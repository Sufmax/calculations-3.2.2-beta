"""
Streamer de cache Blender — upload direct vers Storj.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from config import Config
from s3_uploader import S3Uploader
from utils import format_bytes

logger = logging.getLogger(__name__)

CACHE_EXTENSIONS = {
    '.bphys', '.vdb', '.uni', '.gz',
    '.png', '.exr', '.abc', '.obj', '.ply',
}

PROGRESS_INTERVAL = 5.0
UPLOAD_WORKERS = 3


class CacheFileHandler(FileSystemEventHandler):
    def __init__(self, streamer: 'CacheStreamer'):
        self.streamer = streamer
        super().__init__()

    def _should_process(self, src_path: str) -> bool:
        return Path(src_path).suffix.lower() in CACHE_EXTENSIONS

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process(event.src_path):
            self.streamer.schedule_file(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process(event.src_path):
            self.streamer.schedule_file(Path(event.src_path))


class CacheStreamer:
    def __init__(
        self,
        cache_dir: Path,
        ws_client,
        s3_credentials: Dict[str, str],
    ):
        self.cache_dir = cache_dir
        self.ws_client = ws_client

        self.uploader = S3Uploader(
            endpoint=s3_credentials['endpoint'],
            bucket=s3_credentials['bucket'],
            region=s3_credentials['region'],
            access_key_id=s3_credentials['accessKeyId'],
            secret_access_key=s3_credentials['secretAccessKey'],
            cache_prefix=s3_credentials.get('cachePrefix', 'cache/'),
        )

        self.queue: asyncio.Queue = asyncio.Queue()
        self.uploaded_files: Set[str] = set()
        self.pending_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        self.is_running = False
        self.observer: Optional[Observer] = None
        self.upload_task: Optional[asyncio.Task] = None
        self.progress_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS)

        self.start_time = time.time()
        self.last_log_time = 0

    def start(self):
        logger.info(f"Démarrage du streamer (upload direct Storj): {self.cache_dir}")
        self.is_running = True
        self._loop = asyncio.get_event_loop()

        self._start_watching()
        self.upload_task = asyncio.create_task(self._upload_loop())
        self.progress_task = asyncio.create_task(self._progress_loop())
        self._scan_existing_files()

    def stop(self):
        logger.info("Arrêt du streamer de cache")
        self.is_running = False

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

        if self.upload_task:
            self.upload_task.cancel()
        if self.progress_task:
            self.progress_task.cancel()

        self._executor.shutdown(wait=False)

    def _start_watching(self):
        self.observer = Observer()
        handler = CacheFileHandler(self)
        self.observer.schedule(handler, str(self.cache_dir), recursive=True)
        self.observer.start()
        logger.info("Surveillance du cache activée")

    def _scan_existing_files(self):
        if not self.cache_dir.exists():
            return

        count = 0
        for ext in CACHE_EXTENSIONS:
            for fp in self.cache_dir.rglob(f'*{ext}'):
                if fp.is_file():
                    self._queue_file(fp)
                    count += 1
        logger.info(f"{count} fichiers de cache existants trouvés")

    def schedule_file(self, file_path: Path):
        if self._loop is None or not self.is_running:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue_file, file_path)
        except RuntimeError:
            pass

    def _queue_file(self, file_path: Path):
        key = str(file_path)
        if key in self.uploaded_files or key in self.pending_files:
            return
        self.pending_files.add(key)
        self.queue.put_nowait(file_path)

    async def _upload_loop(self):
        logger.info("Boucle d'upload démarrée")
        try:
            while self.is_running:
                try:
                    file_path = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                    await self._upload_file(file_path)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Erreur dans upload_loop: {e}", exc_info=True)
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Upload loop annulée")

    async def _upload_file(self, file_path: Path):
        file_key = str(file_path)

        if file_key in self.uploaded_files:
            self.pending_files.discard(file_key)
            return

        if not file_path.exists():
            self.pending_files.discard(file_key)
            return

        await self._wait_file_stable(file_path)

        try:
            file_size = file_path.stat().st_size
        except OSError:
            self.pending_files.discard(file_key)
            return

        if file_size == 0:
            self.uploaded_files.add(file_key)
            self.pending_files.discard(file_key)
            return

        s3_key = self.uploader.build_s3_key(self.cache_dir, file_path)

        logger.debug(f"Upload start: {file_path.name}")

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self._executor,
            self.uploader.upload_file,
            file_path,
            s3_key,
        )

        if success:
            self.uploaded_files.add(file_key)
            self.pending_files.discard(file_key)
            
            now = time.time()
            if file_size > 1024 * 1024 or (now - self.last_log_time > 5.0):
                logger.info(f"✓ {file_path.name} uploadé ({format_bytes(file_size)})")
                self.last_log_time = now
            else:
                logger.debug(f"✓ {file_path.name} uploadé")
        else:
            self.failed_files.add(file_key)
            self.pending_files.discard(file_key)
            logger.error(f"✗ {file_path.name} échoué")

    async def _wait_file_stable(self, file_path: Path, max_wait: float = 5.0):
        last_size = -1
        waited = 0.0
        interval = 0.5
        while waited < max_wait:
            try:
                size = file_path.stat().st_size
                if size == last_size and size > 0:
                    return
                last_size = size
            except OSError:
                pass
            await asyncio.sleep(interval)
            waited += interval

    async def _progress_loop(self):
        try:
            while self.is_running:
                await asyncio.sleep(PROGRESS_INTERVAL)
                await self._send_progress()
        except asyncio.CancelledError:
            pass

    def _get_disk_usage(self):
        """Fonction synchrone pour scanner le disque (exécutée dans thread)."""
        total_bytes = 0
        total_files = 0
        if self.cache_dir.exists():
            for ext in CACHE_EXTENSIONS:
                for fp in self.cache_dir.rglob(f'*{ext}'):
                    try:
                        if fp.is_file():
                            total_bytes += fp.stat().st_size
                            total_files += 1
                    except OSError:
                        pass
        return total_bytes, total_files

    async def _send_progress(self):
        stats = self.uploader.get_stats()
        uploaded_bytes = stats['total_bytes_uploaded']
        uploaded_files = stats['total_files_uploaded']
        errors = stats['total_errors']

        # Scan disque en thread séparé pour ne pas bloquer
        loop = asyncio.get_running_loop()
        try:
            disk_bytes, disk_files = await loop.run_in_executor(
                None, self._get_disk_usage
            )
        except Exception as e:
            logger.warning(f"Erreur scan disque: {e}")
            disk_bytes = uploaded_bytes
            disk_files = uploaded_files

        elapsed = time.time() - self.start_time
        rate = uploaded_bytes / elapsed if elapsed > 0 else 0

        percent = 0
        if disk_bytes > 0:
            percent = min(100, int((uploaded_bytes / disk_bytes) * 100))
        elif uploaded_files > 0 and disk_files == 0:
            percent = 100

        if int(elapsed) % 30 == 0 or percent == 100:
             logger.info(
                 f"Status: {percent}% | "
                 f"Disque: {format_bytes(disk_bytes)} ({disk_files} f) | "
                 f"Envoyé: {format_bytes(uploaded_bytes)} ({uploaded_files} f) | "
                 f"Vitesse: {format_bytes(rate)}/s"
             )

        if self.ws_client.is_connected():
            await self.ws_client.send_progress(
                upload_percent=percent,
                disk_bytes=disk_bytes,
                disk_files=disk_files,
                uploaded_bytes=uploaded_bytes,
                uploaded_files=uploaded_files,
                errors=errors,
                rate_bytes_per_sec=rate,
            )

    async def finalize(self):
        logger.info("Finalisation du streaming...")

        timeout = 60.0
        waited = 0.0
        while not self.queue.empty() and waited < timeout:
            await asyncio.sleep(0.5)
            waited += 0.5

        waited = 0.0
        while self.pending_files and waited < timeout:
            await asyncio.sleep(0.5)
            waited += 0.5

        await self._send_progress()
        await self.ws_client.send_cache_complete()

        stats = self.uploader.get_stats()
        logger.info(f"Upload terminé: {stats['total_files_uploaded']} fichiers.")

    def get_stats(self) -> dict:
        stats = self.uploader.get_stats()
        elapsed = time.time() - self.start_time
        return {
            **stats,
            'elapsed_seconds': elapsed,
            'files_pending': len(self.pending_files),
        }