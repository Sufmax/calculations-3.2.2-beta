#!/usr/bin/env python3
"""
Point d'entrée principal du script VM
"""

import asyncio
import logging
import sys
from urllib.request import urlopen
from urllib.error import URLError

from config import Config
from utils import setup_logging, decode_base64_to_file
from ws_client import WSClient
from cache_streamer import CacheStreamer
from blender_runner import BlenderRunner

logger = logging.getLogger(__name__)

ws_client: WSClient = None
cache_streamer: CacheStreamer = None
blender_runner: BlenderRunner = None
heartbeat_task: asyncio.Task = None
shutdown_event = asyncio.Event()
s3_credentials: dict = None
_blender_done_event = asyncio.Event()


async def heartbeat_loop():
    logger.info(f"Démarrage heartbeat (interval: {Config.HEARTBEAT_INTERVAL}s)")
    try:
        while not shutdown_event.is_set():
            if ws_client and ws_client.is_connected():
                await ws_client.send_heartbeat()
            await asyncio.sleep(Config.HEARTBEAT_INTERVAL)
    except asyncio.CancelledError:
        pass


async def on_authenticated(message: dict):
    logger.info(f"Authentifié.")
    global heartbeat_task
    heartbeat_task = asyncio.create_task(heartbeat_loop())


async def on_message(message: dict):
    global s3_credentials
    msg_type = message.get('type')

    if msg_type == 'S3_CREDENTIALS':
        s3_credentials = {
            'endpoint': message.get('endpoint'),
            'bucket': message.get('bucket'),
            'region': message.get('region'),
            'accessKeyId': message.get('accessKeyId'),
            'secretAccessKey': message.get('secretAccessKey'),
            'cachePrefix': message.get('cachePrefix', 'cache/'),
        }
        logger.info(f"Credentials S3 reçues: prefix={s3_credentials['cachePrefix']}")

    elif msg_type == 'BLEND_FILE_URL':
        await handle_blend_file_url(message)

    elif msg_type == 'TERMINATE':
        reason = message.get('reason', 'Non spécifié')
        logger.warning(f"Demande de terminaison: {reason}")
        await shutdown()


async def handle_blend_file_url(message: dict):
    url = message.get('url')
    if not url:
        return

    logger.info("Téléchargement .blend...")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _download_url, url)

        Config.BLEND_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(Config.BLEND_FILE, 'wb') as f:
            f.write(data)

        logger.info(f"Fichier .blend sauvegardé ({len(data)} bytes)")
        asyncio.create_task(start_blender())

    except Exception as e:
        logger.error(f"Erreur téléchargement .blend: {e}")


def _download_url(url: str) -> bytes:
    try:
        with urlopen(url, timeout=300) as response:
            return response.read()
    except URLError as e:
        raise RuntimeError(f"Erreur téléchargement: {e}")


async def start_blender():
    global cache_streamer, blender_runner
    await asyncio.sleep(2.0)

    if s3_credentials is None:
        logger.error("Pas de credentials S3 reçues.")
        _blender_done_event.set()
        return

    logger.info("Démarrage Blender + Streamer...")

    try:
        cache_streamer = CacheStreamer(Config.CACHE_DIR, ws_client, s3_credentials)
        cache_streamer.start()

        blender_runner = BlenderRunner(Config.BLEND_FILE, Config.CACHE_DIR)
        return_code = await blender_runner.run()

        logger.info(f"Blender terminé (code: {return_code})")

        if cache_streamer:
            await cache_streamer.finalize()

        if ws_client and ws_client.is_connected():
            await ws_client.send_ready_to_terminate()

    except Exception as e:
        logger.error(f"Erreur exécution: {e}", exc_info=True)
    finally:
        if cache_streamer:
            cache_streamer.stop()
        _blender_done_event.set()


async def shutdown():
    logger.info("Arrêt en cours...")
    shutdown_event.set()

    if not _blender_done_event.is_set():
        try:
            await asyncio.wait_for(_blender_done_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    if blender_runner:
        blender_runner.terminate()

    if cache_streamer:
        cache_streamer.stop()

    if heartbeat_task:
        heartbeat_task.cancel()

    if ws_client:
        ws_client.disconnect()

    logger.info("Arrêt terminé")


async def main():
    global ws_client
    setup_logging(logging.INFO)
    logger.info("Blender VM Worker - Démarrage")

    try:
        Config.validate()
        
        loop = asyncio.get_running_loop()
        try:
            import signal
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except (NotImplementedError, AttributeError):
            pass

        ws_client = WSClient(Config.WS_URL, Config.VM_PASSWORD)
        ws_client.on_authenticated = on_authenticated
        ws_client.on_message = on_message
        ws_client.on_disconnected = lambda: logger.warning("Déconnecté")
        ws_client.on_error = lambda e: logger.error(f"Erreur WS: {e}")

        await ws_client.connect()
        await shutdown_event.wait()

    except KeyboardInterrupt:
        await shutdown()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        await shutdown()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)