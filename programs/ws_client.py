"""
Client WebSocket robuste pour Blender Coordinator.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional
import websockets
from websockets.client import WebSocketClientProtocol

from config import Config

logger = logging.getLogger(__name__)


class WSClient:
    def __init__(self, url: str, password: str):
        self.url = url
        self.password = password
        self.ws: Optional[WebSocketClientProtocol] = None
        self.token: Optional[str] = None
        self.reconnect_attempts = 0
        self.is_running = False
        self.is_authenticated = False

        self.on_authenticated: Optional[Callable] = None
        self.on_message: Optional[Callable[[dict], None]] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

        self._total_bytes_sent = 0
        self._total_messages_sent = 0

    async def connect(self):
        self.is_running = True

        while self.is_running:
            try:
                logger.info(f"Connexion à {self.url}...")
                
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    max_size=10 * 1024 * 1024
                ) as ws:
                    self.ws = ws
                    self.reconnect_attempts = 0
                    logger.info("Connecté au serveur")

                    if self.token:
                        await self.resume_session()
                    else:
                        await self.authenticate()

                    await self.receive_loop()

            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                logger.warning(f"Connexion fermée/perdue: {e}")
            except Exception as e:
                logger.error(f"Erreur connexion: {e}", exc_info=True)
            
            self.is_authenticated = False
            
            if self.is_running:
                self.reconnect_attempts += 1
                delay = min(30, Config.RECONNECT_DELAY * self.reconnect_attempts)
                logger.info(f"Reconnexion dans {delay}s (tentative {self.reconnect_attempts})")
                await asyncio.sleep(delay)

    async def authenticate(self):
        logger.info("Authentification (nouveau token)...")
        await self.send({
            'type': 'AUTH',
            'password': self.password,
            'timestamp': int(time.time() * 1000)
        })
        await self._wait_for_auth_response()

    async def resume_session(self):
        logger.info(f"Reprise de session (token: {self.token[:8]}...)...")
        await self.authenticate()

    async def _wait_for_auth_response(self):
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
            message = json.loads(response)

            if message.get('type') == 'AUTH_SUCCESS':
                new_token = message.get('token')
                if self.token and self.token != new_token:
                    logger.warning("Nouveau token reçu (session réinitialisée)")
                
                self.token = new_token
                self.is_authenticated = True
                logger.info(f"Authentifié (token: {self.token[:8]}...)")

                if self.on_authenticated:
                    await self.on_authenticated(message)

            elif message.get('type') == 'AUTH_FAILED':
                logger.error(f"Authentification échouée: {message.get('reason')}")
                await asyncio.sleep(5)

        except asyncio.TimeoutError:
            logger.error("Timeout auth response")

    async def receive_loop(self):
        while self.is_running and self.ws:
            try:
                message_str = await self.ws.recv()
                message = json.loads(message_str)
                await self.handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"Erreur receive_loop: {e}")
                break

    async def handle_message(self, message: dict):
        msg_type = message.get('type')

        if msg_type == 'BLEND_FILE_URL':
            logger.info("URL .blend reçue")
        elif msg_type == 'S3_CREDENTIALS':
            logger.info("Credentials S3 reçues")
        elif msg_type == 'TERMINATE':
            logger.warning(f"Demande de terminaison: {message.get('reason')}")
            self.is_running = False
        elif msg_type != 'PONG':
            logger.debug(f"Message reçu: {msg_type}")

        if self.on_message:
            await self.on_message(message)

    async def send(self, message: dict):
        if not self.ws or not self.is_running:
            return False
        try:
            await self.ws.send(json.dumps(message))
            return True
        except Exception:
            return False

    async def send_heartbeat(self):
        return await self.send({'type': 'ALIVE'})

    async def send_progress(
        self,
        upload_percent: int,
        disk_bytes: int,
        disk_files: int,
        uploaded_bytes: int,
        uploaded_files: int,
        errors: int,
        rate_bytes_per_sec: float,
    ):
        return await self.send({
            'type': 'PROGRESS_UPDATE',
            'uploadPercent': upload_percent,
            'diskBytes': disk_bytes,
            'diskFiles': disk_files,
            'uploadedBytes': uploaded_bytes,
            'uploadedFiles': uploaded_files,
            'errors': errors,
            'rateBytesPerSec': int(rate_bytes_per_sec),
        })

    async def send_cache_complete(self):
        return await self.send({'type': 'CACHE_COMPLETE'})

    async def send_ready_to_terminate(self):
        return await self.send({'type': 'READY_TO_TERMINATE'})

    def disconnect(self):
        logger.info("Déconnexion...")
        self.is_running = False
        if self.ws:
            asyncio.create_task(self.ws.close())

    def is_connected(self) -> bool:
        return self.ws is not None and self.is_authenticated