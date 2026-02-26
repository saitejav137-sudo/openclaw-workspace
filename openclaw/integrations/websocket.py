"""WebSocket server integration"""

import asyncio
import json
import threading
from typing import Set, Dict, Any, Optional, Callable
from dataclasses import dataclass

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

from ..core.logger import get_logger

logger = get_logger("websocket")


@dataclass
class WSMessage:
    """WebSocket message"""
    type: str
    data: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        })


class WebSocketClient:
    """WebSocket client connection"""

    def __init__(self, websocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.subscriptions: Set[str] = set()

    async def send(self, message: WSMessage):
        """Send message to client"""
        try:
            await self.websocket.send(message.to_json())
        except Exception as e:
            logger.error(f"Send error: {e}")

    async def handle_message(self, message: str):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await self.send(WSMessage("pong", {}))

            elif msg_type == "subscribe":
                self.subscriptions.add(data.get("channel", ""))
                logger.debug(f"Client {self.client_id} subscribed to {data.get('channel')}")

            elif msg_type == "unsubscribe":
                self.subscriptions.discard(data.get("channel", ""))

            elif msg_type == "trigger":
                await self.send(WSMessage("trigger_ack", data.get("data", {})))

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")


class WebSocketManager:
    """WebSocket server for real-time updates"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8766,
        auth_token: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.clients: Set[WebSocketClient] = set()
        self.server = None
        self._client_counter = 0
        self._running = False

    async def handler(self, websocket, path: str):
        """Handle WebSocket connections"""
        if self.auth_token:
            # Check auth
            try:
                auth_message = await asyncio.wait_for(websocket.recv(), timeout=5)
                auth_data = json.loads(auth_message)
                if auth_data.get("token") != self.auth_token:
                    await websocket.close(4001, "Unauthorized")
                    return
            except asyncio.TimeoutError:
                await websocket.close(4001, "Auth required")
                return

        self._client_counter += 1
        client_id = f"client_{self._client_counter}"
        client = WebSocketClient(websocket, client_id)
        self.clients.add(client)

        logger.info(f"WebSocket client connected: {client_id}")

        try:
            async for message in websocket:
                await client.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.clients.discard(client)

    async def broadcast(self, message: WSMessage):
        """Broadcast message to all clients"""
        if not self.clients:
            return

        message_json = message.to_json()
        await asyncio.gather(
            *[client.websocket.send(message_json) for client in self.clients],
            return_exceptions=True
        )

    async def broadcast_trigger(self, mode: str, result: bool):
        """Broadcast trigger event"""
        await self.broadcast(WSMessage("trigger", {
            "mode": mode,
            "result": result
        }))

    async def start(self):
        """Start WebSocket server"""
        if not WS_AVAILABLE:
            logger.warning("websockets library not available")
            return

        self._running = True
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    def stop(self):
        """Stop WebSocket server"""
        self._running = False
        if self.server:
            self.server.close()
            logger.info("WebSocket server stopped")


class WebSocketManagerSync:
    """Synchronous wrapper for WebSocket manager"""

    def __init__(self, host: str = "localhost", port: int = 8766):
        self.host = host
        self.port = port
        self.ws_manager = WebSocketManager(host, port)
        self._loop = None
        self._thread = None

    def start(self):
        """Start WebSocket server in background thread"""
        if not WS_AVAILABLE:
            logger.warning("websockets not installed")
            return

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.ws_manager.start())
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket server started on port {self.port}")

    def stop(self):
        """Stop WebSocket server"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def broadcast_trigger(self, mode: str, result: bool):
        """Broadcast trigger (async, fire and forget)"""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self.ws_manager.broadcast_trigger(mode, result),
                self._loop
            )

    @property
    def _running(self) -> bool:
        return self.ws_manager._running


# Export classes
__all__ = [
    "WebSocketManager",
    "WebSocketManagerSync",
    "WebSocketClient",
    "WSMessage",
]
