"""Integrations module"""

from .http import VisionHTTPServer, RateLimiter, APIKeyAuth
from .telegram import TelegramBot
from .websocket import WebSocketManager, WebSocketManagerSync
from .streaming import StreamingServer, StreamManager, create_default_stream
from .rest_api import RESTServer, RESTAPIHandler, TriggerStore, OPENAPI_SPEC
from .auth import (
    AuthManager,
    User,
    UserRole,
    Session,
    Permission,
    RBACMiddleware,
    get_auth_manager,
    init_auth_manager,
    ROLE_PERMISSIONS
)

__all__ = [
    "VisionHTTPServer",
    "RateLimiter",
    "APIKeyAuth",
    "TelegramBot",
    "WebSocketManager",
    "WebSocketManagerSync",
    "StreamingServer",
    "StreamManager",
    "create_default_stream",
    "RESTServer",
    "RESTAPIHandler",
    "TriggerStore",
    "OPENAPI_SPEC",
    "AuthManager",
    "User",
    "UserRole",
    "Session",
    "Permission",
    "RBACMiddleware",
    "get_auth_manager",
    "init_auth_manager",
    "ROLE_PERMISSIONS",
]
