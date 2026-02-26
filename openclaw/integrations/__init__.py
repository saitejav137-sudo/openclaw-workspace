"""Integrations module"""

from .http import VisionHTTPServer, RateLimiter, APIKeyAuth
from .telegram import TelegramBot
from .websocket import WebSocketManager, WebSocketManagerSync
from .streaming import StreamingServer, StreamManager, create_default_stream
from .rest_api import RESTServer, RESTAPIHandler, TriggerStore, OPENAPI_SPEC
from .fastapi_server import app as FastAPIApp, create_app, ConnectionManager, manager
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
from .search import (
    SearchProvider,
    SearchResult,
    SearchResponse,
    DuckDuckGoSearch,
    BraveSearch,
    SearchEngine,
    get_search_engine,
    search,
    quick_search,
    answer,
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
    "FastAPIApp",
    "create_app",
    "ConnectionManager",
    "manager",
    "AuthManager",
    "User",
    "UserRole",
    "Session",
    "Permission",
    "RBACMiddleware",
    "get_auth_manager",
    "init_auth_manager",
    "ROLE_PERMISSIONS",
    # Search module
    "SearchProvider",
    "SearchResult",
    "SearchResponse",
    "DuckDuckGoSearch",
    "BraveSearch",
    "SearchEngine",
    "get_search_engine",
    "search",
    "quick_search",
    "answer",
]
