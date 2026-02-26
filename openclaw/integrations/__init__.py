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
from .browser_fetch import (
    BrowserFetcher,
    SimpleWebFetcher,
    BrowserFetchResult,
    get_browser_fetcher,
    get_simple_fetcher,
    fetch_url,
    web_search,
)
from .browser_agent import (
    BrowserAction,
    BrowserResult,
    BrowserAgent,
    get_browser_agent,
    close_browser_agent,
)
from .browser_api import (
    browser_start,
    browser_goto,
    browser_click,
    browser_click_text,
    browser_type,
    browser_input,
    browser_submit,
    browser_extract,
    browser_extract_all,
    browser_screenshot,
    browser_info,
    browser_close,
    execute_browser_action,
    quick_browse,
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
    # Browser fetch module
    "BrowserFetcher",
    "SimpleWebFetcher",
    "BrowserFetchResult",
    "get_browser_fetcher",
    "get_simple_fetcher",
    "fetch_url",
    "web_search",
    # Browser agent module
    "BrowserAction",
    "BrowserResult",
    "BrowserAgent",
    "get_browser_agent",
    "close_browser_agent",
]
