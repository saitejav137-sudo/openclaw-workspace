"""
Integrations module for OpenClaw

This module uses lazy imports to avoid loading all submodules at startup.
Import specific modules directly when needed for better performance.

Example:
    from openclaw.integrations.http import VisionHTTPServer
    from openclaw.integrations.browser_api import browser_start
"""

# Lazy imports for commonly used integrations
def __getattr__(name):
    """Lazy import attributes on demand"""

    # HTTP server
    if name == "VisionHTTPServer":
        from .http import VisionHTTPServer
        return VisionHTTPServer
    elif name == "RateLimiter":
        from .http import RateLimiter
        return RateLimiter
    elif name == "APIKeyAuth":
        from .http import APIKeyAuth
        return APIKeyAuth

    # Telegram
    elif name == "TelegramBot":
        from .telegram import TelegramBot
        return TelegramBot

    # WebSocket
    elif name == "WebSocketManager":
        from .websocket import WebSocketManager
        return WebSocketManager
    elif name == "WebSocketManagerSync":
        from .websocket import WebSocketManagerSync
        return WebSocketManagerSync

    # Browser
    elif name == "BrowserAgent":
        from .browser_agent import BrowserAgent
        return BrowserAgent
    elif name == "browser_start":
        from .browser_api import browser_start
        return browser_start
    elif name == "browser_goto":
        from .browser_api import browser_goto
        return browser_goto

    # Auth
    elif name == "AuthManager":
        from .auth import AuthManager
        return AuthManager
    elif name == "UserRole":
        from .auth import UserRole
        return UserRole

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Direct imports for backwards compatibility
# These are loaded on first import of the module
from .http import VisionHTTPServer, RateLimiter, APIKeyAuth
from .telegram import TelegramBot
