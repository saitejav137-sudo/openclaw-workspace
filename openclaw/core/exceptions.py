"""
Custom exceptions for OpenClaw.

Provides a hierarchy of specific exception types for different
error conditions in the application.
"""

from typing import Optional, Any
from datetime import datetime


class OpenClawError(Exception):
    """Base exception for all OpenClaw errors.

    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        timestamp: When the error occurred
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        code: str = "OC_ERROR",
        details: Optional[dict] = None
    ):
        self.message = message
        self.code = code
        self.timestamp = datetime.utcnow()
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details
            }
        }


# ============== Vision Errors ==============

class VisionError(OpenClawError):
    """Base class for vision-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="VISION_ERROR", details=details)


class OCRError(VisionError):
    """OCR processing error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "OCR_ERROR"


class YOLOError(VisionError):
    """YOLO model error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "YOLO_ERROR"


class TemplateMatchError(VisionError):
    """Template matching error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "TEMPLATE_ERROR"


class ScreenCaptureError(VisionError):
    """Screen capture error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "CAPTURE_ERROR"


# ============== Configuration Errors ==============

class ConfigError(OpenClawError):
    """Base class for configuration errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="CONFIG_ERROR", details=details)


class ConfigValidationError(ConfigError):
    """Configuration validation error."""
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, details=details)
        self.code = "CONFIG_VALIDATION_ERROR"


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    def __init__(self, config_path: str):
        super().__init__(
            f"Configuration file not found: {config_path}",
            details={"path": config_path}
        )
        self.code = "CONFIG_NOT_FOUND"


# ============== Action Errors ==============

class ActionError(OpenClawError):
    """Base class for action execution errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="ACTION_ERROR", details=details)


class KeyPressError(ActionError):
    """Keyboard action error."""
    def __init__(self, message: str, key: Optional[str] = None):
        details = {"key": key} if key else {}
        super().__init__(message, details=details)
        self.code = "KEY_PRESS_ERROR"


class MouseActionError(ActionError):
    """Mouse action error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "MOUSE_ACTION_ERROR"


class ActionTimeoutError(ActionError):
    """Action execution timeout."""
    def __init__(self, action: str, timeout: float):
        super().__init__(
            f"Action '{action}' timed out after {timeout}s",
            details={"action": action, "timeout": timeout}
        )
        self.code = "ACTION_TIMEOUT"


# ============== Browser Errors ==============

class BrowserError(OpenClawError):
    """Base class for browser automation errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="BROWSER_ERROR", details=details)


class BrowserNotFoundError(BrowserError):
    """Browser not found or not installed."""
    def __init__(self, browser: str = "chromium"):
        super().__init__(
            f"Browser '{browser}' not found",
            details={"browser": browser}
        )
        self.code = "BROWSER_NOT_FOUND"


class BrowserNavigationError(BrowserError):
    """Browser navigation error (failed to load page)."""
    def __init__(self, url: str, error: str):
        super().__init__(
            f"Failed to navigate to {url}: {error}",
            details={"url": url, "error": error}
        )
        self.code = "BROWSER_NAVIGATION_ERROR"


class ElementNotFoundError(BrowserError):
    """Browser element not found."""
    def __init__(self, selector: str):
        super().__init__(
            f"Element not found: {selector}",
            details={"selector": selector}
        )
        self.code = "ELEMENT_NOT_FOUND"


# ============== Network Errors ==============

class NetworkError(OpenClawError):
    """Base class for network-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="NETWORK_ERROR", details=details)


class HTTPError(NetworkError):
    """HTTP request error."""
    def __init__(self, status_code: int, message: str, url: Optional[str] = None):
        details = {"status_code": status_code, "url": url} if url else {"status_code": status_code}
        super().__init__(message, details=details)
        self.code = "HTTP_ERROR"


class WebSocketError(NetworkError):
    """WebSocket error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "WEBSOCKET_ERROR"


# ============== Authentication Errors ==============

class AuthError(OpenClawError):
    """Base class for authentication errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="AUTH_ERROR", details=details)


class InvalidAPIKeyError(AuthError):
    """Invalid API key."""
    def __init__(self):
        super().__init__("Invalid or missing API key")
        self.code = "INVALID_API_KEY"


class RateLimitExceededError(AuthError):
    """Rate limit exceeded."""
    def __init__(self, limit: int, window: float):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}s",
            details={"limit": limit, "window": window}
        )
        self.code = "RATE_LIMIT_EXCEEDED"


# ============== Storage Errors ==============

class StorageError(OpenClawError):
    """Base class for storage-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="STORAGE_ERROR", details=details)


class DatabaseError(StorageError):
    """Database operation error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "DATABASE_ERROR"


# ============== Memory Errors ==============

class MemoryError(OpenClawError):
    """Base class for memory-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="MEMORY_ERROR", details=details)


class MemoryNotFoundError(MemoryError):
    """Memory entry not found."""
    def __init__(self, memory_id: str, details: Optional[dict] = None):
        super().__init__(f"Memory not found: {memory_id}", details=details)
        self.code = "MEMORY_NOT_FOUND"
        self.memory_id = memory_id


class MemoryValidationError(MemoryError):
    """Memory validation error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "MEMORY_VALIDATION_ERROR"


class EmbeddingError(MemoryError):
    """Embedding computation error."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)
        self.code = "EMBEDDING_ERROR"


# ============== InterBot Errors ==============

class InterBotError(OpenClawError):
    """Base class for inter-bot communication errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="INTERBOT_ERROR", details=details)


class InterBotConnectionError(InterBotError):
    """Connection to other bot failed."""
    def __init__(self, bot_id: str, details: Optional[dict] = None):
        super().__init__(f"Failed to connect to bot: {bot_id}", details=details)
        self.code = "INTERBOT_CONNECTION_ERROR"
        self.bot_id = bot_id


class InterBotTimeoutError(InterBotError):
    """Message delivery timeout."""
    def __init__(self, message_id: str, details: Optional[dict] = None):
        super().__init__(f"Message delivery timed out: {message_id}", details=details)
        self.code = "INTERBOT_TIMEOUT"
        self.message_id = message_id


# ============== Utility Functions ==============

def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the operation should be retried
    """
    retryable_codes = {
        "NETWORK_ERROR",
        "HTTP_ERROR",
        "WEBSOCKET_ERROR",
        "DATABASE_ERROR",
        "ACTION_TIMEOUT",
        "BROWSER_NAVIGATION_ERROR",
    }

    if isinstance(error, OpenClawError):
        return error.code in retryable_codes

    # Retry common transient errors
    transient_errors = (
        ConnectionError,
        TimeoutError,
        BrokenPipeError,
    )
    return isinstance(error, transient_errors)


__all__ = [
    # Base
    "OpenClawError",
    # Vision
    "VisionError",
    "OCRError",
    "YOLOError",
    "TemplateMatchError",
    "ScreenCaptureError",
    # Config
    "ConfigError",
    "ConfigValidationError",
    "ConfigNotFoundError",
    # Action
    "ActionError",
    "KeyPressError",
    "MouseActionError",
    "ActionTimeoutError",
    # Browser
    "BrowserError",
    "BrowserNotFoundError",
    "BrowserNavigationError",
    "ElementNotFoundError",
    # Network
    "NetworkError",
    "HTTPError",
    "WebSocketError",
    # Auth
    "AuthError",
    "InvalidAPIKeyError",
    "RateLimitExceededError",
    # Storage
    "StorageError",
    "DatabaseError",
    # Utility
    "is_retryable",
]
