"""
API Connector Framework for OpenClaw

Pre-built connectors for popular services:
- GitHub, Slack, Discord, Google, generic REST/GraphQL
- Unified interface for all connectors
- Auth management and rate limiting
- Response caching
"""

import time
import json
import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger

logger = get_logger("api_connectors")


class AuthType(Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class APIConfig:
    """Configuration for an API connector."""
    name: str
    base_url: str
    auth_type: AuthType = AuthType.NONE
    api_key: str = ""
    token: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit: int = 60  # Requests per minute
    timeout: float = 30.0
    cache_ttl: float = 300.0  # Cache for 5 minutes


@dataclass
class APIResponse:
    """Standardized API response."""
    success: bool
    status_code: int = 200
    data: Any = None
    error: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    cached: bool = False


class ResponseCache:
    """Simple in-memory response cache with TTL."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached response if not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry["timestamp"] < entry["ttl"]:
                return entry["data"]
            elif entry:
                del self._cache[key]
            return None

    def set(self, key: str, data: Any, ttl: float = 300.0):
        """Cache a response."""
        with self._lock:
            self._cache[key] = {
                "data": data,
                "timestamp": time.time(),
                "ttl": ttl
            }

    def clear(self):
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests: int = 60, window: float = 60.0):
        self.max_requests = max_requests
        self.window = window
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Check if a request is allowed."""
        now = time.time()
        with self._lock:
            self._timestamps = [t for t in self._timestamps if now - t < self.window]
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False

    def wait_time(self) -> float:
        """Get time to wait before next allowed request."""
        with self._lock:
            if not self._timestamps:
                return 0.0
            oldest = min(self._timestamps)
            return max(0, self.window - (time.time() - oldest))


class APIConnector:
    """
    Base API connector with auth, rate limiting, and caching.

    Usage:
        connector = APIConnector(APIConfig(
            name="github",
            base_url="https://api.github.com",
            auth_type=AuthType.BEARER,
            token="ghp_xxx"
        ))

        response = connector.get("/user/repos")
    """

    def __init__(self, config: APIConfig, http_fn: Optional[Callable] = None):
        self.config = config
        self.http_fn = http_fn  # Custom HTTP function
        self.cache = ResponseCache()
        self.rate_limiter = RateLimiter(
            config.rate_limit,
            window=60.0
        )
        self._request_count = 0
        self._error_count = 0

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with auth."""
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.headers)

        if self.config.auth_type == AuthType.API_KEY:
            headers["X-API-Key"] = self.config.api_key
        elif self.config.auth_type == AuthType.BEARER:
            headers["Authorization"] = f"Bearer {self.config.token}"
        elif self.config.auth_type == AuthType.BASIC:
            import base64
            creds = base64.b64encode(
                f"{self.config.api_key}:{self.config.token}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {creds}"

        return headers

    def _cache_key(self, method: str, endpoint: str, params: Dict = None) -> str:
        """Generate cache key."""
        key = f"{method}:{self.config.base_url}{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        use_cache: bool = True
    ) -> APIResponse:
        """Make an API request."""
        # Check cache for GET requests
        if method == HTTPMethod.GET and use_cache:
            cache_key = self._cache_key(method.value, endpoint, params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return APIResponse(
                    success=True, data=cached, cached=True
                )

        # Rate limiting
        if not self.rate_limiter.allow():
            wait = self.rate_limiter.wait_time()
            return APIResponse(
                success=False,
                status_code=429,
                error=f"Rate limited. Retry after {wait:.1f}s"
            )

        # Make request
        url = f"{self.config.base_url}{endpoint}"
        headers = self._build_headers()
        start = time.time()

        try:
            self._request_count += 1

            if self.http_fn:
                result = self.http_fn(
                    method=method.value,
                    url=url,
                    headers=headers,
                    data=data,
                    params=params,
                    timeout=self.config.timeout
                )
                response = APIResponse(
                    success=True,
                    status_code=result.get("status_code", 200),
                    data=result.get("data"),
                    headers=result.get("headers", {}),
                    duration=time.time() - start
                )
            else:
                # No HTTP function available — report it
                response = APIResponse(
                    success=False,
                    error="No HTTP function configured. Set http_fn in constructor.",
                    duration=time.time() - start
                )

            # Cache successful GET responses
            if (
                response.success and
                method == HTTPMethod.GET and
                use_cache
            ):
                cache_key = self._cache_key(method.value, endpoint, params)
                self.cache.set(cache_key, response.data, self.config.cache_ttl)

            return response

        except Exception as e:
            self._error_count += 1
            logger.error(f"API request failed: {e}")
            return APIResponse(
                success=False,
                error=str(e),
                duration=time.time() - start
            )

    def get(self, endpoint: str, params: Dict = None) -> APIResponse:
        """GET request."""
        return self.request(HTTPMethod.GET, endpoint, params=params)

    def post(self, endpoint: str, data: Dict = None) -> APIResponse:
        """POST request."""
        return self.request(HTTPMethod.POST, endpoint, data=data, use_cache=False)

    def put(self, endpoint: str, data: Dict = None) -> APIResponse:
        """PUT request."""
        return self.request(HTTPMethod.PUT, endpoint, data=data, use_cache=False)

    def delete(self, endpoint: str) -> APIResponse:
        """DELETE request."""
        return self.request(HTTPMethod.DELETE, endpoint, use_cache=False)

    def get_stats(self) -> Dict:
        """Get connector statistics."""
        return {
            "name": self.config.name,
            "base_url": self.config.base_url,
            "requests": self._request_count,
            "errors": self._error_count,
            "error_rate": (
                round(self._error_count / self._request_count, 3)
                if self._request_count > 0 else 0
            )
        }


# ============== Pre-built Connector Templates ==============

class ConnectorTemplates:
    """Pre-built connector configurations for popular services."""

    @staticmethod
    def github(token: str = "", http_fn: Callable = None) -> APIConnector:
        """GitHub API connector."""
        return APIConnector(
            APIConfig(
                name="github",
                base_url="https://api.github.com",
                auth_type=AuthType.BEARER,
                token=token,
                headers={"Accept": "application/vnd.github.v3+json"},
                rate_limit=30  # GitHub has 60/hour for unauthenticated
            ),
            http_fn=http_fn
        )

    @staticmethod
    def slack(token: str = "", http_fn: Callable = None) -> APIConnector:
        """Slack API connector."""
        return APIConnector(
            APIConfig(
                name="slack",
                base_url="https://slack.com/api",
                auth_type=AuthType.BEARER,
                token=token,
                rate_limit=50
            ),
            http_fn=http_fn
        )

    @staticmethod
    def discord(token: str = "", http_fn: Callable = None) -> APIConnector:
        """Discord API connector."""
        return APIConnector(
            APIConfig(
                name="discord",
                base_url="https://discord.com/api/v10",
                auth_type=AuthType.BEARER,
                token=token,
                rate_limit=50,
                headers={"User-Agent": "OpenClaw Agent"}
            ),
            http_fn=http_fn
        )

    @staticmethod
    def generic(
        name: str,
        base_url: str,
        auth_type: AuthType = AuthType.NONE,
        api_key: str = "",
        token: str = "",
        http_fn: Callable = None
    ) -> APIConnector:
        """Generic REST API connector."""
        return APIConnector(
            APIConfig(
                name=name,
                base_url=base_url,
                auth_type=auth_type,
                api_key=api_key,
                token=token
            ),
            http_fn=http_fn
        )


# ============== Connector Registry ==============

class ConnectorRegistry:
    """Registry of available API connectors."""

    def __init__(self):
        self._connectors: Dict[str, APIConnector] = {}
        self._lock = threading.Lock()

    def register(self, connector: APIConnector):
        """Register a connector."""
        with self._lock:
            self._connectors[connector.config.name] = connector
            logger.info(f"Registered API connector: {connector.config.name}")

    def get(self, name: str) -> Optional[APIConnector]:
        """Get a connector by name."""
        with self._lock:
            return self._connectors.get(name)

    def list_connectors(self) -> List[str]:
        """List all registered connectors."""
        with self._lock:
            return list(self._connectors.keys())

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats from all connectors."""
        with self._lock:
            return {
                name: c.get_stats()
                for name, c in self._connectors.items()
            }


# ============== Global Instance ==============

_registry: Optional[ConnectorRegistry] = None


def get_connector_registry() -> ConnectorRegistry:
    """Get global connector registry."""
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
    return _registry


__all__ = [
    "AuthType",
    "HTTPMethod",
    "APIConfig",
    "APIResponse",
    "ResponseCache",
    "RateLimiter",
    "APIConnector",
    "ConnectorTemplates",
    "ConnectorRegistry",
    "get_connector_registry",
]
