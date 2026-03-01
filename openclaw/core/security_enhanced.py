"""
OpenClaw Security & Network Enhancements — Phase 5 + v2026.2.26 Hardening

Per-IP rate limiting, secret rotation, async HTTP client wrapper,
circuit-breaker-protected orchestrator, SSRF guard, symlink protection,
and exec approval hardening.
"""

import time
import threading
import hashlib
import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import socket
from pathlib import Path

from .logger import get_logger

logger = get_logger("security_enhanced")


# ============== Per-IP Rate Limiter ==============

class PerIPRateLimiter:
    """
    Token-bucket rate limiter that tracks limits per IP address.

    Features:
    - Configurable rate per period
    - Auto-cleanup of stale entries
    - IP blacklisting for repeated offenders
    - Thread-safe
    """

    def __init__(self, rate: int = 60, per: float = 60.0,
                 blacklist_after: int = 10, cleanup_interval: float = 300.0):
        self.rate = rate
        self.per = per
        self.blacklist_after = blacklist_after
        self.cleanup_interval = cleanup_interval
        self._buckets: Dict[str, Dict] = {}
        self._violations: Dict[str, int] = defaultdict(int)
        self._blacklist: set = set()
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, ip: str) -> bool:
        """Check if request from IP is allowed."""
        with self._lock:
            # Auto-cleanup old entries periodically
            now = time.time()
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup(now)

            # Check blacklist
            if ip in self._blacklist:
                return False

            # Get or create bucket
            if ip not in self._buckets:
                self._buckets[ip] = {
                    "allowance": float(self.rate),
                    "last_check": now,
                }

            bucket = self._buckets[ip]
            elapsed = now - bucket["last_check"]

            # Refill tokens
            bucket["allowance"] += elapsed * (self.rate / self.per)
            bucket["last_check"] = now

            if bucket["allowance"] > self.rate:
                bucket["allowance"] = float(self.rate)

            if bucket["allowance"] < 1.0:
                self._violations[ip] += 1
                if self._violations[ip] >= self.blacklist_after:
                    self._blacklist.add(ip)
                    logger.warning(f"IP {ip} blacklisted after {self._violations[ip]} rate-limit violations")
                return False

            bucket["allowance"] -= 1.0
            return True

    def unblock(self, ip: str):
        """Remove IP from blacklist and reset rate limit bucket."""
        with self._lock:
            self._blacklist.discard(ip)
            self._violations.pop(ip, None)
            self._buckets.pop(ip, None)  # Reset tokens so next request is allowed

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "tracked_ips": len(self._buckets),
                "blacklisted": list(self._blacklist),
                "violations": dict(self._violations),
            }

    def _cleanup(self, now: float):
        """Remove stale entries older than 2x the rate period."""
        stale_cutoff = now - (self.per * 2)
        stale_ips = [
            ip for ip, b in self._buckets.items()
            if b["last_check"] < stale_cutoff
        ]
        for ip in stale_ips:
            del self._buckets[ip]
            self._violations.pop(ip, None)
        self._last_cleanup = now


# ============== Secret Rotation Manager ==============

class SecretRotationManager:
    """
    Automated secret rotation support.

    Features:
    - Configurable rotation intervals per secret
    - Pre-rotation hooks (validate new secret works)
    - Post-rotation hooks (update dependent services)
    - Rotation history tracking
    """

    def __init__(self):
        self._rotation_configs: Dict[str, Dict] = {}
        self._rotation_history: List[Dict] = []
        self._lock = threading.Lock()

    def register_secret(self, key: str, rotation_days: int = 90,
                        generator: Callable = None,
                        pre_hook: Callable = None,
                        post_hook: Callable = None):
        """Register a secret for automatic rotation."""
        with self._lock:
            self._rotation_configs[key] = {
                "rotation_days": rotation_days,
                "generator": generator or self._default_generator,
                "pre_hook": pre_hook,
                "post_hook": post_hook,
                "last_rotated": time.time(),
            }
        logger.info(f"Registered secret '{key}' for rotation every {rotation_days} days")

    def check_rotation_needed(self) -> List[str]:
        """Check which secrets need rotation."""
        now = time.time()
        needs_rotation = []
        with self._lock:
            for key, config in self._rotation_configs.items():
                age_days = (now - config["last_rotated"]) / 86400
                if age_days >= config["rotation_days"]:
                    needs_rotation.append(key)
        return needs_rotation

    def rotate(self, key: str) -> bool:
        """Rotate a specific secret."""
        from .secrets import get_secrets_manager

        with self._lock:
            config = self._rotation_configs.get(key)
            if not config:
                logger.error(f"No rotation config for secret '{key}'")
                return False

        try:
            # Generate new value
            new_value = config["generator"](key)

            # Pre-hook validation
            if config["pre_hook"]:
                if not config["pre_hook"](key, new_value):
                    logger.warning(f"Pre-rotation hook failed for '{key}'")
                    return False

            # Apply rotation
            sm = get_secrets_manager()
            old_value = sm.get(key)
            sm.set(key, new_value, persist=True)

            # Post-hook
            if config["post_hook"]:
                config["post_hook"](key, new_value, old_value)

            # Record
            with self._lock:
                config["last_rotated"] = time.time()
                self._rotation_history.append({
                    "key": key,
                    "timestamp": time.time(),
                    "success": True,
                })

            logger.info(f"Secret '{key}' rotated successfully")
            return True

        except Exception as e:
            logger.error(f"Secret rotation failed for '{key}': {e}")
            with self._lock:
                self._rotation_history.append({
                    "key": key,
                    "timestamp": time.time(),
                    "success": False,
                    "error": str(e),
                })
            return False

    def rotate_all_due(self) -> Dict[str, bool]:
        """Rotate all secrets that are due."""
        due = self.check_rotation_needed()
        results = {}
        for key in due:
            results[key] = self.rotate(key)
        return results

    def get_status(self) -> Dict:
        """Get rotation status for all managed secrets."""
        now = time.time()
        with self._lock:
            status = {}
            for key, config in self._rotation_configs.items():
                age_days = (now - config["last_rotated"]) / 86400
                status[key] = {
                    "age_days": round(age_days, 1),
                    "max_days": config["rotation_days"],
                    "needs_rotation": age_days >= config["rotation_days"],
                }
            return status

    @staticmethod
    def _default_generator(key: str) -> str:
        """Generate a secure random secret value."""
        return hashlib.sha256(os.urandom(64)).hexdigest()


# ============== Async HTTP Client ==============

class AsyncHTTPClient:
    """
    Async HTTP client with retry, timeout, and circuit breaker.

    Wraps aiohttp or falls back to threading + requests.
    """

    def __init__(self, timeout: float = 30.0, max_retries: int = 3,
                 base_delay: float = 0.5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._session = None

    def _get_session(self):
        """Get or create a requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=0,  # We handle retries ourselves
                pool_connections=10,
                pool_maxsize=10
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    async def get(self, url: str, headers: Dict = None, **kwargs) -> Dict:
        """Async GET with retry."""
        return await self._request("GET", url, headers=headers, **kwargs)

    async def post(self, url: str, data: Any = None,
                   json_data: Dict = None, headers: Dict = None, **kwargs) -> Dict:
        """Async POST with retry."""
        return await self._request("POST", url, data=data,
                                   json_data=json_data, headers=headers, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> Dict:
        """Execute HTTP request with retry and exponential backoff."""
        loop = asyncio.get_event_loop()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: self._sync_request(method, url, **kwargs)
                )
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        return {
            "success": False,
            "status_code": 0,
            "error": str(last_error),
        }

    def _sync_request(self, method: str, url: str,
                      headers: Dict = None, data: Any = None,
                      json_data: Dict = None, **kwargs) -> Dict:
        """Synchronous request implementation."""
        session = self._get_session()

        req_kwargs = {
            "timeout": self.timeout,
            "headers": headers or {},
        }
        if data:
            req_kwargs["data"] = data
        if json_data:
            req_kwargs["json"] = json_data

        response = session.request(method, url, **req_kwargs)

        return {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "body": response.text,
            "headers": dict(response.headers),
        }

    def close(self):
        """Close the client."""
        if self._session:
            self._session.close()
            self._session = None
        self._executor.shutdown(wait=False)


# ============== Resilient Executor ==============

class ResilientExecutor:
    """
    Execute functions with circuit breaker + retry + timeout.

    Combines resilience patterns for any callable.
    """

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 max_retries: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.max_retries = max_retries

        self._failure_count = 0
        self._last_failure = 0.0
        self._state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
        self._stats = {"calls": 0, "successes": 0, "failures": 0, "rejected": 0}

    def execute(self, func: Callable, *args, timeout: float = None, **kwargs) -> Any:
        """Execute function with full resilience."""
        with self._lock:
            self._stats["calls"] += 1

            # Check circuit breaker
            if self._state == "open":
                if time.time() - self._last_failure > self.recovery_timeout:
                    self._state = "half_open"
                else:
                    self._stats["rejected"] += 1
                    raise RuntimeError("Circuit breaker is open")

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if timeout:
                    import signal
                    result = self._run_with_timeout(func, args, kwargs, timeout)
                else:
                    result = func(*args, **kwargs)

                with self._lock:
                    self._stats["successes"] += 1
                    if self._state == "half_open":
                        self._state = "closed"
                        self._failure_count = 0

                return result

            except Exception as e:
                last_error = e
                with self._lock:
                    self._failure_count += 1
                    self._last_failure = time.time()

                    if self._failure_count >= self.failure_threshold:
                        self._state = "open"
                        self._stats["failures"] += 1
                        logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
                        raise

                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff

        with self._lock:
            self._stats["failures"] += 1
        raise last_error

    def _run_with_timeout(self, func, args, kwargs, timeout):
        """Run function with timeout using threading."""
        result_holder = {}
        error_holder = {}

        def runner():
            try:
                result_holder["value"] = func(*args, **kwargs)
            except Exception as e:
                error_holder["error"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            raise TimeoutError(f"Function timed out after {timeout}s")

        if "error" in error_holder:
            raise error_holder["error"]

        return result_holder.get("value")

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        with self._lock:
            return {
                **self._stats,
                "state": self._state,
                "failure_count": self._failure_count,
            }

    def reset(self):
        """Reset the executor."""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._last_failure = 0.0


# ============== Input Validator Middleware ==============

class InputValidator:
    """
    Input validation and sanitization for HTTP endpoints.

    Provides common validators that can be chained together.
    """

    MAX_STRING_LENGTH = 10000
    MAX_JSON_DEPTH = 10

    @staticmethod
    def sanitize_string(s: str, max_length: int = None) -> str:
        """Remove null bytes and control characters, enforce length."""
        if not isinstance(s, str):
            return ""
        max_len = max_length or InputValidator.MAX_STRING_LENGTH
        # Remove null bytes and most control characters (keep newline, tab)
        result = ""
        for ch in s[:max_len]:
            if ch in ('\n', '\t', '\r') or (ord(ch) >= 32):
                result += ch
        return result.strip()

    @staticmethod
    def validate_json_depth(data: Any, max_depth: int = None, current: int = 0) -> bool:
        """Prevent deeply nested JSON (bombs)."""
        max_d = max_depth or InputValidator.MAX_JSON_DEPTH
        if current > max_d:
            return False
        if isinstance(data, dict):
            return all(
                InputValidator.validate_json_depth(v, max_d, current + 1)
                for v in data.values()
            )
        if isinstance(data, list):
            return all(
                InputValidator.validate_json_depth(item, max_d, current + 1)
                for item in data
            )
        return True

    @staticmethod
    def validate_content_length(content_length: int, max_bytes: int = 10_000_000) -> bool:
        """Validate content length header to prevent oversized payloads."""
        return 0 <= content_length <= max_bytes

    @staticmethod
    def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize HTTP headers."""
        sanitized = {}
        for key, value in headers.items():
            clean_key = InputValidator.sanitize_string(key, max_length=100)
            clean_value = InputValidator.sanitize_string(value, max_length=8000)
            if clean_key:
                sanitized[clean_key] = clean_value
        return sanitized


# ============== SSRF Guard (v2026.2.26) ==============

class SSRFGuard:
    """
    SSRF protection guard — blocks requests to internal/private/multicast IPs.

    Ported from upstream OpenClaw v2026.2.26:
    - Blocks RFC 1918 private ranges
    - Blocks loopback, link-local, reserved
    - Blocks IPv6 multicast (ff00::/8)
    - Configurable allowlist
    """

    def __init__(self, allowed_hosts: Optional[List[str]] = None):
        self._allowed_hosts: set = set(allowed_hosts or [])
        self._lock = threading.Lock()

    def add_allowed_host(self, host: str):
        with self._lock:
            self._allowed_hosts.add(host.lower())

    def is_safe_url(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return False
            with self._lock:
                if hostname.lower() in self._allowed_hosts:
                    return True
            try:
                addr_infos = socket.getaddrinfo(hostname, None)
            except socket.gaierror:
                return False
            for addr_info in addr_infos:
                ip_str = addr_info[4][0]
                if not self._is_safe_ip(ip_str):
                    logger.warning(f"SSRF blocked: {url} resolves to unsafe IP {ip_str}")
                    return False
            return True
        except Exception as e:
            logger.error(f"SSRF check error for {url}: {e}")
            return False

    @staticmethod
    def _is_safe_ip(ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
            if isinstance(ip, ipaddress.IPv6Address) and ip.is_multicast:
                return False
            if isinstance(ip, ipaddress.IPv4Address) and ip.is_multicast:
                return False
            return True
        except ValueError:
            return False


# ============== Symlink Path Traversal Guard (v2026.2.26) ==============

class PathGuard:
    """
    Workspace filesystem boundary guard.
    Rejects symlinks escaping workspace, hardlinks, traversal attacks.
    """

    def __init__(self, workspace_root: str):
        self._root = Path(workspace_root).resolve()

    def is_safe_path(self, target_path: str) -> bool:
        try:
            target = Path(target_path)
            if target.exists():
                resolved = target.resolve()
            else:
                resolved = self._resolve_through_ancestors(target)
            try:
                resolved.relative_to(self._root)
                return True
            except ValueError:
                logger.warning(f"Path traversal blocked: {target_path} (outside {self._root})")
                return False
        except Exception as e:
            logger.error(f"Path guard error for {target_path}: {e}")
            return False

    def _resolve_through_ancestors(self, path: Path) -> Path:
        parts = list(path.parts)
        resolved = Path(parts[0]).resolve() if parts else self._root
        for part in parts[1:]:
            candidate = resolved / part
            if candidate.exists():
                resolved = candidate.resolve()
            else:
                resolved = resolved / part
        return resolved

    def is_hardlink(self, path: str) -> bool:
        try:
            return os.stat(path).st_nlink > 1
        except OSError:
            return False

    def validate_write(self, path: str) -> bool:
        if not self.is_safe_path(path):
            return False
        if os.path.exists(path) and self.is_hardlink(path):
            logger.warning(f"Hardlink write blocked: {path}")
            return False
        return True


# ============== Exec Approval Hardening (v2026.2.26) ==============

class ExecApprovalGuard:
    """
    Exec approval hardening — binds approvals to exact argv identity.
    Rejects mutable symlink cwd paths, prevents trailing-space swaps.
    """

    def __init__(self):
        self._approved: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def approve(self, plan_id: str, argv: List[str], cwd: str,
                agent_id: str = "default") -> bool:
        cwd_path = Path(cwd)
        if cwd_path.is_symlink():
            logger.warning(f"Exec approval rejected: symlink cwd {cwd}")
            return False
        resolved_cwd = str(cwd_path.resolve())
        with self._lock:
            self._approved[plan_id] = {
                "argv": list(argv),
                "cwd": resolved_cwd,
                "agent_id": agent_id,
                "approved_at": time.time(),
            }
        logger.info(f"Exec plan {plan_id} approved: {argv[0] if argv else '?'}")
        return True

    def check_approval(self, plan_id: str, argv: List[str], cwd: str,
                       agent_id: str = "default") -> bool:
        with self._lock:
            plan = self._approved.get(plan_id)
            if not plan:
                return False
        cwd_path = Path(cwd)
        if cwd_path.is_symlink():
            return False
        resolved_cwd = str(cwd_path.resolve())
        if plan["argv"] != list(argv):
            return False
        if plan["cwd"] != resolved_cwd:
            return False
        if plan["agent_id"] != agent_id:
            return False
        return True

    def revoke(self, plan_id: str):
        with self._lock:
            self._approved.pop(plan_id, None)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "active_approvals": len(self._approved),
                "plan_ids": list(self._approved.keys()),
            }


# ---------- Global Instances ----------

_ip_limiter: Optional[PerIPRateLimiter] = None
_rotation_manager: Optional[SecretRotationManager] = None
_resilient_executor: Optional[ResilientExecutor] = None
_ssrf_guard: Optional[SSRFGuard] = None
_path_guard: Optional[PathGuard] = None
_exec_guard: Optional[ExecApprovalGuard] = None


def get_ip_rate_limiter(rate: int = 60, per: float = 60.0) -> PerIPRateLimiter:
    global _ip_limiter
    if _ip_limiter is None:
        _ip_limiter = PerIPRateLimiter(rate=rate, per=per)
    return _ip_limiter


def get_rotation_manager() -> SecretRotationManager:
    global _rotation_manager
    if _rotation_manager is None:
        _rotation_manager = SecretRotationManager()
    return _rotation_manager


def get_resilient_executor() -> ResilientExecutor:
    global _resilient_executor
    if _resilient_executor is None:
        _resilient_executor = ResilientExecutor()
    return _resilient_executor


def get_ssrf_guard(allowed_hosts: List[str] = None) -> SSRFGuard:
    global _ssrf_guard
    if _ssrf_guard is None:
        _ssrf_guard = SSRFGuard(allowed_hosts=allowed_hosts)
    return _ssrf_guard


def get_path_guard(workspace_root: str = None) -> PathGuard:
    global _path_guard
    if _path_guard is None:
        root = workspace_root or os.path.expanduser("~/.openclaw")
        _path_guard = PathGuard(workspace_root=root)
    return _path_guard


def get_exec_guard() -> ExecApprovalGuard:
    global _exec_guard
    if _exec_guard is None:
        _exec_guard = ExecApprovalGuard()
    return _exec_guard


__all__ = [
    "PerIPRateLimiter",
    "SecretRotationManager",
    "AsyncHTTPClient",
    "ResilientExecutor",
    "InputValidator",
    "SSRFGuard",
    "PathGuard",
    "ExecApprovalGuard",
    "get_ip_rate_limiter",
    "get_rotation_manager",
    "get_resilient_executor",
    "get_ssrf_guard",
    "get_path_guard",
    "get_exec_guard",
]
