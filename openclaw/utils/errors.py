"""Comprehensive error handling module"""

import traceback
import threading
import functools
import time
from typing import Optional, Callable, Any, Dict, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from openclaw.core.logger import get_logger

logger = get_logger("errors")


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error"""
    timestamp: float
    error_type: str
    message: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    traceback: Optional[str]
    resolved: bool = False


class ErrorHandler:
    """Central error handler with logging and recovery"""

    _instance = None
    _max_errors = 1000

    def __init__(self):
        self._errors: List[ErrorRecord] = []
        self._lock = __import__('threading').Lock()
        self._errors: List[ErrorRecord] = []
        self._lock = threading.Lock()
        self.error_callbacks: List[Callable] = []

    @classmethod
    def get_instance(cls) -> 'ErrorHandler':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_callback(self, callback: Callable):
        """Register callback for error notifications"""
        self.error_callbacks.append(callback)

    def record_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ):
        """Record an error"""
        record = ErrorRecord(
            timestamp=time.time(),
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            context=context or {},
            traceback=traceback.format_exc()
        )

        with self._lock:
            self._errors.append(record)

            # Keep only recent errors
            if len(self._errors) > self._max_errors:
                self._errors = self._errors[-self._max_errors:]

        # Log the error
        log_method = getattr(logger, severity.value, logger.error)
        log_method(f"{record.error_type}: {record.message}")

        # Call callbacks
        for callback in self.error_callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def get_recent_errors(self, limit: int = 100) -> List[ErrorRecord]:
        """Get recent errors"""
        with self._lock:
            return self._errors[-limit:]

    def get_unresolved_errors(self) -> List[ErrorRecord]:
        """Get unresolved errors"""
        with self._lock:
            return [e for e in self._errors if not e.resolved]

    def resolve_error(self, index: int):
        """Mark error as resolved"""
        if 0 <= index < len(self._errors):
            self._errors[index].resolved = True

    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        total = len(self._errors)
        resolved = sum(1 for e in self._errors if e.resolved)
        by_type = {}
        by_severity = {}

        for error in self._errors:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
            by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1

        return {
            "total": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "by_type": by_type,
            "by_severity": by_severity
        }


def handle_errors(
    context: Dict[str, Any] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    reraise: bool = False,
    default_value: Any = None
):
    """Decorator to handle errors in functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler.get_instance()
                ctx = context or {}
                ctx["function"] = func.__name__
                ctx["args"] = str(args)[:100]  # Truncate long args
                ctx["kwargs"] = str(kwargs)[:100]

                error_handler.record_error(e, ctx, severity)

                if reraise:
                    raise
                return default_value

        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for external services"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "closed"  # closed, open, half_open

    @property
    def is_open(self) -> bool:
        """Check if circuit is open"""
        if self._state == "open":
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                return False
            return True
        return False

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker"""
        if self.is_open:
            raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            # Success - reset
            if self._state == "half_open":
                self._state = "closed"
                self._failure_count = 0
            return result

        except self.expected_exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

            raise

    def reset(self):
        """Manually reset the circuit breaker"""
        self._state = "closed"
        self._failure_count = 0
        self._last_failure_time = 0


class RetryContext:
    """Context for retry operations"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_delay: float = 30.0
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self.attempts = 0

    def calculate_delay(self) -> float:
        """Calculate delay for current attempt"""
        delay = self.base_delay * (self.backoff_multiplier ** self.attempts)
        return min(delay, self.max_delay)

    def should_retry(self) -> bool:
        """Check if should retry"""
        return self.attempts < self.max_attempts

    def record_attempt(self):
        """Record an attempt"""
        self.attempts += 1


# Export
__all__ = [
    "ErrorSeverity",
    "ErrorRecord",
    "ErrorHandler",
    "handle_errors",
    "CircuitBreaker",
    "RetryContext",
]
