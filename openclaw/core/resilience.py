"""
Resilience Module for OpenClaw

Production stability patterns:
- Circuit Breaker: prevents cascading failures
- Health Check: aggregate system health monitoring
- Graceful Shutdown: drain in-flight tasks before exit
"""

import time
import threading
import signal
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger

logger = get_logger("resilience")


# ============== Circuit Breaker ==============

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal — requests pass through
    OPEN = "open"           # Failing — requests rejected immediately
    HALF_OPEN = "half_open" # Testing — limited requests pass through


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying half-open
    half_open_max_calls: int = 3        # Test calls in half-open state
    success_threshold: int = 2          # Successes in half-open to close
    excluded_exceptions: Tuple = ()     # Exceptions that don't count as failures


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    States:
    - CLOSED: Normal operation. Failures are counted.
    - OPEN: All calls fail immediately. After recovery_timeout, transitions to HALF_OPEN.
    - HALF_OPEN: Limited calls allowed. If they succeed, go CLOSED. If they fail, go OPEN.

    Usage:
        cb = CircuitBreaker("api_calls")

        @cb.protect
        def call_external_api():
            ...

        # Or use manually:
        try:
            with cb:
                result = risky_operation()
        except CircuitOpenError:
            # Fallback logic
            pass
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._state_changes: List[Dict] = []

    @property
    def state(self) -> CircuitState:
        """Get current state, auto-transitioning from OPEN to HALF_OPEN if timeout elapsed."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.time() - self._last_failure_time >= self.config.recovery_timeout
            ):
                self._transition(CircuitState.HALF_OPEN)
            return self._state

    def _transition(self, new_state: CircuitState):
        """Transition to a new state (must hold lock)."""
        old_state = self._state
        self._state = new_state
        self._state_changes.append({
            "from": old_state.value,
            "to": new_state.value,
            "time": time.time()
        })

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        logger.info(f"Circuit '{self.name}': {old_state.value} → {new_state.value}")

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._total_successes += 1
            self._total_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, error: Exception = None):
        """Record a failed call."""
        # Don't count excluded exceptions
        if error and isinstance(error, self.config.excluded_exceptions):
            return

        with self._lock:
            self._total_failures += 1
            self._total_calls += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition(CircuitState.OPEN)

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        current_state = self.state  # This may auto-transition OPEN → HALF_OPEN

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        # OPEN
        return False

    def __enter__(self):
        """Context manager entry."""
        if not self.allow_request():
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN — request rejected"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with the circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN — call to {func.__name__} rejected"
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        return wrapper

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition(CircuitState.CLOSED)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_rate": (
                    round(self._total_failures / self._total_calls, 3)
                    if self._total_calls > 0 else 0.0
                ),
                "state_changes": len(self._state_changes),
            }


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""
    pass


# ============== Health Check System ==============

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Aggregate health monitoring for all subsystems.

    Usage:
        health = HealthChecker()

        # Register health check functions
        health.register("database", check_db_health)
        health.register("rag_engine", check_rag_health)

        # Get system health
        status = health.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, ComponentHealth] = {}
        self._lock = threading.Lock()

    def register(self, name: str, check_fn: Callable):
        """
        Register a health check function.
        The function should return a ComponentHealth or (status, message) tuple.
        """
        with self._lock:
            self._checks[name] = check_fn
            logger.info(f"Registered health check: {name}")

    def unregister(self, name: str):
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._last_results.pop(name, None)

    def check(self, name: str) -> ComponentHealth:
        """Run a single health check."""
        check_fn = self._checks.get(name)
        if not check_fn:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"No health check registered for '{name}'"
            )

        start = time.time()
        try:
            result = check_fn()
            latency = (time.time() - start) * 1000

            if isinstance(result, ComponentHealth):
                result.latency_ms = latency
                result.last_check = time.time()
                health = result
            elif isinstance(result, tuple) and len(result) == 2:
                health = ComponentHealth(
                    name=name,
                    status=result[0],
                    message=result[1],
                    latency_ms=latency
                )
            else:
                health = ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    latency_ms=latency
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                latency_ms=latency
            )

        with self._lock:
            self._last_results[name] = health

        return health

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks and return aggregate status."""
        results = {}
        for name in list(self._checks.keys()):
            results[name] = self.check(name)

        # Determine overall status
        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return {
            "status": overall.value,
            "timestamp": time.time(),
            "components": {
                name: {
                    "status": h.status.value,
                    "message": h.message,
                    "latency_ms": round(h.latency_ms, 2),
                }
                for name, h in results.items()
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get last known health status without re-checking."""
        with self._lock:
            if not self._last_results:
                return {"status": "unknown", "components": {}}

            statuses = [r.status for r in self._last_results.values()]
            if all(s == HealthStatus.HEALTHY for s in statuses):
                overall = HealthStatus.HEALTHY
            elif any(s == HealthStatus.UNHEALTHY for s in statuses):
                overall = HealthStatus.UNHEALTHY
            else:
                overall = HealthStatus.DEGRADED

            return {
                "status": overall.value,
                "components": {
                    name: {
                        "status": h.status.value,
                        "message": h.message,
                        "latency_ms": round(h.latency_ms, 2),
                    }
                    for name, h in self._last_results.items()
                }
            }


# ============== Graceful Shutdown ==============

class GracefulShutdown:
    """
    Manages graceful shutdown of the application.

    Features:
    - Registers shutdown handlers
    - Drains in-flight tasks before exiting
    - Configurable drain timeout
    - Signal handling (SIGTERM, SIGINT)

    Usage:
        shutdown = GracefulShutdown(drain_timeout=15.0)
        shutdown.register_handler(cleanup_database)
        shutdown.register_handler(flush_logs)
        shutdown.install_signal_handlers()

        # In your main loop:
        while not shutdown.is_shutting_down:
            process_work()
    """

    def __init__(self, drain_timeout: float = 10.0):
        self.drain_timeout = drain_timeout
        self._is_shutting_down = False
        self._handlers: List[Callable] = []
        self._in_flight: Dict[str, float] = {}  # task_id -> start_time
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down

    def register_handler(self, handler: Callable):
        """Register a shutdown handler (called in reverse order)."""
        with self._lock:
            self._handlers.append(handler)

    def track_task(self, task_id: str):
        """Track an in-flight task."""
        with self._lock:
            self._in_flight[task_id] = time.time()

    def complete_task(self, task_id: str):
        """Mark a task as complete."""
        with self._lock:
            self._in_flight.pop(task_id, None)

    def in_flight_count(self) -> int:
        """Get number of in-flight tasks."""
        with self._lock:
            return len(self._in_flight)

    def initiate_shutdown(self):
        """Start graceful shutdown."""
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        logger.info("Graceful shutdown initiated...")

        # Wait for in-flight tasks to drain
        start = time.time()
        while self.in_flight_count() > 0:
            elapsed = time.time() - start
            if elapsed >= self.drain_timeout:
                remaining = self.in_flight_count()
                logger.warning(
                    f"Drain timeout ({self.drain_timeout}s) reached. "
                    f"{remaining} tasks still in-flight — forcing shutdown."
                )
                break

            remaining = self.in_flight_count()
            logger.info(f"Draining {remaining} in-flight tasks... ({elapsed:.1f}s)")
            time.sleep(0.5)

        # Run shutdown handlers in reverse order
        for handler in reversed(self._handlers):
            try:
                handler_name = getattr(handler, '__name__', str(handler))
                logger.info(f"Running shutdown handler: {handler_name}")
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")

        self._shutdown_event.set()
        logger.info("Graceful shutdown complete.")

    def wait_for_shutdown(self, timeout: float = None) -> bool:
        """Block until shutdown is complete. Returns True if shutdown completed."""
        return self._shutdown_event.wait(timeout=timeout)

    def install_signal_handlers(self):
        """Install SIGTERM and SIGINT handlers."""
        def handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name} — initiating graceful shutdown")
            threading.Thread(
                target=self.initiate_shutdown,
                daemon=True
            ).start()

        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
            logger.info("Signal handlers installed (SIGTERM, SIGINT)")
        except (ValueError, OSError) as e:
            # Can't install signals from non-main thread
            logger.warning(f"Could not install signal handlers: {e}")


# ============== Global Instances ==============

_health_checker: Optional[HealthChecker] = None
_shutdown: Optional[GracefulShutdown] = None
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_shutdown_manager() -> GracefulShutdown:
    """Get global shutdown manager."""
    global _shutdown
    if _shutdown is None:
        _shutdown = GracefulShutdown()
    return _shutdown


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator factory for circuit breaker protection."""
    cb = get_circuit_breaker(name, config)
    return cb.protect


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    "HealthStatus",
    "ComponentHealth",
    "HealthChecker",
    "GracefulShutdown",
    "get_health_checker",
    "get_shutdown_manager",
    "get_circuit_breaker",
    "circuit_breaker",
]
