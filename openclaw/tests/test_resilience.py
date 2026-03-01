"""
Tests for Resilience Module
"""

import time
import pytest
from unittest.mock import Mock, patch


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def _create_cb(self, **kwargs):
        from openclaw.core.resilience import CircuitBreaker, CircuitBreakerConfig
        config = CircuitBreakerConfig(**kwargs)
        return CircuitBreaker("test_circuit", config)

    def test_initial_state_closed(self):
        cb = self._create_cb()
        from openclaw.core.resilience import CircuitState
        assert cb.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self):
        cb = self._create_cb()
        assert cb.allow_request() is True

    def test_opens_after_failure_threshold(self):
        from openclaw.core.resilience import CircuitState
        cb = self._create_cb(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_rejects_requests_when_open(self):
        cb = self._create_cb(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is False

    def test_transitions_to_half_open(self):
        from openclaw.core.resilience import CircuitState
        cb = self._create_cb(failure_threshold=2, recovery_timeout=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        cb = self._create_cb(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        assert cb.allow_request() is True
        assert cb.allow_request() is True
        assert cb.allow_request() is False

    def test_half_open_closes_on_success(self):
        from openclaw.core.resilience import CircuitState
        cb = self._create_cb(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        _ = cb.state  # Trigger transition
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_opens_on_failure(self):
        from openclaw.core.resilience import CircuitState
        cb = self._create_cb(failure_threshold=2, recovery_timeout=0.1)

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        _ = cb.state
        cb.allow_request()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_context_manager_success(self):
        cb = self._create_cb()
        with cb:
            pass  # Success
        stats = cb.get_stats()
        assert stats["total_successes"] == 1

    def test_context_manager_failure(self):
        cb = self._create_cb()
        try:
            with cb:
                raise ValueError("test error")
        except ValueError:
            pass
        stats = cb.get_stats()
        assert stats["total_failures"] == 1

    def test_context_manager_rejects_when_open(self):
        from openclaw.core.resilience import CircuitOpenError
        cb = self._create_cb(failure_threshold=1)
        cb.record_failure()

        with pytest.raises(CircuitOpenError):
            with cb:
                pass

    def test_protect_decorator(self):
        cb = self._create_cb()

        @cb.protect
        def my_func():
            return 42

        assert my_func() == 42
        assert cb.get_stats()["total_successes"] == 1

    def test_protect_decorator_failure(self):
        cb = self._create_cb()

        @cb.protect
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_func()

        assert cb.get_stats()["total_failures"] == 1

    def test_reset(self):
        from openclaw.core.resilience import CircuitState
        cb = self._create_cb(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_get_stats(self):
        cb = self._create_cb()
        cb.record_success()
        cb.record_success()
        cb.record_failure()

        stats = cb.get_stats()
        assert stats["total_calls"] == 3
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 1
        assert stats["name"] == "test_circuit"

    def test_excluded_exceptions(self):
        cb = self._create_cb(
            failure_threshold=2,
            excluded_exceptions=(ValueError,)
        )
        cb.record_failure(ValueError("ignored"))
        cb.record_failure(ValueError("also ignored"))

        # Should still be closed since ValueError is excluded
        from openclaw.core.resilience import CircuitState
        assert cb.state == CircuitState.CLOSED

    def test_success_reduces_failure_count(self):
        cb = self._create_cb(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Should reduce count
        cb.record_failure()

        from openclaw.core.resilience import CircuitState
        assert cb.state == CircuitState.CLOSED  # Not at threshold yet


class TestHealthChecker:
    """Tests for health check system."""

    def _create_checker(self):
        from openclaw.core.resilience import HealthChecker
        return HealthChecker()

    def test_register_check(self):
        hc = self._create_checker()
        hc.register("test", lambda: None)
        result = hc.check_all()
        assert "test" in result["components"]

    def test_healthy_check(self):
        from openclaw.core.resilience import HealthStatus
        hc = self._create_checker()
        hc.register("db", lambda: (HealthStatus.HEALTHY, "OK"))
        result = hc.check_all()
        assert result["status"] == "healthy"
        assert result["components"]["db"]["status"] == "healthy"

    def test_unhealthy_check(self):
        hc = self._create_checker()

        def broken():
            raise ConnectionError("DB down")

        hc.register("db", broken)
        result = hc.check_all()
        assert result["status"] == "unhealthy"

    def test_degraded_when_mixed(self):
        from openclaw.core.resilience import HealthStatus
        hc = self._create_checker()
        hc.register("good", lambda: (HealthStatus.HEALTHY, "OK"))
        hc.register("slow", lambda: (HealthStatus.DEGRADED, "Slow"))
        result = hc.check_all()
        assert result["status"] == "degraded"

    def test_latency_measured(self):
        hc = self._create_checker()
        hc.register("slow", lambda: time.sleep(0.05))
        result = hc.check("slow")
        assert result.latency_ms >= 40


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    def _create_shutdown(self, **kwargs):
        from openclaw.core.resilience import GracefulShutdown
        return GracefulShutdown(**kwargs)

    def test_not_shutting_down_initially(self):
        gs = self._create_shutdown()
        assert gs.is_shutting_down is False

    def test_track_and_complete_task(self):
        gs = self._create_shutdown()
        gs.track_task("task1")
        assert gs.in_flight_count() == 1
        gs.complete_task("task1")
        assert gs.in_flight_count() == 0

    def test_handlers_called_on_shutdown(self):
        gs = self._create_shutdown()
        called = []
        gs.register_handler(lambda: called.append("h1"))
        gs.register_handler(lambda: called.append("h2"))
        gs.initiate_shutdown()
        assert "h1" in called
        assert "h2" in called

    def test_handlers_called_in_reverse_order(self):
        gs = self._create_shutdown()
        order = []
        gs.register_handler(lambda: order.append("first"))
        gs.register_handler(lambda: order.append("second"))
        gs.initiate_shutdown()
        assert order == ["second", "first"]

    def test_drain_timeout(self):
        gs = self._create_shutdown(drain_timeout=0.5)
        gs.track_task("stuck_task")
        gs.initiate_shutdown()
        # Should complete within drain timeout
        assert gs.is_shutting_down is True
