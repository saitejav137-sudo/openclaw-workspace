"""
Tests for Task Queue Module
"""

import time
import pytest
from unittest.mock import Mock


class TestPriority:
    """Tests for priority ordering."""

    def test_priority_values(self):
        from openclaw.core.task_queue import Priority
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.LOW.value

    def test_queued_task_ordering(self):
        from openclaw.core.task_queue import QueuedTask, Priority
        t1 = QueuedTask(name="low", priority=Priority.LOW)
        t2 = QueuedTask(name="high", priority=Priority.HIGH)
        assert t2 < t1  # HIGH should sort before LOW


class TestTaskRateLimiter:

    def _create_limiter(self, **kwargs):
        from openclaw.core.task_queue import TaskRateLimiter
        return TaskRateLimiter(**kwargs)

    def test_allows_under_limit(self):
        rl = self._create_limiter(max_per_agent=3)
        assert rl.allow("agent1") is True

    def test_blocks_over_limit(self):
        rl = self._create_limiter(max_per_agent=2, window_seconds=60)
        rl.record("agent1")
        rl.record("agent1")
        assert rl.allow("agent1") is False

    def test_per_agent_tracking(self):
        rl = self._create_limiter(max_per_agent=1)
        rl.record("agent1")
        assert rl.allow("agent1") is False
        assert rl.allow("agent2") is True

    def test_window_expiry(self):
        rl = self._create_limiter(max_per_agent=1, window_seconds=0.1)
        rl.record("agent1")
        time.sleep(0.15)
        assert rl.allow("agent1") is True


class TestDeadlockDetector:

    def _create_detector(self):
        from openclaw.core.task_queue import DeadlockDetector
        return DeadlockDetector()

    def test_no_cycle(self):
        dd = self._create_detector()
        dd.add_task("A", [])
        dd.add_task("B", ["A"])
        dd.add_task("C", ["B"])
        assert dd.detect_cycle() is None

    def test_simple_cycle(self):
        dd = self._create_detector()
        dd.add_task("A", ["B"])
        dd.add_task("B", ["A"])
        cycle = dd.detect_cycle()
        assert cycle is not None
        assert len(cycle) > 0

    def test_would_create_cycle(self):
        dd = self._create_detector()
        dd.add_task("A", [])
        dd.add_task("B", ["A"])
        assert dd.would_create_cycle("C", ["B"]) is False
        assert dd.would_create_cycle("A", ["B"]) is True

    def test_remove_task(self):
        dd = self._create_detector()
        dd.add_task("A", ["B"])
        dd.add_task("B", ["A"])
        dd.remove_task("A")
        assert dd.detect_cycle() is None


class TestRetryDelay:

    def test_exponential_backoff(self):
        from openclaw.core.task_queue import calculate_retry_delay
        d1 = calculate_retry_delay(0, base_delay=1.0, jitter=False)
        d2 = calculate_retry_delay(1, base_delay=1.0, jitter=False)
        d3 = calculate_retry_delay(2, base_delay=1.0, jitter=False)
        assert d1 == 1.0
        assert d2 == 2.0
        assert d3 == 4.0

    def test_max_delay(self):
        from openclaw.core.task_queue import calculate_retry_delay
        d = calculate_retry_delay(100, base_delay=1.0, max_delay=10.0, jitter=False)
        assert d == 10.0

    def test_jitter_is_random(self):
        from openclaw.core.task_queue import calculate_retry_delay
        delays = [calculate_retry_delay(2, jitter=True) for _ in range(10)]
        # With jitter, not all values should be identical
        assert len(set(delays)) > 1

    def test_jitter_within_bounds(self):
        from openclaw.core.task_queue import calculate_retry_delay
        for _ in range(50):
            d = calculate_retry_delay(2, base_delay=1.0, max_delay=10.0, jitter=True)
            assert 0 <= d <= 4.0  # max is min(10, 1*2^2) = 4


class TestTaskQueue:

    def _create_queue(self, **kwargs):
        from openclaw.core.task_queue import TaskQueue
        return TaskQueue(**kwargs)

    def test_enqueue_and_process(self):
        q = self._create_queue()
        q.enqueue("test", lambda: 42)
        task = q.process_next()
        assert task is not None
        assert task.result == 42

    def test_priority_ordering(self):
        from openclaw.core.task_queue import Priority
        q = self._create_queue()

        results = []
        q.enqueue("low", lambda: results.append("low"), priority=Priority.LOW)
        q.enqueue("high", lambda: results.append("high"), priority=Priority.HIGH)
        q.enqueue("critical", lambda: results.append("critical"), priority=Priority.CRITICAL)

        q.process_next()
        q.process_next()
        q.process_next()

        assert results == ["critical", "high", "low"]

    def test_max_concurrent_limit(self):
        q = self._create_queue(max_concurrent=1)
        q.enqueue("t1", lambda: time.sleep(0.1))

        # Manually set a task as running
        from openclaw.core.task_queue import QueuedTask, QueuedTaskStatus
        with q._lock:
            fake = QueuedTask(name="running")
            fake.status = QueuedTaskStatus.RUNNING
            q._running["fake"] = fake

        q.enqueue("t2", lambda: "second")
        result = q.process_next()
        assert result is None  # Can't process, at max concurrent

    def test_retry_on_failure(self):
        from openclaw.core.task_queue import QueuedTaskStatus
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient error")
            return "ok"

        q = self._create_queue()
        q.enqueue("flaky", flaky, max_retries=3)
        q.process_next()  # Fails, queued for retry
        task = q.process_next()  # Succeeds

        assert task is not None
        assert task.result == "ok"
        assert call_count == 2

    def test_cancel_task(self):
        from openclaw.core.task_queue import QueuedTaskStatus
        q = self._create_queue()
        task_id = q.enqueue("cancel_me", lambda: None)
        assert q.cancel(task_id) is True

        task = q.get_task(task_id)
        assert task.status == QueuedTaskStatus.CANCELLED

    def test_timeout_detection(self):
        from openclaw.core.task_queue import QueuedTask, QueuedTaskStatus
        q = self._create_queue()

        # Simulate a stuck running task
        with q._lock:
            stuck = QueuedTask(
                name="stuck",
                timeout_seconds=0.1,
                status=QueuedTaskStatus.RUNNING
            )
            stuck.started_at = time.time() - 1  # Started 1s ago
            q._running[stuck.id] = stuck
            q._tasks[stuck.id] = stuck

        timed_out = q.check_timeouts()
        assert len(timed_out) == 1
        assert timed_out[0].status == QueuedTaskStatus.TIMED_OUT

    def test_deadlock_prevention(self):
        from openclaw.core.task_queue import DeadlockError
        q = self._create_queue()

        # Manually set up a cycle in the deadlock detector
        q.deadlock_detector.add_task("A", ["B"])
        q.deadlock_detector.add_task("B", [])

        # This should detect a cycle: A -> B -> A
        assert q.deadlock_detector.would_create_cycle("B", ["A"]) is True
        # No cycle here
        assert q.deadlock_detector.would_create_cycle("C", ["A"]) is False

    def test_get_stats(self):
        q = self._create_queue()
        q.enqueue("t1", lambda: 1)
        stats = q.get_stats()
        assert stats["queued"] == 1
        assert stats["total_enqueued"] == 1

    def test_queue_size(self):
        q = self._create_queue()
        q.enqueue("t1", lambda: None)
        q.enqueue("t2", lambda: None)
        assert q.queue_size() == 2
