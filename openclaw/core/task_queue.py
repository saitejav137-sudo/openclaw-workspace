"""
Priority Task Queue for OpenClaw — v2026.2.26 Aligned

Production-grade task scheduling:
- Priority-based execution (CRITICAL > HIGH > NORMAL > LOW)
- Rate limiting per agent
- Timeout enforcement with deadlock detection
- Retry with jitter to prevent thundering herd
- Delivery queue backoff persistence (v2026.2.26)
- Drain safety with guaranteed flag reset (v2026.2.26)
"""

import time
import random
import threading
import heapq
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .logger import get_logger

logger = get_logger("task_queue")


# ============== Priority Levels ==============

class Priority(Enum):
    """Task priority levels (lower value = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class QueuedTaskStatus(Enum):
    """Status of a queued task."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


# ============== Task Definition ==============

@dataclass
class QueuedTask:
    """A task in the priority queue."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    priority: Priority = Priority.NORMAL
    func: Optional[Callable] = None
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    max_retries: int = 2
    retry_count: int = 0
    agent_id: Optional[str] = None
    status: QueuedTaskStatus = QueuedTaskStatus.QUEUED
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_attempt_at: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)

    def __lt__(self, other):
        """For heap ordering (lower priority value = higher priority)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


# ============== Rate Limiter ==============

class TaskRateLimiter:
    """
    Rate limiter for tasks per agent.
    Prevents any single agent from being overwhelmed.
    """

    def __init__(self, max_per_agent: int = 5, window_seconds: float = 60.0):
        self.max_per_agent = max_per_agent
        self.window_seconds = window_seconds
        self._agent_tasks: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def allow(self, agent_id: str) -> bool:
        """Check if a new task is allowed for this agent."""
        with self._lock:
            now = time.time()
            if agent_id not in self._agent_tasks:
                self._agent_tasks[agent_id] = []

            # Clean old entries
            self._agent_tasks[agent_id] = [
                t for t in self._agent_tasks[agent_id]
                if now - t < self.window_seconds
            ]

            return len(self._agent_tasks[agent_id]) < self.max_per_agent

    def record(self, agent_id: str):
        """Record a task execution for this agent."""
        with self._lock:
            if agent_id not in self._agent_tasks:
                self._agent_tasks[agent_id] = []
            self._agent_tasks[agent_id].append(time.time())

    def get_stats(self) -> Dict[str, int]:
        """Get current task counts per agent."""
        now = time.time()
        with self._lock:
            return {
                agent_id: len([
                    t for t in timestamps
                    if now - t < self.window_seconds
                ])
                for agent_id, timestamps in self._agent_tasks.items()
            }


# ============== Deadlock Detector ==============

class DeadlockDetector:
    """
    Detects circular dependencies in the task graph.
    Uses DFS to find cycles.
    """

    def __init__(self):
        self._graph: Dict[str, List[str]] = {}  # task_id -> [dependency_ids]
        self._lock = threading.Lock()

    def add_task(self, task_id: str, dependencies: List[str]):
        """Add a task with its dependencies to the graph."""
        with self._lock:
            self._graph[task_id] = dependencies

    def remove_task(self, task_id: str):
        """Remove a completed task from the graph."""
        with self._lock:
            self._graph.pop(task_id, None)

    def detect_cycle(self) -> Optional[List[str]]:
        """
        Detect if there's a circular dependency.
        Returns the cycle path if found, None otherwise.
        """
        with self._lock:
            visited = set()
            rec_stack = set()

            for node in self._graph:
                cycle = self._dfs(node, visited, rec_stack, [])
                if cycle:
                    return cycle

            return None

    def _dfs(
        self,
        node: str,
        visited: set,
        rec_stack: set,
        path: List[str]
    ) -> Optional[List[str]]:
        """DFS to detect cycle."""
        if node in rec_stack:
            cycle_start = path.index(node)
            return path[cycle_start:] + [node]

        if node in visited:
            return None

        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for dep in self._graph.get(node, []):
            cycle = self._dfs(dep, visited, rec_stack, path)
            if cycle:
                return cycle

        path.pop()
        rec_stack.discard(node)
        return None

    def would_create_cycle(self, task_id: str, dependencies: List[str]) -> bool:
        """Check if adding this task would create a cycle."""
        with self._lock:
            # Temporarily add and check
            old = self._graph.get(task_id)
            self._graph[task_id] = dependencies

            cycle = None
            visited = set()
            rec_stack = set()
            for node in self._graph:
                cycle = self._dfs(node, visited, rec_stack, [])
                if cycle:
                    break

            # Restore
            if old is not None:
                self._graph[task_id] = old
            else:
                del self._graph[task_id]

            return cycle is not None


# ============== Retry with Jitter ==============

def calculate_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate retry delay with exponential backoff and jitter.
    Jitter prevents the thundering herd problem.

    Uses "full jitter" strategy: delay = random(0, min(max_delay, base * 2^attempt))
    """
    exponential = min(max_delay, base_delay * (2 ** attempt))

    if jitter:
        # Full jitter: random value between 0 and the exponential delay
        return random.uniform(0, exponential)
    else:
        return exponential


# ============== Priority Task Queue ==============

class TaskQueue:
    """
    Priority-based task queue with production features.

    Features:
    - Priority ordering (CRITICAL first)
    - Rate limiting per agent
    - Timeout enforcement
    - Deadlock detection
    - Retry with jitter

    Usage:
        queue = TaskQueue()

        # Add tasks
        task_id = queue.enqueue(
            name="process_data",
            func=my_function,
            priority=Priority.HIGH,
            timeout_seconds=60
        )

        # Process next task
        result = queue.process_next()
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_per_agent: int = 5,
        rate_limit_window: float = 60.0
    ):
        self._heap: List[QueuedTask] = []
        self._tasks: Dict[str, QueuedTask] = {}
        self._running: Dict[str, QueuedTask] = {}
        self._completed: List[QueuedTask] = []
        self._lock = threading.Lock()

        self.max_concurrent = max_concurrent
        self.rate_limiter = TaskRateLimiter(max_per_agent, rate_limit_window)
        self.deadlock_detector = DeadlockDetector()

        self._draining = False
        self._drain_lock = threading.Lock()

        self._total_enqueued = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_timed_out = 0

    def enqueue(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Dict = None,
        priority: Priority = Priority.NORMAL,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        agent_id: str = None,
        dependencies: List[str] = None
    ) -> str:
        """Add a task to the queue. Returns task ID."""
        deps = dependencies or []

        # Check for deadlock
        task_id = str(uuid4())[:8]
        if deps and self.deadlock_detector.would_create_cycle(task_id, deps):
            raise DeadlockError(
                f"Adding task '{name}' with deps {deps} would create a circular dependency"
            )

        task = QueuedTask(
            id=task_id,
            name=name,
            priority=priority,
            func=func,
            args=args,
            kwargs=kwargs or {},
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            agent_id=agent_id,
            dependencies=deps
        )

        with self._lock:
            if self._draining:
                raise RuntimeError(f"Queue is draining — rejecting new task '{name}'")

            heapq.heappush(self._heap, task)
            self._tasks[task.id] = task
            self._total_enqueued += 1

            if deps:
                self.deadlock_detector.add_task(task.id, deps)

        logger.info(
            f"Enqueued task '{name}' ({task.id}) "
            f"priority={priority.name}"
        )
        return task.id

    def process_next(self) -> Optional[QueuedTask]:
        """
        Process the next highest-priority task.
        Returns the task if processed, None if queue is empty or at capacity.
        """
        with self._lock:
            # Check concurrent limit
            if len(self._running) >= self.max_concurrent:
                return None

            # Find next eligible task
            task = self._find_next_eligible()
            if not task:
                return None

            # Mark as running
            task.status = QueuedTaskStatus.RUNNING
            task.started_at = time.time()
            self._running[task.id] = task

            # Rate limit tracking
            if task.agent_id:
                self.rate_limiter.record(task.agent_id)

        # Execute outside the lock
        try:
            if task.func:
                task.result = task.func(*task.args, **task.kwargs)
            task.status = QueuedTaskStatus.COMPLETED
            task.completed_at = time.time()

            with self._lock:
                self._running.pop(task.id, None)
                self._completed.append(task)
                self._total_completed += 1
                self.deadlock_detector.remove_task(task.id)

            logger.info(f"Task '{task.name}' ({task.id}) completed")

        except Exception as e:
            task.error = str(e)
            task.completed_at = time.time()

            with self._lock:
                self._running.pop(task.id, None)

            # Retry logic with backoff persistence (v2026.2.26)
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.last_attempt_at = time.time()
                delay = calculate_retry_delay(task.retry_count)
                logger.warning(
                    f"Task '{task.name}' failed (attempt {task.retry_count}/"
                    f"{task.max_retries}), retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

                task.status = QueuedTaskStatus.QUEUED
                task.error = None
                with self._lock:
                    heapq.heappush(self._heap, task)
            else:
                task.status = QueuedTaskStatus.FAILED
                with self._lock:
                    self._completed.append(task)
                    self._total_failed += 1
                    self.deadlock_detector.remove_task(task.id)
                logger.error(
                    f"Task '{task.name}' ({task.id}) failed permanently: {e}"
                )

        return task

    def _find_next_eligible(self) -> Optional[QueuedTask]:
        """Find next eligible task from heap (must hold lock)."""
        skipped = []

        while self._heap:
            task = heapq.heappop(self._heap)

            # Skip cancelled tasks
            if task.status == QueuedTaskStatus.CANCELLED:
                continue

            # Check dependencies
            deps_met = all(
                self._tasks.get(dep, QueuedTask()).status == QueuedTaskStatus.COMPLETED
                for dep in task.dependencies
            )
            if not deps_met:
                skipped.append(task)
                continue

            # Check rate limit
            if task.agent_id and not self.rate_limiter.allow(task.agent_id):
                skipped.append(task)
                continue

            # Put skipped tasks back
            for s in skipped:
                heapq.heappush(self._heap, s)

            return task

        # Put skipped tasks back
        for s in skipped:
            heapq.heappush(self._heap, s)

        return None

    def check_timeouts(self) -> List[QueuedTask]:
        """Check for timed-out tasks and cancel them."""
        timed_out = []
        now = time.time()

        with self._lock:
            for task_id, task in list(self._running.items()):
                if task.started_at and (now - task.started_at) > task.timeout_seconds:
                    task.status = QueuedTaskStatus.TIMED_OUT
                    task.error = f"Timed out after {task.timeout_seconds}s"
                    task.completed_at = now
                    self._running.pop(task_id)
                    self._completed.append(task)
                    self._total_timed_out += 1
                    timed_out.append(task)
                    logger.warning(
                        f"Task '{task.name}' ({task.id}) timed out "
                        f"after {task.timeout_seconds}s"
                    )

        return timed_out

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued or running task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status in (QueuedTaskStatus.QUEUED, QueuedTaskStatus.RUNNING):
                task.status = QueuedTaskStatus.CANCELLED
                task.completed_at = time.time()
                self._running.pop(task_id, None)
                self.deadlock_detector.remove_task(task_id)
                logger.info(f"Cancelled task '{task.name}' ({task_id})")
                return True

            return False

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def queue_size(self) -> int:
        """Get number of queued tasks."""
        with self._lock:
            return len(self._heap)

    def running_count(self) -> int:
        """Get number of running tasks."""
        with self._lock:
            return len(self._running)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "queued": len(self._heap),
                "running": len(self._running),
                "completed": self._total_completed,
                "failed": self._total_failed,
                "timed_out": self._total_timed_out,
                "total_enqueued": self._total_enqueued,
                "draining": self._draining,
                "max_concurrent": self.max_concurrent,
                "rate_limits": self.rate_limiter.get_stats(),
            }

    def start_drain(self):
        with self._drain_lock:
            self._draining = True

    def stop_drain(self):
        with self._drain_lock:
            self._draining = False

    def drain_and_wait(self, timeout: float = 60.0) -> bool:
        self.start_drain()
        try:
            start = time.time()
            while time.time() - start < timeout:
                with self._lock:
                    if len(self._running) == 0:
                        return True
                time.sleep(0.5)
            return False
        finally:
            self.stop_drain()

    def recover_deferred(self) -> List[str]:
        recovered = []
        now = time.time()
        with self._lock:
            to_recover = []
            remaining = []
            while self._heap:
                task = heapq.heappop(self._heap)
                if (task.status == QueuedTaskStatus.QUEUED and
                    task.last_attempt_at is not None):
                    backoff = calculate_retry_delay(task.retry_count, jitter=False)
                    if now - task.last_attempt_at >= backoff:
                        to_recover.append(task)
                    else:
                        remaining.append(task)
                else:
                    remaining.append(task)
            for t in remaining:
                heapq.heappush(self._heap, t)
            for task in to_recover:
                task.last_attempt_at = None
                heapq.heappush(self._heap, task)
                recovered.append(task.id)
        return recovered


class DeadlockError(Exception):
    """Raised when a circular dependency is detected."""
    pass


# ============== Global Instance ==============

_task_queue: Optional[TaskQueue] = None


def get_task_queue(max_concurrent: int = 10) -> TaskQueue:
    """Get global task queue."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(max_concurrent=max_concurrent)
    return _task_queue


__all__ = [
    "Priority",
    "QueuedTaskStatus",
    "QueuedTask",
    "TaskRateLimiter",
    "DeadlockDetector",
    "DeadlockError",
    "TaskQueue",
    "calculate_retry_delay",
    "get_task_queue",
]
