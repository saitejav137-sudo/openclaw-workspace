"""
Async Message Queue for OpenClaw

Redis-based message queue for async task processing.
Provides reliable task queuing and background processing.
"""

import time
import json
import uuid
import threading
import queue
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from ..core.logger import get_logger

logger = get_logger("mq")


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Task message"""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: int = TaskPriority.NORMAL.value
    created_at: float = 0
    scheduled_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({
            "id": self.id,
            "type": self.type,
            "payload": self.payload,
            "priority": self.priority,
            "created_at": self.created_at,
            "scheduled_at": self.scheduled_at,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        })

    @classmethod
    def from_json(cls, data: str) -> 'Task':
        """Deserialize from JSON"""
        d = json.loads(data)
        return cls(
            id=d["id"],
            type=d["type"],
            payload=d["payload"],
            priority=d.get("priority", TaskPriority.NORMAL.value),
            created_at=d.get("created_at", time.time()),
            scheduled_at=d.get("scheduled_at"),
            retries=d.get("retries", 0),
            max_retries=d.get("max_retries", 3),
            metadata=d.get("metadata", {})
        )


class InMemoryMessageQueue:
    """In-memory message queue (fallback when Redis unavailable)"""

    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._workers: List[threading.Thread] = []
        self._handlers: Dict[str, Callable] = {}

    def enqueue(self, task: Task) -> bool:
        """Add task to queue"""
        try:
            # Priority is inverted (lower number = higher priority)
            self._queue.put((-task.priority, task))
            logger.debug(f"Task enqueued: {task.id}")
            return True
        except Exception as e:
            logger.error(f"Enqueue error: {e}")
            return False

    def dequeue(self, timeout: float = 1.0) -> Optional[Task]:
        """Get task from queue"""
        try:
            _, task = self._queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None

    def complete(self, task_id: str, result: Any):
        """Mark task as complete"""
        with self._lock:
            self._results[task_id] = {
                "status": "completed",
                "result": result,
                "completed_at": time.time()
            }

    def fail(self, task_id: str, error: str):
        """Mark task as failed"""
        with self._lock:
            self._results[task_id] = {
                "status": "failed",
                "error": error,
                "failed_at": time.time()
            }

    def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[Dict]:
        """Get task result"""
        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                if task_id in self._results:
                    return self._results.pop(task_id)
            time.sleep(0.1)

        return None

    def register_handler(self, task_type: str, handler: Callable):
        """Register task handler"""
        self._handlers[task_type] = handler
        logger.info(f"Handler registered for: {task_type}")

    def start_workers(self, count: int = 4):
        """Start worker threads"""
        self._running = True

        for i in range(count):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"mq-worker-{i}"
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {count} workers")

    def stop_workers(self):
        """Stop worker threads"""
        self._running = False

        for worker in self._workers:
            worker.join(timeout=5)

        self._workers.clear()
        logger.info("Workers stopped")

    def _worker_loop(self):
        """Worker thread loop"""
        while self._running:
            task = self.dequeue()

            if task is None:
                continue

            logger.debug(f"Processing task: {task.type}")

            try:
                # Get handler
                handler = self._handlers.get(task.type)

                if handler:
                    result = handler(task.payload)
                    self.complete(task.id, result)
                else:
                    logger.warning(f"No handler for task type: {task.type}")

            except Exception as e:
                logger.error(f"Task processing error: {e}")

                # Retry if possible
                if task.retries < task.max_retries:
                    task.retries += 1
                    self.enqueue(task)
                else:
                    self.fail(task.id, str(e))

    def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            "pending": self._queue.qsize(),
            "results": len(self._results),
            "workers": len(self._workers),
            "handlers": list(self._handlers.keys())
        }


class RedisMessageQueue:
    """Redis-based message queue for distributed systems"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._queue_name = "openclaw:tasks"
        self._results_name = "openclaw:results"

    def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import redis
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            self._redis.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
            return True
        except ImportError:
            logger.warning("redis-py not installed, using in-memory queue")
            return False
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            return False

    def enqueue(self, task: Task) -> bool:
        """Add task to Redis queue"""
        if not self._redis:
            return False

        try:
            self._redis.zadd(
                self._queue_name,
                {task.to_json(): -task.priority}  # Lower score = higher priority
            )
            return True
        except Exception as e:
            logger.error(f"Redis enqueue error: {e}")
            return False

    def dequeue(self, timeout: float = 1.0) -> Optional[Task]:
        """Get task from Redis queue"""
        if not self._redis:
            return None

        try:
            result = self._redis.zpopmin(self._queue_name, count=1)

            if result:
                _, data = result[0]
                return Task.from_json(data)

            return None

        except Exception as e:
            logger.error(f"Redis dequeue error: {e}")
            return None

    def complete(self, task_id: str, result: Any):
        """Store task result in Redis"""
        if not self._redis:
            return

        try:
            self._redis.hset(
                self._results_name,
                task_id,
                json.dumps({
                    "status": "completed",
                    "result": result,
                    "completed_at": time.time()
                })
            )

            # Auto-expire results after 1 hour
            self._redis.expire(self._results_name, 3600)

        except Exception as e:
            logger.error(f"Redis complete error: {e}")

    def get_stats(self) -> Dict:
        """Get Redis queue stats"""
        if not self._redis:
            return {}

        try:
            return {
                "pending": self._redis.zcard(self._queue_name),
                "results": self._redis.hlen(self._results_name)
            }
        except Exception:
            return {}


# Task queue factory
def create_message_queue(redis_url: Optional[str] = None) -> InMemoryMessageQueue:
    """Create message queue instance"""
    # Try Redis first if URL provided
    if redis_url:
        redis_mq = RedisMessageQueue(redis_url)
        if redis_mq.connect():
            # Wrap Redis with InMemory for handlers
            # In production, you'd use a proper Redis queue
            pass

    # Fallback to in-memory
    return InMemoryMessageQueue()


# Global queue instance
_queue: Optional[InMemoryMessageQueue] = None


def get_message_queue() -> InMemoryMessageQueue:
    """Get global message queue"""
    global _queue

    if _queue is None:
        _queue = create_message_queue()

    return _queue


def enqueue_task(task_type: str, payload: Dict, priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Quick way to enqueue a task"""
    task = Task(
        id=str(uuid.uuid4()),
        type=task_type,
        payload=payload,
        priority=priority.value
    )

    queue = get_message_queue()
    queue.enqueue(task)

    return task.id


def register_task_handler(task_type: str, handler: Callable):
    """Register a task handler"""
    queue = get_message_queue()
    queue.register_handler(task_type, handler)


__all__ = [
    "Task",
    "TaskPriority",
    "InMemoryMessageQueue",
    "RedisMessageQueue",
    "create_message_queue",
    "get_message_queue",
    "enqueue_task",
    "register_task_handler",
]
