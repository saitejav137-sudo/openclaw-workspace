"""
Scheduled Task Runner for OpenClaw

Cron-like task scheduling for persistent agent operations:
- Cron expression parsing
- Interval-based and time-based scheduling
- Persistent task storage
- Missed run catch-up
"""

import time
import threading
import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import os

from .logger import get_logger

logger = get_logger("scheduler")


class ScheduleType(Enum):
    """Types of schedules."""
    INTERVAL = "interval"      # Every N seconds/minutes/hours
    DAILY = "daily"            # At specific time daily
    CRON = "cron"              # Cron expression
    ONCE = "once"              # Run once at specific time


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled task."""
    schedule_type: ScheduleType
    interval_seconds: Optional[float] = None      # For INTERVAL
    time_of_day: Optional[str] = None              # For DAILY: "HH:MM"
    cron_expression: Optional[str] = None          # For CRON: "*/5 * * * *"
    run_at: Optional[float] = None                 # For ONCE: timestamp


@dataclass
class ScheduledTask:
    """A scheduled task."""
    id: str
    name: str
    schedule: ScheduleConfig
    func: Optional[Callable] = None
    func_name: str = ""  # For serialization
    enabled: bool = True
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    max_runs: int = 0  # 0 = unlimited
    catch_up_missed: bool = True

    @property
    def is_due(self) -> bool:
        """Check if the task is due to run."""
        if not self.enabled:
            return False
        if self.max_runs > 0 and self.run_count >= self.max_runs:
            return False
        if self.next_run is None:
            return True
        return time.time() >= self.next_run


class CronParser:
    """
    Simple cron expression parser.
    Supports: minute hour day_of_month month day_of_week
    Special: * (any), */N (every N), N-M (range), N,M (list)
    """

    @staticmethod
    def parse_field(field_str: str, min_val: int, max_val: int) -> List[int]:
        """Parse a single cron field into a list of valid values."""
        values = set()

        for part in field_str.split(","):
            part = part.strip()

            if part == "*":
                values.update(range(min_val, max_val + 1))
            elif part.startswith("*/"):
                step = int(part[2:])
                values.update(range(min_val, max_val + 1, step))
            elif "-" in part:
                start, end = part.split("-")
                values.update(range(int(start), int(end) + 1))
            else:
                values.add(int(part))

        return sorted(v for v in values if min_val <= v <= max_val)

    @classmethod
    def matches(cls, expression: str, dt: datetime) -> bool:
        """Check if a datetime matches a cron expression."""
        parts = expression.strip().split()
        if len(parts) != 5:
            return False

        minute, hour, dom, month, dow = parts

        minute_vals = cls.parse_field(minute, 0, 59)
        hour_vals = cls.parse_field(hour, 0, 23)
        dom_vals = cls.parse_field(dom, 1, 31)
        month_vals = cls.parse_field(month, 1, 12)
        dow_vals = cls.parse_field(dow, 0, 6)

        return (
            dt.minute in minute_vals and
            dt.hour in hour_vals and
            dt.day in dom_vals and
            dt.month in month_vals and
            dt.weekday() in dow_vals
        )

    @classmethod
    def next_match(cls, expression: str, after: datetime = None) -> Optional[datetime]:
        """Find the next datetime that matches a cron expression."""
        if after is None:
            after = datetime.now()

        # Search up to 366 days ahead
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        max_time = after + timedelta(days=366)

        while current < max_time:
            if cls.matches(expression, current):
                return current
            current += timedelta(minutes=1)

        return None


class TaskScheduler:
    """
    Main scheduler that manages and runs scheduled tasks.

    Usage:
        scheduler = TaskScheduler()

        # Add tasks
        scheduler.add_task(
            "health_check",
            lambda: check_health(),
            ScheduleConfig(ScheduleType.INTERVAL, interval_seconds=300)
        )

        scheduler.add_task(
            "daily_report",
            lambda: generate_report(),
            ScheduleConfig(ScheduleType.DAILY, time_of_day="08:00")
        )

        # Start the scheduler
        scheduler.start()
    """

    def __init__(self, storage_dir: str = "~/.openclaw/scheduler"):
        self.storage_dir = os.path.expanduser(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._check_interval = 10  # seconds

        self._load_state()

    def add_task(
        self,
        name: str,
        func: Callable,
        schedule: ScheduleConfig,
        max_runs: int = 0,
        catch_up_missed: bool = True
    ) -> str:
        """Add a scheduled task."""
        import hashlib
        task_id = hashlib.sha256(name.encode()).hexdigest()[:10]

        task = ScheduledTask(
            id=task_id,
            name=name,
            schedule=schedule,
            func=func,
            func_name=getattr(func, '__name__', str(func)),
            max_runs=max_runs,
            catch_up_missed=catch_up_missed
        )

        task.next_run = self._calculate_next_run(task)

        with self._lock:
            self._tasks[task_id] = task

        logger.info(
            f"Scheduled task '{name}' ({schedule.schedule_type.value}) "
            f"next_run={datetime.fromtimestamp(task.next_run).strftime('%H:%M:%S') if task.next_run else 'now'}"
        )

        return task_id

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        with self._lock:
            if task_id in self._tasks:
                name = self._tasks[task_id].name
                del self._tasks[task_id]
                logger.info(f"Removed scheduled task: {name}")
                return True
            return False

    def enable_task(self, task_id: str):
        """Enable a task."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = True

    def disable_task(self, task_id: str):
        """Disable a task."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = False

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._save_state()
        logger.info("Scheduler stopped")

    def _run_loop(self):
        """Main scheduler loop."""
        while self._running and not self._stop_event.is_set():
            self._check_and_run()
            self._stop_event.wait(timeout=self._check_interval)

    def _check_and_run(self):
        """Check for due tasks and run them."""
        with self._lock:
            due_tasks = [t for t in self._tasks.values() if t.is_due]

        for task in due_tasks:
            self._execute_task(task)

    def _execute_task(self, task: ScheduledTask):
        """Execute a single task."""
        logger.info(f"Running scheduled task: {task.name}")
        start = time.time()

        try:
            if task.func:
                task.func()
            task.run_count += 1
            task.last_run = time.time()
            task.last_error = None
            logger.info(f"Task '{task.name}' completed in {time.time() - start:.2f}s")

        except Exception as e:
            task.error_count += 1
            task.last_error = str(e)
            task.last_run = time.time()
            logger.error(f"Task '{task.name}' failed: {e}")

        # Calculate next run
        task.next_run = self._calculate_next_run(task)

    def _calculate_next_run(self, task: ScheduledTask) -> Optional[float]:
        """Calculate the next run time for a task."""
        schedule = task.schedule

        if schedule.schedule_type == ScheduleType.INTERVAL:
            base = task.last_run or time.time()
            return base + (schedule.interval_seconds or 60)

        elif schedule.schedule_type == ScheduleType.DAILY:
            if schedule.time_of_day:
                now = datetime.now()
                parts = schedule.time_of_day.split(":")
                target = now.replace(
                    hour=int(parts[0]),
                    minute=int(parts[1]) if len(parts) > 1 else 0,
                    second=0, microsecond=0
                )
                if target <= now:
                    target += timedelta(days=1)
                return target.timestamp()
            return time.time() + 86400

        elif schedule.schedule_type == ScheduleType.CRON:
            if schedule.cron_expression:
                next_dt = CronParser.next_match(schedule.cron_expression)
                if next_dt:
                    return next_dt.timestamp()
            return time.time() + 60

        elif schedule.schedule_type == ScheduleType.ONCE:
            return schedule.run_at

        return time.time() + 60

    def run_now(self, task_id: str):
        """Manually trigger a task to run immediately."""
        with self._lock:
            task = self._tasks.get(task_id)
        if task:
            self._execute_task(task)

    def list_tasks(self) -> List[Dict]:
        """List all scheduled tasks."""
        with self._lock:
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "type": t.schedule.schedule_type.value,
                    "enabled": t.enabled,
                    "run_count": t.run_count,
                    "error_count": t.error_count,
                    "last_run": t.last_run,
                    "next_run": t.next_run,
                    "last_error": t.last_error,
                }
                for t in self._tasks.values()
            ]

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "enabled": sum(1 for t in self._tasks.values() if t.enabled),
                "total_runs": sum(t.run_count for t in self._tasks.values()),
                "total_errors": sum(t.error_count for t in self._tasks.values()),
                "running": self._running,
            }

    def _save_state(self):
        """Save scheduler state to disk."""
        try:
            filepath = os.path.join(self.storage_dir, "state.json")
            data = {
                "tasks": {
                    tid: {
                        "id": t.id, "name": t.name,
                        "schedule_type": t.schedule.schedule_type.value,
                        "interval_seconds": t.schedule.interval_seconds,
                        "time_of_day": t.schedule.time_of_day,
                        "cron_expression": t.schedule.cron_expression,
                        "enabled": t.enabled, "last_run": t.last_run,
                        "run_count": t.run_count, "error_count": t.error_count,
                        "func_name": t.func_name
                    }
                    for tid, t in self._tasks.items()
                }
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")

    def _load_state(self):
        """Load scheduler state from disk."""
        try:
            filepath = os.path.join(self.storage_dir, "state.json")
            if not os.path.exists(filepath):
                return
            # State loaded, but functions need to be re-registered
            logger.info("Scheduler state file exists (tasks need re-registration)")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")


# ============== Global Instance ==============

_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def schedule_interval(name: str, func: Callable, seconds: float) -> str:
    """Quick schedule: run every N seconds."""
    return get_scheduler().add_task(
        name, func,
        ScheduleConfig(ScheduleType.INTERVAL, interval_seconds=seconds)
    )


def schedule_daily(name: str, func: Callable, time_str: str) -> str:
    """Quick schedule: run daily at HH:MM."""
    return get_scheduler().add_task(
        name, func,
        ScheduleConfig(ScheduleType.DAILY, time_of_day=time_str)
    )


def schedule_cron(name: str, func: Callable, expression: str) -> str:
    """Quick schedule: run on cron expression."""
    return get_scheduler().add_task(
        name, func,
        ScheduleConfig(ScheduleType.CRON, cron_expression=expression)
    )



# ============== Backward-Compatible Wrapper ==============

class Scheduler(TaskScheduler):
    """
    Backward-compatible wrapper for TaskScheduler.
    Provides the old add_job/get_job API used by existing code.
    """

    def add_job(
        self,
        name: str,
        schedule_type: ScheduleType = ScheduleType.INTERVAL,
        callback: Callable = None,
        interval_seconds: float = 60.0,
        time_of_day: str = None,
        cron_expression: str = None,
        **kwargs
    ) -> str:
        """Add a job (backward-compatible wrapper for add_task)."""
        config = ScheduleConfig(
            schedule_type=schedule_type,
            interval_seconds=interval_seconds,
            time_of_day=time_of_day,
            cron_expression=cron_expression
        )
        task_id = self.add_task(name, callback or (lambda: None), config, **kwargs)
        return task_id

    def get_job(self, name: str):
        """Get a job by name (backward-compatible wrapper)."""
        with self._lock:
            for task in self._tasks.values():
                if task.name == name:
                    return task
        return None


__all__ = [
    "ScheduleType",
    "ScheduleConfig",
    "ScheduledTask",
    "CronParser",
    "TaskScheduler",
    "Scheduler",
    "get_scheduler",
    "schedule_interval",
    "schedule_daily",
    "schedule_cron",
]
