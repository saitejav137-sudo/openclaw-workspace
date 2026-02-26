"""
Scheduler for OpenClaw

Cron-like scheduled triggers and jobs.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re

logger = logging.getLogger("openclaw.scheduler")


class ScheduleType(Enum):
    """Schedule types"""
    INTERVAL = "interval"      # Every N seconds
    CRON = "cron"             # Cron-style schedule
    ONCE = "once"             # Run once at specific time
    DAILY = "daily"           # Run daily at specific time
    WEEKLY = "weekly"          # Run weekly on specific day


@dataclass
class ScheduleJob:
    """Scheduled job definition"""
    name: str
    schedule_type: ScheduleType
    interval_seconds: float = 0
    cron_expression: str = ""
    run_at: Optional[datetime] = None
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    hour: int = 0
    minute: int = 0
    enabled: bool = True
    callback: Callable = None
    config: Dict = field(default_factory=dict)
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    run_count: int = 0
    max_runs: int = 0  # 0 = unlimited


class Scheduler:
    """Scheduler for running jobs at specified times"""

    def __init__(self):
        self.jobs: Dict[str, ScheduleJob] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_job(
        self,
        name: str,
        schedule_type: ScheduleType,
        callback: Callable,
        interval_seconds: float = None,
        cron_expression: str = None,
        run_at: datetime = None,
        day_of_week: int = None,
        hour: int = 0,
        minute: int = 0,
        max_runs: int = 0,
        config: Dict = None,
        enabled: bool = True
    ) -> ScheduleJob:
        """Add a job to the scheduler"""
        with self._lock:
            job = ScheduleJob(
                name=name,
                schedule_type=schedule_type,
                interval_seconds=interval_seconds or 60.0,
                cron_expression=cron_expression or "",
                run_at=run_at,
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                callback=callback,
                config=config or {},
                enabled=enabled,
                max_runs=max_runs
            )

            # Calculate next run time
            job.next_run = self._calculate_next_run(job)

            self.jobs[name] = job
            logger.info(f"Job added: {name} ({schedule_type.value})")

            return job

    def remove_job(self, name: str) -> bool:
        """Remove a job from the scheduler"""
        with self._lock:
            if name in self.jobs:
                del self.jobs[name]
                logger.info(f"Job removed: {name}")
                return True
            return False

    def enable_job(self, name: str) -> bool:
        """Enable a job"""
        with self._lock:
            if name in self.jobs:
                self.jobs[name].enabled = True
                self.jobs[name].next_run = self._calculate_next_run(self.jobs[name])
                logger.info(f"Job enabled: {name}")
                return True
            return False

    def disable_job(self, name: str) -> bool:
        """Disable a job"""
        with self._lock:
            if name in self.jobs:
                self.jobs[name].enabled = False
                self.jobs[name].next_run = None
                logger.info(f"Job disabled: {name}")
                return True
            return False

    def get_job(self, name: str) -> Optional[ScheduleJob]:
        """Get a job by name"""
        return self.jobs.get(name)

    def list_jobs(self) -> List[Dict]:
        """List all jobs"""
        with self._lock:
            return [
                {
                    "name": job.name,
                    "type": job.schedule_type.value,
                    "enabled": job.enabled,
                    "last_run": job.last_run,
                    "next_run": job.next_run,
                    "run_count": job.run_count
                }
                for job in self.jobs.values()
            ]

    def start(self):
        """Start the scheduler"""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def _run_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                current_time = time.time()

                with self._lock:
                    for job in self.jobs.values():
                        if not job.enabled:
                            continue

                        if job.next_run and current_time >= job.next_run:
                            # Run the job
                            logger.info(f"Running job: {job.name}")
                            try:
                                if job.callback:
                                    job.callback(job.config)
                                job.last_run = current_time
                                job.run_count += 1

                                # Check max runs
                                if job.max_runs > 0 and job.run_count >= job.max_runs:
                                    job.enabled = False
                                    logger.info(f"Job {job.name} reached max runs")
                                    continue

                            except Exception as e:
                                logger.error(f"Job {job.name} error: {e}")

                            # Calculate next run
                            job.next_run = self._calculate_next_run(job)

                # Sleep for 1 second
                time.sleep(1)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(1)

    def _calculate_next_run(self, job: ScheduleJob) -> Optional[float]:
        """Calculate next run time for a job"""
        now = datetime.now()

        if job.schedule_type == ScheduleType.INTERVAL:
            if job.last_run:
                return job.last_run + job.interval_seconds
            return time.time() + job.interval_seconds

        elif job.schedule_type == ScheduleType.ONCE:
            if job.run_at:
                if job.run_at > now:
                    return job.run_at.timestamp()
            return None

        elif job.schedule_type == ScheduleType.DAILY:
            next_run = now.replace(hour=job.hour, minute=job.minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run.timestamp()

        elif job.schedule_type == ScheduleType.WEEKLY:
            days_ahead = job.day_of_week - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now.replace(hour=job.hour, minute=job.minute, second=0, microsecond=0)
            next_run += timedelta(days=days_ahead)
            if next_run <= now:
                next_run += timedelta(days=7)
            return next_run.timestamp()

        return None


# Convenience functions
def create_interval_job(
    name: str,
    interval_seconds: float,
    callback: Callable,
    config: Dict = None
) -> ScheduleJob:
    """Create an interval-based job"""
    scheduler = Scheduler.get_instance()
    return scheduler.add_job(
        name=name,
        schedule_type=ScheduleType.INTERVAL,
        callback=callback,
        interval_seconds=interval_seconds,
        config=config
    )


def create_daily_job(
    name: str,
    hour: int,
    minute: int,
    callback: Callable,
    config: Dict = None
) -> ScheduleJob:
    """Create a daily job"""
    scheduler = Scheduler.get_instance()
    return scheduler.add_job(
        name=name,
        schedule_type=ScheduleType.DAILY,
        callback=callback,
        hour=hour,
        minute=minute,
        config=config
    )


# Global scheduler instance
_global_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get the global scheduler instance"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = Scheduler()
    return _global_scheduler


class SchedulerInstance:
    """Singleton for scheduler access"""

    _instance: Optional[Scheduler] = None

    @classmethod
    def get_instance(cls) -> Scheduler:
        if cls._instance is None:
            cls._instance = Scheduler()
        return cls._instance


# Export
__all__ = [
    "ScheduleType",
    "ScheduleJob",
    "Scheduler",
    "create_interval_job",
    "create_daily_job",
    "get_scheduler",
]
