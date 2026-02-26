"""Storage module - Database and Cache"""

import os
import time
import json
import sqlite3
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime

from core.logger import get_logger

logger = get_logger("storage")


@dataclass
class TriggerRecord:
    """Trigger event record"""
    id: int
    timestamp: float
    mode: str
    condition_met: bool
    triggered: bool
    screenshot_path: Optional[str]
    action_taken: Optional[str]
    success: bool
    context_json: Optional[str]


class DatabaseManager:
    """SQLite database for trigger history"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.expanduser("~/.openclaw/triggers.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = None
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> 'DatabaseManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def _init_db(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Triggers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                mode TEXT NOT NULL,
                condition_met INTEGER NOT NULL,
                triggered INTEGER NOT NULL,
                screenshot_path TEXT,
                action_taken TEXT,
                success INTEGER DEFAULT 1,
                context_json TEXT
            )
        """)

        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON triggers(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mode ON triggers(mode)
        """)

        conn.commit()
        logger.debug("Database initialized")

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def log_trigger(
        self,
        mode: str,
        condition_met: bool,
        triggered: bool,
        screenshot_path: Optional[str] = None,
        action_taken: Optional[str] = None,
        success: bool = True,
        context: Optional[Dict] = None
    ) -> int:
        """Log a trigger event"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO triggers (
                timestamp, mode, condition_met, triggered,
                screenshot_path, action_taken, success, context_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            mode,
            int(condition_met),
            int(triggered),
            screenshot_path,
            action_taken,
            int(success),
            json.dumps(context) if context else None
        ))

        conn.commit()
        return cursor.lastrowid

    def get_recent(self, limit: int = 100) -> List[TriggerRecord]:
        """Get recent trigger records"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, mode, condition_met, triggered,
                   screenshot_path, action_taken, success, context_json
            FROM triggers
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [
            TriggerRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                mode=row["mode"],
                condition_met=bool(row["condition_met"]),
                triggered=bool(row["triggered"]),
                screenshot_path=row["screenshot_path"],
                action_taken=row["action_taken"],
                success=bool(row["success"]),
                context_json=row["context_json"]
            )
            for row in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM triggers")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM triggers WHERE triggered = 1")
        triggered = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM triggers WHERE success = 0")
        failed = cursor.fetchone()[0]

        cursor.execute("""
            SELECT mode, COUNT(*) as count
            FROM triggers
            GROUP BY mode
        """)
        by_mode = {row[0]: row[1] for row in cursor.fetchall()}

        success_rate = ((triggered - failed) / triggered * 100) if triggered > 0 else 0.0

        return {
            "total": total,
            "triggered": triggered,
            "failed": failed,
            "success_rate": success_rate,
            "by_mode": by_mode
        }

    def get_triggers_by_time_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[TriggerRecord]:
        """Get triggers within time range"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, mode, condition_met, triggered,
                   screenshot_path, action_taken, success, context_json
            FROM triggers
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_time, end_time))

        rows = cursor.fetchall()
        return [
            TriggerRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                mode=row["mode"],
                condition_met=bool(row["condition_met"]),
                triggered=bool(row["triggered"]),
                screenshot_path=row["screenshot_path"],
                action_taken=row["action_taken"],
                success=bool(row["success"]),
                context_json=row["context_json"]
            )
            for row in rows
        ]

    def cleanup_old_records(self, days: int = 30) -> int:
        """Delete records older than specified days"""
        cutoff = time.time() - (days * 86400)
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM triggers WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old records")

        return deleted

    def vacuum(self):
        """Optimize database"""
        conn = self.get_connection()
        conn.execute("VACUUM")
        logger.debug("Database vacuumed")

    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None


# Export classes
__all__ = [
    "DatabaseManager",
    "TriggerRecord",
]

# Import new database module
from .database import DatabaseManager as AdvancedDatabaseManager, DatabaseConfig, DatabaseType

# Import vector database
from .vector_db import (
    VectorDatabase,
    VectorEntry,
    SearchResult,
    DistanceMetric,
    get_vector_database,
    semantic_search,
    add_to_index,
)

__all__.extend([
    "VectorDatabase",
    "VectorEntry",
    "SearchResult",
    "DistanceMetric",
    "get_vector_database",
    "semantic_search",
    "add_to_index",
])
