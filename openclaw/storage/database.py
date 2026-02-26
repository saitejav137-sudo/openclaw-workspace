"""
Database Backend Support for OpenClaw

Supports SQLite, PostgreSQL, and MySQL databases.
Includes connection pooling and migrations.
"""

import os
import time
import json
import threading
import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from urllib.parse import urlparse

from core.logger import get_logger

logger = get_logger("database")


class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "openclaw"
    username: str = ""
    password: str = ""
    path: str = ""  # For SQLite
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

    @classmethod
    def from_url(cls, url: str) -> "DatabaseConfig":
        """Parse database URL"""
        parsed = urlparse(url)

        if parsed.scheme == "sqlite":
            return cls(
                type=DatabaseType.SQLITE,
                path=parsed.path or "~/.openclaw/openclaw.db"
            )

        if parsed.scheme == "postgresql":
            return cls(
                type=DatabaseType.POSTGRESQL,
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else "openclaw",
                username=parsed.username or "",
                password=parsed.password or ""
            )

        if parsed.scheme == "mysql":
            return cls(
                type=DatabaseType.MYSQL,
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                database=parsed.path[1:] if parsed.path else "openclaw",
                username=parsed.username or "",
                password=parsed.password or ""
            )

        raise ValueError(f"Unsupported database: {parsed.scheme}")


class DatabaseManager:
    """
    Multi-database manager with connection pooling.
    Supports SQLite, PostgreSQL, and MySQL.
    """

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None
        self._pool = None
        self._connected = False

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "DatabaseManager":
        """Get or create singleton instance"""
        with cls._lock:
            if cls._instance is None:
                if db_path:
                    # Determine type from path
                    if db_path.endswith(".db") or db_path.endswith(".sqlite"):
                        config = DatabaseConfig(type=DatabaseType.SQLITE, path=db_path)
                    else:
                        config = DatabaseConfig.from_url(db_path)
                else:
                    config = DatabaseConfig(type=DatabaseType.SQLITE)

                cls._instance = cls(config)

            return cls._instance

    def connect(self) -> bool:
        """Connect to database"""
        try:
            if self.config.type == DatabaseType.SQLITE:
                return self._connect_sqlite()
            elif self.config.type == DatabaseType.POSTGRESQL:
                return self._connect_postgresql()
            elif self.config.type == DatabaseType.MYSQL:
                return self._connect_mysql()

            return False

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def _connect_sqlite(self) -> bool:
        """Connect to SQLite"""
        try:
            import sqlite3

            # Expand path
            db_path = os.path.expanduser(self.config.path)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            self._connection = sqlite3.connect(
                db_path,
                check_same_thread=False,
                timeout=30
            )
            self._connection.row_factory = sqlite3.Row

            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")

            self._connected = True
            logger.info(f"SQLite connected: {db_path}")

            # Run migrations
            self._migrate()

            return True

        except Exception as e:
            logger.error(f"SQLite connection error: {e}")
            return False

    def _connect_postgresql(self) -> bool:
        """Connect to PostgreSQL"""
        try:
            import psycopg2
            from psycopg2 import pool

            # Create connection pool
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.config.pool_size,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password
            )

            self._connected = True
            logger.info(f"PostgreSQL connected: {self.config.host}:{self.config.port}/{self.config.database}")

            # Run migrations
            self._migrate()

            return True

        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL connection error: {e}")
            return False

    def _connect_mysql(self) -> bool:
        """Connect to MySQL"""
        try:
            import pymysql
            import DBUtils.PooledDB

            # Create connection pool
            self._pool = DBUtils.PooledDB.PooledDB(
                creator=pymysql,
                maxconnections=self.config.pool_size,
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                charset='utf8mb4'
            )

            self._connected = True
            logger.info(f"MySQL connected: {self.config.host}:{self.config.port}/{self.config.database}")

            # Run migrations
            self._migrate()

            return True

        except ImportError:
            logger.error("pymysql not installed. Install with: pip install pymysql")
            return False
        except Exception as e:
            logger.error(f"MySQL connection error: {e}")
            return False

    def _migrate(self):
        """Run database migrations"""
        if self.config.type == DatabaseType.SQLITE:
            self._migrate_sqlite()
        else:
            self._migrate_sql()

    def _migrate_sqlite(self):
        """Migrate SQLite schema"""
        cursor = self._connection.cursor()

        # Triggers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                mode TEXT NOT NULL,
                config TEXT,
                enabled INTEGER DEFAULT 1,
                created_at REAL,
                updated_at REAL
            )
        """)

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_id INTEGER,
                triggered INTEGER,
                timestamp REAL,
                metadata TEXT,
                FOREIGN KEY (trigger_id) REFERENCES triggers(id)
            )
        """)

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT,
                role TEXT DEFAULT 'viewer',
                api_key TEXT,
                created_at REAL,
                last_login REAL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_trigger ON events(trigger_id)")

        self._connection.commit()
        logger.info("SQLite migrations complete")

    def _migrate_sql(self):
        """Migrate SQL (PostgreSQL/MySQL) schema"""
        # Get connection
        conn = self.get_connection()
        if not conn:
            return

        cursor = conn.cursor()

        # Triggers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triggers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                mode VARCHAR(50) NOT NULL,
                config JSONB,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                trigger_id INTEGER REFERENCES triggers(id),
                triggered BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255),
                password_hash VARCHAR(255),
                role VARCHAR(50) DEFAULT 'viewer',
                api_key VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_trigger ON events(trigger_id)")

        conn.commit()
        logger.info("SQL migrations complete")

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        if not self._connected:
            self.connect()

        if self.config.type == DatabaseType.SQLITE:
            yield self._connection
            return

        if self._pool:
            conn = self._pool.connection()
            try:
                yield conn
            finally:
                conn.close()
        else:
            raise RuntimeError("No database connection")

    def execute(self, query: str, params: tuple = ()) -> Optional[Any]:
        """Execute a query"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            if query.strip().upper().startswith("SELECT"):
                if self.config.type == DatabaseType.SQLITE:
                    return cursor.fetchall()
                else:
                    return cursor.fetchall()

            conn.commit()
            return cursor.lastrowid if cursor.lastrowid else cursor.rowcount

    def execute_many(self, query: str, params: List[tuple]) -> int:
        """Execute many queries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params)
            conn.commit()
            return cursor.rowcount

    def fetch_all(self, query: str, params: tuple = ()) -> List:
        """Fetch all rows"""
        return self.execute(query, params) or []

    def fetch_one(self, query: str, params: tuple = ()) -> Optional:
        """Fetch one row"""
        rows = self.execute(query, params)
        return rows[0] if rows else None

    # Trigger operations
    def create_trigger(self, name: str, mode: str, config: Dict) -> int:
        """Create a new trigger"""
        now = time.time()
        config_json = json.dumps(config)

        return self.execute(
            "INSERT INTO triggers (name, mode, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name, mode, config_json, now, now)
        )

    def get_trigger(self, trigger_id: int) -> Optional[Dict]:
        """Get trigger by ID"""
        row = self.fetch_one(
            "SELECT * FROM triggers WHERE id = ?",
            (trigger_id,)
        )

        if row:
            if self.config.type == DatabaseType.SQLITE:
                return dict(row)
            return row._asdict() if hasattr(row, '_asdict') else dict(row)

        return None

    def list_triggers(self) -> List[Dict]:
        """List all triggers"""
        rows = self.fetch_all("SELECT * FROM triggers ORDER BY created_at DESC")

        results = []
        for row in rows:
            if self.config.type == DatabaseType.SQLITE:
                results.append(dict(row))
            else:
                results.append(row._asdict() if hasattr(row, '_asdict') else dict(row))

        return results

    def update_trigger(self, trigger_id: int, **kwargs) -> bool:
        """Update trigger"""
        if not kwargs:
            return False

        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values())
        values.append(time.time())  # updated_at
        values.append(trigger_id)

        self.execute(
            f"UPDATE triggers SET {set_clause}, updated_at = ? WHERE id = ?",
            tuple(values)
        )

        return True

    def delete_trigger(self, trigger_id: int) -> bool:
        """Delete trigger"""
        self.execute("DELETE FROM triggers WHERE id = ?", (trigger_id,))
        return True

    # Event operations
    def log_event(self, trigger_id: Optional[int], triggered: bool, metadata: Dict = None):
        """Log a trigger event"""
        now = time.time()
        metadata_json = json.dumps(metadata or {})

        self.execute(
            "INSERT INTO events (trigger_id, triggered, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (trigger_id, triggered, now, metadata_json)
        )

    def get_events(self, trigger_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Get events"""
        if trigger_id:
            rows = self.fetch_all(
                "SELECT * FROM events WHERE trigger_id = ? ORDER BY timestamp DESC LIMIT ?",
                (trigger_id, limit)
            )
        else:
            rows = self.fetch_all(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

        results = []
        for row in rows:
            if self.config.type == DatabaseType.SQLITE:
                results.append(dict(row))
            else:
                results.append(row._asdict() if hasattr(row, '_asdict') else dict(row))

        return results

    def get_stats(self) -> Dict:
        """Get trigger statistics"""
        total = self.fetch_one("SELECT COUNT(*) as count FROM triggers")
        total = total[0] if total else 0

        events = self.fetch_one("SELECT COUNT(*) as count FROM events WHERE triggered = 1")
        triggered = events[0] if events else 0

        failed = self.fetch_one("SELECT COUNT(*) as count FROM events WHERE triggered = 0")
        failed = failed[0] if failed else 0

        success_rate = (triggered / (triggered + failed) * 100) if (triggered + failed) > 0 else 0

        return {
            "total": total,
            "triggered": triggered,
            "failed": failed,
            "success_rate": success_rate
        }

    def close(self):
        """Close database connection"""
        if self._connected:
            if self.config.type == DatabaseType.SQLITE and self._connection:
                self._connection.close()
            elif self._pool:
                self._pool.closeall()

            self._connected = False
            logger.info("Database connection closed")


# Legacy compatibility
def get_database_manager(db_path: str = None) -> DatabaseManager:
    """Get database manager instance"""
    return DatabaseManager.get_instance(db_path)


__all__ = [
    "DatabaseManager",
    "DatabaseConfig",
    "DatabaseType",
    "get_database_manager",
]
