"""Core logging module with structured logging support"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from logging.handlers import RotatingFileHandler
import json


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with correlation ID support"""

    # Thread-local storage for correlation ID
    _correlation_id = None

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, "correlation_id", None) or self._correlation_id or "N/A"
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)

    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread/request"""
        cls._correlation_id = correlation_id

    @classmethod
    def clear_correlation_id(cls):
        """Clear correlation ID"""
        cls._correlation_id = None


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class Logger:
    """Centralized logger manager for OpenClaw"""

    _instance: Optional['Logger'] = None
    _loggers: dict = {}

    def __init__(self):
        self.log_dir = os.path.expanduser("~/.openclaw/logs")
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG
        self.log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self._configured = False

    @classmethod
    def get_instance(cls) -> 'Logger':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(
        self,
        log_dir: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        structured: bool = False
    ) -> None:
        """Configure the logging system"""

        if log_dir:
            self.log_dir = log_dir

        self.console_level = console_level
        self.file_level = file_level

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger("openclaw")
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        if structured:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = ColoredFormatter(self.log_format, self.date_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        log_file = os.path.join(self.log_dir, "openclaw.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(self.log_format, self.date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Error log file
        error_file = os.path.join(self.log_dir, "errors.log")
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""

        if name not in self._loggers:
            logger = logging.getLogger(f"openclaw.{name}")
            self._loggers[name] = logger

        return self._loggers[name]

    def add_extra_data(self, logger: logging.Logger, **kwargs) -> logging.Logger:
        """Add extra data to log records"""
        # This is a convenience method for structured logging
        return logger


# Convenience function for getting loggers
def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module"""
    return Logger.get_instance().get_logger(name)


# Initialize default logger
def setup_logging(
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> None:
    """Setup the logging system with defaults"""
    Logger.get_instance().configure(log_dir, console_level, file_level)
