"""
Structured Audit Logging for OpenClaw

SIEM-compatible JSON logging for:
- Agent actions (tool used, input/output, duration, success/fail)
- Security events (auth, permission, suspicious activity)
- Workflow events (start, complete, node transitions)
- System events (startup, shutdown, errors)
"""

import os
import json
import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone

from .logger import get_logger

logger = get_logger("audit")


class AuditEventType(Enum):
    """Types of audit events."""
    # Agent events
    AGENT_ACTION = "agent.action"
    AGENT_TOOL_USE = "agent.tool_use"
    AGENT_DECISION = "agent.decision"
    AGENT_ERROR = "agent.error"

    # Orchestration events
    TASK_CREATED = "orchestrator.task_created"
    TASK_COMPLETED = "orchestrator.task_completed"
    TASK_FAILED = "orchestrator.task_failed"
    SUBTASK_ASSIGNED = "orchestrator.subtask_assigned"
    SUBTASK_COMPLETED = "orchestrator.subtask_completed"

    # Security events
    AUTH_SUCCESS = "security.auth_success"
    AUTH_FAILURE = "security.auth_failure"
    PERMISSION_DENIED = "security.permission_denied"
    SECRET_ACCESSED = "security.secret_accessed"
    SUSPICIOUS_INPUT = "security.suspicious_input"

    # Workflow events
    WORKFLOW_START = "workflow.start"
    WORKFLOW_COMPLETE = "workflow.complete"
    WORKFLOW_APPROVAL = "workflow.approval"
    WORKFLOW_ERROR = "workflow.error"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    CONFIG_CHANGE = "system.config_change"


class AuditSeverity(Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """A single audit event."""
    event_type: str
    severity: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    channel: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize as JSON line."""
        data = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
        }
        # Only include non-None optional fields
        if self.agent_id:
            data["agent_id"] = self.agent_id
        if self.session_id:
            data["session_id"] = self.session_id
        if self.user_id:
            data["user_id"] = self.user_id
        if self.channel:
            data["channel"] = self.channel
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.metadata:
            data["metadata"] = self.metadata

        return json.dumps(data, default=str)


class AuditLogger:
    """
    Structured audit logger.

    Writes events as JSON Lines to audit log file.
    Thread-safe with buffered writes for performance.

    Usage:
        audit = get_audit_logger()
        audit.log_action("vision_agent", "screen_capture", success=True, duration_ms=150)
        audit.log_security("auth_failure", user="unknown", ip="1.2.3.4")
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_buffer: int = 50,
        flush_interval: float = 5.0
    ):
        self._log_dir = log_dir or os.path.expanduser("~/.openclaw/logs")
        self._max_buffer = max_buffer
        self._flush_interval = flush_interval
        self._buffer: List[str] = []
        self._buffer_lock = threading.Lock()
        self._handlers: List[callable] = []

        os.makedirs(self._log_dir, exist_ok=True)

        # Start periodic flush
        self._flush_timer = None
        self._schedule_flush()

    @classmethod
    def get_instance(cls, **kwargs) -> "AuditLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        if cls._instance and cls._instance._flush_timer:
            cls._instance._flush_timer.cancel()
        cls._instance = None

    def _schedule_flush(self):
        self._flush_timer = threading.Timer(self._flush_interval, self._periodic_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _periodic_flush(self):
        self.flush()
        self._schedule_flush()

    # ---- Logging Methods ----

    def log(self, event: AuditEvent):
        """Log an audit event."""
        json_line = event.to_json()

        with self._buffer_lock:
            self._buffer.append(json_line)
            if len(self._buffer) >= self._max_buffer:
                self._flush_buffer()

        # Call registered handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass

    def log_action(
        self,
        agent_id: str,
        action: str,
        success: bool = True,
        duration_ms: float = None,
        input_data: Any = None,
        output_data: Any = None,
        **kwargs
    ):
        """Log an agent action."""
        self.log(AuditEvent(
            event_type=AuditEventType.AGENT_ACTION.value,
            severity=AuditSeverity.INFO.value if success else AuditSeverity.WARNING.value,
            message=f"Agent '{agent_id}' performed '{action}': {'success' if success else 'failed'}",
            agent_id=agent_id,
            duration_ms=duration_ms,
            metadata={
                "action": action,
                "success": success,
                "input": str(input_data)[:200] if input_data else None,
                "output": str(output_data)[:200] if output_data else None,
                **kwargs
            }
        ))

    def log_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        duration_ms: float = None,
        success: bool = True,
        error: str = None,
        **kwargs
    ):
        """Log a tool invocation."""
        self.log(AuditEvent(
            event_type=AuditEventType.AGENT_TOOL_USE.value,
            severity=AuditSeverity.INFO.value if success else AuditSeverity.ERROR.value,
            message=f"Tool '{tool_name}' by '{agent_id}': {'ok' if success else error}",
            agent_id=agent_id,
            duration_ms=duration_ms,
            metadata={"tool": tool_name, "success": success, "error": error, **kwargs}
        ))

    def log_security(
        self,
        event_type: str,
        message: str = "",
        severity: str = "warning",
        **kwargs
    ):
        """Log a security event."""
        self.log(AuditEvent(
            event_type=f"security.{event_type}",
            severity=severity,
            message=message or f"Security event: {event_type}",
            metadata=kwargs
        ))

    def log_workflow(
        self,
        workflow_id: str,
        event: str,
        node: str = None,
        **kwargs
    ):
        """Log a workflow event."""
        self.log(AuditEvent(
            event_type=f"workflow.{event}",
            severity=AuditSeverity.INFO.value,
            message=f"Workflow '{workflow_id}': {event}" + (f" at node '{node}'" if node else ""),
            metadata={"workflow_id": workflow_id, "node": node, **kwargs}
        ))

    def log_orchestration(
        self,
        plan_id: str,
        event: str,
        **kwargs
    ):
        """Log an orchestration event."""
        self.log(AuditEvent(
            event_type=f"orchestrator.{event}",
            severity=AuditSeverity.INFO.value,
            message=f"Plan '{plan_id}': {event}",
            metadata={"plan_id": plan_id, **kwargs}
        ))

    # ---- Handler Registration ----

    def add_handler(self, handler: callable):
        """Add a custom event handler (e.g., for Telegram alerts)."""
        self._handlers.append(handler)

    # ---- Flush & Query ----

    def flush(self):
        """Flush buffer to disk."""
        with self._buffer_lock:
            self._flush_buffer()

    def _flush_buffer(self):
        """Internal flush (must hold _buffer_lock)."""
        if not self._buffer:
            return

        log_file = os.path.join(
            self._log_dir,
            f"audit_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                for line in self._buffer:
                    f.write(line + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush audit log: {e}")

    def query_recent(self, count: int = 100, event_type: str = None) -> List[Dict]:
        """Query recent audit events from today's log file."""
        log_file = os.path.join(
            self._log_dir,
            f"audit_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )

        events = []
        if not os.path.exists(log_file):
            return events

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in reversed(lines):
                if len(events) >= count:
                    break
                try:
                    event = json.loads(line.strip())
                    if event_type and event.get("event_type") != event_type:
                        continue
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Failed to query audit log: {e}")

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        log_files = list(Path(self._log_dir).glob("audit_*.jsonl"))
        total_size = sum(f.stat().st_size for f in log_files)

        return {
            "log_dir": self._log_dir,
            "log_files": len(log_files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "buffer_size": len(self._buffer),
            "handlers": len(self._handlers),
        }


# ============== Global Access ==============

def get_audit_logger(**kwargs) -> AuditLogger:
    """Get global audit logger."""
    return AuditLogger.get_instance(**kwargs)


__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "AuditEvent",
    "AuditLogger",
    "get_audit_logger",
]
