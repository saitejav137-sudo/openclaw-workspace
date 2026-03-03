"""
Event Bus for OpenClaw

Centralized event system inspired by ComposioHQ/agent-orchestrator's
OrchestratorEvent pattern with priority-based notification routing.

Event categories:
- Agent lifecycle:  agent.spawned, agent.working, agent.stuck, agent.errored, agent.completed
- Task lifecycle:   task.created, task.assigned, task.completed, task.failed
- Swarm lifecycle:  swarm.started, swarm.completed, swarm.failed
- Reactions:        reaction.triggered, reaction.escalated
- Health:           health.degraded, health.recovered
- System:           system.startup, system.shutdown
"""

import time
import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from collections import deque

from .logger import get_logger

logger = get_logger("event_bus")


# ============== Event Types ==============

class EventPriority(str, Enum):
    """Priority levels — determines notification routing."""
    URGENT = "urgent"    # Requires immediate human attention
    ACTION = "action"    # Human action needed but not urgent
    WARNING = "warning"  # Something may need attention
    INFO = "info"        # Informational, no action needed

    @property
    def level(self) -> int:
        """Numeric level for comparison (lower = higher priority)."""
        return {"urgent": 0, "action": 1, "warning": 2, "info": 3}[self.value]


class EventType(str, Enum):
    """All event types emitted by the system."""

    # Agent lifecycle
    AGENT_SPAWNED = "agent.spawned"
    AGENT_WORKING = "agent.working"
    AGENT_IDLE = "agent.idle"
    AGENT_STUCK = "agent.stuck"
    AGENT_NEEDS_INPUT = "agent.needs_input"
    AGENT_ERRORED = "agent.errored"
    AGENT_COMPLETED = "agent.completed"
    AGENT_KILLED = "agent.killed"
    AGENT_RECOVERED = "agent.recovered"

    # Task lifecycle
    TASK_CREATED = "task.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_RUNNING = "task.running"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_RETRYING = "task.retrying"
    TASK_CANCELLED = "task.cancelled"

    # Swarm lifecycle
    SWARM_STARTED = "swarm.started"
    SWARM_PROGRESS = "swarm.progress"
    SWARM_COMPLETED = "swarm.completed"
    SWARM_FAILED = "swarm.failed"

    # Reactions
    REACTION_TRIGGERED = "reaction.triggered"
    REACTION_SUCCEEDED = "reaction.succeeded"
    REACTION_FAILED = "reaction.failed"
    REACTION_ESCALATED = "reaction.escalated"

    # Health
    HEALTH_DEGRADED = "health.degraded"
    HEALTH_RECOVERED = "health.recovered"
    HEALTH_CHECK_FAILED = "health.check_failed"

    # System
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_CONFIG_RELOADED = "system.config_reloaded"
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"

    # Inter-bot
    INTERBOT_MESSAGE_SENT = "interbot.message_sent"
    INTERBOT_MESSAGE_RECEIVED = "interbot.message_received"
    INTERBOT_TASK_DELEGATED = "interbot.task_delegated"


# Default priorities for event types
EVENT_DEFAULT_PRIORITY: Dict[EventType, EventPriority] = {
    EventType.AGENT_STUCK: EventPriority.WARNING,
    EventType.AGENT_ERRORED: EventPriority.ACTION,
    EventType.AGENT_KILLED: EventPriority.WARNING,
    EventType.AGENT_NEEDS_INPUT: EventPriority.ACTION,
    EventType.TASK_FAILED: EventPriority.WARNING,
    EventType.SWARM_FAILED: EventPriority.ACTION,
    EventType.REACTION_ESCALATED: EventPriority.URGENT,
    EventType.HEALTH_DEGRADED: EventPriority.WARNING,
    EventType.HEALTH_CHECK_FAILED: EventPriority.ACTION,
}


# ============== Event Data Model ==============

@dataclass
class Event:
    """An event emitted by the orchestrator."""
    id: str = field(default_factory=lambda: str(uuid4())[:12])
    type: EventType = EventType.SYSTEM_STARTUP
    priority: EventPriority = EventPriority.INFO
    source: str = ""              # who emitted: "swarm", "orchestrator", "agent:researcher"
    message: str = ""             # human-readable description
    data: Dict[str, Any] = field(default_factory=dict)  # structured payload
    timestamp: float = field(default_factory=time.time)

    @property
    def category(self) -> str:
        """Get the event category (first part before the dot)."""
        return self.type.value.split(".")[0]

    def __str__(self) -> str:
        return f"[{self.priority.value.upper()}] {self.type.value}: {self.message}"


# ============== Subscription ==============

@dataclass
class Subscription:
    """A subscription to specific event types."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    handler: Callable = None
    event_types: Optional[Set[EventType]] = None  # None = all events
    categories: Optional[Set[str]] = None          # None = all categories
    min_priority: EventPriority = EventPriority.INFO
    is_async: bool = False

    def matches(self, event: Event) -> bool:
        """Check if this subscription matches the given event."""
        # Priority filter
        if event.priority.level > self.min_priority.level:
            return False
        # Event type filter
        if self.event_types and event.type not in self.event_types:
            return False
        # Category filter
        if self.categories and event.category not in self.categories:
            return False
        return True


# ============== Event Bus ==============

class EventBus:
    """
    Central event bus for the entire OpenClaw system.

    Features:
    - Typed events with priority levels
    - Subscribe by event type, category, or priority
    - Sync and async handlers
    - Event history for debugging
    - Thread-safe

    Usage:
        bus = EventBus()

        # Subscribe to specific events
        bus.subscribe(
            handler=on_agent_stuck,
            event_types={EventType.AGENT_STUCK},
        )

        # Subscribe to all events in a category
        bus.subscribe(
            handler=log_all_agent_events,
            categories={"agent"},
        )

        # Subscribe to urgent events only
        bus.subscribe(
            handler=alert_human,
            min_priority=EventPriority.URGENT,
        )

        # Emit an event
        bus.emit(EventType.AGENT_STUCK, "Agent researcher is stuck", {
            "agent_id": "abc123",
            "stuck_since": time.time() - 300,
        })
    """

    def __init__(self, max_history: int = 1000):
        self._subscriptions: Dict[str, Subscription] = {}
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._event_count = 0
        self._error_count = 0
        logger.info("EventBus initialized (max_history=%d)", max_history)

    def subscribe(
        self,
        handler: Callable,
        event_types: Set[EventType] = None,
        categories: Set[str] = None,
        min_priority: EventPriority = EventPriority.INFO,
    ) -> str:
        """
        Subscribe to events. Returns subscription ID for unsubscribing.

        Args:
            handler: Function to call when event matches. Receives Event as arg.
            event_types: Set of specific event types to listen for (None = all).
            categories: Set of categories to listen for (None = all).
            min_priority: Only receive events at this priority or higher.

        Returns:
            Subscription ID string.
        """
        is_async = asyncio.iscoroutinefunction(handler)
        sub = Subscription(
            handler=handler,
            event_types=event_types,
            categories=categories,
            min_priority=min_priority,
            is_async=is_async,
        )

        with self._lock:
            self._subscriptions[sub.id] = sub

        filter_desc = []
        if event_types:
            filter_desc.append(f"types={[t.value for t in event_types]}")
        if categories:
            filter_desc.append(f"categories={list(categories)}")
        if min_priority != EventPriority.INFO:
            filter_desc.append(f"min_priority={min_priority.value}")

        logger.debug(
            "Subscription '%s' registered: %s",
            sub.id, ", ".join(filter_desc) or "all events",
        )
        return sub.id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription. Returns True if it existed."""
        with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                logger.debug("Subscription '%s' removed", subscription_id)
                return True
        return False

    def emit(
        self,
        event_type: EventType,
        message: str = "",
        data: Dict[str, Any] = None,
        source: str = "",
        priority: EventPriority = None,
    ) -> Event:
        """
        Emit an event to all matching subscribers.

        Args:
            event_type: The type of event.
            message: Human-readable description.
            data: Structured payload.
            source: Who emitted this event.
            priority: Override default priority.

        Returns:
            The created Event object.
        """
        # Determine priority
        if priority is None:
            priority = EVENT_DEFAULT_PRIORITY.get(event_type, EventPriority.INFO)

        event = Event(
            type=event_type,
            priority=priority,
            source=source,
            message=message,
            data=data or {},
        )

        # Record in history
        with self._lock:
            self._history.append(event)
            self._event_count += 1
            # Snapshot subscribers to avoid holding lock during dispatch
            subscribers = list(self._subscriptions.values())

        # Log the event
        log_fn = (
            logger.warning if priority == EventPriority.URGENT
            else logger.info if priority in (EventPriority.ACTION, EventPriority.WARNING)
            else logger.debug
        )
        log_fn("Event [%s] %s: %s", event.id, event.type.value, message)

        # Dispatch to matching subscribers
        for sub in subscribers:
            if sub.matches(event):
                try:
                    if sub.is_async:
                        # Schedule async handler
                        self._schedule_async(sub.handler, event)
                    else:
                        sub.handler(event)
                except Exception as e:
                    self._error_count += 1
                    logger.error(
                        "Error in event handler for '%s': %s",
                        event.type.value, e,
                        exc_info=True,
                    )

        return event

    def emit_event(self, event: Event) -> Event:
        """Emit a pre-constructed event."""
        return self.emit(
            event_type=event.type,
            message=event.message,
            data=event.data,
            source=event.source,
            priority=event.priority,
        )

    def _schedule_async(self, handler: Callable, event: Event) -> None:
        """Schedule an async handler for execution."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(handler(event))
        except RuntimeError:
            # No event loop running — run in a thread
            thread = threading.Thread(
                target=lambda: asyncio.run(handler(event)),
                daemon=True,
            )
            thread.start()

    # ============== History & Querying ==============

    def get_history(
        self,
        limit: int = 50,
        event_type: EventType = None,
        category: str = None,
        min_priority: EventPriority = None,
        since: float = None,
    ) -> List[Event]:
        """
        Query event history with filters.

        Args:
            limit: Max number of events to return.
            event_type: Filter by specific event type.
            category: Filter by category (e.g., "agent", "task").
            min_priority: Filter by minimum priority.
            since: Only events after this timestamp.

        Returns:
            List of matching events, newest first.
        """
        with self._lock:
            events = list(self._history)

        # Apply filters
        if event_type:
            events = [e for e in events if e.type == event_type]
        if category:
            events = [e for e in events if e.category == category]
        if min_priority:
            events = [e for e in events if e.priority.level <= min_priority.level]
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Newest first, limited
        return list(reversed(events))[:limit]

    def get_recent(self, limit: int = 20) -> List[Event]:
        """Get the most recent events."""
        return self.get_history(limit=limit)

    def count_by_type(self, since: float = None) -> Dict[str, int]:
        """Count events by type since a timestamp."""
        with self._lock:
            events = list(self._history)

        if since:
            events = [e for e in events if e.timestamp >= since]

        counts: Dict[str, int] = {}
        for e in events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts

    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._history.clear()
        logger.info("Event history cleared")

    # ============== Stats ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            history_size = len(self._history)
            sub_count = len(self._subscriptions)

        return {
            "total_events_emitted": self._event_count,
            "total_errors": self._error_count,
            "history_size": history_size,
            "subscriptions": sub_count,
            "events_last_hour": len(
                self.get_history(limit=10000, since=time.time() - 3600)
            ),
        }


# ============== Persistent Event Store ==============

import json
import os
from datetime import datetime


class PersistentEventStore:
    """
    Writes events to daily JSONL files for persistence.

    Events survive restarts. Auto-rotates by date.
    Files: ~/.openclaw/events/2026-03-02.jsonl

    Usage:
        store = PersistentEventStore()
        bus.subscribe(handler=store.on_event)  # Subscribe to all events
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.expanduser("~/.openclaw/events")
        os.makedirs(self.base_dir, exist_ok=True)
        self._current_date: str = ""
        self._current_file = None
        self._lock = threading.Lock()
        self._total_written: int = 0

    def on_event(self, event: Event) -> None:
        """Handler to subscribe to the event bus."""
        today = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            # Rotate file if date changed
            if today != self._current_date:
                if self._current_file:
                    self._current_file.close()
                filepath = os.path.join(self.base_dir, f"{today}.jsonl")
                self._current_file = open(filepath, "a")
                self._current_date = today

            # Write event as JSON line
            record = {
                "id": event.id,
                "type": event.type.value,
                "priority": event.priority.value,
                "source": event.source,
                "message": event.message,
                "data": event.data,
                "timestamp": event.timestamp,
            }

            try:
                self._current_file.write(json.dumps(record, default=str) + "\n")
                self._current_file.flush()
                self._total_written += 1
            except Exception as e:
                logger.error("Failed to persist event: %s", e)

    def load_events(self, date: str = None, limit: int = 100) -> List[Dict]:
        """Load persisted events from a date file."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        filepath = os.path.join(self.base_dir, f"{date}.jsonl")
        if not os.path.exists(filepath):
            return []

        events = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return events[-limit:]  # Return most recent

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence stats."""
        files = []
        total_size = 0
        if os.path.exists(self.base_dir):
            for f in sorted(os.listdir(self.base_dir)):
                if f.endswith(".jsonl"):
                    path = os.path.join(self.base_dir, f)
                    size = os.path.getsize(path)
                    files.append(f)
                    total_size += size

        return {
            "base_dir": self.base_dir,
            "total_written": self._total_written,
            "files": files,
            "total_size_bytes": total_size,
        }

    def close(self):
        """Close the current file."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None



_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def emit(
    event_type: EventType,
    message: str = "",
    data: Dict[str, Any] = None,
    source: str = "",
    priority: EventPriority = None,
) -> Event:
    """Convenience: emit an event on the global bus."""
    return get_event_bus().emit(event_type, message, data, source, priority)


def subscribe(
    handler: Callable,
    event_types: Set[EventType] = None,
    categories: Set[str] = None,
    min_priority: EventPriority = EventPriority.INFO,
) -> str:
    """Convenience: subscribe to events on the global bus."""
    return get_event_bus().subscribe(handler, event_types, categories, min_priority)


__all__ = [
    # Enums
    "EventPriority",
    "EventType",
    # Data model
    "Event",
    "Subscription",
    # Bus
    "EventBus",
    "get_event_bus",
    # Persistence
    "PersistentEventStore",
    # Convenience
    "emit",
    "subscribe",
    # Constants
    "EVENT_DEFAULT_PRIORITY",
]
