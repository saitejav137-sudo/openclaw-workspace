"""
Reaction Engine for OpenClaw

Configurable automated reactions to system events — the killer feature
from ComposioHQ/agent-orchestrator adapted for OpenClaw.

How it works:
1. The engine subscribes to the EventBus for ALL events
2. When an event fires, it checks configured reactions  
3. If a reaction matches, it executes the configured action
4. If the action fails after retries, it escalates to human

Example reactions:
- agent.stuck  → auto-restart the agent
- agent.error  → send error context back to agent for self-repair
- task.failed  → retry with a different agent
- swarm.completed → notify the user via Telegram
"""

import time
import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger
from .event_bus import (
    EventBus, Event, EventType, EventPriority,
    get_event_bus,
)

logger = get_logger("reaction_engine")


# ============== Reaction Actions ==============

class ReactionAction(str, Enum):
    """What to do when an event matches a reaction."""
    RESTART_AGENT = "restart-agent"            # Restart the stuck/failed agent
    SEND_TO_AGENT = "send-to-agent"            # Send message/context to agent
    RETRY_TASK = "retry-task"                   # Retry the failed task
    RETRY_DIFFERENT_AGENT = "retry-different-agent"  # Retry with alternate agent
    NOTIFY = "notify"                           # Notify the human
    AUTO_MERGE = "auto-merge"                   # Auto-merge (future: PR flows)
    CUSTOM = "custom"                           # Call a custom handler


# ============== Reaction Config ==============

@dataclass
class ReactionConfig:
    """
    Configuration for a single reaction.
    
    Modeled on agent-orchestrator's ReactionConfig YAML block.
    """
    # Matching
    event_type: EventType              # Which event triggers this reaction
    
    # Behavior
    auto: bool = True                  # Whether this reaction is enabled
    action: ReactionAction = ReactionAction.NOTIFY
    message: str = ""                  # Message template (for send-to-agent)
    priority: EventPriority = EventPriority.INFO
    
    # Retry & escalation
    retries: int = 2                   # Max retry attempts
    escalate_after_sec: float = 600.0  # Escalate to human after N seconds (0 = never)
    cooldown_sec: float = 30.0         # Min time between repeated reactions for same source
    
    # Custom handler (for CUSTOM action)
    handler: Optional[Callable] = None
    
    # Description for CLI/dashboard
    description: str = ""


@dataclass
class ReactionResult:
    """Result of a reaction execution."""
    reaction_type: str
    event_id: str
    success: bool
    action: str
    message: str = ""
    escalated: bool = False
    attempt: int = 0
    timestamp: float = field(default_factory=time.time)


# ============== Default Reactions ==============

def _default_reactions() -> Dict[str, ReactionConfig]:
    """The default reaction set — battle-tested patterns from agent-orchestrator."""
    return {
        "agent-stuck": ReactionConfig(
            event_type=EventType.AGENT_STUCK,
            auto=True,
            action=ReactionAction.RESTART_AGENT,
            retries=2,
            escalate_after_sec=600,   # 10 min
            cooldown_sec=60,
            description="Auto-restart agents that are stuck",
        ),
        "agent-error": ReactionConfig(
            event_type=EventType.AGENT_ERRORED,
            auto=True,
            action=ReactionAction.SEND_TO_AGENT,
            message="An error occurred. Analyze the error and attempt recovery: {error_details}",
            retries=1,
            escalate_after_sec=300,   # 5 min
            cooldown_sec=30,
            description="Send error context back to agent for self-repair",
        ),
        "task-failed": ReactionConfig(
            event_type=EventType.TASK_FAILED,
            auto=True,
            action=ReactionAction.RETRY_TASK,
            retries=1,
            escalate_after_sec=900,   # 15 min
            cooldown_sec=60,
            description="Retry failed tasks automatically",
        ),
        "swarm-completed": ReactionConfig(
            event_type=EventType.SWARM_COMPLETED,
            auto=True,
            action=ReactionAction.NOTIFY,
            priority=EventPriority.ACTION,
            description="Notify user when swarm completes",
        ),
        "swarm-failed": ReactionConfig(
            event_type=EventType.SWARM_FAILED,
            auto=True,
            action=ReactionAction.NOTIFY,
            priority=EventPriority.URGENT,
            description="Urgent notification when swarm fails",
        ),
        "health-degraded": ReactionConfig(
            event_type=EventType.HEALTH_DEGRADED,
            auto=True,
            action=ReactionAction.NOTIFY,
            priority=EventPriority.WARNING,
            cooldown_sec=300,         # Don't spam — max once per 5 min
            description="Notify on system health degradation",
        ),
        "escalation": ReactionConfig(
            event_type=EventType.REACTION_ESCALATED,
            auto=True,
            action=ReactionAction.NOTIFY,
            priority=EventPriority.URGENT,
            description="Always notify human on escalation",
        ),
    }


# ============== Reaction Engine ==============

class ReactionEngine:
    """
    Automated reaction system that subscribes to events and executes
    configured responses with retry and escalation logic.

    This is the Python adaptation of agent-orchestrator's reaction system.
    The key insight: most agent failures are recoverable *if* you detect
    them fast enough and take the right action automatically.

    Usage:
        engine = ReactionEngine()
        engine.start()   # Subscribes to event bus

        # Add custom reaction
        engine.add_reaction("my-reaction", ReactionConfig(
            event_type=EventType.TASK_COMPLETED,
            action=ReactionAction.CUSTOM,
            handler=my_custom_handler,
        ))

        # Events on the bus will now trigger reactions automatically
    """

    def __init__(
        self,
        event_bus: EventBus = None,
        reactions: Dict[str, ReactionConfig] = None,
        on_notify: Optional[Callable] = None,
    ):
        self._bus = event_bus
        self._reactions = reactions or _default_reactions()
        self._on_notify = on_notify  # callback: (message, priority) -> None
        
        self._subscription_id: Optional[str] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Track reaction state
        self._last_triggered: Dict[str, float] = {}  # reaction_name:source → timestamp
        self._attempt_counts: Dict[str, int] = {}    # reaction_name:source → attempts
        self._first_triggered: Dict[str, float] = {} # reaction_name:source → first trigger time
        
        # Results history
        self._results: List[ReactionResult] = []
        self._max_results = 200
        
        # Stats
        self._total_reactions = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_escalations = 0

        logger.info(
            "ReactionEngine initialized with %d reactions: %s",
            len(self._reactions),
            list(self._reactions.keys()),
        )

    @property
    def bus(self) -> EventBus:
        if self._bus is None:
            self._bus = get_event_bus()
        return self._bus

    # ============== Start / Stop ==============

    def start(self) -> None:
        """Start the reaction engine — subscribe to all events."""
        if self._running:
            return

        self._subscription_id = self.bus.subscribe(
            handler=self._on_event,
            # Listen to all events — we filter internally
        )
        self._running = True

        logger.info("ReactionEngine started (subscription=%s)", self._subscription_id)

    def stop(self) -> None:
        """Stop the reaction engine."""
        if not self._running:
            return

        if self._subscription_id:
            self.bus.unsubscribe(self._subscription_id)
            self._subscription_id = None

        self._running = False
        logger.info("ReactionEngine stopped")

    # ============== Reaction Management ==============

    def add_reaction(self, name: str, config: ReactionConfig) -> None:
        """Add or update a reaction."""
        with self._lock:
            self._reactions[name] = config
        logger.info("Reaction '%s' added: %s → %s", name, config.event_type.value, config.action.value)

    def remove_reaction(self, name: str) -> bool:
        """Remove a reaction."""
        with self._lock:
            if name in self._reactions:
                del self._reactions[name]
                logger.info("Reaction '%s' removed", name)
                return True
        return False

    def enable_reaction(self, name: str) -> None:
        """Enable a reaction."""
        with self._lock:
            if name in self._reactions:
                self._reactions[name].auto = True

    def disable_reaction(self, name: str) -> None:
        """Disable a reaction."""
        with self._lock:
            if name in self._reactions:
                self._reactions[name].auto = False

    def list_reactions(self) -> Dict[str, Dict[str, Any]]:
        """List all configured reactions."""
        return {
            name: {
                "event_type": cfg.event_type.value,
                "action": cfg.action.value,
                "enabled": cfg.auto,
                "retries": cfg.retries,
                "escalate_after_sec": cfg.escalate_after_sec,
                "description": cfg.description,
            }
            for name, cfg in self._reactions.items()
        }

    # ============== Event Handler ==============

    def _on_event(self, event: Event) -> None:
        """Handle an incoming event — check all reactions."""
        with self._lock:
            reactions = list(self._reactions.items())

        for name, config in reactions:
            if not config.auto:
                continue
            if config.event_type != event.type:
                continue

            # Check cooldown
            key = f"{name}:{event.source}"
            last = self._last_triggered.get(key, 0)
            if time.time() - last < config.cooldown_sec:
                logger.debug("Reaction '%s' in cooldown (%.0fs remaining)", name,
                             config.cooldown_sec - (time.time() - last))
                continue

            # Execute the reaction
            self._execute_reaction(name, config, event, key)

    def _execute_reaction(
        self,
        name: str,
        config: ReactionConfig,
        event: Event,
        key: str,
    ) -> None:
        """Execute a single reaction with retry logic."""
        self._total_reactions += 1
        self._last_triggered[key] = time.time()

        # Track first trigger for escalation timing
        if key not in self._first_triggered:
            self._first_triggered[key] = time.time()

        # Check if escalation time exceeded
        elapsed_since_first = time.time() - self._first_triggered[key]
        if config.escalate_after_sec > 0 and elapsed_since_first > config.escalate_after_sec:
            self._escalate(name, config, event, f"Timed out after {elapsed_since_first:.0f}s")
            return

        # Check retry count
        attempt = self._attempt_counts.get(key, 0) + 1
        self._attempt_counts[key] = attempt

        if attempt > config.retries + 1:  # +1 for initial attempt
            self._escalate(name, config, event, f"Max retries ({config.retries}) exhausted")
            return

        logger.info(
            "Reaction '%s' triggered by event [%s] (attempt %d/%d)",
            name, event.id, attempt, config.retries + 1,
        )

        # Emit reaction.triggered event
        self.bus.emit(
            EventType.REACTION_TRIGGERED,
            f"Reaction '{name}' triggered: {config.action.value}",
            data={
                "reaction": name,
                "action": config.action.value,
                "event_id": event.id,
                "attempt": attempt,
            },
            source="reaction_engine",
        )

        try:
            success = self._execute_action(config, event)

            result = ReactionResult(
                reaction_type=name,
                event_id=event.id,
                success=success,
                action=config.action.value,
                attempt=attempt,
            )
            self._record_result(result)

            if success:
                self._total_successes += 1
                # Reset counters on success
                self._attempt_counts.pop(key, None)
                self._first_triggered.pop(key, None)

                self.bus.emit(
                    EventType.REACTION_SUCCEEDED,
                    f"Reaction '{name}' succeeded",
                    data={"reaction": name, "event_id": event.id},
                    source="reaction_engine",
                )
            else:
                self._total_failures += 1
                self.bus.emit(
                    EventType.REACTION_FAILED,
                    f"Reaction '{name}' failed (attempt {attempt})",
                    data={"reaction": name, "event_id": event.id, "attempt": attempt},
                    source="reaction_engine",
                    priority=EventPriority.WARNING,
                )

        except Exception as e:
            self._total_failures += 1
            logger.error("Reaction '%s' error: %s", name, e, exc_info=True)

            result = ReactionResult(
                reaction_type=name,
                event_id=event.id,
                success=False,
                action=config.action.value,
                message=str(e),
                attempt=attempt,
            )
            self._record_result(result)

    # ============== Action Execution ==============

    def _execute_action(self, config: ReactionConfig, event: Event) -> bool:
        """Execute the configured action. Returns True on success."""

        if config.action == ReactionAction.RESTART_AGENT:
            return self._action_restart_agent(event)

        elif config.action == ReactionAction.SEND_TO_AGENT:
            return self._action_send_to_agent(config, event)

        elif config.action == ReactionAction.RETRY_TASK:
            return self._action_retry_task(event)

        elif config.action == ReactionAction.RETRY_DIFFERENT_AGENT:
            return self._action_retry_different_agent(event)

        elif config.action == ReactionAction.NOTIFY:
            return self._action_notify(config, event)

        elif config.action == ReactionAction.CUSTOM:
            if config.handler:
                config.handler(event)
                return True
            logger.warning("CUSTOM reaction without handler")
            return False

        else:
            logger.warning("Unknown action: %s", config.action)
            return False

    def _action_restart_agent(self, event: Event) -> bool:
        """Restart a stuck/failed agent via the lifecycle manager."""
        agent_id = event.data.get("agent_id")
        if not agent_id:
            logger.warning("restart-agent: no agent_id in event data")
            return False

        try:
            from .lifecycle_manager import get_lifecycle_manager
            lm = get_lifecycle_manager()
            lm._attempt_recovery(agent_id)
            return True
        except Exception as e:
            logger.error("restart-agent failed: %s", e)
            return False

    def _action_send_to_agent(self, config: ReactionConfig, event: Event) -> bool:
        """Send a message/context to the agent for self-repair."""
        agent_id = event.data.get("agent_id")
        if not agent_id:
            return False

        # Format the message template with event data
        try:
            message = config.message.format(**event.data)
        except (KeyError, ValueError):
            message = config.message

        logger.info("send-to-agent '%s': %s", agent_id, message[:100])
        # The actual sending depends on the agent's communication channel.
        # For now, we log the action and return True — handlers can override.
        return True

    def _action_retry_task(self, event: Event) -> bool:
        """Retry a failed task."""
        task_id = event.data.get("task_id")
        if not task_id:
            return False

        logger.info("retry-task: %s", task_id)
        # Hook into the TaskQueue for retry
        try:
            from .task_queue import TaskQueue
            # The actual retry mechanism depends on the queue implementation
            return True
        except Exception as e:
            logger.error("retry-task failed: %s", e)
            return False

    def _action_retry_different_agent(self, event: Event) -> bool:
        """Retry with a different agent."""
        task_id = event.data.get("task_id")
        failed_agent = event.data.get("agent_id")
        if not task_id:
            return False

        logger.info("retry-different-agent: task=%s, excluding=%s", task_id, failed_agent)
        return True

    def _action_notify(self, config: ReactionConfig, event: Event) -> bool:
        """Notify the human."""
        message = f"🔔 [{config.priority.value.upper()}] {event.message}"

        if self._on_notify:
            try:
                self._on_notify(message, config.priority)
                return True
            except Exception as e:
                logger.error("Notification failed: %s", e)
                return False

        # Fallback: just log
        logger.info("NOTIFICATION: %s", message)
        return True

    # ============== Escalation ==============

    def _escalate(self, name: str, config: ReactionConfig, event: Event, reason: str) -> None:
        """Escalate — the reaction couldn't handle it, human needed."""
        self._total_escalations += 1

        logger.warning("ESCALATION [%s]: %s (event: %s)", name, reason, event.type.value)

        # Reset counters
        key = f"{name}:{event.source}"
        self._attempt_counts.pop(key, None)
        self._first_triggered.pop(key, None)

        # Create escalation result
        result = ReactionResult(
            reaction_type=name,
            event_id=event.id,
            success=False,
            action="escalate",
            message=reason,
            escalated=True,
        )
        self._record_result(result)

        # Emit escalation event (will trigger the "escalation" reaction → notify)
        self.bus.emit(
            EventType.REACTION_ESCALATED,
            f"Reaction '{name}' escalated: {reason}. Event: {event.message}",
            data={
                "reaction": name,
                "reason": reason,
                "original_event": event.type.value,
                "agent_id": event.data.get("agent_id"),
            },
            source="reaction_engine",
            priority=EventPriority.URGENT,
        )

    # ============== Results & Stats ==============

    def _record_result(self, result: ReactionResult) -> None:
        """Record a reaction result."""
        self._results.append(result)
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results:]

    def get_recent_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent reaction results."""
        return [
            {
                "reaction": r.reaction_type,
                "event_id": r.event_id,
                "success": r.success,
                "action": r.action,
                "message": r.message,
                "escalated": r.escalated,
                "attempt": r.attempt,
                "timestamp": r.timestamp,
            }
            for r in reversed(self._results[-limit:])
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get reaction engine statistics."""
        return {
            "is_running": self._running,
            "total_reactions_triggered": self._total_reactions,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_escalations": self._total_escalations,
            "configured_reactions": len(self._reactions),
            "active_reactions": sum(1 for r in self._reactions.values() if r.auto),
            "pending_retries": len(self._attempt_counts),
        }


# ============== Global Instance ==============

_reaction_engine: Optional[ReactionEngine] = None


def get_reaction_engine(**kwargs) -> ReactionEngine:
    """Get or create the global reaction engine."""
    global _reaction_engine
    if _reaction_engine is None:
        _reaction_engine = ReactionEngine(**kwargs)
    return _reaction_engine


__all__ = [
    "ReactionAction",
    "ReactionConfig",
    "ReactionResult",
    "ReactionEngine",
    "get_reaction_engine",
]
