"""
Lifecycle Manager for OpenClaw

Polling-based lifecycle manager inspired by ComposioHQ/agent-orchestrator's
LifecycleManager. Monitors agent health and auto-recovers failures.

Features:
- Polls agent states at configurable intervals
- Detects stuck agents (no activity beyond threshold)
- Auto-restores crashed agents from checkpoints
- Emits events via the event bus
- Coordinates with circuit breakers from resilience module
"""

import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .logger import get_logger
from .agent_state import (
    AgentStatus, ActivityState, AgentStateManager, get_state_manager,
    TERMINAL_STATUSES, RECOVERABLE_STATUSES,
    DEFAULT_IDLE_THRESHOLD_SEC, DEFAULT_STUCK_THRESHOLD_SEC,
)
from .event_bus import get_event_bus, EventType, EventPriority

logger = get_logger("lifecycle_manager")


# ============== Configuration ==============

@dataclass
class LifecycleConfig:
    """Configuration for the lifecycle manager."""
    poll_interval_sec: float = 10.0       # How often to check agent states
    stuck_threshold_sec: float = DEFAULT_STUCK_THRESHOLD_SEC
    idle_threshold_sec: float = DEFAULT_IDLE_THRESHOLD_SEC
    max_auto_recoveries: int = 3          # Max recovery attempts per agent
    auto_recover: bool = True             # Whether to auto-recover stuck/errored agents
    checkpoint_before_recover: bool = True # Create checkpoint before recovery
    escalate_after_max_retries: bool = True  # Notify human after max retries


# ============== Lifecycle Manager ==============

class LifecycleManager:
    """
    Monitors all agent sessions and takes automatic action.

    Modeled on agent-orchestrator's LifecycleManager. Runs a background
    polling loop that:
    1. Checks each agent's activity state
    2. Detects stuck/idle agents
    3. Auto-recovers failed agents from checkpoints
    4. Emits events for each state change
    5. Escalates to human (Telegram notification) if auto-recovery fails

    Usage:
        manager = LifecycleManager()
        manager.start()

        # ... agents run and may get stuck/crash ...

        manager.stop()
    """

    def __init__(
        self,
        config: LifecycleConfig = None,
        state_manager: AgentStateManager = None,
        on_escalate: Optional[Callable] = None,
    ):
        self.config = config or LifecycleConfig()
        self._state_manager = state_manager
        self._on_escalate = on_escalate  # callback for human escalation

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Track recovery attempts per agent
        self._recovery_attempts: Dict[str, int] = {}

        # Stats
        self._total_checks = 0
        self._stuck_detected = 0
        self._recoveries_attempted = 0
        self._recoveries_succeeded = 0
        self._escalations = 0

        logger.info(
            "LifecycleManager initialized (poll_interval=%.1fs, stuck_threshold=%.0fs, auto_recover=%s)",
            self.config.poll_interval_sec,
            self.config.stuck_threshold_sec,
            self.config.auto_recover,
        )

    @property
    def state_manager(self) -> AgentStateManager:
        if self._state_manager is None:
            self._state_manager = get_state_manager()
        return self._state_manager

    # ============== Start / Stop ==============

    def start(self) -> None:
        """Start the lifecycle polling loop."""
        with self._lock:
            if self._running:
                logger.warning("LifecycleManager already running")
                return

            self._running = True
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="lifecycle-manager",
                daemon=True,
            )
            self._thread.start()

            # Emit startup event
            bus = get_event_bus()
            bus.emit(
                EventType.SYSTEM_STARTUP,
                "LifecycleManager started",
                source="lifecycle_manager",
            )

            logger.info("LifecycleManager started")

    def stop(self) -> None:
        """Stop the lifecycle polling loop."""
        with self._lock:
            if not self._running:
                return

            self._running = False

        if self._thread:
            self._thread.join(timeout=self.config.poll_interval_sec + 2)
            self._thread = None

        logger.info("LifecycleManager stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ============== Polling Loop ==============

    def _poll_loop(self) -> None:
        """Main polling loop — runs in background thread."""
        logger.info("Lifecycle polling loop started (interval=%.1fs)", self.config.poll_interval_sec)

        while self._running:
            try:
                self._check_all()
            except Exception as e:
                logger.error("Error in lifecycle check: %s", e, exc_info=True)

            # Sleep in small increments so stop() is responsive
            slept = 0.0
            while slept < self.config.poll_interval_sec and self._running:
                time.sleep(min(1.0, self.config.poll_interval_sec - slept))
                slept += 1.0

    def _check_all(self) -> None:
        """Check all agent states and take action as needed."""
        self._total_checks += 1
        sm = self.state_manager

        for agent_id, state in list(sm._states.items()):
            # Skip terminal agents
            if state.is_terminal:
                continue

            # Detect activity
            activity = state.detect_activity()
            old_activity = state.activity
            if activity != old_activity:
                state.activity = activity
                logger.debug(
                    "Agent '%s' activity changed: %s → %s",
                    agent_id, old_activity.value, activity.value,
                )

            # Check for stuck agents
            if self._is_stuck(state):
                self._handle_stuck(agent_id, state)

            # Check for errored agents that need recovery
            elif state.status == AgentStatus.ERROR and self.config.auto_recover:
                self._handle_error_recovery(agent_id, state)

    def check(self, agent_id: str) -> None:
        """Force-check a specific agent now."""
        state = self.state_manager.get_state(agent_id)
        if state and not state.is_terminal:
            activity = state.detect_activity()
            state.activity = activity

            if self._is_stuck(state):
                self._handle_stuck(agent_id, state)
            elif state.status == AgentStatus.ERROR:
                self._handle_error_recovery(agent_id, state)

    # ============== Stuck Detection ==============

    def _is_stuck(self, state) -> bool:
        """Check if an agent is stuck (inactive beyond threshold)."""
        if state.status in {AgentStatus.STUCK, AgentStatus.IDLE, AgentStatus.PAUSED}:
            return False  # Already handled or intentionally paused

        if state.status not in {AgentStatus.RUNNING, AgentStatus.WAITING}:
            return False

        elapsed = time.time() - state.last_activity_at
        return elapsed > self.config.stuck_threshold_sec

    def _handle_stuck(self, agent_id: str, state) -> None:
        """Handle a stuck agent."""
        self._stuck_detected += 1
        elapsed = time.time() - state.last_activity_at

        logger.warning(
            "Agent '%s' is stuck (inactive for %.0fs, threshold=%.0fs)",
            agent_id, elapsed, self.config.stuck_threshold_sec,
        )

        # Transition to STUCK status
        self.state_manager.update_state(
            agent_id,
            status=AgentStatus.STUCK,
            validate_transition=True,
        )

        # Attempt auto-recovery if enabled
        if self.config.auto_recover:
            self._attempt_recovery(agent_id)

    # ============== Recovery ==============

    def _handle_error_recovery(self, agent_id: str, state) -> None:
        """Handle recovery for errored agents."""
        if self.config.auto_recover:
            self._attempt_recovery(agent_id)

    def _attempt_recovery(self, agent_id: str) -> None:
        """Attempt to auto-recover an agent."""
        attempts = self._recovery_attempts.get(agent_id, 0)

        if attempts >= self.config.max_auto_recoveries:
            # Max retries exceeded — escalate to human
            if self.config.escalate_after_max_retries:
                self._escalate(agent_id, f"Max recovery attempts ({attempts}) exceeded")
            return

        self._recovery_attempts[agent_id] = attempts + 1
        self._recoveries_attempted += 1

        logger.info(
            "Attempting auto-recovery for agent '%s' (attempt %d/%d)",
            agent_id, attempts + 1, self.config.max_auto_recoveries,
        )

        # Emit recovery event
        bus = get_event_bus()
        bus.emit(
            EventType.REACTION_TRIGGERED,
            f"Auto-recovering agent '{agent_id}' (attempt {attempts + 1})",
            data={
                "agent_id": agent_id,
                "attempt": attempts + 1,
                "max_attempts": self.config.max_auto_recoveries,
            },
            source="lifecycle_manager",
            priority=EventPriority.WARNING,
        )

        try:
            # Transition to RECOVERING
            self.state_manager.update_state(
                agent_id,
                status=AgentStatus.RECOVERING,
                validate_transition=True,
            )

            # Create checkpoint of current (broken) state for debugging
            if self.config.checkpoint_before_recover:
                self.state_manager.create_checkpoint(agent_id)

            # Restore from last good checkpoint
            restored = self.state_manager.restore_checkpoint(agent_id)

            if restored:
                # Transition back to IDLE (ready for new work)
                self.state_manager.update_state(
                    agent_id,
                    status=AgentStatus.IDLE,
                    validate_transition=True,
                )
                self._recoveries_succeeded += 1
                self._recovery_attempts[agent_id] = 0  # Reset counter on success

                bus.emit(
                    EventType.AGENT_RECOVERED,
                    f"Agent '{agent_id}' recovered successfully",
                    data={"agent_id": agent_id},
                    source="lifecycle_manager",
                    priority=EventPriority.INFO,
                )
                logger.info("Agent '%s' recovered successfully", agent_id)
            else:
                # No checkpoint available — escalate
                self.state_manager.update_state(
                    agent_id,
                    status=AgentStatus.ERROR,
                    validate_transition=True,
                )
                self._escalate(agent_id, "No checkpoint available for recovery")

        except Exception as e:
            logger.error("Recovery failed for agent '%s': %s", agent_id, e)
            self.state_manager.update_state(
                agent_id,
                status=AgentStatus.ERROR,
                validate_transition=False,  # Force it
            )
            self._escalate(agent_id, f"Recovery error: {e}")

    # ============== Escalation ==============

    def _escalate(self, agent_id: str, reason: str) -> None:
        """Escalate to human — the agent needs manual attention."""
        self._escalations += 1

        logger.warning("ESCALATION for agent '%s': %s", agent_id, reason)

        # Emit escalation event
        bus = get_event_bus()
        bus.emit(
            EventType.REACTION_ESCALATED,
            f"Agent '{agent_id}' needs human attention: {reason}",
            data={"agent_id": agent_id, "reason": reason},
            source="lifecycle_manager",
            priority=EventPriority.URGENT,
        )

        # Call escalation callback if provided
        if self._on_escalate:
            try:
                self._on_escalate(agent_id, reason)
            except Exception as e:
                logger.error("Escalation callback failed: %s", e)

    # ============== Stats ==============

    def get_states(self) -> Dict[str, str]:
        """Get current state for all agents."""
        return {
            agent_id: state.status.value
            for agent_id, state in self.state_manager._states.items()
        }

    def get_activities(self) -> Dict[str, str]:
        """Get current activity for all agents."""
        return {
            agent_id: state.detect_activity().value
            for agent_id, state in self.state_manager._states.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get lifecycle manager statistics."""
        return {
            "is_running": self._running,
            "total_checks": self._total_checks,
            "stuck_detected": self._stuck_detected,
            "recoveries_attempted": self._recoveries_attempted,
            "recoveries_succeeded": self._recoveries_succeeded,
            "escalations": self._escalations,
            "recovery_attempts_per_agent": dict(self._recovery_attempts),
            "config": {
                "poll_interval_sec": self.config.poll_interval_sec,
                "stuck_threshold_sec": self.config.stuck_threshold_sec,
                "auto_recover": self.config.auto_recover,
                "max_auto_recoveries": self.config.max_auto_recoveries,
            },
        }

    def reset_recovery_count(self, agent_id: str) -> None:
        """Reset recovery counter for an agent (after manual intervention)."""
        self._recovery_attempts.pop(agent_id, None)
        logger.info("Reset recovery counter for agent '%s'", agent_id)


# ============== Global Instance ==============

_lifecycle_manager: Optional[LifecycleManager] = None


def get_lifecycle_manager(**kwargs) -> LifecycleManager:
    """Get or create the global lifecycle manager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager(**kwargs)
    return _lifecycle_manager


__all__ = [
    "LifecycleConfig",
    "LifecycleManager",
    "get_lifecycle_manager",
]
