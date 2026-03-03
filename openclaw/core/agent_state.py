"""
Persistent Agent State Management for OpenClaw

Manages agent state persistence, recovery, and checkpointing.

Upgraded with patterns from ComposioHQ/agent-orchestrator:
- 12-state lifecycle (was 6)
- Activity detection (active, ready, idle, blocked, exited)
- State transition validation
- Event bus integration
"""

import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import shutil

from .logger import get_logger

logger = get_logger("agent_state")


class AgentStatus(str, Enum):
    """Agent lifecycle status — 12 states inspired by agent-orchestrator."""
    # Original 6
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    # New 6 from agent-orchestrator patterns
    SPAWNING = "spawning"         # Agent is being initialized
    STUCK = "stuck"               # Agent has been inactive for too long
    RECOVERING = "recovering"     # Agent is being restored from checkpoint
    NEEDS_INPUT = "needs_input"   # Agent is waiting for human input
    COMPLETED = "completed"       # Agent finished its task successfully
    TERMINATED = "terminated"     # Agent was forcefully killed


# Terminal states — agent is done and won't transition further
TERMINAL_STATUSES: Set[AgentStatus] = {
    AgentStatus.STOPPED,
    AgentStatus.COMPLETED,
    AgentStatus.TERMINATED,
}

# States that can be auto-recovered
RECOVERABLE_STATUSES: Set[AgentStatus] = {
    AgentStatus.ERROR,
    AgentStatus.STUCK,
}

# Valid state transitions (from → set of allowed destinations)
VALID_TRANSITIONS: Dict[AgentStatus, Set[AgentStatus]] = {
    AgentStatus.SPAWNING:    {AgentStatus.IDLE, AgentStatus.RUNNING, AgentStatus.ERROR},
    AgentStatus.IDLE:        {AgentStatus.RUNNING, AgentStatus.PAUSED, AgentStatus.STOPPED, AgentStatus.TERMINATED},
    AgentStatus.RUNNING:     {AgentStatus.IDLE, AgentStatus.WAITING, AgentStatus.PAUSED, AgentStatus.STUCK, AgentStatus.NEEDS_INPUT, AgentStatus.ERROR, AgentStatus.COMPLETED, AgentStatus.TERMINATED},
    AgentStatus.WAITING:     {AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.STUCK, AgentStatus.ERROR, AgentStatus.TERMINATED},
    AgentStatus.PAUSED:      {AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.STOPPED, AgentStatus.TERMINATED},
    AgentStatus.STUCK:       {AgentStatus.RECOVERING, AgentStatus.RUNNING, AgentStatus.ERROR, AgentStatus.TERMINATED},
    AgentStatus.NEEDS_INPUT: {AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.STUCK, AgentStatus.TERMINATED},
    AgentStatus.RECOVERING:  {AgentStatus.IDLE, AgentStatus.RUNNING, AgentStatus.ERROR, AgentStatus.TERMINATED},
    AgentStatus.ERROR:       {AgentStatus.RECOVERING, AgentStatus.IDLE, AgentStatus.RUNNING, AgentStatus.TERMINATED},
    AgentStatus.COMPLETED:   {AgentStatus.IDLE, AgentStatus.SPAWNING},  # can be reused
    AgentStatus.STOPPED:     {AgentStatus.IDLE, AgentStatus.SPAWNING},
    AgentStatus.TERMINATED:  set(),  # truly terminal
}


class ActivityState(str, Enum):
    """
    Activity state as detected by monitoring — finer-grained than AgentStatus.
    Inspired by agent-orchestrator's ActivityState.
    """
    ACTIVE = "active"           # Agent is actively processing
    READY = "ready"             # Agent finished its turn, waiting for work
    IDLE = "idle"               # Agent has been inactive for a while
    WAITING_INPUT = "waiting_input"  # Agent needs human input
    BLOCKED = "blocked"         # Agent hit an error
    EXITED = "exited"           # Agent process is no longer running


# Threshold before a "ready" agent becomes "idle"
DEFAULT_IDLE_THRESHOLD_SEC = 300  # 5 minutes
# Threshold before an "idle" agent is considered "stuck"
DEFAULT_STUCK_THRESHOLD_SEC = 600  # 10 minutes


@dataclass
class AgentCheckpoint:
    """Agent state checkpoint"""
    agent_id: str
    state: Dict[str, Any]
    status: AgentStatus
    timestamp: float
    version: int


@dataclass
class AgentState:
    """Complete agent state with activity tracking."""
    agent_id: str
    name: str
    status: AgentStatus
    context: Dict[str, Any]
    memory_ids: List[str]
    tools_used: List[str]
    workflow_id: Optional[str]
    current_node: Optional[str]
    error_count: int
    success_count: int
    last_update: float
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # New fields from agent-orchestrator patterns
    activity: ActivityState = ActivityState.READY
    last_activity_at: float = field(default_factory=time.time)
    stuck_since: Optional[float] = None
    last_error: Optional[str] = None
    total_tasks: int = 0
    recovery_count: int = 0

    def detect_activity(self) -> ActivityState:
        """Detect current activity state based on timing thresholds."""
        if self.status in TERMINAL_STATUSES:
            return ActivityState.EXITED
        if self.status == AgentStatus.ERROR:
            return ActivityState.BLOCKED
        if self.status == AgentStatus.NEEDS_INPUT:
            return ActivityState.WAITING_INPUT
        if self.status == AgentStatus.RUNNING:
            return ActivityState.ACTIVE

        # Check idle/stuck thresholds
        elapsed = time.time() - self.last_activity_at
        if elapsed > DEFAULT_STUCK_THRESHOLD_SEC:
            return ActivityState.IDLE
        if elapsed > DEFAULT_IDLE_THRESHOLD_SEC:
            return ActivityState.IDLE
        return ActivityState.READY

    @property
    def is_terminal(self) -> bool:
        """Check if the agent is in a terminal state."""
        return self.status in TERMINAL_STATUSES

    @property
    def is_recoverable(self) -> bool:
        """Check if the agent can be auto-recovered."""
        return self.status in RECOVERABLE_STATUSES


class AgentStateManager:
    """
    Persistent state management for agents.

    Features:
    - Save/restore agent state
    - Checkpointing for recovery
    - State versioning
    - Auto-save intervals
    - State export/import
    """

    def __init__(
        self,
        storage_dir: str = "~/.openclaw/agent_states",
        max_checkpoints: int = 10,
        auto_save_interval: float = 60.0
    ):
        self.storage_dir = os.path.expanduser(storage_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval

        # In-memory state
        self._states: Dict[str, AgentState] = {}
        self._checkpoints: Dict[str, List[AgentCheckpoint]] = {}
        self._lock = threading.Lock()

        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load existing states
        self._load_all_states()

    def create_agent(
        self,
        agent_id: str,
        name: str,
        metadata: Dict[str, Any] = None
    ) -> AgentState:
        """Create new agent state"""
        with self._lock:
            if agent_id in self._states:
                logger.warning(f"Agent {agent_id} already exists")
                return self._states[agent_id]

            state = AgentState(
                agent_id=agent_id,
                name=name,
                status=AgentStatus.IDLE,
                context={},
                memory_ids=[],
                tools_used=[],
                workflow_id=None,
                current_node=None,
                error_count=0,
                success_count=0,
                last_update=time.time(),
                created_at=time.time(),
                metadata=metadata or {}
            )

            self._states[agent_id] = state
            self._checkpoints[agent_id] = []

            # Save to disk
            self._save_state(state)

            logger.info(f"Created agent state: {agent_id}")
            return state

    def get_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state"""
        return self._states.get(agent_id)

    def update_state(
        self,
        agent_id: str,
        status: AgentStatus = None,
        context: Dict[str, Any] = None,
        workflow_id: str = None,
        current_node: str = None,
        validate_transition: bool = True,
        **kwargs
    ) -> bool:
        """Update agent state with optional transition validation."""
        with self._lock:
            state = self._states.get(agent_id)
            if not state:
                return False

            if status and status != state.status:
                # Validate the state transition
                if validate_transition:
                    allowed = VALID_TRANSITIONS.get(state.status, set())
                    if status not in allowed:
                        logger.warning(
                            "Invalid transition for agent '%s': %s → %s (allowed: %s)",
                            agent_id, state.status.value, status.value,
                            [s.value for s in allowed],
                        )
                        return False

                old_status = state.status
                state.status = status
                state.last_activity_at = time.time()

                # Track stuck timing
                if status == AgentStatus.STUCK:
                    state.stuck_since = time.time()
                elif old_status == AgentStatus.STUCK:
                    state.stuck_since = None

                # Track recoveries
                if status == AgentStatus.RECOVERING:
                    state.recovery_count += 1

                # Update activity state
                state.activity = state.detect_activity()

                logger.info(
                    "Agent '%s' transition: %s → %s (activity: %s)",
                    agent_id, old_status.value, status.value, state.activity.value,
                )

                # Emit event via event bus (if available)
                self._emit_status_event(agent_id, old_status, status)

            if context:
                state.context.update(context)
            if workflow_id:
                state.workflow_id = workflow_id
            if current_node:
                state.current_node = current_node

            # Update counters
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)

            state.last_update = time.time()

            return True

    def _emit_status_event(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
        """Emit a status change event to the event bus."""
        try:
            from .event_bus import get_event_bus, EventType
            bus = get_event_bus()

            event_map = {
                AgentStatus.SPAWNING: EventType.AGENT_SPAWNED,
                AgentStatus.RUNNING: EventType.AGENT_WORKING,
                AgentStatus.STUCK: EventType.AGENT_STUCK,
                AgentStatus.ERROR: EventType.AGENT_ERRORED,
                AgentStatus.NEEDS_INPUT: EventType.AGENT_NEEDS_INPUT,
                AgentStatus.COMPLETED: EventType.AGENT_COMPLETED,
                AgentStatus.TERMINATED: EventType.AGENT_KILLED,
                AgentStatus.RECOVERING: EventType.AGENT_RECOVERED,
            }

            event_type = event_map.get(new_status)
            if event_type:
                bus.emit(
                    event_type=event_type,
                    message=f"Agent '{agent_id}' transitioned: {old_status.value} → {new_status.value}",
                    data={"agent_id": agent_id, "old_status": old_status.value, "new_status": new_status.value},
                    source=f"agent:{agent_id}",
                )
        except Exception:
            pass  # Event bus not available yet — that's fine

    def record_success(self, agent_id: str):
        """Record successful action"""
        with self._lock:
            state = self._states.get(agent_id)
            if state:
                state.success_count += 1
                state.last_update = time.time()

    def record_error(self, agent_id: str):
        """Record error"""
        with self._lock:
            state = self._states.get(agent_id)
            if state:
                state.error_count += 1
                state.last_update = time.time()

    def create_checkpoint(self, agent_id: str) -> Optional[AgentCheckpoint]:
        """Create state checkpoint"""
        with self._lock:
            state = self._states.get(agent_id)
            if not state:
                return None

            checkpoint = AgentCheckpoint(
                agent_id=agent_id,
                state=asdict(state),
                status=state.status,
                timestamp=time.time(),
                version=len(self._checkpoints.get(agent_id, [])) + 1
            )

            if agent_id not in self._checkpoints:
                self._checkpoints[agent_id] = []

            self._checkpoints[agent_id].append(checkpoint)

            # Limit checkpoints
            if len(self._checkpoints[agent_id]) > self.max_checkpoints:
                self._checkpoints[agent_id].pop(0)

            # Save checkpoint
            self._save_checkpoint(checkpoint)

            return checkpoint

    def restore_checkpoint(
        self,
        agent_id: str,
        version: int = None
    ) -> bool:
        """Restore from checkpoint"""
        with self._lock:
            checkpoints = self._checkpoints.get(agent_id, [])
            if not checkpoints:
                return False

            # Get checkpoint
            if version:
                checkpoint = next((c for c in checkpoints if c.version == version), None)
            else:
                checkpoint = checkpoints[-1]

            if not checkpoint:
                return False

            # Restore state
            state_data = checkpoint.state
            state = self._states.get(agent_id)
            if state:
                for key, value in state_data.items():
                    if hasattr(state, key):
                        setattr(state, key, value)

            logger.info(f"Restored agent {agent_id} to checkpoint {checkpoint.version}")
            return True

    def list_checkpoints(self, agent_id: str) -> List[Dict]:
        """List available checkpoints"""
        checkpoints = self._checkpoints.get(agent_id, [])
        return [
            {
                "version": c.version,
                "timestamp": c.timestamp,
                "status": c.status.value
            }
            for c in checkpoints
        ]

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent state"""
        with self._lock:
            if agent_id not in self._states:
                return False

            # Remove from memory
            del self._states[agent_id]
            if agent_id in self._checkpoints:
                del self._checkpoints[agent_id]

            # Remove files
            self._delete_state_file(agent_id)

            logger.info(f"Deleted agent state: {agent_id}")
            return True

    def export_state(self, agent_id: str) -> Optional[Dict]:
        """Export agent state as JSON"""
        state = self._states.get(agent_id)
        if not state:
            return None

        return asdict(state)

    def import_state(self, state_data: Dict) -> bool:
        """Import agent state from JSON"""
        try:
            state = AgentState(**state_data)
            self._states[state.agent_id] = state
            self._save_state(state)
            return True
        except Exception as e:
            logger.error(f"Import error: {e}")
            return False

    def _save_state(self, state: AgentState):
        """Save state to disk"""
        try:
            filepath = os.path.join(self.storage_dir, f"{state.agent_id}.json")
            with open(filepath, 'w') as f:
                json.dump(asdict(state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _save_checkpoint(self, checkpoint: AgentCheckpoint):
        """Save checkpoint to disk"""
        try:
            filepath = os.path.join(
                self.storage_dir,
                f"{checkpoint.agent_id}_checkpoint_{checkpoint.version}.json"
            )
            with open(filepath, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _delete_state_file(self, agent_id: str):
        """Delete state file from disk"""
        try:
            filepath = os.path.join(self.storage_dir, f"{agent_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)

            # Delete checkpoints
            for f in Path(self.storage_dir).glob(f"{agent_id}_checkpoint_*.json"):
                f.unlink()
        except Exception as e:
            logger.error(f"Failed to delete files: {e}")

    def _load_all_states(self):
        """Load all states from disk"""
        try:
            for filepath in Path(self.storage_dir).glob("*.json"):
                if "checkpoint" not in filepath.name:
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            state = AgentState(**data)
                            self._states[state.agent_id] = state
                    except Exception as e:
                        logger.error(f"Failed to load {filepath}: {e}")

            logger.info(f"Loaded {len(self._states)} agent states")

        except Exception as e:
            logger.error(f"Failed to load states: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        return {
            "total_agents": len(self._states),
            "total_checkpoints": sum(len(c) for c in self._checkpoints.values()),
            "storage_dir": self.storage_dir
        }


# Global state manager
_state_manager: Optional[AgentStateManager] = None


def get_state_manager() -> AgentStateManager:
    """Get global state manager"""
    global _state_manager
    if _state_manager is None:
        _state_manager = AgentStateManager()
    return _state_manager


def create_agent(agent_id: str, name: str, metadata: Dict = None) -> AgentState:
    """Quick create agent state"""
    return get_state_manager().create_agent(agent_id, name, metadata)


def save_checkpoint(agent_id: str) -> Optional[AgentCheckpoint]:
    """Quick create checkpoint"""
    return get_state_manager().create_checkpoint(agent_id)


def restore_agent(agent_id: str, version: int = None) -> bool:
    """Quick restore agent"""
    return get_state_manager().restore_checkpoint(agent_id, version)


__all__ = [
    "AgentStatus",
    "ActivityState",
    "AgentCheckpoint",
    "AgentState",
    "AgentStateManager",
    "get_state_manager",
    "create_agent",
    "save_checkpoint",
    "restore_agent",
    "TERMINAL_STATUSES",
    "RECOVERABLE_STATUSES",
    "VALID_TRANSITIONS",
    "DEFAULT_IDLE_THRESHOLD_SEC",
    "DEFAULT_STUCK_THRESHOLD_SEC",
]
