"""
Persistent Agent State Management for OpenClaw

Manages agent state persistence, recovery, and checkpointing.
"""

import os
import json
import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import shutil

from .logger import get_logger

logger = get_logger("agent_state")


class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


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
    """Complete agent state"""
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
        **kwargs
    ) -> bool:
        """Update agent state"""
        with self._lock:
            state = self._states.get(agent_id)
            if not state:
                return False

            if status:
                state.status = status
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
    "AgentCheckpoint",
    "AgentState",
    "AgentStateManager",
    "get_state_manager",
    "create_agent",
    "save_checkpoint",
    "restore_agent",
]
