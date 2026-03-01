"""
Agent Swarm Mode for OpenClaw

Multiple specialized agents collaborating on complex tasks:
- Role assignment (researcher, coder, reviewer, etc.)
- Communication protocols between agents
- Task decomposition and assignment
- Result aggregation and conflict resolution
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .logger import get_logger

logger = get_logger("agent_swarm")


class AgentRole(Enum):
    """Pre-defined agent roles."""
    RESEARCHER = "researcher"      # Gathers information
    CODER = "coder"                # Writes and fixes code
    REVIEWER = "reviewer"          # Reviews work quality
    PLANNER = "planner"            # Creates plans and strategies
    EXECUTOR = "executor"          # Executes actions
    MONITOR = "monitor"            # Monitors system health
    COMMUNICATOR = "communicator"  # Handles external comms
    CUSTOM = "custom"


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK = "task"              # Task assignment
    RESULT = "result"          # Task result
    QUERY = "query"            # Information request
    RESPONSE = "response"      # Response to query
    BROADCAST = "broadcast"    # Broadcast to all
    STATUS = "status"          # Status update
    ERROR = "error"            # Error notification


@dataclass
class SwarmMessage:
    """Message between agents in the swarm."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    sender: str = ""
    receiver: str = ""  # Empty = broadcast
    message_type: MessageType = MessageType.TASK
    content: Any = None
    priority: int = 0  # Higher = more important
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None


@dataclass
class SwarmAgent:
    """An agent in the swarm."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    role: AgentRole = AgentRole.EXECUTOR
    capabilities: List[str] = field(default_factory=list)
    handler: Optional[Callable] = None  # Function that processes tasks
    status: str = "idle"  # idle, busy, error
    current_task: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_active: float = field(default_factory=time.time)


@dataclass
class SwarmTask:
    """A task for the swarm to complete."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    description: str = ""
    assigned_to: Optional[str] = None
    required_role: Optional[AgentRole] = None
    required_capabilities: List[str] = field(default_factory=list)
    priority: int = 0
    status: str = "pending"  # pending, assigned, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)  # IDs of subtasks
    parent_task: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class MessageBus:
    """
    Communication bus for inter-agent messaging.
    Supports direct messages, broadcasts, and pub/sub.
    """

    def __init__(self):
        self._queues: Dict[str, List[SwarmMessage]] = {}  # agent_id -> messages
        self._broadcast_log: List[SwarmMessage] = []
        self._subscribers: Dict[str, List[Callable]] = {}  # topic -> handlers
        self._lock = threading.Lock()

    def send(self, message: SwarmMessage):
        """Send a message to a specific agent or broadcast."""
        with self._lock:
            if message.receiver:
                # Direct message
                if message.receiver not in self._queues:
                    self._queues[message.receiver] = []
                self._queues[message.receiver].append(message)
            else:
                # Broadcast
                self._broadcast_log.append(message)
                for queue in self._queues.values():
                    queue.append(message)

            logger.debug(
                f"Message {message.id}: {message.sender} → "
                f"{message.receiver or 'ALL'} ({message.message_type.value})"
            )

    def receive(self, agent_id: str, limit: int = 10) -> List[SwarmMessage]:
        """Receive messages for an agent."""
        with self._lock:
            if agent_id not in self._queues:
                return []

            messages = self._queues[agent_id][:limit]
            self._queues[agent_id] = self._queues[agent_id][limit:]
            return messages

    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)

    def register_agent(self, agent_id: str):
        """Register an agent's message queue."""
        with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = []

    def pending_count(self, agent_id: str) -> int:
        """Get number of pending messages for an agent."""
        with self._lock:
            return len(self._queues.get(agent_id, []))


class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks for the swarm.
    """

    def decompose(
        self,
        task: SwarmTask,
        available_roles: List[AgentRole]
    ) -> List[SwarmTask]:
        """
        Decompose a task into subtasks based on available roles.
        Returns a list of subtasks.
        """
        subtasks = []
        description = task.description.lower()

        # Research phase
        if AgentRole.RESEARCHER in available_roles:
            subtasks.append(SwarmTask(
                description=f"Research: {task.description}",
                required_role=AgentRole.RESEARCHER,
                parent_task=task.id,
                priority=task.priority
            ))

        # Planning phase
        if AgentRole.PLANNER in available_roles:
            subtasks.append(SwarmTask(
                description=f"Plan approach for: {task.description}",
                required_role=AgentRole.PLANNER,
                parent_task=task.id,
                priority=task.priority
            ))

        # Execution phase
        if any(word in description for word in ["code", "implement", "build", "fix"]):
            if AgentRole.CODER in available_roles:
                subtasks.append(SwarmTask(
                    description=f"Implement: {task.description}",
                    required_role=AgentRole.CODER,
                    parent_task=task.id,
                    priority=task.priority
                ))
        else:
            if AgentRole.EXECUTOR in available_roles:
                subtasks.append(SwarmTask(
                    description=f"Execute: {task.description}",
                    required_role=AgentRole.EXECUTOR,
                    parent_task=task.id,
                    priority=task.priority
                ))

        # Review phase
        if AgentRole.REVIEWER in available_roles:
            subtasks.append(SwarmTask(
                description=f"Review results of: {task.description}",
                required_role=AgentRole.REVIEWER,
                parent_task=task.id,
                priority=task.priority - 1  # Lower priority, run after
            ))

        # Update parent task
        task.subtasks = [st.id for st in subtasks]

        if not subtasks:
            # No decomposition possible, return as-is
            subtasks.append(SwarmTask(
                description=task.description,
                parent_task=task.id,
                priority=task.priority
            ))

        return subtasks


class AgentSwarm:
    """
    Multi-agent swarm for collaborative task execution.

    Usage:
        swarm = AgentSwarm()

        # Add agents
        swarm.add_agent("researcher", AgentRole.RESEARCHER, research_handler)
        swarm.add_agent("coder", AgentRole.CODER, code_handler)
        swarm.add_agent("reviewer", AgentRole.REVIEWER, review_handler)

        # Submit a task
        result = swarm.submit_task("Build a data processing pipeline")
    """

    def __init__(self):
        self._agents: Dict[str, SwarmAgent] = {}
        self._tasks: Dict[str, SwarmTask] = {}
        self.message_bus = MessageBus()
        self.decomposer = TaskDecomposer()
        self._lock = threading.Lock()

    def add_agent(
        self,
        name: str,
        role: AgentRole,
        handler: Callable = None,
        capabilities: List[str] = None
    ) -> str:
        """Add an agent to the swarm."""
        agent = SwarmAgent(
            name=name,
            role=role,
            capabilities=capabilities or [],
            handler=handler
        )

        with self._lock:
            self._agents[agent.id] = agent
            self.message_bus.register_agent(agent.id)

        logger.info(f"Agent '{name}' ({role.value}) joined swarm")
        return agent.id

    def remove_agent(self, agent_id: str):
        """Remove an agent from the swarm."""
        with self._lock:
            if agent_id in self._agents:
                name = self._agents[agent_id].name
                del self._agents[agent_id]
                logger.info(f"Agent '{name}' left swarm")

    def submit_task(
        self,
        description: str,
        priority: int = 0,
        decompose: bool = True
    ) -> SwarmTask:
        """Submit a task to the swarm. Blocks until all subtasks are complete."""
        task = SwarmTask(
            description=description,
            priority=priority
        )

        with self._lock:
            self._tasks[task.id] = task

        logger.info(f"Task submitted: {description}")

        if decompose:
            available_roles = list(set(a.role for a in self._agents.values()))
            subtasks = self.decomposer.decompose(task, available_roles)

            for st in subtasks:
                with self._lock:
                    self._tasks[st.id] = st
                self._assign_task(st)
                
            # Wait for all subtasks to complete
            start_time = time.time()
            timeout = 180  # Max 3 minutes
            while True:
                all_done = True
                for st in subtasks:
                    # Refresh status from dict
                    with self._lock:
                        curr_st = self._tasks.get(st.id)
                    if not curr_st or curr_st.status not in ("completed", "failed"):
                        all_done = False
                        break
                        
                if all_done or time.time() - start_time > timeout:
                    break
                    
                time.sleep(1)  # Poll every second
                
            with self._lock:
                task.status = "completed"
        else:
            self._assign_task(task)
            
            # Wait for single task
            start_time = time.time()
            timeout = 180
            while True:
                with self._lock:
                    curr_task = self._tasks.get(task.id)
                if not curr_task or curr_task.status in ("completed", "failed") or time.time() - start_time > timeout:
                    break
                time.sleep(1)

        return task

    def _assign_task(self, task: SwarmTask):
        """Assign a task to the best available agent."""
        with self._lock:
            best_agent = None

            for agent in self._agents.values():
                if agent.status != "idle":
                    continue

                # Role match
                if task.required_role and agent.role != task.required_role:
                    continue

                # Capability match
                if task.required_capabilities:
                    has_all = all(
                        cap in agent.capabilities
                        for cap in task.required_capabilities
                    )
                    if not has_all:
                        continue

                best_agent = agent
                break

            if best_agent:
                task.assigned_to = best_agent.id
                task.status = "assigned"
                best_agent.status = "busy"
                best_agent.current_task = task.id

                # Send task message
                self.message_bus.send(SwarmMessage(
                    sender="swarm",
                    receiver=best_agent.id,
                    message_type=MessageType.TASK,
                    content=task.description,
                    priority=task.priority
                ))

                logger.info(f"Task assigned to {best_agent.name}: {task.description}")

                # Execute in parallel thread
                thread = threading.Thread(
                    target=self._execute_task,
                    args=(best_agent, task),
                    daemon=True
                )
                thread.start()
            else:
                logger.warning(f"No available agent for task: {task.description}")

    def _execute_task(self, agent: SwarmAgent, task: SwarmTask):
        """Execute a task using the agent's handler."""
        if not agent.handler:
            task.status = "completed"
            task.result = "No handler - task acknowledged"
            agent.status = "idle"
            agent.current_task = None
            agent.completed_tasks += 1
            return

        task.status = "running"

        try:
            result = agent.handler(task.description)
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
            agent.completed_tasks += 1

            # Broadcast result
            self.message_bus.send(SwarmMessage(
                sender=agent.id,
                message_type=MessageType.RESULT,
                content={"task": task.id, "result": result}
            ))

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            agent.failed_tasks += 1
            logger.error(f"Agent '{agent.name}' task failed: {e}")

        finally:
            agent.status = "idle"
            agent.current_task = None
            agent.last_active = time.time()

    def get_task_status(self, task_id: str) -> Optional[SwarmTask]:
        """Get task status."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get overall swarm status."""
        with self._lock:
            agents = {
                a.id: {
                    "name": a.name,
                    "role": a.role.value,
                    "status": a.status,
                    "completed": a.completed_tasks,
                    "failed": a.failed_tasks
                }
                for a in self._agents.values()
            }

            task_stats = {
                "total": len(self._tasks),
                "pending": sum(1 for t in self._tasks.values() if t.status == "pending"),
                "running": sum(1 for t in self._tasks.values() if t.status == "running"),
                "completed": sum(1 for t in self._tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self._tasks.values() if t.status == "failed"),
            }

            return {
                "agents": agents,
                "agent_count": len(self._agents),
                "tasks": task_stats
            }

    def get_results(self, task_id: str) -> Dict[str, Any]:
        """Get aggregated results for a task and its subtasks."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"error": "Task not found"}

            results = {"main": task.result}

            for subtask_id in task.subtasks:
                subtask = self._tasks.get(subtask_id)
                if subtask:
                    results[subtask.description] = {
                        "status": subtask.status,
                        "result": subtask.result,
                        "error": subtask.error
                    }

            return results


# ============== Global Instance ==============

_swarm: Optional[AgentSwarm] = None


def get_swarm() -> AgentSwarm:
    """Get global agent swarm."""
    global _swarm
    if _swarm is None:
        _swarm = AgentSwarm()
    return _swarm


__all__ = [
    "AgentRole",
    "MessageType",
    "SwarmMessage",
    "SwarmAgent",
    "SwarmTask",
    "MessageBus",
    "TaskDecomposer",
    "AgentSwarm",
    "get_swarm",
]
