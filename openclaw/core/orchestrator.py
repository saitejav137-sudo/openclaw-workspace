"""
Multi-Agent Orchestrator for OpenClaw

Hierarchical coordinator pattern inspired by CrewAI + AutoGen.
Implements task decomposition, agent selection, parallel execution,
result aggregation, and reflect-and-verify loops.

Architecture:
    AgentOrchestrator (coordinator)
        └── TaskDecomposer (plan-and-execute)
            ├── Agent1 (parallel/sequential)
            ├── Agent2
            └── AgentN
        └── ResultAggregator (synthesize + verify)
"""

import time
import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logger import get_logger
from .agent_comm import (
    AgentCommunication, AgentMessage, MessageType, Priority,
    get_communication
)
from .agent_state import (
    AgentStateManager, AgentStatus, get_state_manager
)
from .agent_tools import ToolRegistry, get_tool_registry, ToolResult

logger = get_logger("orchestrator")


# ============== Data Models ==============

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    ASSIGNED = "assigned"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    AGGREGATING = "aggregating"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStrategy(Enum):
    """How to execute subtasks."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


@dataclass
class SubTask:
    """A decomposed subtask."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    description: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # IDs of subtasks this depends on
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: float = 180.0
    max_retries: int = 2
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Complete plan for executing a complex task."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    original_task: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return all(
            st.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for st in self.subtasks
        )

    @property
    def has_failures(self) -> bool:
        return any(st.status == TaskStatus.FAILED for st in self.subtasks)

    @property
    def progress(self) -> float:
        if not self.subtasks:
            return 0.0
        done = sum(1 for st in self.subtasks if st.status == TaskStatus.COMPLETED)
        return done / len(self.subtasks)


@dataclass
class OrchestratorResult:
    """Final result from orchestration."""
    plan_id: str
    success: bool
    result: Any = None
    subtask_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============== Agent Capability Registry ==============

@dataclass
class AgentCapability:
    """Describes what an agent can do."""
    agent_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    max_concurrent: int = 1
    active_tasks: int = 0
    success_rate: float = 1.0
    avg_duration: float = 0.0
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def is_available(self) -> bool:
        return self.is_healthy and self.active_tasks < self.max_concurrent


class AgentRegistry:
    """
    Central registry of all agent capabilities.

    Features:
    - Register/unregister agents with their capabilities
    - Find best agent for a given task type
    - Health monitoring via heartbeats
    - Load balancing across agents
    """

    def __init__(self):
        self._agents: Dict[str, AgentCapability] = {}
        self._lock = threading.Lock()
        self._heartbeat_timeout = 60.0  # seconds

    def register(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        max_concurrent: int = 1
    ) -> AgentCapability:
        """Register an agent with its capabilities."""
        with self._lock:
            agent = AgentCapability(
                agent_id=agent_id,
                name=name,
                capabilities=capabilities,
                max_concurrent=max_concurrent
            )
            self._agents[agent_id] = agent
            logger.info(f"Registered agent '{name}' ({agent_id}) with capabilities: {capabilities}")
            return agent

    def unregister(self, agent_id: str):
        """Unregister an agent."""
        with self._lock:
            if agent_id in self._agents:
                name = self._agents[agent_id].name
                del self._agents[agent_id]
                logger.info(f"Unregistered agent '{name}' ({agent_id})")

    def heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = time.time()
                self._agents[agent_id].is_healthy = True

    def find_agents(
        self,
        required_capabilities: List[str],
        available_only: bool = True
    ) -> List[AgentCapability]:
        """
        Find agents that match ALL required capabilities.
        Returns agents sorted by: availability > success_rate > avg_duration.
        """
        with self._lock:
            matches = []
            for agent in self._agents.values():
                # Check health
                if time.time() - agent.last_heartbeat > self._heartbeat_timeout:
                    agent.is_healthy = False

                # Check availability
                if available_only and not agent.is_available:
                    continue

                # Check capabilities
                if all(cap in agent.capabilities for cap in required_capabilities):
                    matches.append(agent)

            # Sort: available first, then by success rate (desc), then by speed
            matches.sort(
                key=lambda a: (-int(a.is_available), -a.success_rate, a.avg_duration)
            )
            return matches

    def get_best_agent(self, required_capabilities: List[str]) -> Optional[AgentCapability]:
        """Get the single best agent for a task."""
        agents = self.find_agents(required_capabilities)
        return agents[0] if agents else None

    def record_completion(self, agent_id: str, duration: float, success: bool):
        """Record task completion for an agent (updates stats)."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.active_tasks = max(0, agent.active_tasks - 1)
                # Update rolling average duration
                if agent.avg_duration == 0:
                    agent.avg_duration = duration
                else:
                    agent.avg_duration = (agent.avg_duration * 0.8) + (duration * 0.2)
                # Update success rate (exponential moving average)
                agent.success_rate = (agent.success_rate * 0.9) + (float(success) * 0.1)

    def list_all(self) -> List[AgentCapability]:
        """List all registered agents."""
        with self._lock:
            return list(self._agents.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total = len(self._agents)
            healthy = sum(1 for a in self._agents.values() if a.is_healthy)
            available = sum(1 for a in self._agents.values() if a.is_available)
            return {
                "total_agents": total,
                "healthy": healthy,
                "available": available,
                "agents": {
                    a.agent_id: {
                        "name": a.name,
                        "capabilities": a.capabilities,
                        "healthy": a.is_healthy,
                        "available": a.is_available,
                        "success_rate": round(a.success_rate, 3),
                        "avg_duration": round(a.avg_duration, 2),
                    }
                    for a in self._agents.values()
                }
            }


# ============== Task Decomposer ==============

class TaskDecomposer:
    """
    Breaks complex tasks into subtasks using configurable strategies.

    Supports:
    - Rule-based decomposition (keyword matching)
    - LLM-based decomposition (via AI module)
    - Template-based decomposition (pre-defined plans)
    """

    def __init__(self):
        self._templates: Dict[str, List[Dict]] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        """Register built-in task templates."""
        self._templates["research"] = [
            {"name": "search", "capabilities": ["web_search"], "description": "Search for information"},
            {"name": "analyze", "capabilities": ["analysis"], "description": "Analyze search results"},
            {"name": "summarize", "capabilities": ["summarization"], "description": "Summarize findings"},
        ]

        self._templates["vision_pipeline"] = [
            {"name": "capture", "capabilities": ["screen_capture"], "description": "Capture screen"},
            {"name": "detect", "capabilities": ["vision", "ocr"], "description": "Detect objects/text"},
            {"name": "analyze", "capabilities": ["analysis"], "description": "Analyze detections"},
            {"name": "alert", "capabilities": ["notification"], "description": "Send alerts"},
        ]

        self._templates["automation"] = [
            {"name": "plan", "capabilities": ["planning"], "description": "Plan automation steps"},
            {"name": "execute", "capabilities": ["execution", "browser"], "description": "Execute steps"},
            {"name": "verify", "capabilities": ["verification"], "description": "Verify results"},
        ]

        self._templates["code_review"] = [
            {"name": "fetch", "capabilities": ["code_access"], "description": "Fetch code changes"},
            {"name": "analyze", "capabilities": ["code_analysis"], "description": "Analyze code quality"},
            {"name": "security_scan", "capabilities": ["security"], "description": "Security analysis"},
            {"name": "report", "capabilities": ["summarization"], "description": "Generate report"},
        ]

    def register_template(self, name: str, steps: List[Dict]):
        """Register a custom task template."""
        self._templates[name] = steps
        logger.info(f"Registered task template: {name} ({len(steps)} steps)")

    def decompose(
        self,
        task: str,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        template: Optional[str] = None
    ) -> TaskPlan:
        """
        Decompose a task into subtasks.

        If a template is specified, use it directly.
        Otherwise, try to match a template by keywords.
        Falls back to creating a single subtask.
        """
        plan = TaskPlan(
            original_task=task,
            strategy=strategy
        )

        # Use specified template
        if template and template in self._templates:
            plan.subtasks = self._create_from_template(template, task)
            logger.info(f"Decomposed task using template '{template}': {len(plan.subtasks)} subtasks")
            return plan

        # Try to match template by keywords
        matched = self._match_template(task)
        if matched:
            plan.subtasks = self._create_from_template(matched, task)
            logger.info(f"Decomposed task using matched template '{matched}': {len(plan.subtasks)} subtasks")
            return plan

        # Fallback: single task
        plan.subtasks = [
            SubTask(
                name="execute",
                description=task,
                required_capabilities=["general"]
            )
        ]
        logger.info("No template matched, created single subtask")
        return plan

    def _match_template(self, task: str) -> Optional[str]:
        """Match task text to a template."""
        task_lower = task.lower()
        keyword_map = {
            "research": ["research", "search", "find", "investigate", "look up"],
            "vision_pipeline": ["screenshot", "capture", "detect", "vision", "ocr", "screen"],
            "automation": ["automate", "click", "navigate", "fill", "submit", "browser"],
            "code_review": ["code review", "pull request", "pr", "review code", "code quality"],
        }

        for template_name, keywords in keyword_map.items():
            if any(kw in task_lower for kw in keywords):
                return template_name

        return None

    def _create_from_template(self, template_name: str, task: str) -> List[SubTask]:
        """Create subtasks from a template."""
        template = self._templates[template_name]
        subtasks = []

        for i, step in enumerate(template):
            subtask = SubTask(
                name=step["name"],
                description=f"{step.get('description', step['name'])} for: {task}",
                required_capabilities=step.get("capabilities", []),
                dependencies=[subtasks[i-1].id] if i > 0 else [],
                timeout_seconds=step.get("timeout", 180.0)
            )
            subtasks.append(subtask)

        return subtasks


# ============== Result Aggregator ==============

class ResultAggregator:
    """
    Collects, validates, and synthesizes results from multiple agents.

    Supports:
    - Simple concatenation
    - Weighted merge (by agent success rate)
    - Conflict resolution
    """

    def aggregate(
        self,
        plan: TaskPlan,
        strategy: str = "merge"
    ) -> OrchestratorResult:
        """Aggregate subtask results into a final result."""
        start_time = plan.created_at
        duration = time.time() - start_time

        errors = []
        subtask_results = {}
        agents_used = set()

        for subtask in plan.subtasks:
            subtask_results[subtask.id] = {
                "name": subtask.name,
                "status": subtask.status.value,
                "result": subtask.result,
                "error": subtask.error,
                "agent": subtask.assigned_agent,
                "duration": (
                    (subtask.completed_at - subtask.started_at)
                    if subtask.started_at and subtask.completed_at
                    else None
                )
            }

            if subtask.error:
                errors.append(f"[{subtask.name}] {subtask.error}")

            if subtask.assigned_agent:
                agents_used.add(subtask.assigned_agent)

        # Determine overall success
        success = not plan.has_failures and plan.is_complete

        # Build combined result
        if strategy == "merge":
            combined = self._merge_results(plan)
        elif strategy == "last":
            combined = self._last_result(plan)
        else:
            combined = self._merge_results(plan)

        return OrchestratorResult(
            plan_id=plan.id,
            success=success,
            result=combined,
            subtask_results=subtask_results,
            errors=errors,
            duration_seconds=round(duration, 2),
            agents_used=list(agents_used),
        )

    def _merge_results(self, plan: TaskPlan) -> Any:
        """Merge all subtask results."""
        results = []
        for subtask in plan.subtasks:
            if subtask.status == TaskStatus.COMPLETED and subtask.result is not None:
                if isinstance(subtask.result, str):
                    results.append(f"## {subtask.name}\n{subtask.result}")
                else:
                    results.append(subtask.result)

        if all(isinstance(r, str) for r in results):
            return "\n\n".join(results)

        return results

    def _last_result(self, plan: TaskPlan) -> Any:
        """Return the last subtask's result (pipeline pattern)."""
        for subtask in reversed(plan.subtasks):
            if subtask.status == TaskStatus.COMPLETED and subtask.result is not None:
                return subtask.result
        return None


# ============== Main Orchestrator ==============

class AgentOrchestrator:
    """
    Hierarchical multi-agent orchestrator.

    Coordinates task decomposition, agent selection, execution,
    and result aggregation for complex multi-step tasks.

    Usage:
        orchestrator = AgentOrchestrator()

        # Register agents
        orchestrator.registry.register("vision_agent", "Vision", ["vision", "ocr"])
        orchestrator.registry.register("browser_agent", "Browser", ["browser", "navigation"])

        # Execute complex task
        result = orchestrator.execute("research AI agent best practices")
    """

    def __init__(
        self,
        max_parallel: int = 4,
        default_timeout: float = 300.0
    ):
        self.registry = AgentRegistry()
        self.decomposer = TaskDecomposer()
        self.aggregator = ResultAggregator()

        self._max_parallel = max_parallel
        self._default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        self._active_plans: Dict[str, TaskPlan] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "on_task_start": [],
            "on_task_complete": [],
            "on_subtask_start": [],
            "on_subtask_complete": [],
            "on_subtask_fail": [],
            "on_agent_assigned": [],
        }
        self._lock = threading.Lock()

    # ---- Hook Registration ----

    def on(self, event: str, handler: Callable):
        """Register an event hook."""
        if event in self._hooks:
            self._hooks[event].append(handler)

    def _fire(self, event: str, **kwargs):
        """Fire event hooks."""
        for handler in self._hooks.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                logger.error(f"Hook error ({event}): {e}")

    # ---- Task Handler Registration ----

    def register_handler(self, capability: str, handler: Callable):
        """
        Register a handler function for a capability.

        The handler receives (subtask: SubTask, context: Dict) and
        should return the result or raise an exception.
        """
        self._task_handlers[capability] = handler
        logger.info(f"Registered task handler for capability: {capability}")

    # ---- Main Execution ----

    def execute(
        self,
        task: str,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        template: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> OrchestratorResult:
        """
        Execute a complex task with multi-agent orchestration.

        Steps:
        1. Decompose task into subtasks
        2. Assign agents to subtasks
        3. Execute subtasks (sequential/parallel)
        4. Aggregate results
        5. Return final result
        """
        context = context or {}
        start_time = time.time()

        logger.info(f"Orchestrating task: {task[:100]}...")
        self._fire("on_task_start", task=task, strategy=strategy)

        # 1. Decompose
        plan = self.decomposer.decompose(task, strategy, template)
        plan.status = TaskStatus.DECOMPOSING

        with self._lock:
            self._active_plans[plan.id] = plan

        logger.info(
            f"Plan {plan.id}: {len(plan.subtasks)} subtasks, "
            f"strategy={strategy.value}"
        )

        # 2. Assign agents
        self._assign_agents(plan)
        plan.status = TaskStatus.ASSIGNED

        # 3. Execute
        try:
            plan.status = TaskStatus.RUNNING

            if strategy == ExecutionStrategy.PARALLEL:
                self._execute_parallel(plan, context)
            elif strategy == ExecutionStrategy.PIPELINE:
                self._execute_pipeline(plan, context)
            else:
                self._execute_sequential(plan, context)

        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            plan.status = TaskStatus.FAILED

        # 4. Aggregate
        plan.status = TaskStatus.AGGREGATING
        result = self.aggregator.aggregate(plan)
        result.duration_seconds = round(time.time() - start_time, 2)

        # 5. Finalize
        plan.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        plan.completed_at = time.time()

        self._fire(
            "on_task_complete",
            plan_id=plan.id,
            success=result.success,
            duration=result.duration_seconds
        )

        logger.info(
            f"Plan {plan.id} {'completed' if result.success else 'failed'} "
            f"in {result.duration_seconds}s "
            f"({len(result.agents_used)} agents used)"
        )

        return result

    def _assign_agents(self, plan: TaskPlan):
        """Assign best-fit agents to each subtask."""
        for subtask in plan.subtasks:
            if subtask.required_capabilities:
                agent = self.registry.get_best_agent(subtask.required_capabilities)
                if agent:
                    subtask.assigned_agent = agent.agent_id
                    agent.active_tasks += 1
                    self._fire(
                        "on_agent_assigned",
                        subtask=subtask.name,
                        agent=agent.name
                    )
                    logger.info(
                        f"Assigned '{agent.name}' to subtask '{subtask.name}'"
                    )
                else:
                    logger.warning(
                        f"No agent found for capabilities: {subtask.required_capabilities}"
                    )

    def _execute_sequential(self, plan: TaskPlan, context: Dict):
        """Execute subtasks one by one, passing results forward."""
        for subtask in plan.subtasks:
            # Check dependencies
            if not self._dependencies_met(subtask, plan):
                subtask.status = TaskStatus.FAILED
                subtask.error = "Unmet dependencies"
                continue

            self._execute_subtask(subtask, context)

            # Pass result to context for next subtask
            if subtask.result is not None:
                context[f"result_{subtask.name}"] = subtask.result
                context["last_result"] = subtask.result

    def _execute_parallel(self, plan: TaskPlan, context: Dict):
        """Execute independent subtasks in parallel."""
        # Group by dependency level
        levels = self._get_dependency_levels(plan)

        for level_subtasks in levels:
            if len(level_subtasks) == 1:
                # Single task — run directly
                self._execute_subtask(level_subtasks[0], context)
            else:
                # Multiple independent tasks — run in parallel
                futures = {}
                for subtask in level_subtasks:
                    future = self._executor.submit(
                        self._execute_subtask, subtask, context.copy()
                    )
                    futures[future] = subtask

                for future in as_completed(futures, timeout=self._default_timeout):
                    subtask = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        subtask.status = TaskStatus.FAILED
                        subtask.error = str(e)

            # Update context with results
            for subtask in level_subtasks:
                if subtask.result is not None:
                    context[f"result_{subtask.name}"] = subtask.result

    def _execute_pipeline(self, plan: TaskPlan, context: Dict):
        """Pipeline: each subtask's output feeds into the next."""
        pipeline_data = None

        for subtask in plan.subtasks:
            if pipeline_data is not None:
                context["pipeline_input"] = pipeline_data

            self._execute_subtask(subtask, context)
            pipeline_data = subtask.result

    def _execute_subtask(self, subtask: SubTask, context: Dict):
        """Execute a single subtask with retry logic."""
        subtask.started_at = time.time()
        subtask.status = TaskStatus.RUNNING

        self._fire("on_subtask_start", subtask=subtask)

        retries = 0
        while retries <= subtask.max_retries:
            try:
                # Find handler
                handler = self._find_handler(subtask)

                if handler:
                    subtask.result = handler(subtask, context)
                    subtask.status = TaskStatus.COMPLETED
                    subtask.completed_at = time.time()

                    # Update agent stats
                    if subtask.assigned_agent:
                        duration = subtask.completed_at - subtask.started_at
                        self.registry.record_completion(
                            subtask.assigned_agent, duration, True
                        )

                    self._fire("on_subtask_complete", subtask=subtask)
                    logger.info(f"Subtask '{subtask.name}' completed")
                    return

                else:
                    # No handler — mark as completed with no result
                    subtask.status = TaskStatus.COMPLETED
                    subtask.completed_at = time.time()
                    subtask.result = f"[No handler for {subtask.required_capabilities}]"
                    logger.warning(f"No handler for subtask '{subtask.name}'")
                    return

            except Exception as e:
                retries += 1
                if retries <= subtask.max_retries:
                    wait = 2 ** retries  # Exponential backoff
                    logger.warning(
                        f"Subtask '{subtask.name}' failed (attempt {retries}/"
                        f"{subtask.max_retries}), retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    subtask.status = TaskStatus.FAILED
                    subtask.error = str(e)
                    subtask.completed_at = time.time()

                    if subtask.assigned_agent:
                        duration = subtask.completed_at - subtask.started_at
                        self.registry.record_completion(
                            subtask.assigned_agent, duration, False
                        )

                    self._fire("on_subtask_fail", subtask=subtask, error=e)
                    logger.error(f"Subtask '{subtask.name}' failed after {retries} retries: {e}")

    def _find_handler(self, subtask: SubTask) -> Optional[Callable]:
        """Find the best handler for a subtask."""
        # Check exact match first
        for cap in subtask.required_capabilities:
            if cap in self._task_handlers:
                return self._task_handlers[cap]

        # Check general handler
        if "general" in self._task_handlers:
            return self._task_handlers["general"]

        return None

    def _dependencies_met(self, subtask: SubTask, plan: TaskPlan) -> bool:
        """Check if all dependencies for a subtask are satisfied."""
        if not subtask.dependencies:
            return True

        for dep_id in subtask.dependencies:
            dep = next((st for st in plan.subtasks if st.id == dep_id), None)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False

        return True

    def _get_dependency_levels(self, plan: TaskPlan) -> List[List[SubTask]]:
        """
        Group subtasks into levels based on dependencies.
        Level 0: No dependencies
        Level 1: Depends on level 0
        etc.
        """
        levels = []
        assigned = set()

        while len(assigned) < len(plan.subtasks):
            level = []
            for subtask in plan.subtasks:
                if subtask.id in assigned:
                    continue
                # All deps satisfied?
                deps = set(subtask.dependencies)
                if deps <= assigned:
                    level.append(subtask)

            if not level:
                # Avoid infinite loop — add all remaining
                level = [st for st in plan.subtasks if st.id not in assigned]
                levels.append(level)
                break

            for st in level:
                assigned.add(st.id)
            levels.append(level)

        return levels

    # ---- Status & Management ----

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get a plan by ID."""
        return self._active_plans.get(plan_id)

    def cancel_plan(self, plan_id: str):
        """Cancel a running plan."""
        plan = self._active_plans.get(plan_id)
        if plan:
            plan.status = TaskStatus.CANCELLED
            for subtask in plan.subtasks:
                if subtask.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    subtask.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled plan {plan_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "active_plans": len(self._active_plans),
            "registered_handlers": list(self._task_handlers.keys()),
            "registry": self.registry.get_stats(),
            "plans": {
                pid: {
                    "task": p.original_task[:50],
                    "status": p.status.value,
                    "progress": round(p.progress, 2),
                    "subtasks": len(p.subtasks),
                }
                for pid, p in self._active_plans.items()
            }
        }

    def shutdown(self):
        """Shutdown the orchestrator."""
        self._executor.shutdown(wait=False)
        logger.info("Orchestrator shutdown")


# ============== Global Access ==============

_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator(**kwargs) -> AgentOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(**kwargs)
    return _orchestrator


__all__ = [
    "TaskPriority",
    "TaskStatus",
    "ExecutionStrategy",
    "SubTask",
    "TaskPlan",
    "OrchestratorResult",
    "AgentCapability",
    "AgentRegistry",
    "TaskDecomposer",
    "ResultAggregator",
    "AgentOrchestrator",
    "get_orchestrator",
]
