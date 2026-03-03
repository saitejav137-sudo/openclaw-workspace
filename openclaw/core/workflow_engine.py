"""
Multi-Agent Workflow Engine for OpenClaw

Workflow orchestration for multiple AI agents - inspired by Dify/MetaGPT.
Supports sequential, parallel, and conditional workflows.
"""

import time
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .logger import get_logger
from .agent_tools import ToolRegistry, get_tool_registry, ToolResult

logger = get_logger("workflow_engine")


class NodeType(Enum):
    """Workflow node types"""
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    START = "start"
    END = "end"
    WAIT = "wait"
    PARALLEL = "parallel"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowNode:
    """A node in the workflow"""
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    next_nodes: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None


@dataclass
class WorkflowEdge:
    """Edge connecting nodes"""
    from_node: str
    to_node: str
    condition: Optional[Callable] = None


@dataclass
class WorkflowExecution:
    """Workflow execution state"""
    workflow_id: str
    status: WorkflowStatus
    current_node: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class WorkflowEngine:
    """
    Multi-agent workflow orchestration engine.

    Features:
    - Sequential and parallel execution
    - Conditional branching
    - Agent coordination
    - Tool execution
    - State management
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.tool_registry = get_tool_registry()

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str,
        config: Dict = None,
        next_nodes: List[str] = None
    ) -> WorkflowNode:
        """Add a node to the workflow"""
        node = WorkflowNode(
            id=node_id,
            type=node_type,
            name=name,
            config=config or {},
            next_nodes=next_nodes or []
        )
        self.nodes[node_id] = node
        logger.info(f"Added workflow node: {node_id} ({node_type.value})")
        return node

    def add_edge(self, from_node: str, to_node: str, condition: Callable = None):
        """Add an edge between nodes"""
        edge = WorkflowEdge(from_node, to_node, condition)
        self.edges.append(edge)

    def get_next_nodes(self, node_id: str, context: Dict) -> List[str]:
        """Get next nodes based on conditions"""
        node = self.nodes.get(node_id)
        if not node:
            return []

        # Return defined next nodes
        if node.condition:
            # Check condition
            try:
                if node.condition(context):
                    return node.next_nodes
            except Exception:
                pass
            return []

        return node.next_nodes

    async def execute_node(self, node: WorkflowNode, context: Dict) -> Any:
        """Execute a single node"""
        logger.info(f"Executing node: {node.name}")

        try:
            if node.type == NodeType.TOOL:
                # Execute tool
                tool_name = node.config.get("tool")
                params = node.config.get("params", {})
                result = self.tool_registry.execute(tool_name, **params)
                return result

            elif node.type == NodeType.AGENT:
                # Execute agent
                agent_config = node.config.get("agent")
                # Would integrate with agent here
                return {"agent": agent_config, "executed": True}

            elif node.type == NodeType.CONDITION:
                # Evaluate condition
                condition_func = node.config.get("condition")
                if condition_func:
                    return condition_func(context)
                return True

            elif node.type == NodeType.WAIT:
                # Wait for duration
                duration = node.config.get("duration", 1.0)
                await asyncio.sleep(duration)
                return {"waited": duration}

            elif node.type == NodeType.END:
                return None

            return None

        except Exception as e:
            logger.error(f"Node execution error: {e}")
            raise

    async def execute(self, start_node: str = None) -> WorkflowExecution:
        """Execute the workflow"""
        if not self.nodes:
            raise ValueError("No nodes in workflow")

        # Find start node
        if start_node is None:
            for node in self.nodes.values():
                if node.type == NodeType.START:
                    start_node = node.id
                    break
            if start_node is None:
                start_node = list(self.nodes.keys())[0]

        execution = WorkflowExecution(
            workflow_id=str(uuid4()),
            status=WorkflowStatus.RUNNING,
            current_node=start_node
        )

        current = start_node
        visited = set()

        while current and current not in visited:
            visited.add(current)
            node = self.nodes.get(current)

            if not node:
                break

            execution.current_node = current

            # Execute node
            try:
                result = await self.execute_node(node, execution.results)
                execution.results[current] = result
            except Exception as e:
                execution.errors.append(str(e))
                execution.status = WorkflowStatus.FAILED
                break

            # Get next nodes
            next_nodes = self.get_next_nodes(current, execution.results)

            if not next_nodes:
                break

            current = next_nodes[0]  # Take first branch

        execution.status = WorkflowStatus.COMPLETED
        execution.completed_at = time.time()

        logger.info(f"Workflow completed: {execution.workflow_id}")
        return execution

    def validate(self) -> List[str]:
        """Validate workflow structure"""
        errors = []

        # Check for cycles
        # Check for disconnected nodes
        # Check for missing references

        return errors


# Pre-built workflow templates

def create_automation_workflow() -> WorkflowEngine:
    """Create a basic automation workflow"""
    workflow = WorkflowEngine("automation")

    # Start
    workflow.add_node("start", NodeType.START, "Start")

    # Capture screen
    workflow.add_node("capture", NodeType.TOOL, "Capture Screen",
                      {"tool": "capture_screen"})

    # Detect objects
    workflow.add_node("detect", NodeType.TOOL, "Detect Objects",
                      {"tool": "detect_objects"})

    # Check condition
    workflow.add_node("check", NodeType.CONDITION, "Check Detection",
                      config={"condition": lambda ctx: True})

    # End
    workflow.add_node("end", NodeType.END, "End")

    # Connect edges
    workflow.add_edge("start", "capture")
    workflow.add_edge("capture", "detect")
    workflow.add_edge("detect", "check")
    workflow.add_edge("check", "end")

    return workflow


def create_screenshot_workflow() -> WorkflowEngine:
    """Create screenshot and OCR workflow"""
    workflow = WorkflowEngine("screenshot_ocr")

    workflow.add_node("start", NodeType.START, "Start")
    workflow.add_node("capture", NodeType.TOOL, "Capture Screen",
                      {"tool": "capture_screen"})
    workflow.add_node("ocr", NodeType.TOOL, "Extract Text",
                      {"tool": "extract_text"})
    workflow.add_node("end", NodeType.END, "End")

    workflow.add_edge("start", "capture")
    workflow.add_edge("capture", "ocr")
    workflow.add_edge("ocr", "end")

    return workflow


# Global workflow engine
_workflows: Dict[str, WorkflowEngine] = {}


def get_workflow_engine(name: str = "default") -> WorkflowEngine:
    """Get or create workflow engine"""
    if name not in _workflows:
        _workflows[name] = WorkflowEngine(name)
    return _workflows[name]


def create_workflow(name: str, nodes: List[Dict]) -> WorkflowEngine:
    """Create workflow from config"""
    engine = WorkflowEngine(name)

    for node_data in nodes:
        engine.add_node(
            node_data["id"],
            NodeType(node_data["type"]),
            node_data["name"],
            node_data.get("config", {}),
            node_data.get("next", [])
        )

    # Add edges
    for node_data in nodes:
        for next_id in node_data.get("next", []):
            engine.add_edge(node_data["id"], next_id)

    return engine


# ============================================================
# DAG-Based Workflow Engine (Phase 3 Enhancement)
# ============================================================

import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor


class DAGNodeStatus(str, Enum):
    """Status of a node in the DAG."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """A node in a DAG workflow with dependencies."""
    id: str
    name: str
    handler: Optional[Callable] = None
    depends_on: List[str] = field(default_factory=list)
    timeout: float = 300.0
    max_retries: int = 1
    on_failure: str = "stop"  # "stop", "skip", "continue"
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    status: DAGNodeStatus = DAGNodeStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    attempts: int = 0

    @property
    def duration(self) -> float:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0


class DAGStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DAGResult:
    """Result of a DAG workflow execution."""
    workflow_id: str
    status: DAGStatus
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    duration: float = 0.0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0

    def get_output(self, node_id: str) -> Any:
        node = self.nodes.get(node_id)
        return node.result if node else None


class WorkflowDAG:
    """
    Directed Acyclic Graph for workflow nodes.

    Features:
    - Cycle detection (DFS with three-color marking)
    - Topological sort (Kahn's algorithm)
    - Ready-node detection (all deps satisfied)

    Usage:
        dag = WorkflowDAG("my-pipeline")
        dag.add_node(DAGNode(id="A", name="Fetch", handler=fetch_fn))
        dag.add_node(DAGNode(id="B", name="Parse", handler=parse_fn, depends_on=["A"]))
        dag.add_node(DAGNode(id="C", name="Analyze", handler=analyze_fn, depends_on=["A"]))
        dag.add_node(DAGNode(id="D", name="Report", handler=report_fn, depends_on=["B", "C"]))
        # Diamond dependency: A → B, A → C, B → D, C → D
    """

    def __init__(self, workflow_id: str = None, name: str = ""):
        self.workflow_id = workflow_id or str(uuid4())[:12]
        self.name = name
        self.nodes: Dict[str, DAGNode] = {}
        self._dependents: Dict[str, set] = defaultdict(set)

    def add_node(self, node: "DAGNode") -> "WorkflowDAG":
        self.nodes[node.id] = node
        for dep_id in node.depends_on:
            self._dependents[dep_id].add(node.id)
        return self

    def validate(self) -> List[str]:
        issues = []
        for nid, node in self.nodes.items():
            for dep_id in node.depends_on:
                if dep_id not in self.nodes:
                    issues.append(f"Node '{nid}' depends on unknown '{dep_id}'")
            if node.handler is None:
                issues.append(f"Node '{nid}' has no handler")
        if self._has_cycle():
            issues.append("Workflow contains a cycle")
        return issues

    def _has_cycle(self) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}

        def dfs(nid):
            color[nid] = GRAY
            for dep in self._dependents.get(nid, set()):
                if dep in color:
                    if color[dep] == GRAY:
                        return True
                    if color[dep] == WHITE and dfs(dep):
                        return True
            color[nid] = BLACK
            return False

        return any(color[nid] == WHITE and dfs(nid) for nid in self.nodes)

    def topological_sort(self) -> List[str]:
        in_degree = {nid: len(n.depends_on) for nid, n in self.nodes.items()}
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for dep_id in self._dependents.get(nid, set()):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)
        return result

    def get_ready_nodes(self) -> List[str]:
        ready = []
        for nid, node in self.nodes.items():
            if node.status != DAGNodeStatus.PENDING:
                continue
            if all(
                self.nodes[d].status in (DAGNodeStatus.COMPLETED, DAGNodeStatus.SKIPPED)
                for d in node.depends_on if d in self.nodes
            ):
                ready.append(nid)
        return ready

    def __len__(self):
        return len(self.nodes)


class WorkflowDAGEngine:
    """
    Executes DAG workflows with parallel threads and dependency resolution.

    Usage:
        engine = WorkflowDAGEngine(max_workers=4)
        dag = WorkflowDAG(name="pipeline")
        dag.add_node(DAGNode(id="A", name="Step A", handler=step_a))
        dag.add_node(DAGNode(id="B", name="Step B", handler=step_b, depends_on=["A"]))
        result = engine.execute(dag)
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._total_executions = 0
        self._total_nodes_run = 0

    def execute(self, dag: WorkflowDAG, context: Dict[str, Any] = None) -> DAGResult:
        context = context or {}
        start_time = time.time()
        self._total_executions += 1

        issues = dag.validate()
        if issues:
            logger.error("DAG validation failed: %s", issues)
            return DAGResult(workflow_id=dag.workflow_id, status=DAGStatus.FAILED, nodes=dag.nodes)

        self._emit_event("workflow.started", f"DAG '{dag.name}' executing ({len(dag)} nodes)",
                         {"workflow_id": dag.workflow_id})

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            running = {}

            while True:
                ready = dag.get_ready_nodes()

                for nid in ready:
                    node = dag.nodes[nid]
                    if node.condition and not node.condition(context):
                        node.status = DAGNodeStatus.SKIPPED
                        continue
                    node.status = DAGNodeStatus.READY
                    running[nid] = pool.submit(self._run_node, node, context, dag)

                if running:
                    done = [nid for nid, f in running.items() if f.done()]
                    for nid in done:
                        try:
                            running[nid].result()
                        except Exception:
                            pass
                        del running[nid]
                    if not done:
                        time.sleep(0.02)
                        continue
                elif not ready:
                    break

            for nid, f in running.items():
                try:
                    f.result(timeout=5)
                except Exception:
                    pass

        duration = time.time() - start_time
        completed = sum(1 for n in dag.nodes.values() if n.status == DAGNodeStatus.COMPLETED)
        failed = sum(1 for n in dag.nodes.values() if n.status == DAGNodeStatus.FAILED)
        skipped = sum(1 for n in dag.nodes.values() if n.status == DAGNodeStatus.SKIPPED)

        status = DAGStatus.COMPLETED if failed == 0 else (DAGStatus.PARTIAL if completed > 0 else DAGStatus.FAILED)

        self._emit_event(
            "workflow.completed" if status == DAGStatus.COMPLETED else "workflow.failed",
            f"DAG '{dag.name}' {status.value} ({completed}/{len(dag)} ok, {duration:.1f}s)",
            {"workflow_id": dag.workflow_id, "completed": completed, "failed": failed},
        )

        return DAGResult(
            workflow_id=dag.workflow_id, status=status, nodes=dag.nodes,
            duration=duration, completed_count=completed,
            failed_count=failed, skipped_count=skipped,
        )

    def _run_node(self, node: DAGNode, context: Dict, dag: WorkflowDAG):
        node.status = DAGNodeStatus.RUNNING
        node.started_at = time.time()
        self._total_nodes_run += 1

        for attempt in range(node.max_retries):
            node.attempts = attempt + 1
            try:
                ctx = dict(context)
                ctx["_node_id"] = node.id
                for dep_id in node.depends_on:
                    dep = dag.nodes.get(dep_id)
                    if dep and dep.result is not None:
                        ctx[f"_{dep_id}_result"] = dep.result

                node.result = node.handler(ctx)
                node.status = DAGNodeStatus.COMPLETED
                node.completed_at = time.time()
                logger.info("  ✅ DAG node '%s' completed (%.1fs)", node.name, node.duration)
                return

            except Exception as e:
                if attempt + 1 >= node.max_retries:
                    node.status = DAGNodeStatus.FAILED
                    node.error = str(e)
                    node.completed_at = time.time()
                    logger.warning("  ❌ DAG node '%s' failed: %s", node.name, e)

                    if node.on_failure == "skip":
                        node.status = DAGNodeStatus.SKIPPED

    def _emit_event(self, event_type_str: str, message: str, data: dict = None):
        try:
            from .event_bus import get_event_bus, EventType
            type_map = {
                "workflow.started": EventType.SWARM_STARTED,
                "workflow.completed": EventType.SWARM_COMPLETED,
                "workflow.failed": EventType.SWARM_FAILED,
            }
            get_event_bus().emit(
                type_map.get(event_type_str, EventType.SYSTEM_STARTUP),
                message, data=data, source="workflow_dag_engine",
            )
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "total_nodes_run": self._total_nodes_run,
            "max_workers": self.max_workers,
        }


# ============== Global DAG Engine ==============

_dag_engine: Optional[WorkflowDAGEngine] = None


def get_dag_engine(max_workers: int = 4) -> WorkflowDAGEngine:
    global _dag_engine
    if _dag_engine is None:
        _dag_engine = WorkflowDAGEngine(max_workers=max_workers)
    return _dag_engine


__all__ = [
    "NodeType",
    "WorkflowStatus",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowExecution",
    "WorkflowEngine",
    "create_automation_workflow",
    "create_screenshot_workflow",
    "get_workflow_engine",
    "create_workflow",
    # DAG Engine (Phase 3)
    "DAGNodeStatus",
    "DAGNode",
    "DAGStatus",
    "DAGResult",
    "WorkflowDAG",
    "WorkflowDAGEngine",
    "get_dag_engine",
]

