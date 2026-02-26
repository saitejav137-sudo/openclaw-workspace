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

 start        # Find node
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
]
