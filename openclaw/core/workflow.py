"""
Workflow Engine for OpenClaw

Visual automation workflow builder and executor.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json

logger = logging.getLogger("openclaw.workflow")


class NodeType(Enum):
    """Workflow node types"""
    TRIGGER = "trigger"
    CONDITION = "condition"
    ACTION = "action"
    WAIT = "wait"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    SET_VARIABLE = "set_variable"
    GET_VARIABLE = "get_variable"
    WEBHOOK = "webhook"
    NOTIFICATION = "notification"
    LOG = "log"


class OperatorType(Enum):
    """Condition operators"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUALS = ">="
    LESS_EQUALS = "<="
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


@dataclass
class WorkflowNode:
    """Workflow node definition"""
    id: str
    type: NodeType
    name: str
    config: Dict = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    position: tuple = (0, 0)


@dataclass
class WorkflowEdge:
    """Connection between nodes"""
    source: str
    target: str
    source_port: str = "output"
    target_port: str = "input"
    condition: str = None  # For conditional edges


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str
    name: str
    description: str = ""
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class BaseNodeExecutor(ABC):
    """Base executor for workflow nodes"""

    def __init__(self, node: WorkflowNode, context: Dict):
        self.node = node
        self.context = context

    @abstractmethod
    def execute(self) -> Any:
        """Execute the node"""
        pass


class TriggerExecutor(BaseNodeExecutor):
    """Execute trigger node"""

    def execute(self) -> bool:
        """Check trigger condition"""
        # Implement trigger logic
        return True


class ConditionExecutor(BaseNodeExecutor):
    """Execute condition node"""

    def execute(self) -> bool:
        """Evaluate condition"""
        config = self.node.config
        left = config.get("left", "")
        operator = config.get("operator", OperatorType.EQUALS.value)
        right = config.get("right", "")

        # Get actual values from context
        left_val = self.context.get(left, left)
        right_val = self.context.get(right, right)

        # Evaluate
        op = OperatorType(operator)
        if op == OperatorType.EQUALS:
            return left_val == right_val
        elif op == OperatorType.NOT_EQUALS:
            return left_val != right_val
        elif op == OperatorType.GREATER_THAN:
            return float(left_val) > float(right_val)
        elif op == OperatorType.LESS_THAN:
            return float(left_val) < float(right_val)
        elif op == OperatorType.CONTAINS:
            return right_val in left_val

        return False


class ActionExecutorNode(BaseNodeExecutor):
    """Execute action node"""

    def execute(self) -> bool:
        """Execute action"""
        from openclaw.core.actions import TriggerAction, ActionSequence

        action_type = self.node.config.get("action_type", "key")
        action = self.node.config.get("action", "alt+o")
        delay = self.node.config.get("delay", 0)

        if action_type == "key":
            return TriggerAction.execute(action, delay)
        elif action_type == "sequence":
            sequence = ActionSequence()
            return sequence.execute(self.node.config.get("actions", []))

        return False


class WaitExecutor(BaseNodeExecutor):
    """Execute wait node"""

    def execute(self) -> bool:
        """Wait for specified time"""
        duration = self.node.config.get("duration", 1.0)
        time.sleep(duration)
        return True


class WebhookExecutor(BaseNodeExecutor):
    """Execute webhook node"""

    def execute(self) -> bool:
        """Execute webhook"""
        import requests

        url = self.node.config.get("url", "")
        method = self.node.config.get("method", "POST")
        data = self.node.config.get("data", {})

        try:
            response = requests.request(method, url, json=data, timeout=10)
            self.context["webhook_response"] = response.json()
            self.context["webhook_status"] = response.status_code
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False


class NotificationExecutor(BaseNodeExecutor):
    """Execute notification node"""

    def execute(self) -> bool:
        """Send notification"""
        from openclaw.integrations.alerting import alert

        title = self.node.config.get("title", "Workflow Alert")
        message = self.node.config.get("message", "")
        severity = self.node.config.get("severity", "info")

        return alert(title, message, severity=severity)


class LogExecutor(BaseNodeExecutor):
    """Execute log node"""

    def execute(self) -> bool:
        """Log message"""
        message = self.node.config.get("message", "")
        level = self.node.config.get("level", "info")

        log_func = getattr(logger, level, logger.info)
        log_func(f"[Workflow] {message}")

        return True


class WorkflowExecutor:
    """Executes workflows"""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.context: Dict = workflow.variables.copy()
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def execute(self) -> bool:
        """Execute the workflow"""
        self.running = True

        try:
            # Build execution graph
            node_map = {node.id: node for node in self.workflow.nodes}

            # Find start nodes (nodes with no incoming edges)
            incoming = {edge.target for edge in self.workflow.edges}
            start_nodes = [n for n in self.workflow.nodes if n.id not in incoming]

            # Execute each path
            for start_node in start_nodes:
                if not self._execute_node(start_node, node_map):
                    return False

            return True

        finally:
            self.running = False

    def _execute_node(self, node: WorkflowNode, node_map: Dict) -> bool:
        """Execute a single node"""
        # Create executor
        executor = self._create_executor(node)

        # Execute
        try:
            result = executor.execute()

            # Update context
            self.context[f"{node.id}_result"] = result

            # Find next nodes
            next_nodes = self._get_next_nodes(node.id)

            # Execute next nodes
            for next_node in next_nodes:
                if not self._execute_node(next_node, node_map):
                    return False

            return True

        except Exception as e:
            logger.error(f"Node {node.id} error: {e}")
            return False

    def _create_executor(self, node: WorkflowNode) -> BaseNodeExecutor:
        """Create executor for node type"""
        executors = {
            NodeType.TRIGGER: TriggerExecutor,
            NodeType.CONDITION: ConditionExecutor,
            NodeType.CONDITIONAL: ConditionExecutor,
            NodeType.ACTION: ActionExecutorNode,
            NodeType.WAIT: WaitExecutor,
            NodeType.WEBHOOK: WebhookExecutor,
            NodeType.NOTIFICATION: NotificationExecutor,
            NodeType.LOG: LogExecutor,
        }

        executor_class = executors.get(node.type, BaseNodeExecutor)
        return executor_class(node, self.context)

    def _get_next_nodes(self, node_id: str) -> List[WorkflowNode]:
        """Get next nodes in workflow"""
        # This would use the edges to find connected nodes
        # Simplified for now
        return []


class WorkflowManager:
    """Manages workflows"""

    _instance = None

    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'WorkflowManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_workflow(
        self,
        name: str,
        description: str = ""
    ) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            id=f"wf_{int(time.time())}",
            name=name,
            description=description
        )
        with self._lock:
            self.workflows[workflow.id] = workflow
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow"""
        with self._lock:
            if workflow_id in self.workflows:
                del self.workflows[workflow_id]
                return True
        return False

    def list_workflows(self) -> List[Dict]:
        """List all workflows"""
        with self._lock:
            return [
                {
                    "id": wf.id,
                    "name": wf.name,
                    "description": wf.description,
                    "enabled": wf.enabled,
                    "node_count": len(wf.nodes)
                }
                for wf in self.workflows.values()
            ]

    def execute_workflow(self, workflow_id: str) -> bool:
        """Execute a workflow"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False

        executor = WorkflowExecutor(workflow)
        return executor.execute()


# Example workflow JSON
EXAMPLE_WORKFLOW = {
    "id": "example_workflow",
    "name": "Example Workflow",
    "description": "An example workflow",
    "nodes": [
        {
            "id": "trigger1",
            "type": "trigger",
            "name": "Screen Change Detected",
            "config": {"threshold": 0.05}
        },
        {
            "id": "action1",
            "type": "action",
            "name": "Press Alt+O",
            "config": {"action": "alt+o", "delay": 1.5}
        }
    ],
    "edges": [
        {"source": "trigger1", "target": "action1"}
    ]
}


# Export
__all__ = [
    "NodeType",
    "OperatorType",
    "WorkflowNode",
    "WorkflowEdge",
    "Workflow",
    "WorkflowExecutor",
    "WorkflowManager",
    "EXAMPLE_WORKFLOW",
]
