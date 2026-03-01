"""
Pre-built Workflow Templates for OpenClaw

Production-ready workflow templates that can be used directly
or customized for specific use cases.
"""

from typing import Dict, List, Optional, Any
from .workflow_engine import (
    WorkflowEngine, WorkflowNode, NodeType, WorkflowStatus,
    get_workflow_engine, create_workflow
)
from .logger import get_logger

logger = get_logger("workflow_templates")


# ============== Template Definitions ==============

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "deep_research": {
        "name": "Deep Research Pipeline",
        "description": "Query → Search → Analyze → Summarize → Deliver",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "name": "Start Research",
                "next_nodes": ["plan"],
            },
            {
                "id": "plan",
                "type": "agent",
                "name": "Create Research Plan",
                "config": {
                    "agent_capability": "planning",
                    "prompt_template": "Create a research plan for: {task}",
                    "max_duration": 30,
                },
                "next_nodes": ["search"],
            },
            {
                "id": "search",
                "type": "agent",
                "name": "Search & Gather",
                "config": {
                    "agent_capability": "web_search",
                    "prompt_template": "Search for information about: {task}",
                    "max_duration": 120,
                },
                "next_nodes": ["analyze"],
            },
            {
                "id": "analyze",
                "type": "agent",
                "name": "Analyze Results",
                "config": {
                    "agent_capability": "analysis",
                    "prompt_template": "Analyze these search results: {previous_result}",
                    "max_duration": 60,
                },
                "next_nodes": ["approval"],
            },
            {
                "id": "approval",
                "type": "wait",
                "name": "Human Approval Gate",
                "config": {
                    "approval_message": "Research analysis complete. Review before summarizing?",
                    "timeout": 3600,
                    "auto_approve": False,
                },
                "next_nodes": ["summarize"],
            },
            {
                "id": "summarize",
                "type": "agent",
                "name": "Summarize Findings",
                "config": {
                    "agent_capability": "summarization",
                    "prompt_template": "Create a comprehensive summary of: {analysis}",
                    "max_duration": 60,
                },
                "next_nodes": ["deliver"],
            },
            {
                "id": "deliver",
                "type": "tool",
                "name": "Deliver Report",
                "config": {
                    "tool_name": "send_message",
                    "channel": "telegram",
                },
                "next_nodes": ["end"],
            },
            {
                "id": "end",
                "type": "end",
                "name": "Research Complete",
            },
        ],
    },

    "vision_pipeline": {
        "name": "Vision Detection Pipeline",
        "description": "Capture → Detect → Analyze → Alert → Log",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "name": "Start Vision Pipeline",
                "next_nodes": ["capture"],
            },
            {
                "id": "capture",
                "type": "tool",
                "name": "Screen Capture",
                "config": {
                    "tool_name": "screen_capture",
                    "region": None,  # Full screen by default
                },
                "next_nodes": ["detect"],
            },
            {
                "id": "detect",
                "type": "agent",
                "name": "Object/Text Detection",
                "config": {
                    "agent_capability": "vision",
                    "detection_modes": ["ocr", "yolo", "template"],
                    "confidence_threshold": 0.7,
                },
                "next_nodes": ["check_detection"],
            },
            {
                "id": "check_detection",
                "type": "condition",
                "name": "Detection Found?",
                "config": {
                    "condition": "detection_count > 0",
                    "true_node": "analyze",
                    "false_node": "end",
                },
                "next_nodes": ["analyze", "end"],
            },
            {
                "id": "analyze",
                "type": "agent",
                "name": "Analyze Detection",
                "config": {
                    "agent_capability": "analysis",
                    "include_screenshot": True,
                },
                "next_nodes": ["alert"],
            },
            {
                "id": "alert",
                "type": "tool",
                "name": "Send Alert",
                "config": {
                    "tool_name": "notify",
                    "channels": ["telegram", "desktop"],
                },
                "next_nodes": ["log"],
            },
            {
                "id": "log",
                "type": "tool",
                "name": "Log Event",
                "config": {
                    "tool_name": "audit_log",
                    "log_level": "info",
                },
                "next_nodes": ["end"],
            },
            {
                "id": "end",
                "type": "end",
                "name": "Pipeline Complete",
            },
        ],
    },

    "code_review": {
        "name": "Automated Code Review",
        "description": "Fetch PR → Analyze → Security Scan → Report",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "name": "Start Code Review",
                "next_nodes": ["fetch"],
            },
            {
                "id": "fetch",
                "type": "tool",
                "name": "Fetch Code Changes",
                "config": {
                    "tool_name": "git_diff",
                },
                "next_nodes": ["parallel_analysis"],
            },
            {
                "id": "parallel_analysis",
                "type": "parallel",
                "name": "Parallel Analysis",
                "config": {
                    "parallel_nodes": ["quality", "security", "style"],
                },
                "next_nodes": ["quality", "security", "style"],
            },
            {
                "id": "quality",
                "type": "agent",
                "name": "Code Quality Analysis",
                "config": {
                    "agent_capability": "code_analysis",
                    "checks": ["complexity", "duplication", "patterns"],
                },
                "next_nodes": ["report"],
            },
            {
                "id": "security",
                "type": "agent",
                "name": "Security Scan",
                "config": {
                    "agent_capability": "security",
                    "checks": ["injection", "secrets", "vulnerabilities"],
                },
                "next_nodes": ["report"],
            },
            {
                "id": "style",
                "type": "agent",
                "name": "Style Check",
                "config": {
                    "agent_capability": "code_analysis",
                    "checks": ["formatting", "naming", "documentation"],
                },
                "next_nodes": ["report"],
            },
            {
                "id": "report",
                "type": "agent",
                "name": "Generate Report",
                "config": {
                    "agent_capability": "summarization",
                    "format": "markdown",
                },
                "next_nodes": ["end"],
            },
            {
                "id": "end",
                "type": "end",
                "name": "Review Complete",
            },
        ],
    },

    "email_monitor": {
        "name": "Email Monitor & Summarizer",
        "description": "Check → Filter → Summarize → Notify → Archive",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "name": "Start Email Monitor",
                "next_nodes": ["check"],
            },
            {
                "id": "check",
                "type": "tool",
                "name": "Check Inbox",
                "config": {
                    "tool_name": "email_check",
                    "protocol": "imap",
                    "unread_only": True,
                },
                "next_nodes": ["has_emails"],
            },
            {
                "id": "has_emails",
                "type": "condition",
                "name": "New Emails?",
                "config": {
                    "condition": "email_count > 0",
                    "true_node": "filter",
                    "false_node": "end",
                },
                "next_nodes": ["filter", "end"],
            },
            {
                "id": "filter",
                "type": "agent",
                "name": "Filter & Prioritize",
                "config": {
                    "agent_capability": "analysis",
                    "rules": ["importance", "sender", "subject"],
                },
                "next_nodes": ["summarize"],
            },
            {
                "id": "summarize",
                "type": "agent",
                "name": "Summarize Emails",
                "config": {
                    "agent_capability": "summarization",
                    "max_summary_length": 500,
                },
                "next_nodes": ["notify"],
            },
            {
                "id": "notify",
                "type": "tool",
                "name": "Send Summary",
                "config": {
                    "tool_name": "notify",
                    "channels": ["telegram"],
                },
                "next_nodes": ["end"],
            },
            {
                "id": "end",
                "type": "end",
                "name": "Monitor Complete",
            },
        ],
    },

    "automation_with_approval": {
        "name": "Automation with Human Approval",
        "description": "Plan → Preview → Approve → Execute → Verify",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "name": "Start Automation",
                "next_nodes": ["plan"],
            },
            {
                "id": "plan",
                "type": "agent",
                "name": "Generate Automation Plan",
                "config": {
                    "agent_capability": "planning",
                },
                "next_nodes": ["preview"],
            },
            {
                "id": "preview",
                "type": "agent",
                "name": "Preview Changes",
                "config": {
                    "agent_capability": "analysis",
                    "dry_run": True,
                },
                "next_nodes": ["approve"],
            },
            {
                "id": "approve",
                "type": "wait",
                "name": "Human Approval",
                "config": {
                    "approval_message": "Review automation plan. Proceed?",
                    "timeout": 7200,
                    "auto_approve": False,
                },
                "next_nodes": ["execute"],
            },
            {
                "id": "execute",
                "type": "agent",
                "name": "Execute Automation",
                "config": {
                    "agent_capability": "execution",
                    "retry_on_failure": True,
                    "max_retries": 2,
                },
                "next_nodes": ["verify"],
            },
            {
                "id": "verify",
                "type": "agent",
                "name": "Verify Results",
                "config": {
                    "agent_capability": "verification",
                },
                "next_nodes": ["end"],
            },
            {
                "id": "end",
                "type": "end",
                "name": "Automation Complete",
            },
        ],
    },
}


# ============== Template Utilities ==============

def list_templates() -> List[Dict[str, str]]:
    """List available workflow templates."""
    return [
        {"name": name, "description": t["description"]}
        for name, t in TEMPLATES.items()
    ]


def get_template(name: str) -> Optional[Dict]:
    """Get a template by name."""
    return TEMPLATES.get(name)


def create_from_template(
    template_name: str,
    workflow_name: Optional[str] = None,
    overrides: Optional[Dict] = None
) -> Optional[WorkflowEngine]:
    """
    Create a workflow engine from a template.

    Args:
        template_name: The template to use
        workflow_name: Custom name for the workflow
        overrides: Dict of node_id -> config overrides
    """
    template = TEMPLATES.get(template_name)
    if not template:
        logger.error(f"Template not found: {template_name}")
        return None

    name = workflow_name or template["name"]
    engine = WorkflowEngine(name=name)

    # Apply overrides
    nodes = template["nodes"]
    if overrides:
        for node in nodes:
            if node["id"] in overrides:
                node["config"] = {**node.get("config", {}), **overrides[node["id"]]}

    # Add nodes
    for node_def in nodes:
        engine.add_node(
            node_id=node_def["id"],
            node_type=NodeType(node_def["type"]),
            name=node_def["name"],
            config=node_def.get("config", {}),
            next_nodes=node_def.get("next_nodes", [])
        )

    logger.info(f"Created workflow from template '{template_name}': {len(nodes)} nodes")
    return engine


__all__ = [
    "TEMPLATES",
    "list_templates",
    "get_template",
    "create_from_template",
]
