"""
Natural Language to Automation for OpenClaw

Converts plain English commands into executable automation sequences:
- Parse intent and parameters from natural language
- Map to available tools and actions
- Generate step-by-step action plans
- Execute with validation
"""

import time
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger
from .agent_tools import get_tool_registry, ToolResult

logger = get_logger("nl_automation")


class ActionIntent(Enum):
    """Recognized action intents."""
    CLICK = "click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    OPEN = "open"
    SEARCH = "search"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    SCROLL = "scroll"
    READ = "read"
    CUSTOM = "custom"


@dataclass
class ParsedCommand:
    """Parsed result from natural language command."""
    original: str
    intent: ActionIntent
    target: str = ""
    value: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class AutomationStep:
    """A single step in an automation sequence."""
    step_number: int
    description: str
    tool_name: str
    tool_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed
    duration: float = 0.0


@dataclass
class AutomationPlan:
    """A complete automation plan derived from NL command."""
    command: str
    steps: List[AutomationStep] = field(default_factory=list)
    status: str = "planned"  # planned, running, completed, failed
    total_duration: float = 0.0
    error: Optional[str] = None


class IntentParser:
    """
    Parse natural language commands into structured intents.
    Uses keyword patterns for baseline, designed for LLM enhancement.
    """

    # Intent patterns — keyword → (intent, extraction_fn)
    PATTERNS = {
        ActionIntent.CLICK: [
            r"click\s+(?:on\s+)?(.+)",
            r"press\s+(?:the\s+)?(.+?)(?:\s+button)?$",
            r"tap\s+(?:on\s+)?(.+)",
        ],
        ActionIntent.TYPE: [
            r"type\s+['\"](.+?)['\"]",
            r"type\s+(.+?)(?:\s+in|\s+into|\s*$)",
            r"write\s+['\"](.+?)['\"]",
            r"enter\s+['\"](.+?)['\"]",
        ],
        ActionIntent.PRESS_KEY: [
            r"press\s+(enter|escape|tab|space|backspace|delete|f\d+|ctrl\+\w+|alt\+\w+)",
            r"hit\s+(enter|escape|tab|space|return)",
        ],
        ActionIntent.OPEN: [
            r"open\s+(.+)",
            r"launch\s+(.+)",
            r"start\s+(.+)",
            r"run\s+(.+)",
        ],
        ActionIntent.SEARCH: [
            r"search\s+(?:for\s+)?(.+)",
            r"find\s+(.+)",
            r"look\s+(?:for\s+|up\s+)?(.+)",
        ],
        ActionIntent.NAVIGATE: [
            r"go\s+to\s+(.+)",
            r"navigate\s+to\s+(.+)",
            r"visit\s+(.+)",
            r"browse\s+(?:to\s+)?(.+)",
        ],
        ActionIntent.SCREENSHOT: [
            r"(?:take\s+a?\s*)?screenshot",
            r"capture\s+(?:the\s+)?screen",
            r"grab\s+(?:the\s+)?screen",
        ],
        ActionIntent.WAIT: [
            r"wait\s+(\d+)\s*(?:seconds?|secs?|s)",
            r"pause\s+(?:for\s+)?(\d+)",
            r"sleep\s+(\d+)",
        ],
        ActionIntent.READ: [
            r"read\s+(?:the\s+)?(.+)",
            r"extract\s+text\s+(?:from\s+)?(.+)?",
            r"ocr\s+(.+)?",
            r"what\s+(?:does\s+)?(?:the\s+)?screen\s+say",
        ],
    }

    def parse(self, command: str) -> ParsedCommand:
        """Parse a natural language command into structured intent."""
        command_lower = command.strip().lower()

        best_match = None
        best_confidence = 0.0

        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    confidence = len(match.group(0)) / len(command_lower)
                    if confidence > best_confidence:
                        target = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                        best_match = ParsedCommand(
                            original=command,
                            intent=intent,
                            target=target.strip(),
                            confidence=confidence
                        )
                        best_confidence = confidence

        if best_match:
            return best_match

        # Fallback: treat as custom
        return ParsedCommand(
            original=command,
            intent=ActionIntent.CUSTOM,
            target=command,
            confidence=0.1
        )

    def parse_multi(self, command: str) -> List[ParsedCommand]:
        """Parse a command that may contain multiple steps."""
        # Split on "then", "and then", "after that", commas
        parts = re.split(r'\s+then\s+|\s+and\s+then\s+|\s+after\s+that\s+|,\s*', command)
        return [self.parse(part.strip()) for part in parts if part.strip()]


class AutomationPlanner:
    """
    Converts parsed intents into executable automation plans.
    Maps intents to tool calls.
    """

    # Intent → tool mapping
    TOOL_MAP = {
        ActionIntent.CLICK: "click",
        ActionIntent.TYPE: "type_text",
        ActionIntent.PRESS_KEY: "press_key",
        ActionIntent.SCREENSHOT: "capture_screen",
        ActionIntent.WAIT: "wait",
        ActionIntent.READ: "extract_text",
    }

    def create_plan(self, commands: List[ParsedCommand]) -> AutomationPlan:
        """Create an automation plan from parsed commands."""
        plan = AutomationPlan(
            command=" → ".join(c.original for c in commands)
        )

        for i, cmd in enumerate(commands):
            step = self._create_step(i + 1, cmd)
            plan.steps.append(step)

        return plan

    def _create_step(self, step_num: int, cmd: ParsedCommand) -> AutomationStep:
        """Create a single automation step from a parsed command."""
        tool_name = self.TOOL_MAP.get(cmd.intent, "")
        tool_args = {}

        if cmd.intent == ActionIntent.CLICK:
            tool_args = {"x": 0, "y": 0}  # Will need screen understanding
            return AutomationStep(
                step_number=step_num,
                description=f"Click on '{cmd.target}'",
                tool_name=tool_name,
                tool_args=tool_args
            )

        elif cmd.intent == ActionIntent.TYPE:
            tool_args = {"text": cmd.target}
            return AutomationStep(
                step_number=step_num,
                description=f"Type '{cmd.target}'",
                tool_name=tool_name,
                tool_args=tool_args
            )

        elif cmd.intent == ActionIntent.PRESS_KEY:
            tool_args = {"key": cmd.target}
            return AutomationStep(
                step_number=step_num,
                description=f"Press {cmd.target}",
                tool_name=tool_name,
                tool_args=tool_args
            )

        elif cmd.intent == ActionIntent.WAIT:
            seconds = float(cmd.target) if cmd.target else 1.0
            tool_args = {"seconds": seconds}
            return AutomationStep(
                step_number=step_num,
                description=f"Wait {seconds}s",
                tool_name=tool_name,
                tool_args=tool_args
            )

        elif cmd.intent == ActionIntent.SCREENSHOT:
            return AutomationStep(
                step_number=step_num,
                description="Take screenshot",
                tool_name=tool_name,
                tool_args={}
            )

        elif cmd.intent == ActionIntent.READ:
            return AutomationStep(
                step_number=step_num,
                description=f"Read text from screen",
                tool_name=tool_name,
                tool_args={}
            )

        elif cmd.intent == ActionIntent.OPEN:
            return AutomationStep(
                step_number=step_num,
                description=f"Open {cmd.target}",
                tool_name="",  # Will use subprocess or xdg-open
                tool_args={"target": cmd.target}
            )

        elif cmd.intent == ActionIntent.NAVIGATE:
            return AutomationStep(
                step_number=step_num,
                description=f"Navigate to {cmd.target}",
                tool_name="",
                tool_args={"url": cmd.target}
            )

        elif cmd.intent == ActionIntent.SEARCH:
            return AutomationStep(
                step_number=step_num,
                description=f"Search for '{cmd.target}'",
                tool_name="",
                tool_args={"query": cmd.target}
            )

        else:
            return AutomationStep(
                step_number=step_num,
                description=f"Custom: {cmd.original}",
                tool_name="",
                tool_args={"command": cmd.original}
            )


class NLAutomationEngine:
    """
    Main engine for natural language to automation conversion.

    Usage:
        engine = NLAutomationEngine()

        # Parse and plan
        plan = engine.parse_and_plan("open Chrome then go to google.com")

        # Execute
        result = engine.execute(plan)

        # Or do it all at once
        result = engine.run("type 'hello world' then press enter")
    """

    def __init__(self, llm_fn: Optional[Callable] = None):
        self.parser = IntentParser()
        self.planner = AutomationPlanner()
        self.tool_registry = get_tool_registry()
        self.llm_fn = llm_fn  # Optional LLM for better parsing
        self._history: List[AutomationPlan] = []

    def parse_and_plan(self, command: str) -> AutomationPlan:
        """Parse a natural language command and create an execution plan."""
        commands = self.parser.parse_multi(command)
        plan = self.planner.create_plan(commands)
        logger.info(f"Plan created: {len(plan.steps)} steps from '{command}'")
        return plan

    def execute(self, plan: AutomationPlan) -> AutomationPlan:
        """Execute an automation plan step by step."""
        plan.status = "running"
        start_time = time.time()

        for step in plan.steps:
            step.status = "running"
            step_start = time.time()

            try:
                if step.tool_name and step.tool_name in [t["name"] for t in self.tool_registry.list_tools()]:
                    result = self.tool_registry.execute(step.tool_name, **step.tool_args)
                    step.result = result.result if result.status.value == "success" else result.error
                    step.status = "completed" if result.status.value == "success" else "failed"
                else:
                    step.result = f"Tool '{step.tool_name}' not available"
                    step.status = "completed"

            except Exception as e:
                step.result = str(e)
                step.status = "failed"
                logger.error(f"Step {step.step_number} failed: {e}")

            step.duration = time.time() - step_start
            logger.info(f"Step {step.step_number}: {step.description} → {step.status}")

        plan.total_duration = time.time() - start_time
        plan.status = "completed" if all(s.status == "completed" for s in plan.steps) else "failed"

        self._history.append(plan)
        return plan

    def run(self, command: str) -> AutomationPlan:
        """Parse, plan, and execute a natural language command."""
        plan = self.parse_and_plan(command)
        return self.execute(plan)

    def get_history(self) -> List[AutomationPlan]:
        """Get execution history."""
        return self._history


# ============== Global Instance ==============

_engine: Optional[NLAutomationEngine] = None


def get_nl_engine() -> NLAutomationEngine:
    """Get global NL automation engine."""
    global _engine
    if _engine is None:
        _engine = NLAutomationEngine()
    return _engine


def run_command_nl(command: str) -> AutomationPlan:
    """Quick-run a natural language command."""
    return get_nl_engine().run(command)


__all__ = [
    "ActionIntent",
    "ParsedCommand",
    "AutomationStep",
    "AutomationPlan",
    "IntentParser",
    "AutomationPlanner",
    "NLAutomationEngine",
    "get_nl_engine",
    "run_command_nl",
]
