"""
ReAct Agent Loop for OpenClaw

Implements the Reason+Act paradigm:
1. THINK — Analyze the current situation
2. ACT — Choose and execute a tool
3. OBSERVE — Process the result
4. REPEAT until goal is achieved

Based on ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al.)
"""

import time
import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .logger import get_logger
from .agent_tools import ToolRegistry, get_tool_registry, ToolResult

logger = get_logger("react_agent")


class StepType(Enum):
    """Types of steps in the ReAct loop."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning chain."""
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0


@dataclass
class ReActTrace:
    """Complete trace of a ReAct execution."""
    goal: str
    steps: List[ReActStep] = field(default_factory=list)
    status: str = "running"
    final_answer: Optional[str] = None
    total_duration: float = 0.0
    started_at: float = field(default_factory=time.time)


class ReActAgent:
    """
    Autonomous agent using Reason+Act loop.

    The agent:
    1. Receives a goal/task
    2. Thinks about what to do
    3. Selects and executes a tool
    4. Observes the result
    5. Repeats until the goal is achieved

    Usage:
        agent = ReActAgent(
            think_fn=my_llm_think,
            tool_registry=get_tool_registry()
        )

        result = agent.run("Find the current weather in Tokyo")
    """

    def __init__(
        self,
        think_fn: Optional[Callable] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        max_retries: int = 2,
        verbose: bool = True
    ):
        self.think_fn = think_fn or self._default_think
        self.tool_registry = tool_registry or get_tool_registry()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
        self._traces: List[ReActTrace] = []

    def run(self, goal: str, context: Dict[str, Any] = None) -> ReActTrace:
        """
        Execute the ReAct loop for a given goal.

        Args:
            goal: What the agent should accomplish
            context: Additional context for the agent

        Returns:
            ReActTrace with the full reasoning chain
        """
        trace = ReActTrace(goal=goal)
        context = context or {}
        context["available_tools"] = self.tool_registry.list_tools()

        logger.info(f"ReAct agent starting: {goal}")

        for step_num in range(self.max_steps):
            # THINK — What should I do next?
            thought = self._think(goal, trace.steps, context)
            trace.steps.append(ReActStep(
                step_type=StepType.THOUGHT,
                content=thought.get("reasoning", "")
            ))

            if self.verbose:
                logger.info(f"Step {step_num + 1} THINK: {thought.get('reasoning', '')[:100]}")

            # Check if agent wants to give final answer
            if thought.get("action") == "final_answer":
                answer = thought.get("answer", "")
                trace.steps.append(ReActStep(
                    step_type=StepType.FINAL_ANSWER,
                    content=answer
                ))
                trace.final_answer = answer
                trace.status = "completed"
                break

            # ACT — Execute the chosen tool
            tool_name = thought.get("action", "")
            tool_args = thought.get("action_args", {})

            act_step = ReActStep(
                step_type=StepType.ACTION,
                content=f"Execute: {tool_name}",
                tool_name=tool_name,
                tool_args=tool_args
            )

            start = time.time()
            result = self._execute_action(tool_name, tool_args)
            act_step.duration = time.time() - start
            act_step.tool_result = result
            trace.steps.append(act_step)

            if self.verbose:
                logger.info(f"Step {step_num + 1} ACT: {tool_name} → {str(result)[:100]}")

            # OBSERVE — Process the result
            observation = self._format_observation(tool_name, result)
            trace.steps.append(ReActStep(
                step_type=StepType.OBSERVATION,
                content=observation
            ))

            # Add observation to context for next iteration
            context["last_observation"] = observation

        else:
            # Max steps reached
            trace.status = "max_steps_reached"
            trace.final_answer = self._summarize_progress(trace)
            logger.warning(f"ReAct agent reached max steps ({self.max_steps})")

        trace.total_duration = time.time() - trace.started_at
        self._traces.append(trace)

        logger.info(
            f"ReAct agent finished: status={trace.status} "
            f"steps={len(trace.steps)} duration={trace.total_duration:.1f}s"
        )

        return trace

    def _think(self, goal: str, history: List[ReActStep], context: Dict) -> Dict:
        """Use the think function to reason about next action."""
        try:
            return self.think_fn(goal, history, context)
        except Exception as e:
            logger.error(f"Think function error: {e}")
            return {
                "reasoning": f"Error in reasoning: {e}",
                "action": "final_answer",
                "answer": f"Encountered an error while reasoning: {e}"
            }

    def _execute_action(self, tool_name: str, args: Dict) -> Any:
        """Execute a tool action with retry logic."""
        for attempt in range(self.max_retries + 1):
            result = self.tool_registry.execute(tool_name, **args)
            if result.status.value == "success":
                return result.result
            if attempt < self.max_retries:
                logger.warning(f"Action {tool_name} failed (attempt {attempt + 1}), retrying...")

        return f"Error: {result.error}" if result.error else "Unknown error"

    def _format_observation(self, tool_name: str, result: Any) -> str:
        """Format tool result as observation text."""
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def _summarize_progress(self, trace: ReActTrace) -> str:
        """Summarize what was accomplished when max steps are reached."""
        observations = [s for s in trace.steps if s.step_type == StepType.OBSERVATION]
        if observations:
            obs_text = observations[-1].content[:500]
            return f"I researched this extensively but couldn't get a definitive final answer in time. Here is what I found last:\n{obs_text}"
        return "I encountered an internal limit while trying to solve this. Please try asking in a different way or summarizing your request."

    def _default_think(self, goal: str, history: List[ReActStep], context: Dict) -> Dict:
        """Default think function — simple rule-based (replace with LLM)."""
        tools = context.get("available_tools", [])

        if not history:
            # First step — pick the most relevant tool
            if tools:
                return {
                    "reasoning": f"Goal: {goal}. Available tools: {[t['name'] for t in tools]}. "
                                 f"Will start with {tools[0]['name']}.",
                    "action": tools[0]["name"],
                    "action_args": {}
                }

        # If we have observations, conclude
        observations = [s for s in history if s.step_type == StepType.OBSERVATION]
        if observations:
            return {
                "reasoning": "Have gathered information. Providing answer.",
                "action": "final_answer",
                "answer": observations[-1].content
            }

        return {
            "reasoning": "No more actions needed.",
            "action": "final_answer",
            "answer": "Task completed."
        }

    def get_traces(self) -> List[ReActTrace]:
        """Get all execution traces."""
        return self._traces

    def clear_traces(self):
        """Clear execution traces."""
        self._traces.clear()


# ============== ReAct Prompt Builder ==============

class ReActPromptBuilder:
    """
    Builds prompts for LLM-powered ReAct reasoning.

    Formats the history into a prompt that an LLM can reason about.
    """

    SYSTEM_PROMPT = """You are an AI agent that uses the ReAct (Reason+Act) framework.

For each step, you must:
1. THINK: Reason about the current situation and what to do
2. ACT: Choose a tool and provide arguments
3. OBSERVE: The tool result will be provided to you

When you have enough information, respond with action "final_answer".

Available tools:
{tools}

Respond in JSON format:
{{
    "reasoning": "your step-by-step thinking",
    "action": "tool_name" or "final_answer",
    "action_args": {{}},  // for tool calls
    "answer": ""  // for final_answer
}}"""

    @classmethod
    def build_prompt(
        cls,
        goal: str,
        history: List[ReActStep],
        tools: List[Dict]
    ) -> str:
        """Build the prompt for the LLM."""
        tools_str = "\n".join(
            f"- {t['name']}: {t.get('description', '')}"
            for t in tools
        )

        prompt_parts = [
            cls.SYSTEM_PROMPT.format(tools=tools_str),
            f"\nGoal: {goal}\n"
        ]

        for step in history:
            if step.step_type == StepType.THOUGHT:
                prompt_parts.append(f"Thought: {step.content}")
            elif step.step_type == StepType.ACTION:
                prompt_parts.append(f"Action: {step.tool_name}({step.tool_args})")
            elif step.step_type == StepType.OBSERVATION:
                prompt_parts.append(f"Observation: {step.content}")

        prompt_parts.append("\nWhat is your next step?")

        return "\n".join(prompt_parts)


# ============== Global Instance ==============

_react_agent: Optional[ReActAgent] = None


def get_react_agent(**kwargs) -> ReActAgent:
    """Get global ReAct agent."""
    global _react_agent
    if _react_agent is None:
        _react_agent = ReActAgent(**kwargs)
    return _react_agent


__all__ = [
    "StepType",
    "ReActStep",
    "ReActTrace",
    "ReActAgent",
    "ReActPromptBuilder",
    "get_react_agent",
]
