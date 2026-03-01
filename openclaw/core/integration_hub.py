"""
OpenClaw Integration Hub — Phase 4B

Central wiring point for all Phase 3 advanced modules.
Registers new tools, initialises background services, and
connects everything to the orchestrator.
"""

import time
from typing import Dict, Any, Optional, List
from .logger import get_logger

logger = get_logger("integration_hub")


class IntegrationHub:
    """
    Central integration point for all OpenClaw subsystems.

    Lazy-initialises each subsystem on first access and provides
    a unified startup/shutdown lifecycle.
    """

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._started = False

    # ---------- Component Access (lazy init) ----------

    @property
    def scheduler(self):
        """Get the task scheduler."""
        if "scheduler" not in self._components:
            from .scheduler import TaskScheduler
            self._components["scheduler"] = TaskScheduler()
        return self._components["scheduler"]

    @property
    def react_agent(self):
        """Get the ReAct agent."""
        if "react_agent" not in self._components:
            from .react_agent import ReActAgent
            self._components["react_agent"] = ReActAgent()
        return self._components["react_agent"]

    @property
    def sandbox(self):
        """Get the sandbox executor."""
        if "sandbox" not in self._components:
            from .sandbox import Sandbox
            self._components["sandbox"] = Sandbox()
        return self._components["sandbox"]

    @property
    def research_agent(self):
        """Get the research agent."""
        if "research_agent" not in self._components:
            from .research_agent import ResearchAgent
            self._components["research_agent"] = ResearchAgent()
        return self._components["research_agent"]

    @property
    def knowledge_graph(self):
        """Get the knowledge graph."""
        if "knowledge_graph" not in self._components:
            from .knowledge_graph import KnowledgeGraph
            self._components["knowledge_graph"] = KnowledgeGraph()
        return self._components["knowledge_graph"]

    @property
    def adaptive_learner(self):
        """Get the adaptive learning engine."""
        if "adaptive_learner" not in self._components:
            from .adaptive_learning import AdaptiveLearner
            self._components["adaptive_learner"] = AdaptiveLearner()
        return self._components["adaptive_learner"]

    @property
    def nl_automation(self):
        """Get the NL automation engine."""
        if "nl_automation" not in self._components:
            from .nl_automation import NLAutomationEngine
            self._components["nl_automation"] = NLAutomationEngine()
        return self._components["nl_automation"]

    @property
    def screen_engine(self):
        """Get the screen understanding engine."""
        if "screen_engine" not in self._components:
            from .screen_understanding import ScreenUnderstandingEngine
            self._components["screen_engine"] = ScreenUnderstandingEngine()
        return self._components["screen_engine"]

    @property
    def api_connectors(self):
        """Get the API connector registry."""
        if "api_connectors" not in self._components:
            from .api_connectors import APIConnectorRegistry
            self._components["api_connectors"] = APIConnectorRegistry()
        return self._components["api_connectors"]

    @property
    def agent_swarm(self):
        """Get the agent swarm."""
        if "agent_swarm" not in self._components:
            from .agent_swarm import AgentSwarm
            self._components["agent_swarm"] = AgentSwarm()
        return self._components["agent_swarm"]

    # ---------- Tool Registration ----------

    def register_tools(self):
        """Register all Phase 3 capabilities as tools in the global ToolRegistry."""
        from .agent_tools import Tool, get_tool_registry

        registry = get_tool_registry()

        # Sandbox execution tool
        def run_code(code: str, language: str = "python", timeout: int = 30) -> Dict:
            """Execute code in a sandboxed environment."""
            result = self.sandbox.execute(code, language=language, timeout=timeout)
            return {"output": result.output, "exit_code": result.exit_code,
                    "error": result.error, "duration": result.duration}

        registry.register(Tool(
            name="run_code",
            description="Execute Python/Bash/Node code in a safe sandbox with timeout enforcement",
            func=run_code,
            parameters={"code": "str", "language": "str", "timeout": "int"}
        ))

        # Research tool
        def research(query: str, max_sources: int = 5) -> Dict:
            """Research a topic using the autonomous research agent."""
            result = self.research_agent.research(query, max_sources=max_sources)
            return {"summary": result.summary, "sources": result.sources,
                    "confidence": result.confidence}

        registry.register(Tool(
            name="research",
            description="Autonomously research a topic across the web, synthesise findings",
            func=research,
            parameters={"query": "str", "max_sources": "int"}
        ))

        # NL Automation tool
        def automate(instruction: str) -> Dict:
            """Execute a natural language automation instruction."""
            result = self.nl_automation.execute(instruction)
            return {"success": result.success, "steps": result.steps_completed,
                    "result": result.output}

        registry.register(Tool(
            name="automate",
            description="Execute natural language instructions as desktop automation sequences",
            func=automate,
            parameters={"instruction": "str"}
        ))

        # Knowledge graph query tool
        def query_knowledge(query: str) -> Dict:
            """Query the knowledge graph for entities and relationships."""
            entities = self.knowledge_graph.search(query)
            return {"entities": [e.to_dict() for e in entities[:10]]}

        registry.register(Tool(
            name="query_knowledge",
            description="Search the knowledge graph for entities and relationships",
            func=query_knowledge,
            parameters={"query": "str"}
        ))

        # Schedule task tool
        def schedule_task(name: str, command: str, interval_seconds: int = 60) -> Dict:
            """Schedule a recurring task."""
            from .scheduler import ScheduleConfig, ScheduleType
            config = ScheduleConfig(
                schedule_type=ScheduleType.INTERVAL,
                interval_seconds=interval_seconds
            )
            task_id = self.scheduler.add_task(
                name, lambda: self.sandbox.execute(command), config
            )
            return {"task_id": task_id, "name": name, "interval": interval_seconds}

        registry.register(Tool(
            name="schedule_task",
            description="Schedule a recurring command to run at a fixed interval",
            func=schedule_task,
            parameters={"name": "str", "command": "str", "interval_seconds": "int"}
        ))

        logger.info(f"Registered 5 Phase 3 tools in ToolRegistry")

    # ---------- Lifecycle ----------

    def start(self):
        """Start all background services."""
        if self._started:
            return

        logger.info("Starting IntegrationHub…")
        self.register_tools()

        # Start the scheduler background loop
        try:
            self.scheduler.start()
            logger.info("Scheduler started")
        except Exception as e:
            logger.warning(f"Scheduler start failed (non-fatal): {e}")

        self._started = True
        logger.info("IntegrationHub started — all subsystems available")

    def stop(self):
        """Stop all background services gracefully."""
        if not self._started:
            return

        logger.info("Stopping IntegrationHub…")

        # Stop scheduler
        try:
            self.scheduler.stop()
        except Exception as e:
            logger.warning(f"Scheduler stop error: {e}")

        self._started = False
        logger.info("IntegrationHub stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all subsystems."""
        return {
            "started": self._started,
            "components_loaded": list(self._components.keys()),
            "component_count": len(self._components),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ---------- Global Singleton ----------

_hub: Optional[IntegrationHub] = None


def get_integration_hub() -> IntegrationHub:
    """Get or create the global IntegrationHub."""
    global _hub
    if _hub is None:
        _hub = IntegrationHub()
    return _hub


__all__ = [
    "IntegrationHub",
    "get_integration_hub",
]
