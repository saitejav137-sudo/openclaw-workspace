"""
Tests for Phase 3 Advanced Features
Covers all 10 new modules
"""

import time
import os
import pytest
from unittest.mock import Mock, patch


# ============== ReAct Agent Tests ==============

class TestReActAgent:

    def test_basic_run(self):
        from openclaw.core.react_agent import ReActAgent
        agent = ReActAgent(max_steps=3)
        trace = agent.run("test goal")
        assert trace.status in ("completed", "max_steps_reached")
        assert len(trace.steps) > 0

    def test_final_answer(self):
        from openclaw.core.react_agent import ReActAgent
        def think_fn(goal, history, ctx):
            return {"reasoning": "Done", "action": "final_answer", "answer": "42"}
        agent = ReActAgent(think_fn=think_fn)
        trace = agent.run("what is the answer")
        assert trace.final_answer == "42"
        assert trace.status == "completed"

    def test_max_steps(self):
        from openclaw.core.react_agent import ReActAgent
        def never_finish(goal, history, ctx):
            tools = ctx.get("available_tools", [])
            if tools:
                return {"reasoning": "Keep going", "action": tools[0]["name"], "action_args": {}}
            return {"reasoning": "No tools", "action": "final_answer", "answer": "no tools"}
        agent = ReActAgent(think_fn=never_finish, max_steps=2)
        trace = agent.run("infinite task")
        assert trace.status in ("completed", "max_steps_reached")

    def test_prompt_builder(self):
        from openclaw.core.react_agent import ReActPromptBuilder
        prompt = ReActPromptBuilder.build_prompt("test goal", [], [{"name": "search", "description": "Search"}])
        assert "test goal" in prompt
        assert "search" in prompt


# ============== Adaptive Learning Tests ==============

class TestAdaptiveLearning:

    def test_record_experience(self):
        from openclaw.core.adaptive_learning import AdaptiveLearner
        learner = AdaptiveLearner(storage_dir="/tmp/test_learning")
        exp = learner.record_experience("context", "click", "success", 0.8)
        assert exp.reward == 0.8
        assert exp.action == "click"

    def test_experience_replay_sampling(self):
        from openclaw.core.adaptive_learning import ExperienceReplay, Experience
        replay = ExperienceReplay(max_size=100)
        for i in range(20):
            replay.add(Experience(id=str(i), context="ctx", action="act", result="res", reward=i/20))
        samples = replay.sample(5)
        assert len(samples) <= 5

    def test_strategy_manager(self):
        from openclaw.core.adaptive_learning import StrategyManager, Strategy
        sm = StrategyManager()
        sm.add_strategy(Strategy(id="s1", name="Test", pattern="search", actions=["search"], score=0.8))
        result = sm.select_strategy("search for files")
        assert result is not None
        assert result.name == "Test"

    def test_strategy_update(self):
        from openclaw.core.adaptive_learning import StrategyManager, Strategy
        sm = StrategyManager()
        sm.add_strategy(Strategy(id="s1", name="Test", pattern="test", actions=["act"], score=0.5))
        sm.update_strategy("s1", success=True)
        top = sm.get_top_strategies(1)
        assert top[0].uses == 1
        assert top[0].successes == 1

    def test_insights(self):
        from openclaw.core.adaptive_learning import AdaptiveLearner
        learner = AdaptiveLearner(storage_dir="/tmp/test_learning2")
        learner.record_experience("ctx", "act", "res", 0.9)
        insights = learner.get_insights()
        assert insights["total_experiences"] >= 1


# ============== Knowledge Graph Tests ==============

class TestKnowledgeGraph:

    def test_add_entity(self):
        from openclaw.core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(storage_path="/tmp/test_kg")
        entity = kg.add_entity("Python", "tool", {"version": "3.10"})
        assert entity.name == "Python"

    def test_add_relationship(self):
        from openclaw.core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(storage_path="/tmp/test_kg2")
        kg.add_entity("Python", "tool")
        kg.add_entity("Data Analysis", "task")
        rel = kg.add_relationship("Data Analysis", "Python", "requires", source_type="task", target_type="tool")
        assert rel.relation_type == "requires"

    def test_get_related(self):
        from openclaw.core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(storage_path="/tmp/test_kg3")
        kg.add_entity("A", "concept")
        kg.add_entity("B", "concept")
        kg.add_relationship("A", "B", "connects_to")
        related = kg.get_related("A")
        assert len(related) >= 1

    def test_find_path(self):
        from openclaw.core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(storage_path="/tmp/test_kg4")
        kg.add_entity("A", "concept")
        kg.add_entity("B", "concept")
        kg.add_entity("C", "concept")
        kg.add_relationship("A", "B", "connects")
        kg.add_relationship("B", "C", "connects")
        path = kg.find_path("A", "C")
        assert path is not None
        assert len(path) == 3

    def test_search(self):
        from openclaw.core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(storage_path="/tmp/test_kg5")
        kg.add_entity("Machine Learning", "concept")
        results = kg.search("machine")
        assert len(results) >= 1

    def test_stats(self):
        from openclaw.core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(storage_path="/tmp/test_kg6")
        kg.add_entity("X", "concept")
        stats = kg.get_stats()
        assert stats["total_entities"] >= 1


# ============== NL Automation Tests ==============

class TestNLAutomation:

    def test_parse_click(self):
        from openclaw.core.nl_automation import IntentParser, ActionIntent
        parser = IntentParser()
        result = parser.parse("click on the Settings button")
        assert result.intent == ActionIntent.CLICK
        assert "settings" in result.target.lower()

    def test_parse_type(self):
        from openclaw.core.nl_automation import IntentParser, ActionIntent
        parser = IntentParser()
        result = parser.parse("type 'hello world'")
        assert result.intent == ActionIntent.TYPE
        assert "hello world" in result.target

    def test_parse_wait(self):
        from openclaw.core.nl_automation import IntentParser, ActionIntent
        parser = IntentParser()
        result = parser.parse("wait 5 seconds")
        assert result.intent == ActionIntent.WAIT
        assert "5" in result.target

    def test_parse_multi(self):
        from openclaw.core.nl_automation import IntentParser
        parser = IntentParser()
        results = parser.parse_multi("click on start then type 'hello'")
        assert len(results) == 2

    def test_create_plan(self):
        from openclaw.core.nl_automation import NLAutomationEngine
        engine = NLAutomationEngine()
        plan = engine.parse_and_plan("type 'test' then wait 2 seconds")
        assert len(plan.steps) == 2

    def test_parse_screenshot(self):
        from openclaw.core.nl_automation import IntentParser, ActionIntent
        parser = IntentParser()
        result = parser.parse("take a screenshot")
        assert result.intent == ActionIntent.SCREENSHOT


# ============== Screen Understanding Tests ==============

class TestScreenUnderstanding:

    def test_element_detection_from_ocr(self):
        from openclaw.core.screen_understanding import ElementDetector
        detector = ElementDetector()
        ocr = [{"text": "Submit", "x": 100, "y": 200, "width": 80, "height": 30}]
        elements = detector.detect_from_ocr(ocr)
        assert len(elements) == 1
        assert elements[0].label == "Submit"

    def test_bounding_box(self):
        from openclaw.core.screen_understanding import BoundingBox
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        assert bb.center == (60, 45)
        assert bb.contains((50, 40)) is True
        assert bb.contains((200, 200)) is False

    def test_element_finder(self):
        from openclaw.core.screen_understanding import (
            ScreenUnderstanding, UIElement, UIElementType, BoundingBox, ScreenState
        )
        su = ScreenUnderstanding()
        ocr = [
            {"text": "Settings", "x": 100, "y": 50, "width": 80, "height": 30},
            {"text": "OK", "x": 200, "y": 300, "width": 60, "height": 30},
        ]
        su.analyze_from_ocr(ocr)
        results = su.find("Settings")
        assert len(results) >= 1

    def test_get_click_target(self):
        from openclaw.core.screen_understanding import ScreenUnderstanding
        su = ScreenUnderstanding()
        su.analyze_from_ocr([
            {"text": "Submit", "x": 100, "y": 200, "width": 80, "height": 30}
        ])
        target = su.get_click_target("Submit")
        assert target is not None
        assert isinstance(target, tuple)


# ============== Scheduler Tests ==============

class TestScheduler:

    def test_cron_parser(self):
        from openclaw.core.scheduler import CronParser
        from datetime import datetime
        dt = datetime(2026, 1, 15, 10, 30)
        assert CronParser.matches("30 10 * * *", dt) is True
        assert CronParser.matches("0 10 * * *", dt) is False

    def test_cron_every_5_min(self):
        from openclaw.core.scheduler import CronParser
        from datetime import datetime
        dt = datetime(2026, 1, 15, 10, 15)
        assert CronParser.matches("*/5 * * * *", dt) is True
        dt2 = datetime(2026, 1, 15, 10, 13)
        assert CronParser.matches("*/5 * * * *", dt2) is False

    def test_add_task(self):
        from openclaw.core.scheduler import TaskScheduler, ScheduleConfig, ScheduleType
        sched = TaskScheduler(storage_dir="/tmp/test_sched")
        tid = sched.add_task("test", lambda: None, ScheduleConfig(ScheduleType.INTERVAL, interval_seconds=60))
        assert tid is not None

    def test_task_is_due(self):
        from openclaw.core.scheduler import ScheduledTask, ScheduleConfig, ScheduleType
        task = ScheduledTask(
            id="t1", name="test",
            schedule=ScheduleConfig(ScheduleType.INTERVAL, interval_seconds=1),
            next_run=time.time() - 1
        )
        assert task.is_due is True

    def test_list_tasks(self):
        from openclaw.core.scheduler import TaskScheduler, ScheduleConfig, ScheduleType
        sched = TaskScheduler(storage_dir="/tmp/test_sched2")
        sched.add_task("t1", lambda: None, ScheduleConfig(ScheduleType.INTERVAL, interval_seconds=60))
        tasks = sched.list_tasks()
        assert len(tasks) == 1


# ============== Research Agent Tests ==============

class TestResearchAgent:

    def test_search_provider(self):
        from openclaw.core.research_agent import SearchProvider
        results = [{"url": "https://en.wikipedia.org/test", "title": "Test"}]
        sp = SearchProvider(search_fn=lambda q, n: results)
        sources = sp.search("test")
        assert len(sources) == 1
        assert sources[0].credibility > 0.7  # Wikipedia is high credibility

    def test_content_extractor(self):
        from openclaw.core.research_agent import ContentExtractor
        ce = ContentExtractor()
        content = "Python is a programming language. It is used for data science. The weather is nice today."
        facts = ce.extract_facts(content, "Python programming")
        assert len(facts) >= 1

    def test_research_no_search_fn(self):
        from openclaw.core.research_agent import ResearchAgent
        agent = ResearchAgent()
        report = agent.research("test query")
        assert report.query == "test query"


# ============== API Connectors Tests ==============

class TestAPIConnectors:

    def test_response_cache(self):
        from openclaw.core.api_connectors import ResponseCache
        cache = ResponseCache()
        cache.set("key1", {"data": "value"}, ttl=10)
        assert cache.get("key1") == {"data": "value"}

    def test_cache_expiry(self):
        from openclaw.core.api_connectors import ResponseCache
        cache = ResponseCache()
        cache.set("key1", "value", ttl=0.1)
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_rate_limiter(self):
        from openclaw.core.api_connectors import RateLimiter
        rl = RateLimiter(max_requests=2, window=60)
        assert rl.allow() is True
        assert rl.allow() is True
        assert rl.allow() is False

    def test_connector_no_http_fn(self):
        from openclaw.core.api_connectors import APIConnector, APIConfig, AuthType
        conn = APIConnector(APIConfig(name="test", base_url="https://example.com"))
        resp = conn.get("/endpoint")
        assert resp.success is False  # No http_fn configured

    def test_connector_templates(self):
        from openclaw.core.api_connectors import ConnectorTemplates
        gh = ConnectorTemplates.github(token="test")
        assert gh.config.name == "github"
        assert gh.config.base_url == "https://api.github.com"

    def test_connector_registry(self):
        from openclaw.core.api_connectors import ConnectorRegistry, ConnectorTemplates
        reg = ConnectorRegistry()
        reg.register(ConnectorTemplates.github())
        assert "github" in reg.list_connectors()


# ============== Sandbox Tests ==============

class TestSandbox:

    def test_python_execution(self):
        from openclaw.core.sandbox import Sandbox
        sb = Sandbox()
        result = sb.execute("print('hello')")
        assert result.success is True
        assert "hello" in result.stdout
        sb.cleanup()

    def test_python_math(self):
        from openclaw.core.sandbox import Sandbox
        sb = Sandbox()
        result = sb.execute("print(2 + 3)")
        assert result.success is True
        assert "5" in result.stdout
        sb.cleanup()

    def test_code_validation(self):
        from openclaw.core.sandbox import CodeValidator, Language, SandboxConfig
        validator = CodeValidator()
        result = validator.validate("import shutil; shutil.rmtree('/')", Language.PYTHON, SandboxConfig())
        assert result["safe"] is False

    def test_timeout(self):
        from openclaw.core.sandbox import Sandbox
        sb = Sandbox()
        result = sb.execute("import time; time.sleep(10)", timeout=1)
        assert result.success is False
        assert result.timed_out is True
        sb.cleanup()

    def test_bash_execution(self):
        from openclaw.core.sandbox import Sandbox, Language
        sb = Sandbox()
        result = sb.execute("echo 'hello bash'", Language.BASH)
        assert result.success is True
        assert "hello bash" in result.stdout
        sb.cleanup()

    def test_stats(self):
        from openclaw.core.sandbox import Sandbox
        sb = Sandbox()
        sb.execute("print(1)")
        stats = sb.get_stats()
        assert stats["total_executions"] == 1
        sb.cleanup()


# ============== Agent Swarm Tests ==============

class TestAgentSwarm:

    def test_add_agent(self):
        from openclaw.core.agent_swarm import AgentSwarm, AgentRole
        swarm = AgentSwarm()
        aid = swarm.add_agent("researcher", AgentRole.RESEARCHER)
        assert aid is not None

    def test_submit_task(self):
        from openclaw.core.agent_swarm import AgentSwarm, AgentRole
        swarm = AgentSwarm()
        swarm.add_agent("exec", AgentRole.EXECUTOR, handler=lambda desc: "done")
        task = swarm.submit_task("do something", decompose=False)
        assert task.status in ("assigned", "completed")

    def test_message_bus(self):
        from openclaw.core.agent_swarm import MessageBus, SwarmMessage, MessageType
        bus = MessageBus()
        bus.register_agent("agent1")
        bus.send(SwarmMessage(sender="system", receiver="agent1", message_type=MessageType.TASK, content="test"))
        messages = bus.receive("agent1")
        assert len(messages) == 1

    def test_task_decomposition(self):
        from openclaw.core.agent_swarm import TaskDecomposer, SwarmTask, AgentRole
        decomposer = TaskDecomposer()
        task = SwarmTask(description="Build a web scraper")
        subtasks = decomposer.decompose(task, [AgentRole.RESEARCHER, AgentRole.CODER, AgentRole.REVIEWER])
        assert len(subtasks) >= 2

    def test_swarm_status(self):
        from openclaw.core.agent_swarm import AgentSwarm, AgentRole
        swarm = AgentSwarm()
        swarm.add_agent("a1", AgentRole.EXECUTOR)
        status = swarm.get_swarm_status()
        assert status["agent_count"] == 1

    def test_broadcast(self):
        from openclaw.core.agent_swarm import MessageBus, SwarmMessage, MessageType
        bus = MessageBus()
        bus.register_agent("a1")
        bus.register_agent("a2")
        bus.send(SwarmMessage(sender="system", message_type=MessageType.BROADCAST, content="hello"))
        assert bus.pending_count("a1") == 1
        assert bus.pending_count("a2") == 1
