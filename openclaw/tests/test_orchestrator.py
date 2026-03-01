"""
Tests for AgentOrchestrator
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAgentRegistry:
    """Tests for agent registration and discovery."""

    def _create_registry(self):
        from openclaw.core.orchestrator import AgentRegistry
        return AgentRegistry()

    def test_register_agent(self):
        reg = self._create_registry()
        agent = reg.register("a1", "Vision Agent", ["vision", "ocr"])
        assert agent.agent_id == "a1"
        assert agent.name == "Vision Agent"
        assert "vision" in agent.capabilities
        assert "ocr" in agent.capabilities

    def test_unregister_agent(self):
        reg = self._create_registry()
        reg.register("a1", "Agent1", ["test"])
        reg.unregister("a1")
        assert reg.get_best_agent(["test"]) is None

    def test_find_agents_by_capability(self):
        reg = self._create_registry()
        reg.register("a1", "Vision", ["vision", "ocr"])
        reg.register("a2", "Browser", ["browser", "navigation"])
        reg.register("a3", "Multi", ["vision", "browser"])

        vision_agents = reg.find_agents(["vision"])
        assert len(vision_agents) == 2  # a1 and a3
        agent_ids = [a.agent_id for a in vision_agents]
        assert "a1" in agent_ids
        assert "a3" in agent_ids

    def test_find_agents_multi_capability(self):
        reg = self._create_registry()
        reg.register("a1", "Vision", ["vision"])
        reg.register("a2", "Multi", ["vision", "ocr"])

        # Only a2 has both
        agents = reg.find_agents(["vision", "ocr"])
        assert len(agents) == 1
        assert agents[0].agent_id == "a2"

    def test_get_best_agent(self):
        reg = self._create_registry()
        reg.register("a1", "Agent1", ["test"], max_concurrent=1)
        agent = reg.get_best_agent(["test"])
        assert agent is not None
        assert agent.agent_id == "a1"

    def test_get_best_agent_none(self):
        reg = self._create_registry()
        assert reg.get_best_agent(["nonexistent"]) is None

    def test_heartbeat(self):
        reg = self._create_registry()
        agent = reg.register("a1", "Agent1", ["test"])
        old_heartbeat = agent.last_heartbeat
        time.sleep(0.01)
        reg.heartbeat("a1")
        assert agent.last_heartbeat > old_heartbeat

    def test_record_completion(self):
        reg = self._create_registry()
        agent = reg.register("a1", "Agent1", ["test"])
        agent.active_tasks = 1

        reg.record_completion("a1", duration=5.0, success=True)
        assert agent.active_tasks == 0
        assert agent.avg_duration > 0

    def test_available_agent_preferred(self):
        reg = self._create_registry()
        a1 = reg.register("a1", "Busy", ["test"], max_concurrent=1)
        a1.active_tasks = 1  # busy
        reg.register("a2", "Free", ["test"], max_concurrent=1)

        agent = reg.get_best_agent(["test"])
        assert agent.agent_id == "a2"

    def test_get_stats(self):
        reg = self._create_registry()
        reg.register("a1", "Agent1", ["test"])
        stats = reg.get_stats()
        assert stats["total_agents"] == 1
        assert stats["healthy"] == 1
        assert "a1" in stats["agents"]


class TestTaskDecomposer:
    """Tests for task decomposition."""

    def _create_decomposer(self):
        from openclaw.core.orchestrator import TaskDecomposer
        return TaskDecomposer()

    def test_decompose_with_template(self):
        d = self._create_decomposer()
        plan = d.decompose("test task", template="research")
        assert len(plan.subtasks) == 3
        names = [st.name for st in plan.subtasks]
        assert "search" in names
        assert "analyze" in names
        assert "summarize" in names

    def test_decompose_keyword_match(self):
        d = self._create_decomposer()
        plan = d.decompose("research AI best practices")
        assert len(plan.subtasks) == 3  # matched "research" template

    def test_decompose_vision_match(self):
        d = self._create_decomposer()
        plan = d.decompose("take a screenshot and detect text")
        assert len(plan.subtasks) == 4  # matched "vision_pipeline"

    def test_decompose_fallback(self):
        d = self._create_decomposer()
        plan = d.decompose("do something random")
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].name == "execute"

    def test_custom_template(self):
        d = self._create_decomposer()
        d.register_template("custom", [
            {"name": "step1", "capabilities": ["cap1"]},
            {"name": "step2", "capabilities": ["cap2"]},
        ])
        plan = d.decompose("test", template="custom")
        assert len(plan.subtasks) == 2

    def test_dependencies_chain(self):
        d = self._create_decomposer()
        plan = d.decompose("research task", template="research")
        # Each step depends on the previous
        assert len(plan.subtasks[0].dependencies) == 0
        assert len(plan.subtasks[1].dependencies) == 1
        assert len(plan.subtasks[2].dependencies) == 1


class TestTaskPlan:
    """Tests for TaskPlan data model."""

    def test_is_complete(self):
        from openclaw.core.orchestrator import TaskPlan, SubTask, TaskStatus
        plan = TaskPlan(subtasks=[
            SubTask(name="t1", status=TaskStatus.COMPLETED),
            SubTask(name="t2", status=TaskStatus.COMPLETED),
        ])
        assert plan.is_complete is True

    def test_is_not_complete(self):
        from openclaw.core.orchestrator import TaskPlan, SubTask, TaskStatus
        plan = TaskPlan(subtasks=[
            SubTask(name="t1", status=TaskStatus.COMPLETED),
            SubTask(name="t2", status=TaskStatus.PENDING),
        ])
        assert plan.is_complete is False

    def test_has_failures(self):
        from openclaw.core.orchestrator import TaskPlan, SubTask, TaskStatus
        plan = TaskPlan(subtasks=[
            SubTask(name="t1", status=TaskStatus.FAILED),
        ])
        assert plan.has_failures is True

    def test_progress(self):
        from openclaw.core.orchestrator import TaskPlan, SubTask, TaskStatus
        plan = TaskPlan(subtasks=[
            SubTask(name="t1", status=TaskStatus.COMPLETED),
            SubTask(name="t2", status=TaskStatus.PENDING),
        ])
        assert plan.progress == 0.5


class TestResultAggregator:
    """Tests for result aggregation."""

    def test_merge_results(self):
        from openclaw.core.orchestrator import (
            ResultAggregator, TaskPlan, SubTask, TaskStatus
        )
        agg = ResultAggregator()
        plan = TaskPlan(subtasks=[
            SubTask(name="search", status=TaskStatus.COMPLETED, result="Found: AI best practices"),
            SubTask(name="summarize", status=TaskStatus.COMPLETED, result="Summary: AI is great"),
        ])
        result = agg.aggregate(plan)
        assert result.success is True
        assert "search" in result.result
        assert "Summary" in result.result

    def test_aggregate_with_errors(self):
        from openclaw.core.orchestrator import (
            ResultAggregator, TaskPlan, SubTask, TaskStatus
        )
        agg = ResultAggregator()
        plan = TaskPlan(subtasks=[
            SubTask(name="ok", status=TaskStatus.COMPLETED, result="good"),
            SubTask(name="bad", status=TaskStatus.FAILED, error="timeout"),
        ])
        result = agg.aggregate(plan)
        assert result.success is False
        assert len(result.errors) == 1


class TestAgentOrchestrator:
    """Tests for the full orchestrator."""

    def _create_orchestrator(self):
        from openclaw.core.orchestrator import AgentOrchestrator
        return AgentOrchestrator()

    def test_simple_execution(self):
        orch = self._create_orchestrator()

        # Register agent
        orch.registry.register("a1", "General", ["general"])

        # Register handler
        def handler(subtask, context):
            return "done!"
        orch.register_handler("general", handler)

        result = orch.execute("do something")
        assert result.success is True

    def test_sequential_execution(self):
        from openclaw.core.orchestrator import ExecutionStrategy
        orch = self._create_orchestrator()

        orch.registry.register("a1", "Searcher", ["web_search"])
        orch.registry.register("a2", "Analyzer", ["analysis"])
        orch.registry.register("a3", "Summarizer", ["summarization"])

        call_order = []

        def search_handler(st, ctx):
            call_order.append("search")
            return "search results"

        def analyze_handler(st, ctx):
            call_order.append("analyze")
            return "analyzed"

        def summarize_handler(st, ctx):
            call_order.append("summarize")
            return "summary"

        orch.register_handler("web_search", search_handler)
        orch.register_handler("analysis", analyze_handler)
        orch.register_handler("summarization", summarize_handler)

        result = orch.execute(
            "research something",
            strategy=ExecutionStrategy.SEQUENTIAL,
            template="research"
        )

        assert result.success is True
        assert call_order == ["search", "analyze", "summarize"]

    def test_parallel_execution(self):
        from openclaw.core.orchestrator import ExecutionStrategy
        orch = self._create_orchestrator()

        orch.registry.register("a1", "Worker", ["general"])

        results_collected = []

        def handler(subtask, context):
            results_collected.append(subtask.name)
            return f"result_{subtask.name}"

        orch.register_handler("general", handler)

        # Custom template with independent tasks
        orch.decomposer.register_template("parallel_test", [
            {"name": "task_a", "capabilities": ["general"]},
            {"name": "task_b", "capabilities": ["general"]},
        ])
        # Remove dependencies for parallel
        plan_test = orch.decomposer.decompose("test", template="parallel_test")
        for st in plan_test.subtasks:
            st.dependencies = []  # Make all independent

        result = orch.execute(
            "test parallel",
            strategy=ExecutionStrategy.PARALLEL,
            template="parallel_test"
        )

        assert result.success is True
        assert len(results_collected) == 2

    def test_retry_on_failure(self):
        orch = self._create_orchestrator()
        orch.registry.register("a1", "General", ["general"])

        call_count = 0

        def flaky_handler(subtask, context):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        orch.register_handler("general", flaky_handler)

        result = orch.execute("test retry")
        assert result.success is True
        assert call_count == 2  # Failed once, succeeded on retry

    def test_hooks_called(self):
        orch = self._create_orchestrator()
        orch.registry.register("a1", "General", ["general"])

        def handler(st, ctx):
            return "ok"
        orch.register_handler("general", handler)

        events = []
        orch.on("on_task_start", lambda **kw: events.append("start"))
        orch.on("on_task_complete", lambda **kw: events.append("complete"))

        orch.execute("test hooks")
        assert "start" in events
        assert "complete" in events

    def test_cancel_plan(self):
        from openclaw.core.orchestrator import TaskStatus
        orch = self._create_orchestrator()

        # Create a plan manually
        from openclaw.core.orchestrator import TaskPlan, SubTask
        plan = TaskPlan(
            original_task="test",
            subtasks=[SubTask(name="t1", status=TaskStatus.RUNNING)]
        )
        orch._active_plans[plan.id] = plan

        orch.cancel_plan(plan.id)
        assert plan.status == TaskStatus.CANCELLED
        assert plan.subtasks[0].status == TaskStatus.CANCELLED

    def test_get_stats(self):
        orch = self._create_orchestrator()
        orch.registry.register("a1", "Agent1", ["test"])
        stats = orch.get_stats()
        assert "active_plans" in stats
        assert "registry" in stats
        assert stats["registry"]["total_agents"] == 1

    def test_shutdown(self):
        orch = self._create_orchestrator()
        orch.shutdown()  # Should not raise
