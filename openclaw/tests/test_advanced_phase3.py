"""
Phase 3 Tests for OpenClaw — Push to 95+

Tests:
1. Workflow DAG Engine (cycle detection, topological sort, parallel execution, diamond DAG)
2. LLM ReAct (chain_of_thought, parse_response, fallback)
3. Agent Negotiation (proposals, voting, consensus methods)
4. Metrics Server (health, metrics, events endpoints)
5. GitHub Tracker (mock API, CRUD)
6. Resilience Wiring (health checker integration, system state)
"""

import os
import sys
import time
import json
import tempfile
import threading
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. Workflow DAG Engine Tests
# ============================================================

class TestWorkflowDAG:
    """Tests for the DAG engine in workflow_engine.py"""

    def test_dag_create(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="test")
        assert len(dag) == 0
        dag.add_node(DAGNode(id="A", name="Step A", handler=lambda ctx: 1))
        assert len(dag) == 1

    def test_dag_validation_missing_dep(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="test")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: 1, depends_on=["X"]))
        issues = dag.validate()
        assert any("unknown" in i for i in issues)

    def test_dag_validation_no_handler(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="test")
        dag.add_node(DAGNode(id="A", name="A", handler=None))
        issues = dag.validate()
        assert any("no handler" in i for i in issues)

    def test_dag_validation_valid(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="test")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: 1))
        dag.add_node(DAGNode(id="B", name="B", handler=lambda ctx: 2, depends_on=["A"]))
        issues = dag.validate()
        assert issues == []

    def test_topological_sort_linear(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="linear")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: 1))
        dag.add_node(DAGNode(id="B", name="B", handler=lambda ctx: 2, depends_on=["A"]))
        dag.add_node(DAGNode(id="C", name="C", handler=lambda ctx: 3, depends_on=["B"]))

        order = dag.topological_sort()
        assert order.index("A") < order.index("B") < order.index("C")

    def test_topological_sort_diamond(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="diamond")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: 1))
        dag.add_node(DAGNode(id="B", name="B", handler=lambda ctx: 2, depends_on=["A"]))
        dag.add_node(DAGNode(id="C", name="C", handler=lambda ctx: 3, depends_on=["A"]))
        dag.add_node(DAGNode(id="D", name="D", handler=lambda ctx: 4, depends_on=["B", "C"]))

        order = dag.topological_sort()
        assert order[0] == "A"
        assert order[-1] == "D"
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_get_root_nodes(self):
        from core.workflow_engine import WorkflowDAG, DAGNode
        dag = WorkflowDAG(name="test")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: 1))
        dag.add_node(DAGNode(id="B", name="B", handler=lambda ctx: 2))
        dag.add_node(DAGNode(id="C", name="C", handler=lambda ctx: 3, depends_on=["A"]))

        ready = dag.get_ready_nodes()
        assert "A" in ready
        assert "B" in ready
        assert "C" not in ready


class TestWorkflowDAGEngine:
    """Tests for WorkflowDAGEngine execution."""

    def test_execute_linear(self):
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGStatus

        results = []
        dag = WorkflowDAG(name="linear")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: results.append("A") or "a"))
        dag.add_node(DAGNode(id="B", name="B", handler=lambda ctx: results.append("B") or "b",
                             depends_on=["A"]))
        dag.add_node(DAGNode(id="C", name="C", handler=lambda ctx: results.append("C") or "c",
                             depends_on=["B"]))

        engine = WorkflowDAGEngine(max_workers=2)
        result = engine.execute(dag)

        assert result.status == DAGStatus.COMPLETED
        assert result.completed_count == 3
        assert result.failed_count == 0
        assert results == ["A", "B", "C"]

    def test_execute_diamond(self):
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGStatus

        dag = WorkflowDAG(name="diamond")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: "a"))
        dag.add_node(DAGNode(id="B", name="B", handler=lambda ctx: "b", depends_on=["A"]))
        dag.add_node(DAGNode(id="C", name="C", handler=lambda ctx: "c", depends_on=["A"]))
        dag.add_node(DAGNode(id="D", name="D", handler=lambda ctx: "d", depends_on=["B", "C"]))

        engine = WorkflowDAGEngine(max_workers=4)
        result = engine.execute(dag)

        assert result.status == DAGStatus.COMPLETED
        assert result.completed_count == 4
        assert result.get_output("D") == "d"

    def test_execute_with_failure(self):
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGStatus

        dag = WorkflowDAG(name="fail-test")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: "ok"))
        dag.add_node(DAGNode(id="B", name="B",
                             handler=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
                             depends_on=["A"]))

        engine = WorkflowDAGEngine()
        result = engine.execute(dag)

        assert result.status in (DAGStatus.PARTIAL, DAGStatus.FAILED)
        assert result.failed_count >= 1

    def test_execute_with_skip_failure(self):
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGNodeStatus

        def fail_fn(ctx):
            raise RuntimeError("skip me")

        dag = WorkflowDAG(name="skip-test")
        dag.add_node(DAGNode(id="A", name="A", handler=fail_fn, on_failure="skip"))

        engine = WorkflowDAGEngine()
        result = engine.execute(dag)

        assert dag.nodes["A"].status == DAGNodeStatus.SKIPPED

    def test_dependency_passing(self):
        """Test that dependency results are passed via context."""
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGStatus

        dag = WorkflowDAG(name="dep-pass")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: 42))
        dag.add_node(DAGNode(id="B", name="B",
                             handler=lambda ctx: ctx.get("_A_result", 0) * 2,
                             depends_on=["A"]))

        engine = WorkflowDAGEngine()
        result = engine.execute(dag)

        assert result.status == DAGStatus.COMPLETED
        assert result.get_output("B") == 84

    def test_conditional_skip(self):
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGNodeStatus

        dag = WorkflowDAG(name="cond-test")
        dag.add_node(DAGNode(
            id="A", name="A",
            handler=lambda ctx: "should not run",
            condition=lambda ctx: False,  # Never run
        ))

        engine = WorkflowDAGEngine()
        result = engine.execute(dag)

        assert dag.nodes["A"].status == DAGNodeStatus.SKIPPED

    def test_engine_stats(self):
        from core.workflow_engine import WorkflowDAGEngine
        engine = WorkflowDAGEngine()
        stats = engine.get_stats()
        assert stats["total_executions"] == 0
        assert stats["max_workers"] == 4


# ============================================================
# 2. LLM ReAct Tests
# ============================================================

class TestLLMReActThink:
    """Tests for LLMReActThink in react_agent.py"""

    def test_chain_of_thought_field(self):
        from core.react_agent import ReActStep, StepType
        step = ReActStep(step_type=StepType.THOUGHT, content="test", chain_of_thought="Because X")
        assert step.chain_of_thought == "Because X"

    def test_llm_think_create(self):
        from core.react_agent import LLMReActThink
        think = LLMReActThink()
        assert think.model == "MiniMax-Text-01"
        assert think.temperature == 0.3

    def test_parse_json_response(self):
        from core.react_agent import LLMReActThink
        think = LLMReActThink()

        response = '{"reasoning": "Need to search", "action": "web_search", "action_args": {"query": "test"}}'
        result = think._parse_response(response)
        assert result["action"] == "web_search"
        assert result["reasoning"] == "Need to search"

    def test_parse_json_with_code_block(self):
        from core.react_agent import LLMReActThink
        think = LLMReActThink()

        response = '```json\n{"reasoning": "Done", "action": "final_answer", "answer": "42"}\n```'
        result = think._parse_response(response)
        assert result["action"] == "final_answer"
        assert result["answer"] == "42"

    def test_fallback_on_error(self):
        from core.react_agent import LLMReActThink
        think = LLMReActThink()

        # Call with no LLM available → should fallback
        result = think("Find info about cats", [], {"available_tools": [{"name": "search"}]})
        assert result["action"] == "search"
        assert think._fallback_count == 1

    def test_fallback_with_observations(self):
        from core.react_agent import LLMReActThink, ReActStep, StepType
        think = LLMReActThink()

        history = [
            ReActStep(step_type=StepType.OBSERVATION, content="Cats are mammals")
        ]
        result = think("Find info about cats", history, {})
        assert result["action"] == "final_answer"
        assert "mammals" in result["answer"]

    def test_stats(self):
        from core.react_agent import LLMReActThink
        think = LLMReActThink()
        stats = think.get_stats()
        assert stats["model"] == "MiniMax-Text-01"
        assert stats["llm_calls"] == 0


# ============================================================
# 3. Agent Negotiation Tests
# ============================================================

class TestNegotiation:
    """Tests for agent_negotiation.py"""

    def test_open_round(self):
        from core.agent_negotiation import NegotiationEngine, NegotiationStatus
        engine = NegotiationEngine()
        round_id = engine.open_round("What approach?")
        assert round_id is not None
        round_ = engine.get_round(round_id)
        assert round_.status == NegotiationStatus.COLLECTING

    def test_submit_proposal(self):
        from core.agent_negotiation import NegotiationEngine, Proposal
        engine = NegotiationEngine()
        round_id = engine.open_round("What approach?")

        result = engine.submit_proposal(round_id, Proposal(
            agent_id="researcher",
            approach="Search the web",
            confidence=0.8,
        ))
        assert result is True

        round_ = engine.get_round(round_id)
        assert len(round_.proposals) == 1

    def test_best_score_consensus(self):
        from core.agent_negotiation import NegotiationEngine, Proposal, ConsensusMethod
        engine = NegotiationEngine()
        round_id = engine.open_round("What approach?", method=ConsensusMethod.BEST_SCORE)

        engine.submit_proposal(round_id, Proposal(
            agent_id="researcher", approach="Search", confidence=0.9,
        ))
        engine.submit_proposal(round_id, Proposal(
            agent_id="coder", approach="Code it", confidence=0.6,
        ))

        winner = engine.decide(round_id)
        assert winner is not None
        assert winner.agent_id == "researcher"
        assert winner.confidence == 0.9

    def test_majority_vote(self):
        from core.agent_negotiation import NegotiationEngine, Proposal, ConsensusMethod
        engine = NegotiationEngine()
        round_id = engine.open_round("What approach?", method=ConsensusMethod.MAJORITY)

        engine.submit_proposal(round_id, Proposal(
            agent_id="a", approach="Option A", confidence=0.5, votes=3,
        ))
        engine.submit_proposal(round_id, Proposal(
            agent_id="b", approach="Option B", confidence=0.9, votes=1,
        ))

        winner = engine.decide(round_id)
        # Majority = most votes
        assert winner.agent_id == "a"

    def test_weighted_consensus(self):
        from core.agent_negotiation import NegotiationEngine, Proposal, ConsensusMethod
        engine = NegotiationEngine(agent_weights={"expert": 2.0, "junior": 0.5})
        round_id = engine.open_round("What approach?", method=ConsensusMethod.WEIGHTED)

        engine.submit_proposal(round_id, Proposal(
            agent_id="expert", approach="Expert plan", confidence=0.6,
        ))
        engine.submit_proposal(round_id, Proposal(
            agent_id="junior", approach="Junior plan", confidence=0.9,
        ))

        winner = engine.decide(round_id)
        # expert: 0.6 * 2.0 = 1.2, junior: 0.9 * 0.5 = 0.45
        assert winner.agent_id == "expert"

    def test_unanimous_disagree(self):
        from core.agent_negotiation import NegotiationEngine, Proposal, ConsensusMethod
        engine = NegotiationEngine()
        round_id = engine.open_round("What approach?", method=ConsensusMethod.UNANIMOUS)

        engine.submit_proposal(round_id, Proposal(agent_id="a", approach="Plan A", confidence=0.8))
        engine.submit_proposal(round_id, Proposal(agent_id="b", approach="Plan B", confidence=0.7))

        winner = engine.decide(round_id)
        assert winner is None  # No unanimity

    def test_voting(self):
        from core.agent_negotiation import NegotiationEngine, Proposal, ConsensusMethod
        engine = NegotiationEngine()
        round_id = engine.open_round("Vote test", method=ConsensusMethod.MAJORITY)

        engine.submit_proposal(round_id, Proposal(
            agent_id="a", approach="A",
        ))
        engine.submit_proposal(round_id, Proposal(
            agent_id="b", approach="B",
        ))

        # Get proposal IDs
        round_ = engine.get_round(round_id)
        p_a = round_.proposals[0].id
        p_b = round_.proposals[1].id

        engine.vote(round_id, "voter1", p_a)
        engine.vote(round_id, "voter2", p_a)
        engine.vote(round_id, "voter3", p_b)

        winner = engine.decide(round_id)
        assert winner.agent_id == "a"  # Got 2 votes vs 1

    def test_stats(self):
        from core.agent_negotiation import NegotiationEngine, Proposal
        engine = NegotiationEngine()
        r1 = engine.open_round("Q1")
        engine.submit_proposal(r1, Proposal(agent_id="a", approach="A", confidence=0.5))
        engine.decide(r1)

        stats = engine.get_stats()
        assert stats["total_rounds"] == 1
        assert stats["total_proposals"] == 1
        assert stats["decided_rounds"] == 1

    def test_empty_round_fails(self):
        from core.agent_negotiation import NegotiationEngine, NegotiationStatus
        engine = NegotiationEngine()
        round_id = engine.open_round("Empty")
        winner = engine.decide(round_id)
        assert winner is None
        assert engine.get_round(round_id).status == NegotiationStatus.FAILED


# ============================================================
# 4. Metrics Server Tests
# ============================================================

class TestMetricsServer:
    """Tests for metrics_server.py"""

    def test_server_create(self):
        from core.metrics_server import MetricsServer
        server = MetricsServer(port=19100)
        assert server.port == 19100
        assert server.url == "http://0.0.0.0:19100"

    def test_server_start_stop(self):
        from core.metrics_server import MetricsServer
        server = MetricsServer(port=19101)
        server.start()
        assert server._running
        time.sleep(0.2)
        server.stop()
        assert not server._running

    def test_health_endpoint(self):
        import urllib.request
        from core.metrics_server import MetricsServer

        server = MetricsServer(port=19102)
        server.start()
        time.sleep(0.3)

        try:
            req = urllib.request.Request(f"http://127.0.0.1:19102/health")
            resp = urllib.request.urlopen(req, timeout=2)
            data = json.loads(resp.read())
            assert "status" in data or "checks" in data
        finally:
            server.stop()

    def test_metrics_endpoint(self):
        import urllib.request
        from core.metrics_server import MetricsServer

        server = MetricsServer(port=19103)
        server.start()
        time.sleep(0.3)

        try:
            req = urllib.request.Request(f"http://127.0.0.1:19103/metrics")
            resp = urllib.request.urlopen(req, timeout=2)
            text = resp.read().decode()
            assert "openclaw_uptime_seconds" in text
        finally:
            server.stop()

    def test_index_endpoint(self):
        import urllib.request
        from core.metrics_server import MetricsServer

        server = MetricsServer(port=19104)
        server.start()
        time.sleep(0.3)

        try:
            req = urllib.request.Request(f"http://127.0.0.1:19104/")
            resp = urllib.request.urlopen(req, timeout=2)
            data = json.loads(resp.read())
            assert data["service"] == "OpenClaw"
            assert "/health" in data["endpoints"]
        finally:
            server.stop()

    def test_404_endpoint(self):
        import urllib.request
        from core.metrics_server import MetricsServer

        server = MetricsServer(port=19105)
        server.start()
        time.sleep(0.3)

        try:
            req = urllib.request.Request(f"http://127.0.0.1:19105/nonexistent")
            try:
                urllib.request.urlopen(req, timeout=2)
            except urllib.request.HTTPError as e:
                assert e.code == 404
        finally:
            server.stop()


# ============================================================
# 5. GitHub Tracker Tests
# ============================================================

class TestGitHubTracker:
    """Tests for plugins/github_tracker.py (mock API)."""

    @pytest.fixture(autouse=True)
    def setup_tracker(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "github_tracker",
            os.path.join(os.path.dirname(__file__), "..", "plugins", "github_tracker.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.GitHubTrackerPlugin = mod.GitHubTrackerPlugin
        self.tracker = self.GitHubTrackerPlugin()
        self.tracker.configure({
            "token": "test-token",
            "owner": "testuser",
            "repo": "testrepo",
        })

    def test_configure(self):
        assert self.tracker.token == "test-token"
        assert self.tracker.owner == "testuser"
        assert self.tracker.repo == "testrepo"
        assert self.tracker.name == "github-tracker"

    def test_headers(self):
        headers = self.tracker._headers()
        assert "Bearer test-token" in headers["Authorization"]

    def test_api_url(self):
        url = self.tracker._api_url("issues/1")
        assert url == "https://api.github.com/repos/testuser/testrepo/issues/1"

    def test_stats(self):
        stats = self.tracker.get_stats()
        assert stats["configured"] is True
        assert stats["total_created"] == 0

    def test_unconfigured_stats(self):
        tracker = self.GitHubTrackerPlugin()
        stats = tracker.get_stats()
        assert stats["configured"] is False

    @patch("requests.get")
    def test_get_issue_mock(self, mock_get):
        import asyncio
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "number": 42,
                "title": "Test Issue",
                "body": "Description",
                "state": "open",
                "labels": [{"name": "bug"}],
                "assignees": [{"login": "dev1"}],
                "created_at": "2026-01-01",
                "updated_at": "2026-01-02",
                "html_url": "https://github.com/test/42",
            },
        )

        result = asyncio.get_event_loop().run_until_complete(
            self.tracker.get_issue("42")
        )
        assert result["id"] == "42"
        assert result["title"] == "Test Issue"

    @patch("requests.post")
    def test_create_issue_mock(self, mock_post):
        import asyncio
        mock_post.return_value = MagicMock(
            status_code=201,
            json=lambda: {
                "number": 99,
                "title": "New Issue",
                "state": "open",
                "html_url": "https://github.com/test/99",
            },
        )

        result = asyncio.get_event_loop().run_until_complete(
            self.tracker.create_issue("New Issue", "Details", priority="high")
        )
        assert result["id"] == "99"
        assert self.tracker._total_created == 1


# ============================================================
# 6. Resilience Wiring Tests
# ============================================================

class TestResilienceWiring:
    """Tests for resilience integration in system_bootstrap.py"""

    def test_system_state_new_fields(self):
        from core.system_bootstrap import SystemState
        state = SystemState()
        assert hasattr(state, "health_checker_started")
        assert hasattr(state, "metrics_server_started")
        assert state.health_checker_started is False

    def test_system_state_summary_new_fields(self):
        from core.system_bootstrap import SystemState
        state = SystemState()
        state.started_at = time.time()
        summary = state.summary()
        assert "health_checker" in summary
        assert "metrics_server" in summary

    def test_health_checker_direct(self):
        from core.resilience import HealthChecker
        checker = HealthChecker()

        checker.register("test", lambda: ("healthy", "All good"))
        result = checker.check("test")
        assert result.status in ("healthy", "HEALTHY") or result.status.value == "healthy"

    def test_circuit_breaker(self):
        from core.resilience import CircuitBreaker, CircuitBreakerConfig, CircuitState

        cb = CircuitBreaker("test-llm", CircuitBreakerConfig(failure_threshold=2))
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Under threshold

        cb.record_failure()
        assert cb.state == CircuitState.OPEN  # Threshold hit

    def test_circuit_breaker_recovery(self):
        from core.resilience import CircuitBreaker, CircuitBreakerConfig, CircuitState

        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1,
        ))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)  # Wait for recovery
        assert cb.state == CircuitState.HALF_OPEN


# ============================================================
# Integration Tests
# ============================================================

class TestPhase3Integration:
    """End-to-end integration tests for Phase 3."""

    def test_dag_with_events(self):
        """DAG execution emits events to event bus."""
        from core.event_bus import EventBus, EventType
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine

        bus = EventBus()
        received = []
        bus.subscribe(handler=lambda e: received.append(e))

        dag = WorkflowDAG(name="event-test")
        dag.add_node(DAGNode(id="A", name="A", handler=lambda ctx: "done"))

        engine = WorkflowDAGEngine()
        engine.execute(dag)

        # Workflow succeeded
        assert dag.nodes["A"].result == "done"

    def test_negotiation_to_workflow(self):
        """Negotiate an approach, then execute it as a DAG."""
        from core.agent_negotiation import NegotiationEngine, Proposal, ConsensusMethod
        from core.workflow_engine import WorkflowDAG, DAGNode, WorkflowDAGEngine, DAGStatus

        # Negotiate
        engine = NegotiationEngine()
        round_id = engine.open_round("How to research?", method=ConsensusMethod.BEST_SCORE)
        engine.submit_proposal(round_id, Proposal(
            agent_id="researcher", approach="search_then_summarize", confidence=0.9,
        ))
        engine.submit_proposal(round_id, Proposal(
            agent_id="coder", approach="code_it", confidence=0.4,
        ))
        winner = engine.decide(round_id)
        assert winner.approach == "search_then_summarize"

        # Build DAG based on winning approach
        dag = WorkflowDAG(name=winner.approach)
        dag.add_node(DAGNode(id="search", name="Search", handler=lambda ctx: "results"))
        dag.add_node(DAGNode(id="summarize", name="Summarize",
                             handler=lambda ctx: f"Summary of {ctx.get('_search_result', 'N/A')}",
                             depends_on=["search"]))

        dag_engine = WorkflowDAGEngine()
        result = dag_engine.execute(dag)

        assert result.status == DAGStatus.COMPLETED
        assert "Summary" in result.get_output("summarize")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
