"""
Tests for Smart Context Module
"""

import time
import pytest
from unittest.mock import Mock


class TestContextCompressor:
    """Tests for context compression."""

    def _create_compressor(self, **kwargs):
        from openclaw.core.smart_context import ContextCompressor
        return ContextCompressor(**kwargs)

    def _make_entry(self, content, role="user", importance=0.5):
        from openclaw.core.smart_context import ContextEntry
        return ContextEntry(role=role, content=content, importance=importance)

    def test_no_compression_under_limit(self):
        cc = self._create_compressor(max_tokens=10000)
        entries = [self._make_entry(f"Short message {i}") for i in range(5)]
        result = cc.compress(entries)
        assert len(result) == 5

    def test_compresses_over_limit(self):
        cc = self._create_compressor(max_tokens=100, preserve_recent=3)
        entries = [self._make_entry(f"A longer message number {i} " * 10) for i in range(20)]
        result = cc.compress(entries)
        assert len(result) < len(entries)

    def test_preserves_recent_messages(self):
        cc = self._create_compressor(max_tokens=50, preserve_recent=3)
        entries = [self._make_entry(f"Message {i} " * 20) for i in range(10)]
        result = cc.compress(entries)
        # Last 3 should be intact
        recent_contents = [e.content for e in result[-3:]]
        original_recent = [e.content for e in entries[-3:]]
        assert recent_contents == original_recent

    def test_preserves_high_importance(self):
        cc = self._create_compressor(max_tokens=50, preserve_recent=2)
        entries = [
            self._make_entry("Important fact", importance=0.9),
            self._make_entry("Normal message " * 20, importance=0.3),
            self._make_entry("Another normal " * 20, importance=0.3),
            self._make_entry("Recent 1"),
            self._make_entry("Recent 2"),
        ]
        result = cc.compress(entries)
        contents = [e.content for e in result]
        assert "Important fact" in contents

    def test_compressed_entry_marked(self):
        cc = self._create_compressor(max_tokens=50, preserve_recent=2)
        entries = [self._make_entry(f"Message {i} " * 20) for i in range(10)]
        result = cc.compress(entries)
        compressed_entries = [e for e in result if e.compressed]
        assert len(compressed_entries) >= 1


class TestConversationChain:
    """Tests for conversation chaining."""

    def _create_chain(self, session_id="test_session"):
        from openclaw.core.smart_context import ConversationChain
        return ConversationChain(session_id)

    def test_link_conversation(self):
        chain = self._create_chain()
        chain.link_conversation("prev_1", "Discussed database schema")
        context = chain.get_context()
        assert "database schema" in context

    def test_add_key_fact(self):
        chain = self._create_chain()
        chain.add_key_fact("User prefers PostgreSQL")
        context = chain.get_context()
        assert "PostgreSQL" in context

    def test_no_duplicate_facts(self):
        chain = self._create_chain()
        chain.add_key_fact("Fact A")
        chain.add_key_fact("Fact A")
        facts = chain.get_all_facts()
        assert facts.count("Fact A") == 1

    def test_get_all_facts_includes_linked(self):
        chain = self._create_chain()
        chain.link_conversation("prev", "Summary", key_facts=["Linked fact"])
        chain.add_key_fact("Local fact")
        all_facts = chain.get_all_facts()
        assert "Linked fact" in all_facts
        assert "Local fact" in all_facts

    def test_relevance_ordering(self):
        chain = self._create_chain()
        chain.link_conversation("low", "Low relevance", relevance=0.2)
        chain.link_conversation("high", "High relevance", relevance=0.9)
        context = chain.get_context()
        # High relevance should appear before low
        high_pos = context.find("High relevance")
        low_pos = context.find("Low relevance")
        assert high_pos < low_pos

    def test_empty_context(self):
        chain = self._create_chain()
        assert chain.get_context() == ""


class TestToolAnalytics:
    """Tests for tool usage tracking."""

    def _create_analytics(self):
        from openclaw.core.smart_context import ToolAnalytics
        return ToolAnalytics()

    def test_record_success(self):
        ta = self._create_analytics()
        ta.record_call("search", duration=1.5, success=True)
        m = ta.get_metrics("search")
        assert m.total_calls == 1
        assert m.successes == 1
        assert m.success_rate == 1.0

    def test_record_failure(self):
        ta = self._create_analytics()
        ta.record_call("search", duration=0.1, success=False, error="Timeout")
        m = ta.get_metrics("search")
        assert m.failures == 1
        assert m.last_error == "Timeout"

    def test_success_rate(self):
        ta = self._create_analytics()
        ta.record_call("search", success=True)
        ta.record_call("search", success=True)
        ta.record_call("search", success=False)
        m = ta.get_metrics("search")
        assert abs(m.success_rate - 0.667) < 0.01

    def test_avg_duration(self):
        ta = self._create_analytics()
        ta.record_call("search", duration=1.0)
        ta.record_call("search", duration=3.0)
        m = ta.get_metrics("search")
        assert m.avg_duration == 2.0

    def test_health_status_healthy(self):
        ta = self._create_analytics()
        for _ in range(10):
            ta.record_call("search", success=True)
        assert ta.get_metrics("search").health_status == "healthy"

    def test_health_status_degraded(self):
        ta = self._create_analytics()
        for _ in range(7):
            ta.record_call("search", success=True)
        for _ in range(3):
            ta.record_call("search", success=False)
        assert ta.get_metrics("search").health_status == "degraded"

    def test_health_status_unhealthy(self):
        ta = self._create_analytics()
        for _ in range(2):
            ta.record_call("search", success=True)
        for _ in range(8):
            ta.record_call("search", success=False)
        assert ta.get_metrics("search").health_status == "unhealthy"

    def test_get_report(self):
        ta = self._create_analytics()
        ta.record_call("search", duration=1.0, success=True)
        ta.record_call("click", duration=0.5, success=True)
        report = ta.get_report()
        assert "search" in report["tools"]
        assert "click" in report["tools"]
        assert report["summary"]["total_calls"] == 2

    def test_recommendations(self):
        ta = self._create_analytics()
        for _ in range(10):
            ta.record_call("broken_tool", success=False, error="Always fails")
        recs = ta.get_recommendations()
        assert len(recs) > 0
        assert "broken_tool" in recs[0]

    def test_unknown_tool(self):
        ta = self._create_analytics()
        assert ta.get_metrics("nonexistent") is None


class TestAgentSelfHealer:
    """Tests for agent self-healing."""

    def _create_healer(self, **kwargs):
        from openclaw.core.smart_context import AgentSelfHealer
        return AgentSelfHealer(**kwargs)

    def test_record_success_resets_failures(self):
        healer = self._create_healer(failure_threshold=3)
        healer.record_failure("agent1")
        healer.record_failure("agent1")
        healer.record_success("agent1")
        status = healer.get_status()
        assert status["agent1"]["consecutive_failures"] == 0

    def test_auto_restart_on_threshold(self):
        restarted = []
        healer = self._create_healer(failure_threshold=2)
        healer.register_restart_handler("agent1", lambda: restarted.append(True))

        healer.record_failure("agent1")
        result = healer.record_failure("agent1")

        assert result is True
        assert len(restarted) == 1

    def test_max_restarts_limit(self):
        healer = self._create_healer(
            failure_threshold=1,
            max_restarts=2,
            cooldown_seconds=0
        )
        healer.register_restart_handler("agent1", lambda: None)

        for _ in range(3):
            healer.record_failure("agent1")
            healer.record_success("agent1")  # reset counter

        # After max restarts, should not restart
        healer.record_failure("agent1")
        result = healer.record_failure("agent1")
        assert result is False

    def test_cooldown_prevents_rapid_restarts(self):
        healer = self._create_healer(
            failure_threshold=1,
            cooldown_seconds=10.0
        )
        healer.register_restart_handler("agent1", lambda: None)

        # First restart works
        healer.record_failure("agent1")
        # Reset failures for next round
        healer.record_success("agent1")

        # Second restart should be on cooldown
        result = healer.record_failure("agent1")
        assert result is False

    def test_no_handler_no_restart(self):
        healer = self._create_healer(failure_threshold=1)
        # No handler registered
        result = healer.record_failure("agent1")
        assert result is False

    def test_get_unhealthy(self):
        healer = self._create_healer(failure_threshold=2)
        healer.record_failure("agent1")
        healer.record_failure("agent1")
        healer.record_failure("agent1")  # No handler, stays unhealthy

        unhealthy = healer.get_unhealthy()
        assert "agent1" in unhealthy

    def test_healthy_agent_not_in_unhealthy(self):
        healer = self._create_healer(failure_threshold=5)
        healer.record_failure("agent1")
        healer.record_success("agent1")

        unhealthy = healer.get_unhealthy()
        assert "agent1" not in unhealthy
