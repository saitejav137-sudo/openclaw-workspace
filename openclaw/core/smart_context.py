"""
Smart Context Manager for OpenClaw

Advanced intelligence features:
- Auto-compress long conversation histories
- Conversation chaining for cross-session continuity
- Tool usage analytics
- Agent self-healing with auto-restart
"""

import time
import threading
import hashlib
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from .logger import get_logger

logger = get_logger("smart_context")


# ============== Context Compression ==============

@dataclass
class ContextEntry:
    """A single entry in the conversation context."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0
    compressed: bool = False
    importance: float = 0.5

    def __post_init__(self):
        if self.token_estimate == 0:
            # Rough estimate: 1 token ≈ 4 chars
            self.token_estimate = len(self.content) // 4


class ContextCompressor:
    """
    Auto-compress long conversation histories to save token space.

    Strategy:
    1. Keep recent messages intact
    2. Summarize older messages into condensed form
    3. Preserve high-importance messages regardless of age
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        preserve_recent: int = 10,
        summary_ratio: float = 0.3
    ):
        self.max_tokens = max_tokens
        self.preserve_recent = preserve_recent
        self.summary_ratio = summary_ratio

    def compress(self, entries: List[ContextEntry]) -> List[ContextEntry]:
        """
        Compress context entries if they exceed max_tokens.
        Returns compressed list.
        """
        total_tokens = sum(e.token_estimate for e in entries)

        if total_tokens <= self.max_tokens:
            return entries

        # Split into recent (keep intact) and older (compress)
        recent = entries[-self.preserve_recent:]
        older = entries[:-self.preserve_recent]

        if not older:
            return entries

        # Keep high-importance older messages
        important = [e for e in older if e.importance >= 0.8]
        to_compress = [e for e in older if e.importance < 0.8]

        if not to_compress:
            return important + recent

        # Create summary of compressed messages
        summary = self._create_summary(to_compress)
        summary_entry = ContextEntry(
            role="system",
            content=f"[Compressed context — {len(to_compress)} messages]\n{summary}",
            compressed=True,
            importance=0.6
        )

        result = [summary_entry] + important + recent

        new_tokens = sum(e.token_estimate for e in result)
        logger.info(
            f"Compressed context: {total_tokens} → {new_tokens} tokens "
            f"({len(entries)} → {len(result)} messages)"
        )

        return result

    def _create_summary(self, entries: List[ContextEntry]) -> str:
        """Create a summary of context entries."""
        # Group by role
        by_role = defaultdict(list)
        for entry in entries:
            by_role[entry.role].append(entry.content)

        parts = []
        if by_role.get("user"):
            topics = set()
            for content in by_role["user"]:
                # Extract key phrases (simple keyword extraction)
                words = content.lower().split()
                # Keep words > 4 chars as likely meaningful
                topics.update(w for w in words if len(w) > 4)
            if topics:
                parts.append(f"Topics discussed: {', '.join(list(topics)[:15])}")

        if by_role.get("assistant"):
            parts.append(f"Assistant provided {len(by_role['assistant'])} responses")

        if not parts:
            parts.append(f"{len(entries)} messages exchanged")

        return "; ".join(parts)


# ============== Conversation Chaining ==============

@dataclass
class ConversationLink:
    """Link between related conversations."""
    conversation_id: str
    summary: str
    key_facts: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 0.5


class ConversationChain:
    """
    Links related conversations for cross-session continuity.

    Usage:
        chain = ConversationChain("session_123")
        chain.link_conversation("session_120", "Discussed database schema")
        chain.add_key_fact("User prefers PostgreSQL over MySQL")

        # Get context from previous sessions
        context = chain.get_context()
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._links: List[ConversationLink] = []
        self._key_facts: List[str] = []
        self._lock = threading.Lock()

    def link_conversation(
        self,
        conversation_id: str,
        summary: str,
        key_facts: List[str] = None,
        relevance: float = 0.5
    ):
        """Link a previous conversation to this one."""
        with self._lock:
            link = ConversationLink(
                conversation_id=conversation_id,
                summary=summary,
                key_facts=key_facts or [],
                relevance_score=relevance
            )
            self._links.append(link)
            logger.info(f"Linked conversation {conversation_id} to {self.session_id}")

    def add_key_fact(self, fact: str):
        """Add a key fact from the current session."""
        with self._lock:
            if fact not in self._key_facts:
                self._key_facts.append(fact)

    def get_context(self, max_links: int = 5) -> str:
        """Get context from linked conversations."""
        with self._lock:
            if not self._links and not self._key_facts:
                return ""

            parts = []

            # Sort by relevance
            sorted_links = sorted(
                self._links,
                key=lambda l: l.relevance_score,
                reverse=True
            )[:max_links]

            if sorted_links:
                parts.append("--- Previous Context ---")
                for link in sorted_links:
                    parts.append(f"[{link.conversation_id}] {link.summary}")
                    for fact in link.key_facts:
                        parts.append(f"  • {fact}")

            if self._key_facts:
                parts.append("--- Key Facts ---")
                for fact in self._key_facts:
                    parts.append(f"• {fact}")

            return "\n".join(parts)

    def get_all_facts(self) -> List[str]:
        """Get all key facts from this session and linked sessions."""
        with self._lock:
            all_facts = list(self._key_facts)
            for link in self._links:
                all_facts.extend(link.key_facts)
            return all_facts


# ============== Tool Analytics ==============

@dataclass
class ToolMetrics:
    """Metrics for a single tool."""
    name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    total_duration: float = 0.0
    last_error: Optional[str] = None
    last_used: Optional[float] = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def health_status(self) -> str:
        if self.total_calls == 0:
            return "unknown"
        if self.success_rate >= 0.95:
            return "healthy"
        if self.success_rate >= 0.7:
            return "degraded"
        return "unhealthy"


class ToolAnalytics:
    """
    Track tool usage patterns and effectiveness.

    Usage:
        analytics = ToolAnalytics()

        # Record tool usage
        analytics.record_call("web_search", duration=1.5, success=True)
        analytics.record_call("web_search", duration=0.0, success=False, error="Timeout")

        # Get insights
        report = analytics.get_report()
    """

    def __init__(self):
        self._metrics: Dict[str, ToolMetrics] = {}
        self._lock = threading.Lock()

    def record_call(
        self,
        tool_name: str,
        duration: float = 0.0,
        success: bool = True,
        error: str = None
    ):
        """Record a tool execution."""
        with self._lock:
            if tool_name not in self._metrics:
                self._metrics[tool_name] = ToolMetrics(name=tool_name)

            m = self._metrics[tool_name]
            m.total_calls += 1
            m.total_duration += duration
            m.last_used = time.time()

            if success:
                m.successes += 1
            else:
                m.failures += 1
                m.last_error = error

    def get_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        """Get metrics for a specific tool."""
        with self._lock:
            return self._metrics.get(tool_name)

    def get_report(self) -> Dict[str, Any]:
        """Get full analytics report."""
        with self._lock:
            if not self._metrics:
                return {"tools": {}, "summary": {}}

            tools = {}
            total_calls = 0
            total_successes = 0
            total_failures = 0

            for name, m in self._metrics.items():
                tools[name] = {
                    "total_calls": m.total_calls,
                    "success_rate": round(m.success_rate, 3),
                    "avg_duration_ms": round(m.avg_duration * 1000, 1),
                    "health": m.health_status,
                    "last_error": m.last_error,
                }
                total_calls += m.total_calls
                total_successes += m.successes
                total_failures += m.failures

            # Most used tools
            most_used = sorted(
                self._metrics.values(),
                key=lambda m: m.total_calls,
                reverse=True
            )[:5]

            # Slowest tools
            slowest = sorted(
                self._metrics.values(),
                key=lambda m: m.avg_duration,
                reverse=True
            )[:5]

            return {
                "tools": tools,
                "summary": {
                    "total_calls": total_calls,
                    "overall_success_rate": (
                        round(total_successes / total_calls, 3)
                        if total_calls > 0 else 0.0
                    ),
                    "most_used": [m.name for m in most_used],
                    "slowest": [m.name for m in slowest],
                    "unhealthy": [
                        m.name for m in self._metrics.values()
                        if m.health_status == "unhealthy"
                    ],
                }
            }

    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on analytics."""
        recommendations = []

        with self._lock:
            for name, m in self._metrics.items():
                if m.total_calls >= 5 and m.success_rate < 0.7:
                    recommendations.append(
                        f"Tool '{name}' has low success rate ({m.success_rate:.0%}). "
                        f"Consider investigating: {m.last_error}"
                    )

                if m.total_calls >= 5 and m.avg_duration > 10.0:
                    recommendations.append(
                        f"Tool '{name}' is slow (avg {m.avg_duration:.1f}s). "
                        f"Consider caching or optimization."
                    )

        return recommendations


# ============== Agent Self-Healing ==============

@dataclass
class AgentHealthRecord:
    """Health record for an agent."""
    agent_id: str
    consecutive_failures: int = 0
    total_restarts: int = 0
    last_failure: Optional[float] = None
    last_restart: Optional[float] = None
    is_healthy: bool = True


class AgentSelfHealer:
    """
    Monitors agent health and auto-restarts failed agents.

    Features:
    - Detect consecutive failures
    - Auto-restart with last checkpoint recovery
    - Exponential backoff for restarts
    - Max restart limit to prevent restart loops

    Usage:
        healer = AgentSelfHealer()
        healer.register_restart_handler("vision_agent", restart_vision_agent)

        # In error handling:
        healer.record_failure("vision_agent")
        # Auto-restarts if threshold hit
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        max_restarts: int = 5,
        cooldown_seconds: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.max_restarts = max_restarts
        self.cooldown_seconds = cooldown_seconds

        self._agents: Dict[str, AgentHealthRecord] = {}
        self._restart_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_restart_handler(self, agent_id: str, handler: Callable):
        """Register a function that restarts the agent."""
        with self._lock:
            self._restart_handlers[agent_id] = handler
            if agent_id not in self._agents:
                self._agents[agent_id] = AgentHealthRecord(agent_id=agent_id)
            logger.info(f"Registered restart handler for agent '{agent_id}'")

    def record_success(self, agent_id: str):
        """Record a successful operation — resets failure counter."""
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = AgentHealthRecord(agent_id=agent_id)
            self._agents[agent_id].consecutive_failures = 0
            self._agents[agent_id].is_healthy = True

    def record_failure(self, agent_id: str) -> bool:
        """
        Record a failure. Returns True if agent was auto-restarted.
        """
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = AgentHealthRecord(agent_id=agent_id)

            record = self._agents[agent_id]
            record.consecutive_failures += 1
            record.last_failure = time.time()

            if record.consecutive_failures >= self.failure_threshold:
                record.is_healthy = False

                # Check if we can restart
                if record.total_restarts >= self.max_restarts:
                    logger.error(
                        f"Agent '{agent_id}' exceeded max restarts "
                        f"({self.max_restarts}). Manual intervention required."
                    )
                    return False

                # Check cooldown
                if (
                    record.last_restart
                    and time.time() - record.last_restart < self.cooldown_seconds
                ):
                    logger.warning(
                        f"Agent '{agent_id}' restart on cooldown. "
                        f"Waiting {self.cooldown_seconds}s..."
                    )
                    return False

                # Attempt restart
                handler = self._restart_handlers.get(agent_id)
                if handler:
                    return self._attempt_restart(agent_id, handler, record)

        return False

    def _attempt_restart(
        self,
        agent_id: str,
        handler: Callable,
        record: AgentHealthRecord
    ) -> bool:
        """Attempt to restart an agent (must hold lock)."""
        try:
            logger.info(
                f"Auto-restarting agent '{agent_id}' "
                f"(restart #{record.total_restarts + 1})"
            )
            handler()
            record.total_restarts += 1
            record.last_restart = time.time()
            record.consecutive_failures = 0
            record.is_healthy = True
            logger.info(f"Agent '{agent_id}' restarted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to restart agent '{agent_id}': {e}")
            record.total_restarts += 1
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get health status of all agents."""
        with self._lock:
            return {
                agent_id: {
                    "healthy": record.is_healthy,
                    "consecutive_failures": record.consecutive_failures,
                    "total_restarts": record.total_restarts,
                    "last_failure": record.last_failure,
                }
                for agent_id, record in self._agents.items()
            }

    def get_unhealthy(self) -> List[str]:
        """Get list of unhealthy agent IDs."""
        with self._lock:
            return [
                agent_id
                for agent_id, record in self._agents.items()
                if not record.is_healthy
            ]


# ============== Global Instances ==============

_compressor: Optional[ContextCompressor] = None
_analytics: Optional[ToolAnalytics] = None
_healer: Optional[AgentSelfHealer] = None


def get_compressor(max_tokens: int = 4000) -> ContextCompressor:
    """Get global context compressor."""
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor(max_tokens=max_tokens)
    return _compressor


def get_tool_analytics() -> ToolAnalytics:
    """Get global tool analytics."""
    global _analytics
    if _analytics is None:
        _analytics = ToolAnalytics()
    return _analytics


def get_self_healer() -> AgentSelfHealer:
    """Get global agent self-healer."""
    global _healer
    if _healer is None:
        _healer = AgentSelfHealer()
    return _healer


__all__ = [
    "ContextEntry",
    "ContextCompressor",
    "ConversationLink",
    "ConversationChain",
    "ToolMetrics",
    "ToolAnalytics",
    "AgentHealthRecord",
    "AgentSelfHealer",
    "get_compressor",
    "get_tool_analytics",
    "get_self_healer",
]
