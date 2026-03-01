"""
Advanced Smart Context Manager for OpenClaw

Features:
- Hierarchical memory system (working, episodic, semantic, procedural)
- Token budget management with smart allocation
- Importance-based retention with time decay
- Semantic clustering and deduplication
- LRU caching for compressed contexts
- RAG integration for retrieval
- Streaming context for long conversations

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              SmartContextManager                │
    ├─────────────────────────────────────────────────┤
    │  Working Memory (current session)               │
    │    - Recent messages, high fidelity            │
    │    - Token budget: 40%                         │
    ├─────────────────────────────────────────────────┤
    │  Episodic Memory (past sessions)               │
    │    - Summarized conversations                  │
    │    - Token budget: 30%                         │
    ├─────────────────────────────────────────────────┤
    │  Semantic Memory (learned facts)               │
    │    - Extracted entities, preferences           │
    │    - Token budget: 20%                         │
    ├─────────────────────────────────────────────────┤
    │  Procedural Memory (patterns)                   │
    │    - Learned workflows, tool sequences         │
    │    - Token budget: 10%                         │
    └─────────────────────────────────────────────────┘
"""

import time
import threading
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from abc import ABC, abstractmethod

from .logger import get_logger

logger = get_logger("smart_context")


# ============== Enums and Constants ==============

class MemoryType(Enum):
    """Types of memory in the hierarchy."""
    WORKING = "working"       # Current conversation
    EPISODIC = "episodic"     # Past sessions
    SEMANTIC = "semantic"     # Learned facts
    PROCEDURAL = "procedural" # Learned patterns


class CompressionStrategy(Enum):
    """Strategies for context compression."""
    FULL = "full"           # Keep everything (under budget)
    SUMMARIZE = "summarize" # Create summary
    EXTRACT = "extract"     # Extract key points
    PRUNE = "prune"         # Remove low-importance
    HYBRID = "hybrid"       # Combination


# Token estimation constants
CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE_HEADER = 10


# ============== Data Classes ==============

@dataclass
class ContextMessage:
    """A single message in the context."""
    role: str              # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    importance: float = 0.5  # 0-1 scale
    memory_type: MemoryType = MemoryType.WORKING
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = ""
    compressed: bool = False  # Backwards compatibility

    @property
    def token_estimate(self) -> int:
        """Backwards compatible alias for token_count."""
        return self.token_count

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = (len(self.content) // CHARS_PER_TOKEN) + TOKENS_PER_MESSAGE_HEADER
        if not self.message_id:
            self.message_id = hashlib.md5(
                f"{self.content}{self.timestamp}".encode()
            ).hexdigest()[:8]


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""
    fact: str
    category: str          # "preference", "entity", "action", "constraint"
    confidence: float       # 0-1
    source_message_id: str
    timestamp: float = field(default_factory=time.time)
    last_referenced: float = field(default_factory=time.time)
    reference_count: int = 0


@dataclass
class LearnedPattern:
    """A learned procedural pattern."""
    pattern: str           # Description of the pattern
    sequence: List[str]    # Tool/action sequence
    frequency: int = 1
    last_used: float = field(default_factory=time.time)
    success_rate: float = 1.0
    context: str = ""      # When this pattern applies


@dataclass
class MemoryBudget:
    """Token budget allocation for memory types."""
    working_tokens: int = 2000    # ~40%
    episodic_tokens: int = 1500    # ~30%
    semantic_tokens: int = 1000    # ~20%
    procedural_tokens: int = 500   # ~10%
    total_budget: int = 5000

    def __post_init__(self):
        self.total_budget = (
            self.working_tokens + self.episodic_tokens +
            self.semantic_tokens + self.procedural_tokens
        )

    def get_limit(self, memory_type: MemoryType) -> int:
        """Get token limit for a memory type."""
        return {
            MemoryType.WORKING: self.working_tokens,
            MemoryType.EPISODIC: self.episodic_tokens,
            MemoryType.SEMANTIC: self.semantic_tokens,
            MemoryType.PROCEDURAL: self.procedural_tokens,
        }[memory_type]


# ============== Importance Scorer ==============

class ImportanceScorer:
    """
    Scores message importance based on multiple factors.
    """

    # Keywords that indicate high importance
    HIGH_IMPORTANCE_KEYWORDS = {
        "error", "fail", "exception", "bug", "issue", "problem",
        "important", "critical", "urgent", "must", "required",
        "config", "configuration", "setup", "install", "deploy",
        "api", "key", "secret", "password", "token",
        "fix", "solution", "worked", "success", "done"
    }

    LOW_IMPORTANCE_KEYWORDS = {
        "okay", "ok", "thanks", "thank you", "sure", "yes", "no",
        "hello", "hi", "hey", "bye", "goodbye"
    }

    def __init__(self, decay_rate: float = 0.01):
        """
        Args:
            decay_rate: How quickly old messages lose importance (per minute)
        """
        self.decay_rate = decay_rate

    def score(self, message: ContextMessage) -> float:
        """Calculate importance score (0-1)."""
        score = 0.5  # Base score

        # Factor 1: Keyword analysis
        content_lower = message.content.lower()
        words = set(content_lower.split())

        high_keywords = words.intersection(self.HIGH_IMPORTANCE_KEYWORDS)
        low_keywords = words.intersection(self.LOW_IMPORTANCE_KEYWORDS)

        score += len(high_keywords) * 0.1
        score -= len(low_keywords) * 0.05

        # Factor 2: Message length (longer = often more substantive)
        if message.token_count > 200:
            score += 0.1
        elif message.token_count < 30:
            score -= 0.1

        # Factor 3: Role analysis
        if message.role == "system":
            score += 0.2  # System messages are important
        elif message.role == "tool":
            # Tool results are important if they contain errors
            if "error" in content_lower or "fail" in content_lower:
                score += 0.3
            else:
                score += 0.1

        # Factor 4: Time decay
        age_minutes = (time.time() - message.timestamp) / 60
        decay = self.decay_rate * age_minutes
        score -= decay

        # Factor 5: Metadata signals
        if message.metadata.get("has_code"):
            score += 0.15
        if message.metadata.get("has_error"):
            score += 0.25
        if message.metadata.get("user_rating", 0) > 0:
            score += message.metadata["user_rating"] * 0.1

        return max(0.0, min(1.0, score))


# ============== Fact Extractor ==============

class FactExtractor:
    """
    Extracts structured facts from conversation messages.
    """

    # Patterns for fact extraction
    ENTITY_PATTERNS = {
        "preference": [
            r"prefer[s]?\s+(\w+)",
            r"like[s]?\s+(\w+)",
            r"instead of\s+(\w+)",
            r"rather than\s+(\w+)",
        ],
        "constraint": [
            r"must\s+(\w+)",
            r"need[s]?\s+(\w+)",
            r"require[s]?\s+(\w+)",
            r"can't\s+(\w+)",
            r"cannot\s+(\w+)",
        ],
        "action": [
            r"always\s+(\w+)",
            r"never\s+(\w+)",
            r"usually\s+(\w+)",
            r"typically\s+(\w+)",
        ],
    }

    def __init__(self):
        import re
        self._patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.ENTITY_PATTERNS.items()
        }

    def extract(self, messages: List[ContextMessage]) -> List[ExtractedFact]:
        """Extract facts from a list of messages."""
        facts = []

        for msg in messages:
            if msg.role not in ("user", "assistant"):
                continue

            content = msg.content

            # Check each category
            for category, patterns in self._patterns.items():
                for pattern in patterns:
                    match = pattern.search(content)
                    if match:
                        fact_text = match.group(0)
                        # Determine confidence based on specificity
                        confidence = 0.7 if match.groups() else 0.5

                        facts.append(ExtractedFact(
                            fact=fact_text,
                            category=category,
                            confidence=confidence,
                            source_message_id=msg.message_id,
                            timestamp=msg.timestamp
                        ))

        # Deduplicate similar facts
        return self._deduplicate(facts)

    def _deduplicate(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Remove duplicate or very similar facts."""
        if not facts:
            return []

        unique = []
        seen_texts: Set[str] = set()

        for fact in facts:
            # Normalize for comparison
            normalized = fact.fact.lower().strip()
            # Check if too similar to existing
            is_duplicate = any(
                normalized in seen or seen in normalized
                for seen in seen_texts
            )

            if not is_duplicate:
                unique.append(fact)
                seen_texts.add(normalized)

        return unique


# ============== Pattern Learner ==============

class PatternLearner:
    """
    Learns procedural patterns from tool usage sequences.
    """

    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency
        self._sequences: Dict[str, List[str]] = defaultdict(list)
        self._pattern_stats: Dict[str, Dict[str, Any]] = {}

    def record_sequence(self, sequence: List[str], success: bool = True):
        """Record a tool/action sequence."""
        if len(sequence) < 2:
            return

        # Create pattern key (sequence as string)
        pattern_key = " -> ".join(sequence)

        if pattern_key not in self._pattern_stats:
            self._pattern_stats[pattern_key] = {
                "sequence": sequence,
                "frequency": 0,
                "successes": 0,
                "failures": 0,
            }

        stats = self._pattern_stats[pattern_key]
        stats["frequency"] += 1
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    def get_patterns(self, min_success_rate: float = 0.7) -> List[LearnedPattern]:
        """Get learned patterns meeting the success threshold."""
        patterns = []

        for pattern_key, stats in self._pattern_stats.items():
            if stats["frequency"] >= self.min_frequency:
                success_rate = (
                    stats["successes"] / stats["frequency"]
                    if stats["frequency"] > 0 else 0
                )

                if success_rate >= min_success_rate:
                    patterns.append(LearnedPattern(
                        pattern=pattern_key,
                        sequence=stats["sequence"],
                        frequency=stats["frequency"],
                        success_rate=success_rate
                    ))

        return sorted(patterns, key=lambda p: p.frequency, reverse=True)


# ============== Context Compressor (Advanced) ==============

class AdvancedCompressor:
    """
    Advanced context compression with multiple strategies.
    """

    def __init__(
        self,
        budget: MemoryBudget = None,
        scorer: ImportanceScorer = None,
        extractor: FactExtractor = None
    ):
        self.budget = budget or MemoryBudget()
        self.scorer = scorer or ImportanceScorer()
        self.extractor = extractor or FactExtractor()

    def compress(
        self,
        messages: List[ContextMessage],
        target_tokens: int = None
    ) -> List[ContextMessage]:
        """
        Compress messages to fit within token budget.

        Uses hybrid strategy:
        1. Score all messages by importance
        2. Keep recent high-importance messages
        3. Summarize older important messages
        4. Extract and preserve key facts
        """
        if not messages:
            return []

        target = target_tokens or self.budget.total_budget
        total_tokens = sum(m.token_count for m in messages)

        if total_tokens <= target:
            return messages

        # Score all messages
        for msg in messages:
            if msg.importance == 0.5:  # Not explicitly set
                msg.importance = self.scorer.score(msg)

        # Split into recent and older
        recent_count = min(10, len(messages) // 3)
        recent = messages[-recent_count:]
        older = messages[:-recent_count]

        # Keep recent messages (they're most relevant)
        compressed = list(recent)
        recent_tokens = sum(m.token_count for m in recent)

        # If under budget, add important older messages
        remaining_budget = target - recent_tokens

        if remaining_budget > 0 and older:
            # Sort older by importance
            sorted_older = sorted(older, key=lambda m: m.importance, reverse=True)

            # Add high-importance older messages
            for msg in sorted_older:
                if msg.token_count <= remaining_budget and msg.importance > 0.6:
                    compressed.append(msg)
                    remaining_budget -= msg.token_count

        # If still over budget, summarize
        if sum(m.token_count for m in compressed) > target:
            compressed = self._hybrid_compress(messages, target)

        logger.info(
            f"Compressed: {total_tokens} -> {sum(m.token_count for m in compressed)} tokens "
            f"({len(messages)} -> {len(compressed)} messages)"
        )

        return compressed

    def _hybrid_compress(
        self,
        messages: List[ContextMessage],
        target: int
    ) -> List[ContextMessage]:
        """Hybrid compression: keep recent + summary of older."""
        if not messages:
            return []

        # Keep last N messages
        keep_count = min(8, len(messages))
        recent = messages[-keep_count:]
        older = messages[:-keep_count]

        # Create summary of older
        summary_tokens = max(50, target - sum(m.token_count for m in recent))

        # Extract topics from older messages
        topics = self._extract_topics(older)
        summary = f"[Previous conversation: {len(older)} messages] Topics: {topics}"

        summary_msg = ContextMessage(
            role="system",
            content=summary,
            importance=0.6,
            memory_type=MemoryType.EPISODIC,
            token_count=len(summary) // CHARS_PER_TOKEN
        )

        return [summary_msg] + recent

    def _extract_topics(self, messages: List[ContextMessage]) -> str:
        """Extract key topics from messages."""
        word_freq: Dict[str, int] = defaultdict(int)

        for msg in messages:
            # Simple word frequency
            words = msg.content.lower().split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    word_freq[word] += 1

        # Get top topics
        top = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return ", ".join(w for w, _ in top)


# ============== Main Smart Context Manager ==============

class SmartContextManager:
    """
    Main context manager with hierarchical memory.

    Features:
    - Token budget management
    - Importance-based retention
    - Fact extraction and preservation
    - Pattern learning
    - RAG integration
    - LRU caching
    """

    def __init__(
        self,
        max_tokens: int = 5000,
        enable_rag: bool = True,
        enable_learning: bool = True
    ):
        self.max_tokens = max_tokens
        self.enable_rag = enable_rag
        self.enable_learning = enable_learning

        # Budget
        self.budget = MemoryBudget(
            working_tokens=int(max_tokens * 0.4),
            episodic_tokens=int(max_tokens * 0.3),
            semantic_tokens=int(max_tokens * 0.2),
            procedural_tokens=int(max_tokens * 0.1)
        )

        # Components
        self.scorer = ImportanceScorer()
        self.extractor = FactExtractor()
        self.compressor = AdvancedCompressor(self.budget, self.scorer, self.extractor)
        self.pattern_learner = PatternLearner()

        # Memory stores
        self._working_memory: List[ContextMessage] = []
        self._episodic_memory: List[ContextMessage] = []  # Past session summaries
        self._semantic_memory: List[ExtractedFact] = []   # Extracted facts
        self._procedural_memory: List[LearnedPattern] = [] # Learned patterns

        # Cache
        self._context_cache: Dict[str, List[ContextMessage]] = {}
        self._cache_lock = threading.Lock()

        # Thread safety
        self._lock = threading.RLock()

        # RAG integration
        self._rag_engine = None
        if enable_rag:
            self._init_rag()

        logger.info(f"SmartContextManager initialized with {max_tokens} token budget")

    def _init_rag(self):
        """Initialize RAG integration."""
        try:
            from .rag_engine import get_rag_engine
            self._rag_engine = get_rag_engine()
            logger.info("RAG integration enabled")
        except Exception as e:
            logger.warning(f"RAG not available: {e}")

    # --- Core API ---

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> ContextMessage:
        """
        Add a message to working memory.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata (has_code, has_error, etc.)

        Returns:
            The added message
        """
        with self._lock:
            msg = ContextMessage(
                role=role,
                content=content,
                metadata=metadata or {},
                memory_type=MemoryType.WORKING
            )
            msg.importance = self.scorer.score(msg)

            self._working_memory.append(msg)

            # Periodically extract facts
            if len(self._working_memory) % 10 == 0:
                self._extract_facts()

            # Invalidate cache
            self._context_cache.clear()

            return msg

    def get_context(
        self,
        include_rag: bool = True,
        max_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get optimized context for AI model.

        Args:
            include_rag: Whether to include RAG-retrieved context
            max_tokens: Override max tokens

        Returns:
            List of message dicts ready for the model
        """
        with self._lock:
            target = max_tokens or self.max_tokens

            # Compress working memory
            compressed = self.compressor.compress(
                self._working_memory,
                target - self.budget.episodic_tokens
            )

            # Build context
            context = []

            # 1. System prompt with semantic memory (key facts)
            if self._semantic_memory:
                facts_text = self._format_facts()
                context.append({
                    "role": "system",
                    "content": f"Key facts from previous sessions:\n{facts_text}"
                })

            # 2. Episodic memory (past sessions)
            if self._episodic_memory:
                context.extend([
                    {"role": m.role, "content": m.content}
                    for m in self._episodic_memory[-3:]
                ])

            # 3. Current working memory
            context.extend([
                {"role": m.role, "content": m.content}
                for m in compressed
            ])

            # 4. Procedural memory (if relevant)
            if self._procedural_memory and self.enable_learning:
                patterns = self._get_relevant_patterns()
                if patterns:
                    pattern_text = "\n".join(
                        f"- {p.pattern} (used {p.frequency}x, {p.success_rate:.0%} success)"
                        for p in patterns[:3]
                    )
                    context.append({
                        "role": "system",
                        "content": f"Learned patterns:\n{pattern_text}"
                    })

            # 5. RAG context
            if include_rag and self._rag_engine and self._working_memory:
                # Get last user message for retrieval
                last_user_msg = None
                for m in reversed(self._working_memory):
                    if m.role == "user":
                        last_user_msg = m.content
                        break

                if last_user_msg:
                    try:
                        rag_results = self._rag_engine.retrieve(last_user_msg, top_k=3)
                        if rag_results:
                            rag_text = "\n\n".join(
                                f"[{r.source}]\n{r.text[:200]}..."
                                for r in rag_results
                            )
                            context.append({
                                "role": "system",
                                "content": f"Relevant knowledge:\n{rag_text}"
                            })
                    except Exception as e:
                        logger.warning(f"RAG retrieval failed: {e}")

            # Check token count
            total = sum(
                len(m["content"]) // CHARS_PER_TOKEN
                for m in context
            )
            logger.debug(f"Context size: {total} tokens ({len(context)} messages)")

            return context

    def get_context_text(self, **kwargs) -> str:
        """Get context as formatted text string."""
        context = self.get_context(**kwargs)
        return "\n\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in context
        )

    # --- Session Management ---

    def end_session(self) -> Dict[str, Any]:
        """
        End current session and archive to episodic memory.

        Returns:
            Session summary
        """
        with self._lock:
            if not self._working_memory:
                return {"status": "no_messages"}

            # Extract facts before clearing
            self._extract_facts()

            # Create session summary
            summary = self._create_session_summary()

            # Archive to episodic
            summary_msg = ContextMessage(
                role="system",
                content=summary,
                memory_type=MemoryType.EPISODIC,
                importance=0.7
            )
            self._episodic_memory.append(summary_msg)

            # Keep only recent episodes
            max_episodes = 10
            if len(self._episodic_memory) > max_episodes:
                self._episodic_memory = self._episodic_memory[-max_episodes:]

            # Capture stats before clearing
            message_count = len(self._working_memory)

            # Clear working memory
            self._working_memory.clear()
            self._context_cache.clear()

            session_info = {
                "status": "saved",
                "message_count": message_count,
                "facts_extracted": len(self._semantic_memory),
                "patterns_learned": len(self._procedural_memory)
            }

            logger.info(f"Session ended: {session_info}")
            return session_info

    def _extract_facts(self):
        """Extract facts from recent working memory."""
        if not self._working_memory:
            return

        new_facts = self.extractor.extract(self._working_memory[-20:])

        for fact in new_facts:
            # Check if we already have this fact
            existing = any(
                f.fact.lower() == fact.fact.lower()
                for f in self._semantic_memory
            )
            if not existing:
                self._semantic_memory.append(fact)

        # Keep only high-confidence, recent facts
        self._semantic_memory = [
            f for f in self._semantic_memory
            if f.confidence >= 0.5
        ][:100]  # Max 100 facts

    def _create_session_summary(self) -> str:
        """Create a summary of the current session."""
        if not self._working_memory:
            return "Empty session"

        roles = defaultdict(int)
        total_tokens = 0

        for msg in self._working_memory:
            roles[msg.role] += 1
            total_tokens += msg.token_count

        topics = self.compressor._extract_topics(self._working_memory[:10])

        return (
            f"Session: {len(self._working_memory)} messages, "
            f"{total_tokens} tokens. "
            f"Roles: {dict(roles)}. "
            f"Topics: {topics}"
        )

    def _format_facts(self) -> str:
        """Format semantic memory facts for context."""
        if not self._semantic_memory:
            return ""

        # Group by category
        by_category = defaultdict(list)
        for fact in self._semantic_memory[-20:]:  # Recent 20
            by_category[fact.category].append(fact.fact)

        parts = []
        for category, facts in by_category.items():
            parts.append(f"{category}: {', '.join(facts[:5])}")

        return "; ".join(parts)

    def _get_recent_patterns(self) -> List[LearnedPattern]:
        """Get patterns relevant to recent work."""
        if not self._working_memory:
            return []

        # Get tool sequence from recent messages
        recent_tools = [
            m.content.split()[0]
            for m in self._working_memory[-10:]
            if m.role == "tool"
        ]

        if not recent_tools:
            return []

        # Find patterns that start with these tools
        relevant = []
        for pattern in self._procedural_memory[:10]:
            if pattern.sequence[0] in recent_tools:
                relevant.append(pattern)

        return relevant[:3]

    # --- Tool Integration ---

    def record_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        success: bool,
        duration: float
    ):
        """Record a tool call for pattern learning."""
        if not self.enable_learning:
            return

        # Extract sequence
        # This would need to be called with the full sequence
        pass

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                "working_memory": {
                    "messages": len(self._working_memory),
                    "tokens": sum(m.token_count for m in self._working_memory),
                    "budget": self.budget.working_tokens
                },
                "episodic_memory": {
                    "sessions": len(self._episodic_memory),
                    "budget": self.budget.episodic_tokens
                },
                "semantic_memory": {
                    "facts": len(self._semantic_memory),
                    "budget": self.budget.semantic_tokens
                },
                "procedural_memory": {
                    "patterns": len(self._procedural_memory),
                    "budget": self.budget.procedural_tokens
                },
                "total_tokens": self.max_tokens,
                "rag_enabled": self._rag_engine is not None,
                "learning_enabled": self.enable_learning
            }


# ============== Global Instance ==============

_context_manager: Optional[SmartContextManager] = None


def get_context_manager(
    max_tokens: int = 5000,
    enable_rag: bool = True,
    enable_learning: bool = True
) -> SmartContextManager:
    """Get global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = SmartContextManager(
            max_tokens=max_tokens,
            enable_rag=enable_rag,
            enable_learning=enable_learning
        )
    return _context_manager


# ============== Convenience Functions ==============

def add_message(role: str, content: str, **kwargs):
    """Quick add message to context."""
    get_context_manager().add_message(role, content, **kwargs)


def get_context(**kwargs) -> List[Dict[str, Any]]:
    """Quick get context."""
    return get_context_manager().get_context(**kwargs)


def end_session() -> Dict[str, Any]:
    """Quick end session."""
    return get_context_manager().end_session()


def get_stats() -> Dict[str, Any]:
    """Quick get stats."""
    return get_context_manager().get_stats()


# ============== Backwards Compatibility ==============

# Old classes re-exported for compatibility
# Backwards compatible ContextCompressor wrapper
class ContextCompressor:
    """Backwards compatible context compressor."""

    def __init__(
        self,
        max_tokens: int = 4000,
        preserve_recent: int = 10,
        summary_ratio: float = 0.3
    ):
        self.max_tokens = max_tokens
        self.preserve_recent = preserve_recent
        self.summary_ratio = summary_ratio

    def compress(self, entries):
        """Compress entries - old implementation for compatibility."""
        # Calculate total tokens
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
            result = important + recent
            for r in result:
                r.compressed = True
            return result

        # Create summary of compressed messages
        summary = self._create_summary(to_compress)
        summary_entry = ContextMessage(
            role="system",
            content=f"[Compressed context — {len(to_compress)} messages]\n{summary}",
            importance=0.6
        )
        summary_entry.compressed = True

        result = [summary_entry] + important + recent

        # Mark all as compressed
        for r in result:
            r.compressed = True

        return result

    def _create_summary(self, entries):
        """Create a summary of context entries."""
        # Group by role
        by_role = defaultdict(list)
        for entry in entries:
            by_role[entry.role].append(entry.content)

        parts = []
        if by_role.get("user"):
            topics = set()
            for content in by_role["user"]:
                words = content.lower().split()
                topics.update(w for w in words if len(w) > 4)
            if topics:
                parts.append(f"Topics discussed: {', '.join(list(topics)[:15])}")

        if by_role.get("assistant"):
            parts.append(f"Assistant provided {len(by_role['assistant'])} responses")

        if not parts:
            parts.append(f"{len(entries)} messages exchanged")

        return "; ".join(parts)


ContextEntry = ContextMessage

# Legacy classes (implemented separately for backwards compatibility)
class ConversationChain:
    """Legacy conversation chaining (deprecated)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._links: List = []
        self._key_facts: List[str] = []
        self._lock = threading.Lock()

    def link_conversation(self, conversation_id: str, summary: str, key_facts: List = None, relevance: float = 0.5):
        from dataclasses import dataclass, field
        @dataclass
        class Link:
            conversation_id: str
            summary: str
            key_facts: List = field(default_factory=list)
            relevance_score: float = 0.5
        with self._lock:
            self._links.append(Link(conversation_id, summary, key_facts or [], relevance))

    def add_key_fact(self, fact: str):
        with self._lock:
            if fact not in self._key_facts:
                self._key_facts.append(fact)

    def get_context(self, max_links: int = 5) -> str:
        with self._lock:
            if not self._links and not self._key_facts:
                return ""
            parts = []
            sorted_links = sorted(self._links, key=lambda l: l.relevance_score, reverse=True)[:max_links]
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
        with self._lock:
            all_facts = list(self._key_facts)
            for link in self._links:
                all_facts.extend(link.key_facts)
            return all_facts


class ToolAnalytics:
    """Legacy tool analytics (deprecated)."""

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def record_call(self, tool_name: str, duration: float = 0.0, success: bool = True, error: str = None):
        with self._lock:
            if tool_name not in self._metrics:
                self._metrics[tool_name] = {
                    "total": 0, "successes": 0, "failures": 0,
                    "duration": 0.0, "last_error": None, "last_used": None
                }
            m = self._metrics[tool_name]
            m["total"] += 1
            m["duration"] += duration
            m["last_used"] = time.time()
            if success:
                m["successes"] += 1
            else:
                m["failures"] += 1
                m["last_error"] = error

    def get_metrics(self, tool_name: str):
        with self._lock:
            m = self._metrics.get(tool_name)
            if not m:
                return None
            # Return object-like interface
            class Metrics:
                pass
            mm = Metrics()
            mm.total_calls = m["total"]
            mm.successes = m["successes"]
            mm.failures = m["failures"]
            mm.total_duration = m["duration"]
            mm.last_error = m["last_error"]
            mm.last_used = m["last_used"]
            mm.success_rate = m["successes"] / m["total"] if m["total"] > 0 else 0.0
            mm.avg_duration = m["duration"] / m["total"] if m["total"] > 0 else 0.0
            # Compute health status
            if mm.total_calls == 0:
                mm.health_status = "unknown"
            elif mm.success_rate >= 0.95:
                mm.health_status = "healthy"
            elif mm.success_rate >= 0.7:
                mm.health_status = "degraded"
            else:
                mm.health_status = "unhealthy"
            return mm

    def get_report(self) -> Dict:
        with self._lock:
            if not self._metrics:
                return {"tools": {}, "summary": {}}
            tools = {}
            total_calls = sum(m["total"] for m in self._metrics.values())
            total_successes = sum(m["successes"] for m in self._metrics.values())
            for name, m in self._metrics.items():
                tools[name] = {
                    "total_calls": m["total"],
                    "success_rate": round(m["successes"] / m["total"], 3) if m["total"] > 0 else 0,
                    "avg_duration_ms": round(m["duration"] / m["total"] * 1000, 1) if m["total"] > 0 else 0,
                    "health": "healthy" if m["successes"] / m["total"] >= 0.95 else "degraded" if m["successes"] / m["total"] >= 0.7 else "unhealthy",
                    "last_error": m["last_error"],
                }
            return {"tools": tools, "summary": {"total_calls": total_calls, "overall_success_rate": round(total_successes / total_calls, 3) if total_calls > 0 else 0}}

    def get_recommendations(self) -> List[str]:
        recommendations = []
        with self._lock:
            for name, m in self._metrics.items():
                if m["total"] >= 5 and m["successes"] / m["total"] < 0.7:
                    recommendations.append(
                        f"Tool '{name}' has low success rate ({m['successes'] / m['total']:.0%}). "
                        f"Consider investigating: {m['last_error']}"
                    )
                if m["total"] >= 5 and m["duration"] / m["total"] > 10.0:
                    recommendations.append(
                        f"Tool '{name}' is slow (avg {m['duration'] / m['total']:.1f}s). "
                        f"Consider caching or optimization."
                    )
        return recommendations


class AgentSelfHealer:
    """Legacy agent self-healer (deprecated)."""

    def __init__(self, failure_threshold: int = 3, max_restarts: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.max_restarts = max_restarts
        self.cooldown_seconds = cooldown_seconds
        self._agents: Dict[str, Any] = {}
        self._restart_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_restart_handler(self, agent_id: str, handler: Callable):
        with self._lock:
            self._restart_handlers[agent_id] = handler
            if agent_id not in self._agents:
                self._agents[agent_id] = {"failures": 0, "restarts": 0, "healthy": True, "last_failure": None, "last_restart": None}

    def record_success(self, agent_id: str):
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = {"failures": 0, "restarts": 0, "healthy": True, "last_failure": None, "last_restart": None}
            self._agents[agent_id]["failures"] = 0
            self._agents[agent_id]["healthy"] = True

    def record_failure(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = {"failures": 0, "restarts": 0, "healthy": True, "last_failure": None, "last_restart": None}
            record = self._agents[agent_id]
            record["failures"] += 1
            record["last_failure"] = time.time()
            if record["failures"] >= self.failure_threshold:
                record["healthy"] = False
                if record["restarts"] >= self.max_restarts:
                    return False
                # Check cooldown
                if record["last_restart"] and time.time() - record["last_restart"] < self.cooldown_seconds:
                    return False
                handler = self._restart_handlers.get(agent_id)
                if handler:
                    try:
                        handler()
                        record["restarts"] += 1
                        record["last_restart"] = time.time()
                        record["failures"] = 0
                        record["healthy"] = True
                        return True
                    except:
                        record["restarts"] += 1
                        return False
            return False

    def get_status(self) -> Dict:
        with self._lock:
            return {aid: {"healthy": r["healthy"], "consecutive_failures": r["failures"], "total_restarts": r["restarts"], "last_failure": r["last_failure"]} for aid, r in self._agents.items()}

    def get_unhealthy(self) -> List[str]:
        with self._lock:
            return [aid for aid, r in self._agents.items() if not r["healthy"]]


__all__ = [
    # New advanced classes
    "SmartContextManager",
    "ContextMessage",
    "MemoryBudget",
    "ImportanceScorer",
    "FactExtractor",
    "PatternLearner",
    "AdvancedCompressor",
    "MemoryType",
    "CompressionStrategy",
    # Convenience functions
    "get_context_manager",
    "add_message",
    "get_context",
    "end_session",
    "get_stats",
    # Backwards compatibility
    "ContextCompressor",
    "ContextEntry",
    "ConversationChain",
    "ToolAnalytics",
    "AgentSelfHealer",
]
