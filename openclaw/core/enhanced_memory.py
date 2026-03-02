"""
Enhanced Agent Memory System for OpenClaw

Features (2026 Best Practices):
- Git-scoped memory storage for project isolation
- Memory index with lazy loading ()
- HierarchicalClaude Code pattern memory scopes (CrewAI pattern)
- Composite scoring with semantic + recency + importance
- Memory consolidation and deduplication
- Non-blocking async saves
- Path-scoped memory rules
- Memory compaction system
- Subagent memory isolation
- Cross-session synthesis

Based on research from:
- Claude Code memory system
- CrewAI memory architecture
- LangChain Deep Agents
- Mem0 platform patterns
"""

import os
import re
import json
import time
import shutil
import hashlib
import threading
import subprocess
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from .logger import get_logger

logger = get_logger("enhanced_memory")


# ============== Enums ==============

class MemoryType(Enum):
    """Types of memory"""
    EPISODIC = "episodic"      # Specific events/experiences
    SEMANTIC = "semantic"      # Facts and knowledge
    WORKING = "working"        # Current context
    PROCEDURAL = "procedural"  # How to do things
    INSTRUCTION = "instruction"  # User-defined instructions


class MemoryScope(Enum):
    """Memory scope levels (hierarchical)"""
    GLOBAL = "/"                      # Machine-wide
    COMPANY = "/company"              # Organization
    PROJECT = "/project"              # Project-specific
    AGENT = "/agent"                  # Agent-specific
    SUBAGENT = "/agent/subagent"      # Subagent-specific


class MemorySource(Enum):
    """Source of memory content"""
    USER = "user"           # User-defined
    AUTO = "auto"           # System auto-learned
    EXTRACTED = "extracted" # Extracted from conversation
    IMPORTED = "imported"   # Imported from external


# ============== Data Classes ==============

@dataclass
class MemoryEntry:
    """A single memory entry with enhanced metadata"""
    id: str
    content: str
    memory_type: MemoryType
    scope: str = "/project"
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    source: MemorySource = MemorySource.AUTO
    path_patterns: List[str] = field(default_factory=list)  # For path-scoped rules
    version: int = 1
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    private: bool = False  # For privacy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "scope": self.scope,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "source": self.source.value,
            "path_patterns": self.path_patterns,
            "version": self.version,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "private": self.private
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dict"""
        data["memory_type"] = MemoryType(data["memory_type"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


@dataclass
class MemoryIndex:
    """Memory index for quick lookup (MEMORY.md style)"""
    entries: List[Dict[str, Any]] = field(default_factory=list)  # Summarized entries
    total_count: int = 0
    last_updated: float = field(default_factory=time.time)
    version: int = 1


@dataclass
class MemoryQuery:
    """Query for memory search"""
    text: str
    memory_type: Optional[MemoryType] = None
    scope: Optional[str] = None
    path: Optional[str] = None  # Current file path for path-scoped rules
    limit: int = 10
    min_importance: float = 0.0
    min_similarity: float = 0.0
    include_private: bool = True


@dataclass
class MemoryScore:
    """Composite memory score"""
    memory: MemoryEntry
    total_score: float = 0.0
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    path_match: bool = False


# ============== Configuration ==============

class MemoryConfig:
    """Memory system configuration"""

    # Paths
    DEFAULT_BASE_DIR = "~/.openclaw"
    MEMORY_DIR_NAME = "memory"
    INDEX_FILE_NAME = "MEMORY.md"
    RULES_DIR_NAME = "rules"

    # Limits
    MAX_INDEX_LINES = 200
    MAX_MEMORIES = 10000
    MAX_MEMORY_SIZE = 100_000  # chars
    CONSOLIDATION_THRESHOLD = 0.85

    # Weights for composite scoring
    DEFAULT_SEMANTIC_WEIGHT = 0.4
    DEFAULT_RECENCY_WEIGHT = 0.3
    DEFAULT_IMPORTANCE_WEIGHT = 0.3

    # Decay
    RECENCY_HALF_LIFE_DAYS = 7

    # Async
    ASYNC_SAVE = True
    MAX_WORKERS = 4

    # Embedding
    EMBEDDING_DIM = 384  # Matches sentence-transformers/all-MiniLM-L6-v2


# ============== Git Scope Detector ==============

class GitScopeDetector:
    """Detects git repository and worktree for memory scoping"""

    @staticmethod
    def get_repo_path() -> Optional[str]:
        """Get current git repository path"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    @staticmethod
    def get_repo_name() -> Optional[str]:
        """Get repository name"""
        repo_path = GitScopeDetector.get_repo_path()
        if repo_path:
            return Path(repo_path).name
        return None

    @staticmethod
    def get_worktree_name() -> Optional[str]:
        """Get current worktree name"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--absolute-git-dir"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_dir = result.stdout.strip()
                # Check if in worktree
                if ".git/worktrees" in git_dir:
                    return Path(git_dir).parent.name
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    @staticmethod
    def get_memory_path() -> str:
        """Get memory path scoped to git repository"""
        base = os.path.expanduser(MemoryConfig.DEFAULT_BASE_DIR)
        repo_name = GitScopeDetector.get_repo_name()

        if repo_name:
            # Scope to git repo
            return os.path.join(base, "projects", repo_name, MemoryConfig.MEMORY_DIR_NAME)
        else:
            # Fallback to default
            return os.path.join(base, MemoryConfig.MEMORY_DIR_NAME)


# ============== Embedding Provider ==============

class EmbeddingProvider:
    """Simple embedding provider with fallback"""

    def __init__(self, model: str = "simple"):
        self.model = model
        self._provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize embedding provider"""
        try:
            from .embeddings import get_embedding_provider
            self._provider = get_embedding_provider()
        except Exception as e:
            logger.debug(f"Using simple embedding: {e}")

    def embed(self, text: str) -> List[float]:
        """Get embedding for text"""
        if self._provider:
            try:
                result = self._provider.embed(text)
                return result if isinstance(result, list) else result.tolist()
            except Exception as e:
                logger.debug(f"Provider embedding failed: {e}")

        # Fallback to simple hash-based embedding
        return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple bag-of-words embedding"""
        words = text.lower().split()
        dim = MemoryConfig.EMBEDDING_DIM
        embedding = np.zeros(dim)

        for word in words:
            hash_val = hash(word)
            idx = hash_val % dim
            embedding[idx] += 1

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


# ============== Path Scoped Rules ==============

class PathScopedRules:
    """Manages path-scoped memory rules"""

    def __init__(self, rules_dir: str):
        self.rules_dir = rules_dir
        self._rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self):
        """Load rules from directory"""
        if not os.path.exists(self.rules_dir):
            return

        for filepath in Path(self.rules_dir).glob("**/*.md"):
            try:
                content = filepath.read_text()
                # Parse frontmatter
                rule = self._parse_rule_file(filepath.name, content)
                if rule:
                    self._rules.append(rule)
            except Exception as e:
                logger.warning(f"Failed to load rule {filepath}: {e}")

    def _parse_rule_file(self, filename: str, content: str) -> Optional[Dict[str, Any]]:
        """Parse a rule file with frontmatter"""
        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                body = parts[2].strip()

                # Parse frontmatter
                paths = []
                for line in frontmatter.split("\n"):
                    if line.startswith("paths:"):
                        # Extract paths
                        pass

                return {
                    "name": filename,
                    "paths": paths,
                    "content": body
                }

        return None

    def get_matching_rules(self, file_path: str) -> List[str]:
        """Get rules matching a file path"""
        matching = []

        for rule in self._rules:
            patterns = rule.get("paths", [])
            if not patterns:  # No path restriction, apply to all
                matching.append(rule["content"])
            else:
                for pattern in patterns:
                    if self._match_pattern(file_path, pattern):
                        matching.append(rule["content"])
                        break

        return matching

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match path against glob pattern"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)


# ============== Main Enhanced Memory ==============

class EnhancedMemory:
    """
    Enhanced memory system with 2026 best practices.

    Features:
    - Git-scoped storage for project isolation
    - Memory index with lazy loading
    - Hierarchical scopes
    - Composite scoring
    - Path-scoped rules
    - Async saves
    - Memory compaction
    """

    def __init__(
        self,
        agent_id: str = "default",
        memory_dir: str = None,
        max_memories: int = MemoryConfig.MAX_MEMORIES,
        embedding_model: str = "simple",
        semantic_weight: float = MemoryConfig.DEFAULT_SEMANTIC_WEIGHT,
        recency_weight: float = MemoryConfig.DEFAULT_RECENCY_WEIGHT,
        importance_weight: float = MemoryConfig.DEFAULT_IMPORTANCE_WEIGHT,
        enable_async: bool = True
    ):
        self.agent_id = agent_id
        self.memory_dir = memory_dir or GitScopeDetector.get_memory_path()
        self.max_memories = max_memories
        self.enable_async = enable_async

        # Scoring weights
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight

        # Components
        self.embedding_provider = EmbeddingProvider(embedding_model)

        # Storage
        self._memories: Dict[str, MemoryEntry] = {}
        self._working_memory: Dict[str, Any] = {}
        self._index: Optional[MemoryIndex] = None
        self._path_rules: Optional[PathScopedRules] = None

        # Async executor
        self._executor: Optional[ThreadPoolExecutor] = None
        if enable_async:
            self._executor = ThreadPoolExecutor(max_workers=MemoryConfig.MAX_WORKERS)

        # Thread safety
        self._lock = threading.RLock()

        # Create directory
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(os.path.join(self.memory_dir, MemoryConfig.RULES_DIR_NAME), exist_ok=True)

        # Initialize
        self._load_index()
        self._load_memories()
        self._init_path_rules()

        logger.info(f"EnhancedMemory initialized at {self.memory_dir}")

    def _init_path_rules(self):
        """Initialize path-scoped rules"""
        rules_dir = os.path.join(self.memory_dir, MemoryConfig.RULES_DIR_NAME)
        self._path_rules = PathScopedRules(rules_dir)

    # --- Core Operations ---

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        scope: str = "/project",
        metadata: Dict[str, Any] = None,
        source: MemorySource = MemorySource.AUTO,
        path_patterns: List[str] = None,
        tags: List[str] = None,
        private: bool = False,
        async_save: bool = None
    ) -> str:
        """Add a new memory entry"""
        with self._lock:
            # Check for duplicate
            for mem in self._memories.values():
                if mem.content == content:
                    mem.access_count += 1
                    mem.last_accessed = time.time()
                    return mem.id

            # Generate embedding
            embedding = self.embedding_provider.embed(content)

            # Create memory
            memory = MemoryEntry(
                id=self._generate_id(content),
                content=content[:MemoryConfig.MAX_MEMORY_SIZE],  # Truncate if too long
                memory_type=memory_type,
                scope=scope,
                importance=importance,
                embedding=embedding,
                metadata=metadata or {},
                source=source,
                path_patterns=path_patterns or [],
                tags=tags or [],
                private=private
            )

            # Store
            self._memories[memory.id] = memory

            # Cleanup if needed
            if len(self._memories) > self.max_memories:
                self._cleanup()

            # Save (async or sync)
            use_async = async_save if async_save is not None else self.enable_async
            if use_async and self._executor:
                self._executor.submit(self._save_memory, memory)
            else:
                self._save_memory(memory)

            # Update index
            self._update_index(memory)

            logger.debug(f"Added memory: {memory.id} ({memory_type.value})")
            return memory.id

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory"""
        memory = self._memories.get(memory_id)

        if memory:
            with self._lock:
                memory.access_count += 1
                memory.last_accessed = time.time()

        return memory

    def query_memories(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Query memories using composite scoring"""
        results = []

        # Compute query embedding
        query_embedding = self.embedding_provider.embed(query.text)

        for memory in self._memories.values():
            # Filter by type
            if query.memory_type and memory.memory_type != query.memory_type:
                continue

            # Filter by scope
            if query.scope and not memory.scope.startswith(query.scope):
                continue

            # Filter by importance
            if memory.importance < query.min_importance:
                continue

            # Filter by privacy
            if memory.private and not query.include_private:
                continue

            # Filter by path patterns (if path provided)
            path_match = False
            if query.path and memory.path_patterns:
                for pattern in memory.path_patterns:
                    if self._path_rules._match_pattern(query.path, pattern):
                        path_match = True
                        break

            # Calculate composite score
            score = self._calculate_composite_score(
                memory, query_embedding, query.text
            )

            # Apply minimum similarity threshold
            if score.semantic_score < query.min_similarity:
                continue

            # Mark path match
            if path_match or not memory.path_patterns:
                score.path_match = True

            results.append((memory, score.total_score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return [m for m, _ in results[:query.limit]]

    def _calculate_composite_score(
        self,
        memory: MemoryEntry,
        query_embedding: List[float],
        query_text: str
    ) -> MemoryScore:
        """Calculate composite memory score"""
        # 1. Semantic similarity
        semantic = 0.0
        if memory.embedding:
            semantic = self._cosine_similarity(query_embedding, memory.embedding)

        # 2. Recency score (exponential decay)
        age_days = (time.time() - memory.timestamp) / 86400
        half_life = MemoryConfig.RECENCY_HALF_LIFE_DAYS
        recency = 0.5 ** (age_days / half_life)

        # 3. Importance score
        importance = memory.importance

        # Composite
        total = (
            self.semantic_weight * semantic +
            self.recency_weight * recency +
            self.importance_weight * importance
        )

        return MemoryScore(
            memory=memory,
            total_score=total,
            semantic_score=semantic,
            recency_score=recency,
            importance_score=importance
        )

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        a = np.array(a)
        b = np.array(b)

        # Handle dimension mismatch
        if a.shape != b.shape:
            # Try to resize smaller to larger
            if len(a) < len(b):
                a = np.pad(a, (0, len(b) - len(a)))
            elif len(b) < len(a):
                b = np.pad(b, (0, len(a) - len(b)))

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # --- Working Memory ---

    def update_working_memory(self, key: str, value: Any):
        """Update working memory (short-term)"""
        self._working_memory[key] = {
            "value": value,
            "timestamp": time.time()
        }

    def get_working_memory(self, key: str = None) -> Dict[str, Any]:
        """Get working memory"""
        if key:
            return self._working_memory.get(key, {}).get("value")
        return {
            k: v["value"]
            for k, v in self._working_memory.items()
        }

    def clear_working_memory(self):
        """Clear working memory"""
        self._working_memory.clear()

    # --- Index Management ---

    def _load_index(self):
        """Load memory index"""
        index_path = os.path.join(self.memory_dir, MemoryConfig.INDEX_FILE_NAME)

        if os.path.exists(index_path):
            try:
                content = Path(index_path).read_text()
                lines = content.split("\n")

                # Parse entries (simplified)
                entries = []
                for line in lines[:MemoryConfig.MAX_INDEX_LINES]:
                    if line.strip():
                        entries.append({"summary": line.strip()})

                self._index = MemoryIndex(
                    entries=entries,
                    total_count=len(entries),
                    last_updated=os.path.getmtime(index_path)
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index = MemoryIndex()

        if not self._index:
            self._index = MemoryIndex()

    def _update_index(self, memory: MemoryEntry):
        """Update memory index"""
        if not self._index:
            self._index = MemoryIndex()

        # Add summary entry
        summary = self._create_memory_summary(memory)

        self._index.entries.append({
            "id": memory.id,
            "type": memory.memory_type.value,
            "summary": summary,
            "importance": memory.importance,
            "timestamp": memory.timestamp
        })

        self._index.total_count = len(self._memories)
        self._index.last_updated = time.time()

        # Rebuild index if too large
        if len(self._index.entries) > MemoryConfig.MAX_INDEX_LINES:
            self._rebuild_index()

    def _create_memory_summary(self, memory: MemoryEntry) -> str:
        """Create a one-line summary for index"""
        # Take first 100 chars
        summary = memory.content[:100]
        if len(memory.content) > 100:
            summary += "..."
        return summary

    def _rebuild_index(self):
        """Rebuild index with latest memories"""
        # Keep most important and recent
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )

        entries = []
        for memory in sorted_memories[:MemoryConfig.MAX_INDEX_LINES]:
            entries.append({
                "id": memory.id,
                "type": memory.memory_type.value,
                "summary": self._create_memory_summary(memory),
                "importance": memory.importance,
                "timestamp": memory.timestamp
            })

        self._index.entries = entries
        self._index.version += 1

    def get_index_content(self, max_lines: int = None) -> str:
        """Get index content for loading at session start"""
        max_lines = max_lines or MemoryConfig.MAX_INDEX_LINES

        if not self._index:
            return ""

        lines = ["# Memory Index\n"]

        for entry in self._index.entries[:max_lines]:
            lines.append(f"- [{entry['type']}] {entry['summary']}")

        return "\n".join(lines)

    # --- Persistence ---

    def _save_memory(self, memory: MemoryEntry):
        """Save memory to disk"""
        try:
            filepath = os.path.join(self.memory_dir, f"{memory.id}.json")

            with open(filepath, 'w') as f:
                json.dump(memory.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save memory {memory.id}: {e}")

    def _load_memories(self):
        """Load all memories from disk"""
        try:
            for filepath in Path(self.memory_dir).glob("*.json"):
                if filepath.name == MemoryConfig.INDEX_FILE_NAME:
                    continue

                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        memory = MemoryEntry.from_dict(data)
                        self._memories[memory.id] = memory
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")

            logger.info(f"Loaded {len(self._memories)} memories")

        except Exception as e:
            logger.error(f"Failed to load memories: {e}")

    # --- Cleanup and Compaction ---

    def _cleanup(self):
        """Remove least important memories"""
        memories = list(self._memories.values())
        memories.sort(key=lambda x: (x.importance, x.access_count))

        remove_count = len(memories) // 10
        for memory in memories[:remove_count]:
            del self._memories[memory.id]
            self._delete_memory_file(memory.id)

        logger.info(f"Cleaned up {remove_count} memories")

    def _delete_memory_file(self, memory_id: str):
        """Delete memory file"""
        filepath = os.path.join(self.memory_dir, f"{memory_id}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to delete {filepath}: {e}")

    def consolidate(self, threshold: float = MemoryConfig.CONSOLIDATION_THRESHOLD):
        """Consolidate similar memories"""
        memories = list(self._memories.values())

        to_merge = []
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                if mem1.memory_type == mem2.memory_type:
                    # Check similarity
                    if mem1.embedding and mem2.embedding:
                        sim = self._cosine_similarity(mem1.embedding, mem2.embedding)
                        if sim >= threshold:
                            to_merge.append((mem1, mem2))

        # Merge
        for mem1, mem2 in to_merge:
            self._merge_memories(mem1, mem2)

        if to_merge:
            logger.info(f"Consolidated {len(to_merge)} memory pairs")

    def _merge_memories(self, mem1: MemoryEntry, mem2: MemoryEntry):
        """Merge two memories into one"""
        # Keep the more important one
        if mem2.importance > mem1.importance:
            mem1, mem2 = mem2, mem1

        # Update mem1
        mem1.access_count += mem2.access_count
        mem1.version += 1
        mem1.parent_id = mem2.id

        # Tag the old one as merged
        mem2.metadata["merged_into"] = mem1.id
        mem2.metadata["merged_at"] = time.time()

        # Save updated
        self._save_memory(mem1)
        self._save_memory(mem2)

        # Remove from active
        del self._memories[mem2.id]

    def compact(self) -> Dict[str, int]:
        """Compact memory - cleanup, consolidate, rebuild index"""
        stats = {
            "before": len(self._memories)
        }

        # Cleanup
        self._cleanup()

        # Consolidate
        self.consolidate()

        # Rebuild index
        self._rebuild_index()

        stats["after"] = len(self._memories)
        stats["removed"] = stats["before"] - stats["after"]

        logger.info(f"Compaction complete: {stats}")
        return stats

    # --- Path-Scoped Rules ---

    def add_rule(self, name: str, content: str, paths: List[str] = None):
        """Add a path-scoped rule"""
        rule_path = os.path.join(
            self.memory_dir,
            MemoryConfig.RULES_DIR_NAME,
            f"{name}.md"
        )

        # Build frontmatter
        frontmatter = "---\n"
        if paths:
            frontmatter += "paths:\n"
            for p in paths:
                frontmatter += f"  - {p}\n"
        frontmatter += "---\n\n"

        content = frontmatter + content

        Path(rule_path).write_text(content)

        # Reload rules
        self._init_path_rules()

    def get_rules_for_path(self, file_path: str) -> List[str]:
        """Get rules applicable to a file path"""
        if self._path_rules:
            return self._path_rules.get_matching_rules(file_path)
        return []

    # --- Subagent Memory ---

    def create_subagent_memory(self, parent_agent_id: str, subagent_id: str) -> 'EnhancedMemory':
        """Create isolated memory for a subagent"""
        # Create subagent-specific directory
        subagent_dir = os.path.join(self.memory_dir, "subagents", subagent_id)
        os.makedirs(subagent_dir, exist_ok=True)

        subagent_memory = EnhancedMemory(
            agent_id=subagent_id,
            memory_dir=subagent_dir,
            max_memories=self.max_memories // 2,  # Limit for subagents
            semantic_weight=self.semantic_weight,
            recency_weight=self.recency_weight,
            importance_weight=self.importance_weight,
            enable_async=self.enable_async
        )

        # Optionally inherit some parent memories
        return subagent_memory

    def inherit_from_parent(self, parent_memory: 'EnhancedMemory', percentage: float = 0.1):
        """Inherit memories from parent agent"""
        parent_memories = list(parent_memory._memories.values())
        parent_memories.sort(key=lambda m: m.importance, reverse=True)

        # Take top %
        count = int(len(parent_memories) * percentage)
        for memory in parent_memories[:count]:
            self.add_memory(
                content=f"[Inherited from parent] {memory.content}",
                memory_type=memory.memory_type,
                importance=memory.importance * 0.8,  # Slightly lower
                scope=memory.scope,
                source=MemorySource.IMPORTED,
                tags=["inherited"]
            )

    # --- Cross-Session Synthesis ---

    def synthesize_lessons(self, session_count: int = 10) -> str:
        """Synthesize lessons from past sessions"""
        episodic = [
            m for m in self._memories.values()
            if m.memory_type == MemoryType.EPISODIC
        ]

        if len(episodic) < session_count:
            return "Not enough sessions to synthesize"

        # Get recent sessions
        episodic.sort(key=lambda m: m.timestamp, reverse=True)
        recent = episodic[:session_count]

        # Extract patterns
        patterns = defaultdict(int)
        for mem in recent:
            # Extract key topics (simplified)
            words = mem.content.lower().split()
            for word in words:
                if len(word) > 5:
                    patterns[word] += 1

        # Get top patterns
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]

        # Create synthesis
        synthesis = "# Lessons Learned\n\n"
        synthesis += f"Synthesized from {len(recent)} recent sessions:\n\n"
        synthesis += "## Key Patterns\n"
        for word, count in top_patterns:
            synthesis += f"- {word}: {count} occurrences\n"

        synthesis += "\n## Recommendations\n"
        synthesis += "- Focus on high-importance tasks\n"
        synthesis += "- Maintain context across sessions\n"

        return synthesis

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        by_type = defaultdict(int)
        by_scope = defaultdict(int)
        total_access = 0

        for mem in self._memories.values():
            by_type[mem.memory_type.value] += 1
            by_scope[mem.scope] += 1
            total_access += mem.access_count

        return {
            "total": len(self._memories),
            "by_type": dict(by_type),
            "by_scope": dict(by_scope),
            "working_memory": len(self._working_memory),
            "max_memories": self.max_memories,
            "total_access": total_access,
            "index_entries": len(self._index.entries) if self._index else 0,
            "memory_dir": self.memory_dir
        }

    # --- Utilities ---

    def _generate_id(self, content: str) -> str:
        """Generate unique ID"""
        return hashlib.sha256(
            f"{content}{time.time()}{os.urandom(8)}".encode()
        ).hexdigest()[:16]

    def export(self, filepath: str):
        """Export all memories to file"""
        data = {
            "exported_at": time.time(),
            "memory_dir": self.memory_dir,
            "stats": self.get_stats(),
            "memories": [m.to_dict() for m in self._memories.values()]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self._memories)} memories to {filepath}")

    def import_memories(self, filepath: str) -> int:
        """Import memories from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        count = 0
        for mem_data in data.get("memories", []):
            try:
                memory = MemoryEntry.from_dict(mem_data)
                # Only add if not exists
                if memory.id not in self._memories:
                    self._memories[memory.id] = memory
                    self._save_memory(memory)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")

        self._rebuild_index()
        logger.info(f"Imported {count} memories from {filepath}")
        return count

    def close(self):
        """Close memory system"""
        if self._executor:
            self._executor.shutdown(wait=True)


# ============== Hierarchical Memory Manager ==============

class HierarchicalMemoryManager:
    """
    Manages hierarchical memory scopes (CrewAI pattern).

    Tree structure:
        /
          /company
            /company/engineering
            /company/product
          /project
            /project/alpha
            /project/beta
          /agent
            /agent/researcher
            /agent/writer

    MemoryScope restricts operations to a subtree.
    MemorySlice combines multiple scopes.
    """

    def __init__(self, base_memory: EnhancedMemory):
        self.base_memory = base_memory
        self._scopes: Dict[str, EnhancedMemory] = {}
        self._scope_lock = threading.Lock()

    def scope(self, path: str) -> 'ScopedMemory':
        """Get a scoped view of memory"""
        return ScopedMemory(self.base_memory, path)

    def slice(
        self,
        scopes: List[str],
        read_only: bool = False
    ) -> 'MemorySlice':
        """Get a slice combining multiple scopes"""
        return MemorySlice(self.base_memory, scopes, read_only)

    def create_subscope(
        self,
        parent_scope: str,
        child_name: str
    ) -> str:
        """Create a new sub-scope"""
        child_scope = f"{parent_scope}/{child_name}"
        return child_scope


class ScopedMemory:
    """Memory scoped to a specific path"""

    def __init__(self, memory: EnhancedMemory, scope: str):
        self.memory = memory
        self.scope = scope

    def add_memory(self, content: str, **kwargs) -> str:
        """Add memory within this scope"""
        kwargs["scope"] = self.scope
        return self.memory.add_memory(content, **kwargs)

    def query_memories(self, query: MemoryQuery, **kwargs) -> List[MemoryEntry]:
        """Query memories within this scope"""
        query.scope = self.scope
        return self.memory.query_memories(query, **kwargs)

    def get_all(self, limit: int = 100) -> List[MemoryEntry]:
        """Get all memories in this scope"""
        all_mems = [
            m for m in self.memory._memories.values()
            if m.scope.startswith(self.scope)
        ]
        all_mems.sort(key=lambda m: m.timestamp, reverse=True)
        return all_mems[:limit]


class MemorySlice:
    """Memory slice combining multiple scopes"""

    def __init__(
        self,
        memory: EnhancedMemory,
        scopes: List[str],
        read_only: bool = False
    ):
        self.memory = memory
        self.scopes = scopes
        self.read_only = read_only

    def query_memories(
        self,
        query: MemoryQuery,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Query across multiple scopes"""
        results = []

        for scope in self.scopes:
            query.scope = scope
            scope_results = self.memory.query_memories(query)
            results.extend(scope_results)

        # Deduplicate and sort
        seen = set()
        unique = []
        for mem in results:
            if mem.id not in seen:
                seen.add(mem.id)
                unique.append(mem)

        unique.sort(key=lambda m: m.importance, reverse=True)
        return unique[:limit]

    def add_memory(self, content: str, scope: str = None, **kwargs) -> str:
        """Add memory (if not read-only)"""
        if self.read_only:
            raise ValueError("Cannot add to read-only memory slice")

        scope = scope or self.scopes[0]
        kwargs["scope"] = scope
        return self.memory.add_memory(content, **kwargs)


# ============== Recall Flow ==============

class RecallFlow:
    """
    Different recall strategies for memory retrieval.

    Shallow: Vector-only similarity search
    Deep: LLM-assisted reasoning over memories
    """

    @staticmethod
    def shallow_recall(
        memory: EnhancedMemory,
        query: str,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """Shallow recall - vector similarity only"""
        return memory.query_memories(
            MemoryQuery(text=query, limit=limit)
        )

    @staticmethod
    def deep_recall(
        memory: EnhancedMemory,
        query: str,
        limit: int = 5,
        llm_provider: Callable = None
    ) -> List[MemoryEntry]:
        """
        Deep recall - uses LLM to reason over memories.

        This is more accurate but slower.
        """
        # First get candidate memories
        candidates = memory.query_memories(
            MemoryQuery(text=query, limit=limit * 2)
        )

        if not candidates or not llm_provider:
            return candidates[:limit]

        # Use LLM to filter/rank
        try:
            # Build prompt for LLM
            context = "\n\n".join(
                f"[{i+1}] {m.content[:200]}"
                for i, m in enumerate(candidates)
            )

            prompt = f"""Given the query: "{query}"

Which of these memories are most relevant? Consider:
1. Direct relevance to the query
2. Importance and reliability
3. Recency

Relevant memories (list numbers):"""

            response = llm_provider(prompt)

            # Parse response to get indices
            # (simplified - real implementation would be more robust)
            relevant = []
            for line in response.split("\n"):
                for i, mem in enumerate(candidates):
                    if str(i + 1) in line:
                        relevant.append(mem)

            return relevant[:limit] if relevant else candidates[:limit]

        except Exception as e:
            logger.warning(f"Deep recall failed: {e}")
            return candidates[:limit]


# ============== Global Store ==============

_memory_store: Dict[str, EnhancedMemory] = {}
_store_lock = threading.Lock()


def get_enhanced_memory(agent_id: str = "default") -> EnhancedMemory:
    """Get or create enhanced memory"""
    with _store_lock:
        if agent_id not in _memory_store:
            _memory_store[agent_id] = EnhancedMemory(agent_id)
        return _memory_store[agent_id]


def create_memory_for_project(project_name: str = None) -> EnhancedMemory:
    """Create memory scoped to a project"""
    if project_name is None:
        project_name = GitScopeDetector.get_repo_name() or "default"

    memory_dir = os.path.join(
        os.path.expanduser(MemoryConfig.DEFAULT_BASE_DIR),
        "projects",
        project_name,
        MemoryConfig.MEMORY_DIR_NAME
    )

    return EnhancedMemory(
        agent_id=project_name,
        memory_dir=memory_dir
    )


# ============== Convenience Functions ==============

def remember(
    content: str,
    memory_type: MemoryType = MemoryType.EPISODIC,
    importance: float = 0.5,
    agent_id: str = "default"
) -> str:
    """Quick remember function"""
    return get_enhanced_memory(agent_id).add_memory(
        content=content,
        memory_type=memory_type,
        importance=importance,
        source=MemorySource.AUTO
    )


def recall(
    query: str,
    agent_id: str = "default",
    limit: int = 5
) -> List[MemoryEntry]:
    """Quick recall function"""
    return get_enhanced_memory(agent_id).query_memories(
        MemoryQuery(text=query, limit=limit)
    )


def get_memory_stats(agent_id: str = "default") -> Dict[str, Any]:
    """Get memory statistics"""
    return get_enhanced_memory(agent_id).get_stats()


__all__ = [
    "MemoryType",
    "MemoryScope",
    "MemorySource",
    "MemoryEntry",
    "MemoryIndex",
    "MemoryQuery",
    "MemoryScore",
    "EnhancedMemory",
    "GitScopeDetector",
    "PathScopedRules",
    "HierarchicalMemoryManager",
    "ScopedMemory",
    "MemorySlice",
    "RecallFlow",
    "get_enhanced_memory",
    "create_memory_for_project",
    "remember",
    "recall",
    "get_memory_stats",
]
