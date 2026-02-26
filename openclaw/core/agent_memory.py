"""
Agent Memory System for OpenClaw

Persistent memory for AI agents with semantic search.
Based on 2026 trends: memory management for 24/7 proactive agents.
"""

import time
import json
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import os

import numpy as np

from .logger import get_logger

logger = get_logger("agent_memory")


class MemoryType(Enum):
    """Types of memory"""
    EPISODIC = "episodic"  # Specific events/experiences
    SEMANTIC = "semantic"  # Facts and knowledge
    WORKING = "working"    # Current context
    PROCEDURAL = "procedural"  # How to do things


@dataclass
class Memory:
    """A single memory entry"""
    id: str
    content: str
    memory_type: MemoryType
    importance: float = 0.5  # 0-1
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class MemoryQuery:
    """Query for memory search"""
    text: str
    memory_type: Optional[MemoryType] = None
    limit: int = 10
    min_importance: float = 0.0


class AgentMemory:
    """
    Persistent memory system for AI agents.
    Supports episodic, semantic, working, and procedural memory.
    """

    def __init__(
        self,
        agent_id: str = "default",
        memory_dir: str = "~/.openclaw/memory",
        max_memories: int = 10000,
        embedding_model: str = "simple"  # simple, openai, sentence-transformers
    ):
        self.agent_id = agent_id
        self.memory_dir = os.path.expanduser(memory_dir)
        self.max_memories = max_memories
        self.embedding_model = embedding_model

        # In-memory storage
        self._memories: Dict[str, Memory] = {}
        self._working_memory: Dict[str, Any] = {}

        # Create directory
        os.makedirs(self.memory_dir, exist_ok=True)

        # Load existing memories
        self._load_memories()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.sha256(
            f"{content}{time.time()}".encode()
        ).hexdigest()[:16]

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text"""
        if self.embedding_model == "simple":
            # Simple hash-based embedding
            hash_val = hash(text.encode())
            np.random.seed(hash_val % (2**32))
            return np.random.randn(128).tolist()

        # For other models, would integrate with OpenAI or sentence-transformers
        return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple bag-of-words embedding"""
        words = text.lower().split()
        embedding = np.zeros(128)

        for word in words:
            hash_val = hash(word)
            idx = hash_val % 128
            embedding[idx] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a new memory"""
        # Check for duplicate
        for mem in self._memories.values():
            if mem.content == content:
                mem.access_count += 1
                mem.last_accessed = time.time()
                return mem.id

        # Generate embedding
        embedding = self._compute_embedding(content)

        # Create memory
        memory = Memory(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Store
        self._memories[memory.id] = memory

        # Cleanup if needed
        if len(self._memories) > self.max_memories:
            self._cleanup()

        # Save to disk
        self._save_memory(memory)

        logger.debug(f"Added memory: {memory.id} ({memory_type.value})")
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory"""
        memory = self._memories.get(memory_id)

        if memory:
            memory.access_count += 1
            memory.last_accessed = time.time()

        return memory

    def query_memories(self, query: MemoryQuery) -> List[Memory]:
        """Query memories using semantic search"""
        results = []

        # Compute query embedding
        query_embedding = self._compute_embedding(query.text)

        # Search through memories
        for memory in self._memories.values():
            # Filter by type
            if query.memory_type and memory.memory_type != query.memory_type:
                continue

            # Filter by importance
            if memory.importance < query.min_importance:
                continue

            # Calculate similarity
            if memory.embedding:
                similarity = self._cosine_similarity(
                    query_embedding,
                    memory.embedding
                )

                # Update access
                memory.access_count += 1
                memory.last_accessed = time.time()

                results.append((memory, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return [m for m, _ in results[:query.limit]]

    def get_episodic(self, limit: int = 10) -> List[Memory]:
        """Get recent episodic memories"""
        episodic = [
            m for m in self._memories.values()
            if m.memory_type == MemoryType.EPISODIC
        ]
        episodic.sort(key=lambda x: x.timestamp, reverse=True)
        return episodic[:limit]

    def get_semantic(self, limit: int = 10) -> List[Memory]:
        """Get important semantic memories"""
        semantic = [
            m for m in self._memories.values()
            if m.memory_type == MemoryType.SEMANTIC
        ]
        semantic.sort(key=lambda x: x.importance, reverse=True)
        return semantic[:limit]

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

    def remember_action(
        self,
        action: str,
        result: str,
        context: Dict[str, Any] = None
    ):
        """Remember an action and its result (procedural memory)"""
        content = f"Action: {action} -> Result: {result}"
        self.add_memory(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            importance=0.7,
            metadata=context or {}
        )

    def get_recent_context(self, limit: int = 5) -> str:
        """Get recent memories as context string"""
        recent = self.get_episodic(limit)
        if not recent:
            return ""

        context_parts = []
        for mem in recent:
            timestamp = datetime.fromtimestamp(mem.timestamp).strftime("%H:%M:%S")
            context_parts.append(f"[{timestamp}] {mem.content}")

        return "\n".join(context_parts)

    def _cleanup(self):
        """Remove least important/accessed memories"""
        # Sort by importance and access count
        memories = list(self._memories.values())
        memories.sort(key=lambda x: (x.importance, x.access_count))

        # Remove 10% least important
        remove_count = len(memories) // 10
        for memory in memories[:remove_count]:
            del self._memories[memory.id]

        logger.info(f"Cleaned up {remove_count} memories")

    def _save_memory(self, memory: Memory):
        """Save memory to disk"""
        try:
            filepath = os.path.join(
                self.memory_dir,
                f"{self.agent_id}_{memory.id}.json"
            )

            # Convert to dict with JSON-serializable types
            data = {
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,  # Convert enum to string
                "importance": memory.importance,
                "timestamp": memory.timestamp,
                "embedding": memory.embedding,
                "metadata": memory.metadata,
                "access_count": memory.access_count,
                "last_accessed": memory.last_accessed
            }

            with open(filepath, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _load_memories(self):
        """Load memories from disk"""
        try:
            pattern = f"{self.agent_id}_*.json"
            memory_path = Path(self.memory_dir)

            for filepath in memory_path.glob(pattern):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        # Convert string back to enum
                        data["memory_type"] = MemoryType(data["memory_type"])
                        memory = Memory(**data)
                        self._memories[memory.id] = memory
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")

            logger.info(f"Loaded {len(self._memories)} memories")

        except Exception as e:
            logger.error(f"Failed to load memories: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        by_type = {}
        for mem_type in MemoryType:
            by_type[mem_type.value] = sum(
                1 for m in self._memories.values()
                if m.memory_type == mem_type
            )

        return {
            "total": len(self._memories),
            "by_type": by_type,
            "working_memory": len(self._working_memory),
            "max_memories": self.max_memories
        }


# Global memory instance
_memory_store: Dict[str, AgentMemory] = {}


def get_agent_memory(agent_id: str = "default") -> AgentMemory:
    """Get or create agent memory"""
    if agent_id not in _memory_store:
        _memory_store[agent_id] = AgentMemory(agent_id)
    return _memory_store[agent_id]


def remember_episodic(
    agent_id: str,
    content: str,
    importance: float = 0.5
) -> str:
    """Quick remember episodic memory"""
    return get_agent_memory(agent_id).add_memory(
        content, MemoryType.EPISODIC, importance
    )


def remember_fact(
    agent_id: str,
    fact: str,
    importance: float = 0.8
) -> str:
    """Quick remember semantic fact"""
    return get_agent_memory(agent_id).add_memory(
        fact, MemoryType.SEMANTIC, importance
    )


def recall(
    agent_id: str,
    query: str,
    limit: int = 5
) -> List[Memory]:
    """Quick recall memories"""
    return get_agent_memory(agent_id).query_memories(
        MemoryQuery(text=query, limit=limit)
    )


__all__ = [
    "MemoryType",
    "Memory",
    "MemoryQuery",
    "AgentMemory",
    "get_agent_memory",
    "remember_episodic",
    "remember_fact",
    "recall",
]
