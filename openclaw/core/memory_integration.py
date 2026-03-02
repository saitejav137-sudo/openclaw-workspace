"""
Memory Integration Module for OpenClaw

Integrates enhanced memory with the existing system:
- Connects EnhancedMemory to SmartContextManager
- Provides backward compatibility with AgentMemory
- Auto-initializes memory for agents
"""

import os
from typing import Optional, Dict, Any, List
from threading import Lock

from .enhanced_memory import (
    EnhancedMemory,
    get_enhanced_memory,
    create_memory_for_project,
    MemoryType,
    MemorySource,
    MemoryQuery,
    HierarchicalMemoryManager,
    ScopedMemory,
    MemorySlice,
    GitScopeDetector
)
from .smart_context import SmartContextManager
from .logger import get_logger

logger = get_logger("memory_integration")


class MemoryIntegration:
    """
    Integrates enhanced memory with OpenClaw's agent system.

    Features:
    - Auto-initializes memory for agents
    - Connects to SmartContextManager
    - Provides backward compatibility
    - Hierarchical scope management
    """

    _instance: Optional['MemoryIntegration'] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._agent_memories: Dict[str, EnhancedMemory] = {}
        self._context_managers: Dict[str, SmartContextManager] = {}
        self._hierarchical_managers: Dict[str, HierarchicalMemoryManager] = {}

        logger.info("Memory integration initialized")

    def get_agent_memory(
        self,
        agent_id: str = None,
        use_enhanced: bool = True
    ) -> EnhancedMemory:
        """Get memory for an agent"""
        if agent_id is None:
            agent_id = GitScopeDetector.get_repo_name() or "default"

        if agent_id not in self._agent_memories:
            self._agent_memories[agent_id] = get_enhanced_memory(agent_id)

        return self._agent_memories[agent_id]

    def get_context_manager(
        self,
        agent_id: str = None,
        max_tokens: int = 5000
    ) -> SmartContextManager:
        """Get SmartContextManager for an agent"""
        if agent_id is None:
            agent_id = GitScopeDetector.get_repo_name() or "default"

        if agent_id not in self._context_managers:
            self._context_managers[agent_id] = SmartContextManager(
                max_tokens=max_tokens,
                enable_rag=True,
                enable_learning=True
            )

        return self._context_managers[agent_id]

    def get_hierarchical_manager(
        self,
        agent_id: str = None
    ) -> HierarchicalMemoryManager:
        """Get hierarchical memory manager for an agent"""
        if agent_id is None:
            agent_id = GitScopeDetector.get_repo_name() or "default"

        if agent_id not in self._hierarchical_managers:
            memory = self.get_agent_memory(agent_id)
            self._hierarchical_managers[agent_id] = HierarchicalMemoryManager(memory)

        return self._hierarchical_managers[agent_id]

    def scope_memory(
        self,
        scope_path: str,
        agent_id: str = None
    ) -> ScopedMemory:
        """Get scoped memory view"""
        hierarchical = self.get_hierarchical_manager(agent_id)
        return hierarchical.scope(scope_path)

    def slice_memory(
        self,
        scopes: List[str],
        agent_id: str = None,
        read_only: bool = False
    ) -> MemorySlice:
        """Get memory slice across scopes"""
        hierarchical = self.get_hierarchical_manager(agent_id)
        return hierarchical.slice(scopes, read_only)

    def remember_for_agent(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        **kwargs
    ) -> str:
        """Remember something for an agent"""
        memory = self.get_agent_memory(agent_id)

        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            mem_type = MemoryType.EPISODIC

        return memory.add_memory(
            content=content,
            memory_type=mem_type,
            importance=importance,
            source=MemorySource.AUTO,
            **kwargs
        )

    def recall_for_agent(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[Any]:
        """Recall memories for an agent"""
        memory = self.get_agent_memory(agent_id)
        return memory.query_memories(
            MemoryQuery(text=query, limit=limit, **kwargs)
        )

    def get_agent_stats(self, agent_id: str = None) -> Dict[str, Any]:
        """Get memory stats for an agent"""
        memory = self.get_agent_memory(agent_id)
        return memory.get_stats()

    def compact_agent_memory(self, agent_id: str = None) -> Dict[str, int]:
        """Compact memory for an agent"""
        memory = self.get_agent_memory(agent_id)
        return memory.compact()


# ============== Global Instance ==============

_integration: Optional[MemoryIntegration] = None


def get_memory_integration() -> MemoryIntegration:
    """Get global memory integration instance"""
    global _integration
    if _integration is None:
        _integration = MemoryIntegration()
    return _integration


# ============== Convenience Functions ==============

def get_memory(agent_id: str = None) -> EnhancedMemory:
    """Get enhanced memory for an agent"""
    return get_memory_integration().get_agent_memory(agent_id)


def get_context(agent_id: str = None, max_tokens: int = 5000) -> SmartContextManager:
    """Get context manager for an agent"""
    return get_memory_integration().get_context_manager(agent_id, max_tokens)


def remember(
    content: str,
    agent_id: str = None,
    memory_type: str = "episodic",
    importance: float = 0.5,
    **kwargs
) -> str:
    """Quick remember function"""
    return get_memory_integration().remember_for_agent(
        agent_id or "default",
        content,
        memory_type,
        importance,
        **kwargs
    )


def recall(
    query: str,
    agent_id: str = None,
    limit: int = 5,
    **kwargs
) -> List[Any]:
    """Quick recall function"""
    return get_memory_integration().recall_for_agent(
        agent_id or "default",
        query,
        limit,
        **kwargs
    )


def memory_scope(scope_path: str, agent_id: str = None) -> ScopedMemory:
    """Get scoped memory view"""
    return get_memory_integration().scope_memory(scope_path, agent_id)


def memory_slice(
    scopes: List[str],
    agent_id: str = None,
    read_only: bool = False
) -> MemorySlice:
    """Get memory slice across scopes"""
    return get_memory_integration().slice_memory(scopes, agent_id, read_only)


__all__ = [
    "MemoryIntegration",
    "get_memory_integration",
    "get_memory",
    "get_context",
    "remember",
    "recall",
    "memory_scope",
    "memory_slice",
]
