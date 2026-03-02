"""
Tests for Enhanced Memory System
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
from pathlib import Path

sys.path.insert(0, 'openclaw')

from openclaw.core.enhanced_memory import (
    EnhancedMemory,
    MemoryType,
    MemorySource,
    MemoryQuery,
    MemoryConfig,
    GitScopeDetector,
    HierarchicalMemoryManager,
    ScopedMemory,
    MemorySlice,
)


# Test fixtures
@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory(temp_memory_dir):
    """Create an EnhancedMemory instance with temp directory."""
    return EnhancedMemory(
        agent_id="test_agent",
        memory_dir=temp_memory_dir,
        max_memories=100
    )


class TestMemoryConfig:
    """Test MemoryConfig constants."""

    def test_default_values(self):
        """Test default configuration values."""
        assert MemoryConfig.MAX_MEMORIES == 10000
        assert MemoryConfig.MAX_INDEX_LINES == 200
        assert MemoryConfig.CONSOLIDATION_THRESHOLD == 0.85
        assert MemoryConfig.EMBEDDING_DIM == 384

    def test_custom_values(self):
        """Test custom configuration."""
        # This tests that the class constants can be overridden if needed
        assert hasattr(MemoryConfig, 'DEFAULT_BASE_DIR')


class TestGitScopeDetector:
    """Test GitScopeDetector."""

    def test_get_repo_path(self):
        """Test getting git repository path."""
        # Should return None in test environment or actual path if in git repo
        result = GitScopeDetector.get_repo_path()
        # Just check it doesn't crash
        assert result is None or isinstance(result, str)

    def test_get_memory_path(self):
        """Test getting memory path."""
        path = GitScopeDetector.get_memory_path()
        assert isinstance(path, str)
        assert '.openclaw' in path


class TestEnhancedMemory:
    """Test EnhancedMemory core functionality."""

    def test_init(self, memory):
        """Test memory initialization."""
        assert memory.agent_id == "test_agent"
        assert len(memory._memories) == 0

    def test_add_memory(self, memory):
        """Test adding a memory."""
        memory_id = memory.add_memory(
            content="Test memory content",
            memory_type=MemoryType.EPISODIC,
            importance=0.7
        )
        assert memory_id is not None
        assert len(memory._memories) == 1

    def test_add_memory_duplicate(self, memory):
        """Test duplicate memory detection."""
        content = "Test duplicate content"
        id1 = memory.add_memory(content=content, memory_type=MemoryType.SEMANTIC)
        id2 = memory.add_memory(content=content, memory_type=MemoryType.SEMANTIC)
        # Should return same ID for duplicate
        assert id1 == id2

    def test_get_memory(self, memory):
        """Test retrieving a memory."""
        memory_id = memory.add_memory(
            content="Get test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8
        )
        retrieved = memory.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Get test"

    def test_get_memory_not_found(self, memory):
        """Test retrieving non-existent memory."""
        result = memory.get_memory("non_existent_id")
        assert result is None

    def test_query_memories(self, memory):
        """Test querying memories."""
        # Add some memories
        memory.add_memory("Python programming", MemoryType.SEMANTIC, 0.9)
        memory.add_memory("JavaScript web dev", MemoryType.SEMANTIC, 0.7)
        memory.add_memory("Fixed bug yesterday", MemoryType.EPISODIC, 0.8)

        results = memory.query_memories(MemoryQuery(text="programming", limit=5))
        assert len(results) > 0

    def test_query_memories_by_type(self, memory):
        """Test querying by memory type."""
        memory.add_memory("Semantic 1", MemoryType.SEMANTIC, 0.8)
        memory.add_memory("Semantic 2", MemoryType.SEMANTIC, 0.7)
        memory.add_memory("Episodic 1", MemoryType.EPISODIC, 0.8)

        results = memory.query_memories(
            MemoryQuery(text="test", limit=10, memory_type=MemoryType.SEMANTIC)
        )
        # All results should be semantic
        for r in results:
            assert r.memory_type == MemoryType.SEMANTIC

    def test_working_memory(self, memory):
        """Test working memory operations."""
        memory.update_working_memory("task", "testing")
        memory.update_working_memory("user", "admin")

        # Get single key
        task = memory.get_working_memory("task")
        assert task == "testing"

        # Get all
        all_working = memory.get_working_memory()
        assert "task" in all_working
        assert "user" in all_working

    def test_clear_working_memory(self, memory):
        """Test clearing working memory."""
        memory.update_working_memory("key1", "value1")
        memory.update_working_memory("key2", "value2")

        memory.clear_working_memory()
        assert len(memory._working_memory) == 0

    def test_get_stats(self, memory):
        """Test getting memory statistics."""
        memory.add_memory("Test 1", MemoryType.SEMANTIC, 0.8)
        memory.add_memory("Test 2", MemoryType.EPISODIC, 0.7)
        memory.add_memory("Test 3", MemoryType.PROCEDURAL, 0.6)

        stats = memory.get_stats()
        assert stats["total"] == 3
        assert "semantic" in stats["by_type"]
        assert "episodic" in stats["by_type"]
        assert "procedural" in stats["by_type"]

    def test_persistence(self, temp_memory_dir):
        """Test memory persistence to disk."""
        # Create and add memory
        memory1 = EnhancedMemory(agent_id="persist_test", memory_dir=temp_memory_dir)
        memory1.add_memory("Persistent memory", MemoryType.SEMANTIC, 0.9)

        # Create new instance - should load from disk
        memory2 = EnhancedMemory(agent_id="persist_test", memory_dir=temp_memory_dir)
        assert len(memory2._memories) == 1

    def test_delete_memory(self, memory):
        """Test deleting a memory."""
        memory_id = memory.add_memory("To delete", MemoryType.EPISODIC, 0.5)
        assert len(memory._memories) == 1

        # Delete via internal method
        del memory._memories[memory_id]
        memory._delete_memory_file(memory_id)
        assert len(memory._memories) == 0

    def test_compact(self, memory):
        """Test memory compaction."""
        # Add many memories
        for i in range(50):
            memory.add_memory(f"Memory {i}", MemoryType.EPISODIC, 0.5)

        # Compact
        result = memory.compact()
        assert "before" in result
        assert "after" in result


class TestHierarchicalMemory:
    """Test hierarchical memory scopes."""

    def test_scope(self, memory):
        """Test scoped memory."""
        hierarchical = HierarchicalMemoryManager(memory)
        scoped = hierarchical.scope("/project/alpha")

        assert scoped.scope == "/project/alpha"
        assert scoped.memory == memory

    def test_scope_add_memory(self, memory):
        """Test adding memory to scope."""
        hierarchical = HierarchicalMemoryManager(memory)
        scoped = hierarchical.scope("/agent/researcher")

        # Add memory to scope
        memory_id = scoped.add_memory(
            content="Research finding",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9
        )

        assert memory_id is not None

    def test_slice(self, memory):
        """Test memory slice."""
        hierarchical = HierarchicalMemoryManager(memory)
        slice_mem = hierarchical.slice(["/project", "/company"], read_only=False)

        assert slice_mem.scopes == ["/project", "/company"]
        assert slice_mem.read_only == False


class TestMemoryTypes:
    """Test memory type handling."""

    def test_all_memory_types(self, memory):
        """Test all memory types can be added."""
        types = [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
            MemoryType.WORKING,
        ]

        for mem_type in types:
            memory.add_memory(f"Test {mem_type.value}", mem_type, 0.5)

        stats = memory.get_stats()
        assert stats["total"] == 4


class TestMemorySource:
    """Test memory source handling."""

    def test_all_sources(self, memory):
        """Test all memory sources."""
        sources = [
            MemorySource.USER,
            MemorySource.AUTO,
            MemorySource.EXTRACTED,
            MemorySource.IMPORTED,
        ]

        for source in sources:
            memory.add_memory(
                f"Test {source.value}",
                MemoryType.SEMANTIC,
                0.5,
                source=source
            )

        # All should be stored
        assert len(memory._memories) == 4


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_query(self, memory):
        """Test querying with empty results."""
        results = memory.query_memories(MemoryQuery(text="nonexistent", limit=5))
        assert isinstance(results, list)

    def test_long_content(self, memory):
        """Test with very long content."""
        long_content = "x" * 100000  # 100k characters
        memory_id = memory.add_memory(long_content, MemoryType.SEMANTIC, 0.5)
        assert memory_id is not None

    def test_special_characters(self, memory):
        """Test with special characters."""
        content = "Test with emojis 🎉 and unicode: 你好"
        memory_id = memory.add_memory(content, MemoryType.SEMANTIC, 0.5)
        retrieved = memory.get_memory(memory_id)
        assert retrieved.content == content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
