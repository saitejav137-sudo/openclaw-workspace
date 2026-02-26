"""Tests for agent memory system"""

import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.agent_memory import (
    AgentMemory,
    MemoryType,
    get_agent_memory,
    remember_episodic,
    remember_fact,
    recall
)


class TestAgentMemory(unittest.TestCase):
    """Test agent memory system"""

    def test_add_memory(self):
        """Test adding memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = AgentMemory("test_agent", memory_dir=tmpdir)

            # Add episodic memory
            mem_id = memory.add_memory(
                "Saw a cat on the screen",
                MemoryType.EPISODIC,
                importance=0.7
            )

            self.assertIsNotNone(mem_id)
            self.assertEqual(len(memory._memories), 1)

    def test_get_memory(self):
        """Test retrieving memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = AgentMemory("test_agent", memory_dir=tmpdir)

            mem_id = memory.add_memory("Test memory", MemoryType.EPISODIC)
            retrieved = memory.get_memory(mem_id)

            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.content, "Test memory")

    def test_query_memories(self):
        """Test querying memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = AgentMemory("test_agent", memory_dir=tmpdir)

            # Add multiple memories
            memory.add_memory("The cat is sleeping", MemoryType.EPISODIC)
            memory.add_memory("Python is a programming language", MemoryType.SEMANTIC)
            memory.add_memory("User clicked the button", MemoryType.EPISODIC)

            # Query
            from openclaw.core.agent_memory import MemoryQuery
            results = memory.query_memories(
                MemoryQuery(text="cat", limit=10)
            )

            self.assertTrue(len(results) > 0)

    def test_working_memory(self):
        """Test working memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = AgentMemory("test_agent", memory_dir=tmpdir)

            # Set working memory
            memory.update_working_memory("screen_state", {"x": 100, "y": 200})
            memory.update_working_memory("last_action", "click")

            # Get working memory
            state = memory.get_working_memory("screen_state")
            self.assertEqual(state, {"x": 100, "y": 200})

            all_wm = memory.get_working_memory()
            self.assertIn("screen_state", all_wm)

    def test_remember_action(self):
        """Test remembering actions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = AgentMemory("test_agent", memory_dir=tmpdir)

            memory.remember_action(
                "click_button",
                "Button clicked successfully",
                {"x": 100, "y": 200}
            )

            # Verify memory was added
            self.assertEqual(len(memory._memories), 1)

    def test_get_stats(self):
        """Test memory statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = AgentMemory("test_agent", memory_dir=tmpdir)

            memory.add_memory("Memory 1", MemoryType.EPISODIC)
            memory.add_memory("Memory 2", MemoryType.SEMANTIC)

            stats = memory.get_stats()
            self.assertEqual(stats["total"], 2)
            self.assertIn("episodic", stats["by_type"])


class TestQuickFunctions(unittest.TestCase):
    """Test quick helper functions"""

    def test_remember_episodic(self):
        """Test quick episodic memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique agent ID for test
            mem_id = remember_episodic("test_quick", "Quick memory test")
            self.assertIsNotNone(mem_id)

    def test_remember_fact(self):
        """Test quick fact memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_id = remember_fact("test_quick", "The sky is blue")
            self.assertIsNotNone(mem_id)

    def test_recall(self):
        """Test quick recall"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add some memories first
            remember_episodic("test_recall", "Python is awesome")
            remember_fact("test_recall", "Python is a language")

            # Recall
            results = recall("test_recall", "Python")
            # May or may not find results depending on embedding
            self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
