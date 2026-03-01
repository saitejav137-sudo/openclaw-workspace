"""Tests for vector database"""

import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.storage.vector_db import (
    VectorDatabase,
    VectorEntry,
    SearchResult,
    DistanceMetric,
    simple_embed,
    get_vector_database,
    semantic_search,
    add_to_index,
)


class TestVectorDatabase(unittest.TestCase):
    """Test vector database"""

    def test_database_creation(self):
        """Test database creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDatabase("test", dimension=128, storage_dir=tmpdir)
            self.assertEqual(db.name, "test")
            self.assertEqual(db.dimension, 128)
            self.assertEqual(db.count(), 0)

    def test_add_entry(self):
        """Test adding entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDatabase("test", dimension=128, storage_dir=tmpdir)

            vector = [0.1] * 128
            entry_id = db.add("Test text", vector, {"tag": "test"})

            self.assertIsNotNone(entry_id)
            self.assertEqual(db.count(), 1)

    def test_get_entry(self):
        """Test getting entry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDatabase("test", dimension=128, storage_dir=tmpdir)

            vector = [0.1] * 128
            entry_id = db.add("Test text", vector)

            entry = db.get(entry_id)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.text, "Test text")

    def test_search(self):
        """Test similarity search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDatabase("test", dimension=128, storage_dir=tmpdir)

            # Add similar entries
            db.add("The cat is sleeping", [0.9] * 128)
            db.add("Python is a programming language", [0.1] * 128)
            db.add("A dog runs in the park", [0.85] * 128)

            # Search
            query = [0.9] * 128
            results = db.search(query, top_k=2)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].text, "The cat is sleeping")

    def test_delete_entry(self):
        """Test deleting entry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDatabase("test", dimension=128, storage_dir=tmpdir)

            vector = [0.1] * 128
            entry_id = db.add("Test text", vector)

            result = db.delete(entry_id)
            self.assertTrue(result)
            self.assertEqual(db.count(), 0)

    def test_stats(self):
        """Test database stats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VectorDatabase("test", dimension=128, storage_dir=tmpdir)

            vector = [0.1] * 128
            db.add("Test 1", vector)
            db.add("Test 2", vector)

            stats = db.get_stats()
            self.assertEqual(stats["count"], 2)
            self.assertEqual(stats["dimension"], 128)


class TestSimpleEmbed(unittest.TestCase):
    """Test simple embedding"""

    def test_embed_dimension(self):
        """Test embedding dimension"""
        embedding = simple_embed("test text", dimension=256)
        self.assertEqual(len(embedding), 256)

    def test_embed_normalized(self):
        """Test embedding is normalized"""
        embedding = simple_embed("test text", dimension=128)
        norm = sum(x**2 for x in embedding) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestQuickFunctions(unittest.TestCase):
    """Test quick helper functions"""

    def test_semantic_search(self):
        """Test quick semantic search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add some data
            db = get_vector_database("test_quick_search", dimension=128)
            db.storage_dir = tmpdir
            db.add("Python is awesome", [0.9] * 128)
            db.add("JavaScript is for web", [0.1] * 128)

            results = semantic_search("Python", dimension=128)
            self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
