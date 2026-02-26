"""
Vector Database for OpenClaw

Simple in-memory vector database for semantic search and RAG.
Supports embeddings, similarity search, and persistence.
"""

import time
import json
import os
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib

import numpy as np

from core.logger import get_logger

logger = get_logger("vector_db")


class DistanceMetric(Enum):
    """Distance metrics for similarity search"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@dataclass
class VectorEntry:
    """A vector entry with metadata"""
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """Search result with score"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorDatabase:
    """
    Simple vector database for semantic search.

    Features:
    - In-memory storage with disk persistence
    - Multiple distance metrics
    - Hierarchical indexing for fast search
    - Batch operations
    """

    def __init__(
        self,
        name: str = "default",
        dimension: int = 384,
        metric: DistanceMetric = DistanceMetric.COSINE,
        storage_dir: str = "~/.openclaw/vectors"
    ):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.storage_dir = os.path.expanduser(storage_dir)

        # In-memory storage
        self.entries: Dict[str, VectorEntry] = {}
        self._index: Dict[str, np.ndarray] = {}

        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load existing data
        self._load()

    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(a, b))

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        return float(np.linalg.norm(a - b))

    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Manhattan distance"""
        return float(np.sum(np.abs(a - b)))

    def _calculate_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate similarity based on metric"""
        if self.metric == DistanceMetric.COSINE:
            return self._cosine_similarity(self._normalize(a), self._normalize(b))
        elif self.metric == DistanceMetric.EUCLIDEAN:
            return -self._euclidean_distance(a, b)  # Negative for similarity
        elif self.metric == DistanceMetric.MANHATTAN:
            return -self._manhattan_distance(a, b)  # Negative for similarity
        return 0.0

    def add(
        self,
        text: str,
        vector: List[float],
        metadata: Dict[str, Any] = None,
        id: str = None
    ) -> str:
        """Add a vector entry"""
        # Validate dimension
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dimension}")

        # Generate or use provided ID
        entry_id = id or self._generate_id(text)

        # Create entry
        entry = VectorEntry(
            id=entry_id,
            vector=vector,
            text=text,
            metadata=metadata or {}
        )

        # Store
        self.entries[entry_id] = entry
        self._index[entry_id] = np.array(vector)

        # Save to disk
        self._save_entry(entry)

        logger.debug(f"Added vector: {entry_id}")
        return entry_id

    def get(self, id: str) -> Optional[VectorEntry]:
        """Get entry by ID"""
        return self.entries.get(id)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        if not self.entries:
            return []

        query = np.array(query_vector)
        results = []

        for entry_id, entry in self.entries.items():
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    entry.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            # Calculate similarity
            score = self._calculate_similarity(query, entry.vector)

            results.append(SearchResult(
                id=entry_id,
                text=entry.text,
                score=score,
                metadata=entry.metadata
            ))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def delete(self, id: str) -> bool:
        """Delete entry by ID"""
        if id in self.entries:
            del self.entries[id]
            if id in self._index:
                del self._index[id]
            self._delete_entry(id)
            return True
        return False

    def count(self) -> int:
        """Get total number of entries"""
        return len(self.entries)

    def clear(self):
        """Clear all entries"""
        self.entries.clear()
        self._index.clear()

    def _save_entry(self, entry: VectorEntry):
        """Save entry to disk"""
        try:
            filepath = os.path.join(self.storage_dir, f"{self.name}_{entry.id}.json")
            with open(filepath, 'w') as f:
                json.dump(asdict(entry), f)
        except Exception as e:
            logger.error(f"Failed to save entry: {e}")

    def _delete_entry(self, id: str):
        """Delete entry from disk"""
        try:
            filepath = os.path.join(self.storage_dir, f"{self.name}_{id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to delete entry: {e}")

    def _load(self):
        """Load entries from disk"""
        try:
            pattern = f"{self.name}_*.json"
            for filepath in Path(self.storage_dir).glob(pattern):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        entry = VectorEntry(**data)
                        self.entries[entry.id] = entry
                        self._index[entry.id] = np.array(entry.vector)
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")

            logger.info(f"Loaded {len(self.entries)} vector entries")

        except Exception as e:
            logger.error(f"Failed to load vectors: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "count": len(self.entries),
            "metric": self.metric.value,
            "storage_dir": self.storage_dir
        }

    def export_json(self) -> str:
        """Export all entries as JSON"""
        return json.dumps([
            asdict(entry)
            for entry in self.entries.values()
        ], indent=2)

    def import_json(self, json_data: str):
        """Import entries from JSON"""
        try:
            data = json.loads(json_data)
            for item in data:
                entry = VectorEntry(**item)
                self.entries[entry.id] = entry
                self._index[entry.id] = np.array(entry.vector)
            logger.info(f"Imported {len(data)} entries")
        except Exception as e:
            logger.error(f"Failed to import: {e}")


# Simple embedding function (bag of words)
def simple_embed(text: str, dimension: int = 384) -> List[float]:
    """Simple bag-of-words embedding"""
    words = text.lower().split()
    embedding = np.zeros(dimension)

    for word in words:
        hash_val = hash(word)
        idx = hash_val % dimension
        embedding[idx] += 1

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


# Global database instances
_databases: Dict[str, VectorDatabase] = {}


def get_vector_database(
    name: str = "default",
    dimension: int = 384,
    metric: DistanceMetric = DistanceMetric.COSINE
) -> VectorDatabase:
    """Get or create a vector database"""
    if name not in _databases:
        _databases[name] = VectorDatabase(name, dimension, metric)
    return _databases[name]


def semantic_search(
    query: str,
    top_k: int = 5,
    dimension: int = 384
) -> List[SearchResult]:
    """Quick semantic search"""
    # Get or create default database
    db = get_vector_database(dimension=dimension)

    # Embed query
    query_vector = simple_embed(query, dimension)

    # Search
    return db.search(query_vector, top_k)


def add_to_index(
    text: str,
    metadata: Dict[str, Any] = None,
    dimension: int = 384
) -> str:
    """Quick add to index"""
    db = get_vector_database(dimension=dimension)
    vector = simple_embed(text, dimension)
    return db.add(text, vector, metadata)


__all__ = [
    "DistanceMetric",
    "VectorEntry",
    "SearchResult",
    "VectorDatabase",
    "simple_embed",
    "get_vector_database",
    "semantic_search",
    "add_to_index",
]
