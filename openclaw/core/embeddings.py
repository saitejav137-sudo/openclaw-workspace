"""
Embedding Providers for OpenClaw RAG

Multiple backends for generating text embeddings:
- SentenceTransformer (local, fast, default)
- OpenAI Embeddings (API-based, higher quality)
- Simple hash-based (fallback, no ML deps)

Usage:
    from core.embeddings import get_embedding_provider
    provider = get_embedding_provider()
    vector = provider.embed("Hello world")
"""

import hashlib
import math
import threading
from typing import List, Optional
from abc import ABC, abstractmethod
from functools import lru_cache

from .logger import get_logger

logger = get_logger("embeddings")


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._dimension = 384  # Updated when model loads

    def _load_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self._model_name)
                        self._dimension = self._model.get_sentence_embedding_dimension()
                        logger.info(
                            f"Loaded embedding model: {self._model_name} "
                            f"(dim={self._dimension})"
                        )
                    except ImportError:
                        logger.error(
                            "sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers"
                        )
                        raise
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}")
                        raise

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI API-based embedding provider.

    Model: text-embedding-3-small (1536 dimensions)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        import os
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._dimension = 1536
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai not installed. Install with: pip install openai")
        return self._client

    def embed(self, text: str) -> List[float]:
        client = self._get_client()
        response = client.embeddings.create(
            input=text,
            model=self._model
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self._model
        )
        return [d.embedding for d in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Fallback embedding provider using deterministic hashing.

    Not semantically meaningful but:
    - No ML dependencies
    - Deterministic (same text → same vector)
    - Fast
    - Good for testing
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        """Generate deterministic hash-based embedding."""
        # Use multiple hash rounds to fill the dimension
        vectors = []
        for i in range(self._dimension):
            h = hashlib.sha256(f"{text}:{i}".encode()).hexdigest()
            # Convert first 8 hex chars to float in [-1, 1]
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1
            vectors.append(val)

        # Normalize
        magnitude = math.sqrt(sum(v * v for v in vectors))
        if magnitude > 0:
            vectors = [v / magnitude for v in vectors]

        return vectors

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


# ============== Embedding Cache ==============

class EmbeddingCache:
    """
    LRU cache for embeddings to avoid recomputation.
    Thread-safe with configurable max size.
    """

    def __init__(self, provider: EmbeddingProvider, max_size: int = 10000):
        self._provider = provider
        self._max_size = max_size
        self._cache: dict = {}
        self._lock = threading.Lock()

    def embed(self, text: str) -> List[float]:
        """Get cached embedding or compute new one."""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Compute outside lock
        embedding = self._provider.embed(text)

        with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict oldest 10%
                keys_to_remove = list(self._cache.keys())[:self._max_size // 10]
                for k in keys_to_remove:
                    del self._cache[k]

            self._cache[cache_key] = embedding

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with caching."""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        with self._lock:
            for i, text in enumerate(texts):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = self._provider.embed_batch(uncached_texts)
            with self._lock:
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                    cache_key = hashlib.md5(text.encode()).hexdigest()
                    self._cache[cache_key] = emb
                    results[idx] = emb

        return results

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()


# ============== Global Provider ==============

_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider(
    backend: str = "auto",
    **kwargs
) -> EmbeddingProvider:
    """
    Get or create the global embedding provider.

    backend options:
    - "auto": Try sentence-transformers, fall back to hash
    - "sentence-transformer": Use sentence-transformers
    - "openai": Use OpenAI API
    - "hash": Use hash-based fallback
    """
    global _provider
    if _provider is not None:
        return _provider

    if backend == "auto":
        try:
            _provider = EmbeddingCache(SentenceTransformerProvider(**kwargs))
            logger.info("Using SentenceTransformer embeddings")
        except (ImportError, Exception) as e:
            logger.warning(f"SentenceTransformer unavailable ({e}), using hash fallback")
            _provider = EmbeddingCache(HashEmbeddingProvider(**kwargs))
    elif backend == "sentence-transformer":
        _provider = EmbeddingCache(SentenceTransformerProvider(**kwargs))
    elif backend == "openai":
        _provider = EmbeddingCache(OpenAIEmbeddingProvider(**kwargs))
    elif backend == "hash":
        _provider = EmbeddingCache(HashEmbeddingProvider(**kwargs))
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")

    return _provider


def reset_provider():
    """Reset global provider (for testing)."""
    global _provider
    _provider = None


__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
    "HashEmbeddingProvider",
    "EmbeddingCache",
    "get_embedding_provider",
    "reset_provider",
]
