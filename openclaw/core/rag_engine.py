"""
Upgraded RAG Engine for OpenClaw

Drop-in replacement for the existing rag_engine.py.
Key upgrades:
- Real semantic embeddings (via embeddings.py providers)
- ChromaDB support for persistent vector storage
- Auto-indexing of workspace files
- Better document chunking (recursive, respects headings)
- Metadata filtering
"""

import os
import time
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .logger import get_logger
from .embeddings import get_embedding_provider, EmbeddingProvider

logger = get_logger("rag")


@dataclass
class RAGConfig:
    """RAG configuration."""
    vector_db_backend: str = "memory"  # "memory", "chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "openclaw_rag"
    persist_directory: str = "~/.openclaw/vector_store"
    embedding_backend: str = "auto"  # "auto", "sentence-transformer", "openai", "hash"
    top_k: int = 5
    min_similarity: float = 0.3
    chunk_size: int = 500
    chunk_overlap: int = 50
    auto_index_paths: List[str] = field(default_factory=list)
    auto_index_extensions: List[str] = field(
        default_factory=lambda: [".md", ".txt", ".py", ".yaml", ".json"]
    )


@dataclass
class RetrievedContext:
    """Retrieved context from RAG."""
    text: str
    source: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============== Vector Store Backends ==============

class VectorStore:
    """Base vector store interface."""

    def add(self, doc_id: str, text: str, embedding: List[float], metadata: Dict) -> str:
        raise NotImplementedError

    def search(self, query_embedding: List[float], top_k: int, filters: Dict = None) -> List[Dict]:
        raise NotImplementedError

    def delete(self, doc_id: str):
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing and small datasets."""

    def __init__(self):
        self._docs: Dict[str, Dict] = {}

    def add(self, doc_id: str, text: str, embedding: List[float], metadata: Dict) -> str:
        self._docs[doc_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }
        return doc_id

    def search(self, query_embedding: List[float], top_k: int, filters: Dict = None) -> List[Dict]:
        results = []
        for doc_id, doc in self._docs.items():
            # Apply filters
            if filters:
                skip = False
                for k, v in filters.items():
                    if doc["metadata"].get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "similarity": similarity,
                "metadata": doc["metadata"]
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def delete(self, doc_id: str):
        self._docs.pop(doc_id, None)

    def count(self) -> int:
        return len(self._docs)

    def clear(self):
        self._docs.clear()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store for production use."""

    def __init__(
        self,
        collection_name: str = "openclaw_rag",
        host: str = "localhost",
        port: int = 8000,
        persist_directory: Optional[str] = None
    ):
        self._collection_name = collection_name
        self._client = None
        self._collection = None

        try:
            import chromadb

            if persist_directory:
                persist_dir = os.path.expanduser(persist_directory)
                os.makedirs(persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(path=persist_dir)
                logger.info(f"ChromaDB: persistent mode at {persist_dir}")
            else:
                try:
                    self._client = chromadb.HttpClient(host=host, port=port)
                    self._client.heartbeat()
                    logger.info(f"ChromaDB: connected to {host}:{port}")
                except Exception:
                    persist_dir = os.path.expanduser("~/.openclaw/vector_store")
                    os.makedirs(persist_dir, exist_ok=True)
                    self._client = chromadb.PersistentClient(path=persist_dir)
                    logger.info(f"ChromaDB: HTTP unavailable, using persistent at {persist_dir}")

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB collection '{collection_name}': {self._collection.count()} docs")

        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

    def add(self, doc_id: str, text: str, embedding: List[float], metadata: Dict) -> str:
        # ChromaDB doesn't allow None values in metadata
        clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[clean_metadata]
        )
        return doc_id

    def search(self, query_embedding: List[float], top_k: int, filters: Dict = None) -> List[Dict]:
        where = filters if filters else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        output = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # ChromaDB returns distance, not similarity
                output.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "similarity": similarity,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        return output

    def delete(self, doc_id: str):
        try:
            self._collection.delete(ids=[doc_id])
        except Exception:
            pass

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )


# ============== Upgraded RAG Engine ==============

class RAGEngine:
    """
    Upgraded Retrieval Augmented Generation engine.

    Key improvements over original:
    - Real semantic embeddings (not random hashes)
    - ChromaDB persistent storage
    - Auto-indexing of workspace files
    - Recursive chunking that respects document structure
    - Metadata filtering
    - Source deduplication
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

        # Initialize embedding provider
        self._embedder = get_embedding_provider(backend=self.config.embedding_backend)
        logger.info(f"RAG using embedding backend: {self._embedder.__class__.__name__}")

        # Initialize vector store
        if self.config.vector_db_backend == "chroma":
            self._store = ChromaVectorStore(
                collection_name=self.config.collection_name,
                host=self.config.chroma_host,
                port=self.config.chroma_port,
                persist_directory=self.config.persist_directory
            )
        else:
            self._store = InMemoryVectorStore()
            logger.info("RAG using in-memory vector store")

        # Track indexed files
        self._indexed_hashes: Dict[str, str] = {}

    def add_document(
        self,
        text: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add document to RAG index."""
        chunks = self._chunk_text(text)
        doc_ids = []

        # Batch embed for efficiency
        embeddings = self._embedder.embed_batch(chunks) if hasattr(self._embedder, 'embed_batch') else [
            self._embedder.embed(c) for c in chunks
        ]

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = hashlib.md5(f"{source}:{i}:{chunk[:50]}".encode()).hexdigest()[:12]

            self._store.add(
                doc_id=doc_id,
                text=chunk,
                embedding=embedding,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": time.time(),
                    **(metadata or {})
                }
            )
            doc_ids.append(doc_id)

        logger.info(f"Indexed '{source}': {len(chunks)} chunks")
        return doc_ids

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """Retrieve relevant context for a query."""
        top_k = top_k or self.config.top_k
        query_embedding = self._embedder.embed(query)

        results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get extra for filtering
            filters=filters
        )

        contexts = []
        seen_texts = set()

        for result in results:
            if result["similarity"] < self.config.min_similarity:
                continue

            # Deduplicate
            text_hash = hashlib.md5(result["text"].encode()).hexdigest()
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            contexts.append(RetrievedContext(
                text=result["text"],
                source=result["metadata"].get("source", "unknown"),
                similarity=result["similarity"],
                metadata=result["metadata"]
            ))

            if len(contexts) >= top_k:
                break

        return contexts

    def get_context_for_agent(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Get formatted context string for an agent prompt."""
        contexts = self.retrieve(query)

        if not contexts:
            return ""

        parts = ["## Retrieved Context:\n"]
        total_chars = 0

        for i, ctx in enumerate(contexts, 1):
            entry = (
                f"{i}. [{ctx.source}] (relevance: {ctx.similarity:.0%})\n"
                f"   {ctx.text}\n"
            )
            if total_chars + len(entry) > max_tokens * 4:  # ~4 chars per token
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)

    def auto_index_workspace(self, workspace_dir: str):
        """
        Auto-index all matching files in a workspace directory.
        Only re-indexes files that have changed since last index.
        """
        workspace = Path(workspace_dir).expanduser()
        if not workspace.exists():
            logger.warning(f"Workspace not found: {workspace}")
            return

        indexed = 0
        for ext in self.config.auto_index_extensions:
            for file_path in workspace.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    content_hash = hashlib.md5(content.encode()).hexdigest()

                    # Skip if unchanged
                    str_path = str(file_path)
                    if self._indexed_hashes.get(str_path) == content_hash:
                        continue

                    self.add_document(
                        text=content,
                        source=str_path,
                        metadata={
                            "type": ext.lstrip("."),
                            "filename": file_path.name,
                        }
                    )
                    self._indexed_hashes[str_path] = content_hash
                    indexed += 1

                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")

        logger.info(f"Auto-indexed {indexed} files from {workspace}")

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a source."""
        # For ChromaDB, we need to query and delete
        # For in-memory, iterate and delete
        deleted = 0
        try:
            query_emb = self._embedder.embed("delete query")
            results = self._store.search(query_emb, top_k=1000, filters={"source": source})
            for r in results:
                self._store.delete(r["id"])
                deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete source '{source}': {e}")
        return deleted

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with improved logic:
        - Respects markdown headings
        - Breaks at paragraph boundaries
        - Falls back to sentence boundaries
        """
        if len(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []

        # First try splitting by headings
        sections = self._split_by_headings(text)

        for section in sections:
            if len(section) <= self.config.chunk_size:
                if section.strip():
                    chunks.append(section.strip())
            else:
                # Split long sections by paragraphs
                chunks.extend(self._split_by_size(section))

        return chunks

    def _split_by_headings(self, text: str) -> List[str]:
        """Split text by markdown headings."""
        lines = text.split("\n")
        sections = []
        current = []

        for line in lines:
            if line.startswith("#") and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current))

        return sections if len(sections) > 1 else [text]

    def _split_by_size(self, text: str) -> List[str]:
        """Split text by size with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            if end < len(text):
                # Try to break at paragraph, then sentence
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.config.chunk_overlap

        return chunks

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        return {
            "total_documents": self._store.count(),
            "indexed_files": len(self._indexed_hashes),
            "backend": self.config.vector_db_backend,
            "embedding": self._embedder.__class__.__name__,
            "config": {
                "top_k": self.config.top_k,
                "chunk_size": self.config.chunk_size,
                "min_similarity": self.config.min_similarity,
            }
        }


# ============== Global Access ==============

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine(config: RAGConfig = None) -> RAGEngine:
    """Get global RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(config)
    return _rag_engine


def add_knowledge(text: str, source: str = "manual", metadata: Dict = None) -> List[str]:
    """Quick add knowledge to RAG."""
    return get_rag_engine().add_document(text, source, metadata)


def query_knowledge(query: str, top_k: int = 5) -> List[RetrievedContext]:
    """Quick query knowledge base."""
    return get_rag_engine().retrieve(query, top_k)


def get_agent_context(query: str) -> str:
    """Get formatted context for agent."""
    return get_rag_engine().get_context_for_agent(query)


__all__ = [
    "RAGConfig",
    "RetrievedContext",
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "RAGEngine",
    "get_rag_engine",
    "add_knowledge",
    "query_knowledge",
    "get_agent_context",
]
