"""
RAG (Retrieval Augmented Generation) Integration for OpenClaw Agents

Connects vector database to agents for context-aware responses.
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from core.logger import get_logger
from core.agent_memory import AgentMemory, get_agent_memory
from storage.vector_db import VectorDatabase, get_vector_database, simple_embed

logger = get_logger("rag")


@dataclass
class RAGConfig:
    """RAG configuration"""
    vector_db_name: str = "rag"
    vector_dimension: int = 384
    memory_agent_id: str = "rag_agent"
    top_k: int = 5
    min_similarity: float = 0.3
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class RetrievedContext:
    """Retrieved context from RAG"""
    text: str
    source: str
    similarity: float
    metadata: Dict[str, Any]


class RAGEngine:
    """
    Retrieval Augmented Generation engine.

    Features:
    - Document chunking
    - Vector embedding storage
    - Similarity search
    - Context retrieval for agents
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vector_db = get_vector_database(
            self.config.vector_db_name,
            self.config.vector_dimension
        )
        self.memory = get_agent_memory(self.config.memory_agent_id)

    def add_document(
        self,
        text: str,
        source: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add document to RAG index"""
        # Chunk text if too long
        chunks = self._chunk_text(text)

        entry_ids = []
        for i, chunk in enumerate(chunks):
            # Create embedding
            vector = simple_embed(chunk, self.config.vector_dimension)

            # Add to vector DB
            doc_id = self.vector_db.add(
                text=chunk,
                vector=vector,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
            )
            entry_ids.append(doc_id)

            # Also add to agent memory
            self.memory.add_memory(
                content=f"[{source}] {chunk}",
                memory_type=self.memory._memories.values().__iter__().__next__().__class__,
                importance=0.6,
                metadata=metadata
            )

        logger.info(f"Added document to RAG: {source} ({len(chunks)} chunks)")
        return entry_ids[0] if entry_ids else None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Dict[str, Any] = None
    ) -> List[RetrievedContext]:
        """Retrieve relevant context for query"""
        top_k = top_k or self.config.top_k

        # Create query embedding
        query_vector = simple_embed(query, self.config.vector_dimension)

        # Search vector DB
        results = self.vector_db.search(
            query_vector,
            top_k=top_k * 2,  # Get more to filter
            filter_metadata=filters
        )

        # Filter by similarity
        contexts = []
        for result in results:
            if result.score >= self.config.min_similarity:
                contexts.append(RetrievedContext(
                    text=result.text,
                    source=result.metadata.get("source", "unknown"),
                    similarity=result.score,
                    metadata=result.metadata
                ))

        return contexts[:top_k]

    def get_context_for_agent(
        self,
        query: str,
        include_memory: bool = True
    ) -> str:
        """Get formatted context string for agent"""
        contexts = self.retrieve(query)

        if not contexts:
            return ""

        # Format context
        context_parts = ["## Retrieved Context:\n"]
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"{i}. [{ctx.source}] (similarity: {ctx.similarity:.2f})\n"
                f"   {ctx.text}\n"
            )

        # Include recent memory if requested
        if include_memory:
            recent = self.memory.get_recent_context(limit=3)
            if recent:
                context_parts.append("\n## Recent Memory:\n")
                context_parts.append(recent)

        return "\n".join(context_parts)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                for sep in [". ", "! ", "? ", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.config.chunk_overlap

        return chunks

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a source"""
        deleted = 0
        # This would require iterating through vector DB
        # For now, return 0 - implement with more efficient method
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics"""
        return {
            "total_vectors": self.vector_db.count(),
            "config": {
                "top_k": self.config.top_k,
                "chunk_size": self.config.chunk_size,
                "dimension": self.config.vector_dimension
            }
        }


# Built-in document loaders

class DocumentLoader:
    """Base class for document loaders"""

    def load(self, path: str) -> str:
        """Load document content"""
        raise NotImplementedError


class TextLoader(DocumentLoader):
    """Load plain text files"""

    def load(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


class MarkdownLoader(DocumentLoader):
    """Load markdown files"""

    def load(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


# Global RAG engine
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine(config: RAGConfig = None) -> RAGEngine:
    """Get global RAG engine"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(config)
    return _rag_engine


def add_knowledge(text: str, source: str = "manual", metadata: Dict = None):
    """Quick add knowledge to RAG"""
    return get_rag_engine().add_document(text, source, metadata)


def query_knowledge(query: str, top_k: int = 5) -> List[RetrievedContext]:
    """Quick query knowledge base"""
    return get_rag_engine().retrieve(query, top_k)


def get_agent_context(query: str) -> str:
    """Get formatted context for agent"""
    return get_rag_engine().get_context_for_agent(query)


__all__ = [
    "RAGConfig",
    "RetrievedContext",
    "RAGEngine",
    "DocumentLoader",
    "TextLoader",
    "MarkdownLoader",
    "get_rag_engine",
    "add_knowledge",
    "query_knowledge",
    "get_agent_context",
]
