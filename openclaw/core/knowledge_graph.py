"""
Knowledge Graph Builder for OpenClaw

Builds structured knowledge graphs from unstructured data:
- Entity extraction and relationship mapping
- Graph traversal and querying
- Reasoning over connected knowledge
- Persistent storage with JSON serialization
"""

import time
import json
import hashlib
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import os

from .logger import get_logger

logger = get_logger("knowledge_graph")


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: str  # person, tool, concept, file, url, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = ""  # Where this entity was extracted from


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relation_type: str  # "uses", "depends_on", "created_by", etc.
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class KnowledgeGraph:
    """
    In-memory knowledge graph with persistent storage.

    Usage:
        kg = KnowledgeGraph()

        # Add entities
        kg.add_entity("python", "tool", {"version": "3.10"})
        kg.add_entity("data_analysis", "task")

        # Add relationships
        kg.add_relationship("data_analysis", "python", "requires")

        # Query
        related = kg.get_related("python")
        path = kg.find_path("data_analysis", "python")
    """

    def __init__(self, storage_path: str = "~/.openclaw/knowledge_graph"):
        self.storage_path = os.path.expanduser(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # entity_id -> [rel_ids]
        self._reverse_adj: Dict[str, List[str]] = defaultdict(list)  # target -> [rel_ids]
        self._lock = threading.Lock()

        self._load()

    def _gen_id(self, *parts: str) -> str:
        """Generate a deterministic ID."""
        return hashlib.sha256("_".join(parts).encode()).hexdigest()[:12]

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Dict[str, Any] = None,
        source: str = ""
    ) -> Entity:
        """Add or update an entity."""
        entity_id = self._gen_id(name, entity_type)

        with self._lock:
            if entity_id in self._entities:
                # Update existing
                entity = self._entities[entity_id]
                if properties:
                    entity.properties.update(properties)
                entity.updated_at = time.time()
            else:
                entity = Entity(
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    properties=properties or {},
                    source=source
                )
                self._entities[entity_id] = entity

        logger.debug(f"Entity: {name} ({entity_type})")
        return entity

    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        weight: float = 1.0,
        properties: Dict = None,
        source_type: str = "concept",
        target_type: str = "concept"
    ) -> Optional[Relationship]:
        """Add a relationship between two entities."""
        # Ensure entities exist
        source_id = self._gen_id(source_name, source_type)
        target_id = self._gen_id(target_name, target_type)

        with self._lock:
            if source_id not in self._entities:
                self._entities[source_id] = Entity(
                    id=source_id, name=source_name, entity_type=source_type
                )
            if target_id not in self._entities:
                self._entities[target_id] = Entity(
                    id=target_id, name=target_name, entity_type=target_type
                )

            rel_id = self._gen_id(source_id, target_id, relation_type)

            rel = Relationship(
                id=rel_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight,
                properties=properties or {}
            )

            self._relationships[rel_id] = rel
            self._adjacency[source_id].append(rel_id)
            self._reverse_adj[target_id].append(rel_id)

        logger.debug(f"Relationship: {source_name} --{relation_type}--> {target_name}")
        return rel

    def get_entity(self, name: str, entity_type: str = "concept") -> Optional[Entity]:
        """Get an entity by name and type."""
        entity_id = self._gen_id(name, entity_type)
        with self._lock:
            return self._entities.get(entity_id)

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        with self._lock:
            return self._entities.get(entity_id)

    def get_related(
        self,
        entity_name: str,
        entity_type: str = "concept",
        relation_type: str = None,
        direction: str = "outgoing"  # outgoing, incoming, both
    ) -> List[Tuple[Entity, Relationship]]:
        """Get entities related to a given entity."""
        entity_id = self._gen_id(entity_name, entity_type)
        results = []

        with self._lock:
            rel_ids = set()
            if direction in ("outgoing", "both"):
                rel_ids.update(self._adjacency.get(entity_id, []))
            if direction in ("incoming", "both"):
                rel_ids.update(self._reverse_adj.get(entity_id, []))

            for rel_id in rel_ids:
                rel = self._relationships.get(rel_id)
                if not rel:
                    continue
                if relation_type and rel.relation_type != relation_type:
                    continue

                # Get the "other" entity
                other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                other_entity = self._entities.get(other_id)
                if other_entity:
                    results.append((other_entity, rel))

        return results

    def find_path(
        self,
        source_name: str,
        target_name: str,
        source_type: str = "concept",
        target_type: str = "concept",
        max_depth: int = 5
    ) -> Optional[List[Entity]]:
        """Find shortest path between two entities using BFS."""
        source_id = self._gen_id(source_name, source_type)
        target_id = self._gen_id(target_name, target_type)

        with self._lock:
            if source_id not in self._entities or target_id not in self._entities:
                return None

            # BFS
            from collections import deque
            queue = deque([(source_id, [source_id])])
            visited = {source_id}

            while queue:
                current, path = queue.popleft()

                if current == target_id:
                    return [self._entities[eid] for eid in path]

                if len(path) >= max_depth:
                    continue

                # Explore neighbors
                for rel_id in self._adjacency.get(current, []):
                    rel = self._relationships.get(rel_id)
                    if rel and rel.target_id not in visited:
                        visited.add(rel.target_id)
                        queue.append((rel.target_id, path + [rel.target_id]))

        return None

    def query(
        self,
        entity_type: str = None,
        relation_type: str = None,
        limit: int = 50
    ) -> List[Entity]:
        """Query entities by type."""
        with self._lock:
            entities = list(self._entities.values())

            if entity_type:
                entities = [e for e in entities if e.entity_type == entity_type]

            entities.sort(key=lambda e: e.updated_at, reverse=True)
            return entities[:limit]

    def search(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name (case-insensitive)."""
        query_lower = query.lower()
        with self._lock:
            matching = [
                e for e in self._entities.values()
                if query_lower in e.name.lower()
            ]
            matching.sort(key=lambda e: e.updated_at, reverse=True)
            return matching[:limit]

    def get_neighbors(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get entity neighborhood (subgraph)."""
        with self._lock:
            entities = {}
            relationships = []
            to_visit = [(entity_id, 0)]
            visited = set()

            while to_visit:
                current_id, current_depth = to_visit.pop(0)
                if current_id in visited or current_depth > depth:
                    continue
                visited.add(current_id)

                entity = self._entities.get(current_id)
                if entity:
                    entities[current_id] = {
                        "name": entity.name,
                        "type": entity.entity_type
                    }

                for rel_id in self._adjacency.get(current_id, []):
                    rel = self._relationships.get(rel_id)
                    if rel:
                        relationships.append({
                            "from": rel.source_id,
                            "to": rel.target_id,
                            "type": rel.relation_type
                        })
                        if current_depth < depth:
                            to_visit.append((rel.target_id, current_depth + 1))

            return {"entities": entities, "relationships": relationships}

    def remove_entity(self, name: str, entity_type: str = "concept"):
        """Remove an entity and its relationships."""
        entity_id = self._gen_id(name, entity_type)
        with self._lock:
            if entity_id not in self._entities:
                return

            # Remove relationships
            for rel_id in list(self._adjacency.get(entity_id, [])):
                self._relationships.pop(rel_id, None)
            for rel_id in list(self._reverse_adj.get(entity_id, [])):
                self._relationships.pop(rel_id, None)

            self._adjacency.pop(entity_id, None)
            self._reverse_adj.pop(entity_id, None)
            del self._entities[entity_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            entity_types = defaultdict(int)
            for e in self._entities.values():
                entity_types[e.entity_type] += 1

            rel_types = defaultdict(int)
            for r in self._relationships.values():
                rel_types[r.relation_type] += 1

            return {
                "total_entities": len(self._entities),
                "total_relationships": len(self._relationships),
                "entity_types": dict(entity_types),
                "relationship_types": dict(rel_types)
            }

    def save(self):
        """Save graph to disk."""
        try:
            filepath = os.path.join(self.storage_path, "graph.json")
            data = {
                "entities": {
                    eid: {
                        "id": e.id, "name": e.name, "entity_type": e.entity_type,
                        "properties": e.properties, "source": e.source,
                        "created_at": e.created_at, "updated_at": e.updated_at
                    }
                    for eid, e in self._entities.items()
                },
                "relationships": {
                    rid: {
                        "id": r.id, "source_id": r.source_id, "target_id": r.target_id,
                        "relation_type": r.relation_type, "weight": r.weight,
                        "properties": r.properties, "created_at": r.created_at
                    }
                    for rid, r in self._relationships.items()
                },
                "saved_at": time.time()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved graph: {len(self._entities)} entities, {len(self._relationships)} rels")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def _load(self):
        """Load graph from disk."""
        try:
            filepath = os.path.join(self.storage_path, "graph.json")
            if not os.path.exists(filepath):
                return

            with open(filepath, 'r') as f:
                data = json.load(f)

            for eid, edata in data.get("entities", {}).items():
                self._entities[eid] = Entity(**edata)

            for rid, rdata in data.get("relationships", {}).items():
                rel = Relationship(**rdata)
                self._relationships[rid] = rel
                self._adjacency[rel.source_id].append(rid)
                self._reverse_adj[rel.target_id].append(rid)

            logger.info(f"Loaded graph: {len(self._entities)} entities")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")


# ============== Global Instance ==============

_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get global knowledge graph."""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph()
    return _graph


__all__ = [
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "get_knowledge_graph",
]
