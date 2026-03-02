# OpenClaw Memory System

A production-ready memory system for AI agents with 2026 best practices.

## Features

- **Git-scoped storage**: Memories organized by git repository
- **Memory types**: Episodic, Semantic, Procedural, Working
- **Composite scoring**: Semantic similarity + recency + importance
- **Path-scoped rules**: Conditional memory for specific file patterns
- **Hierarchical scopes**: Tree-based memory organization (CrewAI pattern)
- **CLI management**: Full CRUD operations via CLI

## Quick Start

### Python API

```python
from openclaw.core.enhanced_memory import (
    EnhancedMemory,
    MemoryType,
    MemorySource
)

# Create memory
memory = EnhancedMemory(agent_id="my_agent")

# Add memory
memory_id = memory.add_memory(
    content="Remember to run tests",
    memory_type=MemoryType.EPISODIC,
    importance=0.8
)

# Query memory
results = memory.query_memories(MemoryQuery(text="tests", limit=5))

# Working memory
memory.update_working_memory("task", "processing")
task = memory.get_working_memory("task")
```

### CLI Usage

```bash
# Add memory
python -m openclaw.core.memory_cli add "Remember to run tests" --type episodic

# List memories
python -m openclaw.core.memory_cli list --type semantic

# Search
python -m openclaw.core.memory_cli search "testing"

# Stats
python -m openclaw.core.memory_cli stats

# Compact
python -m openclaw.core.memory_cli compact
```

## Memory Types

| Type | Use Case |
|------|----------|
| EPISODIC | Specific events, past experiences |
| SEMANTIC | Facts, knowledge, preferences |
| PROCEDURAL | Action patterns, workflows |
| WORKING | Current session context |

## Composite Scoring

Memory relevance is calculated as:

```
score = 0.4 * semantic_similarity + 0.3 * recency + 0.3 * importance
```

Where:
- **semantic_similarity**: Cosine similarity of embeddings
- **recency**: Exponential decay (half-life: 7 days)
- **importance**: User-specified 0-1 score

## Storage

```
~/.openclaw/projects/<repo>/memory/
├── *.json           # Memory entries
├── MEMORY.md        # Index (loaded at startup)
└── rules/           # Path-scoped rules
    └── *.md
```

## Configuration

```python
from openclaw.core.enhanced_memory import MemoryConfig

# Customize
MemoryConfig.MAX_MEMORIES = 10000
MemoryConfig.EMBEDDING_DIM = 384
MemoryConfig.CONSOLIDATION_THRESHOLD = 0.85
```

## Exceptions

```python
from openclaw.core.exceptions import (
    MemoryError,
    MemoryNotFoundError,
    MemoryValidationError,
    EmbeddingError
)

try:
    memory.add_memory("test", MemoryType.SEMANTIC, 0.5)
except MemoryValidationError as e:
    print(f"Validation error: {e}")
```

## Testing

```bash
# Run tests
python -m pytest tests/test_enhanced_memory.py -v

# Run specific test
python -m pytest tests/test_enhanced_memory.py::TestEnhancedMemory::test_add_memory -v
```

---

*Built for production AI agents*
