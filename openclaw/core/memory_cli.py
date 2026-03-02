"""
Memory CLI for OpenClaw

Provides command-line interface for memory management:
- List memories
- Search memories
- Add/edit/delete memories
- View and toggle memory settings
- Export/import memories
- Memory statistics

Inspired by Claude Code's /memory command
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

from .enhanced_memory import (
    EnhancedMemory,
    get_enhanced_memory,
    create_memory_for_project,
    MemoryType,
    MemorySource,
    MemoryQuery,
    GitScopeDetector
)
from .logger import get_logger

logger = get_logger("memory_cli")


class MemoryCLI:
    """Command-line interface for memory management"""

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or GitScopeDetector.get_repo_name() or "default"
        self.memory = get_enhanced_memory(self.agent_id)

    def list_memories(
        self,
        memory_type: str = None,
        scope: str = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List memories with optional filters"""
        memories = list(self.memory._memories.values())

        # Filter by type
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
                memories = [m for m in memories if m.memory_type == mem_type]
            except ValueError:
                logger.warning(f"Unknown memory type: {memory_type}")

        # Filter by scope
        if scope:
            memories = [m for m in memories if m.scope.startswith(scope)]

        # Sort by timestamp
        memories.sort(key=lambda m: m.timestamp, reverse=True)

        return [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "scope": m.scope,
                "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                "importance": m.importance,
                "access_count": m.access_count,
                "timestamp": m.timestamp,
                "source": m.source.value
            }
            for m in memories[:limit]
        ]

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: str = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search memories by query"""
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                logger.warning(f"Unknown memory type: {memory_type}")

        results = self.memory.query_memories(
            MemoryQuery(
                text=query,
                limit=limit,
                memory_type=mem_type,
                min_importance=min_importance
            )
        )

        return [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                "importance": m.importance,
                "score": getattr(m, '_score', None),
                "timestamp": m.timestamp
            }
            for m in results
        ]

    def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        scope: str = "/project",
        tags: List[str] = None,
        private: bool = False
    ) -> str:
        """Add a new memory"""
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            logger.warning(f"Unknown memory type: {memory_type}, using EPISODIC")
            mem_type = MemoryType.EPISODIC

        return self.memory.add_memory(
            content=content,
            memory_type=mem_type,
            importance=importance,
            scope=scope,
            tags=tags or [],
            private=private,
            source=MemorySource.USER
        )

    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        tags: List[str] = None
    ) -> bool:
        """Update an existing memory"""
        memory = self.memory.get_memory(memory_id)

        if not memory:
            logger.error(f"Memory not found: {memory_id}")
            return False

        if content:
            memory.content = content
            memory.version += 1

        if importance is not None:
            memory.importance = importance

        if tags:
            memory.tags = tags

        self.memory._save_memory(memory)
        logger.info(f"Updated memory: {memory_id}")
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id not in self.memory._memories:
            logger.error(f"Memory not found: {memory_id}")
            return False

        del self.memory._memories[memory_id]
        self.memory._delete_memory_file(memory_id)

        logger.info(f"Deleted memory: {memory_id}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.memory.get_stats()

    def get_index(self, max_lines: int = 50) -> str:
        """Get memory index content"""
        return self.memory.get_index_content(max_lines)

    def compact(self) -> Dict[str, int]:
        """Compact memory"""
        return self.memory.compact()

    def consolidate(self, threshold: float = 0.85):
        """Consolidate similar memories"""
        self.memory.consolidate(threshold)
        return {"status": "consolidated"}

    def export_memories(self, filepath: str = None):
        """Export memories to file"""
        if not filepath:
            filepath = f"openclaw_memory_export_{int(os.times().elapsed * 1000)}.json"

        self.memory.export(filepath)
        return {"filepath": filepath}

    def import_memories(self, filepath: str) -> int:
        """Import memories from file"""
        return self.memory.import_memories(filepath)

    def add_rule(
        self,
        name: str,
        content: str,
        paths: List[str] = None
    ):
        """Add a path-scoped rule"""
        self.memory.add_rule(name, content, paths)
        return {"status": "added", "name": name}

    def get_rules_for_file(self, filepath: str) -> List[str]:
        """Get rules applicable to a file"""
        return self.memory.get_rules_for_path(filepath)

    def synthesize_lessons(self, session_count: int = 10) -> str:
        """Synthesize lessons from past sessions"""
        return self.memory.synthesize_lessons(session_count)

    def get_working_memory(self) -> Dict[str, Any]:
        """Get current working memory"""
        return self.memory.get_working_memory()

    def clear_working_memory(self):
        """Clear working memory"""
        self.memory.clear_working_memory()
        return {"status": "cleared"}


# ============== CLI Entry Points ==============

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OpenClaw Memory CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List memories
  memory-cli list --type episodic --limit 20

  # Search memories
  memory-cli search "error handling" --type semantic

  # Add memory
  memory-cli add "Always use dependency injection" --type semantic --importance 0.8

  # Get stats
  memory-cli stats

  # Export memories
  memory-cli export ~/backups/memories.json

  # Compact memory
  memory-cli compact
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--type", help="Filter by memory type")
    list_parser.add_argument("--scope", help="Filter by scope")
    list_parser.add_argument("--limit", type=int, default=20, help="Limit results")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--type", help="Filter by memory type")
    search_parser.add_argument("--limit", type=int, default=10, help="Limit results")
    search_parser.add_argument("--min-importance", type=float, default=0.0, help="Minimum importance")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("content", help="Memory content")
    add_parser.add_argument("--type", default="episodic", help="Memory type")
    add_parser.add_argument("--importance", type=float, default=0.5, help="Importance (0-1)")
    add_parser.add_argument("--scope", default="/project", help="Memory scope")
    add_parser.add_argument("--tags", nargs="*", help="Tags")
    add_parser.add_argument("--private", action="store_true", help="Private memory")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update a memory")
    update_parser.add_argument("id", help="Memory ID")
    update_parser.add_argument("--content", help="New content")
    update_parser.add_argument("--importance", type=float, help="New importance")
    update_parser.add_argument("--tags", nargs="*", help="New tags")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument("id", help="Memory ID")

    # Stats command
    subparsers.add_parser("stats", help="Get memory statistics")

    # Index command
    index_parser = subparsers.add_parser("index", help="Get memory index")
    index_parser.add_argument("--lines", type=int, default=50, help="Max lines")

    # Compact command
    subparsers.add_parser("compact", help="Compact memory")

    # Consolidate command
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate memories")
    consolidate_parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export memories")
    export_parser.add_argument("filepath", nargs="?", help="Export file path")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import memories")
    import_parser.add_argument("filepath", help="Import file path")

    # Add rule command
    rule_parser = subparsers.add_parser("add-rule", help="Add a path-scoped rule")
    rule_parser.add_argument("name", help="Rule name")
    rule_parser.add_argument("content", help="Rule content")
    rule_parser.add_argument("--paths", nargs="*", help="Path patterns")

    # Rules for file command
    rules_parser = subparsers.add_parser("rules-for", help="Get rules for a file")
    rules_parser.add_argument("filepath", help="File path")

    # Synthesize command
    synthesize_parser = subparsers.add_parser("synthesize", help="Synthesize lessons")
    synthesize_parser.add_argument("--sessions", type=int, default=10, help="Number of sessions")

    # Working memory commands
    subparsers.add_parser("working-get", help="Get working memory")
    subparsers.add_parser("working-clear", help="Clear working memory")

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create CLI
    cli = MemoryCLI()

    try:
        if args.command == "list":
            results = cli.list_memories(
                memory_type=args.type,
                scope=args.scope,
                limit=args.limit
            )
            print(json.dumps(results, indent=2))

        elif args.command == "search":
            results = cli.search_memories(
                query=args.query,
                limit=args.limit,
                memory_type=args.type,
                min_importance=args.min_importance
            )
            print(json.dumps(results, indent=2))

        elif args.command == "add":
            memory_id = cli.add_memory(
                content=args.content,
                memory_type=args.type,
                importance=args.importance,
                scope=args.scope,
                tags=args.tags,
                private=args.private
            )
            print(json.dumps({"id": memory_id, "status": "added"}))

        elif args.command == "update":
            success = cli.update_memory(
                memory_id=args.id,
                content=args.content,
                importance=args.importance,
                tags=args.tags
            )
            print(json.dumps({"status": "updated" if success else "failed"}))

        elif args.command == "delete":
            success = cli.delete_memory(args.id)
            print(json.dumps({"status": "deleted" if success else "failed"}))

        elif args.command == "stats":
            stats = cli.get_stats()
            print(json.dumps(stats, indent=2))

        elif args.command == "index":
            index = cli.get_index(max_lines=args.lines)
            print(index)

        elif args.command == "compact":
            result = cli.compact()
            print(json.dumps(result, indent=2))

        elif args.command == "consolidate":
            result = cli.consolidate(threshold=args.threshold)
            print(json.dumps(result, indent=2))

        elif args.command == "export":
            result = cli.export_memories(args.filepath)
            print(json.dumps(result, indent=2))

        elif args.command == "import":
            count = cli.import_memories(args.filepath)
            print(json.dumps({"imported": count}))

        elif args.command == "add-rule":
            result = cli.add_rule(args.name, args.content, args.paths)
            print(json.dumps(result, indent=2))

        elif args.command == "rules-for":
            rules = cli.get_rules_for_file(args.filepath)
            for rule in rules:
                print(f"\n---\n{rule}")

        elif args.command == "synthesize":
            synthesis = cli.synthesize_lessons(args.sessions)
            print(synthesis)

        elif args.command == "working-get":
            working = cli.get_working_memory()
            print(json.dumps(working, indent=2))

        elif args.command == "working-clear":
            result = cli.clear_working_memory()
            print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()


# ============== Interactive Memory Commands ==============

class MemoryCommands:
    """Interactive memory commands for use in agent context"""

    def __init__(self, agent_id: str = None):
        self.cli = MemoryCLI(agent_id)

    def remember(self, content: str, **kwargs) -> str:
        """Remember something"""
        return self.cli.add_memory(content, **kwargs)

    def recall(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Recall memories"""
        return self.cli.search_memories(query, **kwargs)

    def whatRemembered(self) -> str:
        """Get summary of what has been remembered"""
        stats = self.cli.get_stats()
        index = self.cli.get_index(max_lines=20)

        return f"""# Memory Summary

Total memories: {stats['total']}
- Episodic: {stats['by_type'].get('episodic', 0)}
- Semantic: {stats['by_type'].get('semantic', 0)}
- Procedural: {stats['by_type'].get('procedural', 0)}
- Working: {stats.get('working_memory', 0)}

## Recent Memories

{index}
"""

    def memoryStats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.cli.get_stats()

    def addInstruction(self, content: str, scope: str = "/project") -> str:
        """Add an instruction"""
        return self.cli.add_memory(
            content=content,
            memory_type="semantic",
            importance=0.9,
            scope=scope
        )

    def findMemory(self, query: str) -> List[Dict[str, Any]]:
        """Find memories matching query"""
        return self.cli.search_memories(query)


__all__ = [
    "MemoryCLI",
    "MemoryCommands",
    "main"
]
