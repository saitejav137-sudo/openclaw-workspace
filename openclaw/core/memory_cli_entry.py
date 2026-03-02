#!/usr/bin/env python3
"""
Memory CLI Entry Point for OpenClaw

Usage:
    python -m openclaw.core.memory_cli [command]

Commands:
    list        List memories
    search      Search memories
    add         Add a memory
    update      Update a memory
    delete      Delete a memory
    stats       Get memory statistics
    index       Get memory index
    compact     Compact memory
    consolidate Consolidate memories
    export      Export memories
    import      Import memories
    add-rule    Add a path-scoped rule
    rules-for   Get rules for a file
    synthesize  Synthesize lessons from sessions
    working-get Get working memory
    working-clear Clear working memory
"""

from openclaw.core.memory_cli import main

if __name__ == "__main__":
    main()
