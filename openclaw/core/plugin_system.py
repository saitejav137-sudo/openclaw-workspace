"""
Plugin System for OpenClaw

Inspired by ComposioHQ/agent-orchestrator's 8-slot plugin architecture.
Every abstraction is swappable via a clean interface + PluginModule pattern.

6 Plugin Slots:
1. LLMProvider   — AI backend (MiniMax, OpenAI, Anthropic, Ollama)
2. Gateway       — message platform (Telegram, Discord, Slack, REST API)
3. SearchEngine  — web search (DuckDuckGo, Google, Brave, Tavily)
4. Notifier      — push notifications (Telegram, Email, Webhook, Desktop)
5. Storage       — persistence (File, Redis, SQLite, PostgreSQL)
6. TaskTracker   — issue tracking (GitHub Issues, Linear, Jira, local)
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger

logger = get_logger("plugin_system")


# ============== Plugin Slots ==============

class PluginSlot(str, Enum):
    """The 6 swappable plugin slots."""
    LLM_PROVIDER = "llm_provider"
    GATEWAY = "gateway"
    SEARCH_ENGINE = "search_engine"
    NOTIFIER = "notifier"
    STORAGE = "storage"
    TASK_TRACKER = "task_tracker"


# ============== Plugin Manifest ==============

@dataclass
class PluginManifest:
    """Metadata every plugin must declare."""
    name: str                  # e.g. "minimax", "telegram", "duckduckgo"
    slot: PluginSlot           # which slot this plugin fills
    description: str           # human-readable description
    version: str = "0.1.0"    # semver


# ============== Plugin Interfaces (ABCs) ==============

class LLMProvider(ABC):
    """Interface for AI/LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Stream text generation token by token."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
        return True


class Gateway(ABC):
    """Interface for message platforms (Telegram, Discord, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str, **kwargs) -> Any:
        """Send a text message."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start listening for incoming messages."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the gateway."""
        ...

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator."""
        pass


class SearchEngine(ABC):
    """Interface for web search providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search the web. Returns list of dicts with:
        - title: result title
        - url: result URL
        - snippet: text snippet
        """
        ...

    async def fetch_url(self, url: str) -> str:
        """Fetch and extract text from a URL."""
        raise NotImplementedError


class Notifier(ABC):
    """
    Interface for push notifications.

    The Notifier is the PRIMARY interface between the orchestrator and the human.
    Push, not pull. The human never polls.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def notify(self, message: str, priority: str = "info", **kwargs) -> None:
        """Push a notification to the human."""
        ...

    async def notify_with_actions(
        self,
        message: str,
        actions: List[Dict[str, str]],
        priority: str = "info",
    ) -> None:
        """Push a notification with actionable buttons/links."""
        await self.notify(message, priority)


class Storage(ABC):
    """Interface for persistence backends."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL (seconds)."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with the given prefix."""
        return []


class TaskTracker(ABC):
    """Interface for issue/task tracking systems."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def get_issue(self, identifier: str) -> Dict[str, Any]:
        """Fetch issue details."""
        ...

    @abstractmethod
    async def create_issue(
        self,
        title: str,
        description: str,
        labels: List[str] = None,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """Create a new issue."""
        ...

    @abstractmethod
    async def update_issue(
        self,
        identifier: str,
        state: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        """Update issue state or add a comment."""
        ...

    async def list_issues(
        self,
        state: str = "open",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List issues with filters."""
        return []


# ============== Plugin Module ==============

@dataclass
class PluginModule:
    """
    What every plugin must export — a manifest + a factory function.

    Example:
        plugin = PluginModule(
            manifest=PluginManifest(
                name="minimax",
                slot=PluginSlot.LLM_PROVIDER,
                description="MiniMax LLM API provider",
            ),
            create=lambda config: MiniMaxProvider(api_key=config["api_key"]),
        )
    """
    manifest: PluginManifest
    create: Callable[[Dict[str, Any]], Any]


# ============== Plugin Registry ==============

# Map slot → expected interface type
SLOT_INTERFACE_MAP: Dict[PluginSlot, Type] = {
    PluginSlot.LLM_PROVIDER: LLMProvider,
    PluginSlot.GATEWAY: Gateway,
    PluginSlot.SEARCH_ENGINE: SearchEngine,
    PluginSlot.NOTIFIER: Notifier,
    PluginSlot.STORAGE: Storage,
    PluginSlot.TASK_TRACKER: TaskTracker,
}


class PluginRegistry:
    """
    Central registry for all plugins.

    Features:
    - Register/unregister plugins by slot and name
    - Get the active plugin for a given slot
    - List all plugins in a slot
    - Validate plugin interface compliance
    - Hot-swap plugins at runtime

    Usage:
        registry = PluginRegistry()

        # Register a plugin
        registry.register(PluginModule(
            manifest=PluginManifest(name="minimax", slot=PluginSlot.LLM_PROVIDER, ...),
            create=lambda cfg: MiniMaxProvider(**cfg),
        ))

        # Activate a plugin
        registry.activate("llm_provider", "minimax", {"api_key": "..."})

        # Get the active plugin for a slot
        llm = registry.get(PluginSlot.LLM_PROVIDER)
        await llm.generate("Hello world")
    """

    def __init__(self):
        # slot → {name → PluginModule}
        self._plugins: Dict[PluginSlot, Dict[str, PluginModule]] = {
            slot: {} for slot in PluginSlot
        }
        # slot → active instance
        self._active: Dict[PluginSlot, Any] = {}
        # slot → active plugin name
        self._active_names: Dict[PluginSlot, str] = {}
        self._lock = threading.Lock()
        logger.info("PluginRegistry initialized with %d slots", len(PluginSlot))

    def register(self, module: PluginModule) -> None:
        """Register a plugin module."""
        slot = module.manifest.slot
        name = module.manifest.name

        with self._lock:
            if name in self._plugins[slot]:
                logger.warning(
                    "Overwriting existing plugin '%s' in slot '%s'",
                    name, slot.value,
                )
            self._plugins[slot][name] = module
            logger.info(
                "Registered plugin '%s' v%s in slot '%s': %s",
                name, module.manifest.version, slot.value,
                module.manifest.description,
            )

    def unregister(self, slot: PluginSlot, name: str) -> None:
        """Remove a plugin from the registry."""
        with self._lock:
            if name in self._plugins[slot]:
                # Deactivate if active
                if self._active_names.get(slot) == name:
                    del self._active[slot]
                    del self._active_names[slot]
                del self._plugins[slot][name]
                logger.info("Unregistered plugin '%s' from slot '%s'", name, slot.value)

    def activate(
        self,
        slot: PluginSlot,
        name: str,
        config: Dict[str, Any] = None,
    ) -> Any:
        """
        Activate a registered plugin, creating an instance.
        Returns the created instance.
        """
        config = config or {}

        with self._lock:
            if name not in self._plugins[slot]:
                available = list(self._plugins[slot].keys())
                raise ValueError(
                    f"Plugin '{name}' not found in slot '{slot.value}'. "
                    f"Available: {available}"
                )

            module = self._plugins[slot][name]
            instance = module.create(config)

            # Validate interface compliance
            expected_type = SLOT_INTERFACE_MAP.get(slot)
            if expected_type and not isinstance(instance, expected_type):
                raise TypeError(
                    f"Plugin '{name}' does not implement {expected_type.__name__}. "
                    f"Got {type(instance).__name__}"
                )

            # Deactivate previous if exists
            old_name = self._active_names.get(slot)
            if old_name:
                logger.info(
                    "Swapping plugin in slot '%s': '%s' → '%s'",
                    slot.value, old_name, name,
                )

            self._active[slot] = instance
            self._active_names[slot] = name
            logger.info("Activated plugin '%s' in slot '%s'", name, slot.value)
            return instance

    def get(self, slot: PluginSlot) -> Optional[Any]:
        """Get the active plugin instance for a slot."""
        return self._active.get(slot)

    def get_by_name(self, slot: PluginSlot, name: str) -> Optional[PluginModule]:
        """Get a registered (not necessarily active) plugin module."""
        return self._plugins[slot].get(name)

    def get_active_name(self, slot: PluginSlot) -> Optional[str]:
        """Get the name of the active plugin in a slot."""
        return self._active_names.get(slot)

    def list_plugins(self, slot: PluginSlot) -> List[PluginManifest]:
        """List all registered plugins in a slot."""
        return [m.manifest for m in self._plugins[slot].values()]

    def list_all(self) -> Dict[str, List[PluginManifest]]:
        """List all registered plugins grouped by slot."""
        return {
            slot.value: self.list_plugins(slot) for slot in PluginSlot
        }

    def is_active(self, slot: PluginSlot) -> bool:
        """Check if a slot has an active plugin."""
        return slot in self._active

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_registered": sum(
                len(plugins) for plugins in self._plugins.values()
            ),
            "active_slots": len(self._active),
            "slots": {
                slot.value: {
                    "registered": [m.name for m in self.list_plugins(slot)],
                    "active": self._active_names.get(slot),
                }
                for slot in PluginSlot
            },
        }


# ============== Global Registry ==============

_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get or create the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def register_plugin(module: PluginModule) -> None:
    """Convenience: register a plugin in the global registry."""
    get_plugin_registry().register(module)


def get_plugin(slot: PluginSlot) -> Optional[Any]:
    """Convenience: get the active plugin for a slot."""
    return get_plugin_registry().get(slot)


__all__ = [
    # Enums
    "PluginSlot",
    # Interfaces
    "LLMProvider",
    "Gateway",
    "SearchEngine",
    "Notifier",
    "Storage",
    "TaskTracker",
    # Plugin types
    "PluginManifest",
    "PluginModule",
    # Registry
    "PluginRegistry",
    "get_plugin_registry",
    "register_plugin",
    "get_plugin",
    # Mapping
    "SLOT_INTERFACE_MAP",
]
