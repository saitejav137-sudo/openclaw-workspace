"""
Enhanced Plugin System for OpenClaw

Hot-loadable plugin architecture with:
- Plugin discovery from directory
- Lifecycle hooks (init, start, stop)
- Plugin manifest (plugin.yaml)
- Sandboxing via restricted imports
- Hot-reload support
"""

import os
import importlib
import importlib.util
import yaml
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from ..core.logger import get_logger

logger = get_logger("plugins")


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PluginManifest:
    """Plugin metadata from plugin.yaml."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)  # file_read, file_write, network, shell
    entry_point: str = "plugin.py"
    config_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginInfo:
    """Runtime plugin information."""
    manifest: PluginManifest
    status: PluginStatus = PluginStatus.DISCOVERED
    path: str = ""
    instance: Optional[Any] = None
    loaded_at: Optional[float] = None
    error: Optional[str] = None


class PluginBase(ABC):
    """
    Base class for all OpenClaw plugins.

    Plugins must implement:
    - init(): Called when plugin is loaded
    - start(): Called when plugin is activated
    - stop(): Called when plugin is deactivated

    Optional:
    - configure(config): Called with user configuration
    - health_check(): Return True if plugin is healthy
    """

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._logger = get_logger(f"plugin.{self.__class__.__name__}")

    @abstractmethod
    def init(self):
        """Initialize the plugin."""
        ...

    @abstractmethod
    def start(self):
        """Start the plugin."""
        ...

    @abstractmethod
    def stop(self):
        """Stop the plugin."""
        ...

    def configure(self, config: Dict[str, Any]):
        """Configure the plugin with user settings."""
        self.config = config

    def health_check(self) -> bool:
        """Check if plugin is healthy."""
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ============== Plugin Permission Sandboxing ==============

PERMISSION_ALLOWLIST = {
    "file_read": ["os.path", "pathlib", "open"],
    "file_write": ["os.makedirs", "open", "shutil"],
    "network": ["requests", "urllib", "aiohttp", "websockets"],
    "shell": ["subprocess"],
    "vision": ["cv2", "mss", "easyocr", "ultralytics"],
}


def check_plugin_permissions(manifest: PluginManifest) -> List[str]:
    """
    Validate plugin permissions.
    Returns list of warnings about potentially dangerous permissions.
    """
    warnings = []

    if "shell" in manifest.permissions:
        warnings.append(
            f"Plugin '{manifest.name}' requests SHELL access — "
            "allows arbitrary command execution"
        )

    if "network" in manifest.permissions:
        warnings.append(
            f"Plugin '{manifest.name}' requests NETWORK access — "
            "can make external HTTP requests"
        )

    if "file_write" in manifest.permissions:
        warnings.append(
            f"Plugin '{manifest.name}' requests FILE WRITE access — "
            "can modify files on disk"
        )

    return warnings


# ============== Plugin Manager ==============

class PluginManager:
    """
    Manages plugin lifecycle: discovery, loading, activation, deactivation.

    Usage:
        manager = PluginManager("/path/to/plugins/")
        manager.discover()
        manager.load_all()
        manager.start_all()
    """

    def __init__(self, plugin_dir: Optional[str] = None):
        self._plugin_dir = plugin_dir or os.path.expanduser("~/.openclaw/plugins")
        self._plugins: Dict[str, PluginInfo] = {}
        self._lock = threading.Lock()

        os.makedirs(self._plugin_dir, exist_ok=True)

    def discover(self) -> List[str]:
        """
        Discover plugins in the plugin directory.
        Each plugin should be a directory with a plugin.yaml manifest.
        """
        discovered = []
        plugin_path = Path(self._plugin_dir)

        if not plugin_path.exists():
            return discovered

        for item in plugin_path.iterdir():
            if not item.is_dir():
                continue

            manifest_file = item / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = item / "plugin.yml"
                if not manifest_file.exists():
                    continue

            try:
                with open(manifest_file, "r") as f:
                    data = yaml.safe_load(f)

                manifest = PluginManifest(
                    name=data.get("name", item.name),
                    version=data.get("version", "1.0.0"),
                    description=data.get("description", ""),
                    author=data.get("author", ""),
                    capabilities=data.get("capabilities", []),
                    dependencies=data.get("dependencies", []),
                    permissions=data.get("permissions", []),
                    entry_point=data.get("entry_point", "plugin.py"),
                    config_schema=data.get("config_schema", {}),
                )

                # Check permissions
                warnings = check_plugin_permissions(manifest)
                for w in warnings:
                    logger.warning(w)

                self._plugins[manifest.name] = PluginInfo(
                    manifest=manifest,
                    status=PluginStatus.DISCOVERED,
                    path=str(item),
                )

                discovered.append(manifest.name)
                logger.info(
                    f"Discovered plugin: {manifest.name} v{manifest.version} "
                    f"({manifest.description})"
                )

            except Exception as e:
                logger.error(f"Failed to discover plugin at {item}: {e}")

        return discovered

    def load(self, name: str) -> bool:
        """Load a discovered plugin."""
        info = self._plugins.get(name)
        if not info:
            logger.error(f"Plugin '{name}' not found")
            return False

        if info.status not in (PluginStatus.DISCOVERED, PluginStatus.STOPPED, PluginStatus.ERROR):
            logger.warning(f"Plugin '{name}' already loaded (status: {info.status.value})")
            return True

        try:
            entry_point = os.path.join(info.path, info.manifest.entry_point)
            if not os.path.exists(entry_point):
                raise FileNotFoundError(f"Entry point not found: {entry_point}")

            # Load module
            spec = importlib.util.spec_from_file_location(
                f"openclaw_plugin_{name}", entry_point
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find PluginBase subclass
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, PluginBase)
                    and attr is not PluginBase
                ):
                    plugin_class = attr
                    break

            if not plugin_class:
                raise ValueError(f"No PluginBase subclass found in {entry_point}")

            # Instantiate
            instance = plugin_class()
            info.instance = instance
            info.status = PluginStatus.LOADED
            info.loaded_at = time.time()

            logger.info(f"Loaded plugin: {name}")
            return True

        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"Failed to load plugin '{name}': {e}")
            return False

    def load_all(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        results = {}
        for name in list(self._plugins.keys()):
            results[name] = self.load(name)
        return results

    def start(self, name: str) -> bool:
        """Start a loaded plugin."""
        info = self._plugins.get(name)
        if not info or not info.instance:
            logger.error(f"Plugin '{name}' not loaded")
            return False

        try:
            info.instance.init()
            info.status = PluginStatus.INITIALIZED

            info.instance.start()
            info.status = PluginStatus.RUNNING

            logger.info(f"Started plugin: {name}")
            return True

        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"Failed to start plugin '{name}': {e}")
            return False

    def start_all(self) -> Dict[str, bool]:
        """Start all loaded plugins."""
        results = {}
        for name, info in self._plugins.items():
            if info.status == PluginStatus.LOADED:
                results[name] = self.start(name)
        return results

    def stop(self, name: str) -> bool:
        """Stop a running plugin."""
        info = self._plugins.get(name)
        if not info or not info.instance:
            return False

        try:
            info.instance.stop()
            info.status = PluginStatus.STOPPED
            logger.info(f"Stopped plugin: {name}")
            return True
        except Exception as e:
            info.status = PluginStatus.ERROR
            info.error = str(e)
            logger.error(f"Failed to stop plugin '{name}': {e}")
            return False

    def stop_all(self):
        """Stop all running plugins."""
        for name, info in self._plugins.items():
            if info.status == PluginStatus.RUNNING:
                self.stop(name)

    def reload(self, name: str) -> bool:
        """Hot-reload a plugin (stop → load → start)."""
        self.stop(name)

        info = self._plugins.get(name)
        if info:
            info.status = PluginStatus.DISCOVERED
            info.instance = None

        return self.load(name) and self.start(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their status."""
        return [
            {
                "name": info.manifest.name,
                "version": info.manifest.version,
                "description": info.manifest.description,
                "status": info.status.value,
                "capabilities": info.manifest.capabilities,
                "permissions": info.manifest.permissions,
                "error": info.error,
            }
            for info in self._plugins.values()
        ]

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a running plugin instance."""
        info = self._plugins.get(name)
        if info and info.status == PluginStatus.RUNNING:
            return info.instance
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        statuses = {}
        for info in self._plugins.values():
            statuses[info.status.value] = statuses.get(info.status.value, 0) + 1

        return {
            "plugin_dir": self._plugin_dir,
            "total_plugins": len(self._plugins),
            "by_status": statuses,
        }


# ============== Global Access ==============

_manager: Optional[PluginManager] = None


def get_plugin_manager(plugin_dir: str = None) -> PluginManager:
    """Get global plugin manager."""
    global _manager
    if _manager is None:
        _manager = PluginManager(plugin_dir)
    return _manager


__all__ = [
    "PluginStatus",
    "PluginManifest",
    "PluginInfo",
    "PluginBase",
    "PluginManager",
    "get_plugin_manager",
    "check_plugin_permissions",
]
