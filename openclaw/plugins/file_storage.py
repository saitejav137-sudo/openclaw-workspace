"""
File Storage Plugin for OpenClaw

JSON-backed filesystem storage provider. Implements atomic writes,
key namespacing, and metadata tracking.
"""

import os
import json
import time
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.plugin_system import Storage, PluginManifest, PluginModule, PluginSlot
from core.logger import get_logger

logger = get_logger("plugin.file_storage")


class FileStoragePlugin(Storage):
    """
    Filesystem-based storage provider.

    Uses JSON serialization with atomic writes (write to tmp → rename).
    Organizes data under `~/.openclaw/storage/<namespace>/`.

    Config:
        base_dir: Storage root (default: ~/.openclaw/storage)
    """

    def __init__(self):
        self.base_dir: str = os.path.expanduser("~/.openclaw/storage")
        self._lock = threading.Lock()
        self._total_reads: int = 0
        self._total_writes: int = 0
        self._total_deletes: int = 0

    @property
    def name(self) -> str:
        return "file-storage"

    async def get(self, key: str) -> Any:
        """ABC required: async get."""
        return self.load(key)

    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """ABC required: async set."""
        self.save(key, value)

    def configure(self, config: Dict[str, Any]) -> None:
        if "base_dir" in config:
            self.base_dir = os.path.expanduser(config["base_dir"])

    def _key_path(self, key: str, namespace: str = "default") -> str:
        """Get filesystem path for a key."""
        safe_key = key.replace("/", "__").replace("\\", "__")
        directory = os.path.join(self.base_dir, namespace)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{safe_key}.json")

    def save(self, key: str, value: Any, namespace: str = "default", **kwargs) -> bool:
        """
        Save a value with atomic write.

        Uses write-to-temp-then-rename to prevent corruption on crash.
        """
        path = self._key_path(key, namespace)
        envelope = {
            "key": key,
            "namespace": namespace,
            "value": value,
            "metadata": {
                "created_at": time.time(),
                "type": type(value).__name__,
            },
        }

        try:
            directory = os.path.dirname(path)
            # Atomic write: temp file → rename
            fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(envelope, f, indent=2, default=str)

            os.replace(tmp_path, path)  # Atomic on POSIX

            with self._lock:
                self._total_writes += 1
            return True

        except Exception as e:
            logger.error("Storage save failed for '%s': %s", key, e)
            # Clean up temp file
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except Exception:
                pass
            return False

    def load(self, key: str, namespace: str = "default", default: Any = None, **kwargs) -> Any:
        """Load a value by key."""
        path = self._key_path(key, namespace)

        if not os.path.exists(path):
            return default

        try:
            with open(path, "r") as f:
                envelope = json.load(f)

            with self._lock:
                self._total_reads += 1

            return envelope.get("value", default)

        except Exception as e:
            logger.error("Storage load failed for '%s': %s", key, e)
            return default

    def delete(self, key: str, namespace: str = "default", **kwargs) -> bool:
        """Delete a key."""
        path = self._key_path(key, namespace)

        if not os.path.exists(path):
            return False

        try:
            os.unlink(path)
            with self._lock:
                self._total_deletes += 1
            return True
        except Exception as e:
            logger.error("Storage delete failed for '%s': %s", key, e)
            return False

    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if a key exists."""
        return os.path.exists(self._key_path(key, namespace))

    def list_keys(self, namespace: str = "default", pattern: str = "*", **kwargs) -> List[str]:
        """List all keys in a namespace."""
        import fnmatch
        directory = os.path.join(self.base_dir, namespace)

        if not os.path.exists(directory):
            return []

        keys = []
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                key = filename[:-5].replace("__", "/")
                if fnmatch.fnmatch(key, pattern):
                    keys.append(key)
        return sorted(keys)

    def clear_namespace(self, namespace: str = "default") -> int:
        """Delete all keys in a namespace. Returns count deleted."""
        directory = os.path.join(self.base_dir, namespace)
        if not os.path.exists(directory):
            return 0

        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                try:
                    os.unlink(os.path.join(directory, filename))
                    count += 1
                except Exception:
                    pass
        return count

    def get_stats(self) -> Dict[str, Any]:
        # Count total stored items
        total_items = 0
        total_bytes = 0
        namespaces = []
        if os.path.exists(self.base_dir):
            for ns in os.listdir(self.base_dir):
                ns_path = os.path.join(self.base_dir, ns)
                if os.path.isdir(ns_path):
                    namespaces.append(ns)
                    for f in os.listdir(ns_path):
                        if f.endswith(".json"):
                            total_items += 1
                            total_bytes += os.path.getsize(os.path.join(ns_path, f))

        return {
            "base_dir": self.base_dir,
            "total_items": total_items,
            "total_bytes": total_bytes,
            "namespaces": namespaces,
            "total_reads": self._total_reads,
            "total_writes": self._total_writes,
            "total_deletes": self._total_deletes,
        }


# ============== Plugin Module ==============

def create_plugin(config: Dict[str, Any] = None) -> FileStoragePlugin:
    plugin = FileStoragePlugin()
    if config:
        plugin.configure(config)
    return plugin


MANIFEST = PluginManifest(
    name="file-storage",
    version="1.0.0",
    description="JSON filesystem storage provider with atomic writes",
    slot=PluginSlot.STORAGE,
)

file_module = PluginModule(manifest=MANIFEST, create=create_plugin)


__all__ = ["FileStoragePlugin", "create_plugin", "MANIFEST", "file_module"]
