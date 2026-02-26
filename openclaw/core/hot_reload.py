"""
Hot Reload Support for OpenClaw

Watches for changes in config files and plugins and reloads without restart.
"""

import os
import time
import threading
import logging
import hashlib
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .logger import get_logger

logger = get_logger("hot-reload")


class ReloadType(Enum):
    """Types of reload"""
    CONFIG = "config"
    PLUGIN = "plugin"
    TEMPLATE = "template"
    ALL = "all"


@dataclass
class ReloadEvent:
    """Reload event"""
    type: ReloadType
    path: str
    timestamp: float
    action: str  # created, modified, deleted


class FileWatcher:
    """
    File watcher for hot reload.
    Uses polling or native OS events.
    """

    def __init__(
        self,
        paths: List[str],
        callback: Callable[[ReloadEvent], None],
        poll_interval: float = 1.0,
        use_native: bool = True
    ):
        self.paths = paths
        self.callback = callback
        self.poll_interval = poll_interval
        self.use_native = use_native and self._has_native_support()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._file_hashes: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _has_native_support(self) -> bool:
        """Check if native file watching is available"""
        try:
            import watchdog
            return True
        except ImportError:
            return False

    def start(self):
        """Start watching files"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

        # Initial scan
        self._scan_files()

        logger.info(f"File watcher started for: {self.paths}")

    def stop(self):
        """Stop watching files"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("File watcher stopped")

    def _scan_files(self):
        """Scan files and compute initial hashes"""
        for path in self.paths:
            self._update_hash(path)

    def _update_hash(self, path: str) -> Optional[str]:
        """Compute file hash"""
        try:
            if os.path.isfile(path):
                with open(path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
            elif os.path.isdir(path):
                # Hash directory contents
                hash_obj = hashlib.md5()
                for root, dirs, files in os.walk(path):
                    dirs.sort()
                    files.sort()
                    for f in files:
                        fp = os.path.join(root, f)
                        with open(fp, 'rb') as fh:
                            hash_obj.update(fh.read())
                return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {path}: {e}")

        return None

    def _watch_loop(self):
        """Main watching loop"""
        if self.use_native:
            self._native_watch_loop()
        else:
            self._poll_watch_loop()

    def _native_watch_loop(self):
        """Native file watching using watchdog"""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class WatchHandler(FileSystemEventHandler):
                def __init__(self, watcher):
                    self.watcher = watcher

                def on_any_event(self, event):
                    if event.is_directory:
                        return

                    action = "modified"
                    if event.event_type == "created":
                        action = "created"
                    elif event.event_type == "deleted":
                        action = "deleted"

                    self.watcher._handle_change(event.src_path, action)

            handler = WatchHandler(self)
            observer = Observer()

            for path in self.paths:
                if os.path.isdir(path):
                    observer.schedule(handler, path, recursive=True)
                elif os.path.isfile(path):
                    observer.schedule(handler, os.path.dirname(path) or ".", recursive=False)

            observer.start()

            while self._running:
                time.sleep(1)

            observer.stop()
            observer.join()

        except ImportError:
            # Fallback to polling
            logger.warning("Native watching not available, using polling")
            self.use_native = False
            self._poll_watch_loop()

    def _poll_watch_loop(self):
        """Polling-based file watching"""
        while self._running:
            try:
                for path in self.paths:
                    self._check_changes(path)

            except Exception as e:
                logger.error(f"Watch error: {e}")

            time.sleep(self.poll_interval)

    def _check_changes(self, path: str):
        """Check for file changes"""
        current_hash = self._update_hash(path)
        if not current_hash:
            return

        last_hash = self._file_hashes.get(path)

        if last_hash is None:
            # New file
            self._file_hashes[path] = current_hash
            self._handle_change(path, "created")

        elif current_hash != last_hash:
            # File changed
            self._file_hashes[path] = current_hash
            self._handle_change(path, "modified")

    def _handle_change(self, path: str, action: str):
        """Handle file change"""
        # Determine reload type
        reload_type = ReloadType.ALL
        ext = os.path.splitext(path)[1]

        if ext in (".yaml", ".yml", ".json"):
            reload_type = ReloadType.CONFIG
        elif ext == ".py":
            reload_type = ReloadType.PLUGIN
        elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
            reload_type = ReloadType.TEMPLATE

        event = ReloadEvent(
            type=reload_type,
            path=path,
            timestamp=time.time(),
            action=action
        )

        logger.info(f"File change detected: {action} - {path}")

        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Callback error: {e}")


class HotReloader:
    """
    Hot reload manager for OpenClaw.
    Handles configuration and plugin reloading.
    """

    def __init__(self, config_paths: Optional[List[str]] = None):
        self.config_paths = config_paths or []
        self.plugin_paths: List[str] = []
        self.template_paths: List[str] = []

        self._watcher: Optional[FileWatcher] = None
        self._running = False
        self._callbacks: Dict[ReloadType, List[Callable]] = {
            ReloadType.CONFIG: [],
            ReloadType.PLUGIN: [],
            ReloadType.TEMPLATE: [],
            ReloadType.ALL: []
        }

        # Current config
        self._current_config = None

    def add_config_path(self, path: str):
        """Add config file/directory to watch"""
        if path not in self.config_paths:
            self.config_paths.append(path)

    def add_plugin_path(self, path: str):
        """Add plugin directory to watch"""
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)

    def add_template_path(self, path: str):
        """Add template directory to watch"""
        if path not in self.template_paths:
            self.template_paths.append(path)

    def register_callback(self, reload_type: ReloadType, callback: Callable):
        """Register callback for reload events"""
        self._callbacks[reload_type].append(callback)

    def start(self, poll_interval: float = 1.0):
        """Start hot reload"""
        if self._running:
            return

        all_paths = self.config_paths + self.plugin_paths + self.template_paths

        if not all_paths:
            logger.warning("No paths configured for hot reload")
            return

        self._watcher = FileWatcher(
            paths=all_paths,
            callback=self._handle_reload_event,
            poll_interval=poll_interval
        )

        self._watcher.start()
        self._running = True

        logger.info("Hot reload started")

    def stop(self):
        """Stop hot reload"""
        if self._watcher:
            self._watcher.stop()
        self._running = False
        logger.info("Hot reload stopped")

    def _handle_reload_event(self, event: ReloadEvent):
        """Handle reload event"""
        logger.info(f"Reload event: {event.type.value} - {event.path}")

        # Call type-specific callbacks
        if event.type in self._callbacks:
            for callback in self._callbacks[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Reload callback error: {e}")

        # Call ALL callbacks
        for callback in self._callbacks[ReloadType.ALL]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Reload callback error: {e}")


class ConfigReloader:
    """
    Configuration reloader with validation.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self._last_mtime: float = 0

    def check_and_reload(self) -> bool:
        """Check if config changed and reload if needed"""
        config_file = self.config_manager.config_file

        if not config_file or not os.path.exists(config_file):
            return False

        current_mtime = os.path.getmtime(config_file)

        if current_mtime > self._last_mtime:
            self._last_mtime = current_mtime

            try:
                # Reload config
                new_config = self.config_manager.reload()

                if new_config:
                    logger.info(f"Config reloaded from {config_file}")
                    return True

            except Exception as e:
                logger.error(f"Config reload error: {e}")
                return False

        return False


def create_reloader(config_paths: List[str]) -> HotReloader:
    """Create a hot reloader with common paths"""
    reloader = HotReloader()

    for path in config_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                reloader.add_config_path(path)
            elif os.path.isdir(path):
                # Determine type by path name
                if "config" in path.lower():
                    reloader.add_config_path(path)
                elif "plugin" in path.lower():
                    reloader.add_plugin_path(path)
                elif "template" in path.lower():
                    reloader.add_template_path(path)
                else:
                    # Default to config
                    reloader.add_config_path(path)

    return reloader


__all__ = [
    "FileWatcher",
    "HotReloader",
    "ConfigReloader",
    "ReloadEvent",
    "ReloadType",
    "create_reloader",
]
