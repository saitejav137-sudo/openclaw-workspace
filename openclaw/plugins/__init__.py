"""
Plugin System for OpenClaw

Allows extending detection capabilities with custom plugins.
"""

import os
import sys
import importlib.util
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger("openclaw.plugins")


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str] = field(default_factory=list)


class BasePlugin(ABC):
    """Base class for all plugins"""

    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.enabled: bool = True

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin"""
        pass

    @abstractmethod
    def process(self, frame: Any) -> Any:
        """Process a frame and return result"""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass


class DetectionPlugin(BasePlugin):
    """Plugin for custom detection algorithms"""

    def __init__(self):
        super().__init__()
        self.config: Dict = {}

    def initialize(self) -> bool:
        """Initialize the detection plugin"""
        return True

    def process(self, frame: Any) -> Dict[str, Any]:
        """Process frame and return detection results"""
        return {"detected": False, "confidence": 0.0, "data": None}

    def cleanup(self):
        """Cleanup resources"""
        pass


class ActionPlugin(BasePlugin):
    """Plugin for custom actions"""

    def __init__(self):
        super().__init__()

    def initialize(self) -> bool:
        """Initialize the action plugin"""
        return True

    def process(self, context: Dict) -> bool:
        """Execute custom action"""
        return True

    def cleanup(self):
        """Cleanup resources"""
        pass


class PluginManager:
    """Manages plugin loading and lifecycle"""

    _instance = None
    _plugins: Dict[str, BasePlugin] = {}
    _hooks: Dict[str, List[Callable]] = {}

    def __init__(self):
        self.plugin_dir = os.path.expanduser("~/.openclaw/plugins")
        os.makedirs(self.plugin_dir, exist_ok=True)
        self._enabled = True

    @classmethod
    def get_instance(cls) -> 'PluginManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_plugin(self, name: str, plugin: BasePlugin) -> bool:
        """Register a plugin"""
        try:
            self._plugins[name] = plugin
            logger.info(f"Plugin registered: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register plugin {name}: {e}")
            return False

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin"""
        if name in self._plugins:
            try:
                self._plugins[name].cleanup()
                del self._plugins[name]
                logger.info(f"Plugin unregistered: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to unregister plugin {name}: {e}")
        return False

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name"""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self._plugins.keys())

    def load_plugins(self) -> int:
        """Load all plugins from plugin directory"""
        if not os.path.exists(self.plugin_dir):
            logger.info("Plugin directory does not exist")
            return 0

        loaded = 0
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                plugin_name = filename[:-3]
                if self._load_plugin_from_file(plugin_name):
                    loaded += 1

        logger.info(f"Loaded {loaded} plugins")
        return loaded

    def _load_plugin_from_file(self, plugin_name: str) -> bool:
        """Load a plugin from a Python file"""
        plugin_path = os.path.join(self.plugin_dir, f"{plugin_name}.py")

        try:
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_name] = module
                spec.loader.exec_module(module)

                # Look for plugin class
                if hasattr(module, 'register'):
                    plugin = module.register(self)
                    if plugin:
                        self.register_plugin(plugin_name, plugin)
                        return True

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")

        return False

    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback"""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)

    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Trigger all callbacks for a hook"""
        results = []
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook {hook_name} error: {e}")
        return results

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin"""
        if name in self._plugins:
            self._plugins[name].enabled = True
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin"""
        if name in self._plugins:
            self._plugins[name].enabled = False
            return True
        return False

    def cleanup_all(self):
        """Cleanup all plugins"""
        for name, plugin in self._plugins.items():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {name}: {e}")
        self._plugins.clear()


# Example plugin template
EXAMPLE_PLUGIN = '''
"""
Example OpenClaw Plugin

Copy this to ~/.openclaw/plugins/ and modify it.
"""

from openclaw.plugins import DetectionPlugin, PluginMetadata


class MyCustomPlugin(DetectionPlugin):
    """Custom detection plugin"""

    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="my_custom_plugin",
            version="1.0.0",
            author="Your Name",
            description="My custom detection plugin"
        )

    def initialize(self) -> bool:
        """Initialize the plugin"""
        # Add your initialization code here
        print("My custom plugin initialized!")
        return True

    def process(self, frame):
        """Process a frame"""
        # Add your detection logic here
        return {
            "detected": False,
            "confidence": 0.0,
            "data": None
        }

    def cleanup(self):
        """Cleanup resources"""
        pass


def register(manager):
    """Register the plugin with the manager"""
    return MyCustomPlugin()
'''


def create_example_plugin():
    """Create example plugin file"""
    example_path = os.path.expanduser("~/.openclaw/plugins/example.py")
    os.makedirs(os.path.dirname(example_path), exist_ok=True)

    if not os.path.exists(example_path):
        with open(example_path, "w") as f:
            f.write(EXAMPLE_PLUGIN)
        print(f"Example plugin created at: {example_path}")


# Export classes
__all__ = [
    "BasePlugin",
    "DetectionPlugin",
    "ActionPlugin",
    "PluginManager",
    "PluginMetadata",
    "create_example_plugin",
]
