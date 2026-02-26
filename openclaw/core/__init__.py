"""
Core modules for OpenClaw

This module uses lazy imports to avoid loading all submodules at startup.
Import specific modules directly when needed for better performance.

Example:
    from openclaw.core.vision import VisionEngine
    from openclaw.core.actions import TriggerAction
"""

# Lazy imports for commonly used classes
def __getattr__(name):
    """Lazy import attributes on demand"""

    # Vision module
    if name == "VisionEngine":
        from .vision import VisionEngine
        return VisionEngine
    elif name == "ScreenCapture":
        from .vision import ScreenCapture
        return ScreenCapture
    elif name == "OCREngine":
        from .vision import OCREngine
        return OCREngine
    elif name == "VisionConfig":
        from .config import VisionConfig
        return VisionConfig
    elif name == "VisionMode":
        from .config import VisionMode
        return VisionMode
    elif name == "ConfigManager":
        from .config import ConfigManager
        return ConfigManager

    # Actions module
    elif name == "TriggerAction":
        from .actions import TriggerAction
        return TriggerAction
    elif name == "ActionSequence":
        from .actions import ActionSequence
        return ActionSequence
    elif name == "RetryConfig":
        from .actions import RetryConfig
        return RetryConfig

    # Logger
    elif name == "get_logger":
        from .logger import get_logger
        return get_logger
    elif name == "setup_logging":
        from .logger import setup_logging
        return setup_logging

    # AI
    elif name == "NLInterface":
        from .ai import NLInterface
        return NLInterface

    # Automation
    elif name == "get_automation_backend":
        from .automation import get_automation_backend
        return get_automation_backend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Direct imports for backwards compatibility
# These are loaded on first import of the module
from .config import VisionConfig, VisionMode, ConfigManager, ConfigValidationError
from .logger import Logger, get_logger, setup_logging
