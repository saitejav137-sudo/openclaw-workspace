"""
OpenClaw - Vision-enabled automation framework

Modular structure:
- core: VisionEngine, Actions, Config
- integrations: HTTP, Telegram, WebSocket, Gateway
- storage: Database, Cache
- ui: Dashboard, CLI
- utils: Helpers
"""

__version__ = "2.0.0"
__author__ = "OpenClaw Team"

from openclaw.core.config import VisionConfig, VisionMode
from openclaw.core.vision import VisionEngine, ScreenCapture

__all__ = [
    "VisionConfig",
    "VisionMode",
    "VisionEngine",
    "ScreenCapture",
]
