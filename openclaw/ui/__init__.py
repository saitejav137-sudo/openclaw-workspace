"""UI module for OpenClaw"""

from .dashboard import get_modern_dashboard_html, MODERN_DASHBOARD_HTML
from .config_editor import CONFIG_EDITOR_HTML, ConfigEditorHandler, ConfigEditorServer

__all__ = [
    "get_modern_dashboard_html",
    "MODERN_DASHBOARD_HTML",
    "CONFIG_EDITOR_HTML",
    "ConfigEditorHandler",
    "ConfigEditorServer",
]
