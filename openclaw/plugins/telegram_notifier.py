"""
Telegram Notifier Plugin for OpenClaw

Sends notifications via Telegram when reactions escalate,
system health degrades, or swarms complete.

This is the bridge between the reaction engine and the user.
"""

import os
import time
import threading
from typing import Any, Dict, List, Optional

from core.plugin_system import Notifier, PluginManifest, PluginModule, PluginSlot
from core.event_bus import EventPriority
from core.logger import get_logger

logger = get_logger("plugin.telegram_notifier")


class TelegramNotifierPlugin(Notifier):
    """
    Telegram notification provider.

    Sends messages via the Telegram Bot API when the reaction engine
    needs to notify the human operator.

    Config:
        bot_token: Telegram bot token (or TELEGRAM_BOT_TOKEN env)
        chat_id: Target chat ID (or TELEGRAM_CHAT_ID env)
        urgent: bool — send urgent priority notifications
        action: bool — send action priority notifications
        warning: bool — send warning priority notifications
        info: bool — send info priority notifications
    """

    def __init__(self):
        self.bot_token: Optional[str] = None
        self.chat_id: Optional[str] = None
        self._routing = {
            "urgent": True,
            "action": True,
            "warning": True,
            "info": False,
        }
        self._total_sent: int = 0
        self._total_errors: int = 0
        self._last_sent: float = 0
        self._min_interval: float = 2.0  # Min seconds between messages (anti-spam)
        self._lock = threading.Lock()

    def configure(self, config: Dict[str, Any]) -> None:
        self.bot_token = (
            config.get("bot_token")
            or os.getenv("TELEGRAM_BOT_TOKEN")
            or os.getenv("AJANTA_BOT_TOKEN")
        )
        self.chat_id = (
            str(config.get("chat_id", ""))
            or os.getenv("TELEGRAM_CHAT_ID", "")
        )

        # Routing config
        routing = config.get("routing", {})
        for level in self._routing:
            if level in routing:
                self._routing[level] = bool(routing[level])

    def notify(
        self,
        message: str,
        priority: str = "info",
        data: Dict[str, Any] = None,
        **kwargs,
    ) -> bool:
        """
        Send a notification message.

        Args:
            message: Notification text
            priority: urgent/action/warning/info
            data: Additional context data
        """
        # Check routing
        if not self._routing.get(priority, False):
            logger.debug("Notification suppressed (priority=%s, routing=off)", priority)
            return True  # Not an error, just filtered

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram notifier not configured (token=%s, chat=%s)",
                          bool(self.bot_token), bool(self.chat_id))
            return False

        # Anti-spam throttle
        with self._lock:
            now = time.time()
            if now - self._last_sent < self._min_interval:
                time.sleep(self._min_interval - (now - self._last_sent))
            self._last_sent = time.time()

        # Priority emoji
        emoji_map = {
            "urgent": "🚨",
            "action": "⚡",
            "warning": "⚠️",
            "info": "ℹ️",
        }
        emoji = emoji_map.get(priority, "📢")

        # Format message
        formatted = f"{emoji} *OpenClaw [{priority.upper()}]*\n\n{message}"

        if data:
            # Add key context data
            context_lines = []
            for key in ("agent_id", "task_id", "reaction", "error"):
                if key in data and data[key]:
                    context_lines.append(f"• {key}: `{data[key]}`")
            if context_lines:
                formatted += "\n\n" + "\n".join(context_lines)

        return self._send_message(formatted)

    def _send_message(self, text: str) -> bool:
        """Send a message via Telegram Bot API."""
        import requests

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self._total_sent += 1
                logger.info("Telegram notification sent (priority=%s)", "info")
                return True
            else:
                self._total_errors += 1
                logger.error("Telegram API error: %s", response.text[:200])
                return False
        except Exception as e:
            self._total_errors += 1
            logger.error("Telegram send failed: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_sent": self._total_sent,
            "total_errors": self._total_errors,
            "routing": self._routing,
            "configured": bool(self.bot_token and self.chat_id),
        }


# ============== Plugin Module ==============

def create_plugin(config: Dict[str, Any] = None) -> TelegramNotifierPlugin:
    plugin = TelegramNotifierPlugin()
    if config:
        plugin.configure(config)
    return plugin


MANIFEST = PluginManifest(
    name="telegram-notifier",
    version="1.0.0",
    description="Telegram notification provider for escalations and alerts",
    slot=PluginSlot.NOTIFIER,
)

telegram_module = PluginModule(manifest=MANIFEST, create=create_plugin)


__all__ = ["TelegramNotifierPlugin", "create_plugin", "MANIFEST", "telegram_module"]
