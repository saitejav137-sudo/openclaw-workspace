"""
Alerting System for OpenClaw

Send alerts to various channels (Slack, PagerDuty, Email, etc.)
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger("openclaw.alerting")


class AlertSeverity(Enum):
    """Alert severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert definition"""
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "openclaw"


class AlertChannel(Enum):
    """Alert channel types"""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class BaseAlertChannel:
    """Base class for alert channels"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def send(self, alert: Alert) -> bool:
        """Send an alert"""
        raise NotImplementedError


class ConsoleAlertChannel(BaseAlertChannel):
    """Console alert channel"""

    def __init__(self):
        super().__init__(enabled=True)

    def send(self, alert: Alert) -> bool:
        """Print alert to console"""
        severity_emoji = {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }
        emoji = severity_emoji.get(alert.severity, "ℹ️")
        print(f"{emoji} [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
        return True


class SlackAlertChannel(BaseAlertChannel):
    """Slack alert channel"""

    def __init__(self, webhook_url: str = None, channel: str = None):
        super().__init__(enabled=bool(webhook_url))
        self.webhook_url = webhook_url
        self.channel = channel

    def send(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            import requests

            color = {
                AlertSeverity.DEBUG: "#808080",
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9800",
                AlertSeverity.ERROR: "#f44336",
                AlertSeverity.CRITICAL: "#9c27b0",
            }

            payload = {
                "attachments": [{
                    "color": color.get(alert.severity, "#808080"),
                    "title": alert.title,
                    "text": alert.message,
                    "footer": f"OpenClaw | {alert.source}",
                    "ts": alert.timestamp
                }]
            }

            if self.channel:
                payload["channel"] = self.channel

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Slack alert error: {e}")
            return False


class WebhookAlertChannel(BaseAlertChannel):
    """Generic webhook alert channel"""

    def __init__(self, webhook_url: str, method: str = "POST", headers: Dict = None):
        super().__init__(enabled=bool(webhook_url))
        self.webhook_url = webhook_url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, alert: Alert) -> bool:
        """Send alert to webhook"""
        if not self.enabled:
            return False

        try:
            import requests

            payload = {
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "timestamp": alert.timestamp,
                "tags": alert.tags
            }

            response = requests.request(
                self.method,
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )

            return response.status_code < 400

        except Exception as e:
            logger.error(f"Webhook alert error: {e}")
            return False


class AlertManager:
    """Manages alerts and channels"""

    _instance = None

    def __init__(self):
        self.channels: Dict[AlertChannel, BaseAlertChannel] = {
            AlertChannel.CONSOLE: ConsoleAlertChannel()
        }
        self.handlers: List[Callable] = []
        self.alert_history: List[Alert] = []
        self.max_history = 100
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'AlertManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_channel(self, channel_type: AlertChannel, channel: BaseAlertChannel):
        """Add an alert channel"""
        self.channels[channel_type] = channel
        logger.info(f"Alert channel added: {channel_type.value}")

    def remove_channel(self, channel_type: AlertChannel):
        """Remove an alert channel"""
        if channel_type in self.channels:
            del self.channels[channel_type]
            logger.info(f"Alert channel removed: {channel_type.value}")

    def register_handler(self, handler: Callable):
        """Register custom alert handler"""
        self.handlers.append(handler)

    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        tags: Dict[str, str] = None,
        source: str = "openclaw"
    ) -> bool:
        """Send an alert to all channels"""
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            tags=tags or {},
            source=source
        )

        # Store in history
        with self._lock:
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history:]

        # Send to channels
        success = True
        for channel_type, channel in self.channels.items():
            if channel.enabled:
                try:
                    result = channel.send(alert)
                    if not result:
                        success = False
                        logger.warning(f"Alert failed for channel: {channel_type.value}")
                except Exception as e:
                    logger.error(f"Channel {channel_type.value} error: {e}")
                    success = False

        # Call handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

        return success

    def get_alerts(
        self,
        severity: AlertSeverity = None,
        limit: int = 50
    ) -> List[Alert]:
        """Get alert history"""
        with self._lock:
            alerts = self.alert_history

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            return alerts[-limit:]

    def clear_history(self):
        """Clear alert history"""
        with self._lock:
            self.alert_history.clear()


# Convenience functions
def alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    **kwargs
) -> bool:
    """Send an alert"""
    return AlertManager.get_instance().send_alert(title, message, severity, **kwargs)


def alert_info(title: str, message: str, **kwargs) -> bool:
    """Send info alert"""
    return alert(title, message, AlertSeverity.INFO, **kwargs)


def alert_warning(title: str, message: str, **kwargs) -> bool:
    """Send warning alert"""
    return alert(title, message, AlertSeverity.WARNING, **kwargs)


def alert_error(title: str, message: str, **kwargs) -> bool:
    """Send error alert"""
    return alert(title, message, AlertSeverity.ERROR, **kwargs)


def alert_critical(title: str, message: str, **kwargs) -> bool:
    """Send critical alert"""
    return alert(title, message, AlertSeverity.CRITICAL, **kwargs)


# Export
__all__ = [
    "AlertSeverity",
    "Alert",
    "AlertChannel",
    "BaseAlertChannel",
    "ConsoleAlertChannel",
    "SlackAlertChannel",
    "WebhookAlertChannel",
    "AlertManager",
    "alert",
    "alert_info",
    "alert_warning",
    "alert_error",
    "alert_critical",
]
