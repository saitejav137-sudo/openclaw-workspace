"""
Webhook Automation for OpenClaw

Trigger automations from external services via webhooks.
Supports GitHub, GitLab, Slack, and custom webhooks.
"""

import time
import hashlib
import hmac
import json
import re
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import parse_qs

from ..core.logger import get_logger
from ..core.actions import TriggerAction

logger = get_logger("webhook")


class WebhookEvent(Enum):
    """Webhook event types"""
    # GitHub
    GITHUB_PUSH = "github.push"
    GITHUB_PULL_REQUEST = "github.pull_request"
    GITHUB_ISSUE = "github.issue"

    # GitLab
    GITLAB_PUSH = "gitlab.push"
    GITLAB_MERGE_REQUEST = "gitlab.merge_request"

    # Slack
    SLACK_MESSAGE = "slack.message"
    SLACK_COMMAND = "slack.command"

    # Custom
    CUSTOM = "custom"
    HTTP = "http"


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    secret: str = ""
    enabled: bool = True
    verify_ssl: bool = True
    timeout: float = 30.0


@dataclass
class WebhookRequest:
    """Parsed webhook request"""
    event: WebhookEvent
    payload: Dict[str, Any]
    headers: Dict[str, str]
    source: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WebhookTrigger:
    """Webhook trigger configuration"""
    id: str
    name: str
    event: WebhookEvent
    action: str
    action_delay: float = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    secret: str = ""
    created_at: float = field(default_factory=time.time)


class WebhookParser:
    """Parse and validate webhook requests"""

    @staticmethod
    def detect_event(headers: Dict[str, str], body: str) -> WebhookEvent:
        """Detect webhook event type from headers"""
        # GitHub
        if "x-github-event" in headers:
            event_type = headers.get("x-github-event", "")
            if event_type == "push":
                return WebhookEvent.GITHUB_PUSH
            elif event_type == "pull_request":
                return WebhookEvent.GITHUB_PULL_REQUEST
            elif event_type == "issues":
                return WebhookEvent.GITHUB_ISSUE

        # GitLab
        if "x-gitlab-event" in headers:
            event_type = headers.get("x-gitlab-event", "")
            if event_type == "Push Hook":
                return WebhookEvent.GITLAB_PUSH
            elif event_type == "Merge Request Hook":
                return WebhookEvent.GITLAB_MERGE_REQUEST

        # Slack
        if "x-slack-request-type" in headers or headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            return WebhookEvent.SLACK_MESSAGE

        # Default to custom
        return WebhookEvent.CUSTOM

    @staticmethod
    def verify_signature(secret: str, body: str, signature: str, algorithm: str = "sha256") -> bool:
        """Verify webhook signature"""
        if not secret or not signature:
            return True  # Skip verification if no secret

        try:
            if algorithm == "sha256":
                expected = hmac.new(
                    secret.encode(),
                    body.encode(),
                    hashlib.sha256
                ).hexdigest()
                return hmac.compare_digest(f"sha256={expected}", signature)
            elif algorithm == "sha1":
                expected = hmac.new(
                    secret.encode(),
                    body.encode(),
                    hashlib.sha1
                ).hexdigest()
                return hmac.compare_digest(f"sha1={expected}", signature)
        except Exception as e:
            logger.error(f"Signature verification error: {e}")

        return False

    @staticmethod
    def parse_payload(event: WebhookEvent, body: str) -> Dict[str, Any]:
        """Parse webhook payload"""
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # Try parsing as form data
            try:
                params = parse_qs(body)
                return {k: v[0] if len(v) == 1 else v for k, v in params.items()}
            except Exception:
                return {"raw": body}

    @staticmethod
    def extract_trigger_info(event: WebhookEvent, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trigger information from payload"""
        info = {"event": event.value}

        if event == WebhookEvent.GITHUB_PUSH:
            info.update({
                "repository": payload.get("repository", {}).get("full_name"),
                "branch": payload.get("ref", "").replace("refs/heads/", ""),
                "commits": payload.get("commits", []),
                "pusher": payload.get("pusher", {}).get("name"),
            })

        elif event == WebhookEvent.GITLAB_PUSH:
            info.update({
                "repository": payload.get("project", {}).get("path_with_namespace"),
                "branch": payload.get("ref", "").replace("refs/heads/", ""),
                "commits": payload.get("commits", []),
                "user": payload.get("user_name"),
            })

        elif event == WebhookEvent.SLACK_MESSAGE:
            info.update({
                "channel": payload.get("channel"),
                "user": payload.get("user"),
                "text": payload.get("text"),
                "command": payload.get("command"),
            })

        return info


class WebhookTriggerEngine:
    """Engine for processing webhook triggers"""

    def __init__(self):
        self.triggers: Dict[str, WebhookTrigger] = {}
        self.handlers: Dict[WebhookEvent, List[Callable]] = {
            event: [] for event in WebhookEvent
        }
        self._last_trigger_time: Dict[str, float] = {}

    def register_trigger(self, trigger: WebhookTrigger):
        """Register a webhook trigger"""
        self.triggers[trigger.id] = trigger
        logger.info(f"Registered webhook trigger: {trigger.name} ({trigger.event.value})")

    def unregister_trigger(self, trigger_id: str) -> bool:
        """Unregister a webhook trigger"""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            logger.info(f"Unregistered webhook trigger: {trigger_id}")
            return True
        return False

    def add_handler(self, event: WebhookEvent, handler: Callable):
        """Add event handler"""
        self.handlers[event].append(handler)

    def process_webhook(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process webhook and execute matching triggers"""
        results = []

        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue

            if trigger.event != event:
                continue

            # Check conditions
            if trigger.conditions and conditions:
                if not self._check_conditions(trigger.conditions, conditions):
                    continue

            # Debounce - prevent rapid re-triggering
            last_time = self._last_trigger_time.get(trigger.id, 0)
            if time.time() - last_time < 1.0:  # 1 second debounce
                logger.debug(f"Trigger {trigger.id} debounced")
                continue

            try:
                # Execute action
                TriggerAction.execute(trigger.action, trigger.action_delay)

                results.append({
                    "trigger_id": trigger.id,
                    "name": trigger.name,
                    "status": "success",
                    "action": trigger.action,
                    "timestamp": time.time()
                })

                self._last_trigger_time[trigger.id] = time.time()

                # Call handlers
                for handler in self.handlers.get(event, []):
                    try:
                        handler(event, payload, trigger)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")

            except Exception as e:
                logger.error(f"Trigger execution error: {e}")
                results.append({
                    "trigger_id": trigger.id,
                    "name": trigger.name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                })

        return results

    def _check_conditions(self, conditions: Dict, data: Dict) -> bool:
        """Check if conditions are met"""
        for key, expected in conditions.items():
            # Support dot notation for nested keys
            value = self._get_nested_value(data, key)

            if isinstance(expected, str):
                # Regex match
                if not re.match(expected, str(value)):
                    return False
            elif isinstance(expected, list):
                # Any of list
                if value not in expected:
                    return False
            else:
                # Exact match
                if value != expected:
                    return False

        return True

    def _get_nested_value(self, data: Dict, key: str) -> Any:
        """Get nested value using dot notation"""
        keys = key.split(".")
        value = data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value

    def list_triggers(self) -> List[Dict[str, Any]]:
        """List all webhook triggers"""
        return [
            {
                "id": t.id,
                "name": t.name,
                "event": t.event.value,
                "action": t.action,
                "enabled": t.enabled,
                "conditions": t.conditions
            }
            for t in self.triggers.values()
        ]


# Global instance
_webhook_engine: Optional[WebhookTriggerEngine] = None


def get_webhook_engine() -> WebhookTriggerEngine:
    """Get global webhook engine"""
    global _webhook_engine
    if _webhook_engine is None:
        _webhook_engine = WebhookTriggerEngine()
    return _webhook_engine


# ============================================================
# FASTAPI ENDPOINTS
# ============================================================

def create_webhook_endpoints(app, vision_engine=None):
    """Add webhook endpoints to FastAPI app"""

    @app.post("/api/v1/webhooks/{webhook_id}")
    async def receive_webhook(webhook_id: str, request: Request):
        """Receive webhook and process triggers"""
        body = await request.body()
        body_str = body.decode()

        # Get headers
        headers = {k.lower(): v for k, v in request.headers.items()}

        # Detect event
        event = WebhookParser.detect_event(headers, body_str)

        # Parse payload
        payload = WebhookParser.parse_payload(event, body_str)

        # Get webhook engine
        engine = get_webhook_engine()

        # Extract trigger info
        info = WebhookParser.extract_trigger_info(event, payload)

        # Process webhook
        results = engine.process_webhook(event, payload, headers, info)

        return {
            "status": "processed",
            "event": event.value,
            "triggers": results,
            "timestamp": time.time()
        }

    @app.get("/api/v1/webhooks")
    async def list_webhooks():
        """List all webhook triggers"""
        engine = get_webhook_engine()
        return {"triggers": engine.list_triggers()}

    @app.post("/api/v1/webhooks")
    async def create_webhook(
        name: str,
        event: str,
        action: str,
        conditions: Dict[str, Any] = {},
        enabled: bool = True
    ):
        """Create a new webhook trigger"""
        import uuid

        trigger = WebhookTrigger(
            id=str(uuid.uuid4()),
            name=name,
            event=WebhookEvent(event),
            action=action,
            conditions=conditions,
            enabled=enabled
        )

        engine = get_webhook_engine()
        engine.register_trigger(trigger)

        return {
            "id": trigger.id,
            "name": trigger.name,
            "event": trigger.event.value,
            "status": "created"
        }

    @app.delete("/api/v1/webhooks/{webhook_id}")
    async def delete_webhook(webhook_id: str):
        """Delete a webhook trigger"""
        engine = get_webhook_engine()
        success = engine.unregister_trigger(webhook_id)

        return {
            "status": "deleted" if success else "not_found",
            "id": webhook_id
        }

    return app


__all__ = [
    "WebhookEvent",
    "WebhookConfig",
    "WebhookRequest",
    "WebhookTrigger",
    "WebhookParser",
    "WebhookTriggerEngine",
    "get_webhook_engine",
    "create_webhook_endpoints",
]
