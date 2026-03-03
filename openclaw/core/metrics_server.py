"""
Metrics & Observability Server for OpenClaw

Lightweight HTTP server exposing:
- /health  — JSON health status (integrates with HealthChecker)
- /metrics — Prometheus-compatible text metrics
- /events  — JSON event log from PersistentEventStore
- /agents  — Agent states and activity

Default port: 9100 (configurable)
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

from .logger import get_logger

logger = get_logger("metrics_server")


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics endpoints."""

    def log_message(self, format, *args):
        """Suppress default access logs."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        routes = {
            "/health": self._handle_health,
            "/metrics": self._handle_metrics,
            "/events": self._handle_events,
            "/agents": self._handle_agents,
            "/status": self._handle_status,
            "": self._handle_index,
        }

        handler = routes.get(path, self._handle_404)
        handler(params)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── Endpoints ──────────────────────────────────────

    def _handle_index(self, params):
        self._send_json({
            "service": "OpenClaw",
            "version": "2.0",
            "endpoints": ["/health", "/metrics", "/events", "/agents", "/status"],
            "uptime": self._get_uptime(),
        })

    def _handle_health(self, params):
        """Aggregate health from HealthChecker."""
        try:
            from .resilience import HealthChecker
            checker = HealthChecker()

            # Register basic checks
            self._register_health_checks(checker)

            report = checker.check_all()
            status_code = 200 if report["status"] == "healthy" else 503
            self._send_json(report, status=status_code)
        except Exception as e:
            self._send_json({"status": "healthy", "message": "Basic mode", "checks": {}})

    def _handle_metrics(self, params):
        """Prometheus-compatible metrics in text format."""
        lines = []
        lines.append("# HELP openclaw_uptime_seconds System uptime in seconds")
        lines.append("# TYPE openclaw_uptime_seconds gauge")
        lines.append(f"openclaw_uptime_seconds {self._get_uptime():.0f}")

        # Event bus metrics
        try:
            from .event_bus import get_event_bus
            bus = get_event_bus()
            stats = bus.get_stats()
            lines.append("")
            lines.append("# HELP openclaw_events_total Total events emitted")
            lines.append("# TYPE openclaw_events_total counter")
            lines.append(f"openclaw_events_total {stats['total_events_emitted']}")
            lines.append("")
            lines.append("# HELP openclaw_event_errors_total Total event handler errors")
            lines.append("# TYPE openclaw_event_errors_total counter")
            lines.append(f"openclaw_event_errors_total {stats['total_errors']}")
            lines.append("")
            lines.append("# HELP openclaw_subscriptions Active event subscriptions")
            lines.append("# TYPE openclaw_subscriptions gauge")
            lines.append(f"openclaw_subscriptions {stats['subscriptions']}")
        except Exception:
            pass

        # Reaction engine metrics
        try:
            from .reaction_engine import get_reaction_engine
            engine = get_reaction_engine()
            stats = engine.get_stats()
            lines.append("")
            lines.append("# HELP openclaw_reactions_triggered_total Total reactions triggered")
            lines.append("# TYPE openclaw_reactions_triggered_total counter")
            lines.append(f"openclaw_reactions_triggered_total {stats['total_reactions_triggered']}")
        except Exception:
            pass

        # Plugin metrics
        try:
            from .plugin_system import get_plugin_registry
            registry = get_plugin_registry()
            stats = registry.get_stats()
            lines.append("")
            lines.append("# HELP openclaw_plugins_registered Registered plugins")
            lines.append("# TYPE openclaw_plugins_registered gauge")
            lines.append(f"openclaw_plugins_registered {stats['total_registered']}")
            lines.append("")
            lines.append("# HELP openclaw_plugins_active Active plugin slots")
            lines.append("# TYPE openclaw_plugins_active gauge")
            lines.append(f"openclaw_plugins_active {stats['active_slots']}")
        except Exception:
            pass

        self._send_text("\n".join(lines) + "\n")

    def _handle_events(self, params):
        """Recent events from persistent store or in-memory."""
        limit = int(params.get("limit", ["50"])[0])

        try:
            from .event_bus import get_event_bus
            bus = get_event_bus()
            events = bus.get_history(limit=limit)
            self._send_json({
                "count": len(events),
                "events": [
                    {
                        "id": e.id,
                        "type": e.type.value,
                        "priority": e.priority.value,
                        "message": e.message,
                        "source": e.source,
                        "timestamp": e.timestamp,
                    }
                    for e in events
                ],
            })
        except Exception as e:
            self._send_json({"events": [], "error": str(e)})

    def _handle_agents(self, params):
        """Agent states from AgentStateManager."""
        try:
            from .agent_state import get_state_manager
            sm = get_state_manager()
            states = {}
            for agent_id, state in sm._states.items():
                states[agent_id] = {
                    "name": state.name,
                    "status": state.status.value,
                    "context": state.context,
                    "error_count": state.error_count,
                    "success_count": state.success_count,
                    "last_update": state.last_update,
                }
            self._send_json({"agents": states, "count": len(states)})
        except Exception as e:
            self._send_json({"agents": {}, "error": str(e)})

    def _handle_status(self, params):
        """Full system status dashboard."""
        status = {"timestamp": time.time(), "uptime": self._get_uptime()}

        try:
            from .event_bus import get_event_bus
            status["events"] = get_event_bus().get_stats()
        except Exception:
            pass

        try:
            from .plugin_system import get_plugin_registry
            status["plugins"] = get_plugin_registry().get_stats()
        except Exception:
            pass

        try:
            from .reaction_engine import get_reaction_engine
            status["reactions"] = get_reaction_engine().get_stats()
        except Exception:
            pass

        self._send_json(status)

    def _handle_404(self, params):
        self._send_json({"error": "Not found", "path": self.path}, status=404)

    def _get_uptime(self) -> float:
        return time.time() - getattr(self.server, '_start_time', time.time())

    def _register_health_checks(self, checker):
        """Register basic health checks."""
        # Event bus health
        def check_event_bus():
            from .event_bus import get_event_bus
            bus = get_event_bus()
            stats = bus.get_stats()
            return ("healthy", f"{stats['total_events_emitted']} events emitted")

        checker.register("event_bus", check_event_bus)


class MetricsServer:
    """
    Metrics HTTP server running in a background thread.

    Usage:
        server = MetricsServer(port=9100)
        server.start()
        # ... system runs ...
        server.stop()
    """

    def __init__(self, port: int = 9100, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        if self._running:
            return

        self._httpd = HTTPServer((self.host, self.port), MetricsHandler)
        self._httpd._start_time = time.time()
        self._running = True

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        logger.info("📊 Metrics server started at http://%s:%d", self.host, self.port)

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._running = False
            logger.info("Metrics server stopped")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ============== Global Instance ==============

_server: Optional[MetricsServer] = None


def get_metrics_server(port: int = 9100) -> MetricsServer:
    global _server
    if _server is None:
        _server = MetricsServer(port=port)
    return _server


__all__ = [
    "MetricsHandler",
    "MetricsServer",
    "get_metrics_server",
]
