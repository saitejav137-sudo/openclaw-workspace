"""HTTP server with authentication and rate limiting"""

import time
import hashlib
import threading
import json
import base64
from typing import Optional, Dict, Any, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from ..core.config import VisionConfig
from ..core.vision import VisionEngine
from ..core.actions import TriggerAction, ActionSequence, RetryConfig
from ..core.logger import get_logger

logger = get_logger("http")


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: int = 60, per: float = 60.0):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self._lock:
            current = time.time()
            elapsed = current - self.last_check

            # Refill tokens
            self.allowance += elapsed * (self.rate / self.per)
            self.last_check = current

            if self.allowance > self.rate:
                self.allowance = self.rate

            if self.allowance < 1.0:
                return False

            self.allowance -= 1.0
            return True

    def reset(self) -> None:
        """Reset the rate limiter"""
        with self._lock:
            self.allowance = self.rate
            self.last_check = time.time()


class APIKeyAuth:
    """API Key authentication"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.enabled = bool(api_key)

    def validate(self, request) -> bool:
        """Validate API key from request"""
        if not self.enabled:
            return True

        # Check header
        auth_header = request.headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            key = auth_header[7:]
            if key == self.api_key:
                return True

        # Check query param
        if hasattr(request, "path"):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(request.path)
            params = parse_qs(parsed.query)
            if "api_key" in params and params["api_key"][0] == self.api_key:
                return True

        return False


def require_auth(auth: APIKeyAuth):
    """Decorator to require authentication"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not auth.validate(self):
                self.send_error(401, "Unauthorized")
                return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def require_rate_limit(limiter: RateLimiter):
    """Decorator to require rate limiting"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not limiter.is_allowed():
                self.send_error(429, "Too Many Requests")
                return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


@dataclass
class TriggerResult:
    """Result of trigger check"""
    status: str
    triggered: bool
    condition_met: bool
    trigger_count: int
    mode: str
    message: Optional[str] = None
    screenshot_path: Optional[str] = None


class VisionHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for vision triggers"""

    vision_engine: Optional[VisionEngine] = None
    auth: Optional[APIKeyAuth] = None
    rate_limiter: Optional[RateLimiter] = None

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"HTTP: {args[0]}")

    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, status: int, message: str):
        """Send error response"""
        self._send_json({"status": "error", "message": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        # Rate limiting
        if self.rate_limiter and not self.rate_limiter.is_allowed():
            self._send_error(429, "Rate limit exceeded")
            return

        # Authentication
        if self.auth and self.auth.enabled:
            if not self.auth.validate(self):
                self._send_error(401, "Invalid API key")
                return

        # Route handling
        path = self.path.split("?")[0]

        if path == "/health":
            self._handle_health()
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/screenshot":
            self._handle_screenshot()
        elif path == "/dashboard":
            self._handle_dashboard()
        elif path == "/api/config":
            self._handle_get_config()
        else:
            # Default: run vision trigger
            self._handle_trigger()

    def do_POST(self):
        """Handle POST requests"""
        # Rate limiting
        if self.rate_limiter and not self.rate_limiter.is_allowed():
            self._send_error(429, "Rate limit exceeded")
            return

        path = self.path.split("?")[0]

        if path == "/api/trigger":
            self._handle_trigger()
        elif path == "/api/config":
            self._handle_set_config()
        else:
            self._send_error(404, "Not found")

    def _handle_health(self):
        """Health check endpoint"""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "vision": self.vision_engine is not None,
                "auth": self.auth.enabled if self.auth else False,
                "rate_limit": self.rate_limiter is not None
            }
        }
        self._send_json(health)

    def _handle_stats(self):
        """Get trigger statistics"""
        # This would integrate with database
        self._send_json({
            "total": 0,
            "triggered": 0,
            "failed": 0,
            "success_rate": 0.0
        })

    def _handle_screenshot(self):
        """Get latest screenshot"""
        import os
        import glob

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()

        try:
            record_dir = "/tmp/openclaw_records"
            files = sorted(glob.glob(f"{record_dir}/*.png"), key=os.path.getmtime)
            if files:
                with open(files[-1], "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"")
        except:
            self.wfile.write(b"")

    def _handle_trigger(self):
        """Run vision trigger"""
        if not self.vision_engine:
            self._send_json({
                "status": "error",
                "message": "Vision not configured"
            }, 500)
            return

        try:
            # Run vision analysis
            result = self.vision_engine.process()

            response = {
                "status": "ok",
                "triggered": result,
                "condition_met": result,
                "mode": self.vision_engine.config.mode.value
            }

            if result:
                logger.info(f">>> VISION TRIGGER! Mode: {self.vision_engine.config.mode.value}")

                # Execute action
                TriggerAction.execute(
                    self.vision_engine.config.action,
                    self.vision_engine.config.action_delay
                )

                # Execute action sequence if configured
                if self.vision_engine.config.action_sequence:
                    seq = ActionSequence()
                    seq.execute_async(self.vision_engine.config.action_sequence)

            self._send_json(response)

        except Exception as e:
            logger.error(f"Trigger error: {e}")
            self._send_json({
                "status": "error",
                "message": str(e)
            }, 500)

    def _handle_dashboard(self):
        """Serve dashboard HTML"""
        html = self._generate_dashboard()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _handle_get_config(self):
        """Get current configuration"""
        if not self.vision_engine:
            self._send_json({"error": "No config"})
            return

        self._send_json(self.vision_engine.config.to_dict())

    def _handle_set_config(self):
        """Update configuration"""
        # Would update config
        self._send_json({"status": "ok"})

    def _generate_dashboard(self) -> str:
        """Generate dashboard HTML"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>OpenClaw Vision</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .stats { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
        .stat-box { background: #2a2a2a; padding: 20px; border-radius: 8px; min-width: 150px; }
        .stat-box h3 { margin: 0 0 10px 0; color: #888; }
        .stat-box .value { font-size: 32px; font-weight: bold; }
        .green { color: #00ff88; }
        .red { color: #ff4444; }
        .trigger-btn { background: #00ff88; color: #000; padding: 15px 30px; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; }
        .trigger-btn:hover { background: #00cc6a; }
        .screenshot { margin-top: 20px; }
        .screenshot img { max-width: 100%; border-radius: 8px; }
        .log { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-top: 20px; max-height: 300px; overflow-y: auto; }
        .log pre { margin: 0; font-family: monospace; color: #888; }
        .status-badge { display: inline-block; padding: 5px 10px; border-radius: 4px; font-size: 12px; }
        .status-ok { background: #00ff88; color: #000; }
        .status-error { background: #ff4444; color: #fff; }
    </style>
</head>
<body>
    <h1>OpenClaw Vision Dashboard</h1>
    <div class="status">
        <span class="status-badge status-ok" id="healthStatus">Healthy</span>
    </div>
    <button class="trigger-btn" onclick="trigger()">Manual Trigger</button>
    <div class="stats">
        <div class="stat-box">
            <h3>Total</h3>
            <div class="value" id="total">-</div>
        </div>
        <div class="stat-box">
            <h3>Triggered</h3>
            <div class="value green" id="triggered">-</div>
        </div>
        <div class="stat-box">
            <h3>Failed</h3>
            <div class="value red" id="failed">-</div>
        </div>
        <div class="stat-box">
            <h3>Success Rate</h3>
            <div class="value" id="rate">-</div>
        </div>
    </div>
    <div class="screenshot">
        <h3>Latest Screenshot</h3>
        <img id="screenshot" src="/api/screenshot" alt="No screenshot">
    </div>
    <div class="log">
        <h3>Activity Log</h3>
        <pre id="log"></pre>
    </div>
    <script>
        const API_KEY = new URLSearchParams(window.location.search).get('api_key') || '';

        function addApiKey(url) {
            if (!API_KEY) return url;
            const separator = url.includes('?') ? '&' : '?';
            return url + separator + 'api_key=' + API_KEY;
        }

        function loadStats() {
            fetch(addApiKey('/api/stats'))
                .then(r => r.json())
                .then(data => {
                    if (data.total !== undefined) {
                        document.getElementById('total').textContent = data.total;
                        document.getElementById('triggered').textContent = data.triggered;
                        document.getElementById('failed').textContent = data.failed;
                        document.getElementById('rate').textContent = data.success_rate.toFixed(1) + '%';
                    }
                })
                .catch(() => {});
        }
        function trigger() {
            fetch(addApiKey('/api/trigger'))
                .then(r => r.json())
                .then(data => {
                    console.log('Triggered:', data);
                    loadStats();
                })
                .catch(err => console.error(err));
        }
        function loadHealth() {
            fetch(addApiKey('/health'))
                .then(r => r.json())
                .then(data => {
                    const status = document.getElementById('healthStatus');
                    status.textContent = data.status;
                    status.className = 'status-badge ' + (data.status === 'healthy' ? 'status-ok' : 'status-error');
                })
                .catch(() => {});
        }
        loadStats();
        loadHealth();
        setInterval(loadStats, 5000);
        setInterval(loadHealth, 10000);
    </script>
</body>
</html>"""


class VisionHTTPServer:
    """HTTP server with vision capabilities"""

    def __init__(self, port: int, config: VisionConfig):
        self.port = port
        self.config = config
        self.server: Optional[HTTPServer] = None
        self.vision_engine = VisionEngine(config)

        # Setup auth
        self.auth = APIKeyAuth(config.api_key)

        # Setup rate limiter
        self.rate_limiter = RateLimiter(
            rate=config.rate_limit,
            per=60.0
        )

        # Setup handler class
        VisionHTTPHandler.vision_engine = self.vision_engine
        VisionHTTPHandler.auth = self.auth
        VisionHTTPHandler.rate_limiter = self.rate_limiter

    def start(self):
        """Start the HTTP server"""
        self.server = HTTPServer(("", self.port), VisionHTTPHandler)
        logger.info(f"HTTP server started on port {self.port}")
        logger.info(f"API Key: {'Enabled' if self.auth.enabled else 'Disabled'}")
        logger.info(f"Rate Limit: {self.config.rate_limit} req/min")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("HTTP server stopped")
            self.stop()

    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server = None


# Export classes
__all__ = [
    "RateLimiter",
    "APIKeyAuth",
    "VisionHTTPServer",
    "VisionHTTPHandler",
]
