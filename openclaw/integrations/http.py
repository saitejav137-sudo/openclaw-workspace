"""HTTP server with authentication and rate limiting"""

import time
import hashlib
import threading
import json
import base64
import ssl
import os
from typing import Optional, Dict, Any, Callable, List
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


# Input validation functions
import re
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """Validate URL is safe and well-formed.

    Rejects:
    - Empty URLs
    - URLs > 2048 chars
    - Non-http/https schemes
    - IP addresses (including localhost)
    - file:// URLs
    """
    if not url or len(url) > 2048:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        if not parsed.netloc:
            return False
        # Reject IP addresses (including localhost numeric IPs)
        import re
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$|^localhost$|^::1$'
        if re.match(ip_pattern, parsed.netloc, re.IGNORECASE):
            return False
        # Reject file://
        if parsed.scheme == 'file':
            return False
        return True
    except Exception:
        return False


def validate_action(action: str) -> bool:
    """Validate browser action or keyboard shortcut.

    Accepts:
    - Browser actions: start, goto, click, click_text, type, input, submit, extract, extract_all, screenshot, info, close
    - Keyboard shortcuts: alt+o, ctrl+c, ctrl+s, shift+a, Return, etc.
    """
    if not action or not isinstance(action, str):
        return False

    # Browser actions
    valid_browser_actions = {
        "start", "goto", "click", "click_text", "type", "input",
        "submit", "extract", "extract_all", "screenshot", "info", "close"
    }
    if action in valid_browser_actions:
        return True

    # Keyboard shortcuts (case-sensitive: "alt+o", "ctrl+c", "shift+a", "Return", "Delete")
    import re
    # Modifier+key combos: ctrl+c, alt+o, shift+a
    modifier_pattern = r'^(ctrl|alt|shift|super|meta)\+[a-zA-Z0-9]$'
    # Single keys: a, b, 1, 2
    single_key_pattern = r'^[a-zA-Z0-9]$'
    # Special keys (exact case)
    special_key_pattern = r'^(Return|Space|Tab|Escape|Enter|Backspace|Delete|Insert|F1|F2|F3|F4|F5|F6|F7|F8|F9|F10|F11|F12|Up|Down|Left|Right|Home|End|PageUp|PageDown)$'

    if re.match(modifier_pattern, action) or re.match(single_key_pattern, action) or re.match(special_key_pattern, action):
        return True

    return False


def sanitize_string(s: str, max_length: int = 1000) -> str:
    """Sanitize string input"""
    if not isinstance(s, str):
        return ""
    # Remove null bytes and control characters
    s = s.replace('\x00', '')
    # Trim to max length
    return s[:max_length].strip()


def validate_selector(selector: str) -> bool:
    """Validate CSS selector"""
    if not selector or len(selector) > 500:
        return False
    # Basic validation - no dangerous characters
    dangerous = ['<script', 'javascript:', 'onerror', 'onclick']
    lower = selector.lower()
    return not any(d in lower for d in dangerous)


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
    allowed_origins: List[str] = []  # Configure allowed origins

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"HTTP: {args[0]}")

    def _get_allowed_origin(self) -> str:
        """Get the allowed origin for CORS based on request origin.

        Returns:
            The matching allowed origin, or empty string if not allowed
        """
        if not self.allowed_origins:
            # If no origins configured, don't send CORS headers (more secure)
            return ""

        request_origin = self.headers.get("Origin", "")
        if request_origin in self.allowed_origins:
            return request_origin

        # Also check without port for development
        if request_origin:
            # Extract origin without port
            origin_parts = request_origin.split("://")
            if len(origin_parts) == 2:
                host_part = origin_parts[1].split(":")[0]  # Remove port
                for allowed in self.allowed_origins:
                    if allowed.startswith("http://") or allowed.startswith("https://"):
                        allowed_host = allowed.split("://")[1].split(":")[0]
                        if host_part == allowed_host:
                            return request_origin

        return ""

    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")

        # Secure CORS: only allow specific origins
        allowed_origin = self._get_allowed_origin()
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Credentials", "true")

        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, status: int, message: str):
        """Send error response"""
        self._send_json({"status": "error", "message": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight with secure origin checking"""
        allowed_origin = self._get_allowed_origin()

        self.send_response(200)
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "3600")  # Cache preflight for 1 hour
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
        elif path == "/dashboard/modern" or path == "/dashboard/new":
            self._handle_modern_dashboard()
        elif path == "/api/config":
            self._handle_get_config()
        elif path == "/api/browser/info":
            self._handle_browser_info()
        elif path == "/api/browser/extract_all":
            self._handle_browser_extract_all()
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
        elif path == "/api/browser":
            self._handle_browser_action()
        elif path == "/api/smart":
            self._handle_smart_browse()
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

    def _handle_modern_dashboard(self):
        """Serve modern dashboard HTML"""
        try:
            from openclaw.ui.dashboard import MODERN_DASHBOARD_HTML
            html = MODERN_DASHBOARD_HTML
        except ImportError:
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

    def _handle_browser_info(self):
        """Get browser info"""
        try:
            from .browser_api import browser_info
            result = browser_info()
            self._send_json(result)
        except Exception as e:
            self._send_json({"success": False, "error": str(e)})

    def _handle_browser_extract_all(self):
        """Extract all text from browser"""
        try:
            from .browser_api import browser_extract_all
            result = browser_extract_all()
            self._send_json(result)
        except Exception as e:
            self._send_json({"success": False, "error": str(e)})

    def _handle_browser_action(self):
        """Handle browser action"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
            else:
                data = {}

            # Validate action
            action = sanitize_string(data.get("action", ""))
            if not action or not validate_action(action):
                self._send_json({"success": False, "error": "Invalid action"})
                return

            params = data.get("params", {})

            # Validate URL for goto action
            if action == "goto":
                url = params.get("url", "")
                if not validate_url(url):
                    self._send_json({"success": False, "error": "Invalid URL"})
                    return

            # Validate selector for click/type actions
            if action in ("click", "type"):
                selector = params.get("selector", "")
                if not validate_selector(selector):
                    self._send_json({"success": False, "error": "Invalid selector"})
                    return

            from .browser_api import execute_browser_action

            result = execute_browser_action(action, params)
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"success": False, "error": "Invalid JSON"})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)})

    def _handle_smart_browse(self):
        """Handle smart natural language browser control"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
            else:
                data = {}

            # Validate and sanitize instruction
            instruction = sanitize_string(data.get("instruction", ""), max_length=5000)
            if not instruction:
                self._send_json({"success": False, "error": "Empty instruction"})
                return

            from .browser_api import smart_browse

            result = smart_browse(instruction)
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"success": False, "error": "Invalid JSON"})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)})

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

    def __init__(self, port: int, config: VisionConfig, tls_enabled: bool = False,
                 cert_path: str = "", key_path: str = ""):
        self.port = port
        self.config = config
        self.tls_enabled = tls_enabled
        self.cert_path = cert_path
        self.key_path = key_path
        self.server: Optional[HTTPServer] = None
        self.vision_engine = VisionEngine(config)

        # Setup auth
        self.auth = APIKeyAuth(config.api_key)

        # Setup rate limiter - disabled by default (set to None for no limit)
        self.rate_limiter = None

        # Setup handler class with allowed origins from config
        VisionHTTPHandler.vision_engine = self.vision_engine
        VisionHTTPHandler.auth = self.auth
        VisionHTTPHandler.rate_limiter = self.rate_limiter
        VisionHTTPHandler.allowed_origins = config.allowed_origins or []

    @staticmethod
    def generate_self_signed_cert(cert_path: str = "/tmp/openclaw.crt",
                                   key_path: str = "/tmp/openclaw.key"):
        """Generate self-signed certificate for development"""
        import subprocess

        # Check if openssl is available
        try:
            subprocess.run(["openssl", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("OpenSSL not available, cannot generate cert")
            return None, None

        # Generate self-signed certificate
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost/O=OpenClaw/C=US"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Generated self-signed cert: {cert_path}")
            return cert_path, key_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate cert: {e}")
            return None, None

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for HTTPS"""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Load certificate and key
        if self.cert_path and self.key_path:
            if os.path.exists(self.cert_path) and os.path.exists(self.key_path):
                context.load_cert_chain(self.cert_path, self.key_path)
                logger.info(f"Loaded TLS cert: {self.cert_path}")
            else:
                logger.warning(f"Cert or key file not found, generating self-signed")
                cert, key = self.generate_self_signed_cert()
                if cert and key:
                    context.load_cert_chain(cert, key)
        else:
            # Generate self-signed for development
            cert, key = self.generate_self_signed_cert()
            if cert and key:
                context.load_cert_chain(cert, key)

        return context

    def start(self):
        """Start the HTTP server"""
        self.server = HTTPServer(("0.0.0.0", self.port), VisionHTTPHandler)

        # Wrap with TLS if enabled
        if self.tls_enabled:
            ssl_context = self._create_ssl_context()
            self.server.socket = ssl_context.wrap_socket(
                self.server.socket, server_side=True
            )
            protocol = "HTTPS"
        else:
            protocol = "HTTP"

        logger.info(f"{protocol} server started on 0.0.0.0:{self.port}")
        logger.info(f"API Key: {'Enabled' if self.auth.enabled else 'Disabled'}")
        logger.info(f"TLS: {'Enabled' if self.tls_enabled else 'Disabled'}")

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
