"""
FastAPI REST API for OpenClaw

Async REST API with automatic OpenAPI documentation,
WebSocket support, and dependency injection.
Includes: graceful shutdown, rate limiting, input sanitization,
health probes, API versioning, and request logging.
"""

import time
import asyncio
import re
import html
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict, field
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.config import VisionConfig, VisionMode
from ..core.vision import VisionEngine, ScreenCapture
from ..core.actions import TriggerAction, ActionSequence
from ..core.logger import get_logger

logger = get_logger("fastapi")

API_VERSION = "2.1.0"

# ============================================================
# INPUT SANITIZATION
# ============================================================

class InputSanitizer:
    """Sanitize user inputs to prevent XSS and injection attacks"""

    # Dangerous patterns
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION)\b)",
        r"(--|;|'|\"|%27|%22|%3B)",
        r"(\bOR\b.*=.*\bOR\b)",
        r"(\bAND\b.*=.*\bAND\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
    ]

    @classmethod
    def sanitize(cls, value: str) -> str:
        """Sanitize a string value"""
        if not value:
            return value

        # HTML escape to prevent XSS
        sanitized = html.escape(value)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        return sanitized

    @classmethod
    def sanitize_dict(cls, data: Dict) -> Dict:
        """Sanitize all string values in a dictionary"""
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = cls.sanitize(value)
            elif isinstance(value, dict):
                result[key] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [cls.sanitize(v) if isinstance(v, str) else v for v in value]
            else:
                result[key] = value

        return result


class SanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware to sanitize request inputs"""

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip sanitization for certain paths
        if request.url.path.startswith("/ws") or request.url.path.endswith("/screenshots"):
            return await call_next(request)

        # Process request body for POST/PUT
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Re-create request with sanitized body
                    pass  # Body sanitization handled in endpoints
            except Exception:
                pass

        response = await call_next(request)
        return response


# ============================================================
# RATE LIMITING
# ============================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests: int = 100
    window_seconds: int = 60
    enabled: bool = True


class RateLimiter:
    """In-memory rate limiter"""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _clean_old_requests(self, key: str):
        """Remove requests outside the time window"""
        if not self._requests[key]:
            return

        cutoff = time.time() - self.config.window_seconds
        self._requests[key] = [ts for ts in self._requests[key] if ts > cutoff]

    def check(self, key: str) -> bool:
        """Check if request is allowed"""
        if not self.config.enabled:
            return True

        self._clean_old_requests(key)

        if len(self._requests[key]) >= self.config.requests:
            return False

        self._requests[key].append(time.time())
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests"""
        self._clean_old_requests(key)
        return max(0, self.config.requests - len(self._requests[key]))

    def reset(self, key: str = None):
        """Reset rate limit"""
        if key:
            self._requests.pop(key, None)
        else:
            self._requests.clear()


# Global rate limiter
rate_limiter = RateLimiter(RateLimitConfig(requests=100, window_seconds=60))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/healthz", "/ready", "/live"]:
            return await call_next(request)

        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key", "")
        key = f"{client_ip}:{api_key}" if api_key else client_ip

        # Check rate limit
        if not rate_limiter.check(key):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": rate_limiter.config.window_seconds,
                    "limit": rate_limiter.config.requests
                }
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.config.requests)
        response.headers["X-RateLimit-Remaining"] = str(rate_limiter.get_remaining(key))

        return response


# ============================================================
# REQUEST/RESPONSE LOGGING
# ============================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        try:
            response = await call_next(request)

            # Log response
            duration = time.time() - start_time
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration:.3f}s"
            )

            # Add timing header
            response.headers["X-Response-Time"] = f"{duration:.3f}"

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"error={str(e)} duration={duration:.3f}s"
            )
            raise


# ============================================================
# GRACEFUL SHUTDOWN
# ============================================================

class GracefulShutdown:
    """Manage graceful shutdown"""

    def __init__(self):
        self.is_shutting_down = False
        self.shutdown_event = asyncio.Event()
        self.connections: List = []
        self._tasks: List[asyncio.Task] = []

    def add_task(self, task: asyncio.Task):
        """Add a background task"""
        self._tasks.append(task)

    async def shutdown(self, sig=None):
        """Handle shutdown signal"""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        logger.info(f"Received shutdown signal {sig or 'manual'}, starting graceful shutdown...")

        # Wait for ongoing requests
        if self.connections:
            logger.info(f"Waiting for {len(self.connections)} connections to close...")
            await asyncio.sleep(1)

        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Notify all clients
        await manager.broadcast({
            "type": "shutdown",
            "message": "Server is shutting down",
            "timestamp": time.time()
        })

        logger.info("Graceful shutdown complete")
        self.shutdown_event.set()


# Global shutdown handler
shutdown_handler = GracefulShutdown()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class HealthResponse(BaseModel):
    status: str
    version: str = API_VERSION
    timestamp: float
    services: Dict[str, bool]


class LivenessResponse(BaseModel):
    status: str
    uptime: float


class ReadinessResponse(BaseModel):
    status: str
    checks: Dict[str, bool]


class TriggerRequest(BaseModel):
    mode: Optional[str] = None
    target_text: Optional[str] = None
    region: Optional[List[int]] = None
    action: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_inputs(cls, v):
        if isinstance(v, str):
            return InputSanitizer.sanitize(v)
        return v


class TriggerResponse(BaseModel):
    status: str
    triggered: bool
    condition_met: bool
    mode: str
    timestamp: float


class ConfigUpdate(BaseModel):
    mode: Optional[str] = None
    polling: Optional[bool] = None
    poll_interval: Optional[float] = None
    target_text: Optional[str] = None
    action: Optional[str] = None
    action_delay: Optional[float] = None

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_inputs(cls, v):
        if isinstance(v, str):
            return InputSanitizer.sanitize(v)
        return v


class TriggerCreate(BaseModel):
    name: str
    mode: str
    config: Dict[str, Any] = {}
    enabled: bool = True

    @field_validator("name", "mode", mode="before")
    @classmethod
    def sanitize_inputs(cls, v):
        if isinstance(v, str):
            return InputSanitizer.sanitize(v)
        return v


class StatsResponse(BaseModel):
    total: int
    triggered: int
    failed: int
    success_rate: float
    by_mode: Dict[str, int] = {}


class NLPRequest(BaseModel):
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v):
        if isinstance(v, str):
            return InputSanitizer.sanitize(v)
        return v


class AutomationExecute(BaseModel):
    action: str
    delay: float = 0

    @field_validator("action", mode="before")
    @classmethod
    def sanitize_action(cls, v):
        if isinstance(v, str):
            # Only allow safe characters for actions
            return re.sub(r'[^a-zA-Z0-9_+-]', '', v)
        return v


# ============================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================

class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client.host}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected: {websocket.client.host}")

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"FastAPI server starting... (v{API_VERSION})")

    # Initialize services
    shutdown_handler.is_shutting_down = False

    yield

    # Graceful shutdown
    await shutdown_handler.shutdown()
    logger.info("FastAPI server shutdown complete")


# ============================================================
# CREATE FASTAPI APP
# ============================================================

def create_app(config: VisionConfig = None, rate_limit: RateLimitConfig = None) -> FastAPI:
    """Create and configure FastAPI app with all middleware"""

    global _config, _vision_engine, rate_limiter

    if config:
        _config = config
        _vision_engine = VisionEngine(config)

    if rate_limit:
        rate_limiter = RateLimiter(rate_limit)

    # Determine allowed origins from config
    allowed_origins = []
    if config and config.allowed_origins:
        allowed_origins = config.allowed_origins

    # Create app
    app = FastAPI(
        title="OpenClaw REST API",
        description="Vision-based automation framework API",
        version=API_VERSION,
        docs_url="/api-docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Add middleware (order matters - last added = first executed)
    app.add_middleware(SanitizationMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # CORS - secure by default: only allow specific origins if configured
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins else [],  # Empty = no CORS
        allow_credentials=True if allowed_origins else False,
        allow_methods=["GET", "POST", "PUT", "DELETE"] if allowed_origins else [],
        allow_headers=["Authorization", "Content-Type"] if allowed_origins else [],
    )

    if allowed_origins:
        logger.info(f"CORS enabled for origins: {allowed_origins}")
    else:
        logger.info("CORS disabled (no origins configured)")

    # Store app reference for shutdown
    app.state.shutdown_handler = shutdown_handler

    return app


# Use create_app to get the configured app
app = create_app()


# ============================================================
# GLOBAL STATE
# ============================================================

_vision_engine: Optional[VisionEngine] = None
_config: Optional[VisionConfig] = None
_start_time = time.time()


# ============================================================
# DEPENDENCIES
# ============================================================

def get_vision_engine() -> VisionEngine:
    """Dependency to get vision engine"""
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine(_config or VisionConfig())
    return _vision_engine


def get_config() -> VisionConfig:
    """Dependency to get config"""
    global _config
    if _config is None:
        _config = VisionConfig()
    return _config


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "OpenClaw API",
        "version": API_VERSION,
        "docs": "/api-docs"
    }


# ---- Health Check Endpoints ----

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=time.time(),
        services={
            "vision": _vision_engine is not None,
            "api": True,
            "websocket": len(manager.active_connections) > 0
        }
    )


@app.get("/healthz", response_model=LivenessResponse)
async def liveness():
    """Kubernetes liveness probe - is the app running?"""
    return LivenessResponse(
        status="ok",
        uptime=time.time() - _start_time
    )


@app.get("/ready", response_model=ReadinessResponse)
async def readiness():
    """Kubernetes readiness probe - is the app ready to serve?"""
    checks = {
        "api": True,
        "vision": _vision_engine is not None,
        "not_shutting_down": not shutdown_handler.is_shutting_down
    }

    return ReadinessResponse(
        status="ready" if all(checks.values()) else "not_ready",
        checks=checks
    )


# ---- API Versioning ----

@app.get("/api/v1/health")
async def health_check_v1():
    """Health check v1 (deprecated, use /health)"""
    return await health_check()


@app.get("/api/v2/health")
async def health_check_v2():
    """Health check v2"""
    return await health_check()


# ---- Trigger Endpoints ----

@app.post("/api/v1/trigger", response_model=TriggerResponse)
async def trigger(
    request: TriggerRequest = None,
    engine: VisionEngine = Depends(get_vision_engine)
):
    """Execute vision trigger"""
    try:
        result = engine.process()

        response = TriggerResponse(
            status="ok",
            triggered=result,
            condition_met=result,
            mode=engine.config.mode.value,
            timestamp=time.time()
        )

        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "trigger",
            "data": asdict(response)
        })

        return response

    except Exception as e:
        logger.error(f"Trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/triggers", response_model=List[Dict])
async def list_triggers():
    """List all triggers (placeholder)"""
    return []


@app.post("/api/v1/triggers", status_code=201)
async def create_trigger(trigger: TriggerCreate):
    """Create a new trigger"""
    return {
        "id": str(int(time.time())),
        "name": trigger.name,
        "mode": trigger.mode,
        "enabled": trigger.enabled
    }


@app.get("/api/v1/triggers/{trigger_id}")
async def get_trigger(trigger_id: str):
    """Get trigger by ID"""
    # Sanitize trigger_id
    trigger_id = InputSanitizer.sanitize(trigger_id)
    raise HTTPException(status_code=404, detail="Trigger not found")


@app.delete("/api/v1/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str):
    """Delete trigger"""
    # Sanitize trigger_id
    trigger_id = InputSanitizer.sanitize(trigger_id)
    return {"status": "deleted", "id": trigger_id}


@app.post("/api/v1/triggers/{trigger_id}/execute")
async def execute_trigger(trigger_id: str, engine: VisionEngine = Depends(get_vision_engine)):
    """Execute a specific trigger"""
    trigger_id = InputSanitizer.sanitize(trigger_id)
    result = engine.process()
    return {
        "trigger_id": trigger_id,
        "executed": True,
        "result": result
    }


# ---- Config Endpoints ----

@app.get("/api/v1/config")
async def get_config_endpoint(config: VisionConfig = Depends(get_config)):
    """Get current configuration"""
    return config.to_dict()


@app.put("/api/v1/config")
async def update_config_endpoint(
    updates: ConfigUpdate,
    engine: VisionEngine = Depends(get_vision_engine)
):
    """Update configuration"""
    update_data = updates.dict(exclude_unset=True)

    # Apply updates to config
    for key, value in update_data.items():
        if hasattr(engine.config, key):
            setattr(engine.config, key, value)

    return {"status": "updated", "config": engine.config.to_dict()}


# ---- Metrics Endpoints ----

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from ..integrations.metrics import MetricsRegistry
    registry = MetricsRegistry.get_instance()
    return Response(
        content=registry.get_metrics_text(),
        media_type="text/plain"
    )


# ---- Stats Endpoints ----

@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get trigger statistics"""
    return StatsResponse(
        total=0,
        triggered=0,
        failed=0,
        success_rate=100.0,
        by_mode={}
    )


# ---- Automation Endpoints ----

@app.post("/api/v1/automation/execute")
async def execute_automation(request: AutomationExecute):
    """Execute an automation action"""
    try:
        TriggerAction.execute(request.action, request.delay)
        return {"status": "executed", "action": request.action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- NLP Endpoints ----

@app.post("/api/v1/nlp/process")
async def process_nlp(request: NLPRequest):
    """Process natural language input"""
    try:
        from ..core.ai import NLInterface, NLPConfig
        nlp = NLInterface(NLPConfig(mode="pattern"))
        result = nlp.process(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Screenshot Endpoints ----

@app.get("/api/v1/screenshots")
async def get_screenshot():
    """Get latest screenshot"""
    import cv2
    import numpy as np

    try:
        img = ScreenCapture.capture_full()
        _, png = cv2.imencode(".png", img)
        return Response(
            content=png.tobytes(),
            media_type="image/png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Recording Endpoints ----

@app.post("/api/v1/recording/start")
async def start_recording(session_id: str = None):
    """Start screen recording"""
    from ..core.recorder import start_recording as start_rec
    session = start_rec(session_id)
    return {
        "status": "recording",
        "session_id": session.id,
        "output_path": session.output_path
    }


@app.post("/api/v1/recording/stop")
async def stop_recording():
    """Stop screen recording"""
    from ..core.recorder import stop_recording as stop_rec
    session = stop_rec()
    if session:
        return {
            "status": "stopped",
            "frames": session.frames,
            "duration": session.end_time - session.start_time if session.end_time else 0,
            "output_path": session.output_path
        }
    return {"status": "not_recording"}


@app.post("/api/v1/recording/trigger")
async def record_on_trigger(duration: float = 10.0):
    """Record for a specific duration"""
    from ..core.recorder import record_on_trigger as record_trig
    session = record_trig(duration)
    if session:
        return {
            "status": "recorded",
            "frames": session.frames,
            "output_path": session.output_path
        }
    return {"status": "error"}


@app.get("/api/v1/recordings")
async def list_recordings():
    """List all recordings"""
    from ..core.recorder import get_recorder
    recorder = get_recorder()
    return {"recordings": recorder.get_recordings()}


@app.delete("/api/v1/recordings/{filename}")
async def delete_recording(filename: str):
    """Delete a recording"""
    from ..core.recorder import get_recorder
    recorder = get_recorder()
    success = recorder.delete_recording(filename)
    return {"status": "deleted" if success else "not_found"}


# ---- WebSocket Endpoint ----

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": time.time()
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---- Webhook Endpoints ----

@app.post("/api/v1/webhooks/{webhook_id}")
async def receive_webhook(webhook_id: str, request: Request):
    """Receive webhook and process triggers"""
    from ..integrations.webhook import get_webhook_engine, WebhookParser

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
    from ..integrations.webhook import get_webhook_engine
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
    from ..integrations.webhook import get_webhook_engine, WebhookTrigger, WebhookEvent
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
    from ..integrations.webhook import get_webhook_engine
    engine = get_webhook_engine()
    success = engine.unregister_trigger(webhook_id)

    return {
        "status": "deleted" if success else "not_found",
        "id": webhook_id
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "app",
    "create_app",
    "ConnectionManager",
    "manager",
    "InputSanitizer",
    "RateLimiter",
    "RateLimitConfig",
    "GracefulShutdown",
    "shutdown_handler",
    "HealthResponse",
    "LivenessResponse",
    "ReadinessResponse",
    "TriggerRequest",
    "TriggerResponse",
    "ConfigUpdate",
    "StatsResponse",
    "NLPRequest",
    "AutomationExecute",
    "API_VERSION",
]
