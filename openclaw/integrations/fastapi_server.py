"""
FastAPI REST API for OpenClaw

Async REST API with automatic OpenAPI documentation,
WebSocket support, and dependency injection.
"""

import time
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.config import VisionConfig, VisionMode
from ..core.vision import VisionEngine, ScreenCapture
from ..core.actions import TriggerAction, ActionSequence
from ..core.logger import get_logger

logger = get_logger("fastapi")


# Pydantic Models
class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    timestamp: float
    services: Dict[str, bool]


class TriggerRequest(BaseModel):
    mode: Optional[str] = None
    target_text: Optional[str] = None
    region: Optional[List[int]] = None
    action: Optional[str] = None


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


class TriggerCreate(BaseModel):
    name: str
    mode: str
    config: Dict[str, Any] = {}
    enabled: bool = True


class StatsResponse(BaseModel):
    total: int
    triggered: int
    failed: int
    success_rate: float
    by_mode: Dict[str, int] = {}


class NLPRequest(BaseModel):
    text: str


class AutomationExecute(BaseModel):
    action: str
    delay: float = 0


# Connection Manager for WebSocket
class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI server starting...")
    yield
    logger.info("FastAPI server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="OpenClaw REST API",
    description="Vision-based automation framework API",
    version="2.0.0",
    docs_url="/api-docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_vision_engine: Optional[VisionEngine] = None
_config: Optional[VisionConfig] = None


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


# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "OpenClaw API",
        "version": "2.0.0",
        "docs": "/api-docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=time.time(),
        services={
            "vision": _vision_engine is not None,
            "api": True,
            "websocket": True
        }
    )


@app.get("/api/v1/health")
async def health_check_v1():
    """Health check v1"""
    return await health_check()


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
    raise HTTPException(status_code=404, detail="Trigger not found")


@app.delete("/api/v1/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str):
    """Delete trigger"""
    return {"status": "deleted", "id": trigger_id}


@app.post("/api/v1/triggers/{trigger_id}/execute")
async def execute_trigger(trigger_id: str, engine: VisionEngine = Depends(get_vision_engine)):
    """Execute a specific trigger"""
    result = engine.process()
    return {
        "trigger_id": trigger_id,
        "executed": True,
        "result": result
    }


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


@app.post("/api/v1/automation/execute")
async def execute_automation(request: AutomationExecute):
    """Execute an automation action"""
    try:
        TriggerAction.execute(request.action, request.delay)
        return {"status": "executed", "action": request.action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/v1/screenshots")
async def get_screenshot():
    """Get latest screenshot"""
    import cv2
    import numpy as np

    try:
        img = ScreenCapture.capture_full()
        _, png = cv2.imencode(".png", img)
        return JSONResponse(
            content=png.tobytes(),
            media_type="image/png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": time.time()
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Alternative health for load balancers
@app.get("/healthz")
async def healthz():
    """Minimal health check for k8s"""
    return {"ok": True}


@app.get("/ready")
async def ready():
    """Readiness check"""
    return {"ready": True}


def create_app(config: VisionConfig = None) -> FastAPI:
    """Create and configure FastAPI app"""
    global _config, _vision_engine

    if config:
        _config = config
        _vision_engine = VisionEngine(config)

    return app


__all__ = [
    "app",
    "create_app",
    "ConnectionManager",
    "manager",
    "HealthResponse",
    "TriggerRequest",
    "TriggerResponse",
    "ConfigUpdate",
    "StatsResponse",
    "NLPRequest",
    "AutomationExecute",
]
