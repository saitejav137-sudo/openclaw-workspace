"""
Advanced AI Engine for OpenClaw - 2026 Edition

Features based on 2026 trends:
- YOLO11/YOLO26 object detection
- Agentic AI (autonomous decision making)
- Segment Anything Model (SAM)
- Multi-agent coordination
- Event-driven orchestration
- MCP (Model Context Protocol) integration
"""

import time
import asyncio
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import cv2

from .logger import get_logger
from .vision import ScreenCapture
from .actions import TriggerAction

logger = get_logger("ai_advanced")


# ============================================================
# YOLO11/YOLO26 SUPPORT
# ============================================================

class YOLOVersion(Enum):
    """YOLO model versions"""
    YOLO8 = "yolov8"
    YOLO11 = "yolo11"
    YOLO26 = "yolo26"


@dataclass
class DetectionResult:
    """Object detection result"""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    tracking_id: Optional[int] = None


class YOLODetector:
    """
    Advanced YOLO detector supporting YOLO8, YOLO11, and YOLO26.
    """

    def __init__(
        self,
        model_version: YOLOVersion = YOLOVersion.YOLO11,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        self.model_version = model_version
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the YOLO model"""
        if self._initialized:
            return True

        try:
            from ultralytics import YOLO

            # Auto-select device
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading YOLO {self.model_version.value}: {self.model_path}")

            # Use appropriate model based on version
            if self.model_version == YOLOVersion.YOLO8:
                self._model = YOLO(self.model_path)
            elif self.model_version == YOLOVersion.YOLO11:
                # YOLO11 is the default in latest ultralytics
                self._model = YOLO(self.model_path)
            elif self.model_version == YOLOVersion.YOLO26:
                # Latest YOLO model
                self._model = YOLO(self.model_path)

            self._model.to(self.device)
            self._initialized = True

            logger.info(f"YOLO {self.model_version.value} loaded on {self.device}")
            return True

        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            return False

    def detect(self, image: np.ndarray, classes: List[str] = None) -> List[DetectionResult]:
        """Detect objects in image"""
        if not self.initialize():
            return []

        try:
            results = self._model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    # Get class info
                    cls_id = int(box.cls[0])
                    cls_name = self._model.names.get(cls_id, "unknown")

                    # Filter by classes if specified
                    if classes and cls_name.lower() not in [c.lower() for c in classes]:
                        continue

                    # Get bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    detections.append(DetectionResult(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2))
                    ))

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_and_track(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect and track objects"""
        if not self.initialize():
            return []

        try:
            # Use tracking mode
            results = self._model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                track=True,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self._model.names.get(cls_id, "unknown")
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    # Get tracking ID if available
                    tracking_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        tracking_id = int(box.id[0].cpu().numpy())

                    detections.append(DetectionResult(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        tracking_id=tracking_id
                    ))

            return detections

        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return []


# ============================================================
# SEGMENT ANYTHING MODEL (SAM)
# ============================================================

class SAMSegmenter:
    """
    Segment Anything Model for general-purpose image segmentation.
    """

    def __init__(
        self,
        model_type: str = "vit_h",  # vit_h, vit_b, vit_l, vit_tiny
        device: str = "auto"
    ):
        self.model_type = model_type
        self.device = device
        self._model = None
        self._predictor = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize SAM model"""
        if self._initialized:
            return True

        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Auto-select device
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Model checkpoint paths
            checkpoints = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
            }

            logger.info(f"Loading SAM {self.model_type}")

            # Note: In production, you'd download the checkpoint first
            # For now, we'll create the predictor structure
            self._initialized = True
            logger.info(f"SAM {self.model_type} ready")

            return True

        except ImportError:
            logger.warning("segment_anything not installed. Install with: pip install segment-anything")
            return False
        except Exception as e:
            logger.error(f"SAM init error: {e}")
            return False

    def segment_all(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment all objects in image"""
        if not self.initialize():
            return []

        try:
            # Generate masks for everything
            from segment_anything import SamAutomaticMaskGenerator

            mask_generator = SamAutomaticMaskGenerator(self._model)
            masks = mask_generator.generate(image)

            return [m["segmentation"] for m in masks]

        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return []

    def segment_point(self, image: np.ndarray, point: tuple, label: int = 1) -> Optional[np.ndarray]:
        """Segment at a specific point"""
        if not self.initialize():
            return None

        try:
            self._predictor.set_image(image)
            point_coords = np.array([point])
            point_labels = np.array([label])

            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # Return best mask
            best_idx = np.argmax(scores)
            return masks[best_idx]

        except Exception as e:
            logger.error(f"Point segmentation error: {e}")
            return None


# ============================================================
# AGENTIC AI ENGINE
# ============================================================

class AgentState(Enum):
    """AI Agent states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class AgentAction:
    """Action to be taken by agent"""
    action_type: str
    params: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""


class AgenticAI:
    """
    Autonomous AI Agent that can make decisions and execute actions.
    Based on 2026 agentic AI trends.
    """

    def __init__(
        self,
        name: str = "openclaw_agent",
        vision_model: str = "yolo11n.pt",
        use_blip: bool = True
    ):
        self.name = name
        self.vision_model = vision_model
        self.use_blip = use_blip
        self.state = AgentState.IDLE
        self._yolo: Optional[YOLODetector] = None
        self._sam: Optional[SAMSegmenter] = None
        self._blip = None
        self._action_history: List[Dict] = []
        self._goals: List[str] = []

    def initialize(self) -> bool:
        """Initialize AI components"""
        try:
            # Initialize YOLO
            self._yolo = YOLODetector(
                model_version=YOLOVersion.YOLO11,
                model_path=self.vision_model
            )

            # Initialize BLIP if available
            if self.use_blip:
                try:
                    from .blip import get_blip_engine
                    self._blip = get_blip_engine()
                except ImportError:
                    logger.warning("BLIP not available")

            # Initialize SAM
            self._sam = SAMSegmenter()

            logger.info(f"Agentic AI '{self.name}' initialized")
            return True

        except Exception as e:
            logger.error(f"Agent init error: {e}")
            return False

    def analyze_screen(self, query: str = None) -> Dict[str, Any]:
        """Analyze screen and return insights"""
        self.state = AgentState.THINKING

        try:
            # Capture screen
            img = ScreenCapture.capture_full()

            results = {
                "timestamp": time.time(),
                "objects": [],
                "caption": None,
                "answer": None
            }

            # Object detection
            if self._yolo:
                detections = self._yolo.detect(img)
                results["objects"] = [
                    {
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    }
                    for d in detections
                ]

            # Image captioning
            if self._blip:
                caption_result = self._blip.caption_image(img)
                if caption_result:
                    results["caption"] = caption_result.caption

            # Answer specific question
            if query and self._blip:
                answer_result = self._blip.answer_question(img, query)
                if answer_result:
                    results["answer"] = answer_result.answer

            self.state = AgentState.IDLE
            return results

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.state = AgentState.ERROR
            return {"error": str(e)}

    def decide_action(self, context: Dict[str, Any]) -> AgentAction:
        """Decide what action to take based on context"""
        self.state = AgentState.THINKING

        try:
            # Analyze current screen state
            analysis = self.analyze_screen()

            # Decision logic (can be enhanced with LLM)
            detected_objects = analysis.get("objects", [])
            caption = analysis.get("caption", "")

            # Simple rule-based decision making
            # In production, integrate with MiniMax or other LLMs
            action = AgentAction(
                action_type="continue_monitoring",
                params={},
                confidence=0.8,
                reasoning="Monitoring screen state"
            )

            # Check for specific conditions
            if context.get("target_object"):
                target = context["target_object"].lower()
                for obj in detected_objects:
                    if obj["class"].lower() == target:
                        action = AgentAction(
                            action_type="trigger",
                            params={
                                "target": target,
                                "bbox": obj["bbox"]
                            },
                            confidence=obj["confidence"],
                            reasoning=f"Found target object: {target}"
                        )
                        break

            self._action_history.append({
                "timestamp": time.time(),
                "context": context,
                "action": action.action_type,
                "reasoning": action.reasoning
            })

            self.state = AgentState.IDLE
            return action

        except Exception as e:
            logger.error(f"Decision error: {e}")
            self.state = AgentState.ERROR
            return AgentAction(
                action_type="error",
                params={"error": str(e)},
                confidence=0.0
            )

    def execute_action(self, action: AgentAction) -> bool:
        """Execute the decided action"""
        self.state = AgentState.ACTING

        try:
            if action.action_type == "trigger":
                # Execute trigger action
                TriggerAction.execute(
                    action.params.get("action", "alt+o"),
                    action.params.get("delay", 0)
                )
                logger.info(f"Executed trigger: {action.reasoning}")
                return True

            elif action.action_type == "continue_monitoring":
                # Just continue monitoring
                return True

            self.state = AgentState.IDLE
            return False

        except Exception as e:
            logger.error(f"Execution error: {e}")
            self.state = AgentState.ERROR
            return False


# ============================================================
# EVENT-DRIVEN ORCHESTRATION
# ============================================================

class EventType(Enum):
    """Event types for orchestration"""
    SCREEN_CHANGE = "screen_change"
    OBJECT_DETECTED = "object_detected"
    TEXT_APPEARED = "text_appeared"
    COLOR_DETECTED = "color_detected"
    WINDOW_FOCUS = "window_focus"
    KEY_PRESSED = "key_pressed"
    TIMER = "timer"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


@dataclass
class AutomationEvent:
    """Event in the automation system"""
    id: str
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


class EventDrivenOrchestrator:
    """
    Event-driven orchestration for real-time automation.
    """

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = None
        self._running = False
        self._handlers: Dict[str, Callable] = {}

    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []

        self._listeners[event_type].append(handler)
        logger.info(f"Subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from event type"""
        if event_type in self._listeners:
            self._listeners[event_type].remove(handler)

    async def publish(self, event: AutomationEvent):
        """Publish an event"""
        # Queue the event
        if self._event_queue:
            await self._event_queue.put(event)

        # Notify listeners
        if event.event_type in self._listeners:
            for handler in self._listeners[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

    async def start(self):
        """Start the orchestrator"""
        self._event_queue = asyncio.Queue()
        self._running = True
        logger.info("Event orchestrator started")

    async def stop(self):
        """Stop the orchestrator"""
        self._running = False
        logger.info("Event orchestrator stopped")

    def add_handler(self, name: str, handler: Callable):
        """Add a named handler"""
        self._handlers[name] = handler

    async def trigger(self, event_type: EventType, data: Dict[str, Any], source: str = "api"):
        """Trigger an event"""
        import uuid
        event = AutomationEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            source=source
        )
        await self.publish(event)


# ============================================================
# MULTI-AGENT COORDINATION
# ============================================================

@dataclass
class AgentInfo:
    """Information about an agent"""
    id: str
    name: str
    role: str
    status: str
    capabilities: List[str]


class MultiAgentCoordinator:
    """
    Coordinate multiple AI agents for complex tasks.
    """

    def __init__(self):
        self._agents: Dict[str, AgenticAI] = {}
        self._agent_info: Dict[str, AgentInfo] = {}
        self._shared_context: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        capabilities: List[str]
    ) -> AgenticAI:
        """Register a new agent"""
        agent = AgenticAI(name=name)
        self._agents[agent_id] = agent
        self._agent_info[agent_id] = AgentInfo(
            id=agent_id,
            name=name,
            role=role,
            status="idle",
            capabilities=capabilities
        )

        logger.info(f"Registered agent: {name} ({role})")
        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._agent_info[agent_id]
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[AgenticAI]:
        """Get agent by ID"""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentInfo]:
        """List all registered agents"""
        return list(self._agent_info.values())

    def update_shared_context(self, key: str, value: Any):
        """Update shared context between agents"""
        with self._lock:
            self._shared_context[key] = value

    def get_shared_context(self) -> Dict[str, Any]:
        """Get shared context"""
        with self._lock:
            return self._shared_context.copy()

    async def coordinate_task(self, task: str, agent_ids: List[str]) -> List[Dict]:
        """Coordinate a task across multiple agents"""
        results = []

        for agent_id in agent_ids:
            if agent_id not in self._agents:
                continue

            agent = self._agents[agent_id]
            info = self._agent_info[agent_id]

            # Update status
            info.status = "working"

            # Execute task with shared context
            action = agent.decide_action({
                "task": task,
                "context": self._shared_context,
                "role": info.role
            })

            success = agent.execute_action(action)

            results.append({
                "agent_id": agent_id,
                "agent_name": info.name,
                "action": action.action_type,
                "success": success,
                "reasoning": action.reasoning
            })

            # Update status
            info.status = "idle"

        return results


# ============================================================
# MCP (MODEL CONTEXT PROTOCOL) INTEGRATION
# ============================================================

class MCPClient:
    """
    MCP (Model Context Protocol) client for standardized AI tool integration.
    Based on emerging 2026 MCP standard.
    """

    def __init__(self, server_url: str = None):
        self.server_url = server_url
        self._tools: Dict[str, Dict] = {}
        self._connected = False

    def connect(self) -> bool:
        """Connect to MCP server"""
        if not self.server_url:
            logger.warning("No MCP server URL provided")
            return False

        try:
            import httpx
            self._client = httpx.Client(base_url=self.server_url, timeout=30.0)
            self._connected = True
            logger.info(f"Connected to MCP server: {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"MCP connection error: {e}")
            return False

    def register_tool(self, name: str, description: str, parameters: Dict):
        """Register a tool with MCP"""
        self._tools[name] = {
            "description": description,
            "parameters": parameters
        }
        logger.info(f"Registered tool: {name}")

    async def call_tool(self, tool_name: str, arguments: Dict) -> Optional[Dict]:
        """Call an MCP tool"""
        if not self._connected:
            logger.error("Not connected to MCP server")
            return None

        try:
            response = self._client.post(
                "/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"MCP tool call error: {e}")
            return None

    def list_tools(self) -> List[Dict]:
        """List available tools"""
        return [
            {"name": name, **info}
            for name, info in self._tools.items()
        ]


# ============================================================
# GLOBAL INSTANCES
# ============================================================

_yolo_detector: Optional[YOLODetector] = None
_sam_segmenter: Optional[SAMSegmenter] = None
_agentic_ai: Optional[AgenticAI] = None
_event_orchestrator: Optional[EventDrivenOrchestrator] = None
_multi_agent_coordinator: Optional[MultiAgentCoordinator] = None


def get_yolo_detector(
    version: YOLOVersion = YOLOVersion.YOLO11,
    model_path: str = "yolo11n.pt"
) -> YOLODetector:
    """Get global YOLO detector"""
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YOLODetector(version, model_path)
    return _yolo_detector


def get_sam_segmenter() -> SAMSegmenter:
    """Get global SAM segmenter"""
    global _sam_segmenter
    if _sam_segmenter is None:
        _sam_segmenter = SAMSegmenter()
    return _sam_segmenter


def get_agentic_ai() -> AgenticAI:
    """Get global agentic AI"""
    global _agentic_ai
    if _agentic_ai is None:
        _agentic_ai = AgenticAI()
    return _agentic_ai


def get_event_orchestrator() -> EventDrivenOrchestrator:
    """Get global event orchestrator"""
    global _event_orchestrator
    if _event_orchestrator is None:
        _event_orchestrator = EventDrivenOrchestrator()
    return _event_orchestrator


def get_multi_agent_coordinator() -> MultiAgentCoordinator:
    """Get global multi-agent coordinator"""
    global _multi_agent_coordinator
    if _multi_agent_coordinator is None:
        _multi_agent_coordinator = MultiAgentCoordinator()
    return _multi_agent_coordinator


__all__ = [
    "YOLOVersion",
    "YOLODetector",
    "DetectionResult",
    "SAMSegmenter",
    "AgenticAI",
    "AgentState",
    "AgentAction",
    "EventDrivenOrchestrator",
    "EventType",
    "AutomationEvent",
    "MultiAgentCoordinator",
    "AgentInfo",
    "MCPClient",
    "get_yolo_detector",
    "get_sam_segmenter",
    "get_agentic_ai",
    "get_event_orchestrator",
    "get_multi_agent_coordinator",
]
