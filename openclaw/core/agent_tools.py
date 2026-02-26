"""
Agent Tool System for OpenClaw

Framework for AI agents to use tools - inspired by LangChain tools.
Enables agents to perform actions and get results.
"""

import time
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .logger import get_logger

logger = get_logger("agent_tools")


class ToolCallStatus(Enum):
    """Status of tool call"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    status: ToolCallStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Tool:
    """Tool definition for agents"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = False

    def __post_init__(self):
        self.is_async = inspect.iscoroutinefunction(self.func)

    def run(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        start = time.time()
        try:
            result = self.func(**kwargs)
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.SUCCESS,
                result=result,
                execution_time=time.time() - start
            )
        except Exception as e:
            logger.error(f"Tool {self.name} error: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=str(e),
                execution_time=time.time() - start
            )


class ToolRegistry:
    """Registry for agent tools"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def register_function(
        self,
        func: Callable,
        name: str = None,
        description: str = ""
    ):
        """Register a function as a tool"""
        tool = Tool(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            func=func
        )
        self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        """List all available tools"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.ERROR,
                error=f"Tool '{tool_name}' not found"
            )
        return tool.run(**kwargs)

    def remove(self, name: str) -> bool:
        """Remove a tool"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False


# Built-in tools for OpenClaw

def create_screen_capture_tool():
    """Create screen capture tool"""
    from .vision import ScreenCapture

    def capture_full(region: str = None):
        """Capture full screen or region"""
        if region:
            return ScreenCapture.capture_full()
        return ScreenCapture.capture_full()

    return Tool(
        name="capture_screen",
        description="Capture the screen and return image",
        func=capture_full
    )


def create_click_tool():
    """Create mouse click tool"""
    from .automation import click

    def click_at(x: int, y: int, button: str = "left"):
        """Click at coordinates"""
        click(x, y, button)
        return {"clicked": True, "x": x, "y": y}

    return Tool(
        name="click",
        description="Click at specified coordinates",
        func=click_at
    )


def create_type_tool():
    """Create text typing tool"""
    from .automation import type_text

    def type_text_tool(text: str, delay: float = 0):
        """Type text"""
        type_text(text, delay)
        return {"typed": True, "text": text}

    return Tool(
        name="type_text",
        description="Type text at cursor position",
        func=type_text_tool
    )


def create_press_key_tool():
    """Create key press tool"""
    from .automation import press

    def press_key(key: str):
        """Press a key"""
        press(key)
        return {"pressed": True, "key": key}

    return Tool(
        name="press_key",
        description="Press a keyboard key",
        func=press_key
    )


def create_ocr_tool():
    """Create OCR tool"""
    from .vision import OCREngine

    def extract_text(region: List[int] = None):
        """Extract text from screen"""
        from .vision import ScreenCapture

        if region:
            img = ScreenCapture.capture_region(region)
        else:
            img = ScreenCapture.capture_full()

        ocr = OCREngine()
        return ocr.extract_text(img)

    return Tool(
        name="extract_text",
        description="Extract text from screen using OCR",
        func=extract_text
    )


def create_vision_detect_tool():
    """Create object detection tool"""
    from .vision import ScreenCapture

    def detect_objects(class_names: List[str] = None):
        """Detect objects in screen"""
        from .ai_advanced import YOLODetector, YOLOVersion

        img = ScreenCapture.capture_full()
        detector = YOLODetector(YOLOVersion.YOLO11)
        detections = detector.detect(img, class_names)

        return [
            {
                "class": d.class_name,
                "confidence": d.confidence,
                "bbox": d.bbox
            }
            for d in detections
        ]

    return Tool(
        name="detect_objects",
        description="Detect objects in screen using YOLO",
        func=detect_objects
    )


def create_wait_tool():
    """Create wait tool"""
    import time

    def wait_tool(seconds: float):
        """Wait for specified seconds"""
        time.sleep(seconds)
        return {"waited": seconds}

    return Tool(
        name="wait",
        description="Wait for specified seconds",
        func=wait_tool
    )


def create_get_screen_size_tool():
    """Create screen size tool"""
    from .automation import get_screen_size

    def get_size():
        """Get screen size"""
        size = get_screen_size()
        return {"width": size[0], "height": size[1]} if size else None

    return Tool(
        name="get_screen_size",
        description="Get screen resolution",
        func=get_size
    )


# Default registry
_default_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get default tool registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
        # Register default tools
        _default_registry.register(create_screen_capture_tool())
        _default_registry.register(create_click_tool())
        _default_registry.register(create_type_tool())
        _default_registry.register(create_press_key_tool())
        _default_registry.register(create_ocr_tool())
        _default_registry.register(create_vision_detect_tool())
        _default_registry.register(create_wait_tool())
        _default_registry.register(create_get_screen_size_tool())
    return _default_registry


def register_tool(tool: Tool):
    """Quick register tool"""
    get_tool_registry().register(tool)


def execute_tool(tool_name: str, **kwargs) -> ToolResult:
    """Quick execute tool"""
    return get_tool_registry().execute(tool_name, **kwargs)


__all__ = [
    "ToolCallStatus",
    "ToolResult",
    "Tool",
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    "execute_tool",
]
