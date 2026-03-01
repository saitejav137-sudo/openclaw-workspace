"""
Screen Understanding Engine for OpenClaw

Goes beyond OCR to understand UI structure:
- Detect UI elements (buttons, text fields, menus, etc.)
- Build semantic element tree
- Map element descriptions to screen coordinates
- Enable "click the Settings button" style commands
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger

logger = get_logger("screen_understanding")


class UIElementType(Enum):
    """Types of UI elements."""
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    LABEL = "label"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    ICON = "icon"
    IMAGE = "image"
    LINK = "link"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    TAB = "tab"
    WINDOW = "window"
    TOOLBAR = "toolbar"
    SCROLLBAR = "scrollbar"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Screen region bounding box."""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, point: Tuple[int, int]) -> bool:
        px, py = point
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def overlaps(self, other: 'BoundingBox') -> bool:
        return not (
            self.x + self.width < other.x or
            other.x + other.width < self.x or
            self.y + self.height < other.y or
            other.y + other.height < self.y
        )


@dataclass
class UIElement:
    """A detected UI element on screen."""
    id: str
    element_type: UIElementType
    label: str
    bbox: BoundingBox
    confidence: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List['UIElement'] = field(default_factory=list)
    parent_id: Optional[str] = None
    is_interactive: bool = False
    is_visible: bool = True

    @property
    def center(self) -> Tuple[int, int]:
        return self.bbox.center


@dataclass
class ScreenState:
    """Complete understanding of the current screen state."""
    elements: List[UIElement] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    screen_size: Tuple[int, int] = (1920, 1080)
    active_window: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ElementDetector:
    """
    Detects UI elements from screen images.
    Uses heuristics as baseline, can be enhanced with YOLO/SAM models.
    """

    # Common UI element size heuristics (width_range, height_range)
    ELEMENT_HEURISTICS = {
        UIElementType.BUTTON: {"min_w": 50, "max_w": 300, "min_h": 25, "max_h": 60},
        UIElementType.TEXT_FIELD: {"min_w": 100, "max_w": 500, "min_h": 25, "max_h": 40},
        UIElementType.ICON: {"min_w": 16, "max_w": 64, "min_h": 16, "max_h": 64},
        UIElementType.MENU_ITEM: {"min_w": 100, "max_w": 400, "min_h": 20, "max_h": 40},
    }

    def detect_from_ocr(self, ocr_results: List[Dict]) -> List[UIElement]:
        """
        Build UI elements from OCR results.
        OCR results should have: text, x, y, width, height, confidence
        """
        elements = []

        for i, result in enumerate(ocr_results):
            text = result.get("text", "").strip()
            if not text:
                continue

            bbox = BoundingBox(
                x=int(result.get("x", 0)),
                y=int(result.get("y", 0)),
                width=int(result.get("width", 100)),
                height=int(result.get("height", 30))
            )

            # Classify element type based on heuristics
            element_type = self._classify_element(text, bbox)

            element = UIElement(
                id=f"elem_{i}_{hashlib.sha256(text.encode()).hexdigest()[:6]}",
                element_type=element_type,
                label=text,
                bbox=bbox,
                confidence=result.get("confidence", 0.5),
                is_interactive=element_type in (
                    UIElementType.BUTTON, UIElementType.TEXT_FIELD,
                    UIElementType.LINK, UIElementType.CHECKBOX,
                    UIElementType.DROPDOWN, UIElementType.TAB
                )
            )
            elements.append(element)

        return elements

    def detect_from_detections(self, detections: List[Dict]) -> List[UIElement]:
        """Build UI elements from YOLO/object detection results."""
        elements = []
        type_map = {
            "button": UIElementType.BUTTON,
            "textbox": UIElementType.TEXT_FIELD,
            "icon": UIElementType.ICON,
            "menu": UIElementType.MENU,
            "checkbox": UIElementType.CHECKBOX,
            "link": UIElementType.LINK,
        }

        for i, det in enumerate(detections):
            bbox_data = det.get("bbox", [0, 0, 100, 30])
            bbox = BoundingBox(
                x=int(bbox_data[0]),
                y=int(bbox_data[1]),
                width=int(bbox_data[2] - bbox_data[0]) if len(bbox_data) == 4 else 100,
                height=int(bbox_data[3] - bbox_data[1]) if len(bbox_data) == 4 else 30
            )

            class_name = det.get("class", "unknown")
            element_type = type_map.get(class_name, UIElementType.UNKNOWN)

            element = UIElement(
                id=f"det_{i}",
                element_type=element_type,
                label=det.get("label", class_name),
                bbox=bbox,
                confidence=det.get("confidence", 0.5),
                is_interactive=element_type in (
                    UIElementType.BUTTON, UIElementType.TEXT_FIELD,
                    UIElementType.LINK, UIElementType.CHECKBOX
                )
            )
            elements.append(element)

        return elements

    def _classify_element(self, text: str, bbox: BoundingBox) -> UIElementType:
        """Classify element type based on text and size heuristics."""
        text_lower = text.lower()

        # Button-like keywords
        button_keywords = [
            "ok", "cancel", "submit", "save", "close", "apply",
            "delete", "next", "back", "yes", "no", "confirm",
            "sign in", "log in", "register", "send", "search"
        ]
        if any(kw == text_lower or kw in text_lower for kw in button_keywords):
            return UIElementType.BUTTON

        # Link-like (contains URL patterns or is blue/underlined)
        if text.startswith("http") or text.startswith("www."):
            return UIElementType.LINK

        # Tab-like (short text, specific height)
        if len(text) < 20 and 25 <= bbox.height <= 45:
            return UIElementType.TAB

        # Short text = likely label or button
        if len(text) < 30:
            if bbox.height < 35:
                return UIElementType.LABEL
            return UIElementType.BUTTON

        return UIElementType.LABEL


class ElementFinder:
    """
    Find UI elements by description.
    Enables "click the Settings button" style commands.
    """

    def __init__(self):
        self._current_state: Optional[ScreenState] = None

    def update_state(self, state: ScreenState):
        """Update the current screen state."""
        self._current_state = state

    def find_by_label(
        self,
        label: str,
        element_type: UIElementType = None
    ) -> List[UIElement]:
        """Find elements matching a label (case-insensitive, fuzzy)."""
        if not self._current_state:
            return []

        label_lower = label.lower()
        matches = []

        for element in self._current_state.elements:
            # Exact match
            if element.label.lower() == label_lower:
                matches.append((element, 1.0))
                continue

            # Contains match
            if label_lower in element.label.lower():
                score = len(label_lower) / len(element.label)
                matches.append((element, score))
                continue

            # Partial word match
            label_words = set(label_lower.split())
            element_words = set(element.label.lower().split())
            if label_words & element_words:
                score = len(label_words & element_words) / len(label_words | element_words)
                if score > 0.3:
                    matches.append((element, score))

        # Filter by type if specified
        if element_type:
            matches = [(e, s) for e, s in matches if e.element_type == element_type]

        # Sort by relevance
        matches.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in matches]

    def find_interactive(self) -> List[UIElement]:
        """Find all interactive elements."""
        if not self._current_state:
            return []
        return [e for e in self._current_state.elements if e.is_interactive]

    def find_at_position(self, x: int, y: int) -> List[UIElement]:
        """Find elements at a specific screen position."""
        if not self._current_state:
            return []
        return [
            e for e in self._current_state.elements
            if e.bbox.contains((x, y))
        ]

    def find_near(
        self,
        element: UIElement,
        radius: int = 100
    ) -> List[UIElement]:
        """Find elements near a given element."""
        if not self._current_state:
            return []

        cx, cy = element.center
        nearby = []
        for e in self._current_state.elements:
            if e.id == element.id:
                continue
            ex, ey = e.center
            distance = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            if distance <= radius:
                nearby.append((e, distance))

        nearby.sort(key=lambda x: x[1])
        return [e for e, _ in nearby]


class ScreenUnderstanding:
    """
    Main screen understanding engine.

    Usage:
        su = ScreenUnderstanding()

        # Build state from OCR
        state = su.analyze_screen(ocr_results)

        # Find elements
        buttons = su.find("Settings")
        if buttons:
            click_target = buttons[0].center
    """

    def __init__(self):
        self.detector = ElementDetector()
        self.finder = ElementFinder()
        self._state_history: List[ScreenState] = []

    def analyze_from_ocr(self, ocr_results: List[Dict]) -> ScreenState:
        """Build screen understanding from OCR results."""
        elements = self.detector.detect_from_ocr(ocr_results)
        state = ScreenState(elements=elements)
        self.finder.update_state(state)
        self._state_history.append(state)

        logger.info(
            f"Screen analyzed: {len(elements)} elements "
            f"({sum(1 for e in elements if e.is_interactive)} interactive)"
        )
        return state

    def analyze_from_detections(self, detections: List[Dict]) -> ScreenState:
        """Build screen understanding from detection results."""
        elements = self.detector.detect_from_detections(detections)
        state = ScreenState(elements=elements)
        self.finder.update_state(state)
        self._state_history.append(state)
        return state

    def find(
        self,
        label: str,
        element_type: UIElementType = None
    ) -> List[UIElement]:
        """Find UI elements by label."""
        return self.finder.find_by_label(label, element_type)

    def get_click_target(self, label: str) -> Optional[Tuple[int, int]]:
        """Get click coordinates for a labeled element."""
        matches = self.find(label)
        if matches:
            return matches[0].center
        return None

    def describe_screen(self) -> str:
        """Generate a text description of the current screen."""
        state = self.finder._current_state
        if not state:
            return "No screen data available"

        lines = [f"Screen ({state.screen_size[0]}x{state.screen_size[1]}):"]

        interactive = [e for e in state.elements if e.is_interactive]
        labels = [e for e in state.elements if not e.is_interactive]

        if interactive:
            lines.append(f"\nInteractive elements ({len(interactive)}):")
            for e in interactive:
                lines.append(f"  [{e.element_type.value}] '{e.label}' at {e.center}")

        if labels:
            lines.append(f"\nText/Labels ({len(labels)}):")
            for e in labels[:20]:  # Limit to 20
                lines.append(f"  '{e.label}' at {e.center}")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get screen understanding stats."""
        state = self.finder._current_state
        if not state:
            return {"elements": 0, "states_analyzed": len(self._state_history)}

        by_type = {}
        for e in state.elements:
            t = e.element_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "elements": len(state.elements),
            "interactive": sum(1 for e in state.elements if e.is_interactive),
            "by_type": by_type,
            "states_analyzed": len(self._state_history)
        }


# ============== Global Instance ==============

_screen_engine: Optional[ScreenUnderstanding] = None


def get_screen_understanding() -> ScreenUnderstanding:
    """Get global screen understanding engine."""
    global _screen_engine
    if _screen_engine is None:
        _screen_engine = ScreenUnderstanding()
    return _screen_engine


__all__ = [
    "UIElementType",
    "BoundingBox",
    "UIElement",
    "ScreenState",
    "ElementDetector",
    "ElementFinder",
    "ScreenUnderstanding",
    "get_screen_understanding",
]
