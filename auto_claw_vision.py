"""
Enhanced Auto Claw - Vision-enabled automation tool

Capabilities:
1. OCR (text recognition)
2. Region monitoring (change detection)
3. Object/element detection (template matching)
4. Color detection
5. Screenshot analysis

Usage:
    python3 auto_claw_vision.py --mode <mode> [options]

Modes:
    ocr           - Detect text on screen and trigger on match
    monitor       - Monitor region for visual changes
    template      - Detect UI elements via template matching
    color         - Detect colors in a region
    analyze       - General screenshot analysis
"""

import http.server
import socketserver
import os
import time
import json
import threading
import argparse
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
from enum import Enum
import base64

# Vision imports
import cv2
import numpy as np
from PIL import Image
import mss

# OCR - prefer easyocr, fallback to pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    try:
        import pytesseract
        PYTESSERACT_AVAILABLE = True
    except ImportError:
        PYTESSERACT_AVAILABLE = False

# Automation
import subprocess


# ============== Configuration ==============
PORT = 8765
DEBOUNCE_SECONDS = 3


# ============== Core Classes ==============

class VisionMode(Enum):
    OCR = "ocr"
    MONITOR = "monitor"
    TEMPLATE = "template"
    COLOR = "color"
    ANALYZE = "analyze"
    MULTI = "multi"


@dataclass
class TriggerCondition:
    """A single trigger condition"""
    mode: VisionMode
    target_text: Optional[str] = None
    region: Optional[Tuple[int, int, int, int]] = None
    change_threshold: float = 0.05
    template_path: Optional[str] = None
    template_threshold: float = 0.8
    target_color: Optional[Tuple[int, int, int]] = None
    color_tolerance: int = 30
    text_case_sensitive: bool = False


@dataclass
class VisionConfig:
    """Configuration for vision-based triggering"""
    mode: VisionMode
    # Multi-condition support
    conditions: List[TriggerCondition] = None
    condition_logic: str = "or"  # "and" or "or"
    # Polling mode
    polling: bool = False
    poll_interval: float = 0.5  # Check every X seconds
    # OCR settings
    target_text: Optional[str] = None
    text_case_sensitive: bool = False
    # Monitor settings
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    change_threshold: float = 0.05  # 5% change triggers
    # Template settings
    template_path: Optional[str] = None
    template_threshold: float = 0.8
    # Color settings
    target_color: Optional[Tuple[int, int, int]] = None  # BGR
    color_tolerance: int = 30
    # General
    capture_screen: int = 0  # Monitor number
    action: str = "alt+o"  # Keyboard shortcut to trigger
    action_delay: float = 1.5  # Wait before action

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []


class ScreenCapture:
    """Cross-platform screen capture wrapper"""

    @staticmethod
    def capture_region(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture screen or region returns BGR numpy array"""
        with mss.mss() as sct:
            if region:
                x, y, w, h = region
                monitor = {"top": y, "left": x, "width": w, "height": h}
            else:
                # Capture primary monitor
                monitor = sct.monitors[1]

            screenshot = sct.grab(monitor)
            # Convert to BGR numpy array (BGRA -> BGR)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def capture_full(screen: int = 1) -> np.ndarray:
        """Capture full screen"""
        with mss.mss() as sct:
            monitor = sct.monitors[screen]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


class VisionEngine:
    """Main vision processing engine"""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.last_capture: Optional[np.ndarray] = None
        # Track last captures per condition for monitor mode
        self.condition_captures: Dict[int, Optional[np.ndarray]] = {}

    def process(self) -> bool:
        """Process current frame and return True if trigger condition met"""
        # Handle multi-condition mode
        if self.config.mode == VisionMode.MULTI:
            return self._process_multi()

        method = getattr(self, f"_process_{self.config.mode.value}", None)
        if method:
            return method()
        return False

    def _process_single_condition(self, condition: TriggerCondition) -> bool:
        """Process a single condition and return True if met"""
        # Create a temporary config for this condition
        temp_config = VisionConfig(
            mode=condition.mode,
            target_text=condition.target_text,
            region=condition.region,
            change_threshold=condition.change_threshold,
            template_path=condition.template_path,
            template_threshold=condition.template_threshold,
            target_color=condition.target_color,
            color_tolerance=condition.color_tolerance,
            text_case_sensitive=condition.text_case_sensitive
        )

        engine = SingleConditionEngine(temp_config, self.condition_captures)
        return engine.process()

    def _process_multi(self) -> bool:
        """Process multiple conditions with AND/OR logic"""
        if not self.config.conditions:
            return False

        results = []
        for i, condition in enumerate(self.config.conditions):
            result = self._process_single_condition(condition)
            results.append(result)
            print(f"  Condition {i+1} ({condition.mode.value}): {result}")

        if self.config.condition_logic == "and":
            return all(results)
        else:  # "or"
            return any(results)

    def _process_ocr(self) -> bool:
        """OCR text detection"""
        if not self.config.target_text:
            return False

        img = ScreenCapture.capture_region(self.config.region)

        # Use EasyOCR if available, else pytesseract
        if EASYOCR_AVAILABLE:
            # Initialize reader (cached for performance)
            if not hasattr(self, '_ocr_reader'):
                self._ocr_reader = easyocr.Reader(['en'], gpu=True)
            results = self._ocr_reader.readtext(img)
            # Extract text from results
            text = ' '.join([r[1] for r in results])
        elif PYTESSERACT_AVAILABLE:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
        else:
            print("ERROR: No OCR engine available. Install easyocr or tesseract-ocr")
            return False

        # Check for match
        search_text = self.config.target_text
        if not self.config.text_case_sensitive:
            text = text.lower()
            search_text = search_text.lower()

        return search_text in text

    def _process_monitor(self) -> bool:
        """Region change detection"""
        if not self.config.region:
            return False

        current = ScreenCapture.capture_region(self.config.region)

        if self.last_capture is None:
            self.last_capture = current
            return False

        # Calculate difference
        diff = cv2.absdiff(self.last_capture, current)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        non_zero_ratio = np.count_nonzero(gray) / gray.size

        self.last_capture = current
        return non_zero_ratio > self.config.change_threshold

    def _process_template(self) -> bool:
        """Template matching for object detection"""
        if not self.config.template_path or not os.path.exists(self.config.template_path):
            return False

        # Load template
        template = cv2.imread(self.config.template_path)
        if template is None:
            return False

        # Capture screen
        img = ScreenCapture.capture_region(self.config.region)

        # Template matching
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        return max_val >= self.config.template_threshold

    def _process_color(self) -> bool:
        """Color detection in region"""
        if not self.config.target_color:
            return False

        img = ScreenCapture.capture_region(self.config.region)
        if img is None:
            return False

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create range around target color
        target = np.array(self.config.target_color)
        tolerance = self.config.color_tolerance

        lower = np.array([max(0, c - tolerance) for c in target])
        upper = np.array([min(255, c + tolerance) for c in target])

        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / mask.size

        return ratio > 0.01  # At least 1% of region

    def _process_analyze(self) -> bool:
        """General analysis - placeholder for custom logic"""
        img = ScreenCapture.capture_region(self.config.region)
        # TODO: Implement custom analysis logic
        # Could add: edge detection, contour finding, etc.
        return False


class SingleConditionEngine:
    """Engine for processing a single condition in multi-mode"""

    def __init__(self, config: VisionConfig, condition_captures: Dict[int, np.ndarray]):
        self.config = config
        self.condition_captures = condition_captures
        self.condition_id = id(config)

    def process(self) -> bool:
        """Process single condition"""
        method = getattr(self, f"_process_{self.config.mode.value}", None)
        if method:
            return method()
        return False

    # Class-level OCR reader cache
    _ocr_reader = None

    def _process_ocr(self) -> bool:
        if not self.config.target_text:
            return False
        img = ScreenCapture.capture_region(self.config.region)

        # Use EasyOCR if available
        if EASYOCR_AVAILABLE:
            if SingleConditionEngine._ocr_reader is None:
                SingleConditionEngine._ocr_reader = easyocr.Reader(['en'], gpu=True)
            results = SingleConditionEngine._ocr_reader.readtext(img)
            text = ' '.join([r[1] for r in results])
        elif PYTESSERACT_AVAILABLE:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
        else:
            print("ERROR: No OCR engine available")
            return False

        search_text = self.config.target_text
        if not self.config.text_case_sensitive:
            text = text.lower()
            search_text = search_text.lower()
        return search_text in text

    def _process_monitor(self) -> bool:
        if not self.config.region:
            return False
        current = ScreenCapture.capture_region(self.config.region)
        last = self.condition_captures.get(self.condition_id)
        if last is None:
            self.condition_captures[self.condition_id] = current
            return False
        diff = cv2.absdiff(last, current)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ratio = np.count_nonzero(gray) / gray.size
        self.condition_captures[self.condition_id] = current
        return ratio > self.config.change_threshold

    def _process_template(self) -> bool:
        if not self.config.template_path:
            return False
        template = cv2.imread(self.config.template_path)
        if template is None:
            return False
        img = ScreenCapture.capture_region(self.config.region)
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return max_val >= self.config.template_threshold

    def _process_color(self) -> bool:
        if not self.config.target_color:
            return False
        img = ScreenCapture.capture_region(self.config.region)
        if img is None:
            return False
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        target = np.array(self.config.target_color)
        tolerance = self.config.color_tolerance
        lower = np.array([max(0, c - tolerance) for c in target])
        upper = np.array([min(255, c + tolerance) for c in target])
        mask = cv2.inRange(hsv, lower, upper)
        return np.count_nonzero(mask) / mask.size > 0.01

    def _process_analyze(self) -> bool:
        return False


class AutomationAction:
    """Execute keyboard/mouse actions"""

    @staticmethod
    def execute(action: str, delay: float = 1.5):
        """Execute keyboard shortcut"""
        time.sleep(delay)
        cmd = f"xdotool key --clearmodifiers {action}"
        subprocess.run(cmd, shell=True)


class TriggerManager:
    """Manages trigger state and debouncing"""

    def __init__(self, debounce_seconds: float = DEBOUNCE_SECONDS):
        self.last_trigger = 0.0
        self.debounce_seconds = debounce_seconds
        self.trigger_count = 0
        self.last_result = False

    def should_trigger(self, condition: bool) -> bool:
        """Check if we should trigger (debounced)"""
        current_time = time.time()

        # Only trigger on rising edge (False -> True)
        if condition and not self.last_result:
            if current_time - self.last_trigger > self.debounce_seconds:
                self.last_trigger = current_time
                self.trigger_count += 1
                self.last_result = True
                return True
        elif not condition:
            self.last_result = False

        return False


# ============== HTTP Server ==============

class VisionHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that triggers vision analysis"""

    vision_engine: Optional[VisionEngine] = None
    trigger_manager: TriggerManager = None

    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[HTTP] {args[0]}")

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if not self.vision_engine:
            self.wfile.write(json.dumps({"status": "error", "message": "Vision not configured"}).encode())
            return

        # Run vision analysis
        result = self.vision_engine.process()
        triggered = self.trigger_manager.should_trigger(result)

        response = {
            "status": "ok",
            "triggered": triggered,
            "condition_met": result,
            "trigger_count": self.trigger_manager.trigger_count,
            "mode": self.vision_engine.config.mode.value
        }

        if triggered:
            print(f">>> VISION TRIGGER! Mode: {self.vision_engine.config.mode.value}")
            threading.Thread(
                target=AutomationAction.execute,
                args=(self.vision_engine.config.action, self.vision_engine.config.action_delay)
            ).start()

        self.wfile.write(json.dumps(response).encode())


class VisionHTTPServer:
    """HTTP server with vision capabilities"""

    def __init__(self, port: int, config: VisionConfig):
        self.port = port
        self.config = config
        self.server = None
        self.running = True

    def _polling_loop(self):
        """Continuous polling loop - runs in background thread"""
        engine = VisionEngine(self.config)
        trigger_mgr = TriggerManager(self.config.poll_interval * 2)

        print(f"[Polling] Started - checking every {self.config.poll_interval}s")

        while self.running:
            try:
                result = engine.process()
                triggered = trigger_mgr.should_trigger(result)

                if triggered:
                    print(f">>> POLL TRIGGER! Mode: {self.config.mode.value}")
                    threading.Thread(
                        target=AutomationAction.execute,
                        args=(self.config.action, self.config.action_delay)
                    ).start()

            except Exception as e:
                print(f"[Polling] Error: {e}")

            time.sleep(self.config.poll_interval)

    def start(self):
        """Start the HTTP server with optional polling"""
        VisionHTTPHandler.vision_engine = VisionEngine(self.config)
        VisionHTTPHandler.trigger_manager = TriggerManager()

        # Allow handler to access config
        global handler_config
        handler_config = self.config

        # Start polling if enabled
        if self.config.polling:
            poll_thread = threading.Thread(target=self._polling_loop, daemon=True)
            poll_thread.start()

        # Start HTTP server (skip if port is 0)
        if self.port > 0:
            self.server = socketserver.TCPServer(("", self.port), VisionHTTPHandler)
            print(f"Vision HTTP Server started on port {self.port}")
        else:
            print("HTTP server disabled (port 0)")

        print(f"Mode: {self.config.mode.value}")
        print(f"Action: {self.config.action}")
        print(f"Polling: {'enabled' if self.config.polling else 'disabled'}")

        if self.server:
            self.server.serve_forever()
        else:
            # Keep running for polling
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


# ============== CLI ==============

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Auto Claw with Vision")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["ocr", "monitor", "template", "color", "analyze", "multi"],
                        help="Vision mode (use 'multi' for multiple conditions)")

    # ========== Primary Condition Options ==========
    # OCR options
    parser.add_argument("--text", type=str, help="Target text to detect (OCR mode)")

    # Monitor options
    parser.add_argument("--region", type=str, help="Region as 'x,y,w,h'")
    parser.add_argument("--threshold", type=float, default=0.05, help="Change threshold (0-1)")

    # Template options
    parser.add_argument("--template", type=str, help="Path to template image")

    # Color options
    parser.add_argument("--color", type=str, help="Target color as 'B,G,R'")

    # ========== Secondary Condition Options (for multi mode) ==========
    parser.add_argument("--logic", type=str, default="or", choices=["and", "or"],
                        help="Logic for multi-mode: 'and' (all must match) or 'or' (any can match)")

    # Condition 2
    parser.add_argument("--cond2-mode", type=str,
                        choices=["ocr", "monitor", "template", "color", "analyze"],
                        help="Second condition mode (for multi mode)")
    parser.add_argument("--cond2-text", type=str, help="Text for condition 2 (OCR)")
    parser.add_argument("--cond2-region", type=str, help="Region for condition 2")
    parser.add_argument("--cond2-template", type=str, help="Template path for condition 2")
    parser.add_argument("--cond2-color", type=str, help="Color for condition 2")
    parser.add_argument("--cond2-threshold", type=float, default=0.05, help="Threshold for condition 2")

    # General options
    parser.add_argument("--port", type=int, default=PORT, help="HTTP server port (0 to disable)")
    parser.add_argument("--action", type=str, default="alt+o", help="Keyboard action")
    parser.add_argument("--delay", type=float, default=1.5, help="Action delay")
    parser.add_argument("--debounce", type=float, default=DEBOUNCE_SECONDS, help="Debounce seconds")

    # Polling options
    parser.add_argument("--poll", action="store_true", help="Enable continuous polling mode")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds")

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse region
    region = None
    if args.region:
        region = tuple(map(int, args.region.split(",")))

    # Parse color
    target_color = None
    if args.color:
        target_color = tuple(map(int, args.color.split(",")))

    # Build conditions list for multi-mode
    conditions = []

    # Primary condition (always included)
    primary_condition = TriggerCondition(
        mode=VisionMode(args.mode) if args.mode != "multi" else VisionMode.OCR,
        target_text=args.text,
        region=region,
        change_threshold=args.threshold,
        template_path=args.template,
        target_color=target_color
    )
    conditions.append(primary_condition)

    # Secondary condition (for multi-mode)
    if args.cond2_mode:
        cond2_region = None
        if args.cond2_region:
            cond2_region = tuple(map(int, args.cond2_region.split(",")))

        cond2_color = None
        if args.cond2_color:
            cond2_color = tuple(map(int, args.cond2_color.split(",")))

        secondary_condition = TriggerCondition(
            mode=VisionMode(args.cond2_mode),
            target_text=args.cond2_text,
            region=cond2_region,
            change_threshold=args.cond2_threshold,
            template_path=args.cond2_template,
            target_color=cond2_color
        )
        conditions.append(secondary_condition)

    # Determine mode
    if args.mode == "multi" or args.cond2_mode:
        # Multi-mode
        mode = VisionMode.MULTI
        # Use primary condition's mode if only one condition
        if len(conditions) == 1:
            mode = conditions[0].mode
    else:
        mode = VisionMode(args.mode)

    # Create config
    config = VisionConfig(
        mode=mode,
        conditions=conditions if mode == VisionMode.MULTI else [],
        condition_logic=args.logic,
        polling=args.poll,
        poll_interval=args.interval,
        target_text=args.text,
        region=region,
        change_threshold=args.threshold,
        template_path=args.template,
        target_color=target_color,
        action=args.action,
        action_delay=args.delay
    )

    # Start server
    server = VisionHTTPServer(args.port, config)
    server.start()


if __name__ == "__main__":
    main()
