"""
Enhanced Auto Claw - Vision-enabled automation tool

Capabilities:
1. OCR (text recognition)
2. Region monitoring (change detection)
3. Object/element detection (template matching)
4. Color detection
5. Screenshot analysis
6. Adaptive polling
7. Webhook notifications
8. Desktop notifications
9. Mouse actions
10. Action sequences
11. YAML configuration
12. Configuration profiles

Usage:
    python3 auto_claw_vision.py --mode <mode> [options]
    python3 auto_claw_vision.py --config <config.yaml>

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
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
from enum import Enum
import base64
import yaml
import requests

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

# Notifications
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

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
    # Adaptive polling
    adaptive_polling: bool = False
    idle_interval: float = 2.0  # Interval when idle
    active_interval: float = 0.2  # Interval when active
    # OCR settings
    target_text: Optional[str] = None
    text_case_sensitive: bool = False
    # Monitor settings
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    change_threshold: float = 0.05  # 5% change triggers
    # Template settings
    template_path: Optional[str] = None
    template_threshold: float = 0.8
    # Multiple templates
    templates: List[str] = None  # Multiple template paths
    # Color settings
    target_color: Optional[Tuple[int, int, int]] = None  # BGR
    color_tolerance: int = 30
    # Logging
    log_file: Optional[str] = None
    log_enabled: bool = False
    # Screen recording on trigger
    record_on_trigger: bool = False
    record_dir: str = "/tmp/auto_claw_records"
    # Webhooks
    webhook_url: Optional[str] = None
    webhook_enabled: bool = False
    # Desktop notifications
    notify_enabled: bool = False
    notify_title: str = "OpenClaw"
    # Mouse actions
    mouse_actions: List[Dict] = None  # [{"action": "click", "x": 100, "y": 200}, ...]
    # Action sequences
    action_sequence: List[Dict] = None  # [{"action": "key", "key": "alt+o", "delay": 1.5}, ...]
    # Configuration profiles
    profile_name: Optional[str] = None
    config_file: Optional[str] = None  # YAML config file path
    # Dynamic reload
    watch_config: bool = False
    # General
    capture_screen: int = 0  # Monitor number
    action: str = "alt+o"  # Keyboard shortcut to trigger
    action_delay: float = 1.5  # Wait before action

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.templates is None:
            self.templates = []
        if self.mouse_actions is None:
            self.mouse_actions = []
        if self.action_sequence is None:
            self.action_sequence = []

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "mode": self.mode.value if isinstance(self.mode, VisionMode) else self.mode,
            "polling": self.polling,
            "poll_interval": self.poll_interval,
            "adaptive_polling": self.adaptive_polling,
            "target_text": self.target_text,
            "region": self.region,
            "template_path": self.template_path,
            "templates": self.templates,
            "target_color": self.target_color,
            "action": self.action,
            "webhook_url": self.webhook_url,
            "notify_enabled": self.notify_enabled,
            "mouse_actions": self.mouse_actions,
            "action_sequence": self.action_sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VisionConfig':
        """Create config from dictionary"""
        config = cls(
            mode=VisionMode(data.get("mode", "ocr")),
            polling=data.get("polling", False),
            poll_interval=data.get("poll_interval", 0.5),
            adaptive_polling=data.get("adaptive_polling", False),
            target_text=data.get("target_text"),
            region=tuple(data["region"]) if data.get("region") else None,
            template_path=data.get("template_path"),
            templates=data.get("templates", []),
            target_color=tuple(data["target_color"]) if data.get("target_color") else None,
            action=data.get("action", "alt+o"),
            webhook_url=data.get("webhook_url"),
            webhook_enabled=data.get("webhook_enabled", False),
            notify_enabled=data.get("notify_enabled", False),
            mouse_actions=data.get("mouse_actions", []),
            action_sequence=data.get("action_sequence", []),
        )
        return config


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
        """Template matching for object detection - supports multiple templates"""
        # Collect all template paths
        template_paths = []
        if self.config.template_path and os.path.exists(self.config.template_path):
            template_paths.append(self.config.template_path)
        if self.config.templates:
            for t in self.config.templates:
                if os.path.exists(t):
                    template_paths.append(t)

        if not template_paths:
            return False

        # Capture screen
        img = ScreenCapture.capture_region(self.config.region)

        # Check each template
        for template_path in template_paths:
            template = cv2.imread(template_path)
            if template is None:
                continue

            # Template matching
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= self.config.template_threshold:
                return True

        return False

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


class Logger:
    """File and console logging"""

    _instance = None

    @classmethod
    def get_instance(cls, log_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls(log_file)
        return cls._instance

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            # Create directory if needed
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Log a message to file and/or console"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"

        # Console output
        print(log_line)

        # File output
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(log_line + "\n")
            except Exception as e:
                print(f"Log write error: {e}")

    def info(self, msg):
        self.log(msg, "INFO")

    def warning(self, msg):
        self.log(msg, "WARNING")

    def error(self, msg):
        self.log(msg, "ERROR")


class ScreenRecorder:
    """Records screen on trigger events"""

    def __init__(self, record_dir: str = "/tmp/auto_claw_records"):
        self.record_dir = record_dir
        os.makedirs(record_dir, exist_ok=True)

    def capture_trigger(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Capture screenshot and save to file, return path"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trigger_{timestamp}.png"
        filepath = os.path.join(self.record_dir, filename)

        img = ScreenCapture.capture_region(region)
        cv2.imwrite(filepath, img)

        return filepath

    def capture_sequence(self, count: int = 3, delay: float = 0.1,
                        region: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
        """Capture sequence of screenshots"""
        paths = []
        for i in range(count):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"trigger_{timestamp}_{i}.png"
            filepath = os.path.join(self.record_dir, filename)

            img = ScreenCapture.capture_region(region)
            cv2.imwrite(filepath, img)
            paths.append(filepath)

            if i < count - 1:
                time.sleep(delay)

        return paths


class WebhookNotifier:
    """Send HTTP webhooks on trigger events"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send(self, data: Dict):
        """Send webhook notification"""
        if not self.enabled or not self.webhook_url:
            return

        try:
            payload = {
                "event": "trigger",
                "timestamp": time.time(),
                "data": data
            }
            # Fire and forget - don't block
            threading.Thread(
                target=requests.post,
                args=(self.webhook_url,),
                kwargs={"json": payload, "timeout": 5}
            ).start()
            print(f"[Webhook] Sent to {self.webhook_url}")
        except Exception as e:
            print(f"[Webhook] Error: {e}")


class NotificationManager:
    """Send desktop notifications"""

    def __init__(self, enabled: bool = False, title: str = "OpenClaw"):
        self.enabled = enabled
        self.title = title

    def notify(self, message: str):
        """Send desktop notification"""
        if not self.enabled or not PLYER_AVAILABLE:
            return

        try:
            notification.notify(
                title=self.title,
                message=message,
                timeout=5
            )
            print(f"[Notify] {message}")
        except Exception as e:
            print(f"[Notify] Error: {e}")


class MouseController:
    """Mouse automation actions"""

    @staticmethod
    def move(x: int, y: int):
        """Move mouse to coordinates"""
        subprocess.run(["xdotool", "mousemove", str(x), str(y)], check=False)

    @staticmethod
    def click(button: str = "1"):
        """Click mouse button (1=left, 2=middle, 3=right)"""
        subprocess.run(["xdotool", "click", button], check=False)

    @staticmethod
    def double_click(button: str = "1"):
        """Double click"""
        subprocess.run(["xdotool", "click", "--repeat", "2", button], check=False)

    @staticmethod
    def drag(start_x: int, start_y: int, end_x: int, end_y: int):
        """Drag from start to end"""
        subprocess.run(["xdotool", "mousemove", str(start_x), str(start_y)], check=False)
        time.sleep(0.1)
        subprocess.run(["xdotool", "mousedown", "1"], check=False)
        time.sleep(0.1)
        subprocess.run(["xdotool", "mousemove", str(end_x), str(end_y)], check=False)
        time.sleep(0.1)
        subprocess.run(["xdotool", "mouseup", "1"], check=False)

    @staticmethod
    def scroll(clicks: int):
        """Scroll (positive=up, negative=down)"""
        subprocess.run(["xdotool", "click", "4" if clicks > 0 else "5"] * abs(clicks), check=False)

    @classmethod
    def execute_action(cls, action: Dict):
        """Execute a mouse action from config"""
        action_type = action.get("action", "")

        if action_type == "move":
            cls.move(action.get("x", 0), action.get("y", 0))
        elif action_type == "click":
            cls.click(action.get("button", "1"))
        elif action_type == "double_click":
            cls.double_click(action.get("button", "1"))
        elif action_type == "drag":
            cls.drag(
                action.get("start_x", 0), action.get("start_y", 0),
                action.get("end_x", 0), action.get("end_y", 0)
            )
        elif action_type == "scroll":
            cls.scroll(action.get("clicks", 3))


class ActionSequencer:
    """Execute multi-step action sequences"""

    @staticmethod
    def execute_sequence(actions: List[Dict]):
        """Execute a sequence of actions with delays"""
        if not actions:
            return

        def run_actions():
            for i, action in enumerate(actions):
                action_type = action.get("type", "key")

                if action_type == "key":
                    # Keyboard action
                    key = action.get("key", "alt+o")
                    delay = action.get("delay", 1.5)
                    time.sleep(delay)
                    subprocess.run(["xdotool", "key", "--clearmodifiers", key], check=False)
                    print(f"[Sequence] Key: {key}")

                elif action_type == "mouse":
                    # Mouse action
                    MouseController.execute_action(action.get("mouse", {}))
                    delay = action.get("delay", 0.5)
                    time.sleep(delay)
                    print(f"[Sequence] Mouse action")

                elif action_type == "wait":
                    # Just wait
                    delay = action.get("delay", 1.0)
                    time.sleep(delay)
                    print(f"[Sequence] Wait: {delay}s")

        # Run in background thread
        threading.Thread(target=run_actions, daemon=True).start()


class ConfigManager:
    """YAML configuration file management"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.config_file = None
        self.last_modified = 0

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            print(f"[Config] File not found: {config_path}")
            return {}

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"[Config] Loaded: {config_path}")
                self.config_file = config_path
                self.last_modified = os.path.getmtime(config_path)
                return config or {}
        except Exception as e:
            print(f"[Config] Error loading: {e}")
            return {}

    def save_config(self, config_path: str, data: Dict):
        """Save configuration to YAML file"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                print(f"[Config] Saved: {config_path}")
        except Exception as e:
            print(f"[Config] Error saving: {e}")

    def check_reload(self) -> bool:
        """Check if config file was modified"""
        if not self.config_file or not os.path.exists(self.config_file):
            return False

        current_mtime = os.path.getmtime(self.config_file)
        if current_mtime > self.last_modified:
            self.last_modified = current_mtime
            return True
        return False


class AdaptivePoller:
    """Smart polling that adjusts interval based on activity"""

    def __init__(self, idle_interval: float = 2.0, active_interval: float = 0.2):
        self.idle_interval = idle_interval
        self.active_interval = active_interval
        self.current_interval = idle_interval
        self.was_active = False

    def update(self, detected: bool) -> float:
        """Update interval based on detection result"""
        if detected:
            # Active - use fast polling
            if not self.was_active:
                print(f"[Adaptive] Switching to active mode ({self.active_interval}s)")
            self.current_interval = self.active_interval
            self.was_active = True
        else:
            # Idle - use slow polling
            if self.was_active:
                print(f"[Adaptive] Switching to idle mode ({self.idle_interval}s)")
            self.current_interval = self.idle_interval
            self.was_active = False

        return self.current_interval


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
            msg = f">>> VISION TRIGGER! Mode: {self.vision_engine.config.mode.value}"
            print(msg)
            if self.vision_engine.config.log_enabled:
                logger = Logger.get_instance(self.vision_engine.config.log_file)
                logger.info(msg)

            # Record screenshot on trigger
            if self.vision_engine.config.record_on_trigger:
                recorder = ScreenRecorder(self.vision_engine.config.record_dir)
                filepath = recorder.capture_trigger(self.vision_engine.config.region)
                print(f"  [Recorded] {filepath}")

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

        # Initialize all managers
        logger = None
        recorder = None
        webhook = None
        notifier = None
        adaptive_poller = None
        config_manager = None

        if self.config.log_enabled:
            logger = Logger.get_instance(self.config.log_file)
            logger.info(f"Polling started - interval: {self.config.poll_interval}s")

        if self.config.record_on_trigger:
            recorder = ScreenRecorder(self.config.record_dir)

        if self.config.webhook_enabled and self.config.webhook_url:
            webhook = WebhookNotifier(self.config.webhook_url)

        if self.config.notify_enabled:
            notifier = NotificationManager(True, self.config.notify_title)

        if self.config.adaptive_polling:
            adaptive_poller = AdaptivePoller(
                self.config.idle_interval,
                self.config.active_interval
            )

        if self.config.watch_config and self.config.config_file:
            config_manager = ConfigManager.get_instance()

        current_interval = self.config.poll_interval

        while self.running:
            try:
                # Check for config reload
                if config_manager and config_manager.check_reload():
                    new_config = config_manager.load_config(self.config.config_file)
                    print("[Config] Reloaded configuration")

                result = engine.process()
                triggered = trigger_mgr.should_trigger(result)

                # Update adaptive polling
                if adaptive_poller:
                    current_interval = adaptive_poller.update(result)

                if triggered:
                    msg = f">>> POLL TRIGGER! Mode: {self.config.mode.value}"
                    print(msg)

                    # Log
                    if logger:
                        logger.info(msg)

                    # Record screenshot
                    if recorder:
                        filepath = recorder.capture_trigger(self.config.region)
                        print(f"  [Recorded] {filepath}")
                        if logger:
                            logger.info(f"Screenshot saved: {filepath}")

                    # Send webhook
                    if webhook:
                        webhook.send({
                            "mode": self.config.mode.value,
                            "trigger_count": trigger_mgr.trigger_count,
                            "condition_met": result
                        })

                    # Send notification
                    if notifier:
                        notifier.notify(f"Triggered: {self.config.mode.value}")

                    # Execute mouse actions
                    for mouse_action in self.config.mouse_actions:
                        MouseController.execute_action(mouse_action)

                    # Execute action sequence
                    if self.config.action_sequence:
                        ActionSequencer.execute_sequence(self.config.action_sequence)
                    else:
                        # Default keyboard action
                        threading.Thread(
                            target=AutomationAction.execute,
                            args=(self.config.action, self.config.action_delay)
                        ).start()

            except Exception as e:
                err_msg = f"[Polling] Error: {e}"
                print(err_msg)
                if logger:
                    logger.error(err_msg)

            time.sleep(current_interval)

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
    parser.add_argument("--templates", type=str, nargs="+", help="Multiple template paths (OR match)")

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

    # Logging options
    parser.add_argument("--log", type=str, help="Log file path")
    parser.add_argument("--log-enable", action="store_true", help="Enable logging to file")

    # Recording options
    parser.add_argument("--record", action="store_true", help="Record screenshot on trigger")
    parser.add_argument("--record-dir", type=str, default="/tmp/auto_claw_records",
                       help="Directory to save triggered screenshots")

    # Adaptive polling options
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive polling")
    parser.add_argument("--idle-interval", type=float, default=2.0, help="Polling interval when idle")
    parser.add_argument("--active-interval", type=float, default=0.2, help="Polling interval when active")

    # Webhook options
    parser.add_argument("--webhook", type=str, help="Webhook URL to send notifications")
    parser.add_argument("--webhook-enable", action="store_true", help="Enable webhook notifications")

    # Notification options
    parser.add_argument("--notify", action="store_true", help="Enable desktop notifications")
    parser.add_argument("--notify-title", type=str, default="OpenClaw", help="Notification title")

    # Mouse action options (can be specified multiple times)
    parser.add_argument("--mouse-click", type=str, help="Mouse click at x,y (e.g., '100,200')")
    parser.add_argument("--mouse-move", type=str, help="Move mouse to x,y")
    parser.add_argument("--mouse-scroll", type=int, help="Scroll clicks (positive=up, negative=down)")

    # Action sequence (JSON string)
    parser.add_argument("--sequence", type=str, help="Action sequence as JSON string")

    # YAML configuration
    parser.add_argument("--config", type=str, help="YAML configuration file")

    # Dynamic config reload
    parser.add_argument("--watch", action="store_true", help="Watch config file for changes")

    # Configuration profiles
    parser.add_argument("--profile", type=str, help="Configuration profile name")
    parser.add_argument("--save-profile", type=str, help="Save current config as profile")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles")

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle configuration profiles
    config_manager = ConfigManager.get_instance()
    profile_dir = os.path.expanduser("~/.openclaw")
    os.makedirs(profile_dir, exist_ok=True)

    # List profiles
    if args.list_profiles:
        profiles = [f.replace('.yaml', '') for f in os.listdir(profile_dir) if f.endswith('.yaml')]
        print(f"Available profiles: {profiles if profiles else 'None'}")
        return

    # Load from YAML config if provided
    if args.config:
        yaml_config = config_manager.load_config(args.config)
        if yaml_config:
            # Override with CLI args
            if 'mode' not in yaml_config:
                yaml_config['mode'] = args.mode
            config = VisionConfig.from_dict(yaml_config)
            config.config_file = args.config
        else:
            config = None
    else:
        config = None

    # If no config yet, build from args
    if config is None:
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

        # Primary condition
        primary_condition = TriggerCondition(
            mode=VisionMode(args.mode) if args.mode != "multi" else VisionMode.OCR,
            target_text=args.text,
            region=region,
            change_threshold=args.threshold,
            template_path=args.template,
            target_color=target_color
        )
        conditions.append(primary_condition)

        # Secondary condition
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
            mode = VisionMode.MULTI
            if len(conditions) == 1:
                mode = conditions[0].mode
        else:
            mode = VisionMode(args.mode)

        # Build mouse actions
        mouse_actions = []
        if args.mouse_click:
            x, y = map(int, args.mouse_click.split(','))
            mouse_actions.append({"action": "click", "x": x, "y": y})
        if args.mouse_move:
            x, y = map(int, args.mouse_move.split(','))
            mouse_actions.append({"action": "move", "x": x, "y": y})
        if args.mouse_scroll:
            mouse_actions.append({"action": "scroll", "clicks": args.mouse_scroll})

        # Parse action sequence
        action_sequence = []
        if args.sequence:
            try:
                action_sequence = json.loads(args.sequence)
            except json.JSONDecodeError:
                print(f"[Error] Invalid JSON in --sequence: {args.sequence}")

        # Create config
        config = VisionConfig(
            mode=mode,
            conditions=conditions if mode == VisionMode.MULTI else [],
            condition_logic=args.logic,
            polling=args.poll,
            poll_interval=args.interval,
            adaptive_polling=args.adaptive,
            idle_interval=args.idle_interval,
            active_interval=args.active_interval,
            target_text=args.text,
            region=region,
            change_threshold=args.threshold,
            template_path=args.template,
            templates=args.templates if args.templates else [],
            target_color=target_color,
            action=args.action,
            action_delay=args.delay,
            log_file=args.log,
            log_enabled=args.log_enable,
            record_on_trigger=args.record,
            record_dir=args.record_dir,
            webhook_url=args.webhook,
            webhook_enabled=args.webhook_enable,
            notify_enabled=args.notify,
            notify_title=args.notify_title,
            mouse_actions=mouse_actions,
            action_sequence=action_sequence,
            config_file=args.config,
            watch_config=args.watch,
            profile_name=args.profile
        )

    # Save profile if requested
    if args.save_profile:
        profile_path = os.path.join(profile_dir, f"{args.save_profile}.yaml")
        config_manager.save_config(profile_path, config.to_dict())
        print(f"Profile saved: {profile_path}")
        return

    # Initialize logger
    if config.log_enabled:
        logger = Logger.get_instance(args.log)
        if args.log:
            logger.info(f"Logging started - log file: {args.log}")
        logger.info(f"Mode: {mode.value}, Polling: {args.poll}")

    # Start server
    server = VisionHTTPServer(args.port, config)
    server.start()


if __name__ == "__main__":
    main()
