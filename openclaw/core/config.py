"""Configuration management with validation"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class VisionMode(Enum):
    """Vision detection modes"""
    OCR = "ocr"
    MONITOR = "monitor"
    TEMPLATE = "template"
    COLOR = "color"
    ANALYZE = "analyze"
    MULTI = "multi"
    YOLO = "yolo"
    FUZZY = "fuzzy"
    REGRESSION = "regression"
    WINDOW = "window"


# JSON Schema for VisionConfig validation
VISION_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": [m.value for m in VisionMode]
        },
        "polling": {"type": "boolean"},
        "poll_interval": {"type": "number", "minimum": 0.1},
        "adaptive_polling": {"type": "boolean"},
        "idle_interval": {"type": "number", "minimum": 0.1},
        "active_interval": {"type": "number", "minimum": 0.1},
        "target_text": {"type": "string"},
        "text_case_sensitive": {"type": "boolean"},
        "region": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4
        },
        "change_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "template_path": {"type": "string"},
        "templates": {"type": "array", "items": {"type": "string"}},
        "template_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "yolo_model": {"type": "string"},
        "yolo_classes": {"type": "array", "items": {"type": "string"}},
        "yolo_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "target_color": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 255},
            "minItems": 3,
            "maxItems": 3
        },
        "color_tolerance": {"type": "number", "minimum": 0},
        "action": {"type": "string"},
        "action_delay": {"type": "number", "minimum": 0},
        "log_enabled": {"type": "boolean"},
        "log_file": {"type": "string"},
        "record_on_trigger": {"type": "boolean"},
        "record_dir": {"type": "string"},
        "db_enabled": {"type": "boolean"},
        "db_path": {"type": "string"},
        "webhook_enabled": {"type": "boolean"},
        "webhook_url": {"anyOf": [{"type": "string", "format": "uri"}, {"type": "null"}]},
        "notify_enabled": {"type": "boolean"},
        "notify_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "gateway_enabled": {"type": "boolean"},
        "gateway_port": {"type": "number", "minimum": 1, "maximum": 65535},
        "telegram_enabled": {"type": "boolean"},
        "telegram_token": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "telegram_chat_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "websocket_enabled": {"type": "boolean"},
        "websocket_port": {"type": "number", "minimum": 1, "maximum": 65535},
        "stream_enabled": {"type": "boolean"},
        "stream_port": {"type": "number", "minimum": 1, "maximum": 65535},
        "mouse_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["click", "move", "drag", "scroll", "double_click"]},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "button": {"type": "string"},
                    "start_x": {"type": "number"},
                    "start_y": {"type": "number"},
                    "end_x": {"type": "number"},
                    "end_y": {"type": "number"},
                    "clicks": {"type": "number"}
                }
            }
        },
        "action_sequence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["key", "mouse", "wait"]},
                    "key": {"type": "string"},
                    "delay": {"type": "number", "minimum": 0},
                    "mouse": {"type": "object"},
                    "action": {"type": "string"}
                }
            }
        },
        "api_key": {"type": "string"},
        "rate_limit": {"type": "number", "minimum": 1},
        "fuzzy_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "ocr_languages": {
            "type": "array",
            "items": {"type": "string"}
        },
        "regression_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "retry_attempts": {"type": "number", "minimum": 0},
        "retry_delay": {"type": "number", "minimum": 0}
    }
}


class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass


@dataclass
class TriggerCondition:
    """A single trigger condition for multi-mode"""
    mode: VisionMode = VisionMode.OCR
    target_text: Optional[str] = None
    region: Optional[tuple] = None
    change_threshold: float = 0.05
    template_path: Optional[str] = None
    template_threshold: float = 0.8
    target_color: Optional[tuple] = None
    color_tolerance: int = 30
    text_case_sensitive: bool = False

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value if isinstance(self.mode, VisionMode) else self.mode,
            "target_text": self.target_text,
            "region": list(self.region) if self.region else None,
            "change_threshold": self.change_threshold,
            "template_path": self.template_path,
            "template_threshold": self.template_threshold,
            "target_color": list(self.target_color) if self.target_color else None,
            "color_tolerance": self.color_tolerance,
            "text_case_sensitive": self.text_case_sensitive
        }


@dataclass
class VisionConfig:
    """Configuration for vision-based triggering"""
    mode: VisionMode = VisionMode.OCR

    # Multi-condition support
    conditions: List[TriggerCondition] = field(default_factory=list)
    condition_logic: str = "or"

    # Polling settings
    polling: bool = False
    poll_interval: float = 0.5
    adaptive_polling: bool = False
    idle_interval: float = 2.0
    active_interval: float = 0.2

    # OCR settings
    target_text: Optional[str] = None
    text_case_sensitive: bool = False
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    fuzzy_threshold: float = 0.8

    # Monitor settings
    region: Optional[tuple] = None
    change_threshold: float = 0.05
    regression_threshold: float = 0.01

    # Template settings
    template_path: Optional[str] = None
    templates: List[str] = field(default_factory=list)
    template_threshold: float = 0.8

    # YOLO settings
    yolo_model: str = "yolov8n.pt"
    yolo_classes: List[str] = field(default_factory=list)
    yolo_confidence: float = 0.5

    # Window monitoring settings
    window_signal: str = "TRIGGER_CLAW"
    window_poll_interval: float = 0.3
    window_debounce: float = 3.0

    # Color settings
    target_color: Optional[tuple] = None
    color_tolerance: int = 30

    # Logging
    log_file: Optional[str] = None
    log_enabled: bool = False

    # Recording
    record_on_trigger: bool = False
    record_dir: str = "/tmp/openclaw_records"

    # Database
    db_enabled: bool = False
    db_path: Optional[str] = None

    # Webhooks
    webhook_url: Optional[str] = None
    webhook_enabled: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Desktop notifications
    notify_enabled: bool = False
    notify_title: str = "OpenClaw"

    # Gateway
    gateway_enabled: bool = False
    gateway_port: int = 18789
    gateway_host: str = "localhost"

    # Telegram
    telegram_enabled: bool = False
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # WebSocket
    websocket_enabled: bool = False
    websocket_port: int = 8766
    websocket_host: str = "localhost"

    # Streaming
    stream_enabled: bool = False
    stream_port: int = 8888
    stream_host: str = "localhost"

    # Mouse actions
    mouse_actions: List[Dict] = field(default_factory=list)

    # Action sequences
    action_sequence: List[Dict] = field(default_factory=list)

    # Security
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute

    # General
    capture_screen: int = 0
    action: str = "alt+o"
    action_delay: float = 1.5
    config_file: Optional[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.templates is None:
            self.templates = []
        if self.mouse_actions is None:
            self.mouse_actions = []
        if self.action_sequence is None:
            self.action_sequence = []
        if self.yolo_classes is None:
            self.yolo_classes = []
        if self.ocr_languages is None:
            self.ocr_languages = ["en"]

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, tuple):
                result[key] = list(value)
            elif isinstance(value, VisionMode):
                result[key] = value.value
            elif isinstance(value, TriggerCondition):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict, validate: bool = False) -> 'VisionConfig':
        """Create config from dictionary with optional validation"""
        # Validate if requested (validation is done in load_config)
        if validate and JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(data, VISION_CONFIG_SCHEMA)
            except jsonschema.ValidationError as e:
                raise ConfigValidationError(f"Invalid config: {e.message}")

        # Handle mode conversion
        mode = data.get("mode", "ocr")
        if isinstance(mode, str):
            mode = VisionMode(mode)

        # Handle region conversion
        region = data.get("region")
        if region and isinstance(region, list):
            region = tuple(region)

        # Handle color conversion
        target_color = data.get("target_color")
        if target_color and isinstance(target_color, list):
            target_color = tuple(target_color)

        # Handle conditions
        conditions = []
        for cond in data.get("conditions", []):
            if isinstance(cond, dict):
                cond_mode = VisionMode(cond.get("mode", "ocr"))
                cond_region = tuple(cond["region"]) if cond.get("region") else None
                cond_color = tuple(cond["target_color"]) if cond.get("target_color") else None
                conditions.append(TriggerCondition(
                    mode=cond_mode,
                    target_text=cond.get("target_text"),
                    region=cond_region,
                    change_threshold=cond.get("change_threshold", 0.05),
                    template_path=cond.get("template_path"),
                    template_threshold=cond.get("template_threshold", 0.8),
                    target_color=cond_color,
                    color_tolerance=cond.get("color_tolerance", 30),
                    text_case_sensitive=cond.get("text_case_sensitive", False)
                ))

        return cls(
            mode=mode,
            conditions=conditions,
            condition_logic=data.get("condition_logic", "or"),
            polling=data.get("polling", False),
            poll_interval=data.get("poll_interval", 0.5),
            adaptive_polling=data.get("adaptive_polling", False),
            idle_interval=data.get("idle_interval", 2.0),
            active_interval=data.get("active_interval", 0.2),
            target_text=data.get("target_text"),
            text_case_sensitive=data.get("text_case_sensitive", False),
            ocr_languages=data.get("ocr_languages", ["en"]),
            fuzzy_threshold=data.get("fuzzy_threshold", 0.8),
            region=region,
            change_threshold=data.get("change_threshold", 0.05),
            regression_threshold=data.get("regression_threshold", 0.01),
            template_path=data.get("template_path"),
            templates=data.get("templates", []),
            template_threshold=data.get("template_threshold", 0.8),
            yolo_model=data.get("yolo_model", "yolov8n.pt"),
            yolo_classes=data.get("yolo_classes", []),
            yolo_confidence=data.get("yolo_confidence", 0.5),
            target_color=target_color,
            color_tolerance=data.get("color_tolerance", 30),
            log_file=data.get("log_file"),
            log_enabled=data.get("log_enabled", False),
            record_on_trigger=data.get("record_on_trigger", False),
            record_dir=data.get("record_dir", "/tmp/openclaw_records"),
            db_enabled=data.get("db_enabled", False),
            db_path=data.get("db_path"),
            webhook_url=data.get("webhook_url"),
            webhook_enabled=data.get("webhook_enabled", False),
            retry_attempts=data.get("retry_attempts", 3),
            retry_delay=data.get("retry_delay", 1.0),
            notify_enabled=data.get("notify_enabled", False),
            notify_title=data.get("notify_title", "OpenClaw"),
            gateway_enabled=data.get("gateway_enabled", False),
            gateway_port=data.get("gateway_port", 18789),
            gateway_host=data.get("gateway_host", "localhost"),
            telegram_enabled=data.get("telegram_enabled", False),
            telegram_token=data.get("telegram_token"),
            telegram_chat_id=data.get("telegram_chat_id"),
            websocket_enabled=data.get("websocket_enabled", False),
            websocket_port=data.get("websocket_port", 8766),
            websocket_host=data.get("websocket_host", "localhost"),
            stream_enabled=data.get("stream_enabled", False),
            stream_port=data.get("stream_port", 8888),
            stream_host=data.get("stream_host", "localhost"),
            mouse_actions=data.get("mouse_actions", []),
            action_sequence=data.get("action_sequence", []),
            api_key=data.get("api_key"),
            rate_limit=data.get("rate_limit", 60),
            capture_screen=data.get("capture_screen", 0),
            action=data.get("action", "alt+o"),
            action_delay=data.get("action_delay", 1.5),
            config_file=data.get("config_file")
        )


class ConfigManager:
    """YAML configuration file manager with validation"""

    def __init__(self):
        self.config_file: Optional[str] = None
        self.last_modified: float = 0
        self._config: Optional[VisionConfig] = None
        self._validate = JSONSCHEMA_AVAILABLE

    def load_config(self, config_path: str, validate: bool = True) -> VisionConfig:
        """Load configuration from YAML file with validation"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                raise ValueError("Config file is empty")

            # Validate if enabled
            if validate and self._validate:
                try:
                    jsonschema.validate(data, VISION_CONFIG_SCHEMA)
                except jsonschema.ValidationError as e:
                    raise ConfigValidationError(f"Config validation failed: {e.message}")

            config = VisionConfig.from_dict(data)
            config.config_file = config_path

            self.config_file = config_path
            self.last_modified = os.path.getmtime(config_path)
            self._config = config

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

    def save_config(self, config_path: str, config: VisionConfig) -> None:
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

        self.config_file = config_path
        self.last_modified = os.path.getmtime(config_path)

    def check_reload(self) -> bool:
        """Check if config file was modified"""
        if not self.config_file or not os.path.exists(self.config_file):
            return False

        current_mtime = os.path.getmtime(self.config_file)
        if current_mtime > self.last_modified:
            self.last_modified = current_mtime
            return True
        return False

    def reload(self) -> Optional[VisionConfig]:
        """Reload configuration from file"""
        if self.config_file and os.path.exists(self.config_file):
            return self.load_config(self.config_file)
        return None

    @property
    def config(self) -> Optional[VisionConfig]:
        """Get current config"""
        return self._config
