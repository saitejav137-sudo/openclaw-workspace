"""
AI Natural Language Interface for OpenClaw

ChatGPT-like interface for configuring automations using natural language.
Allows users to describe automations in plain English.
"""

import json
import re
from .logger import get_logger
import time
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = get_logger("ai")


class NLPMode(Enum):
    """Natural language processing modes"""
    PATTERN = "pattern"  # Pattern-based parsing
    LLM = "llm"  # LLM-based (OpenAI, Anthropic, etc.)
    HYBRID = "hybrid"  # Both pattern and LLM


@dataclass
class AutomationIntent:
    """Parsed automation intent from natural language"""
    intent_type: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    suggested_config: Optional[Dict] = None


@dataclass
class NLPConfig:
    """Configuration for natural language processing"""
    mode: NLPMode = NLPMode.PATTERN
    llm_provider: str = "openai"  # openai, anthropic, local
    llm_model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    context_window: int = 5


class NLPParser:
    """
    Natural language parser for automation configuration.
    Supports pattern-based and LLM-based parsing.
    """

    def __init__(self, config: NLPConfig):
        self.config = config
        self._llm_client = None
        self._conversation_history: List[Dict] = []
        self._max_history = config.context_window

    def initialize(self) -> bool:
        """Initialize the NLP parser"""
        if self.config.mode in (NLPMode.LLM, NLPMode.HYBRID):
            return self._initialize_llm()
        return True

    def _initialize_llm(self) -> bool:
        """Initialize LLM client"""
        if self.config.llm_provider == "openai":
            try:
                import openai
                openai.api_key = self.config.api_key
                self._llm_client = openai
                logger.info("OpenAI LLM client initialized")
                return True
            except ImportError:
                logger.error("OpenAI package not installed")
                return False
        elif self.config.llm_provider == "anthropic":
            try:
                import anthropic
                self._llm_client = anthropic.Anthropic(
                    api_key=self.config.api_key
                )
                logger.info("Anthropic LLM client initialized")
                return True
            except ImportError:
                logger.error("Anthropic package not installed")
                return False

        logger.warning(f"Unknown LLM provider: {self.config.llm_provider}")
        return False

    def parse(self, text: str) -> AutomationIntent:
        """Parse natural language text into automation intent"""
        # Clean input
        text = text.strip()

        if self.config.mode == NLPMode.PATTERN:
            return self._parse_pattern(text)
        elif self.config.mode == NLPMode.LLM:
            return self._parse_llm(text)
        else:  # HYBRID
            # Try pattern first, fallback to LLM
            intent = self._parse_pattern(text)
            if intent.confidence < 0.7:
                return self._parse_llm(text)
            return intent

    def _parse_pattern(self, text: str) -> AutomationIntent:
        """Parse using pattern matching"""
        text_lower = text.lower()

        # Detect intent type
        intent_type = self._detect_intent(text_lower)
        parameters = {}

        # Extract parameters based on intent
        if intent_type == "trigger_text":
            parameters = self._extract_text_trigger(text)
        elif intent_type == "trigger_template":
            parameters = self._extract_template_trigger(text)
        elif intent_type == "trigger_color":
            parameters = self._extract_color_trigger(text)
        elif intent_type == "trigger_region":
            parameters = self._extract_region_trigger(text)
        elif intent_type == "trigger_schedule":
            parameters = self._extract_schedule(text)
        elif intent_type == "trigger_voice":
            parameters = self._extract_voice_command(text)
        elif intent_type == "action_keyboard":
            parameters = self._extract_keyboard_action(text)
        elif intent_type == "action_mouse":
            parameters = self._extract_mouse_action(text)
        elif intent_type == "action_notification":
            parameters = self._extract_notification_action(text)
        elif intent_type == "configure":
            parameters = self._extract_configuration(text)
        else:
            intent_type = "general"
            parameters = {"query": text}

        confidence = self._calculate_confidence(intent_type, parameters)

        return AutomationIntent(
            intent_type=intent_type,
            confidence=confidence,
            parameters=parameters,
            raw_text=text,
            suggested_config=self._generate_config(intent_type, parameters)
        )

    def _detect_intent(self, text: str) -> str:
        """Detect the intent type from text"""
        # Trigger patterns
        if any(word in text for word in ["detect", "trigger on", "watch for", "look for", "when"]):
            if any(word in text for word in ["text", "word", "button", "label"]):
                return "trigger_text"
            elif any(word in text for word in ["image", "icon", "picture"]):
                return "trigger_template"
            elif any(word in text for word in ["color", "colour"]):
                return "trigger_color"
            elif any(word in text for word in ["region", "area", "screen"]):
                return "trigger_region"
            elif any(word in text for word in ["time", "schedule", "every", "at"]):
                return "trigger_schedule"
            elif any(word in text for word in ["voice", "say", "speak"]):
                return "trigger_voice"
            return "trigger_generic"

        # Action patterns
        if any(word in text for word in ["press", "type", "key", "keyboard"]):
            return "action_keyboard"
        if any(word in text for word in ["click", "mouse", "move"]):
            return "action_mouse"
        if any(word in text for word in ["notify", "alert", "message", "send"]):
            return "action_notification"

        # Configuration patterns
        if any(word in text for word in ["configure", "set", "change", "update"]):
            return "configure"

        return "general"

    def _extract_text_trigger(self, text: str) -> Dict:
        """Extract text trigger parameters"""
        # Extract quoted text
        match = re.search(r'["\']([^"\']+)["\']', text)
        target_text = match.group(1) if match else ""

        # Extract region if mentioned
        region = self._extract_region(text)

        return {
            "target_text": target_text,
            "region": region,
            "case_sensitive": "case sensitive" in text
        }

    def _extract_template_trigger(self, text: str) -> Dict:
        """Extract template trigger parameters"""
        # Extract image filename
        match = re.search(r'(image|icon|picture|template)\s+(?:called|named|of)?\s*["\']?([^"\']+)["\']?', text, re.IGNORECASE)
        template = match.group(2) if match else ""

        region = self._extract_region(text)

        return {
            "template_path": template,
            "region": region,
            "threshold": self._extract_threshold(text)
        }

    def _extract_color_trigger(self, text: str) -> Dict:
        """Extract color trigger parameters"""
        # Extract color values
        color_match = re.search(r'(\d+)[,\s]+(\d+)[,\s]+(\d+)', text)
        if color_match:
            color = (int(color_match.group(1)), int(color_match.group(2)), int(color_match.group(3)))
        else:
            # Named colors
            color_map = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "yellow": (255, 255, 0),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
            }
            for name, color in color_map.items():
                if name in text.lower():
                    break
            else:
                color = None

        return {
            "target_color": color,
            "tolerance": self._extract_number(text, "tolerance", default=30)
        }

    def _extract_region_trigger(self, text: str) -> Dict:
        """Extract region trigger parameters"""
        region = self._extract_region(text)

        return {
            "region": region,
            "change_threshold": self._extract_threshold(text)
        }

    def _extract_schedule(self, text: str) -> Dict:
        """Extract schedule parameters"""
        schedule = {
            "type": "interval"
        }

        # Cron-like patterns
        if "every minute" in text:
            schedule["interval"] = 60
        elif "every hour" in text:
            schedule["interval"] = 3600
        elif "every" in text:
            # Extract number
            num_match = re.search(r'every\s+(\d+)\s+(second|minute|hour)', text)
            if num_match:
                value = int(num_match.group(1))
                unit = num_match.group(2)
                schedule["interval"] = value * (1 if unit == "second" else 60 if unit == "minute" else 3600)

        # Specific time
        time_match = re.search(r'at\s+(\d{1,2}):(\d{2})', text)
        if time_match:
            schedule["type"] = "cron"
            schedule["hour"] = int(time_match.group(1))
            schedule["minute"] = int(time_match.group(2))

        return schedule

    def _extract_voice_command(self, text: str) -> Dict:
        """Extract voice command parameters"""
        # Extract command phrases
        phrases = []

        # Look for quoted phrases
        for match in re.finditer(r'["\']([^"\']+)["\']', text):
            phrases.append(match.group(1))

        return {
            "phrases": phrases,
            "language": "en-US"
        }

    def _extract_keyboard_action(self, text: str) -> Dict:
        """Extract keyboard action parameters"""
        # Extract key combination
        keys = []

        # Common modifiers
        modifiers = ["ctrl", "alt", "shift", "cmd", "super"]
        for mod in modifiers:
            if mod in text.lower():
                keys.append(mod.capitalize())

        # Extract key letter
        key_match = re.search(r'\b([a-zA-Z])\b', text)
        if key_match:
            keys.append(key_match.group(1).lower())

        return {
            "keys": keys,
            "action": "press"
        }

    def _extract_mouse_action(self, text: str) -> Dict:
        """Extract mouse action parameters"""
        action = "click"

        if "double" in text.lower():
            action = "double_click"
        elif "right" in text.lower():
            action = "right_click"
        elif "move" in text.lower():
            action = "move"

        # Extract coordinates
        coord_match = re.search(r'(\d+)[,\s]+(\d+)', text)
        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
        else:
            x, y = None, None

        return {
            "action": action,
            "x": x,
            "y": y
        }

    def _extract_notification_action(self, text: str) -> Dict:
        """Extract notification action parameters"""
        return {
            "message": text,
            "title": "OpenClaw Notification"
        }

    def _extract_configuration(self, text: str) -> Dict:
        """Extract configuration parameters"""
        config = {}

        # Extract key=value pairs
        for match in re.finditer(r'(\w+)\s*[=:]\s*([^,\s]+)', text):
            key = match.group(1).lower()
            value = match.group(2)

            # Try numeric conversion
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            config[key] = value

        return config

    def _extract_region(self, text: str) -> Optional[List[int]]:
        """Extract screen region from text"""
        # Look for coordinates
        coord_match = re.search(r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)', text)
        if coord_match:
            return [
                int(coord_match.group(1)),
                int(coord_match.group(2)),
                int(coord_match.group(3)),
                int(coord_match.group(4))
            ]

        # Look for "center", "top", etc.
        if "center" in text.lower():
            return ["center"]
        elif "left" in text.lower():
            return ["left"]
        elif "right" in text.lower():
            return ["right"]

        return None

    def _extract_threshold(self, text: str) -> float:
        """Extract threshold value"""
        match = re.search(r'threshold\s*[=:]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.8

    def _extract_number(self, text: str, name: str, default: float = 0) -> float:
        """Extract a number from text"""
        match = re.search(rf'{name}\s*[=:]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return default

    def _calculate_confidence(self, intent_type: str, parameters: Dict) -> float:
        """Calculate confidence score"""
        if not parameters:
            return 0.3

        # Higher confidence for more parameters
        base = 0.5 + (len(parameters) * 0.1)

        return min(base, 0.95)

    def _generate_config(self, intent_type: str, parameters: Dict) -> Dict:
        """Generate suggested configuration"""
        config = {
            "mode": "ocr",
            "polling": True,
            "poll_interval": 0.5
        }

        # Map intent to config
        if intent_type == "trigger_text":
            config["mode"] = "fuzzy"
            config["target_text"] = parameters.get("target_text", "")
            config["text_case_sensitive"] = parameters.get("case_sensitive", False)
        elif intent_type == "trigger_template":
            config["mode"] = "template"
            config["template_path"] = parameters.get("template_path", "")
            config["template_threshold"] = parameters.get("threshold", 0.8)
        elif intent_type == "trigger_color":
            config["mode"] = "color"
            config["target_color"] = parameters.get("target_color")
            config["color_tolerance"] = parameters.get("tolerance", 30)
        elif intent_type == "trigger_region":
            config["mode"] = "monitor"
            config["region"] = parameters.get("region")
            config["change_threshold"] = parameters.get("change_threshold", 0.05)

        # Add common settings
        if "region" in parameters and parameters["region"]:
            config["region"] = parameters["region"]

        return config

    def _parse_llm(self, text: str) -> AutomationIntent:
        """Parse using LLM"""
        if not self._llm_client:
            return self._parse_pattern(text)

        # Build prompt
        prompt = self._build_prompt(text)

        try:
            if self.config.llm_provider == "openai":
                response = self._llm_client.ChatCompletion.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.choices[0].message.content

            elif self.config.llm_provider == "anthropic":
                response = self._llm_client.messages.create(
                    model=self.config.llm_model,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.content[0].text

            # Parse LLM response
            return self._parse_llm_response(result, text)

        except Exception as e:
            logger.error(f"LLM parsing error: {e}")
            return self._parse_pattern(text)

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an automation configuration assistant for OpenClaw, a vision-based automation framework.

Your task is to convert natural language descriptions into precise automation configurations.

Available trigger modes:
- ocr: Detect specific text on screen
- fuzzy: Fuzzy text matching
- template: Match template images
- color: Detect specific colors
- monitor: Monitor region for changes
- yolo: Object detection with YOLO
- window: Monitor window titles
- schedule: Time-based triggers

Available actions:
- keyboard: Press keys (e.g., "alt+o", "ctrl+c")
- mouse: Click, move, drag
- notify: Send notifications
- webhook: Call webhooks

Respond with a JSON object containing:
{
  "intent_type": "trigger_text" | "trigger_template" | etc,
  "confidence": 0.0-1.0,
  "parameters": {...},
  "suggested_config": {...}
}

Example:
Input: "When I see the button 'Submit' on the screen, press Enter"
Output: {"intent_type": "trigger_text", "confidence": 0.95, "parameters": {"target_text": "Submit"}, "suggested_config": {"mode": "fuzzy", "target_text": "Submit", "action": "enter"}}
"""

    def _build_prompt(self, text: str) -> str:
        """Build prompt for LLM"""
        return f"""Convert this natural language automation request into a configuration:

"{text}"

Respond with JSON only, no other text."""

    def _parse_llm_response(self, response: str, original_text: str) -> AutomationIntent:
        """Parse LLM response"""
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return AutomationIntent(
                    intent_type=data.get("intent_type", "general"),
                    confidence=data.get("confidence", 0.8),
                    parameters=data.get("parameters", {}),
                    raw_text=original_text,
                    suggested_config=data.get("suggested_config")
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")

        return self._parse_pattern(original_text)

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self._conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

    def clear_history(self):
        """Clear conversation history"""
        self._conversation_history.clear()


class NLInterface:
    """
    Natural Language Interface for OpenClaw.
    Provides a ChatGPT-like conversational interface.
    """

    def __init__(self, config: Optional[NLPConfig] = None):
        self.config = config or NLPConfig()
        self.parser = NLPParser(self.config)
        self._callbacks: Dict[str, Callable] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the NLP interface"""
        self._initialized = self.parser.initialize()
        return self._initialized

    def process(self, text: str) -> Dict[str, Any]:
        """Process natural language input"""
        if not self._initialized:
            self.initialize()

        # Parse intent
        intent = self.parser.parse(text)

        # Generate response
        response = self._generate_response(intent)

        # Add to history
        self.parser.add_to_history("user", text)
        self.parser.add_to_history("assistant", response["message"])

        return {
            "intent": intent,
            "response": response
        }

    def _generate_response(self, intent: AutomationIntent) -> Dict[str, Any]:
        """Generate response based on intent"""
        if intent.intent_type == "general":
            return {
                "message": f"I understand you're asking about: {intent.parameters.get('query', 'general automation')}. How can I help you create an automation?",
                "type": "question"
            }

        # Generate config suggestion message
        if intent.suggested_config:
            mode = intent.suggested_config.get("mode", "unknown")
            return {
                "message": f"I've parsed your request. I'll set up a {mode} trigger with the following configuration: {json.dumps(intent.suggested_config, indent=2)}",
                "type": "config",
                "config": intent.suggested_config
            }

        return {
            "message": f"I detected a {intent.intent_type} intent. Please provide more details.",
            "type": "clarification"
        }

    def register_callback(self, intent_type: str, callback: Callable):
        """Register callback for specific intent"""
        self._callbacks[intent_type] = callback

    def execute_intent(self, intent: AutomationIntent) -> bool:
        """Execute the parsed intent"""
        if intent.intent_type in self._callbacks:
            try:
                self._callbacks[intent.intent_type](intent)
                return True
            except Exception as e:
                logger.error(f"Intent execution error: {e}")
                return False
        return False


def create_nlp_interface(
    mode: NLPMode = NLPMode.PATTERN,
    api_key: Optional[str] = None,
    provider: str = "openai",
    model: str = "gpt-4"
) -> NLInterface:
    """Create a natural language interface"""
    config = NLPConfig(
        mode=mode,
        llm_provider=provider,
        llm_model=model,
        api_key=api_key
    )

    interface = NLInterface(config)
    return interface


__all__ = [
    "NLInterface",
    "NLPParser",
    "NLPConfig",
    "AutomationIntent",
    "NLPMode",
    "create_nlp_interface",
]
