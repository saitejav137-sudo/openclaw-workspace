"""
BLIP Vision Model Integration for OpenClaw

Screen analysis using BLIP (Bootstrapped Language-Image Pre-training)
for visual question answering and image captioning.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2

from .vision import ScreenCapture
from .logger import get_logger

logger = get_logger("blip")


class BLIPModelType(Enum):
    """BLIP model variants"""
    BLIP_BASE = "blip_base"
    BLIP_LARGE = "blip_large"
    BLIP2_BASE = "blip2_base"
    BLIP2_LARGE = "blip2_large"


@dataclass
class CaptionResult:
    """Image caption result"""
    caption: str
    confidence: float
    model: str
    timestamp: float


@dataclass
class VQAResult:
    """Visual Question Answering result"""
    question: str
    answer: str
    confidence: float
    model: str
    timestamp: float


class BLIPEngine:
    """
    BLIP-powered vision-language model for screen analysis.
    Supports image captioning and visual question answering.
    """

    def __init__(
        self,
        model_type: BLIPModelType = BLIPModelType.BLIP_BASE,
        device: str = "cuda"
    ):
        self.model_type = model_type
        self.device = device
        self._model = None
        self._processor = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize BLIP model"""
        if self._initialized:
            return True

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image
            import torch

            logger.info(f"Loading BLIP model: {self.model_type.value}")

            # Load model based on type
            if self.model_type == BLIPModelType.BLIP_BASE:
                model_name = "Salesforce/blip-base"
            elif self.model_type == BLIPModelType.BLIP_LARGE:
                model_name = "Salesforce/blip-large"
            elif self.model_type == BLIPModelType.BLIP2_BASE:
                model_name = "Salesforce/blip2-base"
            elif self.model_type == BLIPModelType.BLIP2_LARGE:
                model_name = "Salesforce/blip2-large"
            else:
                model_name = "Salesforce/blip-base"

            self._processor = BlipProcessor.from_pretrained(model_name)

            # Determine device
            if self.device == "cuda" and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            self._model = BlipForConditionalGeneration.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()

            self._initialized = True
            logger.info(f"BLIP model loaded on {self._device}")
            return True

        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.info("Install with: pip install transformers torch pillow")
            return False
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if BLIP is available"""
        return self._initialized

    def caption_image(
        self,
        image: np.ndarray,
        prompt: str = "a screenshot of",
        max_length: int = 50
    ) -> Optional[CaptionResult]:
        """Generate caption for image"""
        if not self.initialize():
            return None

        try:
            from PIL import Image
            import torch

            # Convert numpy to PIL
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Prepare inputs
            inputs = self._processor(
                pil_image,
                prompt=prompt,
                return_tensors="pt"
            ).to(self._device, torch.float16 if self._device.type == "cuda" else torch.float32)

            # Generate caption
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5
                )

            caption = self._processor.decode(output[0], skip_special_tokens=True)

            return CaptionResult(
                caption=caption,
                confidence=0.9,  # No confidence score in generation
                model=self.model_type.value,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Caption error: {e}")
            return None

    def answer_question(
        self,
        image: np.ndarray,
        question: str,
        max_length: int = 50
    ) -> Optional[VQAResult]:
        """Answer a question about the image"""
        if not self.initialize():
            return None

        try:
            from PIL import Image
            import torch

            # Convert numpy to PIL
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Prepare inputs with question
            question_prompt = f"Question: {question} Answer:"
            inputs = self._processor(
                pil_image,
                text=question_prompt,
                return_tensors="pt"
            ).to(self._device, torch.float16 if self._device.type == "cuda" else torch.float32)

            # Generate answer
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5
                )

            answer = self._processor.decode(output[0], skip_special_tokens=True)

            return VQAResult(
                question=question,
                answer=answer.strip(),
                confidence=0.85,
                model=self.model_type.value,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"VQA error: {e}")
            return None

    def analyze_screen(
        self,
        region: Optional[tuple] = None,
        questions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze screen with optional questions"""
        # Capture screen
        img = ScreenCapture.capture_region(region)

        results = {
            "timestamp": time.time(),
            "image_shape": img.shape,
            "caption": None,
            "questions": {}
        }

        # Generate caption
        caption_result = self.caption_image(img)
        if caption_result:
            results["caption"] = caption_result.caption

        # Answer questions if provided
        if questions:
            for question in questions:
                answer = self.answer_question(img, question)
                if answer:
                    results["questions"][question] = answer.answer

        return results


class MiniMaxAI:
    """
    MiniMax AI integration for advanced NLP and vision tasks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.chat/v1"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def initialize(self) -> bool:
        """Initialize MiniMax client"""
        if not self.api_key:
            logger.warning("MiniMax API key not provided")
            return False

        try:
            import httpx
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            return True

        except ImportError:
            logger.error("httpx not installed")
            return False
        except Exception as e:
            logger.error(f"MiniMax init error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if MiniMax is available"""
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "abab5.5s-chat",
        temperature: float = 0.7
    ) -> Optional[str]:
        """Send chat completion request"""
        if not self.initialize():
            return None

        try:
            response = self._client.post(
                "/text/chatcompletion_v2",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
            )
            response.raise_for_status()

            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content")

        except Exception as e:
            logger.error(f"MiniMax chat error: {e}")
            return None

    def generate_automation(
        self,
        description: str,
        context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate automation config from natural language"""
        if not self.is_available():
            return None

        messages = [
            {"role": "system", "content": """You are an automation expert. Generate OpenClaw configuration from natural language descriptions.

Available modes:
- ocr: Detect text
- fuzzy: Fuzzy text match
- template: Match image template
- color: Detect color
- monitor: Monitor region changes
- yolo: Object detection

Generate JSON config with:
- mode: detection mode
- target_text: text to detect (for ocr/fuzzy)
- template_path: path to template
- region: [x, y, w, h]
- action: keyboard action like "alt+o"
- action_delay: delay in seconds"""}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Current context: {context}"
            })

        messages.append({
            "role": "user",
            "content": description
        })

        result = self.chat(messages)

        if result:
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    import json
                    return json.loads(json_match.group())
            except Exception:
                pass

        return None


# Global instances
_blip_engine: Optional[BLIPEngine] = None
_minimax_ai: Optional[MiniMaxAI] = None


def get_blip_engine() -> BLIPEngine:
    """Get global BLIP engine"""
    global _blip_engine
    if _blip_engine is None:
        _blip_engine = BLIPEngine()
    return _blip_engine


def get_minimax_ai(api_key: Optional[str] = None) -> MiniMaxAI:
    """Get global MiniMax AI"""
    global _minimax_ai

    if _minimax_ai is None:
        _minimax_ai = MiniMaxAI(api_key=api_key)
    elif api_key:
        _minimax_ai.api_key = api_key
        _minimax_ai.initialize()

    return _minimax_ai


def analyze_screen_with_blip(
    questions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Quick screen analysis with BLIP"""
    engine = get_blip_engine()
    return engine.analyze_screen(questions=questions)


def caption_screen() -> Optional[str]:
    """Quick caption of current screen"""
    engine = get_blip_engine()
    img = ScreenCapture.capture_full()
    result = engine.caption_image(img)
    return result.caption if result else None


def ask_screen(question: str) -> Optional[str]:
    """Ask question about current screen"""
    engine = get_blip_engine()
    img = ScreenCapture.capture_full()
    result = engine.answer_question(img, question)
    return result.answer if result else None


__all__ = [
    "BLIPEngine",
    "BLIPModelType",
    "CaptionResult",
    "VQAResult",
    "MiniMaxAI",
    "get_blip_engine",
    "get_minimax_ai",
    "analyze_screen_with_blip",
    "caption_screen",
    "ask_screen",
]
