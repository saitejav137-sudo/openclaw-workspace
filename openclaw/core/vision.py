"""Core vision module with detection algorithms"""

import os
import time
import hashlib
import threading
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import VisionConfig, VisionMode
from .logger import get_logger

logger = get_logger("vision")


class ScreenCapture:
    """Cross-platform screen capture with caching support"""

    _cache: Dict[str, Tuple[np.ndarray, float]] = {}
    _cache_lock = threading.Lock()
    _cache_ttl = 0.5  # Cache TTL in seconds

    @classmethod
    def capture_region(
        cls,
        region: Optional[Tuple[int, int, int, int]] = None,
        use_cache: bool = False
    ) -> np.ndarray:
        """Capture screen or region with optional caching"""
        import mss

        cache_key = str(region) if region else "full"

        # Check cache
        if use_cache:
            with cls._cache_lock:
                if cache_key in cls._cache:
                    img, timestamp = cls._cache[cache_key]
                    if time.time() - timestamp < cls._cache_ttl:
                        return img.copy()

        # Capture
        with mss.mss() as sct:
            if region:
                x, y, w, h = region
                monitor = {"top": y, "left": x, "width": w, "height": h}
            else:
                monitor = sct.monitors[1]

            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Cache result
        if use_cache:
            with cls._cache_lock:
                cls._cache[cache_key] = (img, time.time())

        return img

    @classmethod
    def capture_full(cls, screen: int = 1) -> np.ndarray:
        """Capture full screen"""
        return cls.capture_region(None)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear capture cache"""
        with cls._cache_lock:
            cls._cache.clear()


class OCREngine:
    """OCR engine with EasyOCR and fallback support"""

    _reader = None
    _languages = ["en"]
    _lock = threading.Lock()

    def __init__(self, languages: List[str] = None):
        self.languages = languages or ["en"]
        self._init_reader()

    def _init_reader(self):
        """Initialize OCR reader"""
        try:
            import easyocr
            with self._lock:
                if OCREngine._reader is None or OCREngine._languages != self.languages:
                    OCREngine._reader = easyocr.Reader(self.languages, gpu=True)
                    OCREngine._languages = self.languages
                    logger.info(f"OCR initialized with languages: {self.languages}")
        except ImportError:
            try:
                import pytesseract
                self._pytesseract = pytesseract
                logger.info("Using pytesseract as OCR backend")
            except ImportError:
                logger.error("No OCR engine available")

    def read(self, image: np.ndarray) -> str:
        """Read text from image"""
        if hasattr(self, '_reader') and self._reader:
            results = self._reader.readtext(image)
            return ' '.join([r[1] for r in results])
        elif hasattr(self, '_pytesseract'):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._pytesseract.image_to_string(gray)
        return ""

    @classmethod
    def is_available(cls) -> bool:
        """Check if OCR is available"""
        try:
            import easyocr
            return True
        except ImportError:
            try:
                import pytesseract
                return True
            except ImportError:
                return False


class FuzzyMatcher:
    """Fuzzy text matching using Levenshtein distance"""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return FuzzyMatcher.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio (0-1)"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        distance = FuzzyMatcher.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))

        return 1.0 - (distance / max_len)

    @staticmethod
    def match(text: str, pattern: str, threshold: float = 0.8) -> bool:
        """Check if pattern matches text above threshold"""
        similarity = FuzzyMatcher.similarity(text, pattern)
        return similarity >= threshold


class TemplateMatcher:
    """Template matching with multi-template support"""

    @staticmethod
    def match(
        image: np.ndarray,
        template_paths: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, float, Tuple[int, int]]]:
        """Match templates in image, returns list of (template_path, confidence, location)"""
        results = []

        for template_path in template_paths:
            if not os.path.exists(template_path):
                continue

            template = cv2.imread(template_path)
            if template is None:
                continue

            # Template matching
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                results.append((template_path, max_val, max_loc))

        return results


class ColorDetector:
    """Color detection in HSV color space"""

    @staticmethod
    def detect(
        image: np.ndarray,
        target_color: Tuple[int, int, int],
        tolerance: int = 30
    ) -> float:
        """Detect color ratio in image (0-1)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        target = np.array(target_color)
        lower = np.array([max(0, c - tolerance) for c in target])
        upper = np.array([min(255, c + tolerance) for c in target])

        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / mask.size

        return ratio

    @staticmethod
    def find_color_regions(
        image: np.ndarray,
        target_color: Tuple[int, int, int],
        tolerance: int = 30,
        min_area: int = 100
    ) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes of color regions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        target = np.array(target_color)
        lower = np.array([max(0, c - tolerance) for c in target])
        upper = np.array([min(255, c + tolerance) for c in target])

        mask = cv2.inRange(hsv, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))

        return regions


class ChangeDetector:
    """Region change detection for monitoring"""

    _last_capture: Dict[int, np.ndarray] = {}

    @classmethod
    def detect_change(
        cls,
        image: np.ndarray,
        region_id: int = 0,
        threshold: float = 0.05
    ) -> bool:
        """Detect if there's significant change in the image"""
        if region_id not in cls._last_capture:
            cls._last_capture[region_id] = image
            return False

        last = cls._last_capture[region_id]

        # Ensure same size
        if image.shape != last.shape:
            cls._last_capture[region_id] = image
            return False

        # Calculate difference
        diff = cv2.absdiff(last, image)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        non_zero_ratio = np.count_nonzero(gray) / gray.size

        cls._last_capture[region_id] = image

        return non_zero_ratio > threshold


class RegressionDetector:
    """Visual regression testing with pixel diff"""

    @staticmethod
    def compute_diff(
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute difference between two images"""
        # Resize if needed
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Calculate ratio of changed pixels
        threshold = 10  # Pixel value difference threshold
        changed_pixels = np.count_nonzero(diff > threshold)
        ratio = changed_pixels / diff.size

        # Create visualization
        visualized = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        visualized[diff > threshold] = [0, 0, 255]  # Red for changed

        return ratio, visualized

    @staticmethod
    def detect_regression(
        current: np.ndarray,
        baseline_path: str,
        threshold: float = 0.01
    ) -> Tuple[bool, float]:
        """Detect visual regression against baseline"""
        if not os.path.exists(baseline_path):
            # Save current as baseline
            cv2.imwrite(baseline_path, current)
            logger.info(f"Baseline saved: {baseline_path}")
            return False, 0.0

        baseline = cv2.imread(baseline_path)
        ratio, _ = RegressionDetector.compute_diff(current, baseline)

        return ratio > threshold, ratio


class VisionEngine:
    """Main vision processing engine"""

    def __init__(self, config: VisionConfig):
        self.config = config
        self._ocr_engine: Optional[OCREngine] = None
        self._yolo_model = None
        self._baseline_captures: Dict[int, np.ndarray] = {}

    @property
    def ocr_engine(self) -> OCREngine:
        """Get or create OCR engine"""
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(self.config.ocr_languages)
        return self._ocr_engine

    def process(self) -> bool:
        """Process current frame and return True if trigger condition met"""
        # Handle multi-condition mode
        if self.config.mode == VisionMode.MULTI:
            return self._process_multi()

        method = getattr(self, f"_process_{self.config.mode.value}", None)
        if method and callable(method):
            return method()

        logger.warning(f"Unknown mode: {self.config.mode}")
        return False

    def _process_multi(self) -> bool:
        """Process multiple conditions with AND/OR logic"""
        if not self.config.conditions:
            return False

        results = []
        for i, condition in enumerate(self.config.conditions):
            result = self._process_condition(condition)
            results.append(result)
            logger.debug(f"Condition {i+1} ({condition.mode.value}): {result}")

        if self.config.condition_logic == "and":
            return all(results)
        return any(results)

    def _process_condition(self, condition) -> bool:
        """Process a single condition"""
        # Create temporary config for this condition
        temp_config = VisionConfig(
            mode=condition.mode,
            target_text=condition.target_text,
            region=condition.region,
            change_threshold=condition.change_threshold,
            template_path=condition.template_path,
            template_threshold=condition.template_threshold,
            target_color=condition.target_color,
            color_tolerance=condition.color_tolerance,
            text_case_sensitive=condition.text_case_sensitive,
            ocr_languages=self.config.ocr_languages,
            fuzzy_threshold=self.config.fuzzy_threshold
        )

        engine = VisionEngine(temp_config)
        return engine.process()

    def _process_ocr(self) -> bool:
        """OCR text detection"""
        if not self.config.target_text:
            return False

        img = ScreenCapture.capture_region(self.config.region)
        text = self.ocr_engine.read(img)

        search_text = self.config.target_text
        if not self.config.text_case_sensitive:
            text = text.lower()
            search_text = search_text.lower()

        return search_text in text

    def _process_fuzzy(self) -> bool:
        """Fuzzy text matching for OCR"""
        if not self.config.target_text:
            return False

        img = ScreenCapture.capture_region(self.config.region)
        text = self.ocr_engine.read(img)

        return FuzzyMatcher.match(
            text,
            self.config.target_text,
            self.config.fuzzy_threshold
        )

    def _process_monitor(self) -> bool:
        """Region change detection"""
        if not self.config.region:
            return False

        img = ScreenCapture.capture_region(self.config.region, use_cache=True)
        region_id = hash(str(self.config.region))

        return ChangeDetector.detect_change(
            img,
            region_id,
            self.config.change_threshold
        )

    def _process_template(self) -> bool:
        """Template matching for object detection"""
        template_paths = []

        if self.config.template_path and os.path.exists(self.config.template_path):
            template_paths.append(self.config.template_path)

        for t in self.config.templates:
            if os.path.exists(t):
                template_paths.append(t)

        if not template_paths:
            logger.warning("No valid template paths found")
            return False

        img = ScreenCapture.capture_region(self.config.region)
        results = TemplateMatcher.match(img, template_paths, self.config.template_threshold)

        if results:
            logger.info(f"Template matched: {results[0][0]} ({results[0][1]:.2f})")
            return True

        return False

    def _process_color(self) -> bool:
        """Color detection in region"""
        if not self.config.target_color:
            return False

        img = ScreenCapture.capture_region(self.config.region)
        if img is None:
            return False

        ratio = ColorDetector.detect(
            img,
            self.config.target_color,
            self.config.color_tolerance
        )

        return ratio > 0.01

    def _process_yolo(self) -> bool:
        """YOLO object detection"""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("YOLO not installed")
            return False

        if self._yolo_model is None:
            self._yolo_model = YOLO(self.config.yolo_model)

        img = ScreenCapture.capture_region(self.config.region)
        results = self._yolo_model(img, verbose=False)

        detected_classes = set()
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= self.config.yolo_confidence:
                    cls_id = int(box.cls[0])
                    detected_classes.add(cls_id)

        if self.config.yolo_classes:
            for cls_name in self.config.yolo_classes:
                for cls_id in detected_classes:
                    name = self._yolo_model.names.get(cls_id, "").lower()
                    if cls_name.lower() in name:
                        logger.info(f"YOLO detected: {name}")
                        return True
            return False

        return len(detected_classes) > 0

    def _process_regression(self) -> bool:
        """Visual regression detection"""
        img = ScreenCapture.capture_region(self.config.region)

        # Use config file path for baseline
        baseline_dir = os.path.join(os.path.expanduser("~"), ".openclaw", "baselines")
        os.makedirs(baseline_dir, exist_ok=True)

        baseline_name = f"baseline_{hash(str(self.config.region))}.png"
        baseline_path = os.path.join(baseline_dir, baseline_name)

        has_regression, ratio = RegressionDetector.detect_regression(
            img,
            baseline_path,
            self.config.regression_threshold
        )

        if has_regression:
            logger.warning(f"Regression detected: {ratio:.2%} changed")

        return has_regression

    def _process_analyze(self) -> bool:
        """AI-powered screen analysis"""
        # This would integrate with AIScreenAnalyzer
        # Placeholder for now
        logger.info("AI Analyze mode - requires BLIP integration")
        return False

    def _process_window(self) -> bool:
        """Window title monitoring - checks for trigger signal"""
        from .window import get_window_monitor

        monitor = get_window_monitor()
        if not monitor:
            # Start a new monitor
            from .window import start_window_monitor
            start_window_monitor(
                trigger_signal=self.config.window_signal,
                callback=None,
                poll_interval=self.config.window_poll_interval
            )
            monitor = get_window_monitor()

        if monitor:
            current_title = monitor.get_active_window()
            if current_title and self.config.window_signal in current_title:
                # Check debounce
                if time.time() - monitor.last_trigger_time > self.config.window_debounce:
                    logger.info(f">>> WINDOW SIGNAL DETECTED: {current_title}")
                    monitor.last_trigger_time = time.time()
                    return True

        return False


# Export classes
__all__ = [
    "VisionEngine",
    "ScreenCapture",
    "OCREngine",
    "FuzzyMatcher",
    "TemplateMatcher",
    "ColorDetector",
    "ChangeDetector",
    "RegressionDetector",
    "MultiMonitor",
    "RegionSelector",
]
