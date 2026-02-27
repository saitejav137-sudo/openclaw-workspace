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
    """Cross-platform screen capture with caching support.

    This class provides screen capture functionality with optional caching
    to improve performance for repeated captures.

    Attributes:
        _cache: Dictionary storing cached screenshots with timestamps
        _cache_lock: Thread lock for cache operations
        _default_cache_ttl: Default time-to-live for cached screenshots in seconds
    """

    _cache: Dict[str, Tuple[np.ndarray, float]] = {}
    _cache_lock = threading.Lock()
    _default_cache_ttl: float = 0.5  # Default cache TTL in seconds

    # Class-level configurable TTL
    _cache_ttl: float = 0.5

    @classmethod
    def set_cache_ttl(cls, ttl: float) -> None:
        """Set the cache TTL.

        Args:
            ttl: Time-to-live for cached screenshots in seconds
        """
        cls._cache_ttl = max(0.1, ttl)  # Minimum 0.1 seconds

    @classmethod
    def get_cache_ttl(cls) -> float:
        """Get the current cache TTL."""
        return cls._cache_ttl

    @classmethod
    def capture_region(
        cls,
        region: Optional[Tuple[int, int, int, int]] = None,
        use_cache: bool = False,
        cache_ttl: Optional[float] = None
    ) -> np.ndarray:
        """Capture a region of the screen.

        Args:
            region: Tuple of (x, y, width, height) for the region to capture.
                   If None, captures the entire primary monitor.
            use_cache: If True, uses cached screenshot if available and not expired.
            cache_ttl: Optional custom TTL for this capture (overrides class setting).

        Returns:
            Screenshot as numpy array in BGR format

        Raises:
            ImportError: If mss is not installed
        """
        import mss

        cache_key = str(region) if region else "full"
        ttl = cache_ttl if cache_ttl is not None else cls._cache_ttl

        # Check cache
        if use_cache:
            with cls._cache_lock:
                if cache_key in cls._cache:
                    img, timestamp = cls._cache[cache_key]
                    if time.time() - timestamp < ttl:
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
        """Capture the full primary monitor.

        Args:
            screen: Monitor index (1 = primary, default)

        Returns:
            Screenshot as numpy array in BGR format
        """
        return cls.capture_region(None)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached screenshots."""
        with cls._cache_lock:
            cls._cache.clear()


class OCREngine:
    """OCR engine with EasyOCR and fallback support"""

    def __init__(self, languages: List[str] = None):
        self.languages = languages or ["en"]
        self._reader = None  # Lazy loaded instance variable
        self._pytesseract = None
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize the OCR reader"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return

            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=True)
                logger.info(f"OCR initialized with languages: {self.languages}")
            except ImportError:
                try:
                    import pytesseract
                    self._pytesseract = pytesseract
                    logger.info("Using pytesseract as OCR backend")
                except ImportError:
                    logger.error("No OCR engine available")

            self._initialized = True

    @property
    def reader(self):
        """Get the OCR reader (lazy initialization)"""
        self._ensure_initialized()
        return self._reader

    @property
    def pytesseract(self):
        """Get the pytesseract module (lazy initialization)"""
        self._ensure_initialized()
        return self._pytesseract

    def read(self, image: np.ndarray) -> str:
        """Read text from image.

        Args:
            image: Input image as numpy array

        Returns:
            Extracted text as string
        """
        self._ensure_initialized()

        if self._reader is not None:
            results = self._reader.readtext(image)
            return ' '.join([r[1] for r in results])
        elif self._pytesseract is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._pytesseract.image_to_string(gray)
        return ""

    @staticmethod
    def is_available() -> bool:
        """Check if any OCR engine is available.

        Returns:
            True if EasyOCR or PyTesseract is installed, False otherwise
        """
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
    """Fuzzy text matching using Levenshtein distance.

    Provides methods for calculating string similarity and fuzzy matching
    with configurable thresholds. Uses case-insensitive comparison.

    Example:
        >>> matcher = FuzzyMatcher()
        >>> matcher.similarity("hello", "hello")
        1.0
        >>> matcher.match("Hello World", "hello world", threshold=0.9)
        True
    """

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate the Levenshtein (edit) distance between two strings.

        The Levenshtein distance is the minimum number of single-character
        edits (insertions, deletions, or substitutions) required to change
        one string into the other.

        Args:
            s1: First string
            s2: Second string

        Returns:
            The edit distance between the two strings

        Example:
            >>> FuzzyMatcher.levenshtein_distance("kitten", "sitting")
            3
        """
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
        """Calculate similarity ratio between two strings.

        Uses Levenshtein distance to compute a similarity score between 0 and 1,
        where 1 means identical strings. Comparison is case-insensitive.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0

        Example:
            >>> FuzzyMatcher.similarity("hello", "hello")
            1.0
            >>> FuzzyMatcher.similarity("hello", "world")
            0.0
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        distance = FuzzyMatcher.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))

        return 1.0 - (distance / max_len)

    @staticmethod
    def match(text: str, pattern: str, threshold: float = 0.8) -> bool:
        """Check if pattern matches text above similarity threshold.

        Args:
            text: Text to search in
            pattern: Pattern to match against
            threshold: Minimum similarity score (0.0 to 1.0), default 0.8

        Returns:
            True if similarity >= threshold, False otherwise

        Example:
            >>> FuzzyMatcher.match("Click Here Button", "click here", threshold=0.7)
            True
        """
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
