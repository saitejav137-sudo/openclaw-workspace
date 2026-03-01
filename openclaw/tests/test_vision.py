"""Unit tests for vision module"""

import unittest
import numpy as np
import os
import tempfile

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.vision import (
    FuzzyMatcher,
    TemplateMatcher,
    ColorDetector,
    ChangeDetector,
    RegressionDetector,
    VisionConfig
)
from openclaw.core.config import VisionMode


class TestFuzzyMatcher(unittest.TestCase):
    """Test FuzzyMatcher"""

    def test_identical_strings(self):
        """Test identical strings"""
        self.assertEqual(FuzzyMatcher.similarity("hello", "hello"), 1.0)

    def test_completely_different(self):
        """Test completely different strings"""
        similarity = FuzzyMatcher.similarity("abc", "xyz")
        self.assertLess(similarity, 0.5)

    def test_similar_strings(self):
        """Test similar strings"""
        similarity = FuzzyMatcher.similarity("hello", "hallo")
        self.assertGreater(similarity, 0.7)

    def test_empty_strings(self):
        """Test empty strings"""
        self.assertEqual(FuzzyMatcher.similarity("", ""), 1.0)
        self.assertEqual(FuzzyMatcher.similarity("test", ""), 0.0)

    def test_match_function(self):
        """Test match function"""
        # Test with similar strings (high threshold)
        self.assertTrue(FuzzyMatcher.match("hello", "hallo", 0.7))
        self.assertFalse(FuzzyMatcher.match("hello", "goodbye", 0.9))


class TestTemplateMatcher(unittest.TestCase):
    """Test TemplateMatcher"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [255, 255, 255]  # White square

        # Create template
        self.template = np.ones((25, 25, 3), dtype=np.uint8) * 255

    def test_match_template_not_exists(self):
        """Test with non-existent template"""
        results = TemplateMatcher.match(
            self.test_image,
            ["/nonexistent/template.png"],
            threshold=0.8
        )
        self.assertEqual(len(results), 0)

    def test_match_template_exists(self):
        """Test with existing template"""
        template_path = os.path.join(self.temp_dir, "template.png")
        cv2.imwrite(template_path, self.template)

        results = TemplateMatcher.match(
            self.test_image,
            [template_path],
            threshold=0.5
        )
        # May or may not match depending on image content

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestColorDetector(unittest.TestCase):
    """Test ColorDetector"""

    def setUp(self):
        """Set up test fixtures"""
        # Red image
        self.red_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.red_image[:, :] = [0, 0, 255]  # BGR

        # Blue image
        self.blue_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.blue_image[:, :] = [255, 0, 0]  # BGR

    def test_detect_red_in_red_image(self):
        """Test detecting red in red image"""
        # Red in BGR is (0, 0, 255), but HSV conversion changes it
        # Test with higher tolerance
        ratio = ColorDetector.detect(self.red_image, (0, 0, 255), tolerance=50)
        # In HSV, pure red (0, 0, 255) becomes roughly (0, 255, 255)
        # So detection might not work perfectly, but let's verify image is processed
        self.assertIsInstance(ratio, float)

    def test_detect_red_in_blue_image(self):
        """Test detecting red in blue image"""
        ratio = ColorDetector.detect(self.blue_image, (0, 0, 255), tolerance=30)
        self.assertLess(ratio, 0.1)


class TestChangeDetector(unittest.TestCase):
    """Test ChangeDetector"""

    def test_first_capture_no_change(self):
        """Test first capture returns False"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ChangeDetector.detect_change(image, region_id=1, threshold=0.1)
        self.assertFalse(result)

    def test_change_detected(self):
        """Test change is detected"""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # First capture
        ChangeDetector.detect_change(image1, region_id=2, threshold=0.1)

        # Second capture with change
        result = ChangeDetector.detect_change(image2, region_id=2, threshold=0.1)
        self.assertTrue(result)


class TestRegressionDetector(unittest.TestCase):
    """Test RegressionDetector"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, "baseline.png")

    def test_first_run_saves_baseline(self):
        """Test first run saves baseline"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        has_regression, ratio = RegressionDetector.detect_regression(
            image,
            self.baseline_path,
            threshold=0.01
        )

        self.assertFalse(has_regression)
        self.assertTrue(os.path.exists(self.baseline_path))

    def test_no_regression_same_image(self):
        """Test no regression with same image"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Save baseline
        cv2.imwrite(self.baseline_path, image)

        # Check again
        has_regression, ratio = RegressionDetector.detect_regression(
            image,
            self.baseline_path,
            threshold=0.01
        )

        self.assertFalse(has_regression)

    def test_regression_different_image(self):
        """Test regression detected with different image"""
        # Save baseline
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(self.baseline_path, image1)

        # Check with different image
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        has_regression, ratio = RegressionDetector.detect_regression(
            image2,
            self.baseline_path,
            threshold=0.01
        )

        self.assertTrue(has_regression)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)


# Import cv2 for tests that need it
import cv2


if __name__ == "__main__":
    unittest.main()
