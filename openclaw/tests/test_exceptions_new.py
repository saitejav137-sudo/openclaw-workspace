"""Unit tests for exception handling"""

import unittest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.exceptions import (
    OpenClawError,
    VisionError,
    OCRError,
    YOLOError,
    TemplateMatchError,
    ScreenCaptureError,
    ConfigError,
    ConfigValidationError,
    ActionError,
    KeyPressError,
    BrowserError,
    NetworkError,
    HTTPError,
    AuthError,
    InvalidAPIKeyError,
    RateLimitExceededError,
    StorageError,
    DatabaseError,
    is_retryable,
)


class TestExceptionHierarchy(unittest.TestCase):
    """Test exception hierarchy"""

    def test_base_exception(self):
        """Test OpenClawError base exception"""
        error = OpenClawError("Test error", code="TEST_ERROR")
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.code, "TEST_ERROR")
        self.assertIsNotNone(error.timestamp)
        # details defaults to empty dict, not None
        self.assertEqual(error.details, {})

    def test_exception_to_dict(self):
        """Test exception serialization to dict"""
        error = OpenClawError(
            "Test error",
            code="TEST_ERROR",
            details={"key": "value"}
        )
        result = error.to_dict()
        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "TEST_ERROR")
        self.assertEqual(result["error"]["message"], "Test error")
        self.assertIn("timestamp", result["error"])

    def test_vision_exceptions(self):
        """Test vision-related exceptions"""
        error = VisionError("Vision error")
        self.assertEqual(error.code, "VISION_ERROR")

        error = OCRError("OCR error")
        self.assertEqual(error.code, "OCR_ERROR")

        error = YOLOError("YOLO error")
        self.assertEqual(error.code, "YOLO_ERROR")

        error = TemplateMatchError("Template error")
        self.assertEqual(error.code, "TEMPLATE_ERROR")

        error = ScreenCaptureError("Capture error")
        self.assertEqual(error.code, "CAPTURE_ERROR")

    def test_config_exceptions(self):
        """Test config-related exceptions"""
        error = ConfigError("Config error")
        self.assertEqual(error.code, "CONFIG_ERROR")

        error = ConfigValidationError("Validation failed", field="mode")
        self.assertEqual(error.code, "CONFIG_VALIDATION_ERROR")
        self.assertEqual(error.details.get("field"), "mode")

    def test_action_exceptions(self):
        """Test action-related exceptions"""
        error = ActionError("Action error")
        self.assertEqual(error.code, "ACTION_ERROR")

        error = KeyPressError("Key press failed", key="ctrl+c")
        self.assertEqual(error.code, "KEY_PRESS_ERROR")
        self.assertEqual(error.details.get("key"), "ctrl+c")

    def test_network_exceptions(self):
        """Test network-related exceptions"""
        error = HTTPError(404, "Not found", url="https://example.com")
        self.assertEqual(error.code, "HTTP_ERROR")
        self.assertEqual(error.details.get("status_code"), 404)

    def test_auth_exceptions(self):
        """Test auth-related exceptions"""
        error = InvalidAPIKeyError()
        self.assertEqual(error.code, "INVALID_API_KEY")

        error = RateLimitExceededError(limit=60, window=60)
        self.assertEqual(error.code, "RATE_LIMIT_EXCEEDED")
        self.assertEqual(error.details.get("limit"), 60)

    def test_storage_exceptions(self):
        """Test storage-related exceptions"""
        error = StorageError("Storage error")
        self.assertEqual(error.code, "STORAGE_ERROR")

        error = DatabaseError("DB error")
        self.assertEqual(error.code, "DATABASE_ERROR")


class TestIsRetryable(unittest.TestCase):
    """Test is_retryable utility"""

    def test_retryable_network_errors(self):
        """Test network errors are retryable"""
        error = NetworkError("Network error")
        self.assertTrue(is_retryable(error))

        error = HTTPError(500, "Server error")
        self.assertTrue(is_retryable(error))

    def test_retryable_timeout(self):
        """Test timeout errors are retryable"""
        self.assertTrue(is_retryable(TimeoutError()))

    def test_not_retryable_config_errors(self):
        """Test config errors are not retryable"""
        error = ConfigError("Config error")
        self.assertFalse(is_retryable(error))

    def test_not_retryable_auth_errors(self):
        """Test auth errors are not retryable"""
        error = InvalidAPIKeyError()
        self.assertFalse(is_retryable(error))


if __name__ == "__main__":
    unittest.main()
