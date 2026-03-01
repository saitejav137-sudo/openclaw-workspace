"""Unit tests for custom exceptions"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.exceptions import (
    OpenClawError,
    VisionError, OCRError, YOLOError, TemplateMatchError, ScreenCaptureError,
    ConfigError, ConfigValidationError, ConfigNotFoundError,
    ActionError, KeyPressError, MouseActionError, ActionTimeoutError,
    BrowserError, BrowserNotFoundError, BrowserNavigationError, ElementNotFoundError,
    NetworkError, HTTPError, WebSocketError,
    AuthError, InvalidAPIKeyError, RateLimitExceededError,
    StorageError, DatabaseError,
    is_retryable
)


class TestOpenClawError(unittest.TestCase):
    """Test base OpenClawError"""

    def test_basic_error(self):
        err = OpenClawError("Test error")
        self.assertEqual(err.message, "Test error")
        self.assertEqual(err.code, "OC_ERROR")
        self.assertIsNotNone(err.timestamp)

    def test_error_with_code(self):
        err = OpenClawError("Test", code="TEST_CODE")
        self.assertEqual(err.code, "TEST_CODE")

    def test_error_with_details(self):
        err = OpenClawError("Test", details={"key": "value"})
        self.assertEqual(err.details["key"], "value")

    def test_to_dict(self):
        err = OpenClawError("Test")
        d = err.to_dict()
        self.assertIn("error", d)
        self.assertEqual(d["error"]["message"], "Test")


class TestVisionErrors(unittest.TestCase):
    """Test vision-related errors"""

    def test_vision_error(self):
        err = VisionError("Vision failed")
        self.assertEqual(err.code, "VISION_ERROR")

    def test_ocr_error(self):
        err = OCRError("OCR failed")
        self.assertEqual(err.code, "OCR_ERROR")

    def test_yolo_error(self):
        err = YOLOError("YOLO failed")
        self.assertEqual(err.code, "YOLO_ERROR")

    def test_template_error(self):
        err = TemplateMatchError("Template not found")
        self.assertEqual(err.code, "TEMPLATE_ERROR")

    def test_capture_error(self):
        err = ScreenCaptureError("Capture failed")
        self.assertEqual(err.code, "CAPTURE_ERROR")


class TestConfigErrors(unittest.TestCase):
    """Test configuration errors"""

    def test_config_error(self):
        err = ConfigError("Config invalid")
        self.assertEqual(err.code, "CONFIG_ERROR")

    def test_config_validation_error(self):
        err = ConfigValidationError("Invalid value", field="mode")
        self.assertEqual(err.code, "CONFIG_VALIDATION_ERROR")
        self.assertEqual(err.details["field"], "mode")

    def test_config_not_found(self):
        err = ConfigNotFoundError("/path/to/config.yaml")
        self.assertEqual(err.code, "CONFIG_NOT_FOUND")
        self.assertEqual(err.details["path"], "/path/to/config.yaml")


class TestActionErrors(unittest.TestCase):
    """Test action errors"""

    def test_action_error(self):
        err = ActionError("Action failed")
        self.assertEqual(err.code, "ACTION_ERROR")

    def test_keypress_error(self):
        err = KeyPressError("Key press failed", key="alt+o")
        self.assertEqual(err.code, "KEY_PRESS_ERROR")
        self.assertEqual(err.details["key"], "alt+o")

    def test_mouse_error(self):
        err = MouseActionError("Click failed")
        self.assertEqual(err.code, "MOUSE_ACTION_ERROR")

    def test_timeout_error(self):
        err = ActionTimeoutError("alt+o", 5.0)
        self.assertEqual(err.code, "ACTION_TIMEOUT")
        self.assertEqual(err.details["action"], "alt+o")
        self.assertEqual(err.details["timeout"], 5.0)


class TestBrowserErrors(unittest.TestCase):
    """Test browser errors"""

    def test_browser_error(self):
        err = BrowserError("Browser failed")
        self.assertEqual(err.code, "BROWSER_ERROR")

    def test_browser_not_found(self):
        err = BrowserNotFoundError("chromium")
        self.assertEqual(err.code, "BROWSER_NOT_FOUND")

    def test_navigation_error(self):
        err = BrowserNavigationError("https://example.com", "Timeout")
        self.assertEqual(err.code, "BROWSER_NAVIGATION_ERROR")
        self.assertEqual(err.details["url"], "https://example.com")

    def test_element_not_found(self):
        err = ElementNotFoundError("#button")
        self.assertEqual(err.code, "ELEMENT_NOT_FOUND")
        self.assertEqual(err.details["selector"], "#button")


class TestNetworkErrors(unittest.TestCase):
    """Test network errors"""

    def test_network_error(self):
        err = NetworkError("Network failed")
        self.assertEqual(err.code, "NETWORK_ERROR")

    def test_http_error(self):
        err = HTTPError(404, "Not found", "https://example.com")
        self.assertEqual(err.code, "HTTP_ERROR")
        self.assertEqual(err.details["status_code"], 404)

    def test_websocket_error(self):
        err = WebSocketError("Connection closed")
        self.assertEqual(err.code, "WEBSOCKET_ERROR")


class TestAuthErrors(unittest.TestCase):
    """Test authentication errors"""

    def test_auth_error(self):
        err = AuthError("Auth failed")
        self.assertEqual(err.code, "AUTH_ERROR")

    def test_invalid_api_key(self):
        err = InvalidAPIKeyError()
        self.assertEqual(err.code, "INVALID_API_KEY")

    def test_rate_limit_exceeded(self):
        err = RateLimitExceededError(60, 60.0)
        self.assertEqual(err.code, "RATE_LIMIT_EXCEEDED")
        self.assertEqual(err.details["limit"], 60)


class TestStorageErrors(unittest.TestCase):
    """Test storage errors"""

    def test_storage_error(self):
        err = StorageError("Storage failed")
        self.assertEqual(err.code, "STORAGE_ERROR")

    def test_database_error(self):
        err = DatabaseError("DB error")
        self.assertEqual(err.code, "DATABASE_ERROR")


class TestIsRetryable(unittest.TestCase):
    """Test is_retryable function"""

    def test_retryable_network_error(self):
        err = NetworkError("Network error")
        self.assertTrue(is_retryable(err))

    def test_retryable_http_error(self):
        err = HTTPError(500, "Server error")
        self.assertTrue(is_retryable(err))

    def test_retryable_timeout(self):
        err = ActionTimeoutError("action", 5.0)
        self.assertTrue(is_retryable(err))

    def test_not_retryable_config(self):
        err = ConfigError("Invalid config")
        self.assertFalse(is_retryable(err))

    def test_not_retryable_auth(self):
        err = AuthError("Auth failed")
        self.assertFalse(is_retryable(err))

    def test_retryable_exception(self):
        err = ConnectionError("Connection refused")
        self.assertTrue(is_retryable(err))

    def test_retryable_timeout_error(self):
        err = TimeoutError("Timed out")
        self.assertTrue(is_retryable(err))


if __name__ == "__main__":
    unittest.main()
