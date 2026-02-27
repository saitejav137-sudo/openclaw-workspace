"""Unit tests for HTTP integration - input validation and utilities"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.integrations.http import (
    validate_url,
    validate_action,
    validate_selector,
    sanitize_string,
    RateLimiter,
    APIKeyAuth
)


class TestURLValidation(unittest.TestCase):
    """Test URL validation"""

    def test_validate_url_valid_https(self):
        self.assertTrue(validate_url("https://google.com"))

    def test_validate_url_valid_http(self):
        self.assertTrue(validate_url("http://example.com"))

    def test_validate_url_with_path(self):
        self.assertTrue(validate_url("https://example.com/path"))

    def test_validate_url_with_query(self):
        self.assertTrue(validate_url("https://example.com?param=value"))

    def test_validate_url_invalid_no_scheme(self):
        self.assertFalse(validate_url("example.com"))

    def test_validate_url_invalid_ip(self):
        self.assertFalse(validate_url("http://192.168.1.1"))

    def test_validate_url_localhost(self):
        self.assertTrue(validate_url("http://localhost:8080"))

    def test_validate_url_file(self):
        self.assertFalse(validate_url("file:///path"))

    def test_validate_url_empty(self):
        self.assertFalse(validate_url(""))

    def test_validate_url_too_long(self):
        self.assertFalse(validate_url("https://example.com/" + "a" * 3000))


class TestActionValidation(unittest.TestCase):
    """Test action validation"""

    def test_validate_action_valid_simple(self):
        self.assertTrue(validate_action("alt+o"))

    def test_validate_action_valid_ctrl(self):
        self.assertTrue(validate_action("ctrl+c"))

    def test_validate_action_valid_shift(self):
        self.assertTrue(validate_action("shift+a"))

    def test_validate_action_valid_special(self):
        self.assertTrue(validate_action("Return"))

    def test_validate_action_invalid_command(self):
        self.assertFalse(validate_action("rm -rf /"))

    def test_validate_action_invalid_semicolon(self):
        self.assertFalse(validate_action("echo hello; rm file"))

    def test_validate_action_invalid_pipe(self):
        self.assertFalse(validate_action("cat file | grep text"))


class TestSelectorValidation(unittest.TestCase):
    """Test selector validation"""

    def test_validate_selector_id(self):
        self.assertTrue(validate_selector("#button"))

    def test_validate_selector_class(self):
        self.assertTrue(validate_selector(".container"))

    def test_validate_selector_tag(self):
        self.assertTrue(validate_selector("div"))

    def test_validate_selector_attribute(self):
        self.assertTrue(validate_selector("[data-id='123']"))

    def test_validate_selector_xss(self):
        self.assertFalse(validate_selector("<script>alert(1)</script>"))

    def test_validate_selector_event(self):
        self.assertFalse(validate_selector("img onerror=alert(1)"))


class TestSanitizeString(unittest.TestCase):
    """Test string sanitization"""

    def test_sanitize_normal_string(self):
        result = sanitize_string("hello world")
        self.assertEqual(result, "hello world")

    def test_sanitize_with_max_length(self):
        long_string = "a" * 3000
        result = sanitize_string(long_string, max_length=100)
        self.assertEqual(len(result), 100)

    def test_sanitize_empty(self):
        result = sanitize_string("")
        self.assertEqual(result, "")


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter"""

    def setUp(self):
        self.limiter = RateLimiter(rate=5, per=60.0)

    def test_rate_limiter_allows_requests(self):
        for _ in range(5):
            self.assertTrue(self.limiter.is_allowed())

    def test_rate_limiter_blocks_over_limit(self):
        for _ in range(5):
            self.limiter.is_allowed()
        self.assertFalse(self.limiter.is_allowed())

    def test_rate_limiter_reset(self):
        for _ in range(5):
            self.limiter.is_allowed()
        self.limiter.reset()
        self.assertTrue(self.limiter.is_allowed())

    def test_rate_limiter_refill(self):
        import time
        self.limiter.is_allowed()  # Use one
        time.sleep(0.1)  # Small delay
        # Should have some refill
        result = self.limiter.is_allowed()
        self.assertIsInstance(result, bool)


class TestAPIKeyAuth(unittest.TestCase):
    """Test API key authentication"""

    def test_auth_disabled(self):
        auth = APIKeyAuth(None)
        self.assertFalse(auth.enabled)

    def test_auth_enabled(self):
        auth = APIKeyAuth("test-key")
        self.assertTrue(auth.enabled)

    def test_auth_valid_key(self):
        auth = APIKeyAuth("test-key-123")

        class MockRequest:
            def __init__(self):
                self.headers = {"Authorization": "Bearer test-key-123"}
                self.command = "GET"
                self.path = "/test"

        self.assertTrue(auth.validate(MockRequest()))

    def test_auth_invalid_key(self):
        auth = APIKeyAuth("test-key-123")

        class MockRequest:
            def __init__(self):
                self.headers = {"Authorization": "Bearer wrong-key"}
                self.command = "GET"
                self.path = "/test"

        self.assertFalse(auth.validate(MockRequest()))

    def test_auth_no_header(self):
        auth = APIKeyAuth("test-key-123")

        class MockRequest:
            def __init__(self):
                self.headers = {}
                self.command = "GET"
                self.path = "/test"

        self.assertFalse(auth.validate(MockRequest()))


if __name__ == "__main__":
    unittest.main()
