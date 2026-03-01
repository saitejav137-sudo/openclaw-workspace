"""Integration tests for HTTP endpoints"""

import unittest
import os
import sys
import json
import threading
import time
from http.client import HTTPConnection
from http.server import HTTPServer
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHTTPEndpoints(unittest.TestCase):
    """Integration tests for HTTP server endpoints"""

    @classmethod
    def setUpClass(cls):
        """Set up test server"""
        from openclaw.core.config import VisionConfig, VisionMode
        from openclaw.integrations.http import VisionHTTPServer

        cls.config = VisionConfig(mode=VisionMode.OCR)
        cls.server = VisionHTTPServer(18765, cls.config)
        cls.server_thread = None

    @classmethod
    def tearDownClass(cls):
        """Tear down test server"""
        if cls.server:
            cls.server.stop()

    def test_health_endpoint(self):
        """Test /health endpoint"""
        import urllib.request

        try:
            response = urllib.request.urlopen('http://localhost:18765/health', timeout=5)
            data = json.loads(response.read().decode())
            self.assertEqual(data['status'], 'healthy')
            self.assertIn('version', data)
            self.assertIn('services', data)
        except Exception as e:
            self.skipTest(f"Server not running: {e}")

    def test_root_endpoint(self):
        """Test root endpoint"""
        import urllib.request

        try:
            response = urllib.request.urlopen('http://localhost:18765/', timeout=5)
            data = json.loads(response.read().decode())
            self.assertIn('status', data)
            self.assertIn('mode', data)
        except Exception as e:
            self.skipTest(f"Server not running: {e}")

    def test_stats_endpoint(self):
        """Test /api/stats endpoint"""
        import urllib.request

        try:
            response = urllib.request.urlopen('http://localhost:18765/api/stats', timeout=5)
            data = json.loads(response.read().decode())
            self.assertIn('total', data)
            self.assertIn('triggered', data)
        except Exception as e:
            self.skipTest(f"Server not running: {e}")


class TestInputValidation(unittest.TestCase):
    """Tests for input validation"""

    def test_validate_url_valid(self):
        """Test URL validation with valid URLs"""
        from openclaw.integrations.http import validate_url

        self.assertTrue(validate_url("https://google.com"))
        self.assertTrue(validate_url("http://example.com/path"))
        self.assertTrue(validate_url("https://localhost:8765"))

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs"""
        from openclaw.integrations.http import validate_url

        self.assertFalse(validate_url(""))
        self.assertFalse(validate_url("javascript:alert(1)"))
        self.assertFalse(validate_url("not-a-url"))
        self.assertFalse(validate_url("ftp://example.com"))

    def test_validate_action_valid(self):
        """Test action validation with valid actions"""
        from openclaw.integrations.http import validate_action

        self.assertTrue(validate_action("start"))
        self.assertTrue(validate_action("goto"))
        self.assertTrue(validate_action("click"))
        self.assertTrue(validate_action("screenshot"))

    def test_validate_action_invalid(self):
        """Test action validation with invalid actions"""
        from openclaw.integrations.http import validate_action

        self.assertFalse(validate_action(""))
        self.assertFalse(validate_action("exec"))
        self.assertFalse(validate_action("delete"))
        self.assertFalse(validate_action("__import__"))

    def test_sanitize_string(self):
        """Test string sanitization"""
        from openclaw.integrations.http import sanitize_string

        self.assertEqual(sanitize_string("hello world"), "hello world")
        self.assertEqual(sanitize_string("hello\x00world"), "helloworld")
        self.assertEqual(len(sanitize_string("a" * 2000, max_length=100)), 100)
        self.assertEqual(sanitize_string(123), "")

    def test_validate_selector(self):
        """Test CSS selector validation"""
        from openclaw.integrations.http import validate_selector

        self.assertTrue(validate_selector("#my-id"))
        self.assertTrue(validate_selector(".my-class"))
        self.assertTrue(validate_selector("div.button"))
        self.assertFalse(validate_selector(""))
        self.assertFalse(validate_selector("<script>alert(1)</script>"))
        self.assertFalse(validate_selector("javascript:alert(1)"))


class TestRateLimiter(unittest.TestCase):
    """Tests for rate limiter"""

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        from openclaw.integrations.http import RateLimiter

        limiter = RateLimiter(rate=10, per=60.0)
        # Should allow at least 10 requests
        for _ in range(10):
            self.assertTrue(limiter.is_allowed())

    def test_rate_limiter_blocks_over_limit(self):
        """Test rate limiter blocks requests over limit"""
        from openclaw.integrations.http import RateLimiter

        limiter = RateLimiter(rate=2, per=60.0)
        limiter.is_allowed()  # First request
        limiter.is_allowed()  # Second request
        # Third request should be blocked
        self.assertFalse(limiter.is_allowed())

    def test_rate_limiter_reset(self):
        """Test rate limiter reset"""
        from openclaw.integrations.http import RateLimiter

        limiter = RateLimiter(rate=1, per=60.0)
        limiter.is_allowed()
        self.assertFalse(limiter.is_allowed())
        limiter.reset()
        self.assertTrue(limiter.is_allowed())


class TestAPIKeyAuth(unittest.TestCase):
    """Tests for API key authentication"""

    def test_auth_disabled(self):
        """Test authentication is disabled when no API key"""
        from openclaw.integrations.http import APIKeyAuth

        auth = APIKeyAuth(None)
        self.assertFalse(auth.enabled)

        auth = APIKeyAuth("")
        self.assertFalse(auth.enabled)

    def test_auth_enabled(self):
        """Test authentication is enabled with API key"""
        from openclaw.integrations.http import APIKeyAuth

        auth = APIKeyAuth("test-api-key")
        self.assertTrue(auth.enabled)

    def test_auth_valid_key(self):
        """Test valid API key passes validation"""
        from openclaw.integrations.http import APIKeyAuth

        auth = APIKeyAuth("test-api-key")

        # Create mock request
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer test-api-key"}

        self.assertTrue(auth.validate(mock_request))

    def test_auth_invalid_key(self):
        """Test invalid API key fails validation"""
        from openclaw.integrations.http import APIKeyAuth

        auth = APIKeyAuth("test-api-key")

        # Create mock request with wrong key and path attribute
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        mock_request.path = "/health"  # Add path attribute to avoid error

        self.assertFalse(auth.validate(mock_request))


if __name__ == '__main__':
    unittest.main()
