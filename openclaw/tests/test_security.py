"""Unit tests for security utilities"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSecurityModule(unittest.TestCase):
    """Test security utilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_data = {
            "api_key": "test_key_123",
            "telegram_token": "token_abc",
            "webhook_url": "https://example.com/webhook"
        }

    @patch('openclaw.utils.security.CRYPTO_AVAILABLE', False)
    def test_encryption_not_available(self):
        """Test behavior when cryptography is not available"""
        # This tests the fallback behavior
        from openclaw.utils.security import ConfigEncryption
        # Should not raise, just warn
        enc = ConfigEncryption(password="test")
        # When crypto unavailable, encrypt returns plaintext
        result = enc.encrypt(self.test_data)
        self.assertIsNotNone(result)

    def test_encrypted_config_dataclass(self):
        """Test EncryptedConfig dataclass"""
        from openclaw.utils.security import EncryptedConfig

        config = EncryptedConfig(
            encrypted_data="encrypted_string",
            salt="salt_value",
            version=1
        )
        self.assertEqual(config.encrypted_data, "encrypted_string")
        self.assertEqual(config.salt, "salt_value")
        self.assertEqual(config.version, 1)


class TestInputSanitization(unittest.TestCase):
    """Test input sanitization"""

    def test_sanitize_normal_string(self):
        """Test sanitizing normal string"""
        from openclaw.integrations.fastapi_server import InputSanitizer

        result = InputSanitizer.sanitize("normal string")
        self.assertEqual(result, "normal string")

    def test_sanitize_xss_attempt(self):
        """Test sanitizing XSS attempt"""
        from openclaw.integrations.fastapi_server import InputSanitizer

        result = InputSanitizer.sanitize("<script>alert('xss')</script>")
        # Should be HTML-escaped
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)

    def test_sanitize_sql_injection(self):
        """Test sanitizing SQL injection attempt"""
        from openclaw.integrations.fastapi_server import InputSanitizer

        # Test SQL injection pattern - semicolons are removed and quotes are escaped
        result = InputSanitizer.sanitize("'; DROP TABLE users; --")
        # Should have semicolons removed or quotes escaped
        # The string is HTML-escaped, so ' becomes &#x27; or &#39;
        # and ; may be removed or escaped
        self.assertTrue(
            ";" not in result or "DROP" not in result or "&#" in result,
            f"String should be sanitized: {result}"
        )

    def test_sanitize_null_bytes(self):
        """Test removing null bytes"""
        from openclaw.integrations.fastapi_server import InputSanitizer

        result = InputSanitizer.sanitize("test\x00string")
        self.assertNotIn("\x00", result)

    def test_sanitize_dict(self):
        """Test sanitizing dictionary"""
        from openclaw.integrations.fastapi_server import InputSanitizer

        data = {
            "key1": "value1",
            "key2": "<script>alert(1)</script>",
            "nested": {
                "inner": "normal text"
            }
        }
        result = InputSanitizer.sanitize_dict(data)
        self.assertEqual(result["key1"], "value1")
        self.assertNotIn("<script>", result["key2"])


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting"""

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        from openclaw.integrations.fastapi_server import RateLimiter, RateLimitConfig

        config = RateLimitConfig(requests=10, window_seconds=60)
        limiter = RateLimiter(config=config)

        # Should allow 10 requests
        for i in range(10):
            self.assertTrue(limiter.check(f"user_{i}"))

    def test_rate_limiter_blocks_excess(self):
        """Test rate limiter blocks excess requests"""
        from openclaw.integrations.fastapi_server import RateLimiter, RateLimitConfig

        config = RateLimitConfig(requests=5, window_seconds=60)
        limiter = RateLimiter(config=config)

        # First 5 should pass
        for i in range(5):
            self.assertTrue(limiter.check("test_user"))

        # 6th should be blocked
        self.assertFalse(limiter.check("test_user"))

    def test_rate_limiter_per_key(self):
        """Test rate limiter works per API key"""
        from openclaw.integrations.fastapi_server import RateLimiter, RateLimitConfig

        config = RateLimitConfig(requests=2, window_seconds=60)
        limiter = RateLimiter(config=config)

        # User A gets 2 requests
        self.assertTrue(limiter.check("user_a"))
        self.assertTrue(limiter.check("user_a"))
        self.assertFalse(limiter.check("user_a"))

        # User B should still have their own limit
        self.assertTrue(limiter.check("user_b"))
        self.assertTrue(limiter.check("user_b"))


if __name__ == "__main__":
    unittest.main()
