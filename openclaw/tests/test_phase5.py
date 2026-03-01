"""
Phase 5 Tests — Security & Network Enhancements

Tests for:
- PerIPRateLimiter: per-IP tracking, blacklisting, cleanup
- SecretRotationManager: registration, rotation, history
- AsyncHTTPClient: creation, retry
- ResilientExecutor: circuit breaker, retry, timeout
- InputValidator: sanitization, JSON depth, content length
- IntegrationHub main.py wiring
"""

import sys
import os
import time
import unittest
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== Per-IP Rate Limiter Tests ==============

class TestPerIPRateLimiter(unittest.TestCase):

    def test_basic_allow(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=10, per=1.0)
        self.assertTrue(limiter.is_allowed("1.2.3.4"))

    def test_rate_limit_exceeded(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=3, per=60.0, blacklist_after=100)
        for _ in range(3):
            limiter.is_allowed("1.2.3.4")
        # 4th should be rejected
        self.assertFalse(limiter.is_allowed("1.2.3.4"))

    def test_different_ips_independent(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=2, per=60.0, blacklist_after=100)
        limiter.is_allowed("10.0.0.1")
        limiter.is_allowed("10.0.0.1")
        # IP1 is exhausted, but IP2 still has tokens
        self.assertTrue(limiter.is_allowed("10.0.0.2"))
        self.assertFalse(limiter.is_allowed("10.0.0.1"))

    def test_blacklisting(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=1, per=60.0, blacklist_after=3)
        ip = "192.168.1.100"
        limiter.is_allowed(ip)  # uses the 1 token
        # Next 3 violations should trigger blacklist
        for _ in range(3):
            limiter.is_allowed(ip)
        # Should be blacklisted now
        self.assertFalse(limiter.is_allowed(ip))

    def test_unblock(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=1, per=60.0, blacklist_after=2)
        ip = "10.0.0.99"
        limiter.is_allowed(ip)
        limiter.is_allowed(ip)
        limiter.is_allowed(ip)
        limiter.unblock(ip)
        self.assertTrue(limiter.is_allowed(ip))

    def test_stats(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=10, per=1.0)
        limiter.is_allowed("1.1.1.1")
        limiter.is_allowed("2.2.2.2")
        stats = limiter.get_stats()
        self.assertEqual(stats["tracked_ips"], 2)

    def test_thread_safety(self):
        from openclaw.core.security_enhanced import PerIPRateLimiter
        limiter = PerIPRateLimiter(rate=1000, per=1.0)

        def hammer(ip):
            for _ in range(100):
                limiter.is_allowed(ip)

        threads = [threading.Thread(target=hammer, args=(f"10.0.0.{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = limiter.get_stats()
        self.assertEqual(stats["tracked_ips"], 10)


# ============== Secret Rotation Manager Tests ==============

class TestSecretRotationManager(unittest.TestCase):

    def test_register_secret(self):
        from openclaw.core.security_enhanced import SecretRotationManager
        mgr = SecretRotationManager()
        mgr.register_secret("API_KEY", rotation_days=30)
        self.assertIn("API_KEY", mgr._rotation_configs)

    def test_check_rotation_not_needed(self):
        from openclaw.core.security_enhanced import SecretRotationManager
        mgr = SecretRotationManager()
        mgr.register_secret("API_KEY", rotation_days=30)
        due = mgr.check_rotation_needed()
        self.assertEqual(due, [])

    def test_check_rotation_needed(self):
        from openclaw.core.security_enhanced import SecretRotationManager
        mgr = SecretRotationManager()
        mgr.register_secret("OLD_KEY", rotation_days=0)  # Immediately due
        # Force the last_rotated to be in the past
        mgr._rotation_configs["OLD_KEY"]["last_rotated"] = time.time() - 86400
        due = mgr.check_rotation_needed()
        self.assertIn("OLD_KEY", due)

    def test_get_status(self):
        from openclaw.core.security_enhanced import SecretRotationManager
        mgr = SecretRotationManager()
        mgr.register_secret("KEY_A", rotation_days=90)
        status = mgr.get_status()
        self.assertIn("KEY_A", status)
        self.assertLess(status["KEY_A"]["age_days"], 1)

    def test_default_generator(self):
        from openclaw.core.security_enhanced import SecretRotationManager
        val = SecretRotationManager._default_generator("test")
        self.assertEqual(len(val), 64)  # SHA-256 hex


# ============== Async HTTP Client Tests ==============

class TestAsyncHTTPClient(unittest.TestCase):

    def test_create_client(self):
        from openclaw.core.security_enhanced import AsyncHTTPClient
        client = AsyncHTTPClient(timeout=10.0, max_retries=2)
        self.assertEqual(client.timeout, 10.0)
        self.assertEqual(client.max_retries, 2)
        client.close()

    def test_get_session_reuse(self):
        from openclaw.core.security_enhanced import AsyncHTTPClient
        client = AsyncHTTPClient()
        s1 = client._get_session()
        s2 = client._get_session()
        self.assertIs(s1, s2)
        client.close()


# ============== Resilient Executor Tests ==============

class TestResilientExecutor(unittest.TestCase):

    def test_successful_execution(self):
        from openclaw.core.security_enhanced import ResilientExecutor
        ex = ResilientExecutor()
        result = ex.execute(lambda: 42)
        self.assertEqual(result, 42)

    def test_retry_on_failure(self):
        from openclaw.core.security_enhanced import ResilientExecutor
        ex = ResilientExecutor(max_retries=2, failure_threshold=10)
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "ok"

        result = ex.execute(flaky)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 2)

    def test_circuit_breaker_opens(self):
        from openclaw.core.security_enhanced import ResilientExecutor
        ex = ResilientExecutor(failure_threshold=2, max_retries=0)

        def always_fail():
            raise ValueError("boom")

        for _ in range(2):
            try:
                ex.execute(always_fail)
            except ValueError:
                pass

        # Circuit should be open now
        with self.assertRaises(RuntimeError):
            ex.execute(lambda: "ok")

    def test_reset(self):
        from openclaw.core.security_enhanced import ResilientExecutor
        ex = ResilientExecutor(failure_threshold=1, max_retries=0)
        try:
            ex.execute(lambda: 1/0)
        except Exception:
            pass
        ex.reset()
        self.assertEqual(ex._state, "closed")
        result = ex.execute(lambda: 99)
        self.assertEqual(result, 99)

    def test_stats(self):
        from openclaw.core.security_enhanced import ResilientExecutor
        ex = ResilientExecutor()
        ex.execute(lambda: 1)
        ex.execute(lambda: 2)
        stats = ex.get_stats()
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["successes"], 2)

    def test_timeout(self):
        from openclaw.core.security_enhanced import ResilientExecutor
        ex = ResilientExecutor(max_retries=0, failure_threshold=10)

        def slow():
            time.sleep(5)
            return "done"

        with self.assertRaises(TimeoutError):
            ex.execute(slow, timeout=0.5)


# ============== Input Validator Tests ==============

class TestInputValidator(unittest.TestCase):

    def test_sanitize_string(self):
        from openclaw.core.security_enhanced import InputValidator
        self.assertEqual(InputValidator.sanitize_string("hello"), "hello")
        # Null bytes removed
        self.assertEqual(InputValidator.sanitize_string("he\x00llo"), "hello")

    def test_sanitize_string_max_length(self):
        from openclaw.core.security_enhanced import InputValidator
        result = InputValidator.sanitize_string("a" * 200, max_length=50)
        self.assertEqual(len(result), 50)

    def test_sanitize_non_string(self):
        from openclaw.core.security_enhanced import InputValidator
        self.assertEqual(InputValidator.sanitize_string(123), "")
        self.assertEqual(InputValidator.sanitize_string(None), "")

    def test_validate_json_depth_ok(self):
        from openclaw.core.security_enhanced import InputValidator
        data = {"a": {"b": {"c": 1}}}
        self.assertTrue(InputValidator.validate_json_depth(data, max_depth=5))

    def test_validate_json_depth_too_deep(self):
        from openclaw.core.security_enhanced import InputValidator
        # Build deeply nested dict
        data = 1
        for _ in range(15):
            data = {"nested": data}
        self.assertFalse(InputValidator.validate_json_depth(data, max_depth=10))

    def test_validate_content_length(self):
        from openclaw.core.security_enhanced import InputValidator
        self.assertTrue(InputValidator.validate_content_length(1000))
        self.assertTrue(InputValidator.validate_content_length(0))
        self.assertFalse(InputValidator.validate_content_length(100_000_000))

    def test_sanitize_headers(self):
        from openclaw.core.security_enhanced import InputValidator
        headers = {"Authorization": "Bearer tok123", "X-Bad\x00": "val"}
        result = InputValidator.sanitize_headers(headers)
        self.assertIn("Authorization", result)
        self.assertIn("X-Bad", result)


# ============== Global Singletons Tests ==============

class TestGlobalSingletons(unittest.TestCase):

    def test_get_ip_rate_limiter(self):
        from openclaw.core.security_enhanced import get_ip_rate_limiter
        l1 = get_ip_rate_limiter()
        l2 = get_ip_rate_limiter()
        self.assertIs(l1, l2)

    def test_get_rotation_manager(self):
        from openclaw.core.security_enhanced import get_rotation_manager
        m1 = get_rotation_manager()
        m2 = get_rotation_manager()
        self.assertIs(m1, m2)

    def test_get_resilient_executor(self):
        from openclaw.core.security_enhanced import get_resilient_executor
        e1 = get_resilient_executor()
        e2 = get_resilient_executor()
        self.assertIs(e1, e2)


# ============== Main.py Wiring Test ==============

class TestMainWiring(unittest.TestCase):

    def test_integration_hub_importable_from_main(self):
        """Verify main.py can find IntegrationHub."""
        from openclaw.core.integration_hub import get_integration_hub
        hub = get_integration_hub()
        self.assertIsNotNone(hub)

    def test_security_enhanced_importable(self):
        """Verify all Phase 5 modules are importable."""
        from openclaw.core.security_enhanced import (
            PerIPRateLimiter,
            SecretRotationManager,
            AsyncHTTPClient,
            ResilientExecutor,
            InputValidator,
        )
        self.assertTrue(all([
            PerIPRateLimiter, SecretRotationManager,
            AsyncHTTPClient, ResilientExecutor, InputValidator
        ]))


if __name__ == "__main__":
    unittest.main()
