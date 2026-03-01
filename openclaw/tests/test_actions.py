"""Unit tests for actions module"""

import unittest
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.actions import (
    RetryConfig,
    RetryStrategy,
    ActionExecutor,
    RetryableError,
    KeyboardAction,
    MouseAction,
    ActionSequence
)


class TestRetryConfig(unittest.TestCase):
    """Test RetryConfig"""

    def test_default_config(self):
        """Test default retry config"""
        config = RetryConfig()
        self.assertEqual(config.attempts, 3)
        self.assertEqual(config.delay, 1.0)
        self.assertEqual(config.strategy, RetryStrategy.EXPONENTIAL)

    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            "retry_attempts": 5,
            "retry_delay": 2.0,
            "strategy": "fixed"
        }
        config = RetryConfig.from_dict(data)
        self.assertEqual(config.attempts, 5)
        self.assertEqual(config.delay, 2.0)
        self.assertEqual(config.strategy, RetryStrategy.FIXED)


class TestActionExecutor(unittest.TestCase):
    """Test ActionExecutor"""

    def test_successful_execution(self):
        """Test successful function execution"""
        executor = ActionExecutor(RetryConfig(attempts=3))

        def success_func():
            return "success"

        result = executor.execute_with_retry(success_func)
        self.assertEqual(result, "success")

    def test_failure_then_success(self):
        """Test retry on failure then success"""
        executor = ActionExecutor(RetryConfig(attempts=3, delay=0.1))

        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RetryableError("Temporary error")
            return "success"

        result = executor.execute_with_retry(flaky_func)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 2)

    def test_all_attempts_fail(self):
        """Test all attempts fail"""
        executor = ActionExecutor(RetryConfig(attempts=3, delay=0.1))

        def always_fails():
            raise RetryableError("Permanent error")

        with self.assertRaises(RetryableError):
            executor.execute_with_retry(always_fails)


class TestRetryStrategy(unittest.TestCase):
    """Test retry delay calculations"""

    def test_fixed_strategy(self):
        """Test fixed delay"""
        config = RetryConfig(attempts=3, delay=1.0, strategy=RetryStrategy.FIXED)
        executor = ActionExecutor(config)

        self.assertEqual(executor.calculate_delay(1), 1.0)
        self.assertEqual(executor.calculate_delay(2), 1.0)
        self.assertEqual(executor.calculate_delay(3), 1.0)

    def test_linear_strategy(self):
        """Test linear backoff"""
        config = RetryConfig(attempts=3, delay=1.0, strategy=RetryStrategy.LINEAR)
        executor = ActionExecutor(config)

        self.assertEqual(executor.calculate_delay(1), 1.0)
        self.assertEqual(executor.calculate_delay(2), 2.0)
        self.assertEqual(executor.calculate_delay(3), 3.0)

    def test_exponential_strategy(self):
        """Test exponential backoff"""
        config = RetryConfig(attempts=3, delay=1.0, backoff_multiplier=2.0)
        executor = ActionExecutor(config)

        self.assertEqual(executor.calculate_delay(1), 1.0)
        self.assertEqual(executor.calculate_delay(2), 2.0)
        self.assertEqual(executor.calculate_delay(3), 4.0)

    def test_max_delay(self):
        """Test max delay cap"""
        config = RetryConfig(attempts=5, delay=1.0, backoff_multiplier=2.0, max_delay=2.0)
        executor = ActionExecutor(config)

        self.assertEqual(executor.calculate_delay(1), 1.0)
        self.assertEqual(executor.calculate_delay(2), 2.0)
        self.assertEqual(executor.calculate_delay(3), 2.0)  # Capped at max_delay


class TestActionSequence(unittest.TestCase):
    """Test ActionSequence"""

    def test_empty_sequence(self):
        """Test empty sequence"""
        sequence = ActionSequence()
        result = sequence.execute([])
        self.assertTrue(result)

    def test_wait_action(self):
        """Test wait action"""
        sequence = ActionSequence()
        actions = [{"type": "wait", "delay": 0.1}]

        start = time.time()
        result = sequence.execute(actions)
        elapsed = time.time() - start

        self.assertTrue(result)
        self.assertGreater(elapsed, 0.05)


if __name__ == "__main__":
    unittest.main()
