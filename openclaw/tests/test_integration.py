"""Integration tests for OpenClaw core modules"""

import unittest
import os
import sys
import tempfile
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWorkspaceIntegration(unittest.TestCase):
    """Integration tests for workspace management"""

    def test_workspace_manager_init(self):
        """Test workspace manager initialization"""
        from openclaw.core.workspace import WorkspaceManager

        # Create fresh manager (not singleton)
        manager = WorkspaceManager()
        manager.initialize()

        workspaces = manager.list_workspaces()
        self.assertIn("development", workspaces)
        self.assertIn("staging", workspaces)
        self.assertIn("production", workspaces)

    def test_workspace_switch(self):
        """Test switching between workspaces"""
        from openclaw.core.workspace import (
            WorkspaceManager,
            EnvironmentType
        )

        manager = WorkspaceManager()
        manager.initialize()

        # Switch to staging
        result = manager.set_workspace("staging")
        self.assertTrue(result)

        workspace = manager.get_workspace("staging")
        self.assertEqual(workspace.environment, EnvironmentType.STAGING)

        # Switch to production
        manager.set_workspace("production")
        workspace = manager.get_workspace("production")
        self.assertEqual(workspace.environment, EnvironmentType.PRODUCTION)

    def test_workspace_env_vars(self):
        """Test workspace environment variables"""
        from openclaw.core.workspace import WorkspaceManager

        manager = WorkspaceManager()
        manager.initialize()
        manager.set_workspace("development")

        workspace = manager.get_workspace("development")

        # Check that environment variables are set
        self.assertEqual(os.environ.get("OPENCLAW_LOG_LEVEL"), workspace.log_level)
        self.assertEqual(os.environ.get("OPENCLAW_DEBUG"), str(workspace.debug_mode).lower())

    def test_custom_workspace(self):
        """Test creating custom workspace"""
        from openclaw.core.workspace import WorkspaceManager

        manager = WorkspaceManager()
        manager.initialize()

        # Create custom workspace
        workspace = manager.create_workspace(
            name="test_workspace",
            base_workspace="development",
            log_level="DEBUG",
            api_port=9000
        )

        self.assertEqual(workspace.name, "test_workspace")
        self.assertEqual(workspace.log_level, "DEBUG")
        self.assertEqual(workspace.api_port, 9000)


class TestMessageQueueIntegration(unittest.TestCase):
    """Integration tests for message queue"""

    def test_task_serialization(self):
        """Test task JSON serialization"""
        from openclaw.integrations.message_queue import Task

        task = Task(
            id="test-serial",
            type="test_type",
            payload={"key": "value"},
            priority=1
        )

        # Serialize and deserialize
        json_str = task.to_json()
        restored = Task.from_json(json_str)

        self.assertEqual(task.id, restored.id)
        self.assertEqual(task.type, restored.type)
        self.assertEqual(task.payload, restored.payload)

    def test_task_priority(self):
        """Test task priority ordering"""
        from openclaw.integrations.message_queue import Task, TaskPriority

        # Create tasks with different priorities
        low_task = Task(id="low", type="test", payload={}, priority=TaskPriority.LOW.value)
        high_task = Task(id="high", type="test", payload={}, priority=TaskPriority.HIGH.value)

        # Higher priority should have lower negative value (for PriorityQueue)
        self.assertLess(-high_task.priority, -low_task.priority)


class TestLoggerIntegration(unittest.TestCase):
    """Integration tests for logging"""

    def test_logger_creation(self):
        """Test logger creation"""
        from openclaw.core.logger import get_logger

        logger = get_logger("test")
        self.assertIsNotNone(logger)

    def test_structured_logging(self):
        """Test structured logging with correlation ID"""
        from openclaw.core.logger import Logger, StructuredFormatter

        # Test correlation ID
        StructuredFormatter.set_correlation_id("test-correlation-123")

        logger = Logger.get_instance().get_logger("structured_test")
        logger.info("Test message")

        # Clean up
        StructuredFormatter.clear_correlation_id()

    def test_log_file_creation(self):
        """Test that log files are created"""
        from openclaw.core.logger import Logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger.get_instance()
            logger.configure(
                log_dir=tmpdir,
                console_level=20,
                file_level=10
            )

            test_logger = logger.get_logger("file_test")
            test_logger.info("Test log message")

            # Check log file exists
            log_file = os.path.join(tmpdir, "openclaw.log")
            self.assertTrue(os.path.exists(log_file))


class TestAutomationIntegration(unittest.TestCase):
    """Integration tests for automation backend"""

    def test_platform_detection(self):
        """Test platform detection"""
        from openclaw.core.automation import get_platform, Platform

        platform = get_platform()
        self.assertIsNotNone(platform)
        self.assertIsInstance(platform, Platform)

    def test_automation_backend_factory(self):
        """Test automation backend factory"""
        from openclaw.core.automation import create_automation_backend

        backend = create_automation_backend()
        self.assertIsNotNone(backend)

    def test_screen_size(self):
        """Test getting screen size"""
        from openclaw.core.automation import get_screen_size

        size = get_screen_size()
        # On headless systems this might return None
        if size:
            self.assertGreater(size[0], 0)
            self.assertGreater(size[1], 0)


class TestVisionIntegration(unittest.TestCase):
    """Integration tests for vision module"""

    def test_screen_capture_creation(self):
        """Test screen capture initialization"""
        from openclaw.core.vision import ScreenCapture

        # Just test that the class loads
        self.assertTrue(hasattr(ScreenCapture, 'capture_full'))
        self.assertTrue(hasattr(ScreenCapture, 'capture_region'))

    def test_ocr_engine(self):
        """Test OCR engine"""
        from openclaw.core.vision import OCREngine

        engine = OCREngine()
        self.assertIsNotNone(engine)

    def test_fuzzy_matcher(self):
        """Test fuzzy matcher"""
        from openclaw.core.vision import FuzzyMatcher

        matcher = FuzzyMatcher()
        self.assertIsNotNone(matcher)

        # Test matching
        result = matcher.match("Hello World", "hello world")
        self.assertIsNotNone(result)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration"""

    def test_config_manager(self):
        """Test config manager"""
        from openclaw.core.config import ConfigManager

        manager = ConfigManager()
        self.assertIsNotNone(manager)

    def test_vision_config_defaults(self):
        """Test default vision config"""
        from openclaw.core.config import VisionConfig, VisionMode

        config = VisionConfig()
        self.assertEqual(config.mode, VisionMode.OCR)
        self.assertEqual(config.action, "alt+o")


class TestSchedulerIntegration(unittest.TestCase):
    """Integration tests for scheduler"""

    def test_scheduler_creation(self):
        """Test scheduler creation"""
        from openclaw.core.scheduler import Scheduler

        scheduler = Scheduler()
        self.assertIsNotNone(scheduler)

    def test_add_job(self):
        """Test adding a job to scheduler"""
        from openclaw.core.scheduler import Scheduler, ScheduleType

        scheduler = Scheduler()

        def test_job():
            pass

        # Add a job with interval
        job_id = scheduler.add_job(
            name="test_job",
            schedule_type=ScheduleType.INTERVAL,
            callback=test_job,
            interval_seconds=60.0
        )

        # Verify job was added
        job = scheduler.get_job("test_job")
        self.assertIsNotNone(job)
        self.assertEqual(job.name, "test_job")


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for workflow"""

    def test_workflow_manager(self):
        """Test workflow manager"""
        from openclaw.core.workflow import WorkflowManager

        manager = WorkflowManager()
        self.assertIsNotNone(manager)


class TestCircuitBreakerIntegration(unittest.TestCase):
    """Integration tests for circuit breaker"""

    def test_circuit_breaker_closed(self):
        """Test circuit breaker stays closed on success"""
        from openclaw.utils.errors import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        # Successful call should keep circuit closed
        result = cb.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(cb._state, "closed")

    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold"""
        from openclaw.utils.errors import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Fail twice to reach threshold
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("test")))
            except Exception:
                pass

        self.assertEqual(cb._state, "open")

    def test_circuit_breaker_call_when_open(self):
        """Test circuit breaker blocks calls when open"""
        from openclaw.utils.errors import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Fail once to open circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test")))
        except Exception:
            pass

        # Next call should raise
        with self.assertRaises(Exception) as context:
            cb.call(lambda: "success")

        self.assertIn("Circuit breaker is open", str(context.exception))

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset"""
        from openclaw.utils.errors import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Fail once to open circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test")))
        except Exception:
            pass

        # Reset
        cb.reset()

        # Should work now
        result = cb.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(cb._state, "closed")


if __name__ == "__main__":
    unittest.main()
