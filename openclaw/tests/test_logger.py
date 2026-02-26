"""Unit tests for logger module"""

import unittest
import sys
import os
import logging
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.logger import get_logger, setup_logging


class TestGetLogger(unittest.TestCase):
    """Test get_logger function"""

    def test_get_logger_returns_logger(self):
        logger = get_logger("test")
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_name(self):
        logger = get_logger("test_name")
        # Logger prepends 'openclaw.' prefix
        self.assertIn("test_name", logger.name)


class TestSetupLogging(unittest.TestCase):
    """Test setup_logging function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_logging_default(self):
        # Should not raise
        setup_logging()
        self.assertTrue(True)

    def test_setup_logging_with_dir(self):
        # Should not raise
        setup_logging(log_dir=self.temp_dir)
        self.assertTrue(True)


class TestLoggingLevels(unittest.TestCase):
    """Test different logging levels"""

    def test_debug_level(self):
        logger = get_logger("test_debug")
        logger.debug("Debug message")
        self.assertTrue(True)

    def test_info_level(self):
        logger = get_logger("test_info")
        logger.info("Info message")
        self.assertTrue(True)

    def test_warning_level(self):
        logger = get_logger("test_warning")
        logger.warning("Warning message")
        self.assertTrue(True)

    def test_error_level(self):
        logger = get_logger("test_error")
        logger.error("Error message")
        self.assertTrue(True)

    def test_critical_level(self):
        logger = get_logger("test_critical")
        logger.critical("Critical message")
        self.assertTrue(True)


class TestLoggingException(unittest.TestCase):
    """Test exception logging"""

    def test_log_exception(self):
        logger = get_logger("test_exception")
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("Caught exception")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
