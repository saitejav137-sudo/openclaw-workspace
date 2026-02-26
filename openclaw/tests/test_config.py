"""Unit tests for core config module"""

import unittest
import tempfile
import os
import json
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.config import (
    VisionConfig,
    VisionMode,
    ConfigManager,
    ConfigValidationError,
    VISION_CONFIG_SCHEMA
)


class TestVisionConfig(unittest.TestCase):
    """Test VisionConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = VisionConfig()
        self.assertEqual(config.mode, VisionMode.OCR)
        self.assertEqual(config.action, "alt+o")
        self.assertEqual(config.poll_interval, 0.5)

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            "mode": "template",
            "action": "ctrl+s",
            "poll_interval": 1.0,
            "template_path": "/path/to/template.png"
        }
        config = VisionConfig.from_dict(data)
        self.assertEqual(config.mode, VisionMode.TEMPLATE)
        self.assertEqual(config.action, "ctrl+s")
        self.assertEqual(config.poll_interval, 1.0)

    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = VisionConfig(
            mode=VisionMode.YOLO,
            action="ctrl+c",
            poll_interval=2.0
        )
        data = config.to_dict()
        self.assertEqual(data["mode"], "yolo")
        self.assertEqual(data["action"], "ctrl+c")
        self.assertEqual(data["poll_interval"], 2.0)

    def test_region_conversion(self):
        """Test region tuple/list conversion"""
        data = {"region": [100, 200, 300, 400]}
        config = VisionConfig.from_dict(data)
        self.assertEqual(config.region, (100, 200, 300, 400))

        # Test reverse
        data = config.to_dict()
        self.assertEqual(data["region"], [100, 200, 300, 400])


class TestVisionMode(unittest.TestCase):
    """Test VisionMode enum"""

    def test_all_modes(self):
        """Test all vision modes"""
        expected_modes = [
            "ocr", "monitor", "template", "color",
            "analyze", "multi", "yolo", "fuzzy", "regression"
        ]
        actual_modes = [m.value for m in VisionMode]

        for expected in expected_modes:
            self.assertIn(expected, actual_modes)


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config"""
        config = VisionConfig(
            mode=VisionMode.OCR,
            action="alt+o",
            target_text="test"
        )

        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager.save_config(config_path, config)

        # Load it back
        loaded = self.config_manager.load_config(config_path, validate=False)
        self.assertEqual(loaded.mode, VisionMode.OCR)
        self.assertEqual(loaded.action, "alt+o")
        self.assertEqual(loaded.target_text, "test")

    def test_check_reload(self):
        """Test config reload detection"""
        config_path = os.path.join(self.temp_dir, "reload_test.yaml")

        config = VisionConfig(mode=VisionMode.OCR)
        self.config_manager.save_config(config_path, config)

        # Load once
        self.config_manager.load_config(config_path, validate=False)
        self.assertFalse(self.config_manager.check_reload())

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation"""

    def test_invalid_mode(self):
        """Test invalid mode validation"""
        data = {"mode": "invalid_mode"}
        with self.assertRaises((ConfigValidationError, ValueError)):
            VisionConfig.from_dict(data, validate=True)

    def test_invalid_region(self):
        """Test invalid region validation"""
        data = {"region": [1, 2, 3]}  # Need 4 values
        with self.assertRaises(ConfigValidationError):
            VisionConfig.from_dict(data, validate=True)

    def test_valid_config(self):
        """Test valid config passes validation"""
        data = {
            "mode": "ocr",
            "action": "alt+o",
            "poll_interval": 0.5
        }
        config = VisionConfig.from_dict(data)
        self.assertEqual(config.mode, VisionMode.OCR)


if __name__ == "__main__":
    unittest.main()
