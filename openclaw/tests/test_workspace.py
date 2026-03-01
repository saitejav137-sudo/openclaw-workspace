"""Unit tests for workspace module"""

import unittest
import os
import tempfile
import yaml
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.workspace import (
    WorkspaceConfig,
    WorkspaceManager,
    EnvironmentType,
)


class TestWorkspaceConfig(unittest.TestCase):
    """Test WorkspaceConfig dataclass"""

    def test_default_config(self):
        """Test default workspace config"""
        config = WorkspaceConfig(name="test", environment=EnvironmentType.DEVELOPMENT)
        self.assertEqual(config.name, "test")
        self.assertEqual(config.environment, EnvironmentType.DEVELOPMENT)
        self.assertEqual(config.log_level, "INFO")
        self.assertFalse(config.debug_mode)

    def test_to_dict(self):
        """Test conversion to dict"""
        config = WorkspaceConfig(
            name="prod",
            environment=EnvironmentType.PRODUCTION,
            log_level="WARNING",
            debug_mode=False
        )
        data = config.to_dict()
        self.assertEqual(data["name"], "prod")
        self.assertEqual(data["environment"], "production")
        self.assertEqual(data["log_level"], "WARNING")


class TestWorkspaceManager(unittest.TestCase):
    """Test WorkspaceManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = WorkspaceManager(workspace_dir=self.temp_dir)

    def test_add_workspace(self):
        """Test adding a workspace"""
        config = WorkspaceConfig(
            name="test",
            environment=EnvironmentType.DEVELOPMENT
        )
        self.manager.save_workspace(config)
        self.assertIn("test", self.manager.workspaces)

    def test_set_workspace(self):
        """Test setting active workspace"""
        config = WorkspaceConfig(
            name="test",
            environment=EnvironmentType.DEVELOPMENT
        )
        self.manager.save_workspace(config)
        result = self.manager.set_workspace("test")
        self.assertTrue(result)

    def test_apply_workspace_settings(self):
        """Test workspace settings are returned as dict"""
        config = WorkspaceConfig(
            name="test",
            environment=EnvironmentType.DEVELOPMENT,
            log_level="DEBUG",
            api_port=9000
        )
        settings = self.manager._apply_workspace_settings(config)
        self.assertIsInstance(settings, dict)
        self.assertEqual(settings["OPENCLAW_LOG_LEVEL"], "DEBUG")
        self.assertEqual(settings["OPENCLAW_API_PORT"], "9000")

    def test_get_workspace_settings(self):
        """Test getting current workspace settings"""
        config = WorkspaceConfig(
            name="test",
            environment=EnvironmentType.DEVELOPMENT,
            log_level="INFO"
        )
        self.manager.save_workspace(config)
        self.manager.set_workspace("test")
        settings = self.manager.get_workspace_settings()
        self.assertEqual(settings["OPENCLAW_LOG_LEVEL"], "INFO")

    def test_create_workspace_from_base(self):
        """Test creating workspace from base"""
        base_config = WorkspaceConfig(
            name="base",
            environment=EnvironmentType.DEVELOPMENT,
            api_port=8080
        )
        self.manager.save_workspace(base_config)

        new_config = self.manager.create_workspace("derived", "base", api_port=9000)
        self.assertEqual(new_config.name, "derived")
        self.assertEqual(new_config.api_port, 9000)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
