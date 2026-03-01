"""
Tests for SecretsManager
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock


class TestMaskSecret:
    """Tests for mask_secret utility."""

    def test_mask_normal_key(self):
        from openclaw.core.secrets import mask_secret
        assert mask_secret("sk_ugc6lprk_ihSnovqV8brffMpB0P3unvxx") == "sk_u...nvxx"

    def test_mask_short_key(self):
        from openclaw.core.secrets import mask_secret
        assert mask_secret("short") == "***"

    def test_mask_none(self):
        from openclaw.core.secrets import mask_secret
        assert mask_secret(None) == "***"

    def test_mask_empty(self):
        from openclaw.core.secrets import mask_secret
        assert mask_secret("") == "***"

    def test_mask_custom_chars(self):
        from openclaw.core.secrets import mask_secret
        result = mask_secret("abcdefghijklmnop", show_chars=6)
        assert result == "abcdef...klmnop"


class TestSanitizeForShell:
    """Tests for shell input sanitization."""

    def test_removes_semicolons(self):
        from openclaw.core.secrets import sanitize_for_shell
        assert ";" not in sanitize_for_shell("echo hello; rm -rf /")

    def test_removes_pipes(self):
        from openclaw.core.secrets import sanitize_for_shell
        assert "|" not in sanitize_for_shell("ls | grep secret")

    def test_removes_backticks(self):
        from openclaw.core.secrets import sanitize_for_shell
        assert "`" not in sanitize_for_shell("`whoami`")

    def test_removes_dollar_signs(self):
        from openclaw.core.secrets import sanitize_for_shell
        assert "$" not in sanitize_for_shell("echo $HOME")

    def test_removes_null_bytes(self):
        from openclaw.core.secrets import sanitize_for_shell
        assert "\0" not in sanitize_for_shell("hello\0world")

    def test_safe_input_unchanged(self):
        from openclaw.core.secrets import sanitize_for_shell
        assert sanitize_for_shell("alt+o") == "alt+o"

    def test_removes_newlines(self):
        from openclaw.core.secrets import sanitize_for_shell
        result = sanitize_for_shell("hello\nworld")
        assert "\n" not in result


class TestValidateXdotoolAction:
    """Tests for xdotool action validation."""

    def test_allowed_actions(self):
        from openclaw.core.secrets import validate_xdotool_action
        assert validate_xdotool_action("key alt+o") is True
        assert validate_xdotool_action("mousemove 100 200") is True
        assert validate_xdotool_action("click 1") is True
        assert validate_xdotool_action("type hello") is True

    def test_disallowed_actions(self):
        from openclaw.core.secrets import validate_xdotool_action
        assert validate_xdotool_action("exec rm -rf /") is False
        assert validate_xdotool_action("") is False
        assert validate_xdotool_action("eval something") is False

    def test_getwindow_allowed(self):
        from openclaw.core.secrets import validate_xdotool_action
        assert validate_xdotool_action("getactivewindow") is True
        assert validate_xdotool_action("getwindowname 12345") is True


class TestSecretsManager:
    """Tests for SecretsManager."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Create temp environment for each test."""
        from openclaw.core.secrets import SecretsManager
        SecretsManager.reset()

        self.env_file = str(tmp_path / ".env")
        self.secrets_dir = str(tmp_path / "secrets")
        os.makedirs(self.secrets_dir, exist_ok=True)

    def _create_manager(self):
        from openclaw.core.secrets import SecretsManager
        return SecretsManager(
            env_file=self.env_file,
            secrets_dir=self.secrets_dir
        )

    def test_get_from_env_var(self):
        manager = self._create_manager()
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            assert manager.get("TEST_KEY") == "test_value"

    def test_get_default(self):
        manager = self._create_manager()
        assert manager.get("NONEXISTENT", "default") == "default"

    def test_get_none_default(self):
        manager = self._create_manager()
        assert manager.get("NONEXISTENT") is None

    def test_set_and_get_cached(self):
        manager = self._create_manager()
        manager.set("MY_KEY", "my_value", persist=False)
        assert manager.get("MY_KEY") == "my_value"

    def test_delete_secret(self):
        manager = self._create_manager()
        manager.set("DEL_KEY", "del_value", persist=False)
        manager.delete("DEL_KEY")
        assert manager.get("DEL_KEY") is None

    def test_list_keys(self):
        manager = self._create_manager()
        manager.set("KEY1", "val1", persist=False)
        manager.set("KEY2", "val2", persist=False)
        keys = manager.list_keys()
        assert "KEY1" in keys
        assert "KEY2" in keys

    def test_validate_no_leaks_clean(self, tmp_path):
        manager = self._create_manager()
        manager.set("SECRET_KEY", "super_secret_value_12345", persist=False)

        clean_file = tmp_path / "clean.md"
        clean_file.write_text("This file has no secrets.")

        leaks = manager.validate_no_leaks(str(clean_file))
        assert len(leaks) == 0

    def test_validate_no_leaks_dirty(self, tmp_path):
        manager = self._create_manager()
        manager.set("SECRET_KEY", "super_secret_value_12345", persist=False)

        dirty_file = tmp_path / "dirty.md"
        dirty_file.write_text("API key: super_secret_value_12345")

        leaks = manager.validate_no_leaks(str(dirty_file))
        assert "SECRET_KEY" in leaks

    def test_encryption_roundtrip(self):
        """Test that encrypt -> decrypt preserves value."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography not installed")

        manager = self._create_manager()
        manager.set("ENC_KEY", "encrypted_value_test", persist=True)

        # Clear cache to force reading from encrypted store
        manager._cache.clear()

        # Should load from encrypted file
        result = manager.get("ENC_KEY")
        assert result == "encrypted_value_test"

    def test_singleton_pattern(self):
        from openclaw.core.secrets import SecretsManager
        SecretsManager.reset()

        m1 = SecretsManager.get_instance(
            env_file=self.env_file,
            secrets_dir=self.secrets_dir
        )
        m2 = SecretsManager.get_instance()
        assert m1 is m2

        SecretsManager.reset()
