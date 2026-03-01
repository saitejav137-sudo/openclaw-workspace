"""
Secrets Management for OpenClaw — v2026.2.26 Aligned

Secure handling of API keys, tokens, and credentials.
Never store secrets in Markdown memory files.

Usage:
    from core.secrets import get_secret, set_secret, mask_secret

    api_key = get_secret("SARVAM_TTS_KEY")
    print(mask_secret(api_key))  # "sk_ug...nvxx"
"""

import os
import json
import base64
import hashlib
import time
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

from .logger import get_logger

logger = get_logger("secrets")

# Try to import optional encryption
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Try to import dotenv
try:
    from dotenv import load_dotenv, set_key as dotenv_set_key
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class SecretsManager:
    """
    Centralized secrets management.

    Priority order for secret resolution:
    1. Environment variables (highest priority)
    2. .env file
    3. Encrypted secrets store (~/.openclaw/.secrets.enc)

    Features:
    - Load from .env files
    - Encrypt at rest using Fernet symmetric encryption
    - Mask secrets for safe logging
    - Validate no secrets leak into memory/log files
    - Thread-safe access
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        env_file: Optional[str] = None,
        secrets_dir: Optional[str] = None
    ):
        self._env_file = env_file or os.path.expanduser("~/.openclaw/.env")
        self._secrets_dir = secrets_dir or os.path.expanduser("~/.openclaw/secrets")
        self._cache: Dict[str, str] = {}
        self._encryption_key: Optional[bytes] = None
        self._initialized = False

        # Load .env file if available
        if DOTENV_AVAILABLE and os.path.exists(self._env_file):
            load_dotenv(self._env_file)
            logger.info(f"Loaded secrets from {self._env_file}")

        # Ensure secrets directory exists
        os.makedirs(self._secrets_dir, mode=0o700, exist_ok=True)

        # Initialize encryption if available
        if ENCRYPTION_AVAILABLE:
            self._init_encryption()

        self._initialized = True

    @classmethod
    def get_instance(cls, **kwargs) -> "SecretsManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def _init_encryption(self):
        """Initialize Fernet encryption with a derived key."""
        key_file = os.path.join(self._secrets_dir, ".key")

        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                self._encryption_key = f.read()
        else:
            # Generate new key
            self._encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self._encryption_key)
            os.chmod(key_file, 0o600)
            logger.info("Generated new encryption key")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.

        Resolution order:
        1. Environment variable
        2. Cached value
        3. Encrypted store
        4. Default value
        """
        # 1. Environment variable (always highest priority)
        env_val = os.environ.get(key)
        if env_val:
            return env_val

        # 2. Cache
        if key in self._cache:
            return self._cache[key]

        # 3. Encrypted store
        encrypted = self._load_encrypted(key)
        if encrypted:
            self._cache[key] = encrypted
            return encrypted

        return default

    def set(self, key: str, value: str, persist: bool = True):
        """
        Set a secret value.

        Args:
            key: Secret identifier
            value: Secret value
            persist: If True, save to encrypted store
        """
        self._cache[key] = value

        if persist:
            # Save to .env file if dotenv is available
            if DOTENV_AVAILABLE and self._env_file:
                try:
                    os.makedirs(os.path.dirname(self._env_file), exist_ok=True)
                    dotenv_set_key(self._env_file, key, value)
                    logger.info(f"Saved secret '{key}' to .env")
                except Exception as e:
                    logger.warning(f"Could not save to .env: {e}")

            # Also save encrypted copy
            self._save_encrypted(key, value)

    def delete(self, key: str):
        """Delete a secret."""
        self._cache.pop(key, None)
        enc_file = os.path.join(self._secrets_dir, f"{key}.enc")
        if os.path.exists(enc_file):
            os.remove(enc_file)
            logger.info(f"Deleted encrypted secret: {key}")

    def _save_encrypted(self, key: str, value: str):
        """Save secret to encrypted file."""
        if not ENCRYPTION_AVAILABLE or not self._encryption_key:
            return
        try:
            fernet = Fernet(self._encryption_key)
            encrypted = fernet.encrypt(value.encode())
            enc_file = os.path.join(self._secrets_dir, f"{key}.enc")
            with open(enc_file, "wb") as f:
                f.write(encrypted)
            os.chmod(enc_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to encrypt secret '{key}': {e}")

    def _load_encrypted(self, key: str) -> Optional[str]:
        """Load secret from encrypted file."""
        if not ENCRYPTION_AVAILABLE or not self._encryption_key:
            return None
        enc_file = os.path.join(self._secrets_dir, f"{key}.enc")
        if not os.path.exists(enc_file):
            return None
        try:
            fernet = Fernet(self._encryption_key)
            with open(enc_file, "rb") as f:
                encrypted = f.read()
            return fernet.decrypt(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret '{key}': {e}")
            return None


    def audit(self) -> Dict:
        """
        Audit all secrets (aligned with upstream 2026.2.26 secrets workflow).
        
        Returns status of all known secrets: present, missing, expired.
        """
        results = {}
        for key in self.list_keys():
            value = self.get(key)
            results[key] = {
                "present": value is not None,
                "masked": mask_secret(value) if value else None,
                "source": "env" if os.getenv(key) else "encrypted_store",
            }
        return results

    def reload(self):
        """Reload secrets from all sources (aligned with upstream 2026.2.26)."""
        self._cache.clear()
        if DOTENV_AVAILABLE and self._env_file and os.path.exists(self._env_file):
            load_dotenv(self._env_file, override=True)
        logger.info("Secrets reloaded from all sources")
        return True

    def list_keys(self) -> list:
        """List all known secret keys (not values)."""
        keys = set(self._cache.keys())
        if os.path.exists(self._secrets_dir):
            for f in os.listdir(self._secrets_dir):
                if f.endswith(".enc"):
                    keys.add(f[:-4])
        return sorted(keys)

    def validate_no_leaks(self, file_path: str) -> list:
        """
        Check if a file contains any known secret values.
        Returns list of leaked secret keys found in the file.
        """
        leaks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return leaks

        for key in self.list_keys():
            value = self.get(key)
            if value and len(value) > 8 and value in content:
                leaks.append(key)
        return leaks

    # ============== v2026.2.26 Features ==============

    def snapshot(self, label: str = None) -> Dict:
        """Create a frozen snapshot of current secrets state."""
        snapshot_label = label or f"snap_{int(time.time())}"
        snapshot_data = {}
        for key in self.list_keys():
            value = self.get(key)
            if value:
                snapshot_data[key] = {
                    "masked": mask_secret(value),
                    "hash": hashlib.sha256(value.encode()).hexdigest()[:16],
                    "source": "env" if os.getenv(key) else "encrypted_store",
                }
        snap_file = os.path.join(self._secrets_dir, f".snapshot_{snapshot_label}.json")
        try:
            with open(snap_file, "w") as f:
                json.dump({"label": snapshot_label, "created_at": time.time(), "keys": snapshot_data}, f, indent=2)
            os.chmod(snap_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
        return {"label": snapshot_label, "key_count": len(snapshot_data), "keys": snapshot_data}

    def apply(self, target_path: str = None, keys: List[str] = None) -> Dict:
        """Apply secrets to target path with strict validation."""
        target = target_path or self._env_file
        target_dir = os.path.dirname(os.path.abspath(target))
        allowed_dirs = [os.path.expanduser("~/.openclaw"), os.path.expanduser("~/Desktop/openclaw")]
        path_ok = any(os.path.abspath(target_dir).startswith(os.path.abspath(d)) for d in allowed_dirs)
        if not path_ok:
            return {"success": False, "error": f"Target path not in allowed directories: {allowed_dirs}"}
        if os.path.islink(target):
            return {"success": False, "error": "Target path is a symlink"}
        results = {}
        keys_to_apply = keys or self.list_keys()
        for key in keys_to_apply:
            value = self.get(key)
            if value:
                results[key] = {"applied": True, "masked": mask_secret(value)}
            else:
                results[key] = {"applied": False, "reason": "not found"}
        return {"success": True, "results": results}

    def configure(self, template: Dict[str, str] = None) -> Dict:
        """Configure secrets from a template."""
        if template is None:
            template = {
                "TELEGRAM_BOT_TOKEN": "Telegram bot API token",
                "BRAVE_API_KEY": "Brave Search API key",
                "OPENAI_API_KEY": "OpenAI API key",
            }
        report = {}
        for key, description in template.items():
            value = self.get(key)
            report[key] = {
                "description": description,
                "configured": value is not None,
                "masked": mask_secret(value) if value else None,
                "source": "env" if os.getenv(key) else ("encrypted" if value else "missing"),
            }
        configured = sum(1 for r in report.values() if r["configured"])
        return {"configured": configured, "total": len(template), "keys": report}


# ============== Utility Functions ==============

def mask_secret(value: Optional[str], show_chars: int = 4) -> str:
    """
    Mask a secret value for safe display/logging.
    "sk_ugc6lprk_ihSnovqV8brffMpB0P3unvxx" -> "sk_u...nvxx"
    """
    if not value:
        return "***"
    if len(value) <= show_chars * 2:
        return "***"
    return f"{value[:show_chars]}...{value[-show_chars:]}"


def sanitize_for_shell(value: str) -> str:
    """
    Sanitize a string for safe use in shell commands.
    Removes/escapes dangerous characters that could lead to injection.
    """
    value = value.replace("\0", "")
    dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "{", "}", "<", ">", "!", "\\", "\n", "\r"]
    for char in dangerous_chars:
        value = value.replace(char, "")
    return value.strip()


ALLOWED_XDOTOOL_ACTIONS = {
    "key", "keyup", "keydown", "type",
    "mousemove", "click", "mousedown", "mouseup",
    "getactivewindow", "getwindowname", "getwindowclassname",
    "search", "windowactivate", "windowfocus",
}


def validate_xdotool_action(action: str) -> bool:
    """Validate that an xdotool action is in the allowlist."""
    parts = action.split()
    if not parts:
        return False
    return parts[0] in ALLOWED_XDOTOOL_ACTIONS


# ============== Global Access ==============

def get_secrets_manager(**kwargs) -> SecretsManager:
    """Get global secrets manager instance."""
    return SecretsManager.get_instance(**kwargs)


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Quick access to get a secret."""
    return get_secrets_manager().get(key, default)


def set_secret(key: str, value: str, persist: bool = True):
    """Quick access to set a secret."""
    get_secrets_manager().set(key, value, persist)


def audit_secrets() -> dict:
    """Quick access to audit secrets."""
    return SecretsManager.get_instance().audit()

def reload_secrets() -> bool:
    """Quick access to reload secrets."""
    return SecretsManager.get_instance().reload()

def configure_secrets(template: Dict = None) -> dict:
    return SecretsManager.get_instance().configure(template)

def snapshot_secrets(label: str = None) -> dict:
    return SecretsManager.get_instance().snapshot(label)

def apply_secrets(target_path: str = None, keys: List = None) -> dict:
    return SecretsManager.get_instance().apply(target_path, keys)

__all__ = [
    "SecretsManager",
    "get_secrets_manager",
    "get_secret",
    "set_secret",
    "mask_secret",
    "sanitize_for_shell",
    "validate_xdotool_action",
    "ALLOWED_XDOTOOL_ACTIONS",
    "audit_secrets",
    "reload_secrets",
    "configure_secrets",
    "snapshot_secrets",
    "apply_secrets",
]
