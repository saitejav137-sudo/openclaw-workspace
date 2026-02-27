"""Config encryption module for securing sensitive data"""

import os
import base64
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from openclaw.core.logger import get_logger

logger = get_logger("security")


@dataclass
class EncryptedConfig:
    """Encrypted configuration wrapper"""
    encrypted_data: str
    salt: str
    version: int = 1


class ConfigEncryption:
    """Encrypt and decrypt sensitive configuration data"""

    def __init__(self, password: Optional[str] = None):
        self.fernet = None
        self.salt = None

        if CRYPTO_AVAILABLE:
            # Use password or get from environment - require non-empty value
            password = password or os.getenv("OPENCLAW_ENCRYPTION_KEY")

            if password:
                self._init_fernet(password)
            else:
                # Try to load existing key - warn if neither available
                logger.warning("No encryption key provided. Set OPENCLAW_ENCRYPTION_KEY or generate a key.")
                self._load_key()

    def _init_fernet(self, password: str):
        """Initialize Fernet with password"""
        # Generate salt or load existing
        key_dir = os.path.expanduser("~/.openclaw")
        salt_file = os.path.join(key_dir, ".key_salt")

        if os.path.exists(salt_file):
            with open(salt_file, "rb") as f:
                self.salt = f.read()
        else:
            self.salt = os.urandom(16)
            os.makedirs(key_dir, exist_ok=True)
            with open(salt_file, "wb") as f:
                f.write(self.salt)

        # Derive key from password
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.fernet = Fernet(key)

    def _load_key(self):
        """Load existing encryption key"""
        key_file = os.path.expanduser("~/.openclaw/.key")

        if os.path.exists(key_file):
            try:
                with open(key_file, "rb") as f:
                    key = f.read()
                self.fernet = Fernet(key)
                logger.info("Encryption key loaded")
            except Exception as e:
                logger.warning(f"Failed to load encryption key: {e}")

    @classmethod
    def generate_key(cls) -> str:
        """Generate a new encryption key"""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")

        key = Fernet.generate_key()
        key_file = os.path.expanduser("~/.openclaw/.key")
        os.makedirs(os.path.dirname(key_file), exist_ok=True)

        with open(key_file, "wb") as f:
            f.write(key)

        logger.info(f"New encryption key saved to {key_file}")
        return key.decode()

    def encrypt(self, data: Dict[str, Any]) -> EncryptedConfig:
        """Encrypt configuration data"""
        if not self.fernet:
            logger.warning("Encryption not initialized, returning plaintext")
            return EncryptedConfig(
                encrypted_data=json.dumps(data),
                salt=base64.b64encode(self.salt).decode() if self.salt else "",
                version=1
            )

        # Encrypt the JSON data
        json_data = json.dumps(data)
        encrypted = self.fernet.encrypt(json_data.encode())

        return EncryptedConfig(
            encrypted_data=base64.b64encode(encrypted).decode(),
            salt=base64.b64encode(self.salt).decode() if self.salt else "",
            version=1
        )

    def decrypt(self, encrypted_config: EncryptedConfig) -> Dict[str, Any]:
        """Decrypt configuration data"""
        if not self.fernet:
            # Try to initialize from config salt
            if encrypted_config.salt:
                self.salt = base64.b64decode(encrypted_config.salt.encode())
                # Would need password here - this is simplified
                logger.warning("Cannot decrypt without password")

            return json.loads(encrypted_config.encrypted_data)

        try:
            encrypted = base64.b64decode(encrypted_config.encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except (base64.binascii.Error, ValueError, KeyError) as e:
            logger.error(f"Decryption failed: invalid encoded data: {e}")
            raise ValueError("Invalid encrypted data format") from e
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a config file"""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")

        with open(input_path, "r") as f:
            data = json.load(f)

        encrypted = self.encrypt(data)

        if output_path is None:
            output_path = input_path + ".enc"

        with open(output_path, "w") as f:
            json.dump({
                "encrypted_data": encrypted.encrypted_data,
                "salt": encrypted.salt,
                "version": encrypted.version
            }, f)

        logger.info(f"Config encrypted: {output_path}")
        return output_path

    def decrypt_file(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """Decrypt a config file"""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")

        with open(input_path, "r") as f:
            data = json.load(f)

        encrypted_config = EncryptedConfig(
            encrypted_data=data["encrypted_data"],
            salt=data.get("salt", ""),
            version=data.get("version", 1)
        )

        decrypted = self.decrypt(encrypted_config)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(decrypted, f, indent=2)
            logger.info(f"Config decrypted: {output_path}")

        return decrypted


def encrypt_sensitive_fields(config: Dict, fields: list = None) -> Dict:
    """Helper to encrypt specific sensitive fields"""
    if fields is None:
        fields = ["api_key", "telegram_token", "telegram_chat_id", "webhook_url"]

    encrypter = ConfigEncryption()
    result = config.copy()

    encrypted_data = {}
    for field in fields:
        if field in result and result[field]:
            encrypted_data[field] = result[field]
            result[field] = f"ENCRYPTED:{field}"

    if encrypted_data:
        encrypted = encrypter.encrypt(encrypted_data)
        result["_encrypted"] = {
            "data": encrypted.encrypted_data,
            "salt": encrypted.salt
        }

    return result


def decrypt_sensitive_fields(config: Dict) -> Dict:
    """Helper to decrypt specific sensitive fields"""
    if "_encrypted" not in config:
        return config

    encrypter = ConfigEncryption()
    encrypted = EncryptedConfig(
        encrypted_data=config["_encrypted"]["data"],
        salt=config["_encrypted"]["salt"]
    )

    decrypted = encrypter.decrypt(encrypted)
    result = config.copy()

    for field, value in decrypted.items():
        result[field] = value

    del result["_encrypted"]
    return result


# Export
__all__ = [
    "ConfigEncryption",
    "EncryptedConfig",
    "encrypt_sensitive_fields",
    "decrypt_sensitive_fields",
]
