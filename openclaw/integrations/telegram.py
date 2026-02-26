"""Telegram bot integration"""

import os
import time
import json
import threading
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass

import requests

from ..core.logger import get_logger
from ..core.actions import RetryConfig, ActionExecutor

logger = get_logger("telegram")


@dataclass
class TelegramCommand:
    """Telegram command definition"""
    name: str
    description: str
    handler: Callable


class TelegramBot:
    """Telegram bot for remote control and notifications"""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        retry_config: RetryConfig = None
    ):
        self.token = token or self._load_token()
        self.chat_id = chat_id or self._load_chat_id()
        self.api_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.enabled = bool(self.token and self.chat_id)
        self.retry_config = retry_config or RetryConfig()
        self.executor = ActionExecutor(self.retry_config)
        self._commands: Dict[str, TelegramCommand] = {}
        self._command_callbacks: Dict[str, Callable] = {}

        if self.enabled:
            self._register_default_commands()

    def _load_token(self) -> Optional[str]:
        """Load token from environment or config"""
        import os
        token = os.getenv("TELEGRAM_BOT_TOKEN")

        if not token:
            config_path = os.path.expanduser("~/.openclaw/openclaw.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    token = config.get("channels", {}).get("telegram", {}).get("botToken")
                except Exception as e:
                    logger.warning(f"Failed to load Telegram config: {e}")

        return token

    def _load_chat_id(self) -> Optional[str]:
        """Load chat ID from environment or config"""
        import os
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not chat_id:
            config_path = os.path.expanduser("~/.openclaw/openclaw.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    allow_from = config.get("channels", {}).get("telegram", {}).get("allowFrom", [])
                    if allow_from:
                        chat_id = str(allow_from[0])
                except Exception as e:
                    logger.warning(f"Failed to load Telegram config: {e}")

        return chat_id

    def _register_default_commands(self):
        """Register default bot commands"""
        self.register_command("help", "Show available commands", self._handle_help)
        self.register_command("status", "Get system status", self._handle_status)
        self.register_command("trigger", "Trigger manual check", self._handle_trigger)

    def register_command(self, name: str, description: str, handler: Callable):
        """Register a bot command"""
        self._commands[name] = TelegramCommand(name, description, handler)
        logger.debug(f"Registered command: /{name}")

    def _handle_help(self, args: List[str]) -> str:
        """Handle /help command"""
        lines = ["Available commands:"]
        for cmd in self._commands.values():
            lines.append(f"/{cmd.name} - {cmd.description}")
        return "\n".join(lines)

    def _handle_status(self, args: List[str]) -> str:
        """Handle /status command"""
        return "System is running"

    def _handle_trigger(self, args: List[str]) -> str:
        """Handle /trigger command"""
        return "Trigger executed"

    def _make_request(self, method: str, data: Dict = None, files: Dict = None) -> Optional[Dict]:
        """Make API request with retry"""
        url = f"{self.api_url}/{method}"

        def _request():
            if files:
                return requests.post(url, data=data, files=files, timeout=30)
            else:
                return requests.post(url, json=data, timeout=10)

        try:
            response = self.executor.execute_with_retry(_request)
            if response and response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Telegram API error: {e}")

        return None

    def send_message(self, text: str, parse_mode: str = None) -> bool:
        """Send message to configured chat"""
        if not self.enabled:
            return False

        data = {
            "chat_id": self.chat_id,
            "text": text
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        result = self._make_request("sendMessage", data)
        if result:
            logger.debug(f"Message sent: {text[:50]}...")
            return True

        return False

    def send_photo(self, image_path: str, caption: str = None) -> bool:
        """Send photo to configured chat"""
        if not self.enabled:
            return False

        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption

        files = {"photo": open(image_path, "rb")}

        result = self._make_request("sendPhoto", data, files)
        files["photo"].close()

        if result:
            logger.debug(f"Photo sent: {image_path}")
            return True

        return False

    def send_document(self, file_path: str, caption: str = None) -> bool:
        """Send document to configured chat"""
        if not self.enabled:
            return False

        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption

        files = {"document": open(file_path, "rb")}

        result = self._make_request("sendDocument", data, files)
        files["document"].close()

        return bool(result)

    def get_updates(self, offset: int = 0, timeout: int = 30) -> List[Dict]:
        """Get updates from bot"""
        if not self.enabled:
            return []

        try:
            url = f"{self.api_url}/getUpdates"
            params = {"timeout": timeout, "offset": offset}
            response = requests.get(url, params=params, timeout=timeout + 5)

            if response.status_code == 200:
                return response.json().get("result", [])

        except Exception as e:
            logger.error(f"Get updates error: {e}")

        return []

    def handle_commands(self, text: str) -> str:
        """Handle text as commands"""
        text = text.strip()

        # Handle !! prefix
        if text.startswith("!!"):
            cmd = text[2:].strip().lower().split()
        else:
            cmd = text.strip().lower().split()

        if not cmd:
            return "Use /help for commands"

        command = cmd[0].replace("/", "")
        args = cmd[1:] if len(cmd) > 1 else []

        if command in self._commands:
            handler = self._commands[command].handler
            try:
                return handler(args)
            except Exception as e:
                logger.error(f"Command error: {e}")
                return f"Error: {str(e)}"

        return f"Unknown command: {command}. Use /help for available commands."

    def start_command_listener(self, callback: Callable = None):
        """Start polling for commands"""
        if not self.enabled:
            logger.warning("Telegram bot not enabled")
            return

        def listener():
            offset = 0
            logger.info("Telegram command listener started")

            while True:
                try:
                    updates = self.get_updates(offset)

                    for update in updates:
                        try:
                            offset = update.get("update_id", offset) + 1
                            message = update.get("message", {})
                            text = message.get("text", "")

                            if text:
                                # Validate chat_id
                                chat_id = str(message.get("chat", {}).get("id", ""))
                                if chat_id != self.chat_id:
                                    logger.warning(f"Unauthorized chat: {chat_id}")
                                    continue

                                logger.debug(f"Command received: {text}")
                                response = self.handle_commands(text)

                                if response:
                                    self.send_message(response)

                                if callback:
                                    callback(text, response)

                        except Exception as e:
                            logger.error(f"Update handling error: {e}")

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Listener error: {e}")
                    time.sleep(5)

        thread = threading.Thread(target=listener, daemon=True)
        thread.start()


# Export classes
__all__ = [
    "TelegramBot",
    "TelegramCommand",
]
