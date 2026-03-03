"""
InterBot Communication Bridge for OpenClaw

Enables Ajanta (Python gateway) and Ellora (TypeScript gateway) to
communicate, delegate tasks, and share context.

Communication channels:
1. Shared file bus (~/.openclaw/interbot/) — JSON message queue
2. Telegram cross-posting — send messages via the other bot's token
3. Shared memory (~/.openclaw/memory/) — already exists, both can read
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from .logger import get_logger

logger = get_logger("interbot")


class MessageType(Enum):
    TASK = "task"           # Request the other bot to do something
    RESPONSE = "response"   # Reply to a task
    INFO = "info"           # Informational broadcast (no response expected)
    QUERY = "query"         # Ask a question, expect a response


class MessageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class InterBotMessage:
    """A message between bots."""
    id: str
    from_bot: str           # "ajanta" or "ellora"
    to_bot: str             # "ajanta" or "ellora"
    msg_type: str           # MessageType value
    content: str            # The actual message/task
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # MessageStatus value
    response: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: int = 300          # Time-to-live in seconds (5 min default)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'InterBotMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


# Known bots with their identities (tokens loaded at runtime)
BOT_REGISTRY = {
    "ajanta": {
        "name": "Ajanta",
        "token": None,  # Loaded from AJANTA_BOT_TOKEN env var or openclaw.json
        "gateway": "python",
        "description": "Python gateway bot with ReAct agent, swarm, and deep research",
    },
    "ellora": {
        "name": "Ellora",
        "token": None,  # Loaded from ELLORA_BOT_TOKEN env var or openclaw.json
        "gateway": "typescript",
        "description": "TypeScript gateway bot with OpenClaw native capabilities",
    },
}


def _load_bot_tokens():
    """Load bot tokens from environment variables or openclaw.json config."""
    # Try env vars first
    ajanta_token = os.environ.get("AJANTA_BOT_TOKEN")
    ellora_token = os.environ.get("ELLORA_BOT_TOKEN")

    # Fall back to openclaw.json
    if not ajanta_token or not ellora_token:
        config_path = os.path.expanduser("~/.openclaw/openclaw.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
            telegram_cfg = config.get("channels", {}).get("telegram", {})
            if not ajanta_token:
                ajanta_token = telegram_cfg.get("botToken")
            if not ellora_token:
                ellora_token = telegram_cfg.get("elloraToken")
        except Exception as e:
            logger.debug(f"Could not load bot tokens from config: {e}")

    if ajanta_token:
        BOT_REGISTRY["ajanta"]["token"] = ajanta_token
    if ellora_token:
        BOT_REGISTRY["ellora"]["token"] = ellora_token


# Load tokens on module import
_load_bot_tokens()


class InterBotBridge:
    """
    Bridge for inter-bot communication.

    Handles:
    - File-based message queue for cross-process communication
    - Telegram cross-posting (send messages via the other bot's token)
    - Background listener for incoming messages
    - Response waiting with timeout
    """

    def __init__(
        self,
        my_bot_id: str = "ajanta",
        interbot_dir: str = "~/.openclaw/interbot",
        chat_id: str = None,
        on_message: Callable = None,
    ):
        self.my_bot_id = my_bot_id
        self.interbot_dir = os.path.expanduser(interbot_dir)
        self.chat_id = chat_id
        self.on_message = on_message

        # Create directories
        os.makedirs(os.path.join(self.interbot_dir, "inbox"), exist_ok=True)
        os.makedirs(os.path.join(self.interbot_dir, "outbox"), exist_ok=True)
        os.makedirs(os.path.join(self.interbot_dir, "archive"), exist_ok=True)

        # Pending response waiters
        self._waiters: Dict[str, threading.Event] = {}
        self._responses: Dict[str, str] = {}
        self._lock = threading.Lock()

        # Listener thread
        self._running = False
        self._listener_thread = None

        logger.info(f"InterBot bridge initialized for '{my_bot_id}'")

    # ============== Send ==============

    def send_task(self, to_bot: str, content: str, metadata: dict = None) -> str:
        """Send a task to another bot. Returns the message ID."""
        msg = InterBotMessage(
            id=self._gen_id(content),
            from_bot=self.my_bot_id,
            to_bot=to_bot,
            msg_type=MessageType.TASK.value,
            content=content,
            metadata=metadata or {},
        )
        self._write_message(msg)

        # Also cross-post to Telegram so the other bot sees it
        self._telegram_crosspost(to_bot, f"📨 Task from {self.my_bot_id.title()}:\n\n{content}")

        logger.info(f"Sent task to {to_bot}: {content[:80]}")
        return msg.id

    def send_query(self, to_bot: str, question: str, timeout: float = 120.0) -> str:
        """Send a query and wait for response. Returns the response text."""
        msg = InterBotMessage(
            id=self._gen_id(question),
            from_bot=self.my_bot_id,
            to_bot=to_bot,
            msg_type=MessageType.QUERY.value,
            content=question,
        )

        # Set up response waiter
        event = threading.Event()
        with self._lock:
            self._waiters[msg.id] = event
            self._responses[msg.id] = None

        self._write_message(msg)
        self._telegram_crosspost(to_bot, f"❓ Question from {self.my_bot_id.title()}:\n\n{question}")

        logger.info(f"Sent query to {to_bot}, waiting for response (timeout: {timeout}s)")

        # Wait for response
        event.wait(timeout=timeout)

        with self._lock:
            response = self._responses.pop(msg.id, None)
            self._waiters.pop(msg.id, None)

        if response is None:
            return f"(No response from {to_bot} within {timeout}s)"

        return response

    def send_info(self, to_bot: str, content: str):
        """Send informational broadcast (no response expected)."""
        msg = InterBotMessage(
            id=self._gen_id(content),
            from_bot=self.my_bot_id,
            to_bot=to_bot,
            msg_type=MessageType.INFO.value,
            content=content,
        )
        self._write_message(msg)
        logger.info(f"Sent info to {to_bot}: {content[:80]}")

    def send_response(self, original_msg_id: str, to_bot: str, response: str):
        """Send a response to a previously received task/query."""
        msg = InterBotMessage(
            id=self._gen_id(response),
            from_bot=self.my_bot_id,
            to_bot=to_bot,
            msg_type=MessageType.RESPONSE.value,
            content=response,
            metadata={"in_reply_to": original_msg_id},
        )
        self._write_message(msg)

        # Also update the original message status
        self._update_message_status(original_msg_id, MessageStatus.DONE, response)

        logger.info(f"Sent response to {to_bot} for message {original_msg_id[:8]}")

    # ============== Receive ==============

    def poll_inbox(self) -> List[InterBotMessage]:
        """Poll the inbox for new messages addressed to this bot."""
        inbox = os.path.join(self.interbot_dir, "inbox")
        messages = []

        for filename in sorted(os.listdir(inbox)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(inbox, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)

                msg = InterBotMessage.from_dict(data)

                # Only process messages for this bot
                if msg.to_bot != self.my_bot_id:
                    continue

                # Skip expired messages
                if msg.is_expired:
                    self._archive_message(filepath, "expired")
                    continue

                # Skip already processed or in-progress
                if msg.status in (MessageStatus.DONE.value, MessageStatus.FAILED.value, MessageStatus.PROCESSING.value):
                    self._archive_message(filepath, msg.status)
                    continue

                messages.append(msg)

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Invalid interbot message {filename}: {e}")
                continue

        return messages

    def process_incoming(self, msg: InterBotMessage):
        """Process a single incoming message."""
        # Handle response messages — wake up waiting threads
        if msg.msg_type == MessageType.RESPONSE.value:
            reply_to = msg.metadata.get("in_reply_to")
            if reply_to:
                with self._lock:
                    if reply_to in self._waiters:
                        self._responses[reply_to] = msg.content
                        self._waiters[reply_to].set()
                        logger.info(f"Response received for {reply_to[:8]}: {msg.content[:80]}")
                        return

        # Handle task/query messages — pass to callback
        if self.on_message:
            try:
                self.on_message(msg)
            except Exception as e:
                logger.error(f"Error processing interbot message: {e}")
                self.send_response(msg.id, msg.from_bot, f"Error processing: {e}")

    # ============== Listener ==============

    def start_listener(self):
        """Start background listener thread that polls for incoming messages."""
        if self._running:
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop, daemon=True
        )
        self._listener_thread.start()
        logger.info("InterBot listener started")

    def stop_listener(self):
        """Stop the listener thread."""
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=5.0)

    def _listener_loop(self):
        """Main listener loop — polls inbox every 2 seconds."""
        while self._running:
            try:
                messages = self.poll_inbox()
                for msg in messages:
                    # Mark as processing BEFORE handling to prevent re-reads
                    self._mark_processing(msg)
                    self.process_incoming(msg)
                    # Archive after processing to prevent any re-reads
                    self._archive_processed(msg)
            except Exception as e:
                logger.error(f"InterBot listener error: {e}")
            time.sleep(2)

    # ============== Telegram Cross-Posting ==============

    def _telegram_crosspost(self, to_bot: str, text: str):
        """Send a message via the other bot's Telegram token.

        This makes the message appear in the other bot's chat context,
        allowing the TypeScript gateway to see and process it.
        """
        import requests

        bot_info = BOT_REGISTRY.get(to_bot)
        if not bot_info or not self.chat_id:
            logger.warning(f"Cannot cross-post to {to_bot}: missing info")
            return

        token = bot_info["token"]
        api_url = f"https://api.telegram.org/bot{token}/sendMessage"

        try:
            resp = requests.post(
                api_url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"Cross-posted to {to_bot} via Telegram")
            else:
                logger.warning(f"Cross-post failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Cross-post error: {e}")

    # ============== File Operations ==============

    def _gen_id(self, content: str) -> str:
        return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]

    def _write_message(self, msg: InterBotMessage):
        """Write message to the inbox directory."""
        filename = f"{int(msg.timestamp * 1000)}_{msg.id}.json"
        filepath = os.path.join(self.interbot_dir, "inbox", filename)

        with open(filepath, "w") as f:
            json.dump(msg.to_dict(), f, indent=2)

    def _mark_processing(self, msg: InterBotMessage):
        """Mark a message as being processed."""
        inbox = os.path.join(self.interbot_dir, "inbox")
        for filename in os.listdir(inbox):
            if msg.id in filename:
                filepath = os.path.join(inbox, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    data["status"] = MessageStatus.PROCESSING.value
                    with open(filepath, "w") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass
                break

    def _archive_processed(self, msg: InterBotMessage):
        """Archive a processed message — removes it from inbox permanently."""
        inbox = os.path.join(self.interbot_dir, "inbox")
        for filename in os.listdir(inbox):
            if msg.id in filename:
                filepath = os.path.join(inbox, filename)
                self._archive_message(filepath, "done")
                break

    def _update_message_status(self, msg_id: str, status: MessageStatus, response: str = None):
        """Update the status of a message in the inbox."""
        inbox = os.path.join(self.interbot_dir, "inbox")
        for filename in os.listdir(inbox):
            if msg_id in filename:
                filepath = os.path.join(inbox, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    data["status"] = status.value
                    if response:
                        data["response"] = response
                        data["response_time"] = time.time()
                    with open(filepath, "w") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass
                break

    def _archive_message(self, filepath: str, reason: str):
        """Move a processed/expired message to archive."""
        try:
            archive_dir = os.path.join(self.interbot_dir, "archive")
            basename = os.path.basename(filepath)
            archive_path = os.path.join(archive_dir, f"{reason}_{basename}")
            os.rename(filepath, archive_path)
        except Exception as e:
            logger.error(f"Failed to archive message: {e}")

    # ============== Utilities ==============

    def get_other_bot(self) -> str:
        """Get the ID of the other bot."""
        return "ellora" if self.my_bot_id == "ajanta" else "ajanta"

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status summary."""
        inbox = os.path.join(self.interbot_dir, "inbox")
        inbox_count = len([f for f in os.listdir(inbox) if f.endswith(".json")]) if os.path.exists(inbox) else 0

        archive = os.path.join(self.interbot_dir, "archive")
        archive_count = len([f for f in os.listdir(archive) if f.endswith(".json")]) if os.path.exists(archive) else 0

        return {
            "my_bot": self.my_bot_id,
            "other_bot": self.get_other_bot(),
            "listener_running": self._running,
            "inbox_messages": inbox_count,
            "archived_messages": archive_count,
            "pending_waiters": len(self._waiters),
        }


# ============== Global Instance ==============

_bridge: Optional[InterBotBridge] = None


def get_interbot_bridge(chat_id: str = None) -> InterBotBridge:
    """Get or create the global InterBot bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = InterBotBridge(
            my_bot_id="ajanta",
            chat_id=chat_id,
        )
    elif chat_id and not _bridge.chat_id:
        _bridge.chat_id = chat_id
    return _bridge


__all__ = [
    "InterBotBridge",
    "InterBotMessage",
    "MessageType",
    "MessageStatus",
    "BOT_REGISTRY",
    "get_interbot_bridge",
]
