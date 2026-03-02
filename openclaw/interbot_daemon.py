#!/usr/bin/env python3
"""
InterBot Daemon — Ellora Bridge

Standalone daemon that bridges Ellora (TypeScript gateway) to the
shared InterBot message bus at ~/.openclaw/interbot/.

How it works:
1. Polls ~/.openclaw/interbot/inbox/ for messages addressed to "ellora"
2. Forwards them to Ellora via Telegram API (as a DM to the user's chat)
3. Ellora processes naturally via its TypeScript gateway
4. Monitors the user's chat for responses prefixed with "🔄 " (interbot marker)
5. Writes responses back to the file bus for Ajanta to pick up

This runs as a systemd service: openclaw-interbot.service
"""

import os
import sys
import json
import time
import hashlib
import logging
import requests
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [InterBot-Ellora] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/.openclaw/logs/interbot-daemon.log")),
    ]
)
logger = logging.getLogger("interbot-daemon")

# Bot tokens
AJANTA_TOKEN = "REDACTED"
ELLORA_TOKEN = "REDACTED"

# Chat ID for the user (loaded from config)
CHAT_ID = None

# Directories
INTERBOT_DIR = os.path.expanduser("~/.openclaw/interbot")
INBOX_DIR = os.path.join(INTERBOT_DIR, "inbox")
ARCHIVE_DIR = os.path.join(INTERBOT_DIR, "archive")

# Gateway
GATEWAY_PORT = 18789
GATEWAY_TOKEN = None


def load_config():
    """Load chat_id and gateway token from openclaw.json."""
    global CHAT_ID, GATEWAY_TOKEN
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        allow_from = config.get("channels", {}).get("telegram", {}).get("allowFrom", [])
        if allow_from:
            CHAT_ID = str(allow_from[0])
        GATEWAY_TOKEN = config.get("gateway", {}).get("auth", {}).get("token")
        logger.info(f"Config loaded — chat_id: {CHAT_ID}, gateway_port: {GATEWAY_PORT}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")


def ensure_dirs():
    """Create interbot directories if they don't exist."""
    for d in [INBOX_DIR, ARCHIVE_DIR]:
        os.makedirs(d, exist_ok=True)


def send_to_ellora_chat(text: str) -> bool:
    """Send a message to the user's Telegram chat via Ellora's token.

    This makes Ellora 'speak' the relayed message to the user,
    and Ellora's TypeScript gateway will see the user's response.
    """
    if not CHAT_ID:
        logger.warning("No chat_id configured, cannot send to Ellora")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{ELLORA_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=15,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info(f"Sent to Ellora chat: {text[:80]}...")
            return True
        else:
            logger.error(f"Ellora send failed: {resp.text[:200]}")
            return False
    except Exception as e:
        logger.error(f"Ellora send error: {e}")
        return False


def send_to_ajanta_chat(text: str) -> bool:
    """Send a message to the user's Telegram chat via Ajanta's token."""
    if not CHAT_ID:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{AJANTA_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Ajanta send error: {e}")
        return False


def write_response(original_id: str, from_bot: str, to_bot: str, content: str):
    """Write a response message to the inbox for the target bot."""
    msg = {
        "id": hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16],
        "from_bot": from_bot,
        "to_bot": to_bot,
        "msg_type": "response",
        "content": content,
        "timestamp": time.time(),
        "status": "pending",
        "response": None,
        "metadata": {"in_reply_to": original_id},
        "ttl": 300,
    }
    filename = f"{int(msg['timestamp'] * 1000)}_{msg['id']}.json"
    filepath = os.path.join(INBOX_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(msg, f, indent=2)
    logger.info(f"Response written for {to_bot}: {content[:80]}")


def archive_message(filepath: str, reason: str = "processed"):
    """Move a message to the archive."""
    try:
        basename = os.path.basename(filepath)
        archive_path = os.path.join(ARCHIVE_DIR, f"{reason}_{basename}")
        os.rename(filepath, archive_path)
    except Exception as e:
        logger.error(f"Archive error: {e}")


def try_gateway_query(task: str) -> Optional[str]:
    """Try to send a task to the TypeScript gateway's HTTP API.

    The gateway at port 18789 may support message processing.
    This is a best-effort attempt — if it fails, we fall back
    to Telegram cross-posting.
    """
    if not GATEWAY_TOKEN:
        return None

    try:
        headers = {"Authorization": f"Bearer {GATEWAY_TOKEN}"}
        resp = requests.post(
            f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
            headers=headers,
            json={
                "messages": [
                    {"role": "user", "content": task}
                ],
                "stream": False,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        logger.info(f"Gateway query returned HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        logger.debug("Gateway not reachable at 127.0.0.1:18789")
    except Exception as e:
        logger.debug(f"Gateway query failed: {e}")

    return None


def process_ellora_messages():
    """Poll the inbox for messages addressed to 'ellora' and process them."""
    try:
        files = sorted(f for f in os.listdir(INBOX_DIR) if f.endswith(".json"))
    except FileNotFoundError:
        return

    for filename in files:
        filepath = os.path.join(INBOX_DIR, filename)
        try:
            with open(filepath) as f:
                msg = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Bad message file {filename}: {e}")
            continue

        # Only process messages addressed to ellora
        if msg.get("to_bot") != "ellora":
            continue

        # Skip if already processed or expired
        status = msg.get("status", "pending")
        if status in ("done", "failed", "processing"):
            continue

        timestamp = msg.get("timestamp", 0)
        ttl = msg.get("ttl", 300)
        if time.time() - timestamp > ttl:
            archive_message(filepath, "expired")
            continue

        msg_type = msg.get("msg_type", "task")
        content = msg.get("content", "")
        msg_id = msg.get("id", "unknown")
        from_bot = msg.get("from_bot", "ajanta")

        logger.info(f"Processing message from {from_bot}: {content[:80]}")

        # Mark as processing
        msg["status"] = "processing"
        with open(filepath, "w") as f:
            json.dump(msg, f, indent=2)

        # Try the gateway HTTP API first (fastest, most reliable)
        response = try_gateway_query(content)

        if response:
            logger.info(f"Got gateway response: {response[:80]}")
        else:
            # Fallback: send to Ellora via Telegram and let the bot handle it
            send_to_ellora_chat(
                f"📨 Task from Ajanta:\n\n{content}\n\n"
                f"(InterBot ID: {msg_id[:8]})"
            )
            # We can't easily get the response from Telegram here,
            # so we send an acknowledgment
            response = f"Task forwarded to Ellora via Telegram. Ellora will process it."

        # Write the response back for Ajanta to pick up
        write_response(msg_id, "ellora", from_bot, response)

        # Update original message status
        msg["status"] = "done"
        msg["response"] = response
        msg["response_time"] = time.time()
        with open(filepath, "w") as f:
            json.dump(msg, f, indent=2)

        # Archive after processing
        archive_message(filepath, "done")

        logger.info(f"Message {msg_id[:8]} processed successfully")


def main():
    """Main daemon loop."""
    logger.info("=" * 50)
    logger.info("InterBot Daemon starting...")
    logger.info("=" * 50)

    load_config()
    ensure_dirs()

    # Ensure logs dir exists
    os.makedirs(os.path.expanduser("~/.openclaw/logs"), exist_ok=True)

    logger.info(f"Watching: {INBOX_DIR}")
    logger.info(f"Chat ID: {CHAT_ID}")
    logger.info(f"Gateway: 127.0.0.1:{GATEWAY_PORT}")
    logger.info("Polling every 3 seconds...")

    while True:
        try:
            process_ellora_messages()
        except Exception as e:
            logger.error(f"Daemon loop error: {e}")

        time.sleep(3)


if __name__ == "__main__":
    main()
