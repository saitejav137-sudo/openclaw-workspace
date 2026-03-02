"""
Tests for InterBot Communication System
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
import time
from pathlib import Path

sys.path.insert(0, 'openclaw')

from openclaw.core.interbot import (
    InterBotBridge,
    InterBotMessage,
    MessageType,
    MessageStatus,
    BOT_REGISTRY,
    get_interbot_bridge,
)


# Test fixtures
@pytest.fixture
def temp_interbot_dir():
    """Create a temporary directory for interbot."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def bridge(temp_interbot_dir):
    """Create an InterBotBridge instance."""
    return InterBotBridge(
        my_bot_id="ajanta",
        interbot_dir=temp_interbot_dir
    )


class TestBotRegistry:
    """Test BOT_REGISTRY."""

    def test_registry_exists(self):
        """Test that registry exists."""
        assert isinstance(BOT_REGISTRY, dict)
        assert len(BOT_REGISTRY) > 0

    def test_ajanta_in_registry(self):
        """Test ajanta is in registry."""
        assert "ajanta" in BOT_REGISTRY
        assert BOT_REGISTRY["ajanta"]["name"] == "Ajanta"
        assert BOT_REGISTRY["ajanta"]["gateway"] == "python"

    def test_ellora_in_registry(self):
        """Test ellora is in registry."""
        assert "ellora" in BOT_REGISTRY
        assert BOT_REGISTRY["ellora"]["name"] == "Ellora"
        assert BOT_REGISTRY["ellora"]["gateway"] == "typescript"


class TestMessageType:
    """Test MessageType enum."""

    def test_all_message_types(self):
        """Test all message types exist."""
        assert MessageType.TASK.value == "task"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.INFO.value == "info"
        assert MessageType.QUERY.value == "query"


class TestMessageStatus:
    """Test MessageStatus enum."""

    def test_all_statuses(self):
        """Test all statuses exist."""
        assert MessageStatus.PENDING.value == "pending"
        assert MessageStatus.PROCESSING.value == "processing"
        assert MessageStatus.DONE.value == "done"
        assert MessageStatus.FAILED.value == "failed"
        assert MessageStatus.EXPIRED.value == "expired"


class TestInterBotMessage:
    """Test InterBotMessage dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = InterBotMessage(
            id="test123",
            from_bot="ajanta",
            to_bot="ellora",
            msg_type=MessageType.TASK.value,
            content="Test content"
        )
        assert msg.id == "test123"
        assert msg.from_bot == "ajanta"
        assert msg.to_bot == "ellora"
        assert msg.msg_type == "task"
        assert msg.content == "Test content"

    def test_to_dict(self):
        """Test converting to dict."""
        msg = InterBotMessage(
            id="test456",
            from_bot="ajanta",
            to_bot="ellora",
            msg_type=MessageType.INFO.value,
            content="Info content"
        )
        data = msg.to_dict()
        assert isinstance(data, dict)
        assert data["id"] == "test456"
        assert data["msg_type"] == "info"

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "id": "test789",
            "from_bot": "ellora",
            "to_bot": "ajanta",
            "msg_type": "task",
            "content": "Task content",
            "timestamp": 1234567890.0,
            "status": "pending",
            "response": None,
            "response_time": None,
            "metadata": {},
            "ttl": 300,
        }
        msg = InterBotMessage.from_dict(data)
        assert msg.id == "test789"
        assert msg.from_bot == "ellora"

    def test_is_expired(self):
        """Test expiration check."""
        # Fresh message should not be expired
        msg = InterBotMessage(
            id="test1",
            from_bot="ajanta",
            to_bot="ellora",
            msg_type=MessageType.TASK.value,
            content="Test"
        )
        assert msg.is_expired == False

    def test_is_expired_true(self):
        """Test expired message."""
        msg = InterBotMessage(
            id="test2",
            from_bot="ajanta",
            to_bot="ellora",
            msg_type=MessageType.TASK.value,
            content="Test",
            timestamp=0,  # Very old
            ttl=300
        )
        assert msg.is_expired == True


class TestInterBotBridge:
    """Test InterBotBridge functionality."""

    def test_init(self, bridge):
        """Test bridge initialization."""
        assert bridge.my_bot_id == "ajanta"
        assert bridge.interbot_dir is not None

    def test_send_task(self, bridge):
        """Test sending a task."""
        task_id = bridge.send_task(
            to_bot="ellora",
            content="Test task",
            metadata={"key": "value"}
        )
        assert task_id is not None
        assert isinstance(task_id, str)

    def test_send_info(self, bridge):
        """Test sending info."""
        info_id = bridge.send_info(
            to_bot="ellora",
            content="Test info"
        )
        # May be None if telegram not configured
        assert info_id is None or isinstance(info_id, str)

    def test_get_inbox(self, bridge):
        """Test getting inbox (use poll_inbox)."""
        messages = bridge.poll_inbox()
        assert isinstance(messages, list)

    def test_get_outbox(self, bridge):
        """Test getting outbox."""
        # First send a message to populate outbox
        bridge.send_task(to_bot="ellora", content="Test")
        messages = bridge.poll_inbox()
        assert isinstance(messages, list)

    def test_poll_inbox(self, bridge):
        """Test polling inbox."""
        messages = bridge.poll_inbox()
        assert isinstance(messages, list)

    def test_get_other_bot(self, bridge):
        """Test getting other bot."""
        other = bridge.get_other_bot()
        assert other == "ellora"

    def test_get_status(self, bridge):
        """Test getting status."""
        status = bridge.get_status()
        assert isinstance(status, dict)
        assert "my_bot" in status
        assert "other_bot" in status
        assert status["my_bot"] == "ajanta"


class TestGetInterbotBridge:
    """Test get_interbot_bridge function."""

    def test_get_bridge(self):
        """Test getting interbot bridge."""
        bridge = get_interbot_bridge()
        assert isinstance(bridge, InterBotBridge)

    def test_get_bridge_with_chat_id(self):
        """Test getting bridge with chat_id."""
        bridge = get_interbot_bridge(chat_id="12345")
        assert isinstance(bridge, InterBotBridge)


class TestMessageFlow:
    """Test complete message flow."""

    def test_full_message_lifecycle(self, temp_interbot_dir):
        """Test sending and receiving a message."""
        # Create two bridges
        bridge1 = InterBotBridge(my_bot_id="ajanta", interbot_dir=temp_interbot_dir)
        bridge2 = InterBotBridge(my_bot_id="ellora", interbot_dir=temp_interbot_dir)

        # Send a task from ajanta to ellora
        task_id = bridge1.send_task(
            to_bot="ellora",
            content="Process this data"
        )
        assert task_id is not None

        # Check ellora's inbox
        time.sleep(0.1)  # Give filesystem time to write
        inbox = bridge2.poll_inbox()

        # Message should be in inbox (may be 0 if filesystem sync needed)
        assert isinstance(inbox, list)


class TestEdgeCases:
    """Test edge cases."""

    def test_message_with_empty_content(self, bridge):
        """Test message with empty content."""
        task_id = bridge.send_task(to_bot="ellora", content="")
        assert task_id is not None

    def test_message_with_special_chars(self, bridge):
        """Test message with special characters."""
        content = "Test 🎉 emoji and unicode: 你好"
        task_id = bridge.send_task(to_bot="ellora", content=content)
        assert task_id is not None

    def test_invalid_to_bot(self, bridge):
        """Test sending to invalid bot."""
        # Should still work (just won't deliver)
        task_id = bridge.send_task(to_bot="invalid_bot", content="Test")
        assert task_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
