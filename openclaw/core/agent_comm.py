"""
Agent Communication System for OpenClaw

Message passing between agents for collaboration.
"""

import time
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import threading

from .logger import get_logger

logger = get_logger("agent_comm")


class MessageType(Enum):
    """Agent message types"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    EVENT = "event"
    ERROR = "error"


class Priority(Enum):
    """Message priority"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Any
    priority: Priority
    correlation_id: str = None
    reply_to: str = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Agent information"""
    id: str
    name: str
    role: str
    status: str
    capabilities: List[str]


class AgentMailbox:
    """Mailbox for agent messages"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._messages: List[AgentMessage] = []
        self._lock = threading.Lock()
        self._handlers: List[Callable] = []

    def add_message(self, message: AgentMessage):
        """Add message to mailbox"""
        with self._lock:
            self._messages.append(message)

            # Sort by priority
            self._messages.sort(key=lambda m: m.priority.value, reverse=True)

    def get_messages(self, unread_only: bool = True) -> List[AgentMessage]:
        """Get messages"""
        with self._lock:
            if unread_only:
                return [m for m in self._messages if not m.metadata.get("read")]
            return self._messages.copy()

    def mark_read(self, message_id: str):
        """Mark message as read"""
        with self._lock:
            for m in self._messages:
                if m.id == message_id:
                    m.metadata["read"] = True

    def clear(self):
        """Clear mailbox"""
        with self._lock:
            self._messages.clear()


class AgentCommunication:
    """
    Agent communication hub.

    Features:
    - Direct messaging between agents
    - Broadcasting to all agents
    - Message queues per agent
    - Async message handling
    - Message persistence
    """

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._mailboxes: Dict[str, AgentMailbox] = {}
        self._message_history: List[AgentMessage] = []
        self._lock = threading.Lock()
        self._max_history = 1000
        self._subscribers: Dict[str, List[Callable]] = {}

    def register_agent(
        self,
        agent_id: str,
        name: str,
        role: str = "worker",
        capabilities: List[str] = None
    ) -> Agent:
        """Register an agent"""
        with self._lock:
            agent = Agent(
                id=agent_id,
                name=name,
                role=role,
                status="online",
                capabilities=capabilities or []
            )
            self._agents[agent_id] = agent
            self._mailboxes[agent_id] = AgentMailbox(agent_id)

            logger.info(f"Registered agent: {agent_id}")
            return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                if agent_id in self._mailboxes:
                    del self._mailboxes[agent_id]
                return True
            return False

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent info"""
        return self._agents.get(agent_id)

    def list_agents(self, role: str = None) -> List[Agent]:
        """List all agents"""
        with self._lock:
            if role:
                return [a for a in self._agents.values() if a.role == role]
            return list(self._agents.values())

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: Any,
        message_type: MessageType = MessageType.REQUEST,
        priority: Priority = Priority.NORMAL,
        correlation_id: str = None,
        metadata: Dict = None
    ) -> Optional[AgentMessage]:
        """Send message to agent"""
        with self._lock:
            # Check agents exist
            if from_agent not in self._agents or to_agent not in self._agents:
                logger.warning(f"Agent not found: {from_agent} or {to_agent}")
                return None

            message = AgentMessage(
                id=str(uuid4()),
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                content=content,
                priority=priority,
                correlation_id=correlation_id,
                metadata=metadata or {}
            )

            # Add to recipient's mailbox
            self._mailboxes[to_agent].add_message(message)

            # Add to history
            self._add_to_history(message)

            logger.debug(f"Message sent: {from_agent} -> {to_agent}")
            return message

    def broadcast(
        self,
        from_agent: str,
        content: Any,
        message_type: MessageType = MessageType.BROADCAST,
        priority: Priority = Priority.NORMAL,
        to_roles: List[str] = None
    ):
        """Broadcast message to all agents"""
        with self._lock:
            target_agents = self._agents.values()
            if to_roles:
                target_agents = [a for a in target_agents if a.role in to_roles]

            for agent in target_agents:
                if agent.id != from_agent:
                    self.send_message(
                        from_agent,
                        agent.id,
                        content,
                        message_type,
                        priority
                    )

    def get_messages(
        self,
        agent_id: str,
        unread_only: bool = True
    ) -> List[AgentMessage]:
        """Get messages for agent"""
        mailbox = self._mailboxes.get(agent_id)
        if not mailbox:
            return []
        return mailbox.get_messages(unread_only)

    def reply_to(
        self,
        original_message: AgentMessage,
        content: Any,
        message_type: MessageType = MessageType.RESPONSE
    ) -> Optional[AgentMessage]:
        """Reply to a message"""
        return self.send_message(
            from_agent=original_message.to_agent,
            to_agent=original_message.from_agent,
            content=content,
            message_type=message_type,
            correlation_id=original_message.correlation_id or original_message.id
        )

    def subscribe(self, agent_id: str, handler: Callable):
        """Subscribe to messages"""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)

    def notify(self, agent_id: str, message: AgentMessage):
        """Notify agent of new message"""
        if agent_id in self._subscribers:
            for handler in self._subscribers[agent_id]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

    def _add_to_history(self, message: AgentMessage):
        """Add message to history"""
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)

    def get_history(
        self,
        agent_id: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get message history"""
        messages = self._message_history
        if agent_id:
            messages = [
                m for m in messages
                if m.from_agent == agent_id or m.to_agent == agent_id
            ]
        return [
            {
                "id": m.id,
                "from": m.from_agent,
                "to": m.to_agent,
                "type": m.message_type.value,
                "timestamp": m.timestamp
            }
            for m in messages[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get communication stats"""
        return {
            "total_agents": len(self._agents),
            "total_messages": len(self._message_history),
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "role": a.role,
                    "status": a.status
                }
                for a in self._agents.values()
            ]
        }


# Global communication hub
_communication: Optional[AgentCommunication] = None


def get_communication() -> AgentCommunication:
    """Get global communication hub"""
    global _communication
    if _communication is None:
        _communication = AgentCommunication()
    return _communication


def register_agent(agent_id: str, name: str, role: str = "worker") -> Agent:
    """Quick register agent"""
    return get_communication().register_agent(agent_id, name, role)


def send_to(agent_id: str, content: Any) -> Optional[AgentMessage]:
    """Quick send message"""
    return get_communication().send_message("system", agent_id, content)


def receive_from(agent_id: str) -> List[AgentMessage]:
    """Quick receive messages"""
    return get_communication().get_messages(agent_id)


__all__ = [
    "MessageType",
    "Priority",
    "AgentMessage",
    "Agent",
    "AgentMailbox",
    "AgentCommunication",
    "get_communication",
    "register_agent",
    "send_to",
    "receive_from",
]
