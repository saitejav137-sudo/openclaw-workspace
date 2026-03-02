# InterBot Communication System

Cross-bot communication for OpenClaw (Ajanta ↔ Ellora).

## Overview

InterBot enables AI bots to communicate, delegate tasks, and share context.

## Architecture

```
┌─────────────┐      Messages       ┌─────────────┐
│   Ajanta    │ ◄─────────────────► │   Ellora    │
│  (Python)   │   ~/.openclaw/     │ (TypeScript)│
└─────────────┘     interbot/      └─────────────┘
```

## Message Types

| Type | Description |
|------|-------------|
| TASK | Request the other bot to do something |
| RESPONSE | Reply to a task |
| INFO | Informational broadcast |
| QUERY | Ask a question |

## Statuses

| Status | Description |
|--------|-------------|
| PENDING | Message waiting to be processed |
| PROCESSING | Message being handled |
| DONE | Task completed successfully |
| FAILED | Task failed |
| EXPIRED | TTL exceeded |

## Usage

### Python API

```python
from openclaw.core.interbot import (
    InterBotBridge,
    MessageType,
    get_interbot_bridge
)

# Create bridge
bridge = InterBotBridge(my_bot_id="ajanta")

# Send a task
task_id = bridge.send_task(
    to_bot="ellora",
    content="Run OCR on screenshot",
    metadata={"path": "/tmp/screen.png"}
)

# Send info
bridge.send_info(to_bot="ellora", content="System update")

# Send query (wait for response)
response = bridge.send_query(
    to_bot="ellora",
    question="What's the current status?"
)

# Poll inbox
messages = bridge.poll_inbox()

# Get status
status = bridge.get_status()
```

### Available Bots

```python
from openclaw.core.interbot import BOT_REGISTRY

for bot_id, info in BOT_REGISTRY.items():
    print(f"{bot_id}: {info['name']} ({info['gateway']})")
```

## Storage

```
~/.openclaw/interbot/
├── inbox/       # Messages for this bot
├── outbox/     # Messages sent by this bot
└── archive/    # Processed messages
```

## Message TTL

Default: 300 seconds (5 minutes)

Messages expire if not processed within TTL.

## Testing

```bash
python -m pytest tests/test_interbot.py -v
```

---

*For bot-to-bot communication*
