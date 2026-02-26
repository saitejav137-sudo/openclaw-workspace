# OpenClaw

AI-powered desktop automation tool with browser control capabilities.

## Features

- **Screen Capture & Vision**: OCR and YOLO-based trigger detection
- **Browser Control API**: HTTP API for AI agents to control a headless browser
- **Telegram Integration**: Send alerts and receive commands via Telegram bot
- **Natural Language Automation**: Configure automations using ChatGPT/Claude
- **Multiple Interfaces**: CLI, REST API, WebSocket, Web UI, Mobile App

## Quick Start

```bash
cd openclaw
pip install -r requirements.txt
python main.py
```

The server runs on port 8765 by default.

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

## Architecture

```
openclaw/
├── core/           # Core modules (AI, vision, actions, automation)
├── integrations/   # External integrations (HTTP, Telegram, WebSocket)
├── storage/        # Database management
├── ui/             # Web dashboard and CLI
├── utils/          # Utilities
├── tests/          # Test suite
├── k8s/            # Kubernetes manifests
└── helm/           # Helm charts
```

## Testing

```bash
pytest openclaw/tests/
```

## License

MIT
