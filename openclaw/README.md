# OpenClaw

AI-powered desktop automation tool with browser control capabilities.

## Features

- **Screen Capture & Vision**: OCR and YOLO-based trigger detection (10+ modes)
- **Browser Control API**: HTTP API for AI agents to control a headless browser
- **Telegram Integration**: Send alerts and receive commands via Telegram bot
- **Natural Language Automation**: Configure automations using ChatGPT/Claude
- **Multiple Interface**: CLI, REST API, WebSocket, Web UI, Mobile App
- **Security**: API key authentication, rate limiting, input validation
- **TLS/HTTPS Support**: Optional HTTPS with self-signed certificates

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCLAW_API_KEY` | API key for authentication | - |
| `OPENCLAW_TLS_ENABLED` | Enable HTTPS | false |
| `OPENCLAW_TLS_CERT_PATH` | Path to TLS certificate | - |
| `OPENCLAW_TLS_KEY_PATH` | Path to TLS key | - |
| `OPENCLAW_TELEGRAM_TOKEN` | Telegram bot token | - |
| `OPENCLAW_TELEGRAM_CHAT_ID` | Telegram chat ID | - |

## API

See [docs/API.md](docs/API.md) for complete API documentation.

### Quick Examples

```bash
# Check health
curl http://localhost:8765/health

# Start browser
curl -X POST http://localhost:8765/api/browser \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'

# Navigate to URL
curl -X POST http://localhost:8765/api/browser \
  -H "Content-Type: application/json" \
  -d '{"action": "goto", "params": {"url": "https://google.com"}}'

# Take screenshot
curl -X POST http://localhost:8765/api/browser \
  -H "Content-Type: application/json" \
  -d '{"action": "screenshot"}'
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
├── docs/           # Documentation
├── k8s/            # Kubernetes manifests
└── helm/           # Helm charts
```

## Security Features

- **API Key Authentication**: Secure your API with a key
- **Rate Limiting**: Prevent abuse with configurable rate limits
- **Input Validation**: Sanitize all user inputs
- **TLS/HTTPS**: Encrypt traffic with TLS

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_http_integration.py -v
```

## Docker

```bash
# Build
docker build -t openclaw .

# Run
docker run -d -p 8765:8765 --net=host openclaw

# With Docker Compose
docker-compose up -d
```

## License

MIT
