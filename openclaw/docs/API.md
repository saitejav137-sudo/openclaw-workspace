# OpenClaw API Documentation

## Overview

OpenClaw provides a REST API on port 8765 for:
- Vision-based trigger detection
- Browser automation
- Configuration management
- Statistics and monitoring

## Base URL

```
http://localhost:8765
```

## Authentication

### API Key (Optional)

Send API key in header:
```
Authorization: Bearer YOUR_API_KEY
```

Or as query parameter:
```
?api_key=YOUR_API_KEY
```

## Endpoints

### Health & Status

#### GET /

Check trigger status.

**Response:**
```json
{
  "status": "ok",
  "triggered": false,
  "condition_met": false,
  "mode": "ocr"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1772130437.26,
  "version": "2.0.0",
  "services": {
    "vision": true,
    "auth": false,
    "rate_limit": false
  }
}
```

#### GET /api/stats

Get trigger statistics.

**Response:**
```json
{
  "total": 0,
  "triggered": 0,
  "failed": 0,
  "success_rate": 0.0
}
```

### Browser Control

#### POST /api/browser

Execute browser actions.

**Request:**
```json
{
  "action": "start",
  "params": {
    "url": "https://example.com"
  }
}
```

**Actions:**
| Action | Parameters | Description |
|--------|------------|-------------|
| `start` | - | Start browser |
| `goto` | `url` | Navigate to URL |
| `click` | `selector` | Click element by CSS selector |
| `click_text` | `text` | Click element containing text |
| `type` | `selector`, `text` | Type into element |
| `input` | `text` | Type in first input |
| `submit` | - | Click submit button |
| `extract` | `selector` | Extract text from element |
| `extract_all` | - | Extract all page text |
| `screenshot` | - | Take screenshot |
| `info` | - | Get browser info |
| `close` | - | Close browser |

**Response:**
```json
{
  "success": true,
  "message": "Browser started"
}
```

#### GET /api/browser/info

Get browser information.

**Response:**
```json
{
  "success": true,
  "info": {
    "url": "https://google.com",
    "title": "Google",
    "initialized": true
  }
}
```

#### GET /api/browser/extract_all

Extract all text from current page.

**Response:**
```json
{
  "success": true,
  "text": "Page content..."
}
```

### Smart Browser Control

#### POST /api/smart

Natural language browser control.

**Request:**
```json
{
  "instruction": "Search for Python tutorials on Google"
}
```

**Response:**
```json
{
  "success": true,
  "result": "...",
  "actions": ["goto", "type", "click"]
}
```

### Configuration

#### GET /api/config

Get current configuration.

**Response:**
```json
{
  "mode": "ocr",
  "target_text": "Submit",
  "polling": true
}
```

#### POST /api/config

Update configuration.

**Request:**
```json
{
  "mode": "yolo",
  "yolo_model": "yolov8n.pt",
  "polling": true
}
```

### Screenshots

#### GET /api/screenshot

Get current screenshot.

**Response:**
Returns PNG image.

## Rate Limiting

When enabled, rate limiting uses token bucket algorithm:
- Default: 60 requests per minute

When limit exceeded:
```json
{
  "error": "Rate limit exceeded"
}
```

## Examples

### cURL

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

# Click by text
curl -X POST http://localhost:8765/api/browser \
  -H "Content-Type: application/json" \
  -d '{"action": "click_text", "params": {"text": "Search"}}'

# Type in search box
curl -X POST http://localhost:8765/api/browser \
  -H "Content-Type: application/json" \
  -d '{"action": "input", "params": {"text": "Hello World"}}'

# Extract all text
curl http://localhost:8765/api/browser/extract_all
```

### Python

```python
import requests

# Start browser
requests.post("http://localhost:8765/api/browser",
              json={"action": "start"})

# Navigate
requests.post("http://localhost:8765/api/browser",
              json={"action": "goto", "params": {"url": "https://example.com"}})

# Get page info
info = requests.get("http://localhost:8765/api/browser/info").json()
print(info["info"]["title"])
```

### JavaScript

```javascript
// Start browser
fetch('http://localhost:8765/api/browser', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({action: 'start'})
});

// Navigate
fetch('http://localhost:8765/api/browser', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({action: 'goto', params: {url: 'https://google.com'}})
});
```

## Error Responses

| Status | Description |
|--------|-------------|
| 400 | Bad Request - Invalid JSON or parameters |
| 401 | Unauthorized - Invalid API key |
| 404 | Not Found - Unknown endpoint |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
