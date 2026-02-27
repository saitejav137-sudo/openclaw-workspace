# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenClaw** is a comprehensive AI-powered desktop automation framework with vision-based trigger detection, browser automation, and multi-interface support (HTTP, FastAPI, WebSocket, CLI).

## Running the Application

### Main Entry Point
```bash
cd openclaw
python3 -m openclaw.main
```

### Command Line Options
```bash
# OCR mode detection
python3 -m openclaw.main --mode ocr --text "Click Here" --action "alt+o"

# Template matching
python3 -m openclaw.main --mode template --template /path/to/image.png

# YOLO detection
python3 -m openclaw.main --mode yolo --yolo-classes "person,cat,dog"

# Window title monitoring
python3 -m openclaw.main --mode window --window-signal "TRIGGER_CLAW"

# Start with config file
python3 -m openclaw.main --config ~/.openclaw/config.yaml

# Show statistics
python3 -m openclaw.main --stats
```

## Architecture

OpenClaw is organized into several key modules:

### Core Modules (`openclaw/core/`)
- **vision.py** - Screen capture, OCR (EasyOCR/PyTesseract), template matching, YOLO, color detection, fuzzy matching
- **config.py** - Configuration management with YAML support and validation
- **actions.py** - Keyboard/mouse automation with retry logic
- **exceptions.py** - Comprehensive exception hierarchy (26+ exception types)
- **workspace.py** - Multi-environment configuration (dev/staging/prod)
- **di.py** - Dependency injection container

### Integrations (`openclaw/integrations/`)
- **http.py** - Basic HTTP server (port 8765)
- **fastapi_server.py** - FastAPI REST API with 60+ endpoints
- **websocket.py** - Real-time WebSocket communication
- **browser_agent.py** - AI-powered browser automation
- **telegram.py** - Telegram bot integration

### Storage (`openclaw/storage/`)
- **database.py** - Multi-database support (SQLite, PostgreSQL, MySQL)
- **vector_db.py** - Vector database for semantic search

## Key Features

- **10+ Detection Modes**: OCR, YOLO, template, color, fuzzy, regression, window, monitor, analyze, multi
- **Adaptive Polling**: Intelligent polling intervals (idle: 2.0s, active: 0.2s)
- **Screen Capture Caching**: Configurable TTL (default 0.5s)
- **Action Sequences**: Multi-step automation with async execution
- **Retry Logic**: Configurable retry strategies (FIXED, LINEAR, EXPONENTIAL)
- **Security**: API keys, rate limiting, input sanitization, TLS/HTTPS support

## Dependencies

- **Python 3.10+**
- **OpenCV** (`opencv-python`) - Image processing
- **NumPy** - Numerical operations
- **FastAPI** - REST API framework
- **mss** - Screen capture
- **EasyOCR** / **PyTesseract** - Text recognition
- **xdotool** - Keyboard/mouse automation (Linux)
- **cryptography** - Encryption (optional)

## Development

### Running Tests
```bash
cd openclaw
python -m pytest tests/ -v
```

### Code Quality
- Type hints where possible
- Specific exception handling (no bare `except:`)
- Comprehensive exception hierarchy
- Dependency injection for testability
