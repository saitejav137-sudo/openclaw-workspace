# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a small desktop automation tool that detects a trigger signal (from a Tampermonkey userscript) and presses the Alt+O keyboard shortcut. It has two implementations:

- **auto_claw.py** - Python HTTP server that listens on port 8765
- **auto_claw.sh** - Bash script that watches window titles for "TRIGGER_CLAW"

## Running the Scripts

### Python version (auto_claw.py)
```bash
python3 auto_claw.py
```
Requires: Python 3, xdotool

### Bash version (auto_claw.sh)
```bash
./auto_claw.sh
```
Requires: xdotool

## Architecture

Both scripts perform the same function:
1. Wait for a trigger signal (HTTP request or window title change)
2. Press Alt+O using xdotool after a short delay
3. Implement debouncing to prevent duplicate triggers (3 second cooldown for Python, title-based dedup for Bash)

The scripts work in conjunction with a Tampermonkey userscript (not included in this repo) that sends the trigger signal when certain web pages load.

## Dependencies

- **xdotool** - For simulating keyboard input
- **Python 3** - For running auto_claw.py
- **Tampermonkey** browser extension with a userscript that:
  - Makes an HTTP request to localhost:8765 (for Python version), OR
  - Changes the browser window title to include "TRIGGER_CLAW" (for Bash version)
