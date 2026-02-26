"""
Browser Control API for AI Agents

Simple HTTP API that AI agents can call to control the browser.
No Telegram commands needed!
"""

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger("browser_api")


# Global browser agent instance
_browser_agent = None


def get_browser_controller():
    """Get browser controller singleton"""
    global _browser_agent
    if _browser_agent is None:
        from .browser_agent import BrowserAgent
        _browser_agent = BrowserAgent(headless=False)
    return _browser_agent


def browser_start() -> Dict[str, Any]:
    """Start browser"""
    try:
        agent = get_browser_controller()
        result = agent.start()
        if result:
            info = agent.get_page_info()
            return {"success": True, "message": "Browser started", "url": info.get("url")}
        return {"success": False, "message": "Failed to start browser"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_goto(url: str) -> Dict[str, Any]:
    """Navigate to URL"""
    try:
        agent = get_browser_controller()
        result = agent.navigate(url)
        if result.success:
            info = agent.get_page_info()
            return {"success": True, "message": f"Navigated to {url}", "title": info.get("title")}
        return {"success": False, "message": result.error}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_click(selector: str) -> Dict[str, Any]:
    """Click element by selector"""
    try:
        agent = get_browser_controller()
        result = agent.click(selector)
        if result.success:
            return {"success": True, "message": f"Clicked {selector}"}
        return {"success": False, "message": result.error}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_click_text(text: str) -> Dict[str, Any]:
    """Click element containing text"""
    try:
        agent = get_browser_controller()
        script = f"""
        function() {{
            let elems = document.querySelectorAll('button, a, input[type="button"], input[type="submit"]');
            for (let el of elems) {{
                if (el.innerText.toLowerCase().includes('{text}'.toLowerCase()) ||
                    el.value?.toLowerCase().includes('{text}'.toLowerCase())) {{
                    el.click();
                    return 'clicked';
                }}
            }}
            return 'not_found';
        }}()
        """
        result = agent.evaluate(script)
        if result.success and result.data == 'clicked':
            return {"success": True, "message": f"Clicked element with text '{text}'"}
        return {"success": False, "message": f"Could not find element with text '{text}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_type(selector: str, text: str) -> Dict[str, Any]:
    """Type text into element"""
    try:
        agent = get_browser_controller()
        result = agent.type(selector, text)
        if result.success:
            return {"success": True, "message": f"Typed '{text}' into {selector}"}
        return {"success": False, "message": result.error}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_input(text: str) -> Dict[str, Any]:
    """Type into first available input"""
    try:
        agent = get_browser_controller()
        script = f"""
        function() {{
            let inputs = document.querySelectorAll('input[type="text"], input[type="search"], textarea');
            for (let inp of inputs) {{
                if (!inp.disabled && !inp.readOnly) {{
                    inp.value = '{text}';
                    inp.dispatchEvent(new Event('input', {{bubbles: true}}));
                    return 'typed';
                }}
            }}
            return 'not_found';
        }}()
        """
        result = agent.evaluate(script)
        if result.success and result.data == 'typed':
            return {"success": True, "message": f"Typed '{text}' into input field"}
        return {"success": False, "message": "Could not find input field"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_submit() -> Dict[str, Any]:
    """Click submit button"""
    try:
        agent = get_browser_controller()
        script = """
        function() {
            let btns = document.querySelectorAll('button[type="submit"], input[type="submit"]');
            for (let btn of btns) { btn.click(); return 'clicked'; }
            return 'not_found';
        }()
        """
        result = agent.evaluate(script)
        if result.success and result.data == 'clicked':
            return {"success": True, "message": "Clicked submit button"}
        return {"success": False, "message": "Could not find submit button"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_extract(selector: str) -> Dict[str, Any]:
    """Extract text from element"""
    try:
        agent = get_browser_controller()
        result = agent.extract_text(selector)
        if result.success:
            return {"success": True, "text": result.data}
        return {"success": False, "message": result.error}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_extract_all() -> Dict[str, Any]:
    """Extract all page text"""
    try:
        agent = get_browser_controller()
        script = "function() { return document.body.innerText.replace(/\\s+/g, ' ').trim().substring(0, 5000); }()"
        result = agent.evaluate(script)
        if result.success:
            return {"success": True, "text": result.data}
        return {"success": False, "message": result.error}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_screenshot() -> Dict[str, Any]:
    """Take screenshot (returns base64)"""
    try:
        agent = get_browser_controller()
        result = agent.screenshot()
        if result.success:
            return {"success": True, "screenshot": result.screenshot}
        return {"success": False, "message": result.error}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_info() -> Dict[str, Any]:
    """Get browser info"""
    try:
        agent = get_browser_controller()
        info = agent.get_page_info()
        return {"success": True, "info": info}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_close() -> Dict[str, Any]:
    """Close browser"""
    try:
        global _browser_agent
        if _browser_agent:
            _browser_agent.stop()
            _browser_agent = None
        return {"success": True, "message": "Browser closed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_browser_action(action: str, params: Dict = None) -> Dict[str, Any]:
    """
    Execute browser action from AI agent.

    Actions:
    - start: Start browser
    - goto: Navigate to URL (params: url)
    - click: Click element (params: selector)
    - click_text: Click by text (params: text)
    - type: Type into element (params: selector, text)
    - input: Type in first input (params: text)
    - submit: Click submit
    - extract: Extract text (params: selector)
    - extract_all: Extract all text
    - screenshot: Take screenshot
    - info: Get browser info
    - close: Close browser
    """
    params = params or {}

    action_map = {
        "start": browser_start,
        "goto": lambda: browser_goto(params.get("url", "")),
        "click": lambda: browser_click(params.get("selector", "")),
        "click_text": lambda: browser_click_text(params.get("text", "")),
        "type": lambda: browser_type(params.get("selector", ""), params.get("text", "")),
        "input": lambda: browser_input(params.get("text", "")),
        "submit": browser_submit,
        "extract": lambda: browser_extract(params.get("selector", "")),
        "extract_all": browser_extract_all,
        "screenshot": browser_screenshot,
        "info": browser_info,
        "close": browser_close,
    }

    if action not in action_map:
        return {"success": False, "error": f"Unknown action: {action}"}

    return action_map[action]()


# Quick functions for AI agents
def quick_browse(url: str) -> Dict[str, Any]:
    """Quick browse - start browser, go to URL, return content"""
    # Start browser
    result = browser_start()
    if not result.get("success"):
        return result

    # Go to URL
    result = browser_goto(url)
    if not result.get("success"):
        return result

    # Extract all text
    return browser_extract_all()


__all__ = [
    "browser_start",
    "browser_goto",
    "browser_click",
    "browser_click_text",
    "browser_type",
    "browser_input",
    "browser_submit",
    "browser_extract",
    "browser_extract_all",
    "browser_screenshot",
    "browser_info",
    "browser_close",
    "execute_browser_action",
    "quick_browse",
]
