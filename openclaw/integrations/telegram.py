"""Telegram bot integration"""

import os
import time
import json
import threading
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass

import requests

from ..core.logger import get_logger
from ..core.actions import RetryConfig, ActionExecutor

logger = get_logger("telegram")

# Import search module for internet access
try:
    from .search import SearchEngine, get_search_engine, SearchResponse
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    logger.warning("Search module not available")

# Import browser fetcher for advanced web access
try:
    from .browser_fetch import BrowserFetcher, BrowserFetchResult
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False
    logger.warning("Browser fetch not available")

# Import browser agent for direct browser control
try:
    from .browser_agent import BrowserAgent, get_browser_agent, close_browser_agent
    BROWSER_AGENT_AVAILABLE = True
except ImportError:
    BROWSER_AGENT_AVAILABLE = False
    logger.warning("Browser agent not available")


@dataclass
class TelegramCommand:
    """Telegram command definition"""
    name: str
    description: str
    handler: Callable


class TelegramBot:
    """Telegram bot for remote control and notifications"""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        retry_config: RetryConfig = None,
        search_provider: str = "duckduckgo",
        brave_api_key: Optional[str] = None
    ):
        self.token = token or self._load_token()
        self.chat_id = chat_id or self._load_chat_id()
        self.api_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.enabled = bool(self.token and self.chat_id)
        self.retry_config = retry_config or RetryConfig()
        self.executor = ActionExecutor(self.retry_config)
        self._commands: Dict[str, TelegramCommand] = {}
        self._command_callbacks: Dict[str, Callable] = {}

        # Initialize search engine for internet access
        self.search_available = SEARCH_AVAILABLE
        if self.search_available:
            self.search_engine = get_search_engine(search_provider, brave_api_key)
        else:
            self.search_engine = None

        # Initialize browser fetcher for advanced web access
        self.browser_available = BROWSER_AVAILABLE
        if self.browser_available:
            try:
                self.browser_fetcher = BrowserFetcher(headless=True, timeout=30)
                logger.info("Browser fetcher initialized")
            except Exception as e:
                logger.warning(f"Browser fetcher init failed: {e}")
                self.browser_fetcher = None
                self.browser_available = False
        else:
            self.browser_fetcher = None

        # Initialize browser agent for direct browser control
        self.browser_agent_available = BROWSER_AGENT_AVAILABLE
        self.browser_agent = None

        if self.enabled:
            self._register_default_commands()
            self._register_search_commands()
            self._register_browser_commands()

    def _load_token(self) -> Optional[str]:
        """Load token from environment or config"""
        import os
        token = os.getenv("TELEGRAM_BOT_TOKEN")

        if not token:
            config_path = os.path.expanduser("~/.openclaw/openclaw.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    token = config.get("channels", {}).get("telegram", {}).get("botToken")
                except Exception as e:
                    logger.warning(f"Failed to load Telegram config: {e}")

        return token

    def _load_chat_id(self) -> Optional[str]:
        """Load chat ID from environment or config"""
        import os
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not chat_id:
            config_path = os.path.expanduser("~/.openclaw/openclaw.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    allow_from = config.get("channels", {}).get("telegram", {}).get("allowFrom", [])
                    if allow_from:
                        chat_id = str(allow_from[0])
                except Exception as e:
                    logger.warning(f"Failed to load Telegram config: {e}")

        return chat_id

    def _register_default_commands(self):
        """Register default bot commands"""
        self.register_command("help", "Show available commands", self._handle_help)
        self.register_command("status", "Get system status", self._handle_status)
        self.register_command("trigger", "Trigger manual check", self._handle_trigger)

    def _register_search_commands(self):
        """Register search-related commands"""
        if self.search_available:
            self.register_command("search", "Search the internet", self._handle_search)
            self.register_command("ask", "Quick answer from web", self._handle_ask)
            logger.info("Search commands registered")

    def _register_browser_commands(self):
        """Register browser-based commands"""
        if self.browser_available:
            self.register_command("browse", "Browse URL with real browser", self._handle_browse)
            self.register_command("bs", "Browser search", self._handle_browser_search)

        # Browser agent commands (direct browser control)
        if self.browser_agent_available:
            self.register_command("browser", "Start browser agent", self._handle_browser_start)
            self.register_command("goto", "Navigate to URL", self._handle_browser_goto)
            self.register_command("click", "Click element", self._handle_browser_click)
            self.register_command("type", "Type text", self._handle_browser_type)
            self.register_command("screenshot", "Take screenshot", self._handle_browser_screenshot)
            self.register_command("extract", "Extract text from element", self._handle_browser_extract)
            self.register_command("eval", "Execute JavaScript", self._handle_browser_eval)
            self.register_command("binfo", "Browser info", self._handle_browser_info)
            self.register_command("bclose", "Close browser", self._handle_browser_close)
            logger.info("Browser agent commands registered")

    def register_command(self, name: str, description: str, handler: Callable):
        """Register a bot command"""
        self._commands[name] = TelegramCommand(name, description, handler)
        logger.debug(f"Registered command: /{name}")

    def _handle_help(self, args: List[str]) -> str:
        """Handle /help command"""
        lines = ["Available commands:"]
        for cmd in self._commands.values():
            lines.append(f"/{cmd.name} - {cmd.description}")
        return "\n".join(lines)

    def _handle_status(self, args: List[str]) -> str:
        """Handle /status command"""
        return "System is running"

    def _handle_trigger(self, args: List[str]) -> str:
        """Handle /trigger command"""
        return "Trigger executed"

    def _handle_search(self, args: List[str]) -> str:
        """Handle /search command - search the internet"""
        if not self.search_available:
            return "Search is not available. Please install required dependencies."

        if not args:
            return "Usage: /search <query>\nExample: /search python tutorial"

        query = " ".join(args)

        try:
            # Get search results
            response = self.search_engine.search(query, max_results=5)

            if not response.results:
                return f"No results found for: {query}"

            # Format results
            lines = [f"Results for: {query}\n"]

            for i, result in enumerate(response.results, 1):
                lines.append(f"{i}. {result.title}")
                lines.append(f"   {result.url}")
                if result.snippet:
                    snippet = result.snippet[:150] + "..." if len(result.snippet) > 150 else result.snippet
                    lines.append(f"   {snippet}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search error: {str(e)}"

    def _handle_ask(self, args: List[str]) -> str:
        """Handle /ask command - quick answer from web"""
        if not self.search_available:
            return "Search is not available. Please install required dependencies."

        if not args:
            return "Usage: /ask <question>\nExample: /ask what is python"

        query = " ".join(args)

        try:
            # Try quick answer first
            answer = self.search_engine.quick_answer(query)

            if answer:
                return f"Answer: {answer}"

            # Fall back to search
            response = self.search_engine.search(query, max_results=3)

            if not response.results:
                return f"No information found for: {query}"

            lines = [f"Here's what I found about: {query}\n"]

            for i, result in enumerate(response.results[:3], 1):
                if result.snippet:
                    lines.append(f"{i}. {result.snippet[:200]}...")
                    lines.append(f"   Source: {result.url}\n")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Ask error: {e}")
            return f"Error: {str(e)}"

    def _handle_browse(self, args: List[str]) -> str:
        """Handle /browse command - fetch URL using real browser"""
        if not self.browser_available:
            return "Browser not available. Install playwright: pip install playwright && playwright install chromium"

        if not args:
            return "Usage: /browse <url>\nExample: /browse https://example.com"

        url = " ".join(args)

        # Add https:// if missing
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        try:
            result = self.browser_fetcher.fetch(url)

            if result.success:
                # Extract text content (first 2000 chars of HTML stripped)
                import re
                text = re.sub(r'<[^>]+>', ' ', result.content)
                text = ' '.join(text.split())

                lines = [f"Page: {result.title}", f"URL: {result.url}\n"]
                lines.append("Content preview:")
                lines.append(text[:2000])

                return "\n".join(lines)
            else:
                return f"Failed to fetch: {result.error}"

        except Exception as e:
            logger.error(f"Browse error: {e}")
            return f"Error: {str(e)}"

    def _handle_browser_search(self, args: List[str]) -> str:
        """Handle /bs command - browser-based search (uses regular search for reliability)"""
        if not args:
            return "Usage: /bs <query>\nExample: /bs python tutorial"

        query = " ".join(args)

        # Use the regular search engine (works reliably)
        if self.search_available:
            try:
                response = self.search_engine.search(query, max_results=5)

                if not response.results:
                    return f"No results found for: {query}"

                lines = [f"Search results for: {query}\n"]

                for i, result in enumerate(response.results, 1):
                    lines.append(f"{i}. {result.title}")
                    lines.append(f"   {result.url}")
                    if result.snippet:
                        lines.append(f"   {result.snippet[:150]}...")
                    lines.append("")

                return "\n".join(lines)

            except Exception as e:
                logger.error(f"Search error: {e}")
                return f"Error: {str(e)}"

        return "Search not available"

    def _handle_browser_start(self, args: List[str]) -> str:
        """Handle /browser command - start browser agent"""
        if not self.browser_agent_available:
            return "Browser agent not available. Install playwright: pip install playwright && playwright install chromium"

        try:
            if self.browser_agent is None:
                headless = False  # Show browser for user to see
                self.browser_agent = get_browser_agent(headless=headless)
                info = self.browser_agent.get_page_info()
                return f"Browser started!\nURL: {info.get('url', 'blank')}\nTitle: {info.get('title', 'New tab')}"
            else:
                return "Browser already running. Use /goto, /click, /type, etc."
        except Exception as e:
            logger.error(f"Browser start error: {e}")
            return f"Error starting browser: {str(e)}"

    def _handle_browser_goto(self, args: List[str]) -> str:
        """Handle /goto command - navigate to URL"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if not args:
            return "Usage: /goto <url>\nExample: /goto https://google.com"

        url = " ".join(args)
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        try:
            result = self.browser_agent.navigate(url)
            if result.success:
                info = self.browser_agent.get_page_info()
                return f"Navigated to: {info.get('url')}\nTitle: {info.get('title')}"
            else:
                return f"Navigation failed: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_click(self, args: List[str]) -> str:
        """Handle /click command - click element"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if not args:
            return "Usage: /click <selector>\nExample: /click button#submit"

        selector = " ".join(args)

        try:
            result = self.browser_agent.click(selector)
            if result.success:
                info = self.browser_agent.get_page_info()
                return f"Clicked: {selector}\nCurrent URL: {info.get('url')}"
            else:
                return f"Click failed: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_type(self, args: List[str]) -> str:
        """Handle /type command - type text"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if len(args) < 2:
            return "Usage: /type <selector> <text>\nExample: /type input#search Python"

        selector = args[0]
        text = " ".join(args[1:])

        try:
            result = self.browser_agent.type(selector, text)
            if result.success:
                return f"Typed '{text}' into {selector}"
            else:
                return f"Type failed: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_screenshot(self, args: List[str]) -> str:
        """Handle /screenshot command - take screenshot"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        try:
            result = self.browser_agent.screenshot()
            if result.success and result.screenshot:
                # Return base64 for now - bot can send as photo
                return f"Screenshot taken! ({len(result.screenshot)} bytes base64)\nUse /browse command to view in browser"
            else:
                return f"Screenshot failed: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_extract(self, args: List[str]) -> str:
        """Handle /extract command - extract text"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if not args:
            return "Usage: /extract <selector>\nExample: /extract div.content"

        selector = " ".join(args)

        try:
            result = self.browser_agent.extract_text(selector)
            if result.success:
                texts = result.data
                if texts:
                    # Return first 10 matches
                    output = f"Found {len(texts)} elements:\n"
                    for i, text in enumerate(texts[:10], 1):
                        text = text.strip()[:200]
                        if text:
                            output += f"{i}. {text}\n"
                    return output
                else:
                    return f"No text found in: {selector}"
            else:
                return f"Extract failed: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_eval(self, args: List[str]) -> str:
        """Handle /eval command - execute JavaScript"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if not args:
            return "Usage: /eval <javascript>\nExample: /eval document.title"

        script = " ".join(args)

        try:
            result = self.browser_agent.evaluate(script)
            if result.success:
                data = str(result.data)[:500]
                return f"Result: {data}"
            else:
                return f"Eval failed: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_info(self, args: List[str]) -> str:
        """Handle /binfo command - browser info"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        try:
            info = self.browser_agent.get_page_info()
            return f"Browser Info:\nURL: {info.get('url')}\nTitle: {info.get('title')}\nActive: {info.get('initialized')}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_close(self, args: List[str]) -> str:
        """Handle /bclose command - close browser"""
        try:
            if self.browser_agent:
                self.browser_agent.stop()
                self.browser_agent = None
                return "Browser closed"
            return "Browser not running"
        except Exception as e:
            return f"Error: {str(e)}"

    def _make_request(self, method: str, data: Dict = None, files: Dict = None) -> Optional[Dict]:
        """Make API request with retry"""
        url = f"{self.api_url}/{method}"

        def _request():
            if files:
                return requests.post(url, data=data, files=files, timeout=30)
            else:
                return requests.post(url, json=data, timeout=10)

        try:
            response = self.executor.execute_with_retry(_request)
            if response and response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Telegram API error: {e}")

        return None

    def send_message(self, text: str, parse_mode: str = None) -> bool:
        """Send message to configured chat"""
        if not self.enabled:
            return False

        data = {
            "chat_id": self.chat_id,
            "text": text
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        result = self._make_request("sendMessage", data)
        if result:
            logger.debug(f"Message sent: {text[:50]}...")
            return True

        return False

    def send_photo(self, image_path: str, caption: str = None) -> bool:
        """Send photo to configured chat"""
        if not self.enabled:
            return False

        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption

        files = {"photo": open(image_path, "rb")}

        result = self._make_request("sendPhoto", data, files)
        files["photo"].close()

        if result:
            logger.debug(f"Photo sent: {image_path}")
            return True

        return False

    def send_document(self, file_path: str, caption: str = None) -> bool:
        """Send document to configured chat"""
        if not self.enabled:
            return False

        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption

        files = {"document": open(file_path, "rb")}

        result = self._make_request("sendDocument", data, files)
        files["document"].close()

        return bool(result)

    def get_updates(self, offset: int = 0, timeout: int = 30) -> List[Dict]:
        """Get updates from bot"""
        if not self.enabled:
            return []

        try:
            url = f"{self.api_url}/getUpdates"
            params = {"timeout": timeout, "offset": offset}
            response = requests.get(url, params=params, timeout=timeout + 5)

            if response.status_code == 200:
                return response.json().get("result", [])

        except Exception as e:
            logger.error(f"Get updates error: {e}")

        return []

    def handle_commands(self, text: str) -> str:
        """Handle text as commands"""
        text = text.strip()

        # Handle !! prefix
        if text.startswith("!!"):
            cmd = text[2:].strip().lower().split()
        else:
            cmd = text.strip().lower().split()

        if not cmd:
            return "Use /help for commands"

        command = cmd[0].replace("/", "")
        args = cmd[1:] if len(cmd) > 1 else []

        if command in self._commands:
            handler = self._commands[command].handler
            try:
                return handler(args)
            except Exception as e:
                logger.error(f"Command error: {e}")
                return f"Error: {str(e)}"

        return f"Unknown command: {command}. Use /help for available commands."

    def start_command_listener(self, callback: Callable = None):
        """Start polling for commands"""
        if not self.enabled:
            logger.warning("Telegram bot not enabled")
            return

        def listener():
            offset = 0
            logger.info("Telegram command listener started")

            while True:
                try:
                    updates = self.get_updates(offset)

                    for update in updates:
                        try:
                            offset = update.get("update_id", offset) + 1
                            message = update.get("message", {})
                            text = message.get("text", "")

                            if text:
                                # Validate chat_id
                                chat_id = str(message.get("chat", {}).get("id", ""))
                                if chat_id != self.chat_id:
                                    logger.warning(f"Unauthorized chat: {chat_id}")
                                    continue

                                logger.debug(f"Command received: {text}")
                                response = self.handle_commands(text)

                                if response:
                                    self.send_message(response)

                                if callback:
                                    callback(text, response)

                        except Exception as e:
                            logger.error(f"Update handling error: {e}")

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Listener error: {e}")
                    time.sleep(5)

        thread = threading.Thread(target=listener, daemon=True)
        thread.start()


# Export classes
__all__ = [
    "TelegramBot",
    "TelegramCommand",
]
