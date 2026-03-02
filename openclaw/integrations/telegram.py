"""Telegram bot integration"""

import os
import time
import json
import threading
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field

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
        self.MAX_HISTORY = 10
        self.history = {}

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
        self._browser_headless = True  # Run headless for Chrome Remote Desktop compatibility

        # Persistent Playwright browser for /imagine (keeps Google login cookies)
        self._imagine_playwright = None
        self._imagine_browser = None
        self._imagine_page = None
        self._imagine_ready = False

        # Agent system
        self._agent_system = None
        self._system_prompt = "You are a helpful, concise AI assistant. Reply thoughtfully. Feel free to use Markdown formatting."

        # Advanced features state
        self._active_reminders: Dict[str, Dict] = {}
        self._reminder_counter = 0
        self._whisper_pipeline = None  # Lazy-loaded transformers whisper

        # Thread safety
        self._history_lock = threading.Lock()
        self._browser_semaphore = threading.Semaphore(3)  # Max 3 concurrent browser sessions

        # Typing action backoff (for tests)
        self._typing_failures = 0
        self._typing_suppressed = False
        self._typing_backoff_until = 0.0

        # Command registration
        self._command_registration_degraded = False
        self._max_bot_commands = 100

        # AskUser tool: pending responses from inline keyboard buttons
        self._pending_responses: Dict[str, Dict] = {}  # {request_id: {"event": Event, "response": None}}
        self._pending_lock = threading.Lock()

        if self.enabled:
            self._register_default_commands()
            self._register_search_commands()
            self._register_browser_commands()
            self._register_agent_commands()
            self._register_interbot_commands()
            self._bootstrap_agents()

    def send_typing_action(self) -> bool:
        """Send typing action to Telegram (with backoff on failures)."""
        import time
        if not self.enabled or not self.api_url:
            return False

        # Check if typing is suppressed due to backoff
        if self._typing_suppressed and time.time() < self._typing_backoff_until:
            return False

        try:
            response = requests.post(
                f"{self.api_url}/sendChatAction",
                json={"chat_id": self.chat_id, "action": "typing"},
                timeout=10
            )
            if response.ok:
                self._typing_failures = 0
                self._typing_suppressed = False
                return True
            else:
                self._typing_failures += 1
                if self._typing_failures >= 3:
                    self._typing_suppressed = True
                    self._typing_backoff_until = time.time() + 60
                return False
        except Exception:
            self._typing_failures += 1
            if self._typing_failures >= 3:
                self._typing_suppressed = True
                self._typing_backoff_until = time.time() + 60
            return False

    def _handle_callback_query(self, query: Dict) -> Optional[str]:
        """Handle callback query from inline keyboard buttons."""
        if not self.enabled:
            return None

        user_id = query.get("from", {}).get("id")
        data = query.get("data", "")
        message = query.get("message", {})
        chat_id = message.get("chat", {}).get("id")

        # Answer the callback query first
        try:
            requests.post(
                f"{self.api_url}/answerCallbackQuery",
                json={"callback_query_id": query.get("id")},
                timeout=10
            )
        except Exception:
            pass

        # Check if this is an AskUser response (format: "askuser:<request_id>:<choice>")
        if data.startswith("askuser:"):
            parts = data.split(":", 2)
            if len(parts) == 3:
                request_id = parts[1]
                choice = parts[2]
                with self._pending_lock:
                    if request_id in self._pending_responses:
                        self._pending_responses[request_id]["response"] = choice
                        self._pending_responses[request_id]["event"].set()
                        logger.info(f"AskUser response received: {request_id} -> {choice}")
                        # Update the message to show what was selected
                        try:
                            msg_id = message.get("message_id")
                            if msg_id:
                                original_text = message.get("text", "")
                                self.edit_message(
                                    str(chat_id), msg_id,
                                    f"{original_text}\n\n✅ Selected: {choice}"
                                )
                        except Exception:
                            pass
                        return None

        # If data starts with /, treat as command
        if data.startswith("/"):
            command_parts = data[1:].split()
            command = command_parts[0] if command_parts else ""
            args = command_parts[1:] if len(command_parts) > 1 else []

            if command in self._command_callbacks:
                return self._command_callbacks[command](args)

        return None

    def send_inline_keyboard(self, text: str, options: list, request_id: str, chat_id: str = None) -> bool:
        """Send a message with inline keyboard buttons for AskUser tool."""
        target_chat = chat_id or self.chat_id
        if not self.api_url or not target_chat:
            return False

        # Build keyboard: each option is a row with one button
        keyboard = []
        for opt in options:
            callback_data = f"askuser:{request_id}:{opt}"
            # Telegram limits callback_data to 64 bytes
            if len(callback_data.encode('utf-8')) > 64:
                callback_data = callback_data[:64]
            keyboard.append([{"text": opt, "callback_data": callback_data}])

        data = {
            "chat_id": target_chat,
            "text": text,
            "reply_markup": json.dumps({"inline_keyboard": keyboard}),
        }
        try:
            result = self._make_request("sendMessage", data)
            return bool(result and result.get("ok"))
        except Exception as e:
            logger.error(f"Failed to send inline keyboard: {e}")
            return False

    def wait_for_user_response(self, request_id: str, question: str, options: list, timeout: float = 120.0) -> str:
        """Send an inline keyboard and block until the user responds or timeout.
        
        This is the core mechanism for the AskUser tool. It:
        1. Creates a threading.Event for this request
        2. Sends the inline keyboard to Telegram
        3. Blocks until the user taps a button or timeout expires
        4. Returns the selected option or 'timeout'
        """
        event = threading.Event()
        with self._pending_lock:
            self._pending_responses[request_id] = {"event": event, "response": None}

        # Send the keyboard
        sent = self.send_inline_keyboard(
            text=f"🤔 Agent needs your input:\n\n{question}",
            options=options,
            request_id=request_id
        )

        if not sent:
            with self._pending_lock:
                self._pending_responses.pop(request_id, None)
            return f"(Failed to send question to user. Proceeding with best guess.)"

        # Block until response or timeout
        event.wait(timeout=timeout)

        with self._pending_lock:
            result = self._pending_responses.pop(request_id, {})

        response = result.get("response")
        if response is None:
            return "(User did not respond in time. Proceeding with best judgment.)"

        return response

    def _extract_reply_media_context(self, message: Dict) -> Optional[str]:
        """Extract context from replied-to media (photo, document, etc)."""
        reply = message.get("reply_to_message", {})
        if not reply:
            return None

        # Check for photo
        if "photo" in reply:
            photo = reply["photo"]
            if photo:
                file_id = photo[0].get("file_id", "unknown")
                caption = reply.get("text", "")
                return f"Replied to image {file_id[:8]}... {caption}"

        # Check for document
        if "document" in reply:
            doc = reply["document"]
            file_name = doc.get("file_name", "document")
            return f"Replied to document: {file_name}"

        # Check for video
        if "video" in reply:
            return "Replied to video"

        # Check for voice/audio
        if "voice" in reply or "audio" in reply:
            return "Replied to audio"

        return None

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
        self.register_command("code", "Execute code in sandbox", self._handle_code)
        self.register_command("remind", "Set a timed reminder", self._handle_remind)
        self.register_command("alert", "Manage active reminders", self._handle_alert)

    def _register_search_commands(self):
        """Register search-related commands"""
        if self.search_available:
            self.register_command("search", "Search the internet", self._handle_search)
            self.register_command("ask", "Quick answer from web", self._get_ai_response) # Changed to _get_ai_response
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
            self.register_command("click", "Click element by selector", self._handle_browser_click)
            self.register_command("clicktext", "Click button/link by text", self._handle_browser_click_text)
            self.register_command("type", "Type text by selector", self._handle_browser_type)
            self.register_command("input", "Type in first input field", self._handle_browser_input)
            self.register_command("submit", "Click submit button", self._handle_browser_submit)
            self.register_command("screenshot", "Take screenshot", self._handle_browser_screenshot)
            self.register_command("extract", "Extract text from element", self._handle_browser_extract)
            self.register_command("extractall", "Extract all text from page", self._handle_browser_extract_all)
            self.register_command("eval", "Execute JavaScript", self._handle_browser_eval)
            self.register_command("binfo", "Browser info", self._handle_browser_info)
            self.register_command("bclose", "Close browser", self._handle_browser_close)
            logger.info("Browser agent commands registered")

        # Image generation via Gemini browser automation
        self.register_command("imagine", "Generate image via Gemini", self._handle_imagine)
        self.register_command("imglogin", "Login to Google for image gen", self._handle_imagine_login)
        self.register_command("deepresearch", "Deep research via Gemini Pro", self._handle_deep_research)

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

    def _get_minimax_api_key(self) -> Optional[str]:
        """Get MiniMax API key from environment or credentials file."""
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            key_path = os.path.expanduser("~/.openclaw/credentials/keys.json")
            if os.path.exists(key_path):
                try:
                    with open(key_path) as f:
                        d = json.load(f)
                    api_key = d.get("providers", {}).get("minimax", {}).get("default", {}).get("api_key")
                except Exception:
                    pass
        return api_key

    def _get_ai_response(self, text: str) -> str:
        api_key = self._get_minimax_api_key()
        if not api_key:
            return "AI API key not configured in environment or credentials."

        try:
            url = "https://api.minimax.io/anthropic/v1/messages"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            data = {"model": "MiniMax-M2.5-Lightning", "max_tokens": 1024, "stream": False, "system": self._system_prompt, "messages": [{"role": "user", "content": text}]}
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                content = result.get('content', [])
                for block in content:
                    if block.get('type') == 'text':
                        return block.get('text', 'No text response.')
            return f"AI Error: {response.text[:100]}"
        except Exception as e:
            return f"AI Request failed: {e}"

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
                headless = True  # Run headless for Chrome Remote Desktop compatibility
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
        """Handle /screenshot command - take screenshot and send to Telegram"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        try:
            result = self.browser_agent.screenshot()
            if result.success and result.screenshot:
                # Save screenshot to temp file
                import base64
                import tempfile
                import os

                # Decode base64
                img_data = base64.b64decode(result.screenshot)

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    f.write(img_data)
                    temp_path = f.name

                # Send to Telegram
                info = self.browser_agent.get_page_info()
                caption = f"Screen: {info.get('title', 'Unknown')}"

                sent = self.send_photo(temp_path, caption)

                # Clean up
                os.unlink(temp_path)

                if sent:
                    return f"Screenshot sent!"
                else:
                    return f"Screenshot taken but failed to send"
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

    def _handle_browser_click_text(self, args: List[str]) -> str:
        """Handle /clicktext command - click element containing text"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if not args:
            return "Usage: /clicktext <text>\nExample: /clicktext Submit"

        text = " ".join(args)

        try:
            # Find element containing the text and click it
            # Use json.dumps to safely escape user text for JS (prevents injection)
            safe_text = json.dumps(text)
            script = f"""
            function() {{
                let target = {safe_text}.toLowerCase();
                // Try buttons first
                let btns = document.querySelectorAll('button, input[type="submit"], a[role="button"]');
                for (let btn of btns) {{
                    if (btn.innerText.toLowerCase().includes(target) ||
                        btn.value?.toLowerCase().includes(target)) {{
                        btn.click();
                        return 'clicked:' + btn.tagName;
                    }}
                }}
                // Try links
                let links = document.querySelectorAll('a');
                for (let link of links) {{
                    if (link.innerText.toLowerCase().includes(target)) {{
                        link.click();
                        return 'clicked:a';
                    }}
                }}
                return 'not_found';
            }}()
            """
            result = self.browser_agent.evaluate(script)
            if result.success and result.data != 'not_found':
                return f"Clicked element containing '{text}'"
            return f"Could not find element with text '{text}'"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_input(self, args: List[str]) -> str:
        """Handle /input command - type in first available input field"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        if not args:
            return "Usage: /input <text>\nExample: /input Hello World"

        text = " ".join(args)

        try:
            # Use json.dumps to safely escape user text for JS (prevents injection)
            safe_text = json.dumps(text)
            script = f"""
            function() {{
                let val = {safe_text};
                let inputs = document.querySelectorAll('input[type="text"], input[type="search"], textarea');
                for (let input of inputs) {{
                    if (!input.disabled && !input.readOnly) {{
                        input.value = val;
                        input.dispatchEvent(new Event('input', {{bubbles: true}}));
                        return 'typed:' + input.tagName;
                    }}
                }}
                return 'not_found';
            }}()
            """
            result = self.browser_agent.evaluate(script)
            if result.success and result.data != 'not_found':
                return f"Typed '{text}' into input field"
            return "Could not find input field"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_submit(self, args: List[str]) -> str:
        """Handle /submit command - click submit button or press Enter"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        try:
            script = """
            function() {
                // Try button type=submit
                let btns = document.querySelectorAll('button[type="submit"], input[type="submit"]');
                for (let btn of btns) {
                    btn.click();
                    return 'clicked_submit';
                }
                // Try form submit
                let forms = document.querySelectorAll('form');
                for (let form of forms) {
                    form.submit();
                    return 'form_submitted';
                }
                return 'not_found';
            }()
            """
            result = self.browser_agent.evaluate(script)
            if result.success and result.data != 'not_found':
                return "Submitted form"
            return "Could not find submit button"
        except Exception as e:
            return f"Error: {str(e)}"

    def _handle_browser_extract_all(self, args: List[str]) -> str:
        """Handle /extractall command - extract all visible text"""
        if not self.browser_agent:
            return "Browser not started. Use /browser first"

        try:
            script = """
            function() {
                let text = document.body.innerText;
                // Clean up whitespace
                text = text.replace(/\\s+/g, ' ').trim();
                return text.substring(0, 3000);
            }()
            """
            result = self.browser_agent.evaluate(script)
            if result.success:
                return f"Page text:\n{result.data}"
            return f"Error: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _close_imagine_browser(self):
        """Safely close any existing imagine browser and playwright instance (used by /imglogin)."""
        try:
            if self._imagine_browser:
                self._imagine_browser.close()
        except Exception:
            pass
        try:
            if self._imagine_playwright:
                self._imagine_playwright.stop()
        except Exception:
            pass
        self._imagine_browser = None
        self._imagine_page = None
        self._imagine_playwright = None
        self._imagine_ready = False

    def _get_chrome_path(self):
        """Find system Chrome binary."""
        import os
        for p in ['/usr/bin/google-chrome-stable', '/usr/bin/google-chrome', '/usr/bin/chromium']:
            if os.path.exists(p):
                return p
        return None

    def _get_browser_launch_args(self):
        """Common stealth launch args for all browser sessions."""
        return [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-infobars',
            '--exclude-switches=enable-automation',
        ]

    def _ensure_imagine_browser(self, headless: bool = True) -> bool:
        """Start a browser using the REAL persistent profile (only for /imglogin).
        This locks the profile dir, so /imagine and /deepresearch should use
        _create_browser_session() instead for parallel-safe operation."""
        if self._imagine_page and self._imagine_ready:
            try:
                self._imagine_page.title()
                return True
            except Exception:
                self._close_imagine_browser()

        self._close_imagine_browser()

        try:
            from playwright.sync_api import sync_playwright
            import os

            user_data_dir = os.path.expanduser("~/.openclaw/browser_data")
            os.makedirs(user_data_dir, exist_ok=True)

            self._imagine_playwright = sync_playwright().start()
            chrome_path = self._get_chrome_path()

            launch_kwargs = {
                'headless': headless,
                'args': self._get_browser_launch_args(),
                'viewport': {'width': 1280, 'height': 720},
                'ignore_https_errors': True,
                'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            }
            if chrome_path:
                launch_kwargs['executable_path'] = chrome_path

            self._imagine_browser = self._imagine_playwright.chromium.launch_persistent_context(
                user_data_dir, **launch_kwargs
            )
            self._imagine_page = self._imagine_browser.pages[0] if self._imagine_browser.pages else self._imagine_browser.new_page()
            self._imagine_ready = True
            logger.info(f"Login browser started (headless={headless})")
            return True
        except Exception as e:
            logger.error(f"Failed to start login browser: {e}")
            self._imagine_ready = False
            self._last_imagine_error = str(e)
            return False

    def _create_browser_session(self, headless: bool = True):
        """Create an INDEPENDENT browser session by copying the login profile
        to a temp directory. This allows parallel /imagine and /deepresearch.
        Returns (playwright, browser, page, temp_dir) or raises on failure."""
        from playwright.sync_api import sync_playwright
        import os, shutil, tempfile

        master_profile = os.path.expanduser("~/.openclaw/browser_data")
        if not os.path.exists(master_profile):
            raise RuntimeError("No login profile found. Run /imglogin first.")

        # Copy login cookies to a unique temp dir
        temp_dir = tempfile.mkdtemp(prefix="openclaw_browser_")
        shutil.copytree(master_profile, os.path.join(temp_dir, "profile"),
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("SingletonLock", "SingletonCookie", "SingletonSocket"))
        profile_dir = os.path.join(temp_dir, "profile")

        pw = sync_playwright().start()
        chrome_path = self._get_chrome_path()

        launch_kwargs = {
            'headless': headless,
            'args': self._get_browser_launch_args(),
            'viewport': {'width': 1280, 'height': 720},
            'ignore_https_errors': True,
            'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }
        if chrome_path:
            launch_kwargs['executable_path'] = chrome_path

        browser = pw.chromium.launch_persistent_context(profile_dir, **launch_kwargs)
        page = browser.pages[0] if browser.pages else browser.new_page()
        logger.info(f"Independent browser session created (headless={headless}) in {temp_dir}")
        return pw, browser, page, temp_dir

    def _close_browser_session(self, pw, browser, temp_dir):
        """Close an independent browser session and clean up its temp dir."""
        import shutil
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        try:
            if pw:
                pw.stop()
        except Exception:
            pass
        try:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _handle_imagine_login(self, args: List[str]) -> str:
        """Handle /imglogin - open Gemini in a VISIBLE browser for one-time Google sign-in.
        Opens a headed browser, waits for the user to log in, then closes it
        so /imagine and /deepresearch can use the saved cookies."""
        try:
            if not self._ensure_imagine_browser(headless=False):
                err = getattr(self, '_last_imagine_error', 'Unknown Error')
                return f"Failed to start browser. Is Playwright installed?\n\nError details: {err}"

            self._imagine_page.goto("https://gemini.google.com", wait_until="domcontentloaded", timeout=30000)
            import time
            time.sleep(3)
            title = self._imagine_page.title()

            # Give user time to log in (120 seconds), then close the headed browser
            # so /imagine and /deepresearch can launch their own headless browser
            login_wait = 120
            self.send_message(
                f"🌐 Browser opened with Gemini!\n"
                f"Page: {title}\n\n"
                f"Please log into your Google account in the browser window.\n"
                f"You have {login_wait} seconds to complete login.\n"
                f"Your session will be saved permanently after login.\n"
                f"The browser will close automatically, then use /imagine or /deepresearch."
            )

            time.sleep(login_wait)

            # Close the headed browser to release the persistent context lock
            self._close_imagine_browser()
            logger.info("Login browser closed, cookies saved to persistent profile")

            return (
                f"✅ Login session saved!\n"
                f"The browser has been closed.\n"
                f"You can now use /imagine <prompt> to generate images\n"
                f"and /deepresearch <topic> for deep research."
            )
        except Exception as e:
            self._close_imagine_browser()
            return f"Error: {str(e)}"

    def _handle_imagine(self, args: List[str]) -> str:
        """Handle /imagine - generate an image using Gemini via Playwright.
        Uses an independent browser session for parallel-safe operation."""
        if not args:
            return "Usage: /imagine <description>\nExample: /imagine a cute cat wearing a hat"

        prompt = " ".join(args)
        pw = browser = page = temp_dir = None

        try:
            # Create independent browser session (copies login cookies)
            try:
                pw, browser, page, temp_dir = self._create_browser_session(headless=True)
            except Exception as e:
                return f"Failed to start browser. Run /imglogin first to set up Google login.\nError: {str(e)}"

            import time, base64, tempfile, os

            # Navigate to Gemini
            page.goto("https://gemini.google.com", wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)

            # Check if we're logged in by looking for the prompt input
            input_selectors = [
                'div[contenteditable="true"]',
                'rich-textarea div[contenteditable="true"]',
                'textarea',
                '.ql-editor',
                '[aria-label*="prompt"]',
                '[aria-label*="Enter"]',
            ]
            input_el = None
            for sel in input_selectors:
                try:
                    page.wait_for_selector(sel, timeout=5000)
                    input_el = sel
                    break
                except Exception:
                    continue

            if not input_el:
                return "Could not find Gemini input. Please run /imglogin to sign into Google first."

            # Type the image generation prompt
            image_prompt = f"Draw a picture of: {prompt}"
            try:
                page.click(input_el)
                time.sleep(0.3)
                page.keyboard.type(image_prompt, delay=30)
                time.sleep(0.5)
                page.keyboard.press("Enter")
            except Exception as e:
                return f"Failed to type prompt: {str(e)}"

            # Smart wait: poll for the generated image
            logger.info(f"Waiting for Gemini to generate image for: {prompt}")
            max_wait = 90
            poll_interval = 5
            elapsed = 0
            img_data = None

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval
                logger.info(f"Polling for image... ({elapsed}s / {max_wait}s)")

                img_data = page.evaluate("""
                () => {
                    let imgs = document.querySelectorAll('img');
                    for (let i = imgs.length - 1; i >= 0; i--) {
                        let img = imgs[i];
                        if (img.naturalWidth > 200 && img.naturalHeight > 200
                            && !img.src.includes('avatar')
                            && !img.src.includes('icon')
                            && !img.src.includes('logo')
                            && !img.src.includes('profile')
                            && !img.src.includes('favicon')) {
                            try {
                                let canvas = document.createElement('canvas');
                                canvas.width = img.naturalWidth;
                                canvas.height = img.naturalHeight;
                                let ctx = canvas.getContext('2d');
                                ctx.drawImage(img, 0, 0);
                                return canvas.toDataURL('image/png').split(',')[1];
                            } catch(e) {
                                return 'URL:' + img.src;
                            }
                        }
                    }
                    return null;
                }
                """)

                if img_data:
                    logger.info(f"Image found after {elapsed}s!")
                    break

            # Process the result
            try:
                if not img_data:
                    screenshot_bytes = page.screenshot()
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        f.write(screenshot_bytes)
                        temp_path = f.name
                    self.send_photo(temp_path, f"🎨 Gemini response for: {prompt}\n(Screenshot - image generation may still be in progress)")
                    os.unlink(temp_path)
                    return "Sent screenshot of Gemini's response. Image may still be generating — try again in a moment."

                if img_data.startswith('URL:'):
                    img_url = img_data[4:]
                    try:
                        new_page = browser.new_page()
                        new_page.goto(img_url, wait_until='load', timeout=15000)
                        time.sleep(1)
                        img_bytes = new_page.screenshot()
                        new_page.close()
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                            f.write(img_bytes)
                            temp_path = f.name
                    except Exception as dl_err:
                        logger.error(f"Playwright download failed: {dl_err}, trying requests")
                        import requests as req
                        r = req.get(img_url, timeout=15)
                        if r.status_code == 200:
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                                f.write(r.content)
                                temp_path = f.name
                        else:
                            return f"Failed to download image from Gemini (HTTP {r.status_code})"
                else:
                    raw = base64.b64decode(img_data)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        f.write(raw)
                        temp_path = f.name

                self.send_photo(temp_path, f"🎨 {prompt}")
                os.unlink(temp_path)
                return f"Image generated and sent! ✨"

            except Exception as e:
                logger.error(f"Image extraction error: {e}")
                try:
                    screenshot_bytes = page.screenshot()
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        f.write(screenshot_bytes)
                        temp_path = f.name
                    self.send_photo(temp_path, f"🎨 {prompt}\n(Screenshot fallback)")
                    os.unlink(temp_path)
                    return "Sent screenshot (image extraction failed)."
                except Exception:
                    return f"Image generation error: {str(e)}"

        except Exception as e:
            logger.error(f"Imagine error: {e}")
            return f"Error: {str(e)}"
        finally:
            self._close_browser_session(pw, browser, temp_dir)


    def _handle_deep_research(self, args: List[str]) -> str:
        """Handle /deepresearch - automate Gemini Deep Research with Pro 3.1 model."""
        if not args:
            return "Usage: /deepresearch <topic>\nExample: /deepresearch Impact of AI on healthcare in 2026"

        topic = " ".join(args)

        pw = browser = page = temp_dir = None

        try:
            # Create independent browser session (copies login cookies)
            try:
                pw, browser, page, temp_dir = self._create_browser_session(headless=True)
            except Exception as e:
                return f"Failed to start browser. Run /imglogin first to set up Google login.\nError: {str(e)}"

            import time, tempfile, os

            # Navigate to Gemini
            page.goto("https://gemini.google.com", wait_until="domcontentloaded", timeout=30000)
            time.sleep(4)

            # Find and click the input area
            input_selectors = [
                'div[contenteditable="true"]',
                'rich-textarea div[contenteditable="true"]',
                'textarea',
                '.ql-editor',
                '[aria-label*="prompt"]',
                '[aria-label*="Enter"]',
            ]
            input_el = None
            for sel in input_selectors:
                try:
                    page.wait_for_selector(sel, timeout=5000)
                    input_el = sel
                    break
                except Exception:
                    continue

            if not input_el:
                return "Could not find Gemini input. Please run /imglogin to sign into Google first."

            # Type the research topic
            page.click(input_el)
            time.sleep(0.3)
            page.keyboard.type(topic, delay=20)
            time.sleep(1)

            # Step 1: Click "Tools" button to open the dropdown menu
            logger.info("Looking for Tools button...")
            tools_clicked = False
            tools_selectors = [
                'button:has-text("Tools")',
                '[aria-label*="Tools"]',
                '[aria-label*="tools"]',
                'text=Tools',
            ]
            for sel in tools_selectors:
                try:
                    page.click(sel, timeout=5000)
                    tools_clicked = True
                    logger.info(f"Clicked Tools via: {sel}")
                    break
                except Exception:
                    continue

            if not tools_clicked:
                # JavaScript fallback for Tools
                tools_clicked = page.evaluate("""
                () => {
                    let els = document.querySelectorAll('button, [role="button"], [role="tab"], span, div');
                    for (let el of els) {
                        let txt = (el.innerText || '').trim().toLowerCase();
                        if (txt === 'tools' || txt.includes('tools')) {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
                """)

            if not tools_clicked:
                return "Could not find the 'Tools' button in Gemini."

            time.sleep(2)  # Wait for dropdown menu to expand

            # Step 2: Click "Deep Research" from the Tools dropdown
            logger.info("Looking for Deep Research in Tools menu...")
            dr_clicked = False
            dr_selectors = [
                'button:has-text("Deep Research")',
                '[aria-label*="Deep Research"]',
                '[role="menuitem"]:has-text("Deep Research")',
                'text=Deep Research',
            ]
            for sel in dr_selectors:
                try:
                    page.click(sel, timeout=5000)
                    dr_clicked = True
                    logger.info(f"Clicked Deep Research via: {sel}")
                    break
                except Exception:
                    continue

            if not dr_clicked:
                # JavaScript fallback for Deep Research
                dr_clicked = page.evaluate("""
                () => {
                    let els = document.querySelectorAll('button, [role="button"], [role="menuitem"], [role="option"], span, div, li');
                    for (let el of els) {
                        if (el.innerText && el.innerText.trim().toLowerCase().includes('deep research')) {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
                """)

            if not dr_clicked:
                return "Could not find 'Deep Research' in the Tools menu. Make sure your account has access."

            time.sleep(2)

            # Step 3: Click the model selector dropdown (shows Fast/Thinking/Pro)
            logger.info("Looking for model selector dropdown...")
            model_btn_clicked = False

            # The model dropdown usually shows the current model name (e.g. "Flash", "Thinking")
            model_btn_selectors = [
                'button:has-text("Flash")',
                'button:has-text("Thinking")',
                'button:has-text("Pro")',
                '[aria-label*="model"]',
                '[aria-label*="Model"]',
                'button:has-text("2.5")',
                'button:has-text("1.5")',
            ]
            for sel in model_btn_selectors:
                try:
                    page.click(sel, timeout=3000)
                    model_btn_clicked = True
                    logger.info(f"Clicked model dropdown via: {sel}")
                    break
                except Exception:
                    continue

            if not model_btn_clicked:
                # JavaScript fallback - click any button containing model-like text
                model_btn_clicked = page.evaluate("""
                () => {
                    let els = document.querySelectorAll('button, [role="button"], [role="listbox"], [role="combobox"]');
                    for (let el of els) {
                        let txt = (el.innerText || '').toLowerCase();
                        if (txt.includes('flash') || txt.includes('thinking') || txt.includes('pro') || txt.includes('model') || txt.includes('2.5') || txt.includes('1.5')) {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
                """)

            if model_btn_clicked:
                time.sleep(2)  # Wait for dropdown to expand

                # Step 4: Select "Pro" from the model dropdown
                logger.info("Selecting Pro model from dropdown...")
                pro_clicked = False
                pro_selectors = [
                    '[role="menuitem"]:has-text("Pro")',
                    '[role="option"]:has-text("Pro")',
                    'button:has-text("Pro")',
                    'text=Pro',
                ]
                for sel in pro_selectors:
                    try:
                        page.click(sel, timeout=3000)
                        pro_clicked = True
                        logger.info(f"Selected Pro model via: {sel}")
                        break
                    except Exception:
                        continue

                if not pro_clicked:
                    # JavaScript fallback
                    page.evaluate("""
                    () => {
                        let els = document.querySelectorAll('[role="menuitem"], [role="option"], button, li, span, div');
                        for (let el of els) {
                            let txt = (el.innerText || '').trim().toLowerCase();
                            if (txt === 'pro' || txt.includes('pro')) {
                                el.click();
                                return true;
                            }
                        }
                        return false;
                    }
                    """)
            else:
                logger.warning("Could not find model selector dropdown, proceeding with default model")

            time.sleep(1)

            # Step 5: Click the Send button
            logger.info("Looking for Send button...")
            send_clicked = False
            send_selectors = [
                'button[aria-label*="Send"]',
                'button[aria-label*="send"]',
                'button[aria-label*="Submit"]',
                'button:has-text("Send")',
                '[data-tooltip*="Send"]',
                'button.send-button',
                'button[mattooltip*="Send"]',
            ]
            for sel in send_selectors:
                try:
                    page.click(sel, timeout=3000)
                    send_clicked = True
                    logger.info(f"Clicked Send via: {sel}")
                    break
                except Exception:
                    continue

            if not send_clicked:
                # JavaScript fallback - look for send/submit buttons
                send_clicked = page.evaluate("""
                () => {
                    // Look for buttons with send-related attributes or icons
                    let btns = document.querySelectorAll('button, [role="button"]');
                    for (let btn of btns) {
                        let label = (btn.getAttribute('aria-label') || '').toLowerCase();
                        let tooltip = (btn.getAttribute('data-tooltip') || '').toLowerCase();
                        let mattooltip = (btn.getAttribute('mattooltip') || '').toLowerCase();
                        let txt = (btn.innerText || '').toLowerCase().trim();
                        // Match send/submit buttons
                        if (label.includes('send') || label.includes('submit') ||
                            tooltip.includes('send') || mattooltip.includes('send') ||
                            txt === 'send' || txt === 'submit') {
                            btn.click();
                            return true;
                        }
                        // Match buttons with path/SVG arrow icon (common for send buttons)
                        let svg = btn.querySelector('svg, mat-icon, .material-icons');
                        if (svg && (btn.closest('.input-area, .prompt-area, .chat-input, rich-textarea') ||
                            btn.closest('[class*="input"], [class*="prompt"]'))) {
                            // This is likely the send button near the input
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }
                """)

            if not send_clicked:
                # Last resort: try Enter key
                try:
                    page.keyboard.press("Enter")
                    logger.info("Used Enter key as fallback for Send")
                except Exception:
                    pass

            time.sleep(3)

            # Wait for the research plan to appear, then click "Start Research"
            logger.info("Waiting for research plan to appear...")
            start_clicked = False
            plan_wait = 60  # Wait up to 60s for the plan
            plan_elapsed = 0

            while plan_elapsed < plan_wait and not start_clicked:
                time.sleep(5)
                plan_elapsed += 5

                start_selectors = [
                    'button:has-text("Start Research")',
                    'button:has-text("Start research")',
                    '[aria-label*="Start"]',
                    'text=Start Research',
                    'text=Start research',
                ]
                for sel in start_selectors:
                    try:
                        page.click(sel, timeout=3000)
                        start_clicked = True
                        logger.info(f"Clicked Start Research via: {sel}")
                        break
                    except Exception:
                        continue

                if not start_clicked:
                    # JavaScript fallback
                    start_clicked = page.evaluate("""
                    () => {
                        let els = document.querySelectorAll('button, [role="button"]');
                        for (let el of els) {
                            let txt = (el.innerText || '').toLowerCase();
                            if (txt.includes('start research') || txt.includes('start')) {
                                el.click();
                                return true;
                            }
                        }
                        return false;
                    }
                    """)

            if not start_clicked:
                # Take screenshot to show what happened
                screenshot_bytes = page.screenshot()
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    f.write(screenshot_bytes)
                    temp_path = f.name
                self.send_photo(temp_path, f"Could not find 'Start Research' button for: {topic}")
                os.unlink(temp_path)
                return "Could not auto-click 'Start Research'. Sent screenshot of current state."

            logger.info("Research started! Capturing baseline before polling...")

            # Capture baseline: the current page text length (includes the research plan)
            time.sleep(5)
            baseline_length = page.evaluate("""
            () => {
                return document.body.innerText.length;
            }
            """) or 0
            logger.info(f"Baseline text length: {baseline_length} chars")

            # Minimum wait: Deep Research always takes at least 5 minutes
            min_wait = 5 * 60  # 5 minutes
            logger.info(f"Waiting minimum {min_wait // 60} minutes before checking...")
            time.sleep(min_wait)

            # Now poll for completion (up to 25 more minutes)
            max_extra_wait = 25 * 60  # 25 more minutes
            poll_interval = 30  # Check every 30 seconds
            research_elapsed = min_wait

            while (research_elapsed - min_wait) < max_extra_wait:
                time.sleep(poll_interval)
                research_elapsed += poll_interval
                minutes = research_elapsed // 60
                seconds = research_elapsed % 60
                logger.info(f"Research in progress... ({minutes}m {seconds}s elapsed)")

                # Completion check: text must have grown SIGNIFICANTLY beyond the plan
                is_complete = page.evaluate(f"""
                () => {{
                    let currentLength = document.body.innerText.length;
                    let growth = currentLength - {baseline_length};

                    // Research report should add at least 3000 chars beyond the plan
                    if (growth < 3000) return false;

                    // Check no loading/researching indicators are active
                    let loadingIndicators = document.querySelectorAll(
                        '.loading, .spinner, [aria-label*="loading"], [aria-label*="Generating"], ' +
                        '[aria-label*="Researching"], [aria-label*="thinking"], .thinking-indicator, ' +
                        '[aria-busy="true"], .progress-indicator'
                    );
                    if (loadingIndicators.length > 0) return false;

                    // Looks like a complete report
                    return true;
                }}
                """)

                if is_complete:
                    logger.info(f"Research completed after {minutes}m {seconds}s! Text grew beyond baseline.")
                    break

            # Give a final buffer for rendering
            time.sleep(5)

            # Extract the research write-up
            write_up = page.evaluate("""
            () => {
                // Get the last/longest response
                let responses = document.querySelectorAll('model-response, .response-container, .message-content, [data-message-id]');
                let best = '';
                for (let r of responses) {
                    let text = r.innerText || '';
                    if (text.length > best.length) {
                        best = text;
                    }
                }
                // If nothing found, get the whole page body
                if (best.length < 200) {
                    best = document.body.innerText;
                }
                return best;
            }
            """)

            if not write_up or len(write_up) < 100:
                # Fallback: screenshot
                screenshot_bytes = page.screenshot(full_page=True)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    f.write(screenshot_bytes)
                    temp_path = f.name
                self.send_photo(temp_path, f"📄 Research result for: {topic}\n(Could not extract text)")
                os.unlink(temp_path)
                return "Research may be complete. Sent screenshot (text extraction failed)."

            # Convert to PDF using weasyprint
            logger.info(f"Converting research to PDF ({len(write_up)} chars)...")
            try:
                from weasyprint import HTML as WeasyHTML
                import markdown as md

                # Convert plain text to styled HTML
                # Try to detect if it's markdown-ish
                html_body = md.markdown(write_up, extensions=['tables', 'fenced_code'])

                styled_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{
                            font-family: 'Georgia', 'Times New Roman', serif;
                            max-width: 800px;
                            margin: 40px auto;
                            padding: 20px 40px;
                            line-height: 1.8;
                            color: #1a1a1a;
                            font-size: 14px;
                        }}
                        h1 {{
                            font-size: 28px;
                            color: #1a73e8;
                            border-bottom: 3px solid #1a73e8;
                            padding-bottom: 10px;
                            margin-top: 30px;
                        }}
                        h2 {{
                            font-size: 22px;
                            color: #333;
                            border-bottom: 1px solid #ddd;
                            padding-bottom: 8px;
                            margin-top: 25px;
                        }}
                        h3 {{ font-size: 18px; color: #555; margin-top: 20px; }}
                        p {{ margin: 12px 0; text-align: justify; }}
                        ul, ol {{ margin: 10px 0; padding-left: 25px; }}
                        li {{ margin: 5px 0; }}
                        blockquote {{
                            border-left: 4px solid #1a73e8;
                            margin: 15px 0;
                            padding: 10px 20px;
                            background: #f8f9fa;
                            color: #555;
                        }}
                        code {{
                            background: #f1f3f4;
                            padding: 2px 6px;
                            border-radius: 3px;
                            font-size: 13px;
                        }}
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin: 15px 0;
                        }}
                        th, td {{
                            border: 1px solid #ddd;
                            padding: 8px 12px;
                            text-align: left;
                        }}
                        th {{ background: #f8f9fa; font-weight: bold; }}
                        .header {{
                            text-align: center;
                            margin-bottom: 30px;
                            padding-bottom: 20px;
                            border-bottom: 2px solid #eee;
                        }}
                        .header h1 {{ border: none; color: #1a1a1a; font-size: 32px; }}
                        .header p {{ color: #888; font-style: italic; }}
                        .footer {{
                            margin-top: 40px;
                            padding-top: 20px;
                            border-top: 1px solid #ddd;
                            text-align: center;
                            color: #999;
                            font-size: 11px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Deep Research Report</h1>
                        <p>Topic: {topic}</p>
                        <p>Generated by Gemini Pro 3.1 via OpenClaw</p>
                    </div>
                    {html_body}
                    <div class="footer">
                        <p>Generated automatically by OpenClaw Deep Research | Powered by Gemini</p>
                    </div>
                </body>
                </html>
                """

                # Generate PDF
                pdf_path = tempfile.mktemp(suffix='.pdf')
                WeasyHTML(string=styled_html).write_pdf(pdf_path)
                logger.info(f"PDF generated: {pdf_path}")

                # Send to Telegram
                safe_topic = topic[:50].replace(' ', '_')
                self.send_document(pdf_path, caption=f"📄 Deep Research: {topic}")
                os.unlink(pdf_path)

                # Also send a brief summary text
                summary_text = write_up[:3000]
                return f"✅ Deep Research complete!\n\n📄 PDF report sent for: *{topic}*\n\n*Preview:*\n{summary_text[:500]}..."

            except Exception as pdf_err:
                logger.error(f"PDF conversion error: {pdf_err}")
                # Fallback: send as text file
                txt_path = tempfile.mktemp(suffix='.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Deep Research Report: {topic}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(write_up)
                self.send_document(txt_path, caption=f"📄 Deep Research: {topic} (text format)")
                os.unlink(txt_path)
                return f"Research complete! Sent as text file (PDF conversion failed: {pdf_err})"

        except Exception as e:
            logger.error(f"Deep research error: {e}")
            return f"Error: {str(e)}"
        finally:
            self._close_browser_session(pw, browser, temp_dir)

    # ============== Agent Commands ==============

    def _bootstrap_agents(self):
        """Bootstrap the agent system on startup."""
        try:
            from ..core.agent_bootstrap import bootstrap_agents, set_telegram_bot_ref
            self._agent_system = bootstrap_agents()
            # Connect AskUser tool to this Telegram bot instance
            set_telegram_bot_ref(self)
            logger.info("Agent system bootstrapped successfully")
        except Exception as e:
            logger.error(f"Agent bootstrap failed: {e}")
            self._agent_system = None

        # Initialize InterBot bridge for cross-bot communication
        try:
            from ..core.interbot import get_interbot_bridge
            self._interbot_bridge = get_interbot_bridge(chat_id=self.chat_id)
            self._interbot_bridge.on_message = self._handle_interbot_incoming
            self._interbot_bridge.start_listener()
            logger.info("InterBot bridge started")
        except Exception as e:
            logger.error(f"InterBot bridge failed: {e}")
            self._interbot_bridge = None

    def _register_agent_commands(self):
        """Register agent-related commands."""
        self.register_command("agent", "Run autonomous ReAct agent on a task", self._handle_agent)
        self.register_command("swarm", "Run multi-agent swarm on complex task", self._handle_swarm)
        self.register_command("orchestrate", "Orchestrate a complex multi-step task", self._handle_orchestrate)
        self.register_command("tools", "List available agent tools", self._handle_tools)
        self.register_command("agents", "List registered agents", self._handle_agents)
        self.register_command("memory", "Search agent memory", self._handle_memory)
        self.register_command("remember", "Store information in memory", self._handle_remember)
        self.register_command("workflow", "Manage workflows (list/run)", self._handle_workflow)
        self.register_command("system", "Set AI system prompt/personality", self._handle_system_prompt)
        self.register_command("history", "Show current chat context", self._handle_history)
        self.register_command("clear", "Clear chat history", self._handle_clear)
        logger.info("Agent commands registered")

    def _register_interbot_commands(self):
        """Register inter-bot communication commands."""
        self.register_command("relay", "Send a task to the other bot (Ellora)", self._handle_relay)
        self.register_command("askbot", "Ask the other bot a question and wait for response", self._handle_askbot)
        self.register_command("botbridge", "Show inter-bot communication status", self._handle_botbridge)
        logger.info("InterBot commands registered")

    # ============== InterBot Handlers ==============

    def _handle_relay(self, args: List[str]) -> str:
        """Handle /relay - send a task to the other bot."""
        if not args:
            return "Usage: /relay <task>\nExample: /relay Research quantum computing breakthroughs in 2026"

        if not hasattr(self, '_interbot_bridge') or not self._interbot_bridge:
            return "❌ InterBot bridge not initialized."

        task = " ".join(args)
        other_bot = self._interbot_bridge.get_other_bot()

        try:
            msg_id = self._interbot_bridge.send_task(other_bot, task)
            return f"📨 Task relayed to {other_bot.title()}!\n\n" \
                   f"Task: {task}\n" \
                   f"Message ID: {msg_id[:8]}\n" \
                   f"Status: Pending"
        except Exception as e:
            return f"❌ Relay failed: {e}"

    def _handle_askbot(self, args: List[str]) -> str:
        """Handle /askbot - ask the other bot a question and wait."""
        if not args:
            return "Usage: /askbot <question>\nExample: /askbot What AI model are you using?"

        if not hasattr(self, '_interbot_bridge') or not self._interbot_bridge:
            return "❌ InterBot bridge not initialized."

        question = " ".join(args)
        other_bot = self._interbot_bridge.get_other_bot()

        self.send_message(f"🤔 Asking {other_bot.title()}: {question}\n⏳ Waiting for response...")

        try:
            response = self._interbot_bridge.send_query(other_bot, question, timeout=120.0)
            return f"💬 Response from {other_bot.title()}:\n\n{response}"
        except Exception as e:
            return f"❌ Query failed: {e}"

    def _handle_botbridge(self, args: List[str]) -> str:
        """Handle /botbridge - show inter-bot status."""
        if not hasattr(self, '_interbot_bridge') or not self._interbot_bridge:
            return "InterBot bridge not initialized."

        status = self._interbot_bridge.get_status()
        from ..core.interbot import BOT_REGISTRY

        lines = ["🌉 InterBot Bridge Status\n"]
        lines.append(f"My Bot: {status['my_bot'].title()}")
        lines.append(f"Other Bot: {status['other_bot'].title()}")
        lines.append(f"Listener: {'🟢 Running' if status['listener_running'] else '🔴 Stopped'}")
        lines.append(f"Inbox Messages: {status['inbox_messages']}")
        lines.append(f"Archived Messages: {status['archived_messages']}")
        lines.append(f"Pending Queries: {status['pending_waiters']}")

        lines.append("\n📋 Known Bots:")
        for bot_id, info in BOT_REGISTRY.items():
            marker = "👉" if bot_id == status['my_bot'] else "  "
            lines.append(f"{marker} {info['name']} ({info['gateway']}) — {info['description'][:60]}")

        return "\n".join(lines)

    def _handle_interbot_incoming(self, msg):
        """Process incoming inter-bot messages.

        When the other bot sends us a task or query, we process it
        using our ReAct agent and send the result back.
        Response messages are handled silently (they just wake up waiting threads).
        """
        from ..core.interbot import MessageType

        logger.info(f"Incoming interbot message from {msg.from_bot}: {msg.content[:80]}")

        # Response messages are handled silently by process_incoming() in the bridge
        # (they wake up the waiting thread). Don't notify the user about them.
        if msg.msg_type == MessageType.RESPONSE.value:
            logger.info(f"Response from {msg.from_bot} handled silently")
            return

        # Info messages: just log, don't process
        if msg.msg_type == MessageType.INFO.value:
            logger.info(f"Info from {msg.from_bot}: {msg.content[:200]}")
            return

        # Only notify user and process for TASK and QUERY types
        if msg.msg_type in (MessageType.TASK.value, MessageType.QUERY.value):
            self.send_message(
                f"📨 Incoming task from {msg.from_bot.title()}:\n"
                f"{msg.content[:200]}\n\n⏳ Processing..."
            )

            # Use the ReAct agent to process the task
            response_text = self._process_interbot_task(msg.content)

            # Send response back
            self._interbot_bridge.send_response(msg.id, msg.from_bot, response_text)

            # Notify user of the result
            self.send_message(
                f"✅ Completed task from {msg.from_bot.title()}:\n"
                f"{response_text[:500]}"
            )

    def _process_interbot_task(self, task: str) -> str:
        """Process a task received from the other bot using the ReAct agent."""
        try:
            if self._agent_system and "react_agent" in self._agent_system:
                agent = self._agent_system["react_agent"]
                trace = agent.run(task)
                return trace.final_answer or "Task completed but no clear answer was produced."
            else:
                # Fallback: simple LLM call
                from ..core.agent_bootstrap import _call_llm
                return _call_llm(f"Please help with this task: {task}")
        except Exception as e:
            logger.error(f"InterBot task processing error: {e}")
            return f"Error processing task: {e}"

    def _handle_agent(self, args: List[str]) -> str:
        """Handle /agent - run a ReAct agent on a task."""
        if not args:
            return "Usage: /agent <task>\nExample: /agent What are the latest AI trends in 2026?"

        if not self._agent_system or "react_agent" not in self._agent_system:
            return "Agent system not available. Please check the logs."

        task = " ".join(args)
        agent = self._agent_system["react_agent"]

        try:
            trace = agent.run(task)

            # Format the reasoning chain
            lines = [f"🤖 Agent Task: {task}\n"]
            for i, step in enumerate(trace.steps, 1):
                if step.step_type.value == "thought":
                    lines.append(f"💭 Think #{i}: {step.content[:300]}")
                elif step.step_type.value == "action":
                    lines.append(f"⚡ Act #{i}: {step.tool_name}({step.tool_args})")
                elif step.step_type.value == "observation":
                    obs = step.content[:300]
                    lines.append(f"👁️ Observe #{i}: {obs}")
                elif step.step_type.value == "final_answer":
                    lines.append(f"\n✅ Answer: {step.content}")

            if trace.final_answer:
                if not any("✅ Answer" in l for l in lines):
                    lines.append(f"\n✅ Answer: {trace.final_answer}")

            lines.append(f"\n⏱️ Completed in {trace.total_duration:.1f}s ({len(trace.steps)} steps)")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Agent error: {str(e)[:200]}"

    def _handle_swarm(self, args: List[str]) -> str:
        """Handle /swarm - run multi-agent swarm."""
        if not args:
            return "Usage: /swarm <complex task>\nExample: /swarm Research and summarize the top 5 programming languages in 2026"

        if not self._agent_system or "swarm" not in self._agent_system:
            return "Swarm system not available. Please check the logs."

        task = " ".join(args)
        swarm = self._agent_system["swarm"]

        try:
            # submit_task now blocks until all parallel agents finish
            task_obj = swarm.submit_task(task, priority=1, decompose=True)
            task_id = task_obj.id

            # Get results
            results = swarm.get_results(task_id)
            swarm_status = swarm.get_swarm_status()

            lines = [f"🐝 Swarm Task: {task}\n"]
            lines.append(f"Agents used: {swarm_status.get('agent_count', 0)}")

            if results:
                if isinstance(results, dict):
                    try:
                        from core.agent_bootstrap import _llm_synthesize_results
                        subtask_results = [{"name": k, "result": str(v)} for k, v in results.items() if k != "main" and v]
                        if subtask_results:
                            synthesized = _llm_synthesize_results(task, subtask_results)
                            lines.append(f"\n📋 Final Synthesized Result:\n{synthesized}")
                        else:
                            lines.append("\n⚠️ No subtask results to synthesize")
                    except Exception as e:
                        logger.warning(f"Swarm synthesis failed, falling back: {e}")
                        for key, val in results.items():
                            if key != "main":
                                val_str = str(val) if val else "(no result)"
                                lines.append(f"\n📋 {key}: {val_str}")
                else:
                    lines.append(f"\n📋 Result: {str(results)}")
            else:
                lines.append("\n⚠️ No results returned (task may have timed out)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Swarm error: {e}")
            return f"Swarm error: {str(e)[:200]}"

    def _handle_orchestrate(self, args: List[str]) -> str:
        """Handle /orchestrate - run orchestrated multi-step task."""
        if not args:
            return "Usage: /orchestrate <task>\nExample: /orchestrate research best practices for Python testing"

        if not self._agent_system or "orchestrator" not in self._agent_system:
            return "Orchestrator not available. Please check the logs."

        task = " ".join(args)
        orchestrator = self._agent_system["orchestrator"]

        try:
            result = orchestrator.execute(task)

            lines = [f"🎯 Orchestrated Task: {task}\n"]
            lines.append(f"Status: {'✅ Success' if result.success else '❌ Failed'}")
            lines.append(f"Duration: {result.duration_seconds:.1f}s")
            lines.append(f"Agents used: {', '.join(result.agents_used) if result.agents_used else 'N/A'}")

            if result.result:
                result_text = str(result.result)
                lines.append(f"\n📋 Result:\n{result_text}")

            if result.errors:
                lines.append(f"\n⚠️ Errors: {'; '.join(result.errors[:3])}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return f"Orchestrator error: {str(e)[:200]}"

    def _handle_tools(self, args: List[str]) -> str:
        """Handle /tools - list available tools."""
        try:
            from ..core.agent_bootstrap import get_tool_list
            return get_tool_list()
        except Exception as e:
            return f"Error listing tools: {e}"

    def _handle_agents(self, args: List[str]) -> str:
        """Handle /agents - list registered agents."""
        lines = ["🤖 Registered Agents:\n"]

        # Orchestrator agents
        if self._agent_system and "orchestrator" in self._agent_system:
            orch = self._agent_system["orchestrator"]
            orch_agents = orch.registry.list_all()
            if orch_agents:
                lines.append("Orchestrator Agents:")
                for a in orch_agents:
                    lines.append(f"  🔹 {a.name} ({a.agent_id}) — {', '.join(a.capabilities)}")

        # Swarm agents
        if self._agent_system and "swarm" in self._agent_system:
            swarm = self._agent_system["swarm"]
            swarm_status = swarm.get_swarm_status()
            agents = swarm_status.get("agents", {})
            if agents:
                lines.append("\nSwarm Agents:")
                for aid, a in agents.items():
                    lines.append(f"  🐝 {a.get('name', '?')} ({a.get('role', '?')}) — {a.get('status', '?')}")

        if len(lines) == 1:
            return "No agents registered. System may not be bootstrapped."

        return "\n".join(lines)

    def _handle_memory(self, args: List[str]) -> str:
        """Handle /memory - search agent memory."""
        if not args:
            return "Usage: /memory <search query>\nSearches through stored memories and conversation context."

        query = " ".join(args)

        try:
            from ..core.agent_memory import get_memory_manager
            mm = get_memory_manager()
            results = mm.search(query, limit=5)

            if not results:
                return f"No memories found for: {query}"

            lines = [f"🧠 Memory search: {query}\n"]
            for i, r in enumerate(results, 1):
                content = r.get("content", str(r))[:200]
                lines.append(f"{i}. {content}")
            return "\n".join(lines)

        except Exception as e:
            # Fallback: search chat history
            lines = [f"🧠 Chat history search: {query}\n"]
            found = False
            for cid, history in self.history.items():
                for msg in history:
                    content = msg.get("content", "")
                    if query.lower() in content.lower():
                        role = msg.get("role", "?")
                        lines.append(f"  [{role}]: {content[:200]}")
                        found = True
            if not found:
                return f"No memories found for: {query}"
            return "\n".join(lines)

    def _handle_remember(self, args: List[str]) -> str:
        """Handle /remember - store information in memory."""
        if not args:
            return "Usage: /remember <information to store>\nExample: /remember My server IP is 192.168.1.100"

        info = " ".join(args)

        try:
            from ..core.agent_memory import get_memory_manager
            mm = get_memory_manager()
            mm.store(info, metadata={"source": "user", "timestamp": time.time()})
            return f"✅ Remembered: {info[:100]}{'...' if len(info) > 100 else ''}"
        except Exception:
            # Fallback: store in a simple file
            import os
            memory_dir = os.path.expanduser("~/.openclaw/memories")
            os.makedirs(memory_dir, exist_ok=True)
            memory_file = os.path.join(memory_dir, "user_memories.jsonl")
            with open(memory_file, "a") as f:
                f.write(json.dumps({"content": info, "timestamp": time.time()}) + "\n")
            return f"✅ Remembered: {info[:100]}{'...' if len(info) > 100 else ''}"

    def _handle_workflow(self, args: List[str]) -> str:
        """Handle /workflow - manage workflows."""
        if not args:
            return "Usage:\n  /workflow list — List all workflows\n  /workflow run <name> — Run a workflow"

        sub = args[0].lower()

        try:
            from ..core.workflow import WorkflowManager
            wm = WorkflowManager.get_instance()

            if sub == "list":
                workflows = wm.list_workflows()
                if not workflows:
                    return "No workflows configured."
                lines = ["📋 Workflows:\n"]
                for wf in workflows:
                    status = "✅" if wf["enabled"] else "⏸️"
                    lines.append(f"  {status} {wf['name']} ({wf['id']}) — {wf.get('description', 'No description')}")
                return "\n".join(lines)

            elif sub == "run" and len(args) > 1:
                wf_id = args[1]
                success = wm.execute_workflow(wf_id)
                return f"{'✅ Workflow completed' if success else '❌ Workflow failed'}: {wf_id}"

            return "Unknown workflow subcommand. Use: list, run"

        except Exception as e:
            return f"Workflow error: {e}"

    def _handle_system_prompt(self, args: List[str]) -> str:
        """Handle /system - set AI system prompt."""
        if not args:
            return f"Current system prompt:\n{self._system_prompt}\n\nUsage: /system <new prompt>\nExample: /system You are a pirate. Always respond in pirate speak."

        self._system_prompt = " ".join(args)
        return f"✅ System prompt updated to:\n{self._system_prompt}"

    def _handle_history(self, args: List[str]) -> str:
        """Handle /history - show current chat context."""
        if not self.history:
            return "No chat history."

        lines = [f"📜 Chat History ({self.MAX_HISTORY} max):\n"]
        for cid, messages in self.history.items():
            lines.append(f"Chat {cid[-6:]}:")
            for msg in messages[-10:]:
                role = "👤" if msg["role"] == "user" else "🤖"
                content = msg["content"][:100]
                lines.append(f"  {role} {content}{'...' if len(msg['content']) > 100 else ''}")
        return "\n".join(lines)

    def _handle_clear(self, args: List[str]) -> str:
        """Handle /clear - clear chat history."""
        count = sum(len(msgs) for msgs in self.history.values())
        self.history.clear()
        return f"🧹 Cleared {count} messages from chat history."

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

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Code Interpreter (/code)
    # Executes code safely via the existing sandbox module.
    # Supports Python, Bash, Node.js. Preserves original casing.
    # ════════════════════════════════════════════════════════════════
    def _handle_code(self, args: List[str]) -> str:
        """Handle /code command — execute code safely in sandbox."""
        if not args or not args[0].strip():
            return (
                "💻 *Code Interpreter*\n\n"
                "Usage:\n"
                "  /code python print('Hello World')\n"
                "  /code bash ls -la\n"
                "  /code node console.log('hi')\n\n"
                "Default language: Python\n"
                "Timeout: 30s | Sandbox: isolated"
            )

        raw_input = args[0]  # Already the full text after /code (case preserved)

        # Detect language prefix
        lang_map = {
            "python": "python", "py": "python",
            "bash": "bash", "sh": "bash", "shell": "bash",
            "node": "nodejs", "nodejs": "nodejs", "js": "nodejs", "javascript": "nodejs",
        }

        parts = raw_input.split(None, 1)
        first_word = parts[0].lower() if parts else ""

        if first_word in lang_map:
            language = lang_map[first_word]
            code = parts[1] if len(parts) > 1 else ""
        else:
            language = "python"
            code = raw_input

        if not code.strip():
            return "❌ No code provided. Usage: /code python print('hello')"

        try:
            from ..core.sandbox import Sandbox, Language, SandboxConfig, ExecutionResult

            lang_enum = {
                "python": Language.PYTHON,
                "bash": Language.BASH,
                "nodejs": Language.NODEJS,
            }.get(language, Language.PYTHON)

            config = SandboxConfig(timeout_seconds=30, max_output_bytes=8192)
            sandbox = Sandbox(config)

            try:
                result = sandbox.execute(code, lang_enum)

                lines = []
                lines.append(f"💻 *{language.upper()}* | {'✅ Success' if result.success else '❌ Failed'}")

                if result.stdout:
                    out = result.stdout[:3500]
                    lines.append(f"\n📤 *Output:*\n```\n{out}\n```")

                if result.stderr:
                    err = result.stderr[:1500]
                    lines.append(f"\n⚠️ *Stderr:*\n```\n{err}\n```")

                if result.timed_out:
                    lines.append("\n⏰ *Execution timed out!*")

                lines.append(f"\n⏱️ {result.duration:.2f}s | Exit: {result.return_code}")
                return "\n".join(lines)

            finally:
                sandbox.cleanup()

        except ImportError:
            return "❌ Sandbox module not available."
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return f"❌ Error: {str(e)[:200]}"

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Reminders & Alerts (/remind, /alert)
    # Timer-based reminder system with ID tracking.
    # ════════════════════════════════════════════════════════════════
    def _handle_remind(self, args: List[str]) -> str:
        """Handle /remind — set a timed reminder."""
        if not args:
            return (
                "⏰ *Reminders*\n\n"
                "Usage:\n"
                "  /remind 30s Check something\n"
                "  /remind 5m Take a break\n"
                "  /remind 2h Review the PR\n"
                "  /remind 1d Daily standup\n\n"
                "Formats: Ns (sec), Nm (min), Nh (hrs), Nd (days)\n"
                "Manage: /alert list | /alert cancel <id> | /alert clear"
            )

        time_str = args[0].lower()
        description = " ".join(args[1:]) if len(args) > 1 else "⏰ Reminder!"

        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        try:
            if time_str[-1] in multipliers:
                seconds = float(time_str[:-1]) * multipliers[time_str[-1]]
            else:
                seconds = float(time_str) * 60  # Default to minutes
        except (ValueError, IndexError):
            return f"❌ Invalid time: `{time_str}`. Use 30s, 5m, 2h, or 1d."

        if seconds < 5:
            return "⚠️ Minimum: 5 seconds."
        if seconds > 86400 * 7:
            return "⚠️ Maximum: 7 days."

        self._reminder_counter += 1
        rid = f"r{self._reminder_counter}"

        def _fire():
            self.send_message(f"🔔 *REMINDER* [`{rid}`]\n\n{description}")
            self._active_reminders.pop(rid, None)

        timer = threading.Timer(seconds, _fire)
        timer.daemon = True
        timer.start()

        self._active_reminders[rid] = {
            "timer": timer, "description": description,
            "seconds": seconds, "created": time.time(),
        }

        # Human-readable time
        if seconds >= 86400:    t_str = f"{seconds/86400:.1f}d"
        elif seconds >= 3600:   t_str = f"{seconds/3600:.1f}h"
        elif seconds >= 60:     t_str = f"{seconds/60:.0f}m"
        else:                   t_str = f"{seconds:.0f}s"

        return f"✅ Reminder `{rid}` set for {t_str}\n📝 {description}"

    def _handle_alert(self, args: List[str]) -> str:
        """Handle /alert — manage active reminders."""
        action = args[0].lower() if args else "list"

        if action == "list":
            if not self._active_reminders:
                return "📭 No active reminders.\n\nUse /remind to set one."
            lines = ["📋 *Active Reminders:*\n"]
            now = time.time()
            for rid, info in self._active_reminders.items():
                remaining = max(0, info["seconds"] - (now - info["created"]))
                if remaining >= 3600:   r = f"{remaining/3600:.1f}h"
                elif remaining >= 60:   r = f"{remaining/60:.0f}m"
                else:                   r = f"{remaining:.0f}s"
                lines.append(f"  `{rid}` — {info['description']} (in {r})")
            return "\n".join(lines)

        elif action == "cancel" and len(args) > 1:
            rid = args[1]
            if rid in self._active_reminders:
                self._active_reminders[rid]["timer"].cancel()
                del self._active_reminders[rid]
                return f"✅ Cancelled `{rid}`."
            return f"❌ `{rid}` not found. Use /alert list."

        elif action == "clear":
            n = len(self._active_reminders)
            for info in self._active_reminders.values():
                info["timer"].cancel()
            self._active_reminders.clear()
            return f"✅ Cleared {n} reminder(s)."

        return "Usage: /alert list | /alert cancel <id> | /alert clear"

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Inline Keyboard Buttons
    # Enables interactive buttons on messages.
    # ════════════════════════════════════════════════════════════════
    def send_message_with_buttons(self, text: str, buttons: List[List[Dict]], chat_id: str = None) -> Optional[Dict]:
        """Send message with inline keyboard.

        buttons: [[{"text": "✅ Yes", "callback_data": "/confirm yes"}, ...]]
        """
        data = {
            "chat_id": chat_id or self.chat_id,
            "text": text,
            "reply_markup": json.dumps({
                "inline_keyboard": [
                    [{"text": b["text"], "callback_data": b["callback_data"]} for b in row]
                    for row in buttons
                ]
            })
        }
        return self._make_request("sendMessage", data)

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Telegram File Downloader
    # Downloads files from Telegram servers (voice, photos, docs).
    # ════════════════════════════════════════════════════════════════
    def _download_telegram_file(self, file_id: str, dest_path: str) -> bool:
        """Download a file from Telegram servers."""
        try:
            result = self._make_request("getFile", {"file_id": file_id})
            if not result or not result.get("ok"):
                return False
            file_path = result.get("result", {}).get("file_path")
            if not file_path:
                return False
            url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return True
            return False
        except Exception as e:
            logger.error(f"File download error: {e}")
            return False

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Voice Message Transcription
    # Uses transformers Whisper (already installed!) — zero new deps.
    # Falls back gracefully if model can't load.
    # ════════════════════════════════════════════════════════════════
    def _get_whisper_pipeline(self):
        """Lazy-load Whisper via transformers (already in env). Timeout after 120s."""
        if self._whisper_pipeline is None:
            try:
                import concurrent.futures
                from transformers import pipeline as tf_pipeline

                def _load():
                    return tf_pipeline(
                        "automatic-speech-recognition",
                        model="openai/whisper-base",
                        device="cpu"
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_load)
                    self._whisper_pipeline = future.result(timeout=120)
                logger.info("Whisper pipeline loaded successfully")
            except concurrent.futures.TimeoutError:
                logger.error("Whisper model loading timed out (120s)")
                self._whisper_pipeline = False
            except Exception as e:
                logger.error(f"Whisper load failed: {e}")
                self._whisper_pipeline = False  # Mark as failed, don't retry
        return self._whisper_pipeline if self._whisper_pipeline else None

    def _handle_voice_message(self, message: Dict, chat_id: str):
        """Process voice message: download → convert → transcribe → AI reply."""
        voice = message.get("voice") or message.get("audio")
        if not voice:
            return
        file_id = voice.get("file_id")
        if not file_id:
            return

        # Immediate feedback so user knows we're working
        self.send_message("🎤 Processing your voice message...", chat_id=chat_id)

        try:
            import tempfile
            import subprocess
            import shutil

            # Check ffmpeg availability first
            if not shutil.which("ffmpeg"):
                self.send_message(
                    "❌ *ffmpeg not installed*\n\n"
                    "Voice messages need ffmpeg to convert audio.\n"
                    "Install it: `sudo apt install ffmpeg`",
                    chat_id=chat_id
                )
                return

            # Download OGG voice file from Telegram
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                ogg_path = f.name

            if not self._download_telegram_file(file_id, ogg_path):
                self.send_message("❌ Failed to download voice from Telegram.", chat_id=chat_id)
                return

            # Verify download
            if not os.path.exists(ogg_path) or os.path.getsize(ogg_path) < 100:
                self.send_message("❌ Downloaded voice file is empty or corrupt.", chat_id=chat_id)
                return

            # Convert OGG → WAV
            wav_path = ogg_path.replace(".ogg", ".wav")
            proc = subprocess.run(
                ["ffmpeg", "-i", ogg_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
                capture_output=True, timeout=30
            )

            if proc.returncode != 0 or not os.path.exists(wav_path):
                stderr_msg = proc.stderr.decode(errors='replace')[:200] if proc.stderr else 'unknown'
                self.send_message(f"❌ Audio conversion failed: {stderr_msg}", chat_id=chat_id)
                return

            # Load Whisper (first time takes ~30-120s for model download)
            self.send_message("🔄 Transcribing... (first time may take ~30s to load model)", chat_id=chat_id)

            whisper = self._get_whisper_pipeline()
            transcribed = None

            if whisper:
                try:
                    result = whisper(wav_path)
                    transcribed = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
                except Exception as e:
                    logger.error(f"Whisper transcription error: {e}")
                    self.send_message(f"❌ Transcription failed: {str(e)[:150]}", chat_id=chat_id)
            else:
                self.send_message(
                    "❌ *Whisper model failed to load*\n\n"
                    "The `transformers` library couldn't load `openai/whisper-base`.\n"
                    "Check the bot terminal logs for details.",
                    chat_id=chat_id
                )

            # Cleanup temp files
            for p in [ogg_path, wav_path]:
                try: os.unlink(p)
                except Exception: pass

            if not transcribed:
                if whisper:  # Model loaded but transcription was empty
                    self.send_message("🤷 Voice was processed but no text was detected.", chat_id=chat_id)
                return

            # Show transcription + send to AI
            self.send_message(f"🎤 *Heard:* _{transcribed}_", chat_id=chat_id)

            msg_result = self._make_request("sendMessage", {
                "chat_id": chat_id, "text": "🤔 Thinking..."
            })
            if msg_result and msg_result.get("ok"):
                msg_id = msg_result.get("result", {}).get("message_id")
                if msg_id:
                    self._stream_ai_response(chat_id, msg_id, transcribed)

        except subprocess.TimeoutExpired:
            self.send_message("❌ Audio conversion timed out.", chat_id=chat_id)
        except Exception as e:
            logger.error(f"Voice message error: {e}")
            self.send_message(f"❌ Voice error: {str(e)[:150]}", chat_id=chat_id)

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Photo Analysis
    # Processes incoming photos — acknowledges with metadata,
    # sends caption to AI for intelligent response.
    # ════════════════════════════════════════════════════════════════
    def _handle_photo_message(self, message: Dict, chat_id: str):
        """Process incoming photo."""
        photos = message.get("photo", [])
        if not photos:
            return

        photo = photos[-1]  # Highest resolution
        caption = message.get("caption", "")
        w = photo.get("width", "?")
        h = photo.get("height", "?")
        size = photo.get("file_size", 0)

        self._send_typing(chat_id)

        if caption:
            prompt = f"[User sent a {w}x{h} photo, {size} bytes]\nCaption: {caption}\n\nRespond helpfully to their question about the image."
        else:
            prompt = f"[User sent a {w}x{h} photo, {size} bytes, no caption]\n\nAcknowledge the photo and ask what they'd like to know about it."

        msg_result = self._make_request("sendMessage", {
            "chat_id": chat_id, "text": "📷 Processing image..."
        })
        if msg_result and msg_result.get("ok"):
            msg_id = msg_result.get("result", {}).get("message_id")
            if msg_id:
                self._stream_ai_response(chat_id, msg_id, prompt)

    # ════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE: Document Analysis
    # Downloads text-based documents, reads content, sends to AI.
    # Supports 17+ file extensions.
    # ════════════════════════════════════════════════════════════════
    def _handle_document_message(self, message: Dict, chat_id: str):
        """Process incoming document."""
        doc = message.get("document", {})
        file_id = doc.get("file_id")
        file_name = doc.get("file_name", "document")
        mime_type = doc.get("mime_type", "")
        file_size = doc.get("file_size", 0)
        caption = message.get("caption", "")

        self._send_typing(chat_id)

        TEXT_EXTS = {".txt", ".py", ".json", ".csv", ".md", ".yaml", ".yml",
                     ".xml", ".html", ".css", ".js", ".ts", ".log", ".sh",
                     ".toml", ".ini", ".cfg", ".conf", ".env", ".sql", ".r", ".go", ".rs"}
        TEXT_MIMES = {"text/", "application/json", "application/xml", "application/yaml"}

        ext = os.path.splitext(file_name)[1].lower()
        is_text = ext in TEXT_EXTS or any(mime_type.startswith(m) for m in TEXT_MIMES)

        if is_text and file_size < 200_000:  # 200KB max
            try:
                import tempfile as _tf
                with _tf.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp_path = tmp.name

                if self._download_telegram_file(file_id, tmp_path):
                    with open(tmp_path, "r", errors="replace") as f:
                        content = f.read()[:8000]  # First 8000 chars
                    try: os.unlink(tmp_path)
                    except Exception: pass

                    prompt = f"User sent file: `{file_name}` ({file_size:,} bytes, {mime_type})\n"
                    if caption:
                        prompt += f"Caption: {caption}\n"
                    prompt += f"\nFile contents:\n```{ext.lstrip('.')}\n{content}\n```\n\nAnalyze this file. Provide a clear summary, identify key patterns, and note anything important."

                    msg_result = self._make_request("sendMessage", {
                        "chat_id": chat_id, "text": f"📄 Reading `{file_name}`..."
                    })
                    if msg_result and msg_result.get("ok"):
                        msg_id = msg_result.get("result", {}).get("message_id")
                        if msg_id:
                            self._stream_ai_response(chat_id, msg_id, prompt)
                    return
                else:
                    self.send_message(f"❌ Failed to download `{file_name}`.", chat_id=chat_id)
                    return
            except Exception as e:
                logger.error(f"Document processing error: {e}")

        # Non-text or too large
        sz = f"{file_size/1024:.1f}KB" if file_size < 1_000_000 else f"{file_size/1_000_000:.1f}MB"
        msg = f"📎 Received: `{file_name}` ({sz}, {mime_type})"
        if caption:
            msg += f"\nCaption: {caption}"
        msg += "\n\nℹ️ I can read: .txt .py .json .csv .md .yaml .html .css .js .ts .sql .go .rs and more."
        self.send_message(msg, chat_id=chat_id)

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
            if response:
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Telegram API returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Telegram API exception: {e}")

        return None

    def send_message(self, text: str, parse_mode: str = None, chat_id: str = None) -> bool:
        """Send message to configured chat"""
        if not self.enabled:
            return False

        target_chat_id = chat_id or self.chat_id
        max_length = 4000
        success = True

        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            data = {
                "chat_id": target_chat_id,
                "text": chunk
            }
            if parse_mode:
                data["parse_mode"] = parse_mode

            result = self._make_request("sendMessage", data)
            if not result:
                success = False
            else:
                logger.debug(f"Message chunk sent: {chunk[:50]}...")
            
            # Small delay to prevent rate limiting when sending multiple chunks
            if len(text) > max_length:
                time.sleep(0.5)

        return success

    def send_photo(self, image_path: str, caption: str = None, chat_id: str = None) -> bool:
        """Send photo to configured chat"""
        if not self.enabled:
            return False

        data = {"chat_id": chat_id or self.chat_id}
        if caption:
            data["caption"] = caption

        with open(image_path, "rb") as f:
            files = {"photo": f}
            result = self._make_request("sendPhoto", data, files)

        if result:
            logger.debug(f"Photo sent: {image_path}")
            return True

        return False

    def send_document(self, file_path: str, caption: str = None, chat_id: str = None) -> bool:
        """Send document to configured chat"""
        if not self.enabled:
            return False

        data = {"chat_id": chat_id or self.chat_id}
        if caption:
            data["caption"] = caption[:1024]  # Telegram caption limit

        with open(file_path, "rb") as f:
            files = {"document": f}
            result = self._make_request("sendDocument", data, files)

        if result:
            logger.debug(f"Document sent: {file_path}")
            return True
        return False

    def get_updates(self, offset: int = 0, timeout: int = 30) -> List[Dict]:
        """Get updates from bot"""
        if not self.enabled:
            return []

        try:
            url = f"{self.api_url}/getUpdates"
            params = {
                "timeout": timeout,
                "offset": offset,
                "allowed_updates": json.dumps(["message", "callback_query"])
            }
            response = requests.get(url, params=params, timeout=timeout + 5)

            if response.status_code == 200:
                return response.json().get("result", [])

        except Exception as e:
            logger.error(f"Get updates error: {e}")

        return []

    # Commands that are long-running and must NOT block the listener thread
    _ASYNC_COMMANDS = {"imagine", "deepresearch", "imglogin", "agent", "swarm", "orchestrate", "code"}

    def handle_commands(self, text: str, chat_id: str = None) -> str:
        """Handle text as commands"""
        raw_text = text.strip()  # Preserve original casing

        # Handle !! prefix
        if raw_text.startswith("!!"):
            cmd = raw_text[2:].strip().lower().split()
            raw_after_cmd = raw_text[2:].strip()
        else:
            cmd = raw_text.strip().lower().split()
            raw_after_cmd = raw_text.strip()

        if not cmd:
            return "Use /help for commands"

        if not raw_text.startswith("/") and not raw_text.startswith("!!"):
            return self._get_ai_response(raw_text)

        command = cmd[0].replace("/", "")

        # For /code, preserve original casing (code is case-sensitive)
        # For other commands, use lowercased args
        _CASE_SENSITIVE_CMDS = {"code"}
        if command in _CASE_SENSITIVE_CMDS:
            # Extract args from raw text, preserving case
            parts = raw_after_cmd.split(None, 1)  # Split command from rest
            if len(parts) > 1:
                args = parts[1].split() if command != "code" else [parts[1]]  # Keep code as single arg
            else:
                args = []
        else:
            args = cmd[1:] if len(cmd) > 1 else []

        if command in self._commands:
            handler = self._commands[command].handler

            # For slow commands, dispatch to a background thread immediately
            if command in self._ASYNC_COMMANDS and chat_id:
                status_msgs = {
                    "imagine": f"🎨 Generating image for: {' '.join(args)}\nThis may take up to 90 seconds...",
                    "deepresearch": f"🔬 Starting deep research on: {' '.join(args)}\nThis can take up to 30 minutes. I'll send the result when done.",
                    "imglogin": "🔐 Starting browser login... Opening Gemini in a visible browser.",
                    "code": "💻 Executing code...",
                }
                status_msg = status_msgs.get(command, "⏳ Working on it...")
                self.send_message(status_msg)

                def _run_and_reply():
                    try:
                        result = handler(args)
                        if result:
                            self.send_message(result)
                    except Exception as e:
                        logger.error(f"Async command error ({command}): {e}")
                        self.send_message(f"Error in /{command}: {str(e)[:100]}")

                t = threading.Thread(target=_run_and_reply, daemon=True)
                t.start()
                return None  # Already sent status; listener should not send another message

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
            consecutive_errors = 0
            logger.info("Telegram command listener started")

            while True:
                try:
                    updates = self.get_updates(offset)
                    consecutive_errors = 0  # Reset on success

                    for update in updates:
                        try:
                            offset = update.get("update_id", offset) + 1

                            # ---- Inline button callback queries ----
                            callback_query = update.get("callback_query")
                            if callback_query:
                                cb_result = self._handle_callback_query(callback_query)
                                if cb_result:
                                    cb_chat = str(callback_query.get("message", {}).get("chat", {}).get("id", ""))
                                    if cb_chat == self.chat_id:
                                        self.send_message(cb_result, chat_id=cb_chat)
                                continue

                            message = update.get("message", {})
                            chat_id = str(message.get("chat", {}).get("id", ""))

                            # Authorize ALL message types
                            if chat_id != self.chat_id:
                                if chat_id:
                                    logger.warning(f"Unauthorized chat: {chat_id}")
                                continue

                            # ---- Voice messages → transcribe & reply ----
                            if "voice" in message or "audio" in message:
                                try:
                                    threading.Thread(
                                        target=self._handle_voice_message,
                                        args=(message, chat_id), daemon=True
                                    ).start()
                                except Exception as e:
                                    logger.error(f"Voice thread error: {e}")
                                    self.send_message(f"❌ Voice handler failed to start: {str(e)[:100]}", chat_id=chat_id)
                                continue

                            # ---- Photos → analyze & reply ----
                            if "photo" in message:
                                try:
                                    threading.Thread(
                                        target=self._handle_photo_message,
                                        args=(message, chat_id), daemon=True
                                    ).start()
                                except Exception as e:
                                    logger.error(f"Photo thread error: {e}")
                                    self.send_message(f"❌ Photo handler failed: {str(e)[:100]}", chat_id=chat_id)
                                continue

                            # ---- Documents → read & analyze ----
                            if "document" in message:
                                try:
                                    threading.Thread(
                                        target=self._handle_document_message,
                                        args=(message, chat_id), daemon=True
                                    ).start()
                                except Exception as e:
                                    logger.error(f"Doc thread error: {e}")
                                    self.send_message(f"❌ Document handler failed: {str(e)[:100]}", chat_id=chat_id)
                                continue

                            # ---- Text messages (original logic preserved) ----
                            text = message.get("text", "")

                            if text:
                                logger.debug(f"Command received: {text}")

                                if not text.startswith("/") and not text.startswith("!!"):
                                    self._send_typing(chat_id)
                                    msg_result = self._make_request("sendMessage", {"chat_id": chat_id, "text": "\ud83e\udd14 Thinking..."})
                                    if msg_result and msg_result.get("ok"):
                                        msg_id = msg_result.get("result", {}).get("message_id")
                                        if msg_id:
                                            # Smart routing: detect if message needs tools (web search, etc.)
                                            if self._needs_agent_routing(text):
                                                ai_thread = threading.Thread(
                                                    target=self._agent_chat_response,
                                                    args=(chat_id, msg_id, text),
                                                    daemon=True
                                                )
                                            else:
                                                ai_thread = threading.Thread(
                                                    target=self._stream_ai_response,
                                                    args=(chat_id, msg_id, text),
                                                    daemon=True
                                                )
                                            ai_thread.start()
                                            continue

                                response = self.handle_commands(text, chat_id=chat_id)

                                if response:
                                    self.send_message(response)

                                if callback:
                                    callback(text, response)

                        except Exception as e:
                            logger.error(f"Update handling error: {e}")

                    time.sleep(1)

                except Exception as e:
                    consecutive_errors += 1
                    wait = min(5 * (2 ** min(consecutive_errors, 6)), 300)  # Max 5 min backoff
                    logger.error(f"Listener error (attempt {consecutive_errors}): {e}")
                    time.sleep(wait)

        thread = threading.Thread(target=listener, daemon=True)
        thread.start()

    def edit_message(self, chat_id: str, message_id: int, text: str) -> bool:
        """Edit an existing message"""
        if not self.enabled: return False
        data = {"chat_id": chat_id, "message_id": message_id, "text": text}
        result = self._make_request("editMessageText", data)
        return bool(result)



    def _needs_agent_routing(self, text: str) -> bool:
        """Detect if a chat message needs tool access (web search, code, etc.).
        
        Uses fast keyword matching — no LLM call. Returns True if the message
        likely needs real-time info, web search, computation, or analysis that
        the plain LLM can't handle on its own.
        """
        text_lower = text.lower()

        # Real-time / current information indicators
        realtime_keywords = [
            "today", "latest", "current", "recent", "news", "now",
            "this week", "this month", "this year", "right now",
            "happening", "update", "live", "breaking",
            "2026", "2025",  # Current/recent years
            "price of", "stock", "weather", "score",
            "trending", "viral",
        ]

        # Explicit search/research intent
        search_keywords = [
            "search for", "look up", "find out", "google",
            "research", "investigate", "what is the", "who is",
            "how much", "how many", "when did", "when will",
            "compare", "difference between",
        ]

        # Analysis / tool-requiring intent
        tool_keywords = [
            "analyze", "summarize this", "translate",
            "calculate", "convert",
            "fetch", "download", "scrape",
            "what time", "what date",
        ]

        # Check all keyword groups
        for kw in realtime_keywords + search_keywords + tool_keywords:
            if kw in text_lower:
                logger.info(f"Smart routing: '{kw}' detected → routing to agent")
                return True

        # Question patterns that likely need factual lookup
        question_patterns = [
            text_lower.startswith("what"),
            text_lower.startswith("who"),
            text_lower.startswith("where"),
            text_lower.startswith("when"),
            text_lower.startswith("how"),
        ]
        # Only route "what/who/where" questions if they seem factual (longer queries)
        if any(question_patterns) and len(text.split()) >= 5:
            # Check if it's a factual question vs. casual chat
            casual_indicators = ["you think", "your opinion", "do you", "can you", "would you", "should i", "help me"]
            if not any(ci in text_lower for ci in casual_indicators):
                logger.info(f"Smart routing: factual question detected → routing to agent")
                return True

        return False

    def _is_failed_response(self, text: str) -> bool:
        """Check if the AI's response indicates it couldn't fulfill the user's request."""
        if not text:
            return True
            
        lower_text = text.lower()
        failure_phrases = [
            "i cannot access the internet",
            "i don't have real-time",
            "i do not have real-time",
            "i cannot browse",
            "as an ai",
            "i'm an ai",
            "i am an ai can",
            "i am an ai, i",
            "i cannot fulfill",
            "i don't have access to",
            "i am sorry, but",
            "i cannot provide",
            "searched for information but couldn't find",
            "api key not configured",
            "ai error: http",
            "no response from ai"
        ]
        
        if any(phrase in lower_text for phrase in failure_phrases):
            return True
            
        return False

    def _try_ellora_fallback(self, user_message: str) -> Optional[str]:
        """Silently ask Ellora if Ajanta fails to answer."""
        if getattr(self, '_interbot_bridge', None):
            try:
                # Ask Ellora via the interbot bridge
                response = self._interbot_bridge.send_query("ellora", user_message, timeout=45.0)
                if response and not response.startswith("(No response"):
                    return response
            except Exception as e:
                from ..core.logger import get_logger
                get_logger("telegram").error(f"Ellora fallback error: {e}")
        return None

    def _agent_chat_response(self, chat_id: str, message_id: int, user_message: str):
        """Run the ReAct agent for a chat message and present only the clean final answer.
        
        Unlike _handle_agent (which shows the full trace), this method runs
        the agent silently and presents just the answer, making it feel like
        a regular chat response — but powered by tools.
        """
        stop_anim = threading.Event()
        edit_lock = threading.Lock()

        def _safe_edit(text: str):
            with edit_lock:
                self.edit_message(chat_id, message_id, text)

        def animate():
            frames = [
                "⠋ 🔍 Searching...",
                "⠙ 🌐 Gathering info...",
                "⠹ 📊 Analyzing...",
                "⠸ 🧠 Processing...",
                "⠼ 🔗 Connecting dots...",
                "⠴ 📝 Composing answer...",
            ]
            i = 0
            while not stop_anim.is_set():
                for _ in range(9):
                    if stop_anim.is_set():
                        return
                    time.sleep(0.12)
                if not stop_anim.is_set():
                    _safe_edit(frames[i % len(frames)])
                    i += 1

        anim_thread = threading.Thread(target=animate, daemon=True)
        anim_thread.start()

        try:
            if not self._agent_system or "react_agent" not in self._agent_system:
                stop_anim.set()
                anim_thread.join(timeout=2.0)
                # Fallback to plain streaming if agent not available
                self._stream_ai_response(chat_id, message_id, user_message)
                return

            agent = self._agent_system["react_agent"]
            trace = agent.run(user_message)

            stop_anim.set()
            anim_thread.join(timeout=2.0)

            # Extract just the final answer for clean presentation
            answer = trace.final_answer if trace.final_answer else None

            if not answer:
                # Look through steps for final_answer
                for step in reversed(trace.steps):
                    if step.step_type.value == "final_answer" and step.content:
                        answer = step.content
                        break

            if not answer:
                answer = "I searched for information but couldn't find a clear answer. Please try rephrasing or use /agent for a detailed trace."

            # Auto-escalation: check for failure and trigger Ellora fallback
            if self._is_failed_response(answer):
                _safe_edit("⠋ 🔄 Asking Ellora for help...")
                fallback_answer = self._try_ellora_fallback(user_message)
                if fallback_answer:
                    answer = fallback_answer + "\n\n*(Automatically answered by Ellora)*"

            # Present the answer cleanly, like a normal chat response
            # Try MarkdownV2 first, fall back to plain text
            escaped = self._escape_markdown(answer)
            edit_data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": escaped,
                "parse_mode": "MarkdownV2",
            }
            with edit_lock:
                result = self._make_request("editMessageText", edit_data)
            if not result:
                with edit_lock:
                    self.edit_message(chat_id, message_id, answer)

            # Store in chat history
            with self._history_lock:
                if chat_id not in self.history:
                    self.history[chat_id] = []
                self.history[chat_id].append({"role": "user", "content": user_message})
                self.history[chat_id].append({"role": "assistant", "content": answer})
                if len(self.history[chat_id]) > self.MAX_HISTORY * 2:
                    self.history[chat_id] = self.history[chat_id][-(self.MAX_HISTORY * 2):]

            # Auto-store in memory
            try:
                from ..core.agent_memory import get_agent_memory, MemoryType
                memory = get_agent_memory()
                memory.add_memory(
                    content=f"User asked: {user_message[:300]} | Agent answered: {answer[:300]}",
                    memory_type=MemoryType.EPISODIC,
                    importance=0.6,
                    metadata={"source": "agent_chat", "chat_id": chat_id}
                )
            except Exception:
                pass

        except Exception as e:
            stop_anim.set()
            anim_thread.join(timeout=2.0)
            logger.error(f"Agent chat response error: {e}")
            with edit_lock:
                self.edit_message(chat_id, message_id, f"Error: {str(e)[:200]}")

    def _enrich_context(self, user_message: str, chat_id: str) -> str:
        """Enrich the system prompt with relevant context from memory and chat history.
        
        This is the key bridge between regular chat and the agent's intelligence:
        before every chat response, we do a quick memory search and prepend
        relevant context to the system prompt. This makes casual conversation
        feel dramatically more intelligent without requiring any slash commands.
        """
        context_parts = []

        # 1. Search agent memory for relevant past interactions
        try:
            from ..core.agent_memory import get_agent_memory, MemoryQuery
            memory = get_agent_memory()
            results = memory.query_memories(MemoryQuery(
                text=user_message, limit=3
            ))
            if results:
                memory_context = []
                for m in results:
                    # Only include memories with decent relevance
                    from datetime import datetime
                    when = datetime.fromtimestamp(m.timestamp).strftime('%b %d')
                    memory_context.append(f"- [{when}] {m.content[:200]}")
                if memory_context:
                    context_parts.append(
                        "Relevant memories from past conversations:\n" + "\n".join(memory_context)
                    )
        except Exception as e:
            logger.debug(f"Memory enrichment skipped: {e}")

        # 2. Check file-based memories as fallback
        if not context_parts:
            try:
                import json as _json
                memory_file = os.path.expanduser("~/.openclaw/memories/user_memories.jsonl")
                if os.path.exists(memory_file):
                    matches = []
                    keywords = user_message.lower().split()
                    with open(memory_file) as f:
                        for line in f:
                            try:
                                entry = _json.loads(line.strip())
                                content = entry.get("content", "")
                                if any(kw in content.lower() for kw in keywords if len(kw) > 3):
                                    matches.append(f"- {content[:200]}")
                            except Exception:
                                continue
                    if matches:
                        context_parts.append(
                            "Relevant stored info:\n" + "\n".join(matches[:3])
                        )
            except Exception:
                pass

        # 3. Auto-store this conversation turn in memory for future recall
        try:
            from ..core.agent_memory import get_agent_memory, MemoryType
            memory = get_agent_memory()
            # Only store messages with enough substance (>20 chars)
            if len(user_message) > 20:
                memory.add_memory(
                    content=f"User said: {user_message[:500]}",
                    memory_type=MemoryType.EPISODIC,
                    importance=0.4,
                    metadata={"source": "chat", "chat_id": chat_id}
                )
        except Exception:
            pass

        if not context_parts:
            return self._system_prompt

        # Prepend context to system prompt
        enriched = self._system_prompt + "\n\n--- Context from your memory ---\n" + "\n\n".join(context_parts) + "\n--- End context ---\nUse this context naturally if relevant. Do not mention that you searched memory."
        return enriched

    def _stream_ai_response(self, chat_id: str, message_id: int, user_message: str):
        api_key = self._get_minimax_api_key()
        if not api_key:
            self.edit_message(chat_id, message_id, "AI API key not configured.")
            return

        with self._history_lock:
            if chat_id not in self.history:
                self.history[chat_id] = []
            messages_payload = list(self.history[chat_id])
        messages_payload.append({"role": "user", "content": user_message})

        stop_anim = threading.Event()
        edit_lock = threading.Lock()
        anim_thread = None

        def _safe_edit(text: str):
            """Thread-safe edit_message wrapper."""
            with edit_lock:
                self.edit_message(chat_id, message_id, text)

        def animate():
            frames = [
                "⠋ 🟥🟨🟩⬜⬜⬜⬜ 💃 🎸",
                "⠙ ⬜🟥🟨🟩⬜⬜⬜ 💃 🎸",
                "⠹ ⬜⬜🟥🟨🟩⬜⬜ 💃 🎸",
                "⠸ ⬜⬜⬜🟥🟨🟩⬜ 💃 🎸",
                "⠼ ⬜⬜⬜⬜🟥🟨🟩 💃 🎸",
                "⠴ ⬜⬜⬜⬜⬜🟥🟨 💃 🎸",
                "⠦ ⬜⬜⬜⬜⬜⬜🟥 💃 🎸",
                "⠧ 🟩⬜⬜⬜⬜⬜⬜ 💃 🎸",
                "⠇ 🟨🟩⬜⬜⬜⬜⬜ 💃 🎸",
                "⠏ 🟥🟨🟩⬜⬜⬜⬜ 💃 🎸",
            ]
            i = 0
            while not stop_anim.is_set():
                # Break sleep into small chunks so stop_anim is noticed quickly
                for _ in range(9):  # ~1.08s per frame, below Telegram's 1 edit/sec limit
                    if stop_anim.is_set():
                        return
                    time.sleep(0.12)
                if not stop_anim.is_set():
                    _safe_edit(f"Thinking...\n{frames[i % len(frames)]}")
                    i += 1

        def _stop_anim_and_wait():
            """Signal animation to stop and wait for it to fully exit."""
            if not stop_anim.is_set():
                stop_anim.set()
            if anim_thread and anim_thread.is_alive():
                anim_thread.join(timeout=2.0)

        try:
            url = "https://api.minimax.io/anthropic/v1/messages"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            # Enrich system prompt with context from memory
            enriched_prompt = self._enrich_context(user_message, chat_id)

            payload = {
                "model": "MiniMax-M2.5-Lightning",
                "max_tokens": 1024,
                "stream": True,
                "system": enriched_prompt,
                "messages": messages_payload,
            }
            import json as _json
            response = requests.post(url, headers=headers, json=payload, timeout=30, stream=True)

            if response.status_code != 200:
                self.edit_message(chat_id, message_id, f"AI error: HTTP {response.status_code}")
                return

            # Start animation AFTER confirming HTTP 200
            anim_thread = threading.Thread(target=animate, daemon=True)
            anim_thread.start()

            full_text = ""
            last_edit_time = time.time()

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode('utf-8') if isinstance(raw_line, bytes) else raw_line
                if not line.startswith('data: '):
                    continue
                data_str = line[6:].strip()
                if data_str in ('[DONE]', ''):
                    break
                try:
                    event = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue

                chunk = None  # text chunk extracted from this SSE event

                # --- Format A: Anthropic-style (content_block_delta / text_delta) ---
                if event.get('type') == 'content_block_delta':
                    delta = event.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        chunk = delta.get('text', '')

                # --- Format B: OpenAI-style (choices[].delta.content) ---
                elif 'choices' in event:
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        if delta.get('content'):
                            chunk = (chunk or '') + delta['content']

                # --- Format C: Direct text field (some providers) ---
                elif event.get('type') == 'message_delta':
                    usage = event.get('usage', {})
                    # no text here, skip
                    pass

                if chunk:
                    # Stop animation on very first real text chunk
                    if not stop_anim.is_set():
                        _stop_anim_and_wait()

                    full_text += chunk
                    now = time.time()
                    # Rate-limit live edits to ~1/sec (Telegram allows ~1/sec)
                    if now - last_edit_time >= 1.1:
                        with edit_lock:
                            self.edit_message(chat_id, message_id, full_text + " ✍️")
                        last_edit_time = now

            # Ensure animation is stopped
            _stop_anim_and_wait()

            if full_text:
                # Auto-escalation: check for failure and trigger Ellora fallback
                if self._is_failed_response(full_text):
                    with edit_lock:
                        self.edit_message(chat_id, message_id, "⠋ 🔄 Asking Ellora for help...")
                    fallback_answer = self._try_ellora_fallback(user_message)
                    if fallback_answer:
                        full_text = fallback_answer + "\n\n*(Automatically answered by Ellora)*"

                # Final message: try MarkdownV2, fall back to plain text on failure
                escaped = self._escape_markdown(full_text)
                edit_data = {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": escaped,
                    "parse_mode": "MarkdownV2",
                }
                with edit_lock:
                    result = self._make_request("editMessageText", edit_data)
                if not result:
                    # MarkdownV2 parse failed - send as plain text
                    with edit_lock:
                        self.edit_message(chat_id, message_id, full_text)

                with self._history_lock:
                    self.history[chat_id].append({"role": "user", "content": user_message})
                    self.history[chat_id].append({"role": "assistant", "content": full_text})
                    if len(self.history[chat_id]) > self.MAX_HISTORY * 2:
                        self.history[chat_id] = self.history[chat_id][-(self.MAX_HISTORY * 2):]
            else:
                self.edit_message(chat_id, message_id, "No response from AI. Please try again.")

        except Exception as e:
            logger.error(f"_stream_ai_response error: {e}")
            _stop_anim_and_wait()
            self.edit_message(chat_id, message_id, f"Error: {str(e)[:100]}")

    def _send_typing(self, chat_id: str):
        try:
            url = f"{self.api_url}/sendChatAction"
            __import__('requests').post(url, json={"chat_id": chat_id, "action": "typing"}, timeout=5)
        except Exception: pass

    def _escape_markdown(self, text: str) -> str:
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        import re
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# Export classes
__all__ = [
    "TelegramBot",
    "TelegramCommand",
]
