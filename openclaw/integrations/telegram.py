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

        if self.enabled:
            self._register_default_commands()
            self._register_search_commands()
            self._register_browser_commands()
            self._register_agent_commands()
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

        # If data starts with /, treat as command
        if data.startswith("/"):
            command_parts = data[1:].split()
            command = command_parts[0] if command_parts else ""
            args = command_parts[1:] if len(command_parts) > 1 else []

            if command in self._command_callbacks:
                return self._command_callbacks[command](args)

        return None

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
            from ..core.agent_bootstrap import bootstrap_agents
            self._agent_system = bootstrap_agents()
            logger.info("Agent system bootstrapped successfully")
        except Exception as e:
            logger.error(f"Agent bootstrap failed: {e}")
            self._agent_system = None

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
            params = {"timeout": timeout, "offset": offset}
            response = requests.get(url, params=params, timeout=timeout + 5)

            if response.status_code == 200:
                return response.json().get("result", [])

        except Exception as e:
            logger.error(f"Get updates error: {e}")

        return []

    # Commands that are long-running and must NOT block the listener thread
    _ASYNC_COMMANDS = {"imagine", "deepresearch", "imglogin", "agent", "swarm", "orchestrate"}

    def handle_commands(self, text: str, chat_id: str = None) -> str:
        """Handle text as commands"""
        text = text.strip()

        # Handle !! prefix
        if text.startswith("!!"):
            cmd = text[2:].strip().lower().split()
        else:
            cmd = text.strip().lower().split()

        if not cmd:
            return "Use /help for commands"

        if not text.startswith("/") and not text.startswith("!!"):
            return self._get_ai_response(text)

        command = cmd[0].replace("/", "")
        args = cmd[1:] if len(cmd) > 1 else []

        if command in self._commands:
            handler = self._commands[command].handler

            # For slow commands, dispatch to a background thread immediately
            if command in self._ASYNC_COMMANDS and chat_id:
                status_msgs = {
                    "imagine": f"🎨 Generating image for: {' '.join(args)}\nThis may take up to 90 seconds...",
                    "deepresearch": f"🔬 Starting deep research on: {' '.join(args)}\nThis can take up to 30 minutes. I'll send the result when done.",
                    "imglogin": "🔐 Starting browser login... Opening Gemini in a visible browser.",
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
                            message = update.get("message", {})
                            text = message.get("text", "")

                            if text:
                                # Validate chat_id
                                chat_id = str(message.get("chat", {}).get("id", ""))
                                if chat_id != self.chat_id:
                                    logger.warning(f"Unauthorized chat: {chat_id}")
                                    continue

                                logger.debug(f"Command received: {text}")

                                if not text.startswith("/") and not text.startswith("!!"):
                                    self._send_typing(chat_id)
                                    msg_result = self._make_request("sendMessage", {"chat_id": chat_id, "text": "\ud83e\udd14 Thinking..."})
                                    if msg_result and msg_result.get("ok"):
                                        msg_id = msg_result.get("result", {}).get("message_id")
                                        if msg_id:
                                            # Run in own thread so listener loop doesn't block
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
            payload = {
                "model": "MiniMax-M2.5-Lightning",
                "max_tokens": 1024,
                "stream": True,
                "system": self._system_prompt,
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
