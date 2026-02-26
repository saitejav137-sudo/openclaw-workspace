"""
Browser Agent for OpenClaw

Direct browser automation using Playwright - no extension needed!
"""

import time
import base64
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from urllib.parse import urljoin

from ..core.logger import get_logger

logger = get_logger("browser_agent")


@dataclass
class BrowserAction:
    """Action to perform in browser"""
    action_type: str  # goto, click, type, evaluate, screenshot, extract
    selector: Optional[str] = None
    value: Optional[str] = None
    script: Optional[str] = None
    timeout: int = 30


@dataclass
class BrowserResult:
    """Result from browser action"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    screenshot: Optional[str] = None  # base64 encoded


class BrowserAgent:
    """
    Direct browser control agent using Playwright.

    No extension needed - controls Chrome directly!
    """

    def __init__(
        self,
        headless: bool = False,  # Show browser for debugging
        slow_mo: int = 0,  # Slow down actions (ms)
        timeout: int = 30000
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = timeout
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

    def start(self) -> bool:
        """Start browser"""
        try:
            from playwright.sync_api import sync_playwright
            import os
            import subprocess

            # Get current display
            display = os.environ.get('DISPLAY', ':0')
            logger.info(f"Starting browser with DISPLAY={display}")

            self._playwright = sync_playwright().start()

            # Try to use system Chrome first (better display support)
            chrome_paths = [
                '/usr/bin/google-chrome-stable',
                '/usr/bin/google-chrome',
                '/usr/bin/chromium',
                '/usr/bin/chromium-browser',
            ]

            chrome_executable = None
            for path in chrome_paths:
                if os.path.exists(path):
                    chrome_executable = path
                    break

            # Build launch args
            launch_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                f'--display={display}',
                '--remote-debugging-port=9222',
            ]

            # If not headless, add window size args
            if not self.headless:
                launch_args.extend([
                    '--start-maximized',
                    '--window-size=1280,720',
                ])

            logger.info(f"Launch args: {launch_args}")

            if chrome_executable:
                # Use system Chrome with Playwright
                logger.info(f"Using system Chrome: {chrome_executable}")
                self._browser = self._playwright.chromium.launch(
                    headless=False,  # Force visible
                    slow_mo=self.slow_mo,
                    executable_path=chrome_executable,
                    args=launch_args
                )
            else:
                # Fall back to Playwright's Chromium
                self._browser = self._playwright.chromium.launch(
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    args=launch_args
                )

            # Create context with default profile
            self._context = self._browser.new_context(
                viewport={'width': 1280, 'height': 720},
                ignore_https_errors=True
            )

            # Create new page
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.timeout)

            self._initialized = True
            logger.info("Browser agent started")
            return True

        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False

    def stop(self):
        """Stop browser"""
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.error(f"Error stopping browser: {e}")

        self._initialized = False

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> BrowserResult:
        """Navigate to URL"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            self._page.goto(url, wait_until=wait_until, timeout=self.timeout)
            time.sleep(0.5)  # Let content load

            return BrowserResult(
                success=True,
                data={
                    "url": self._page.url,
                    "title": self._page.title()
                }
            )
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def click(self, selector: str) -> BrowserResult:
        """Click element by selector"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            self._page.click(selector, timeout=self.timeout)
            time.sleep(0.3)
            return BrowserResult(success=True)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def type(self, selector: str, text: str, delay: int = 50) -> BrowserResult:
        """Type text into element"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            self._page.fill(selector, text, timeout=self.timeout)
            return BrowserResult(success=True)
        except Exception as e:
            # Try type as fallback
            try:
                self._page.type(selector, text, delay=delay)
                return BrowserResult(success=True)
            except:
                return BrowserResult(success=False, error=str(e))

    def press(self, selector: str, key: str) -> BrowserResult:
        """Press key on element"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            self._page.press(selector, key)
            return BrowserResult(success=True)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def evaluate(self, script: str) -> BrowserResult:
        """Execute JavaScript in page context"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            result = self._page.evaluate(script)
            return BrowserResult(success=True, data=result)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def screenshot(self, full_page: bool = False) -> BrowserResult:
        """Take screenshot"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            img_bytes = self._page.screenshot(full_page=full_page)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return BrowserResult(success=True, screenshot=img_base64)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def extract_text(self, selector: str) -> BrowserResult:
        """Extract text from element(s)"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            elements = self._page.query_selector_all(selector)
            texts = [el.inner_text() for el in elements]
            return BrowserResult(success=True, data=texts)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def extract_html(self, selector: str = None) -> BrowserResult:
        """Extract HTML from page or element"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            if selector:
                el = self._page.query_selector(selector)
                html = el.inner_html() if el else ""
            else:
                html = self._page.content()
            return BrowserResult(success=True, data=html)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def wait_for_selector(self, selector: str, timeout: int = None) -> BrowserResult:
        """Wait for element to appear"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            timeout = timeout or self.timeout
            self._page.wait_for_selector(selector, timeout=timeout)
            return BrowserResult(success=True)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def wait_for_load(self, timeout: int = None) -> BrowserResult:
        """Wait for page to load"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            timeout = timeout or self.timeout
            self._page.wait_for_load_state("networkidle", timeout=timeout)
            return BrowserResult(success=True)
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def get_page_info(self) -> Dict[str, Any]:
        """Get current page info"""
        if not self._initialized:
            return {"error": "Browser not initialized"}

        return {
            "url": self._page.url,
            "title": self._page.title(),
            "initialized": self._initialized
        }

    def close_tab(self, index: int = None):
        """Close tab by index (default: current)"""
        if not self._initialized:
            return

        try:
            if index is not None:
                pages = self._context.pages
                if index < len(pages):
                    pages[index].close()
            else:
                self._page.close()
        except Exception as e:
            logger.error(f"Error closing tab: {e}")

    def new_tab(self, url: str = "about:blank") -> BrowserResult:
        """Open new tab"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            new_page = self._context.new_page()
            if url != "about:blank":
                new_page.goto(url, wait_until="domcontentloaded")
            return BrowserResult(success=True, data={"url": new_page.url})
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def switch_tab(self, index: int):
        """Switch to tab by index"""
        if not self._initialized:
            return BrowserResult(success=False, error="Browser not initialized")

        try:
            pages = self._context.pages
            if index < len(pages):
                self._page = pages[index]
                return BrowserResult(success=True, data={"url": self._page.url})
            return BrowserResult(success=False, error="Tab not found")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Global browser agent instance
_browser_agent: Optional[BrowserAgent] = None


def get_browser_agent(headless: bool = False) -> BrowserAgent:
    """Get global browser agent"""
    global _browser_agent
    if _browser_agent is None:
        _browser_agent = BrowserAgent(headless=headless)
        _browser_agent.start()
    return _browser_agent


def close_browser_agent():
    """Close global browser agent"""
    global _browser_agent
    if _browser_agent:
        _browser_agent.stop()
        _browser_agent = None


__all__ = [
    "BrowserAction",
    "BrowserResult",
    "BrowserAgent",
    "get_browser_agent",
    "close_browser_agent",
]
