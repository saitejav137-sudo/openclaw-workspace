"""
Browser-based Web Fetch for OpenClaw

Uses local browser to fetch web pages - useful for accessing
websites that require browser features (login, JS, etc.)
"""

import time
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin

from ..core.logger import get_logger

logger = get_logger("browser_fetch")


@dataclass
class BrowserFetchResult:
    """Result from browser fetch"""
    content: str
    title: str
    url: str
    success: bool
    error: Optional[str] = None


class BrowserFetcher:
    """
    Fetch web pages using local browser.

    Uses Playwright if available, falls back to simple requests.
    """

    def __init__(self, headless: bool = True, timeout: int = 30):
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None
        self._page = None
        self._use_playwright = False

    def _init_playwright(self) -> bool:
        """Initialize Playwright"""
        try:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            self._page = self._browser.new_page()
            self._use_playwright = True
            logger.info("Browser fetcher initialized with Playwright")
            return True
        except ImportError:
            logger.warning("Playwright not installed, falling back to requests")
            return False
        except Exception as e:
            logger.error(f"Failed to init Playwright: {e}")
            return False

    def fetch(self, url: str) -> BrowserFetchResult:
        """
        Fetch a URL using the browser.

        Args:
            url: URL to fetch

        Returns:
            BrowserFetchResult with content
        """
        if not self._use_playwright and not self._playwright:
            if not self._init_playwright():
                # Fall back to simple requests
                return self._fetch_simple(url)

        try:
            self._page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")
            time.sleep(1)  # Let content render

            content = self._page.content()
            title = self._page.title()

            return BrowserFetchResult(
                content=content,
                title=title,
                url=url,
                success=True
            )

        except Exception as e:
            logger.error(f"Browser fetch error: {e}")
            return BrowserFetchResult(
                content="",
                title="",
                url=url,
                success=False,
                error=str(e)
            )

    def search(self, query: str) -> BrowserFetchResult:
        """
        Search using DuckDuckGo in browser.

        Args:
            query: Search query

        Returns:
            BrowserFetchResult with search results
        """
        # Use DuckDuckGo lite HTML version
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        return self.fetch(url)

    def _fetch_simple(self, url: str) -> BrowserFetchResult:
        """Simple fetch using requests"""
        import requests

        try:
            response = requests.get(url, timeout=self.timeout)
            return BrowserFetchResult(
                content=response.text,
                title=response.url,
                url=response.url,
                success=response.status_code == 200
            )
        except Exception as e:
            return BrowserFetchResult(
                content="",
                title="",
                url=url,
                success=False,
                error=str(e)
            )

    def close(self):
        """Close browser"""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()


class SimpleWebFetcher:
    """
    Simple web fetcher using requests (fallback when no browser needed).

    This uses the remote desktop's direct internet connection.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def fetch(self, url: str) -> BrowserFetchResult:
        """Fetch URL using requests"""
        import requests

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            return BrowserFetchResult(
                content=response.text,
                title=url,
                url=response.url,
                success=response.status_code == 200
            )
        except Exception as e:
            return BrowserFetchResult(
                content="",
                title="",
                url=url,
                success=False,
                error=str(e)
            )

    def search(self, query: str) -> BrowserFetchResult:
        """Search using DuckDuckGo"""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        return self.fetch(url)


# Global fetcher instance
_browser_fetcher: Optional[BrowserFetcher] = None
_simple_fetcher: Optional[SimpleWebFetcher] = None


def get_browser_fetcher(headless: bool = True) -> BrowserFetcher:
    """Get browser fetcher instance"""
    global _browser_fetcher
    if _browser_fetcher is None:
        _browser_fetcher = BrowserFetcher(headless=headless)
    return _browser_fetcher


def get_simple_fetcher() -> SimpleWebFetcher:
    """Get simple fetcher instance"""
    global _simple_fetcher
    if _simple_fetcher is None:
        _simple_fetcher = SimpleWebFetcher()
    return _simple_fetcher


def fetch_url(url: str, use_browser: bool = False) -> BrowserFetchResult:
    """Quick fetch URL"""
    if use_browser:
        return get_browser_fetcher().fetch(url)
    return get_simple_fetcher().fetch(url)


def web_search(query: str, use_browser: bool = False) -> BrowserFetchResult:
    """Quick web search"""
    if use_browser:
        return get_browser_fetcher().search(query)
    return get_simple_fetcher().search(query)


__all__ = [
    "BrowserFetcher",
    "SimpleWebFetcher",
    "BrowserFetchResult",
    "get_browser_fetcher",
    "get_simple_fetcher",
    "fetch_url",
    "web_search",
]
