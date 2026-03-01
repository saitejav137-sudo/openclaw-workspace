"""
Free Search API for OpenClaw

Provides internet search capabilities using free APIs (no API key required).
Uses DuckDuckGo as the primary search provider.
"""

import re
import time
import json
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urlencode

import requests

from ..core.logger import get_logger

logger = get_logger("search")


@dataclass
class SearchResult:
    """Search result from query"""
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"


@dataclass
class SearchResponse:
    """Search API response"""
    results: List[SearchResult]
    query: str
    total_results: int
    time_taken: float


class SearchProvider(Enum):
    """Search provider types"""
    DUCKDUCKGO = "duckduckgo"
    DUCKDUCKGO_INSTANT = "ddg_instant"
    BRAVE = "brave"
    SERPAPI = "serpapi"


class DuckDuckGoSearch:
    """
    DuckDuckGo search implementation using duckduckgo-search package.
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """Perform search query"""
        start_time = time.time()
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Origin": "https://html.duckduckgo.com",
            }
            
            results = []
            res = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=self.timeout)
            
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                for res_div in soup.find_all('div', class_='result'):
                    if len(results) >= max_results:
                        break
                        
                    title_elem = res_div.find('h2', class_='result__title')
                    url_elem = res_div.find('a', class_='result__url')
                    snippet_elem = res_div.find('a', class_='result__snippet')
                    
                    if title_elem and url_elem:
                        url_val = url_elem.get('href', '').strip()
                        if url_val.startswith('//'):  # Ad redirects tracking
                            continue
                            
                        results.append(SearchResult(
                            title=title_elem.get_text(strip=True),
                            url=url_val,
                            snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                            source="DuckDuckGo (Native Scraper)"
                        ))
            
            return SearchResponse(
                results=results,
                query=query,
                total_results=len(results),
                time_taken=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return SearchResponse([], query, 0, time.time() - start_time)

    def instant_answer(self, query: str) -> Optional[str]:
        """Get instant answer from DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                answers = list(ddgs.answers(query))
                if answers and len(answers) > 0:
                    # Answers can be in text or url
                    return answers[0].get("text", answers[0].get("url", ""))
        except Exception as e:
            logger.error(f"Instant answer error: {e}")
            
        return None


class BraveSearch:
    """
    Brave Search API (free tier available).
    Requires API key from https://brave.com/search/api/
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or self._load_api_key()
        self.timeout = timeout
        self.base_url = "https://api.search.brave.com/res/v1"

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment"""
        import os
        return os.getenv("BRAVE_SEARCH_API_KEY")

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """Perform Brave Search"""
        start_time = time.time()

        if not self.api_key:
            logger.warning("Brave Search API key not configured")
            return SearchResponse([], query, 0, time.time() - start_time)

        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }

            params = {
                "q": query,
                "count": max_results
            }

            response = requests.get(
                f"{self.base_url}/web/search",
                headers=headers,
                params=params,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"Brave search failed: {response.status_code}")
                return SearchResponse([], query, 0, time.time() - start_time)

            data = response.json()
            results = []

            for item in data.get("web", {}).get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="Brave Search"
                ))

            return SearchResponse(
                results=results,
                query=query,
                total_results=len(results),
                time_taken=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return SearchResponse([], query, 0, time.time() - start_time)


class SearchEngine:
    """
    Unified search engine with multiple provider support.

    Providers (in order of preference):
    1. DuckDuckGo (free, no key) - DEFAULT
    2. Brave Search (free tier available)
    """

    def __init__(
        self,
        provider: str = "duckduckgo",
        brave_api_key: Optional[str] = None,
        timeout: int = 10
    ):
        self.timeout = timeout
        self.provider_name = provider.lower()

        # Initialize providers
        self.duckduckgo = DuckDuckGoSearch(timeout=timeout)
        self.brave = BraveSearch(brave_api_key, timeout=timeout)

        # Set active provider
        if self.provider_name == "brave" and brave_api_key:
            self.provider = self.brave
        else:
            self.provider = self.duckduckgo
            self.provider_name = "duckduckgo"

        logger.info(f"Search engine initialized with provider: {self.provider_name}")

    def search(
        self,
        query: str,
        max_results: int = 10,
        include_instant: bool = False
    ) -> SearchResponse:
        """
        Perform search query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_instant: Try to include instant answer

        Returns:
            SearchResponse with results
        """
        # Perform search
        response = self.provider.search(query, max_results)

        # Try instant answer if enabled and no results
        if include_instant and not response.results:
            instant = self.duckduckgo.instant_answer(query)
            if instant:
                response.results.insert(0, SearchResult(
                    title="Instant Answer",
                    url="",
                    snippet=instant,
                    source="DuckDuckGo Instant"
                ))
                response.total_results = len(response.results)

        logger.info(f"Search '{query}': {response.total_results} results in {response.time_taken:.2f}s")
        return response

    def get_summary(self, query: str, max_results: int = 5) -> str:
        """
        Get a formatted summary of search results.

        Args:
            query: Search query
            max_results: Number of results to include

        Returns:
            Formatted string with search results
        """
        response = self.search(query, max_results)

        if not response.results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]

        for i, result in enumerate(response.results, 1):
            lines.append(f"{i}. {result.title}")
            lines.append(f"   {result.url}")
            if result.snippet:
                lines.append(f"   {result.snippet[:200]}...")
            lines.append("")

        return "\n".join(lines)

    def quick_answer(self, query: str) -> Optional[str]:
        """Get a quick answer for a query"""
        # Try instant answer first
        answer = self.duckduckgo.instant_answer(query)
        if answer:
            return answer

        # Fall back to first search result snippet
        response = self.search(query, max_results=1)
        if response.results:
            return response.results[0].snippet

        return None


# Global search engine
_search_engine: Optional[SearchEngine] = None


def get_search_engine(
    provider: str = "duckduckgo",
    brave_api_key: Optional[str] = None
) -> SearchEngine:
    """Get global search engine instance"""
    global _search_engine

    if _search_engine is None:
        _search_engine = SearchEngine(provider, brave_api_key)

    return _search_engine


def search(query: str, max_results: int = 10) -> SearchResponse:
    """Quick search function"""
    return get_search_engine().search(query, max_results)


def quick_search(query: str) -> str:
    """Quick formatted search results"""
    return get_search_engine().get_summary(query)


def answer(query: str) -> Optional[str]:
    """Get quick answer"""
    return get_search_engine().quick_answer(query)


__all__ = [
    "SearchProvider",
    "SearchResult",
    "SearchResponse",
    "DuckDuckGoSearch",
    "BraveSearch",
    "SearchEngine",
    "get_search_engine",
    "search",
    "quick_search",
    "answer",
]
