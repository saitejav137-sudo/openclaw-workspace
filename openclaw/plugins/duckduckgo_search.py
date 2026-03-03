"""
DuckDuckGo Search Plugin for OpenClaw

Wraps the existing DuckDuckGo search integration as a proper
SearchEngine plugin for the plugin system.
"""

from typing import Any, Dict, List, Optional

from core.plugin_system import SearchEngine, PluginManifest, PluginModule, PluginSlot
from core.logger import get_logger

logger = get_logger("plugin.duckduckgo")


class DuckDuckGoPlugin(SearchEngine):
    """
    DuckDuckGo search provider plugin.

    Delegates to `integrations.search.DuckDuckGoEngine` but exposes
    the standardized SearchEngine interface for the plugin system.
    """

    def __init__(self):
        self._engine = None
        self._total_searches: int = 0
        self._total_results: int = 0
        self._errors: int = 0

    def _get_engine(self):
        """Lazy-load the DuckDuckGo engine."""
        if self._engine is None:
            try:
                from openclaw.integrations.search import get_search_engine
                self._engine = get_search_engine("duckduckgo")
            except ImportError:
                try:
                    from integrations.search import get_search_engine
                    self._engine = get_search_engine("duckduckgo")
                except ImportError:
                    logger.error("Search integration module not available")
                    raise
        return self._engine

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo and return results.

        Returns list of dicts: [{"title": ..., "url": ..., "snippet": ...}]
        """
        try:
            engine = self._get_engine()
            self._total_searches += 1

            response = engine.search(query, max_results=max_results)

            results = []
            for r in response.results:
                results.append({
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet or "",
                })

            self._total_results += len(results)
            return results

        except Exception as e:
            self._errors += 1
            logger.error("DuckDuckGo search failed: %s", e)
            return []

    def suggest(self, query: str) -> List[str]:
        """Auto-complete suggestions (DuckDuckGo instant answers)."""
        try:
            import requests
            response = requests.get(
                "https://duckduckgo.com/ac/",
                params={"q": query, "type": "list"},
                timeout=5,
                headers={"User-Agent": "OpenClaw/2.0"},
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 1:
                    return data[1][:10]  # Return up to 10 suggestions
        except Exception:
            pass
        return []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_searches": self._total_searches,
            "total_results": self._total_results,
            "errors": self._errors,
        }


# ============== Plugin Module ==============

def create_plugin(config: Dict[str, Any] = None) -> DuckDuckGoPlugin:
    return DuckDuckGoPlugin()


MANIFEST = PluginManifest(
    name="duckduckgo-search",
    version="1.0.0",
    description="DuckDuckGo web search provider",
    slot=PluginSlot.SEARCH_ENGINE,
)

duckduckgo_module = PluginModule(manifest=MANIFEST, create=create_plugin)


__all__ = ["DuckDuckGoPlugin", "create_plugin", "MANIFEST", "duckduckgo_module"]
