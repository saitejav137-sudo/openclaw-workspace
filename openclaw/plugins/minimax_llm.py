"""
MiniMax LLM Provider Plugin for OpenClaw

Wraps the MiniMax AI API as a proper LLMProvider plugin for the
plugin system. Replaces the standalone `_call_llm()` function with
a pluggable, configurable provider.
"""

import os
import json
import time
import requests
from typing import Any, Dict, List, Optional, Generator

from core.plugin_system import LLMProvider, PluginManifest, PluginModule, PluginSlot
from core.logger import get_logger

logger = get_logger("plugin.minimax_llm")


class MiniMaxLLMPlugin(LLMProvider):
    """
    MiniMax AI LLM provider — production plugin.

    Supports:
    - complete(): Synchronous completion
    - stream(): Streaming token-by-token response
    - embed(): Text embeddings (delegates to embeddings.py)

    Config:
        api_key: MiniMax API key (or set MINIMAX_API_KEY env)
        model: Model name (default: MiniMax-M2.5-Lightning)
        base_url: API endpoint
        max_tokens: Default max tokens
        temperature: Default temperature
    """

    def __init__(self):
        self.api_key: Optional[str] = None
        self.model: str = "MiniMax-M2.5-Lightning"
        self.base_url: str = "https://api.minimax.io/anthropic/v1/messages"
        self.max_tokens: int = 1024
        self.temperature: float = 0.7
        self._total_calls: int = 0
        self._total_tokens: int = 0
        self._errors: int = 0

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure from YAML or dict."""
        self.model = config.get("model", self.model)
        self.base_url = config.get("base_url", self.base_url)
        self.max_tokens = config.get("max_tokens", self.max_tokens)
        self.temperature = config.get("temperature", self.temperature)

        # API key: config > env > credentials file
        self.api_key = config.get("api_key") or self._load_api_key()

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or credentials file."""
        api_key = os.getenv("MINIMAX_API_KEY")
        if api_key:
            return api_key

        key_path = os.path.expanduser("~/.openclaw/credentials/keys.json")
        if os.path.exists(key_path):
            try:
                with open(key_path) as f:
                    data = json.load(f)
                return data.get("providers", {}).get("minimax", {}).get("default", {}).get("api_key")
            except Exception:
                pass
        return None

    def complete(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        max_tokens: int = None,
        temperature: float = None,
        **kwargs,
    ) -> str:
        """Synchronous LLM completion."""
        if not self.api_key:
            self.api_key = self._load_api_key()
        if not self.api_key:
            return "Error: MiniMax API key not configured."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            self._total_calls += 1
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)

            if response.status_code == 200:
                result = response.json()
                # Extract text from response blocks
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        self._total_tokens += len(text) // 4  # rough estimate
                        return text

                # Fallback: thinking blocks only
                thinking = []
                for block in result.get("content", []):
                    if block.get("type") == "thinking" and "thinking" in block:
                        thinking.append(block["thinking"])
                if thinking:
                    return "\n".join(thinking)

                return "AI Error: Empty response"

            self._errors += 1
            return f"AI Error: HTTP {response.status_code}"

        except Exception as e:
            self._errors += 1
            return f"AI Error: {e}"

    def stream(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        max_tokens: int = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streaming LLM completion — yields tokens as they arrive."""
        if not self.api_key:
            self.api_key = self._load_api_key()
        if not self.api_key:
            yield "Error: API key not configured"
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            self._total_calls += 1
            response = requests.post(
                self.base_url, headers=headers, json=data,
                timeout=120, stream=True,
            )

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    chunk_str = decoded[6:]
                    if chunk_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            self._errors += 1
            yield f"\nStreaming error: {e}"

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            from core.embeddings import get_embedding_provider
            provider = get_embedding_provider()
            return provider.embed(text)
        except Exception:
            # Fallback: simple bag-of-words embedding
            words = text.lower().split()
            vocab = list(set(words))
            embedding = [words.count(w) / max(len(words), 1) for w in vocab[:128]]
            # Pad to fixed size
            while len(embedding) < 128:
                embedding.append(0.0)
            return embedding[:128]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_calls": self._total_calls,
            "total_tokens_approx": self._total_tokens,
            "errors": self._errors,
        }


# ============== Plugin Module (for plugin_system.py registration) ==============

def create_plugin(config: Dict[str, Any] = None) -> MiniMaxLLMPlugin:
    """Factory function for plugin system."""
    plugin = MiniMaxLLMPlugin()
    if config:
        plugin.configure(config)
    return plugin


MANIFEST = PluginManifest(
    name="minimax-llm",
    version="1.0.0",
    description="MiniMax AI LLM provider (M2.5-Lightning)",
    slot=PluginSlot.LLM_PROVIDER,
)

# PluginModule for the registry
minimax_module = PluginModule(manifest=MANIFEST, create=create_plugin)


__all__ = ["MiniMaxLLMPlugin", "create_plugin", "MANIFEST", "minimax_module"]
