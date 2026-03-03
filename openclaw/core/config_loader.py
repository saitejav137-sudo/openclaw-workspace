"""
Unified YAML Configuration Loader for OpenClaw

Inspired by ComposioHQ/agent-orchestrator's YAML config pattern.
One file to configure everything: plugins, agents, reactions, bots.

Features:
- YAML config with env var substitution (${VAR} syntax)
- Schema validation
- Default values
- Config reloading at runtime
"""

import os
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .logger import get_logger

logger = get_logger("config_loader")


# ============== Config Data Models ==============

@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    role: str = "executor"
    capabilities: List[str] = field(default_factory=list)
    max_concurrent: int = 1
    auto_restart: bool = True


@dataclass
class BotConfig:
    """Configuration for a Telegram bot."""
    gateway: str = "python"
    token: str = ""


@dataclass
class ReactionYAMLConfig:
    """YAML-friendly reaction config."""
    auto: bool = True
    action: str = "notify"
    message: str = ""
    retries: int = 2
    escalate_after: str = "10m"   # e.g. "5m", "1h"
    cooldown: str = "30s"


@dataclass
class NotificationRoutingConfig:
    """What priorities get sent where."""
    urgent: bool = True
    action: bool = True
    warning: bool = True
    info: bool = False


@dataclass
class NotifierChannelConfig:
    """Config for a notification channel."""
    plugin: str = "telegram"
    chat_id: str = ""
    routing: NotificationRoutingConfig = field(default_factory=NotificationRoutingConfig)


@dataclass
class OpenClawConfig:
    """Complete OpenClaw configuration."""
    # Server
    port: int = 8080
    
    # Plugin defaults
    defaults: Dict[str, str] = field(default_factory=lambda: {
        "llm": "minimax",
        "gateway": "telegram",
        "search": "duckduckgo",
        "storage": "file",
    })
    
    # Agent configs
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    
    # Reaction configs
    reactions: Dict[str, ReactionYAMLConfig] = field(default_factory=dict)
    
    # Bot configs
    bots: Dict[str, BotConfig] = field(default_factory=dict)
    
    # Notification channels
    notifications: Dict[str, NotifierChannelConfig] = field(default_factory=dict)
    
    # Lifecycle
    lifecycle: Dict[str, Any] = field(default_factory=lambda: {
        "poll_interval": "10s",
        "stuck_threshold": "10m",
        "idle_threshold": "5m",
        "max_auto_recoveries": 3,
        "auto_recover": True,
    })
    
    # File path (set by loader)
    _config_path: str = ""


# ============== Duration Parsing ==============

def parse_duration(value: str) -> float:
    """
    Parse a duration string to seconds.
    
    Supports: "30s", "5m", "1h", "1.5h", "90" (defaults to seconds)
    """
    if not value:
        return 0.0
    
    value = str(value).strip().lower()
    
    match = re.match(r'^(\d+\.?\d*)\s*(s|sec|m|min|h|hr|d|day)?$', value)
    if not match:
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid duration: '%s', defaulting to 0", value)
            return 0.0
    
    num = float(match.group(1))
    unit = match.group(2) or "s"
    
    multipliers = {
        "s": 1, "sec": 1,
        "m": 60, "min": 60,
        "h": 3600, "hr": 3600,
        "d": 86400, "day": 86400,
    }
    
    return num * multipliers.get(unit, 1)


# ============== Env Var Substitution ==============

def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} and ${VAR:-default} patterns."""
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        def replacer(match):
            var_name = match.group(1)
            default = match.group(3) if match.group(3) is not None else None
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            logger.warning("Environment variable '%s' not set and no default", var_name)
            return match.group(0)  # Leave as-is
        
        return re.sub(r'\$\{(\w+)(:-([^}]*))?\}', replacer, value)
    
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    
    return value


# ============== Config Loader ==============

class ConfigLoader:
    """
    Loads and manages OpenClaw configuration.

    Search order:
    1. Path passed to load()
    2. ./openclaw.yaml
    3. ~/.openclaw/openclaw.yaml
    4. Defaults
    """

    DEFAULT_PATHS = [
        "openclaw.yaml",
        "openclaw.yml",
        os.path.expanduser("~/.openclaw/openclaw.yaml"),
        os.path.expanduser("~/.openclaw/openclaw.yml"),
    ]

    def __init__(self):
        self._config: Optional[OpenClawConfig] = None
        self._raw_data: Dict[str, Any] = {}
        self._config_path: Optional[str] = None
        self._last_loaded: float = 0
        self._lock = threading.Lock()

    def load(self, config_path: str = None) -> OpenClawConfig:
        """Load config from YAML file."""
        import yaml  # Lazy import — only needed when loading

        # Find config file
        path = self._find_config(config_path)

        if path:
            logger.info("Loading config from: %s", path)
            with open(path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            
            # Substitute environment variables
            raw = _substitute_env_vars(raw)
            self._raw_data = raw
            self._config_path = path
        else:
            logger.info("No config file found, using defaults")
            raw = {}
            self._config_path = None

        # Build config from raw data
        config = self._build_config(raw)
        config._config_path = path or ""

        with self._lock:
            self._config = config
            self._last_loaded = time.time()

        logger.info("Config loaded: %d agents, %d reactions, %d bots",
                     len(config.agents), len(config.reactions), len(config.bots))
        return config

    def get_config(self) -> OpenClawConfig:
        """Get current config (loads defaults if not loaded)."""
        if self._config is None:
            return self.load()
        return self._config

    def reload(self) -> OpenClawConfig:
        """Reload config from the same file."""
        config = self.load(self._config_path)

        # Emit reload event
        try:
            from .event_bus import get_event_bus, EventType
            get_event_bus().emit(
                EventType.SYSTEM_CONFIG_RELOADED,
                f"Config reloaded from {self._config_path}",
                source="config_loader",
            )
        except Exception:
            pass

        return config

    def _find_config(self, explicit_path: str = None) -> Optional[str]:
        """Find the config file."""
        if explicit_path:
            path = os.path.expanduser(explicit_path)
            if os.path.exists(path):
                return path
            logger.warning("Config file not found: %s", path)
            return None

        for path in self.DEFAULT_PATHS:
            path = os.path.expanduser(path)
            if os.path.exists(path):
                return path

        return None

    def _build_config(self, raw: Dict[str, Any]) -> OpenClawConfig:
        """Build typed config from raw YAML data."""
        config = OpenClawConfig()

        # Server
        config.port = raw.get("port", config.port)

        # Defaults
        if "defaults" in raw:
            config.defaults.update(raw["defaults"])

        # Agents
        for name, agent_data in raw.get("agents", {}).items():
            if isinstance(agent_data, dict):
                config.agents[name] = AgentConfig(
                    role=agent_data.get("role", "executor"),
                    capabilities=agent_data.get("capabilities", []),
                    max_concurrent=agent_data.get("max_concurrent", 1),
                    auto_restart=agent_data.get("auto_restart", True),
                )

        # Reactions
        for name, reaction_data in raw.get("reactions", {}).items():
            if isinstance(reaction_data, dict):
                config.reactions[name] = ReactionYAMLConfig(
                    auto=reaction_data.get("auto", True),
                    action=reaction_data.get("action", "notify"),
                    message=reaction_data.get("message", ""),
                    retries=reaction_data.get("retries", 2),
                    escalate_after=str(reaction_data.get("escalate_after", "10m")),
                    cooldown=str(reaction_data.get("cooldown", "30s")),
                )

        # Bots
        for name, bot_data in raw.get("bots", {}).items():
            if isinstance(bot_data, dict):
                config.bots[name] = BotConfig(
                    gateway=bot_data.get("gateway", "python"),
                    token=bot_data.get("token", ""),
                )

        # Notifications
        for name, notif_data in raw.get("notifications", {}).items():
            if isinstance(notif_data, dict):
                routing_data = notif_data.get("routing", {})
                config.notifications[name] = NotifierChannelConfig(
                    plugin=notif_data.get("plugin", "telegram"),
                    chat_id=str(notif_data.get("chat_id", "")),
                    routing=NotificationRoutingConfig(
                        urgent=routing_data.get("urgent", True),
                        action=routing_data.get("action", True),
                        warning=routing_data.get("warning", True),
                        info=routing_data.get("info", False),
                    ),
                )

        # Lifecycle
        if "lifecycle" in raw:
            config.lifecycle.update(raw["lifecycle"])

        return config

    def validate(self) -> List[str]:
        """Validate current config and return list of issues."""
        issues = []
        config = self.get_config()

        # Check bot tokens
        for name, bot in config.bots.items():
            if not bot.token or bot.token.startswith("${"):
                issues.append(f"Bot '{name}': token not set or env var not resolved")

        # Check notification chat_ids
        for name, notif in config.notifications.items():
            if not notif.chat_id or notif.chat_id.startswith("${"):
                issues.append(f"Notification '{name}': chat_id not set")

        # Check agent capabilities
        for name, agent in config.agents.items():
            if not agent.capabilities:
                issues.append(f"Agent '{name}': no capabilities defined")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Export current config as dict (for display/debugging)."""
        from dataclasses import asdict
        config = self.get_config()
        return {
            "port": config.port,
            "defaults": config.defaults,
            "agents": {n: {"role": a.role, "capabilities": a.capabilities} for n, a in config.agents.items()},
            "reactions": {n: {"auto": r.auto, "action": r.action} for n, r in config.reactions.items()},
            "bots": {n: {"gateway": b.gateway, "token_set": bool(b.token)} for n, b in config.bots.items()},
            "notifications": {n: {"plugin": nc.plugin} for n, nc in config.notifications.items()},
            "lifecycle": config.lifecycle,
            "config_path": config._config_path,
        }


# ============== Global Instance ==============

_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get or create the global config loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config() -> OpenClawConfig:
    """Get the current config."""
    return get_config_loader().get_config()


__all__ = [
    "OpenClawConfig",
    "AgentConfig",
    "BotConfig",
    "ReactionYAMLConfig",
    "NotifierChannelConfig",
    "NotificationRoutingConfig",
    "LifecycleConfig",
    "ConfigLoader",
    "get_config_loader",
    "get_config",
    "parse_duration",
]
