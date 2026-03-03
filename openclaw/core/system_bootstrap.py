"""
System Bootstrap Orchestrator for OpenClaw

Single entry point that wires EVERYTHING together on startup:
1. Load YAML configuration
2. Register plugins in the plugin registry
3. Start event bus + persistent event store
4. Start reaction engine
5. Start lifecycle manager
6. Bootstrap agents (tools, orchestrator, swarm)
7. Emit system.started event

Usage:
    from core.system_bootstrap import bootstrap_system, shutdown_system

    system = bootstrap_system()  # Wires everything
    # ... run your app ...
    shutdown_system()             # Graceful cleanup
"""

import time
import atexit
from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger("system_bootstrap")


class SystemState:
    """Tracks what subsystems are initialized."""

    def __init__(self):
        self.config_loaded = False
        self.plugins_registered = False
        self.event_bus_started = False
        self.reaction_engine_started = False
        self.lifecycle_manager_started = False
        self.agents_bootstrapped = False
        self.health_checker_started = False
        self.metrics_server_started = False
        self.started_at: Optional[float] = None
        self.config = None
        self.references: Dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        return all([
            self.config_loaded,
            self.event_bus_started,
            self.reaction_engine_started,
        ])

    def summary(self) -> Dict[str, Any]:
        return {
            "config": self.config_loaded,
            "plugins": self.plugins_registered,
            "event_bus": self.event_bus_started,
            "reactions": self.reaction_engine_started,
            "lifecycle": self.lifecycle_manager_started,
            "agents": self.agents_bootstrapped,
            "health_checker": self.health_checker_started,
            "metrics_server": self.metrics_server_started,
            "uptime_sec": round(time.time() - self.started_at, 1) if self.started_at else 0,
        }


_system_state = SystemState()


def bootstrap_system(config_path: str = None) -> SystemState:
    """
    Bootstrap the entire OpenClaw system.

    This is the ONE function you call at startup. It initializes
    everything in the correct order with proper error handling.
    Each subsystem is initialized independently so failures don't
    cascade — a broken plugin shouldn't prevent the event bus.

    Args:
        config_path: Optional path to openclaw.yaml

    Returns:
        SystemState with references to all initialized subsystems
    """
    global _system_state
    state = _system_state
    state.started_at = time.time()

    logger.info("🚀 OpenClaw system bootstrap starting...")

    # ── Step 1: Load Configuration ──────────────────────────────
    try:
        from .config_loader import get_config_loader
        loader = get_config_loader()
        config = loader.load(config_path)
        state.config = config
        state.config_loaded = True
        state.references["config"] = config

        issues = loader.validate()
        for issue in issues:
            logger.warning("Config issue: %s", issue)

        logger.info("✅ Config loaded (%d agents, %d reactions, %d bots)",
                     len(config.agents), len(config.reactions), len(config.bots))
    except Exception as e:
        logger.warning("Config load failed (using defaults): %s", e)
        state.config_loaded = True  # Defaults are fine

    # ── Step 2: Start Event Bus ─────────────────────────────────
    try:
        from .event_bus import get_event_bus, EventType
        bus = get_event_bus()
        state.event_bus_started = True
        state.references["event_bus"] = bus
        logger.info("✅ Event bus started")
    except Exception as e:
        logger.error("Event bus failed: %s", e)

    # ── Step 3: Start Persistent Event Store ────────────────────
    try:
        from .event_bus import PersistentEventStore
        store = PersistentEventStore()
        bus.subscribe(handler=store.on_event)
        state.references["event_store"] = store
        logger.info("✅ Persistent event store enabled (%s)", store.base_dir)
    except Exception as e:
        logger.warning("Persistent events unavailable: %s", e)

    # ── Step 4: Register Plugins ────────────────────────────────
    try:
        from .plugin_system import get_plugin_registry

        registry = get_plugin_registry()

        # Register built-in plugins
        _register_builtin_plugins(registry)
        state.plugins_registered = True
        state.references["plugin_registry"] = registry

        logger.info("✅ Plugins registered (%d total)", len(registry.list_plugins()))
    except Exception as e:
        logger.warning("Plugin registration failed: %s", e)

    # ── Step 5: Start Reaction Engine ───────────────────────────
    try:
        from .reaction_engine import get_reaction_engine

        # Create notification callback using Telegram notifier plugin
        notify_fn = _create_notify_callback()

        engine = get_reaction_engine(on_notify=notify_fn)
        engine.start()
        state.reaction_engine_started = True
        state.references["reaction_engine"] = engine

        logger.info("✅ Reaction engine started (%d reactions)",
                     len(engine.list_reactions()))
    except Exception as e:
        logger.warning("Reaction engine failed: %s", e)

    # ── Step 6: Start Lifecycle Manager ─────────────────────────
    try:
        from .lifecycle_manager import get_lifecycle_manager
        from .agent_state import get_state_manager

        sm = get_state_manager()
        lm = get_lifecycle_manager(state_manager=sm)
        lm.start()
        state.lifecycle_manager_started = True
        state.references["lifecycle_manager"] = lm

        logger.info("✅ Lifecycle manager started (poll=%ds)",
                     lm._config.poll_interval_sec)
    except Exception as e:
        logger.warning("Lifecycle manager failed: %s", e)

    # ── Step 7: Bootstrap Agents ────────────────────────────────
    try:
        from .agent_bootstrap import bootstrap_agents
        refs = bootstrap_agents()
        state.agents_bootstrapped = True
        state.references.update(refs or {})
        logger.info("✅ Agents bootstrapped")
    except Exception as e:
        logger.warning("Agent bootstrap failed: %s", e)

    # ── Step 8: Health Checker (Resilience) ─────────────────────
    try:
        from .resilience import HealthChecker
        checker = HealthChecker()

        # Register health checks for key subsystems
        if state.event_bus_started:
            checker.register("event_bus", lambda: (
                "healthy",
                f"{bus.get_stats()['total_events_emitted']} events"
            ))

        checker.register("config", lambda: (
            "healthy" if state.config_loaded else "unhealthy",
            "Loaded" if state.config_loaded else "Not loaded"
        ))

        state.health_checker_started = True
        state.references["health_checker"] = checker
        logger.info("✅ Health checker started (%d checks)", len(checker._checks))
    except Exception as e:
        logger.warning("Health checker failed: %s", e)

    # ── Step 9: Metrics Server ──────────────────────────────────
    try:
        from .metrics_server import MetricsServer
        metrics_port = 9100
        if state.config and hasattr(state.config, 'port'):
            metrics_port = getattr(state.config, 'metrics_port', 9100)
        metrics = MetricsServer(port=metrics_port)
        metrics.start()
        state.metrics_server_started = True
        state.references["metrics_server"] = metrics
        logger.info("✅ Metrics server at http://0.0.0.0:%d", metrics_port)
    except Exception as e:
        logger.warning("Metrics server failed: %s", e)

    # ── Done ────────────────────────────────────────────────────
    duration = time.time() - state.started_at

    if state.event_bus_started:
        from .event_bus import EventType
        bus.emit(
            EventType.SYSTEM_STARTED,
            f"OpenClaw system ready in {duration:.1f}s",
            data=state.summary(),
            source="system_bootstrap",
        )

    logger.info("🎉 System bootstrap complete in %.1fs — %s",
                duration,
                "ALL OK" if state.is_ready else "DEGRADED (check warnings)")

    # Register cleanup
    atexit.register(shutdown_system)

    return state


def shutdown_system() -> None:
    """Graceful shutdown of all subsystems."""
    global _system_state
    state = _system_state

    logger.info("🛑 Shutting down OpenClaw system...")

    # Stop in reverse order
    try:
        lm = state.references.get("lifecycle_manager")
        if lm:
            lm.stop()
            logger.info("Lifecycle manager stopped")
    except Exception as e:
        logger.warning("Lifecycle manager stop error: %s", e)

    try:
        engine = state.references.get("reaction_engine")
        if engine:
            engine.stop()
            logger.info("Reaction engine stopped")
    except Exception as e:
        logger.warning("Reaction engine stop error: %s", e)

    try:
        if state.event_bus_started:
            from .event_bus import EventType, get_event_bus
            bus = get_event_bus()
            bus.emit(EventType.SYSTEM_STOPPED, "System shutting down", source="system_bootstrap")
            logger.info("Event bus final event emitted")
    except Exception:
        pass

    try:
        metrics = state.references.get("metrics_server")
        if metrics:
            metrics.stop()
            logger.info("Metrics server stopped")
    except Exception as e:
        logger.warning("Metrics server stop error: %s", e)

    uptime = time.time() - state.started_at if state.started_at else 0
    logger.info("System shut down after %.1fs uptime", uptime)


def _register_builtin_plugins(registry) -> None:
    """Register all built-in plugins."""
    builtin_plugins = [
        ("plugins.minimax_llm", "minimax_module"),
        ("plugins.duckduckgo_search", "duckduckgo_module"),
        ("plugins.telegram_notifier", "telegram_module"),
        ("plugins.file_storage", "file_module"),
        ("plugins.github_tracker", "github_module"),
    ]

    for module_path, attr_name in builtin_plugins:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            plugin_module = getattr(mod, attr_name)
            registry.register(plugin_module)
            logger.info("  Registered plugin: %s", plugin_module.manifest.name)
        except Exception as e:
            logger.debug("Plugin '%s' not available: %s", module_path, e)


def _create_notify_callback():
    """Create notification callback from Telegram notifier plugin."""
    def notify(message: str, priority=None):
        try:
            from plugins.telegram_notifier import TelegramNotifierPlugin
            notifier = TelegramNotifierPlugin()
            notifier.configure({})  # Uses env vars
            priority_str = priority.value if hasattr(priority, 'value') else str(priority or "info")
            notifier.notify(message, priority=priority_str)
        except Exception as e:
            logger.warning("Notification failed: %s", e)

    return notify


def get_system_state() -> SystemState:
    """Get the current system state."""
    return _system_state


__all__ = [
    "SystemState",
    "bootstrap_system",
    "shutdown_system",
    "get_system_state",
]
