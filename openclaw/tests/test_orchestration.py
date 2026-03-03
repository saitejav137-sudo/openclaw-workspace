"""
Comprehensive tests for OpenClaw orchestration modules.

Tests:
- core/plugin_system.py
- core/event_bus.py (+ PersistentEventStore)
- core/reaction_engine.py
- core/config_loader.py
- core/lifecycle_manager.py
- core/system_bootstrap.py
- plugins/file_storage.py
"""

import os
import sys
import time
import json
import tempfile
import threading
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Plugin System Tests
# ============================================================

class TestPluginSystem:
    """Tests for core/plugin_system.py"""

    def test_plugin_slots_exist(self):
        from core.plugin_system import PluginSlot
        assert hasattr(PluginSlot, "LLM_PROVIDER")
        assert hasattr(PluginSlot, "GATEWAY")
        assert hasattr(PluginSlot, "SEARCH_ENGINE")
        assert hasattr(PluginSlot, "NOTIFIER")
        assert hasattr(PluginSlot, "STORAGE")
        assert hasattr(PluginSlot, "TASK_TRACKER")

    def test_registry_create(self):
        from core.plugin_system import PluginRegistry, PluginSlot
        registry = PluginRegistry()
        for slot in PluginSlot:
            assert len(registry.list_plugins(slot)) == 0

    def test_registry_register(self):
        from core.plugin_system import (
            PluginRegistry, PluginManifest, PluginModule, PluginSlot
        )

        manifest = PluginManifest(
            name="test-plugin",
            slot=PluginSlot.STORAGE,
            description="Test storage plugin",
        )
        module = PluginModule(manifest=manifest, create=lambda cfg: MagicMock())

        registry = PluginRegistry()
        registry.register(module)

        plugins = registry.list_plugins(PluginSlot.STORAGE)
        assert len(plugins) == 1
        assert plugins[0].name == "test-plugin"

    def test_registry_activate(self):
        from core.plugin_system import (
            PluginRegistry, PluginManifest, PluginModule, PluginSlot, Storage
        )

        class TestStorage(Storage):
            @property
            def name(self): return "test"
            async def get(self, key): return None
            async def set(self, key, value, ttl=None): pass
            async def delete(self, key): return True
            async def exists(self, key): return False

        manifest = PluginManifest(
            name="test-storage",
            slot=PluginSlot.STORAGE,
            description="Test storage",
        )
        module = PluginModule(manifest=manifest, create=lambda cfg: TestStorage())

        registry = PluginRegistry()
        registry.register(module)
        instance = registry.activate(PluginSlot.STORAGE, "test-storage")
        assert instance is not None
        assert isinstance(instance, Storage)

    def test_abstract_interfaces(self):
        from core.plugin_system import LLMProvider, Gateway, SearchEngine, Notifier, Storage
        # All should be importable ABCs
        assert LLMProvider is not None
        assert Gateway is not None
        assert SearchEngine is not None
        assert Notifier is not None
        assert Storage is not None

    def test_list_all(self):
        from core.plugin_system import PluginRegistry
        registry = PluginRegistry()
        all_plugins = registry.list_all()
        assert isinstance(all_plugins, dict)
        assert len(all_plugins) == 6  # 6 slots

    def test_get_stats(self):
        from core.plugin_system import PluginRegistry
        registry = PluginRegistry()
        stats = registry.get_stats()
        assert stats["total_registered"] == 0
        assert stats["active_slots"] == 0
        assert "slots" in stats


# ============================================================
# Event Bus Tests
# ============================================================

class TestEventBus:
    """Tests for core/event_bus.py"""

    def test_create_bus(self):
        from core.event_bus import EventBus
        bus = EventBus(max_history=100)
        assert bus._event_count == 0

    def test_emit_event(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()
        event = bus.emit(EventType.SYSTEM_STARTUP, "test startup")
        assert event.type == EventType.SYSTEM_STARTUP
        assert event.message == "test startup"
        assert bus._event_count == 1

    def test_subscribe_and_receive(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()
        received = []

        bus.subscribe(
            handler=lambda e: received.append(e),
            event_types={EventType.AGENT_STUCK},
        )

        bus.emit(EventType.AGENT_STUCK, "test stuck")
        bus.emit(EventType.AGENT_IDLE, "test idle")  # Should NOT trigger

        assert len(received) == 1
        assert received[0].type == EventType.AGENT_STUCK

    def test_subscribe_by_category(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()
        received = []

        bus.subscribe(
            handler=lambda e: received.append(e),
            categories={"task"},
        )

        bus.emit(EventType.TASK_CREATED, "task 1")
        bus.emit(EventType.TASK_COMPLETED, "task 2")
        bus.emit(EventType.AGENT_IDLE, "agent idle")  # Different category

        assert len(received) == 2

    def test_subscribe_by_priority(self):
        from core.event_bus import EventBus, EventType, EventPriority
        bus = EventBus()
        received = []

        bus.subscribe(
            handler=lambda e: received.append(e),
            min_priority=EventPriority.WARNING,
        )

        bus.emit(EventType.AGENT_STUCK, "stuck", priority=EventPriority.WARNING)
        bus.emit(EventType.AGENT_IDLE, "idle", priority=EventPriority.INFO)

        assert len(received) == 1

    def test_unsubscribe(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()
        received = []
        sub_id = bus.subscribe(handler=lambda e: received.append(e))

        bus.emit(EventType.SYSTEM_STARTUP, "before")
        assert len(received) == 1

        bus.unsubscribe(sub_id)
        bus.emit(EventType.SYSTEM_STARTUP, "after")
        assert len(received) == 1  # No new events

    def test_history(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()

        for i in range(5):
            bus.emit(EventType.TASK_CREATED, f"task {i}")

        history = bus.get_history(limit=3)
        assert len(history) == 3
        assert history[0].message == "task 4"  # Newest first

    def test_history_filter_by_type(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()

        bus.emit(EventType.TASK_CREATED, "task")
        bus.emit(EventType.AGENT_IDLE, "agent")

        history = bus.get_history(event_type=EventType.TASK_CREATED)
        assert len(history) == 1

    def test_event_types_complete(self):
        from core.event_bus import EventType
        types = list(EventType)
        # Should have 30+ event types
        assert len(types) >= 30

    def test_system_started_stopped_types(self):
        from core.event_bus import EventType
        assert EventType.SYSTEM_STARTED == "system.started"
        assert EventType.SYSTEM_STOPPED == "system.stopped"

    def test_get_stats(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()
        bus.emit(EventType.SYSTEM_STARTUP, "test")
        stats = bus.get_stats()
        assert stats["total_events_emitted"] == 1

    def test_count_by_type(self):
        from core.event_bus import EventBus, EventType
        bus = EventBus()
        bus.emit(EventType.TASK_CREATED, "a")
        bus.emit(EventType.TASK_CREATED, "b")
        bus.emit(EventType.AGENT_IDLE, "c")
        counts = bus.count_by_type()
        assert counts["task.created"] == 2
        assert counts["agent.idle"] == 1


# ============================================================
# Persistent Event Store Tests
# ============================================================

class TestPersistentEventStore:
    """Tests for PersistentEventStore in event_bus.py"""

    def test_persist_event(self):
        from core.event_bus import PersistentEventStore, EventBus, EventType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEventStore(base_dir=tmpdir)
            bus = EventBus()
            bus.subscribe(handler=store.on_event)

            bus.emit(EventType.SYSTEM_STARTUP, "test persist")

            events = store.load_events(limit=10)
            assert len(events) == 1
            assert events[0]["type"] == "system.startup"
            assert events[0]["message"] == "test persist"

            store.close()

    def test_multiple_events(self):
        from core.event_bus import PersistentEventStore, EventBus, EventType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEventStore(base_dir=tmpdir)
            bus = EventBus()
            bus.subscribe(handler=store.on_event)

            for i in range(20):
                bus.emit(EventType.TASK_CREATED, f"task {i}")

            events = store.load_events(limit=100)
            assert len(events) == 20

            store.close()

    def test_stats(self):
        from core.event_bus import PersistentEventStore, EventBus, EventType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEventStore(base_dir=tmpdir)
            bus = EventBus()
            bus.subscribe(handler=store.on_event)

            bus.emit(EventType.TASK_CREATED, "task 1")
            bus.emit(EventType.TASK_COMPLETED, "task 2")

            stats = store.get_stats()
            assert stats["total_written"] == 2
            assert len(stats["files"]) == 1

            store.close()


# ============================================================
# Reaction Engine Tests
# ============================================================

class TestReactionEngine:
    """Tests for core/reaction_engine.py"""

    def test_default_reactions(self):
        from core.reaction_engine import ReactionEngine
        engine = ReactionEngine()
        reactions = engine.list_reactions()
        assert "agent-stuck" in reactions
        assert "agent-error" in reactions
        assert "task-failed" in reactions
        assert "swarm-completed" in reactions

    def test_add_remove_reaction(self):
        from core.reaction_engine import ReactionEngine, ReactionConfig, ReactionAction
        from core.event_bus import EventType

        engine = ReactionEngine()
        engine.add_reaction("custom", ReactionConfig(
            event_type=EventType.TASK_COMPLETED,
            action=ReactionAction.NOTIFY,
        ))

        assert "custom" in engine.list_reactions()
        engine.remove_reaction("custom")
        assert "custom" not in engine.list_reactions()

    def test_enable_disable(self):
        from core.reaction_engine import ReactionEngine
        engine = ReactionEngine()

        engine.disable_reaction("agent-stuck")
        reactions = engine.list_reactions()
        assert reactions["agent-stuck"]["enabled"] is False

        engine.enable_reaction("agent-stuck")
        reactions = engine.list_reactions()
        assert reactions["agent-stuck"]["enabled"] is True

    def test_reaction_trigger(self):
        from core.reaction_engine import ReactionEngine
        from core.event_bus import EventBus, EventType

        bus = EventBus()
        engine = ReactionEngine(event_bus=bus)
        engine.start()

        # Emit a swarm-completed event (triggers notify action)
        bus.emit(EventType.SWARM_COMPLETED, "swarm done", data={"task_id": "t1"})

        stats = engine.get_stats()
        assert stats["total_reactions_triggered"] >= 1

        engine.stop()

    def test_cooldown(self):
        from core.reaction_engine import ReactionEngine, ReactionConfig, ReactionAction
        from core.event_bus import EventBus, EventType

        bus = EventBus()
        engine = ReactionEngine(event_bus=bus, reactions={
            "test": ReactionConfig(
                event_type=EventType.TASK_FAILED,
                action=ReactionAction.NOTIFY,
                cooldown_sec=60,  # 60s cooldown
            ),
        })
        engine.start()

        bus.emit(EventType.TASK_FAILED, "fail 1", source="agent-a")
        bus.emit(EventType.TASK_FAILED, "fail 2", source="agent-a")  # Should be in cooldown

        stats = engine.get_stats()
        assert stats["total_reactions_triggered"] == 1  # Only first fires

        engine.stop()

    def test_stats(self):
        from core.reaction_engine import ReactionEngine
        engine = ReactionEngine()
        stats = engine.get_stats()
        assert "is_running" in stats
        assert "total_reactions_triggered" in stats
        assert "configured_reactions" in stats


# ============================================================
# Config Loader Tests
# ============================================================

class TestConfigLoader:
    """Tests for core/config_loader.py"""

    def test_parse_duration_seconds(self):
        from core.config_loader import parse_duration
        assert parse_duration("30s") == 30
        assert parse_duration("30sec") == 30

    def test_parse_duration_minutes(self):
        from core.config_loader import parse_duration
        assert parse_duration("5m") == 300
        assert parse_duration("5min") == 300

    def test_parse_duration_hours(self):
        from core.config_loader import parse_duration
        assert parse_duration("1h") == 3600
        assert parse_duration("1.5h") == 5400

    def test_parse_duration_days(self):
        from core.config_loader import parse_duration
        assert parse_duration("1d") == 86400

    def test_parse_duration_raw_number(self):
        from core.config_loader import parse_duration
        assert parse_duration("90") == 90
        assert parse_duration("") == 0

    def test_default_config(self):
        from core.config_loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load()  # No config file → defaults
        assert config.port == 8080
        assert "llm" in config.defaults
        assert "gateway" in config.defaults

    def test_yaml_loading(self):
        from core.config_loader import ConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
port: 9090
agents:
  researcher:
    role: researcher
    capabilities: [web_search, analysis]
reactions:
  agent-stuck:
    auto: true
    action: restart-agent
""")
            f.flush()
            config_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load(config_path)
            assert config.port == 9090
            assert "researcher" in config.agents
            assert config.agents["researcher"].role == "researcher"
            assert "agent-stuck" in config.reactions
        finally:
            os.unlink(config_path)

    def test_env_var_substitution(self):
        from core.config_loader import _substitute_env_vars

        os.environ["TEST_TOKEN"] = "abc123"
        result = _substitute_env_vars("${TEST_TOKEN}")
        assert result == "abc123"
        del os.environ["TEST_TOKEN"]

    def test_env_var_default(self):
        from core.config_loader import _substitute_env_vars
        result = _substitute_env_vars("${NONEXISTENT_VAR:-fallback}")
        assert result == "fallback"

    def test_validate(self):
        from core.config_loader import ConfigLoader
        loader = ConfigLoader()
        loader.load()
        issues = loader.validate()
        assert isinstance(issues, list)

    def test_to_dict(self):
        from core.config_loader import ConfigLoader
        loader = ConfigLoader()
        loader.load()
        d = loader.to_dict()
        assert "port" in d
        assert "defaults" in d


# ============================================================
# Lifecycle Manager Tests
# ============================================================

class TestLifecycleManager:
    """Tests for core/lifecycle_manager.py"""

    def test_create_manager(self):
        from core.lifecycle_manager import LifecycleManager, LifecycleConfig
        from core.agent_state import AgentStateManager

        sm = AgentStateManager()
        config = LifecycleConfig(poll_interval_sec=1)
        lm = LifecycleManager(config=config, state_manager=sm)
        assert lm is not None

    def test_stats(self):
        from core.lifecycle_manager import LifecycleManager, LifecycleConfig
        from core.agent_state import AgentStateManager

        sm = AgentStateManager()
        lm = LifecycleManager(config=LifecycleConfig(), state_manager=sm)
        stats = lm.get_stats()
        assert "total_checks" in stats
        assert "recoveries_attempted" in stats

    def test_get_states(self):
        from core.lifecycle_manager import LifecycleManager, LifecycleConfig
        from core.agent_state import AgentStateManager

        sm = AgentStateManager()
        lm = LifecycleManager(config=LifecycleConfig(), state_manager=sm)
        states = lm.get_states()
        assert isinstance(states, dict)

    def test_start_stop(self):
        from core.lifecycle_manager import LifecycleManager, LifecycleConfig
        from core.agent_state import AgentStateManager

        sm = AgentStateManager()
        lm = LifecycleManager(config=LifecycleConfig(poll_interval_sec=60), state_manager=sm)
        lm.start()
        assert lm._running
        lm.stop()
        assert not lm._running


# ============================================================
# Agent State Tests (Upgraded)
# ============================================================

class TestAgentState:
    """Tests for the upgraded core/agent_state.py"""

    def test_twelve_statuses(self):
        from core.agent_state import AgentStatus
        assert len(AgentStatus) == 12

    def test_six_activities(self):
        from core.agent_state import ActivityState
        assert len(ActivityState) == 6

    def test_valid_transitions(self):
        from core.agent_state import VALID_TRANSITIONS, AgentStatus
        # SPAWNING should be able to transition to IDLE
        assert AgentStatus.IDLE in VALID_TRANSITIONS[AgentStatus.SPAWNING]

    def test_terminal_statuses(self):
        from core.agent_state import TERMINAL_STATUSES, AgentStatus
        assert AgentStatus.COMPLETED in TERMINAL_STATUSES
        assert AgentStatus.TERMINATED in TERMINAL_STATUSES

    def test_state_manager_create(self):
        from core.agent_state import AgentStateManager
        sm = AgentStateManager()
        assert sm is not None

    def test_state_manager_ops(self):
        from core.agent_state import AgentStateManager
        sm = AgentStateManager()
        assert sm is not None
        assert isinstance(sm._states, dict)


# ============================================================
# File Storage Plugin Tests (using direct import from module file)
# ============================================================

class TestFileStoragePlugin:
    """Tests for plugins/file_storage.py (imported directly)."""

    @pytest.fixture(autouse=True)
    def setup_storage(self, tmp_path):
        """Import FileStoragePlugin directly to avoid plugins/__init__.py issues."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "file_storage",
            os.path.join(os.path.dirname(__file__), "..", "plugins", "file_storage.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.FileStoragePlugin = mod.FileStoragePlugin
        self.storage = self.FileStoragePlugin()
        self.storage.configure({"base_dir": str(tmp_path)})

    def test_save_and_load(self):
        assert self.storage.save("key1", {"data": "hello"})
        result = self.storage.load("key1")
        assert result == {"data": "hello"}

    def test_delete(self):
        self.storage.save("temp_key", "temp_value")
        assert self.storage.exists("temp_key")
        assert self.storage.delete("temp_key")
        assert not self.storage.exists("temp_key")

    def test_list_keys(self):
        self.storage.save("alpha", 1)
        self.storage.save("beta", 2)
        self.storage.save("gamma", 3)

        keys = self.storage.list_keys()
        assert len(keys) == 3
        assert "alpha" in keys

    def test_namespaces(self):
        self.storage.save("k1", "v1", namespace="ns1")
        self.storage.save("k1", "v2", namespace="ns2")

        assert self.storage.load("k1", namespace="ns1") == "v1"
        assert self.storage.load("k1", namespace="ns2") == "v2"

    def test_stats(self):
        self.storage.save("test", "data")
        stats = self.storage.get_stats()
        assert stats["total_items"] == 1
        assert stats["total_writes"] == 1

    def test_clear_namespace(self):
        self.storage.save("a", 1)
        self.storage.save("b", 2)
        deleted = self.storage.clear_namespace()
        assert deleted == 2
        assert self.storage.list_keys() == []

    def test_load_missing_key_returns_default(self):
        result = self.storage.load("nonexistent", default="fallback")
        assert result == "fallback"


# ============================================================
# System Bootstrap Tests
# ============================================================

class TestSystemBootstrap:
    """Tests for core/system_bootstrap.py"""

    def test_system_state(self):
        from core.system_bootstrap import SystemState
        state = SystemState()
        assert not state.is_ready
        state.config_loaded = True
        state.event_bus_started = True
        state.reaction_engine_started = True
        assert state.is_ready

    def test_summary(self):
        from core.system_bootstrap import SystemState
        state = SystemState()
        state.started_at = time.time()
        summary = state.summary()
        assert "config" in summary
        assert "plugins" in summary
        assert "uptime_sec" in summary


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_event_bus_to_reaction_engine(self):
        """Test that events flow from bus to reaction engine."""
        from core.event_bus import EventBus, EventType
        from core.reaction_engine import ReactionEngine

        bus = EventBus()
        engine = ReactionEngine(event_bus=bus)
        engine.start()

        # Emit health degraded event
        bus.emit(EventType.HEALTH_DEGRADED, "CPU at 95%", data={"cpu": 95})

        stats = engine.get_stats()
        assert stats["total_reactions_triggered"] >= 1

        engine.stop()

    def test_event_bus_to_persistent_store(self):
        """Test that events are persisted to disk."""
        from core.event_bus import EventBus, EventType, PersistentEventStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEventStore(base_dir=tmpdir)
            bus = EventBus()
            bus.subscribe(handler=store.on_event)

            for i in range(10):
                bus.emit(EventType.TASK_CREATED, f"task {i}")

            events = store.load_events(limit=100)
            assert len(events) == 10

            store.close()

    def test_plugin_registry_lifecycle(self):
        """Test full plugin lifecycle with proper interface."""
        from core.plugin_system import (
            PluginRegistry, PluginManifest, PluginModule, PluginSlot, Storage
        )

        class InMemStorage(Storage):
            @property
            def name(self): return "in-memory"
            async def get(self, key): return self._data.get(key)
            async def set(self, key, value, ttl=None): self._data[key] = value
            async def delete(self, key): return self._data.pop(key, None) is not None
            async def exists(self, key): return key in self._data
            def __init__(self): self._data = {}

        registry = PluginRegistry()
        module = PluginModule(
            manifest=PluginManifest(
                name="mem-storage",
                slot=PluginSlot.STORAGE,
                description="In-memory storage",
            ),
            create=lambda cfg: InMemStorage(),
        )
        registry.register(module)
        instance = registry.activate(PluginSlot.STORAGE, "mem-storage")
        assert isinstance(instance, Storage)
        assert registry.is_active(PluginSlot.STORAGE)

    def test_full_event_chain(self):
        """Test: event → subscription → handler → persistent store."""
        from core.event_bus import EventBus, EventType, PersistentEventStore

        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            store = PersistentEventStore(base_dir=tmpdir)
            handler_log = []

            bus.subscribe(handler=store.on_event)
            bus.subscribe(
                handler=lambda e: handler_log.append(e.message),
                categories={"swarm"},
            )

            bus.emit(EventType.SWARM_STARTED, "swarm 1")
            bus.emit(EventType.SWARM_COMPLETED, "swarm 2")
            bus.emit(EventType.TASK_CREATED, "task 1")  # Not in "swarm" category

            # Handler should have 2 swarm events
            assert len(handler_log) == 2
            # Store should have ALL 3 events
            assert store.get_stats()["total_written"] == 3

            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
