"""
OpenClaw Telemetry & Observability — Phase 6

Structured distributed tracing, metrics collection, and
self-diagnostic health reporting for production monitoring.
"""

import time
import uuid
import threading
import json
import os
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger("telemetry")


# ============== Trace Context ==============

@dataclass
class Span:
    """A single unit of work in a distributed trace."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation: str
    start_time: float
    end_time: float = 0.0
    status: str = "ok"  # ok, error, timeout
    tags: Dict[str, str] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, attributes: Dict = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "tags": self.tags,
            "events": self.events,
        }


class Tracer:
    """
    Distributed tracing with span collection.

    Provides context-managed span creation, parent-child
    relationship tracking, and exportable trace data.
    """

    def __init__(self, service_name: str = "openclaw", max_spans: int = 10000):
        self.service_name = service_name
        self._max_spans = max_spans
        self._spans: List[Span] = []
        self._active_span: Dict[int, Span] = {}  # thread_id -> active span
        self._lock = threading.Lock()

    @contextmanager
    def span(self, operation: str, tags: Dict[str, str] = None):
        """Create a span as a context manager."""
        thread_id = threading.get_ident()
        parent = self._active_span.get(thread_id)

        s = Span(
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16],
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            operation=operation,
            start_time=time.time(),
            tags={"service": self.service_name, **(tags or {})},
        )

        self._active_span[thread_id] = s
        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise
        finally:
            s.end_time = time.time()
            with self._lock:
                self._spans.append(s)
                if len(self._spans) > self._max_spans:
                    self._spans = self._spans[-self._max_spans:]
            # Restore parent
            if parent:
                self._active_span[thread_id] = parent
            else:
                self._active_span.pop(thread_id, None)

    def get_traces(self, limit: int = 100) -> List[Dict]:
        """Get recent traces grouped by trace_id."""
        with self._lock:
            traces: Dict[str, List] = defaultdict(list)
            for s in self._spans[-limit * 5:]:  # oversample
                traces[s.trace_id].append(s.to_dict())
            result = []
            for trace_id, spans in list(traces.items())[-limit:]:
                result.append({
                    "trace_id": trace_id,
                    "spans": sorted(spans, key=lambda x: x["start_time"]),
                    "total_duration_ms": max(s["end_time"] for s in spans) - min(s["start_time"] for s in spans),
                })
            return result

    def get_span_count(self) -> int:
        with self._lock:
            return len(self._spans)

    def clear(self):
        with self._lock:
            self._spans.clear()


# ============== Metrics Collector ==============

class MetricsCollector:
    """
    Structured metrics collection with counters, gauges, and histograms.

    Thread-safe, lightweight, no external dependencies.
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0, tags: Dict = None):
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, tags: Dict = None):
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, tags: Dict = None):
        """Record a histogram value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)
            # Keep bounded
            if len(self._histograms[key]) > 5000:
                self._histograms[key] = self._histograms[key][-5000:]

    def get_all(self) -> Dict:
        """Get all metrics."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }
            for key, values in self._histograms.items():
                if values:
                    sorted_v = sorted(values)
                    n = len(sorted_v)
                    result["histograms"][key] = {
                        "count": n,
                        "avg": sum(sorted_v) / n,
                        "p50": sorted_v[n // 2],
                        "p95": sorted_v[int(n * 0.95)],
                        "p99": sorted_v[int(n * 0.99)],
                    }
            return result

    @staticmethod
    def _make_key(name: str, tags: Dict = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


# ============== Plugin System ==============

@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str = ""
    hooks: List[str] = field(default_factory=list)


class PluginManager:
    """
    Extensible plugin system for OpenClaw.

    Plugins can register hooks that are called at specific
    lifecycle points: on_startup, on_shutdown, on_task_start,
    on_task_complete, on_error, on_tool_call.
    """

    VALID_HOOKS = {
        "on_startup", "on_shutdown",
        "on_task_start", "on_task_complete",
        "on_error", "on_tool_call",
        "on_message", "on_agent_step",
    }

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, info: PluginInfo, hooks: Dict[str, Callable]):
        """Register a plugin with its hooks."""
        with self._lock:
            for hook_name in hooks:
                if hook_name not in self.VALID_HOOKS:
                    raise ValueError(f"Invalid hook: {hook_name}. Valid: {self.VALID_HOOKS}")

            self._plugins[info.name] = info
            for hook_name, callback in hooks.items():
                self._hooks[hook_name].append(callback)
                info.hooks.append(hook_name)

        logger.info(f"Plugin '{info.name}' v{info.version} registered with hooks: {list(hooks.keys())}")

    def unregister(self, name: str):
        """Unregister a plugin."""
        with self._lock:
            if name in self._plugins:
                del self._plugins[name]
                # Note: hooks are not removed to avoid complexity
                # They will be no-ops if the plugin object is gc'd

    def emit(self, hook_name: str, **kwargs) -> List[Any]:
        """Emit a hook event to all registered handlers."""
        with self._lock:
            handlers = list(self._hooks.get(hook_name, []))

        results = []
        for handler in handlers:
            try:
                result = handler(**kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Plugin hook '{hook_name}' error: {e}")
        return results

    def list_plugins(self) -> List[Dict]:
        """List all registered plugins."""
        with self._lock:
            return [
                {
                    "name": p.name,
                    "version": p.version,
                    "description": p.description,
                    "author": p.author,
                    "hooks": p.hooks,
                }
                for p in self._plugins.values()
            ]


# ============== Health Dashboard Data ==============

class HealthDashboard:
    """
    Self-diagnostic health dashboard data provider.

    Aggregates status from all subsystems into a single
    health report for monitoring and alerting.
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_check(self, name: str, check_fn: Callable):
        """Register a health check function. Should return Dict with 'healthy' bool."""
        with self._lock:
            self._checks[name] = check_fn

    def run_all(self) -> Dict:
        """Run all health checks and return aggregated status."""
        results = {}
        overall_healthy = True

        with self._lock:
            checks = dict(self._checks)

        for name, check_fn in checks.items():
            try:
                start = time.time()
                result = check_fn()
                duration = (time.time() - start) * 1000

                healthy = result.get("healthy", True) if isinstance(result, dict) else bool(result)
                results[name] = {
                    "healthy": healthy,
                    "duration_ms": round(duration, 2),
                    "details": result if isinstance(result, dict) else {},
                }
                if not healthy:
                    overall_healthy = False

            except Exception as e:
                overall_healthy = False
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "duration_ms": 0,
                }

        return {
            "overall": "healthy" if overall_healthy else "degraded",
            "timestamp": time.time(),
            "checks": results,
            "total_checks": len(results),
            "healthy_count": sum(1 for r in results.values() if r["healthy"]),
        }


# ---------- Global Instances ----------

_tracer: Optional[Tracer] = None
_metrics: Optional[MetricsCollector] = None
_plugins: Optional[PluginManager] = None
_dashboard: Optional[HealthDashboard] = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def get_plugin_manager() -> PluginManager:
    global _plugins
    if _plugins is None:
        _plugins = PluginManager()
    return _plugins


def get_health_dashboard() -> HealthDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = HealthDashboard()
    return _dashboard


__all__ = [
    "Span", "Tracer",
    "MetricsCollector",
    "PluginInfo", "PluginManager",
    "HealthDashboard",
    "get_tracer", "get_metrics",
    "get_plugin_manager", "get_health_dashboard",
]
