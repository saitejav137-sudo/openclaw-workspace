"""
Metrics and Monitoring for OpenClaw

Prometheus-compatible metrics export.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("openclaw.metrics")


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Base metric"""
    name: str
    description: str
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0


class Counter:
    """Counter metric"""

    def __init__(self, name: str, description: str, labels: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0):
        """Increment counter"""
        with self._lock:
            self._value += amount

    def get_value(self) -> float:
        """Get current value"""
        with self._lock:
            return self._value

    def reset(self):
        """Reset counter"""
        with self._lock:
            self._value = 0.0


class Gauge:
    """Gauge metric"""

    def __init__(self, name: str, description: str, labels: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float):
        """Set gauge value"""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        """Increment gauge"""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        """Decrement gauge"""
        with self._lock:
            self._value -= amount

    def get_value(self) -> float:
        """Get current value"""
        with self._lock:
            return self._value


class Histogram:
    """Histogram metric"""

    def __init__(
        self,
        name: str,
        description: str,
        buckets: List[float] = None,
        labels: Dict[str, str] = None
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.labels = labels or {}
        self._values = []
        self._lock = threading.Lock()

    def observe(self, value: float):
        """Observe a value"""
        with self._lock:
            self._values.append(value)

    def get_values(self) -> List[float]:
        """Get all observed values"""
        with self._lock:
            return self._values.copy()

    def get_buckets(self) -> Dict[str, int]:
        """Get bucket counts"""
        with self._lock:
            buckets = {}
            for bucket in self.buckets:
                count = sum(1 for v in self._values if v <= bucket)
                buckets[f"le_{bucket}"] = count
            return buckets


class MetricsRegistry:
    """Central metrics registry"""

    _instance = None

    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'MetricsRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Dict[str, str] = None
    ) -> Counter:
        """Get or create a counter"""
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, description, labels)
            return self.counters[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Dict[str, str] = None
    ) -> Gauge:
        """Get or create a gauge"""
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, description, labels)
            return self.gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: List[float] = None,
        labels: Dict[str, str] = None
    ) -> Histogram:
        """Get or create a histogram"""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, description, buckets, labels)
            return self.histograms[name]

    def get_metrics_text(self) -> str:
        """Get Prometheus-format metrics"""
        lines = []

        # Counters
        for name, counter in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"# HELP {name} {counter.description}")
            label_str = ""
            if counter.labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in counter.labels.items()) + "}"
            lines.append(f"{name}{label_str} {counter.get_value()}")

        # Gauges
        for name, gauge in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"# HELP {name} {gauge.description}")
            label_str = ""
            if gauge.labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in gauge.labels.items()) + "}"
            lines.append(f"{name}{label_str} {gauge.get_value()}")

        # Histograms
        for name, histogram in self.histograms.items():
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"# HELP {name} {histogram.description}")

            buckets = histogram.get_buckets()
            for bucket_key, count in buckets.items():
                label_str = ""
                if histogram.labels:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in histogram.labels.items()) + "," + bucket_key + "}"
                else:
                    label_str = "{" + bucket_key + "}"
                lines.append(f"{name}{label_str} {count}")

            # Sum and count
            values = histogram.get_values()
            if values:
                if histogram.labels:
                    lines.append(f"{name}_sum{histogram.labels} {sum(values)}")
                    lines.append(f"{name}_count{histogram.labels} {len(values)}")
                else:
                    lines.append(f"{name}_sum {sum(values)}")
                    lines.append(f"{name}_count {len(values)}")

        return "\n".join(lines)

    def reset(self):
        """Reset all metrics"""
        with self._lock:
            for counter in self.counters.values():
                counter.reset()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


# Default metrics
def get_default_metrics() -> Dict[str, Any]:
    """Get default OpenClaw metrics"""
    registry = MetricsRegistry.get_instance()

    return {
        "triggers_total": registry.counter(
            "openclaw_triggers_total",
            "Total number of triggers"
        ),
        "triggers_success": registry.counter(
            "openclaw_triggers_success_total",
            "Total number of successful triggers"
        ),
        "triggers_failed": registry.counter(
            "openclaw_triggers_failed_total",
            "Total number of failed triggers"
        ),
        "actions_executed": registry.counter(
            "openclaw_actions_executed_total",
            "Total number of actions executed"
        ),
        "detection_time": registry.histogram(
            "openclaw_detection_time_seconds",
            "Time taken for detection"
        ),
        "active_jobs": registry.gauge(
            "openclaw_active_jobs",
            "Number of active scheduled jobs"
        ),
    }


# Export
__all__ = [
    "MetricType",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "get_default_metrics",
]
