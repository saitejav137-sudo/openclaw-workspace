"""
Anomaly Detection for OpenClaw

Statistical and ML-based anomaly detection for automation triggers.
Detects unusual patterns in trigger behavior.
"""

import time
import math
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from statistics import mean, stdev, median

from .logger import get_logger

logger = get_logger("anomaly")


class AnomalyType(Enum):
    """Types of anomalies"""
    SPIKE = "spike"  # Sudden spike in triggers
    DROUGHT = "drought"  # Unusually long gap between triggers
    FREQUENCY = "frequency"  # Abnormal trigger frequency
    PATTERN = "pattern"  # Unusual pattern
    THRESHOLD = "threshold"  # Threshold breach


@dataclass
class Anomaly:
    """Detected anomaly"""
    type: AnomalyType
    score: float  # 0-1, higher = more anomalous
    description: str
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class TriggerEvent:
    """Trigger event for analysis"""
    timestamp: float
    triggered: bool
    value: float = 1.0  # Numeric value (1 for trigger, 0 for no trigger)
    metadata: Dict = field(default_factory=dict)


class StatisticalAnalyzer:
    """
    Statistical anomaly detection using various methods.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.events: deque = deque(maxlen=window_size)

    def add_event(self, event: TriggerEvent):
        """Add event for analysis"""
        self.events.append(event)

    def detect_zscore(self, threshold: float = 3.0) -> Optional[Anomaly]:
        """Detect anomalies using Z-score"""
        if len(self.events) < 10:
            return None

        # Get recent values
        values = [e.value for e in self.events]
        mean_val = mean(values)
        std_val = stdev(values)

        if std_val == 0:
            return None

        # Check latest value
        latest = self.events[-1]
        zscore = abs((latest.value - mean_val) / std_val)

        if zscore > threshold:
            return Anomaly(
                type=AnomalyType.SPIKE,
                score=min(zscore / (threshold * 2), 1.0),
                description=f"Z-score anomaly detected: {zscore:.2f}",
                timestamp=latest.timestamp,
                metadata={"zscore": zscore, "mean": mean_val, "std": std_val}
            )

        return None

    def detect_frequency(self, expected_rate: float, tolerance: float = 2.0) -> Optional[Anomaly]:
        """Detect frequency anomalies"""
        if len(self.events) < 20:
            return None

        # Calculate actual rate
        now = time.time()
        window = 60  # 1 minute window

        recent_events = [e for e in self.events if now - e.timestamp < window]
        actual_rate = len(recent_events)

        if actual_rate > expected_rate * tolerance:
            return Anomaly(
                type=AnomalyType.FREQUENCY,
                score=min(actual_rate / (expected_rate * tolerance), 1.0),
                description=f"High trigger frequency: {actual_rate}/min (expected: {expected_rate})",
                timestamp=now,
                metadata={"actual_rate": actual_rate, "expected_rate": expected_rate}
            )

        return None

    def detect_interarrival(self, expected_interval: float, tolerance: float = 3.0) -> Optional[Anomaly]:
        """Detect anomalies in time between triggers"""
        if len(self.events) < 5:
            return None

        # Calculate inter-arrival times
        triggered = [e for e in self.events if e.triggered]

        if len(triggered) < 2:
            return None

        intervals = []
        for i in range(1, len(triggered)):
            intervals.append(triggered[i].timestamp - triggered[i-1].timestamp)

        if not intervals:
            return None

        mean_interval = mean(intervals)
        std_interval = stdev(intervals) if len(intervals) > 1 else 0

        # Check for drought
        now = time.time()
        last_triggered = triggered[-1]
        time_since = now - last_triggered.timestamp

        if std_interval > 0 and time_since > mean_interval + tolerance * std_interval:
            return Anomaly(
                type=AnomalyType.DROUGHT,
                score=min((time_since - mean_interval) / (tolerance * std_interval), 1.0),
                description=f"Long gap since last trigger: {time_since:.1f}s (expected: {mean_interval:.1f}s)",
                timestamp=now,
                metadata={"time_since": time_since, "expected": mean_interval}
            )

        return None

    def detect_threshold(self, upper: float, lower: float = 0) -> Optional[Anomaly]:
        """Detect threshold breaches"""
        if not self.events:
            return None

        latest = self.events[-1]

        if latest.value > upper:
            return Anomaly(
                type=AnomalyType.THRESHOLD,
                score=min((latest.value - upper) / upper, 1.0) if upper > 0 else 1.0,
                description=f"Value {latest.value} exceeds upper threshold {upper}",
                timestamp=latest.timestamp,
                metadata={"value": latest.value, "threshold": upper}
            )

        if latest.value < lower:
            return Anomaly(
                type=AnomalyType.THRESHOLD,
                score=min((lower - latest.value) / lower, 1.0) if lower > 0 else 1.0,
                description=f"Value {latest.value} below lower threshold {lower}",
                timestamp=latest.timestamp,
                metadata={"value": latest.value, "threshold": lower}
            )

        return None


class MovingAverageDetector:
    """
    Moving average based anomaly detection.
    """

    def __init__(self, window_size: int = 10, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.values: deque = deque(maxlen=window_size)

    def add(self, value: float) -> Optional[Anomaly]:
        """Add value and check for anomaly"""
        self.values.append(value)

        if len(self.values) < self.window_size:
            return None

        # Calculate moving average and standard deviation
        values_list = list(self.values)
        ma = mean(values_list[:-1])  # Exclude latest
        std = stdev(values_list[:-1]) if len(values_list) > 1 else 0

        latest = values_list[-1]

        if std > 0:
            deviation = abs(latest - ma) / std

            if deviation > self.sensitivity:
                return Anomaly(
                    type=AnomalyType.SPIKE,
                    score=min(deviation / (self.sensitivity * 2), 1.0),
                    description=f"Moving average deviation: {deviation:.2f}s",
                    timestamp=time.time(),
                    metadata={"value": latest, "ma": ma, "std": std}
                )

        return None


class PatternDetector:
    """
    Pattern-based anomaly detection using sequence analysis.
    """

    def __init__(self, pattern_length: int = 5):
        self.pattern_length = pattern_length
        self.sequence: deque = deque(maxlen=pattern_length)
        self.patterns: Dict[Tuple, int] = {}

    def add(self, triggered: bool) -> Optional[Anomaly]:
        """Add to sequence and check for pattern anomaly"""
        value = 1 if triggered else 0
        self.sequence.append(value)

        if len(self.sequence) < self.pattern_length:
            return None

        pattern = tuple(self.sequence)

        # Count pattern occurrences
        self.patterns[pattern] = self.patterns.get(pattern, 0) + 1

        # Detect rare patterns
        if self.patterns[pattern] == 1:  # First occurrence
            return Anomaly(
                type=AnomalyType.PATTERN,
                score=0.8,
                description=f"New trigger pattern detected: {pattern}",
                timestamp=time.time(),
                metadata={"pattern": pattern}
            )

        return None


class AnomalyDetector:
    """
    Main anomaly detection system.
    Combines multiple detection methods.
    """

    def __init__(self):
        self.statistical = StatisticalAnalyzer(window_size=100)
        self.moving_avg = MovingAverageDetector(window_size=10)
        self.pattern = PatternDetector(pattern_length=5)

        self._anomalies: deque = deque(maxlen=100)
        self._lock = threading.Lock()

        # Detection settings
        self.enabled = True
        self.zscore_threshold = 3.0
        self.expected_rate = 10  # triggers per minute
        self.expected_interval = 6.0  # seconds between triggers

    def record_trigger(self, triggered: bool, value: float = 1.0, metadata: Optional[Dict] = None):
        """Record a trigger event"""
        if not self.enabled:
            return

        event = TriggerEvent(
            timestamp=time.time(),
            triggered=triggered,
            value=value,
            metadata=metadata or {}
        )

        # Add to analyzers
        self.statistical.add_event(event)
        self.moving_avg.add(value)
        self.pattern.add(triggered)

    def detect(self) -> List[Anomaly]:
        """Run all detection methods and return anomalies"""
        if not self.enabled:
            return []

        anomalies = []

        # Z-score detection
        zscore_anomaly = self.statistical.detect_zscore(self.zscore_threshold)
        if zscore_anomaly:
            anomalies.append(zscore_anomaly)

        # Frequency detection
        freq_anomaly = self.statistical.detect_frequency(self.expected_rate)
        if freq_anomaly:
            anomalies.append(freq_anomaly)

        # Inter-arrival detection
        iat_anomaly = self.statistical.detect_interarrival(self.expected_interval)
        if iat_anomaly:
            anomalies.append(iat_anomaly)

        # Moving average detection
        ma_anomaly = self.moving_avg.add(self.statistical.events[-1].value if self.statistical.events else 0)
        if ma_anomaly:
            anomalies.append(ma_anomaly)

        # Pattern detection
        if self.statistical.events:
            pattern_anomaly = self.pattern.add(self.statistical.events[-1].triggered)
            if pattern_anomaly:
                anomalies.append(pattern_anomaly)

        # Store anomalies
        with self._lock:
            self._anomalies.extend(anomalies)

        if anomalies:
            logger.warning(f"Anomalies detected: {len(anomalies)}")

        return anomalies

    def get_anomalies(self, since: Optional[float] = None) -> List[Anomaly]:
        """Get recorded anomalies"""
        with self._lock:
            if since:
                return [a for a in self._anomalies if a.timestamp >= since]
            return list(self._anomalies)

    def clear_anomalies(self):
        """Clear recorded anomalies"""
        with self._lock:
            self._anomalies.clear()

    def get_stats(self) -> Dict:
        """Get anomaly detection statistics"""
        with self._lock:
            total = len(self._anomalies)

            if total == 0:
                return {"total": 0, "by_type": {}}

            by_type = {}
            for a in self._anomalies:
                t = a.type.value
                by_type[t] = by_type.get(t, 0) + 1

            avg_score = mean([a.score for a in self._anomalies])

            return {
                "total": total,
                "by_type": by_type,
                "avg_score": avg_score,
                "recent_count": len([a for a in self._anomalies if time.time() - a.timestamp < 300])
            }


# Global anomaly detector
_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get global anomaly detector"""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector


__all__ = [
    "AnomalyType",
    "Anomaly",
    "TriggerEvent",
    "StatisticalAnalyzer",
    "MovingAverageDetector",
    "PatternDetector",
    "AnomalyDetector",
    "get_anomaly_detector",
]
