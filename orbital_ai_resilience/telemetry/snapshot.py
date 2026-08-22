"""Telemetry snapshot and time-series history data structures."""

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Immutable timestamped snapshot of node physical and compute telemetry.

    Attributes:
        timestamp: Epoch timestamp of observation.
        power_level: Operating power level (Watts or %).
        temperature: Operating temperature (Celsius).
        latency: Processing/comm latency (milliseconds).
        error_rate: Observed error rate ratio [0.0, 1.0].
        used_compute: Total active compute allocated (TFLOPS).
        used_memory: Total active memory allocated (MB).
    """

    timestamp: float
    power_level: float
    temperature: float
    latency: float
    error_rate: float
    used_compute: float = 0.0
    used_memory: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize telemetry snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "power_level": self.power_level,
            "temperature": self.temperature,
            "latency": self.latency,
            "error_rate": self.error_rate,
            "used_compute": self.used_compute,
            "used_memory": self.used_memory,
        }


class TelemetryHistory:
    """Bounded rolling-window time-series history for telemetry snapshots.

    Attributes:
        max_length: Maximum number of historical snapshots retained.
        history: Bounded deque storing TelemetrySnapshot instances.
    """

    def __init__(self, max_length: int = 100) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length: int = max_length
        self.history: deque[TelemetrySnapshot] = deque(maxlen=max_length)

    def add_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """Append a new snapshot to the history window."""
        self.history.append(snapshot)

    def get_latest(self) -> Optional[TelemetrySnapshot]:
        """Return the most recent telemetry snapshot, or None if empty."""
        if not self.history:
            return None
        return self.history[-1]

    def get_recent(self, n: int) -> List[TelemetrySnapshot]:
        """Return the last n telemetry snapshots ordered from oldest to newest."""
        if n <= 0:
            return []
        slice_start = max(0, len(self.history) - n)
        return list(self.history)[slice_start:]

    def get_series(self, metric_name: str) -> List[float]:
        """Extract a list of values for a specific numeric metric attribute across history.

        Args:
            metric_name: Name of metric field ('power_level', 'temperature', 'latency', 'error_rate').

        Returns:
            List of float metric values ordered chronologically.
        """
        if metric_name not in TelemetrySnapshot.__dataclass_fields__:
            raise AttributeError(f"Invalid telemetry metric name: {metric_name!r}")
        return [getattr(s, metric_name) for s in self.history]

    def clear(self) -> None:
        """Clear all stored telemetry history."""
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"TelemetryHistory(count={len(self.history)}, max_length={self.max_length})"
