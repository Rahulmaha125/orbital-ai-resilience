"""Type definitions and enumerations for the Orbital AI Resilience system."""

from enum import Enum


class NodeStatus(str, Enum):
    """Operational status of a Virtual Compute Node."""
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    ISOLATED = "ISOLATED"


class WorkloadStatus(str, Enum):
    """Lifecycle status of a Workload object."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MIGRATED = "MIGRATED"


class HealthState(str, Enum):
    """Health classification derived from Trust/Health score ranges."""
    HEALTHY = "HEALTHY"    # Score 90.0 - 100.0
    WARNING = "WARNING"    # Score 75.0 - 89.9
    DEGRADED = "DEGRADED"  # Score 60.0 - 74.9
    CRITICAL = "CRITICAL"  # Score 0.0 - 59.9
