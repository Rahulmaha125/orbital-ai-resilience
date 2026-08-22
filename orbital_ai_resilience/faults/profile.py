"""Fault profile definition for deterministic fault injection scenarios."""

from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, Optional
from orbital_ai_resilience.faults.types import FaultType


@dataclass
class FaultProfile:
    """Defines a deterministic fault injection profile targeting a VirtualNode.

    Attributes:
        fault_type: Category of fault to inject.
        target_node_id: ID of the virtual node to be affected.
        start_tick: Simulation tick at which fault becomes active.
        duration: Number of ticks fault remains active (None for indefinite).
        intensity: Scaling factor [0.0, 1.0] representing fault severity.
        seed: Random seed for deterministic replayability.
        fault_id: Unique string identifier for the profile.
        params: Additional configuration parameters specific to the fault type.
    """

    fault_type: FaultType
    target_node_id: str
    start_tick: int = 0
    duration: Optional[int] = None
    intensity: float = 0.05
    seed: int = 42
    fault_id: str = field(default_factory=lambda: f"fault-{uuid.uuid4().hex[:8]}")
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(f"intensity must be in range [0.0, 1.0], got {self.intensity}")
        if self.start_tick < 0:
            raise ValueError("start_tick cannot be negative")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive or None")

    def is_active_at(self, current_tick: int) -> bool:
        """Check if the fault profile is active at the specified simulation tick."""
        if current_tick < self.start_tick:
            return False
        if self.duration is not None and current_tick >= self.start_tick + self.duration:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fault profile to dictionary representation."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "target_node_id": self.target_node_id,
            "start_tick": self.start_tick,
            "duration": self.duration,
            "intensity": self.intensity,
            "seed": self.seed,
            "params": self.params,
        }
