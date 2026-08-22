"""Migration policy defining recovery triggers and target eligibility constraints."""

from dataclasses import dataclass, field
from typing import List
from orbital_ai_resilience.core.types import HealthState
from orbital_ai_resilience.detection.types import BehavioralState, DetectionResult


@dataclass
class MigrationPolicy:
    """Configurable rules governing when workloads are migrated and target node eligibility.

    Attributes:
        trigger_on_silent_degradation: If True, triggers migration when is_silent_degradation is True.
        trigger_on_behavioral_states: BehavioralState classifications that trigger migration.
        min_target_physical_health: Minimum physical health score required for destination node (default 90.0).
        allowed_target_physical_states: Physical HealthStates acceptable for target destination node.
        allowed_target_behavioral_states: Behavioral states acceptable for target destination node.
        weight_health: Target scoring weight for physical health score (default 0.30).
        weight_behavior: Target scoring weight for behavioral integrity score (default 0.35).
        weight_capacity: Target scoring weight for available compute capacity (default 0.20).
        weight_power: Target scoring weight for available power level (default 0.15).
        weight_latency: Target penalty weight for latency (default 0.10).
        max_migration_attempts: Maximum migration attempts allowed per workload (default 3).
        auto_isolate_source: If True, automatically isolates source node upon successful migration.
    """

    trigger_on_silent_degradation: bool = True
    trigger_on_behavioral_states: List[BehavioralState] = field(
        default_factory=lambda: [BehavioralState.DEGRADED, BehavioralState.CRITICAL]
    )
    min_target_physical_health: float = 90.0
    allowed_target_physical_states: List[HealthState] = field(
        default_factory=lambda: [HealthState.HEALTHY, HealthState.WARNING]
    )
    allowed_target_behavioral_states: List[BehavioralState] = field(
        default_factory=lambda: [BehavioralState.NORMAL, BehavioralState.WARNING]
    )

    weight_health: float = 0.30
    weight_behavior: float = 0.35
    weight_capacity: float = 0.20
    weight_power: float = 0.15
    weight_latency: float = 0.10

    max_migration_attempts: int = 3
    auto_isolate_source: bool = True

    def should_migrate(self, detection_result: DetectionResult) -> bool:
        """Determine whether a detection result warrants triggering a workload migration."""
        if self.trigger_on_silent_degradation and detection_result.is_silent_degradation:
            return True
        if detection_result.behavioral_state in self.trigger_on_behavioral_states:
            return True
        return False
