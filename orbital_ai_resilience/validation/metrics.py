"""Quantitative performance metrics dataclasses for Reliability, Efficiency, Safety, and Scalability."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ValidationMetrics:
    """Quantitative performance metrics collected during validation simulations.

    Categories:
        - Reliability: recovery_success_rate, verification_success_rate, workload_recovery_rate, workload_loss_rate
        - Efficiency: average_migration_time, average_recovery_time, average_retries, average_recovery_cost, communication_cost, power_cost, orbital_route_cost
        - Safety: verification_failures, quarantined_targets, isolated_sources, unsafe_target_selections, duplicate_ownership_events, unnecessary_migrations
        - Scalability: total_ticks, simulation_duration_sec, avg_tick_duration_sec, avg_routing_time_sec, memory_usage_mb
    """

    scenario_name: str
    policy_name: str
    node_count: int
    total_ticks: int
    seed: int

    # Reliability
    recovery_success_rate: float = 0.0
    verification_success_rate: float = 0.0
    workload_recovery_rate: float = 0.0
    workload_loss_rate: float = 0.0
    total_workloads_submitted: int = 0
    successful_workloads: int = 0
    lost_workloads: int = 0

    # Efficiency
    average_migration_time: float = 0.0
    average_recovery_time: float = 0.0
    average_retries: float = 0.0
    average_recovery_cost: float = 0.0
    communication_cost: float = 0.0
    power_cost: float = 0.0
    orbital_route_cost: float = 0.0

    # Safety
    verification_failures: int = 0
    quarantined_targets: int = 0
    isolated_sources: int = 0
    unsafe_target_selections: int = 0
    duplicate_ownership_events: int = 0
    unnecessary_migrations: int = 0

    # Scalability
    simulation_duration_sec: float = 0.0
    avg_tick_duration_sec: float = 0.0
    avg_routing_time_sec: float = 0.0
    memory_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary with rounded floats."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d
