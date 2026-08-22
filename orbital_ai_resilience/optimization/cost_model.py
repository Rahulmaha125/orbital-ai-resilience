"""Interpretable recovery cost model calculating resource, communication, and risk costs."""

from dataclasses import dataclass
from typing import Any, Dict
from orbital_ai_resilience.optimization.features import TargetNodeFeatures


@dataclass
class CostBreakdown:
    """Detailed breakdown of individual cost model components."""

    migration_cost: float
    communication_cost: float
    compute_cost: float
    power_cost: float
    risk_cost: float
    verification_cost: float
    total_cost: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize cost breakdown to dictionary."""
        return {
            "migration_cost": round(self.migration_cost, 2),
            "communication_cost": round(self.communication_cost, 2),
            "compute_cost": round(self.compute_cost, 2),
            "power_cost": round(self.power_cost, 2),
            "risk_cost": round(self.risk_cost, 2),
            "verification_cost": round(self.verification_cost, 2),
            "total_cost": round(self.total_cost, 2),
        }


class RecoveryCostModel:
    """Calculates transparent, interpretable recovery costs for target candidate nodes.

    RecoveryCost = migration_cost + communication_cost + compute_cost + power_cost + risk_cost + verification_cost
    """

    def __init__(
        self,
        base_migration_cost: float = 5.0,
        base_verification_cost: float = 2.0,
        comm_weight: float = 0.5,
        risk_weight: float = 1.0,
    ) -> None:
        self.base_migration_cost: float = base_migration_cost
        self.base_verification_cost: float = base_verification_cost
        self.comm_weight: float = comm_weight
        self.risk_weight: float = risk_weight

    def calculate_cost(self, features: TargetNodeFeatures) -> CostBreakdown:
        """Calculate total recovery cost and individual cost component breakdown.

        Args:
            features: TargetNodeFeatures vector for candidate node.

        Returns:
            CostBreakdown instance.
        """
        # 1. Base Migration Overhead
        c_mig = self.base_migration_cost

        # 2. Communication Cost (latency + memory payload scaling)
        c_comm = self.comm_weight * features.latency * (1.0 + (features.workload_memory_req / 8192.0))

        # 3. Compute Resource Cost (higher cost when remaining compute is tight)
        avail_ratio = max(0.01, features.available_compute_ratio)
        c_comp = 10.0 * (1.0 - avail_ratio)

        # 4. Power Cost (penalty for low power reserve)
        c_power = 15.0 * (1.0 - (features.power_level / 100.0))

        # 5. Risk Cost (health deficit, error rate, and historical unreliability penalty)
        health_deficit = 100.0 - features.physical_health_score
        unreliability_penalty = 100.0 - features.node_reliability_score
        c_risk = self.risk_weight * (
            0.3 * health_deficit
            + 50.0 * features.error_rate
            + 0.2 * unreliability_penalty
            + 10.0 * float(features.previous_quarantine_count)
        )

        # 6. Verification Cost
        c_verif = self.base_verification_cost

        total = c_mig + c_comm + c_comp + c_power + c_risk + c_verif

        return CostBreakdown(
            migration_cost=c_mig,
            communication_cost=c_comm,
            compute_cost=c_comp,
            power_cost=c_power,
            risk_cost=c_risk,
            verification_cost=c_verif,
            total_cost=total,
        )
