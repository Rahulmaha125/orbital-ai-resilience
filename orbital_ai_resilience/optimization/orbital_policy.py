"""Phase 9 Orbital & Constellation-Aware Recovery Policy and Cost Model operating above Phase 8."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from orbital_ai_resilience.constellation.bandwidth import BandwidthModel
from orbital_ai_resilience.constellation.routing import ConstellationRoute, ConstellationRouter
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.optimization.cost_model import CostBreakdown, RecoveryCostModel
from orbital_ai_resilience.optimization.features import OptimizationFeatureBuilder
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.policy import AdaptiveRecoveryPolicy, BaseRecoveryPolicy
from orbital_ai_resilience.orbital.eclipse import EclipseModel, EclipseStatus
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.prediction import FutureOrbitalPrediction, OrbitalPredictionModel
from orbital_ai_resilience.orbital.propagation import OrbitalPropagator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector


@dataclass
class OrbitalCostBreakdown:
    """Detailed breakdown of orbital recovery cost components."""

    phase8_base_cost: float
    communication_cost: float
    bandwidth_cost: float
    distance_cost: float
    eclipse_risk_cost: float
    future_power_risk_cost: float
    route_risk_cost: float
    total_orbital_cost: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize orbital cost breakdown to dictionary."""
        return {
            "phase8_base_cost": round(self.phase8_base_cost, 2),
            "communication_cost": round(self.communication_cost, 2),
            "bandwidth_cost": round(self.bandwidth_cost, 2),
            "distance_cost": round(self.distance_cost, 2),
            "eclipse_risk_cost": round(self.eclipse_risk_cost, 2),
            "future_power_risk_cost": round(self.future_power_risk_cost, 2),
            "route_risk_cost": round(self.route_risk_cost, 2),
            "total_orbital_cost": round(self.total_orbital_cost, 2),
        }


class OrbitalRecoveryCostModel:
    """Calculates orbital, eclipse, bandwidth, and multi-hop routing costs."""

    def __init__(self, base_cost_model: Optional[RecoveryCostModel] = None) -> None:
        self.base_cost_model: RecoveryCostModel = base_cost_model or RecoveryCostModel()

    def calculate_orbital_cost(
        self,
        base_cost: float,
        route: ConstellationRoute,
        eclipse_status: EclipseStatus,
        prediction: FutureOrbitalPrediction,
    ) -> OrbitalCostBreakdown:
        """Calculate total normalized orbital recovery cost."""
        c_phase8 = base_cost
        c_comm = route.total_latency_ms * 0.5
        c_bw = 100.0 / max(1.0, route.bottleneck_bandwidth_mbps)
        c_dist = route.total_distance_km / 500.0

        c_eclipse = 25.0 if eclipse_status.is_eclipse else (15.0 * prediction.future_eclipse_risk)
        c_power_risk = 20.0 * (1.0 - (prediction.future_power_reserve / 100.0))
        c_route_risk = 10.0 * max(0, route.hop_count - 1)

        total = c_phase8 + c_comm + c_bw + c_dist + c_eclipse + c_power_risk + c_route_risk

        return OrbitalCostBreakdown(
            phase8_base_cost=c_phase8,
            communication_cost=c_comm,
            bandwidth_cost=c_bw,
            distance_cost=c_dist,
            eclipse_risk_cost=c_eclipse,
            future_power_risk_cost=c_power_risk,
            route_risk_cost=c_route_risk,
            total_orbital_cost=round(total, 2),
        )


class OrbitalAwareRecoveryPolicy(BaseRecoveryPolicy):
    """Phase 9 Orbital & Constellation-Aware Recovery Policy.

    Selects BOTH destination target satellite AND optimal multi-hop communication route
    considering orbital geometry, line-of-sight visibility, bandwidth, power, and future eclipse risk.
    """

    def __init__(
        self,
        policy: Optional[MigrationPolicy] = None,
        propagator: Optional[OrbitalPropagator] = None,
        eclipse_model: Optional[EclipseModel] = None,
        prediction_model: Optional[OrbitalPredictionModel] = None,
        router: Optional[ConstellationRouter] = None,
        bandwidth_model: Optional[BandwidthModel] = None,
    ) -> None:
        super().__init__(name="OrbitalAwareRecoveryPolicy")
        self.policy: MigrationPolicy = policy or MigrationPolicy()
        self.propagator: OrbitalPropagator = propagator or OrbitalPropagator()
        self.eclipse_model: EclipseModel = eclipse_model or EclipseModel()
        self.prediction_model: OrbitalPredictionModel = prediction_model or OrbitalPredictionModel(
            propagator=self.propagator, eclipse_model=self.eclipse_model
        )
        self.router: ConstellationRouter = router or ConstellationRouter(eclipse_model=self.eclipse_model)
        self.bandwidth_model: BandwidthModel = bandwidth_model or BandwidthModel()
        self.adaptive_policy: AdaptiveRecoveryPolicy = AdaptiveRecoveryPolicy(policy=self.policy)
        self.selector: TargetSelector = TargetSelector(policy=self.policy)
        self.orbital_cost_model: OrbitalRecoveryCostModel = OrbitalRecoveryCostModel()

    def select_target_and_route(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        history: RecoveryHistory,
        constellation_states: Optional[Dict[str, OrbitalState]] = None,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], Optional[ConstellationRoute], float, Dict[str, Any]]:
        """Select optimal target satellite AND multi-hop route for workload recovery."""
        tick = cluster.nodes[source_node_id].telemetry_history.get_recent(1)[0].timestamp if source_node_id in cluster.nodes else 0.0
        states = constellation_states or self.propagator.generate_constellation_states(num_satellites=len(cluster.nodes), tick=tick)

        # Excluded relay nodes: quarantined or isolated nodes (source node MUST be allowed as starting point!)
        excluded: Set[str] = set()
        if quarantine_manager:
            excluded.update(quarantine_manager.get_quarantined_node_ids())
            excluded.update(quarantine_manager.get_isolated_node_ids())

        best_target_id: Optional[str] = None
        best_route: Optional[ConstellationRoute] = None
        best_score: float = -1e9
        breakdowns: Dict[str, Any] = {}

        for node_id, node in cluster.nodes.items():
            if self.selector.is_eligible_target(node, source_node_id, workload, quarantine_manager=quarantine_manager):
                # 1. Find multi-hop communication route
                route = self.router.find_route(
                    source_id=source_node_id,
                    target_id=node_id,
                    constellation_states=states,
                    excluded_nodes=excluded,
                )

                if not route.is_route_valid:
                    continue

                # 2. Estimate Bandwidth Transfer
                mem_mb = workload.required_memory / 1024.0
                bw_est = self.bandwidth_model.estimate_transfer(
                    workload_memory_mb=mem_mb,
                    bottleneck_bandwidth_mbps=route.bottleneck_bandwidth_mbps,
                )

                if not bw_est.is_bandwidth_sufficient:
                    continue

                # 3. Evaluate Eclipse and Future Prediction
                orb_state = states.get(node_id, self.propagator.compute_state(node_id, initial_phase_deg=0.0, tick=tick))
                ecl_status = self.eclipse_model.evaluate_illumination(orb_state)
                prediction = self.prediction_model.predict_future_state(
                    satellite_id=node_id,
                    initial_phase_deg=orb_state.orbital_phase_deg,
                    current_tick=tick,
                    current_battery_level=node.power_level,
                    lookahead_ticks=6,
                )

                if prediction.predicted_min_battery_level < 15.0 and prediction.future_eclipse_risk > 0.40:
                    continue

                # 4. Calculate Orbital Recovery Cost
                phase8_score_tuple = self.adaptive_policy.select_target(cluster, source_node_id, workload, history, quarantine_manager)
                base_cost = phase8_score_tuple[2].get("candidate_breakdowns", {}).get(node_id, {}).get("total_cost", 10.0)

                orb_cost_breakdown = self.orbital_cost_model.calculate_orbital_cost(
                    base_cost=base_cost,
                    route=route,
                    eclipse_status=ecl_status,
                    prediction=prediction,
                )

                # 5. Composite Orbital Score
                adaptive_base = phase8_score_tuple[2].get("candidate_breakdowns", {}).get(node_id, {}).get("adaptive_score", 100.0)
                sun_bonus = 20.0 * ecl_status.sunlight_fraction
                ecl_risk_penalty = 30.0 * prediction.future_eclipse_risk
                hop_penalty = 8.0 * max(0, route.hop_count - 1)

                orbital_score = adaptive_base + sun_bonus - (0.40 * orb_cost_breakdown.total_orbital_cost) - ecl_risk_penalty - hop_penalty
                orbital_score = round(max(0.0, orbital_score), 2)

                breakdowns[node_id] = {
                    "orbital_score": orbital_score,
                    "adaptive_base_score": adaptive_base,
                    "route": route.to_dict(),
                    "transfer_estimate": bw_est.to_dict(),
                    "eclipse_status": ecl_status.to_dict(),
                    "prediction": prediction.to_dict(),
                    "orbital_cost_breakdown": orb_cost_breakdown.to_dict(),
                }

                if orbital_score > best_score:
                    best_score = orbital_score
                    best_target_id = node_id
                    best_route = route

        selected_breakdown = breakdowns.get(best_target_id, {}) if best_target_id else {}
        explanation = {
            "policy_type": "Phase 9 Orbital & Constellation-Aware Policy",
            "selected_node": best_target_id,
            "selected_route": best_route.path if best_route else [],
            "route_string": " -> ".join(best_route.path) if best_route else "No Route",
            "orbital_score": best_score,
            "candidate_breakdowns": breakdowns,
            "why_selected": (
                f"Selected target {best_target_id} via route [{' -> '.join(best_route.path)}] "
                f"(Orbital Score: {best_score}) due to clear line-of-sight, bottleneck bandwidth "
                f"({selected_breakdown.get('transfer_estimate', {}).get('bottleneck_bandwidth_mbps', 0)} Mbps), "
                f"sunlight fraction ({selected_breakdown.get('eclipse_status', {}).get('sunlight_fraction', 1.0)}), "
                f"and low future eclipse risk ({selected_breakdown.get('prediction', {}).get('future_eclipse_risk', 0.0)})."
            )
            if best_target_id and best_route
            else "No valid orbital target or route found",
        }

        return best_target_id, best_route, max(0.0, best_score), explanation

    def select_target(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        history: RecoveryHistory,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Adapter implementing BaseRecoveryPolicy interface."""
        target_id, route, score, explanation = self.select_target_and_route(
            cluster=cluster,
            source_node_id=source_node_id,
            workload=workload,
            history=history,
            quarantine_manager=quarantine_manager,
        )
        return target_id, score, explanation
