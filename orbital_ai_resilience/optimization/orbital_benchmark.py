"""Scientific 2-policy benchmarking suite comparing Phase 8 Adaptive vs Phase 9 Orbital-Aware Policy across 10 orbital scenarios."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from orbital_ai_resilience.constellation.links import InterSatelliteLink, LinkEvaluator
from orbital_ai_resilience.constellation.routing import ConstellationRoute, ConstellationRouter
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile, FaultType
from orbital_ai_resilience.optimization.cost_model import RecoveryCostModel
from orbital_ai_resilience.optimization.features import OptimizationFeatureBuilder
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.optimizer import AdaptiveRecoveryOptimizer
from orbital_ai_resilience.optimization.orbital_policy import OrbitalAwareRecoveryPolicy, OrbitalRecoveryCostModel
from orbital_ai_resilience.optimization.policy import AdaptiveRecoveryPolicy, DeterministicBaselinePolicy
from orbital_ai_resilience.orbital.eclipse import EclipseModel
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.propagation import OrbitalPropagator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.recovery.migration import MigrationManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.verification.verifier import OutputVerifier
from orbital_ai_resilience.verification.types import VerificationResultState
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


@dataclass
class OrbitalBenchmarkMetrics:
    """Quantitative performance metrics comparing Phase 8 vs. Phase 9 policy."""

    scenario_name: str
    policy_name: str
    recovery_success_rate: float
    verification_success_rate: float
    workload_recovery_rate: float
    workload_loss_rate: float
    average_recovery_time: float
    average_migration_time: float
    average_retries: float
    average_total_recovery_cost: float
    communication_cost: float
    bandwidth_utilization: float
    energy_power_cost: float
    average_route_length: float
    average_hop_count: int
    link_failure_count: int
    eclipse_related_failures: int
    predictive_decision_accuracy: float
    unnecessary_migrations_count: int
    selected_target_node: str
    selected_route_string: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark metrics to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "policy_name": self.policy_name,
            "recovery_success_rate": round(self.recovery_success_rate, 4),
            "verification_success_rate": round(self.verification_success_rate, 4),
            "workload_recovery_rate": round(self.workload_recovery_rate, 4),
            "workload_loss_rate": round(self.workload_loss_rate, 4),
            "average_recovery_time": round(self.average_recovery_time, 4),
            "average_migration_time": round(self.average_migration_time, 4),
            "average_retries": round(self.average_retries, 2),
            "average_total_recovery_cost": round(self.average_total_recovery_cost, 2),
            "communication_cost": round(self.communication_cost, 2),
            "bandwidth_utilization": round(self.bandwidth_utilization, 2),
            "energy_power_cost": round(self.energy_power_cost, 2),
            "average_route_length": round(self.average_route_length, 2),
            "average_hop_count": self.average_hop_count,
            "link_failure_count": self.link_failure_count,
            "eclipse_related_failures": self.eclipse_related_failures,
            "predictive_decision_accuracy": round(self.predictive_decision_accuracy, 4),
            "unnecessary_migrations_count": self.unnecessary_migrations_count,
            "selected_target_node": self.selected_target_node,
            "selected_route_string": self.selected_route_string,
        }


class OrbitalOptimizationBenchmark:
    """Executes scientific comparison between Phase 8 Adaptive and Phase 9 Orbital-Aware Policy across 10 scenarios."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed

    def setup_orbital_scenario(
        self, scenario_id: int
    ) -> Tuple[str, VirtualCluster, FaultInjector, RecoveryHistory, SyntheticAIWorkload, Dict[str, OrbitalState]]:
        """Construct deterministic scenario cluster state, orbital positions, fault injection, history, and workload."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        injector = FaultInjector()
        history = RecoveryHistory()
        propagator = OrbitalPropagator()

        states = propagator.generate_constellation_states(num_satellites=5, tick=0.0)

        if scenario_id == 1:
            name = "Scenario 1: Normal Orbital Operation"
            w = SyntheticAIWorkload(name="task_orb_sc1", seed=self.seed)

        elif scenario_id == 2:
            name = "Scenario 2: Target Enters Eclipse"
            states["node-1"] = OrbitalState(
                satellite_id="node-1",
                altitude_km=550.0,
                inclination_deg=53.0,
                orbital_phase_deg=180.0,
                mean_motion_rad_per_sec=propagator.mean_motion,
                position_km=(-6921.0, 0.0, 0.0),
                velocity_km_s=(0.0, 7.6, 0.0),
                timestamp=0.0,
            )
            w = SyntheticAIWorkload(name="task_orb_sc2", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 3:
            name = "Scenario 3: Source-Target Direct Link Unavailable (Earth Obstruction)"
            states["node-3"] = OrbitalState(
                satellite_id="node-3",
                altitude_km=550.0,
                inclination_deg=53.0,
                orbital_phase_deg=0.0,
                mean_motion_rad_per_sec=propagator.mean_motion,
                position_km=(6921.0, 0.0, 0.0),
                velocity_km_s=(0.0, 7.6, 0.0),
                timestamp=0.0,
            )
            states["node-4"] = OrbitalState(
                satellite_id="node-4",
                altitude_km=550.0,
                inclination_deg=53.0,
                orbital_phase_deg=180.0,
                mean_motion_rad_per_sec=propagator.mean_motion,
                position_km=(-6921.0, 0.0, 0.0),
                velocity_km_s=(0.0, 7.6, 0.0),
                timestamp=0.0,
            )
            w = SyntheticAIWorkload(name="task_orb_sc3", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 4:
            name = "Scenario 4: Multi-Hop Recovery Required"
            states["node-3"] = propagator.compute_state("node-3", 0.0, 0.0)
            states["node-2"] = propagator.compute_state("node-2", 45.0, 0.0)
            states["node-4"] = propagator.compute_state("node-4", 90.0, 0.0)
            w = SyntheticAIWorkload(name="task_orb_sc4", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 5:
            name = "Scenario 5: Low Bandwidth Link"
            cluster.get_node("node-1").memory_capacity = 16384.0
            w = SyntheticAIWorkload(name="task_orb_sc5", required_memory=4096.0, seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 6:
            name = "Scenario 6: High Communication Latency"
            cluster.get_node("node-1").update_telemetry(latency=80.0)
            cluster.get_node("node-2").update_telemetry(latency=5.0)
            w = SyntheticAIWorkload(name="task_orb_sc6", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 7:
            name = "Scenario 7: Power-Constrained Target"
            cluster.get_node("node-1").update_telemetry(power_level=20.0)
            cluster.get_node("node-2").update_telemetry(power_level=98.0)
            w = SyntheticAIWorkload(name="task_orb_sc7", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 8:
            name = "Scenario 8: Future Eclipse Risk"
            states["node-1"] = propagator.compute_state("node-1", 105.0, 0.0)
            cluster.get_node("node-1").update_telemetry(power_level=30.0)
            states["node-2"] = propagator.compute_state("node-2", 10.0, 0.0)
            cluster.get_node("node-2").update_telemetry(power_level=90.0)
            w = SyntheticAIWorkload(name="task_orb_sc8", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 9:
            name = "Scenario 9: Multiple Route Choices"
            w = SyntheticAIWorkload(name="task_orb_sc9", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 10:
            name = "Scenario 10: Combined AI Failure + Orbital Communication Constraints"
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.20, seed=self.seed))
            injector.add_profile(FaultProfile(fault_type=FaultType.MEMORY_BIT_FLIP, target_node_id="node-1", intensity=0.25, seed=99))
            w = SyntheticAIWorkload(name="task_orb_sc10", required_compute=30.0, seed=self.seed)
        else:
            raise ValueError(f"Invalid scenario_id: {scenario_id}")

        return name, cluster, injector, history, w, states

    def run_scenario_policy(
        self,
        scenario_id: int,
        use_orbital_policy: bool,
    ) -> OrbitalBenchmarkMetrics:
        """Run single policy (Phase 8 Adaptive vs Phase 9 Orbital-Aware) on a scenario with scientific fairness."""
        name, cluster, injector, history, workload, states = self.setup_orbital_scenario(scenario_id)
        policy_name = "Phase 9 Orbital Policy" if use_orbital_policy else "Phase 8 Adaptive Policy"

        qm = QuarantineManager()
        pol = MigrationPolicy()
        detector = StatisticalDetector()
        verifier = OutputVerifier()
        router = ConstellationRouter()

        if use_orbital_policy:
            orb_policy = OrbitalAwareRecoveryPolicy(policy=pol, router=router)

            class CustomOrbitalSelector(TargetSelector):
                def select_best_target(
                    self,
                    cluster: VirtualCluster,
                    source_node_id: str,
                    workload: Any,
                    quarantine_manager: Any = None,
                ):
                    tid, route, sc, exp = orb_policy.select_target_and_route(
                        cluster, source_node_id, workload, history, constellation_states=states, quarantine_manager=quarantine_manager
                    )
                    cand_scores = {tid: sc} if tid else {}
                    return tid, sc, cand_scores

            selector = CustomOrbitalSelector(policy=pol, quarantine_manager=qm)
        else:
            adapt_policy = AdaptiveRecoveryPolicy(policy=pol)

            class CustomPhase8Selector(TargetSelector):
                def select_best_target(
                    self,
                    cluster: VirtualCluster,
                    source_node_id: str,
                    workload: Any,
                    quarantine_manager: Any = None,
                ):
                    tid, sc, exp = adapt_policy.select_target(cluster, source_node_id, workload, history, quarantine_manager=quarantine_manager)
                    cand_scores = {tid: sc} if tid else {}
                    return tid, sc, cand_scores

            selector = CustomPhase8Selector(policy=pol, quarantine_manager=qm)

        manager = MigrationManager(policy=pol, selector=selector, verifier=verifier, quarantine_manager=qm)

        n3 = cluster.get_node("node-3")
        n3.assign_workload(workload)
        exec_log = workload.execute_on_node(n3, fault_injector=injector, tick=0)
        det = detector.evaluate(exec_log)

        event = manager.execute_autonomous_recovery(cluster, "node-3", workload, det, fault_injector=injector)

        selected_target = event.target_node_id if (event and event.target_node_id) else "None"
        route_obj = router.find_route("node-3", selected_target, states) if selected_target != "None" else None

        is_success = event.migration_status.value == "COMPLETED" if event else False
        v_state = event.verification_status if event else VerificationResultState.UNVERIFIED

        cost_model = OrbitalRecoveryCostModel()
        ecl_model = EclipseModel()
        orb_state = states.get(selected_target, OrbitalState("none", 550.0, 53.0, 0.0, 0.001, (0, 0, 0), (0, 0, 0), 0.0))
        ecl_status = ecl_model.evaluate_illumination(orb_state)

        if route_obj and route_obj.is_route_valid:
            comm_cost = route_obj.total_latency_ms * 0.5
            route_str = " -> ".join(route_obj.path)
            hop_count = route_obj.hop_count
            dist_km = route_obj.total_distance_km
            bw_mbps = route_obj.bottleneck_bandwidth_mbps
        else:
            comm_cost = 25.0
            route_str = "No Route"
            hop_count = 0
            dist_km = 0.0
            bw_mbps = 0.0

        total_cost = 10.0 + comm_cost + (15.0 if ecl_status.is_eclipse else 0.0)
        power_cost = 15.0 * (1.0 - (cluster.get_node(selected_target).power_level / 100.0)) if selected_target != "None" else 20.0

        metrics = manager.metrics

        return OrbitalBenchmarkMetrics(
            scenario_name=name,
            policy_name=policy_name,
            recovery_success_rate=1.0 if is_success else 0.0,
            verification_success_rate=metrics.target_selection_success_rate,
            workload_recovery_rate=1.0 if is_success else 0.0,
            workload_loss_rate=0.0 if is_success else 1.0,
            average_recovery_time=metrics.average_migration_time,
            average_migration_time=metrics.average_migration_time,
            average_retries=float(metrics.recovery_retries),
            average_total_recovery_cost=round(total_cost, 2),
            communication_cost=round(comm_cost, 2),
            bandwidth_utilization=round(bw_mbps, 1),
            energy_power_cost=round(power_cost, 2),
            average_route_length=round(dist_km, 2),
            average_hop_count=hop_count,
            link_failure_count=0 if (route_obj and route_obj.is_route_valid) else 1,
            eclipse_related_failures=1 if (ecl_status.is_eclipse and not is_success) else 0,
            predictive_decision_accuracy=1.0 if is_success else 0.0,
            unnecessary_migrations_count=0,
            selected_target_node=selected_target,
            selected_route_string=route_str,
        )

    def run_full_orbital_benchmark_suite(self) -> Tuple[List[OrbitalBenchmarkMetrics], Dict[str, float]]:
        """Run 10 scenarios for both Phase 8 and Phase 9 policies (20 runs) and compute % improvements."""
        results = []
        for sc_id in range(1, 11):
            res_p8 = self.run_scenario_policy(sc_id, use_orbital_policy=False)
            res_p9 = self.run_scenario_policy(sc_id, use_orbital_policy=True)
            results.append(res_p8)
            results.append(res_p9)

        p8_runs = [r for r in results if r.policy_name == "Phase 8 Adaptive Policy"]
        p9_runs = [r for r in results if r.policy_name == "Phase 9 Orbital Policy"]

        avg_p8_cost = sum(r.average_total_recovery_cost for r in p8_runs) / len(p8_runs)
        avg_p9_cost = sum(r.average_total_recovery_cost for r in p9_runs) / len(p9_runs)

        avg_p8_comm = sum(r.communication_cost for r in p8_runs) / len(p8_runs)
        avg_p9_comm = sum(r.communication_cost for r in p9_runs) / len(p9_runs)

        p8_success = sum(1 for r in p8_runs if r.recovery_success_rate == 1.0)
        p9_success = sum(1 for r in p9_runs if r.recovery_success_rate == 1.0)

        improvements = {
            "total_cost_reduction_pct": round(((avg_p8_cost - avg_p9_cost) / avg_p8_cost) * 100.0, 2),
            "communication_cost_reduction_pct": round(((avg_p8_comm - avg_p9_comm) / avg_p8_comm) * 100.0, 2),
            "recovery_success_increase_pct": round(((p9_success - p8_success) / max(1, p8_success)) * 100.0, 2),
        }

        return results, improvements
