"""Scientific 2-policy benchmarking suite comparing Deterministic Baseline vs. Adaptive Optimization."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile, FaultType
from orbital_ai_resilience.optimization.cost_model import RecoveryCostModel
from orbital_ai_resilience.optimization.features import OptimizationFeatureBuilder
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.optimizer import AdaptiveRecoveryOptimizer
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.recovery.migration import MigrationManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.verification.verifier import OutputVerifier
from orbital_ai_resilience.verification.types import VerificationResultState
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


@dataclass
class PolicyComparisonMetrics:
    """Quantitative performance comparison metrics for Baseline vs. Adaptive policy."""

    scenario_name: str
    policy_name: str
    recovery_success_rate: float
    verification_success_rate: float
    workload_recovery_rate: float
    workload_loss_rate: float
    average_migration_time: float
    average_retries: float
    average_recovery_cost: float
    average_communication_cost: float
    average_power_cost: float
    target_selection_accuracy: float
    quarantined_targets_count: int
    total_reward: float
    selected_target_node: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize comparison metrics to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "policy_name": self.policy_name,
            "recovery_success_rate": round(self.recovery_success_rate, 4),
            "verification_success_rate": round(self.verification_success_rate, 4),
            "workload_recovery_rate": round(self.workload_recovery_rate, 4),
            "workload_loss_rate": round(self.workload_loss_rate, 4),
            "average_migration_time": round(self.average_migration_time, 4),
            "average_retries": round(self.average_retries, 2),
            "average_recovery_cost": round(self.average_recovery_cost, 2),
            "average_communication_cost": round(self.average_communication_cost, 2),
            "average_power_cost": round(self.average_power_cost, 2),
            "target_selection_accuracy": round(self.target_selection_accuracy, 4),
            "quarantined_targets_count": self.quarantined_targets_count,
            "total_reward": round(self.total_reward, 2),
            "selected_target_node": self.selected_target_node,
        }


class OptimizationBenchmark:
    """Executes scientific 2-policy comparison across 10 controlled scenarios."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed

    def setup_scenario(self, scenario_id: int) -> Tuple[str, VirtualCluster, FaultInjector, RecoveryHistory, SyntheticAIWorkload]:
        """Construct deterministic scenario cluster state, fault injection, history, and workload."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        injector = FaultInjector()
        history = RecoveryHistory()

        if scenario_id == 1:
            name = "Scenario 1: Normal Healthy Cluster"
            w = SyntheticAIWorkload(name="task_sc1", seed=self.seed)

        elif scenario_id == 2:
            name = "Scenario 2: Gradual Silent AI Degradation"
            w = SyntheticAIWorkload(name="task_sc2", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.05, seed=self.seed))

        elif scenario_id == 3:
            name = "Scenario 3: Sudden Severe AI Degradation"
            w = SyntheticAIWorkload(name="task_sc3", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.25, seed=self.seed))

        elif scenario_id == 4:
            name = "Scenario 4: Multiple Healthy Targets (Different Latency)"
            cluster.get_node("node-2").update_telemetry(latency=25.0)
            cluster.get_node("node-4").update_telemetry(latency=40.0)
            w = SyntheticAIWorkload(name="task_sc4", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 5:
            name = "Scenario 5: Multiple Healthy Targets (Different Power Levels)"
            cluster.get_node("node-1").update_telemetry(power_level=50.0)
            cluster.get_node("node-2").update_telemetry(power_level=95.0)
            cluster.get_node("node-4").update_telemetry(power_level=30.0)
            w = SyntheticAIWorkload(name="task_sc5", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 6:
            name = "Scenario 6: Different Compute Capacities"
            cluster.get_node("node-1").compute_capacity = 50.0
            cluster.get_node("node-2").compute_capacity = 200.0
            w = SyntheticAIWorkload(name="task_sc6", required_compute=40.0, seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 7:
            name = "Scenario 7: Historical Unreliable Target"
            history.record_outcome("node-1", VerificationResultState.VERIFICATION_FAILED, was_quarantined=True)
            history.record_outcome("node-1", VerificationResultState.VERIFICATION_FAILED, was_quarantined=True)
            w = SyntheticAIWorkload(name="task_sc7", seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 8:
            name = "Scenario 8: Cascading Target Failure"
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))
            injector.add_profile(FaultProfile(fault_type=FaultType.MEMORY_BIT_FLIP, target_node_id="node-1", intensity=0.20, seed=99))
            w = SyntheticAIWorkload(name="task_sc8", seed=self.seed)

        elif scenario_id == 9:
            name = "Scenario 9: Resource-Constrained Cluster"
            for n in cluster.nodes.values():
                n.memory_capacity = 1000.0
            w = SyntheticAIWorkload(name="task_sc9", required_memory=800.0, seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15, seed=self.seed))

        elif scenario_id == 10:
            name = "Scenario 10: High-Criticality Workload"
            w = SyntheticAIWorkload(name="high_criticality_defense_task", required_compute=50.0, required_memory=4096.0, seed=self.seed)
            injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.20, seed=self.seed))
        else:
            raise ValueError(f"Invalid scenario_id: {scenario_id}")

        return name, cluster, injector, history, w

    def run_scenario_policy(
        self,
        scenario_id: int,
        use_adaptive: bool,
    ) -> PolicyComparisonMetrics:
        """Run single policy (Baseline or Adaptive) on a scenario with scientific fairness."""
        name, cluster, injector, history, workload = self.setup_scenario(scenario_id)
        policy_name = "Adaptive Policy" if use_adaptive else "Baseline Policy"

        qm = QuarantineManager()
        pol = MigrationPolicy()
        detector = StatisticalDetector()
        verifier = OutputVerifier()
        optimizer = AdaptiveRecoveryOptimizer(history=history)

        class CustomSelector(TargetSelector):
            def select_best_target(
                self,
                cluster: VirtualCluster,
                source_node_id: str,
                workload: Any,
                quarantine_manager: Any = None,
            ):
                p_name = "adaptive" if use_adaptive else "baseline"
                tid, sc, exp = optimizer.select_target_node(
                    cluster, source_node_id, workload, policy_name=p_name, quarantine_manager=quarantine_manager
                )
                cand_scores = exp.get("candidate_scores", {}) if "candidate_scores" in exp else {tid: sc}
                return tid, sc, cand_scores

        selector = CustomSelector(policy=pol, quarantine_manager=qm)
        manager = MigrationManager(policy=pol, selector=selector, verifier=verifier, quarantine_manager=qm)

        n3 = cluster.get_node("node-3")
        n3.assign_workload(workload)
        exec_log = workload.execute_on_node(n3, fault_injector=injector, tick=0)
        det = detector.evaluate(exec_log)

        event = manager.execute_autonomous_recovery(cluster, "node-3", workload, det, fault_injector=injector)

        selected_node_id = event.target_node_id if (event and event.target_node_id) else "None"
        selected_node = cluster.get_node(selected_node_id) if selected_node_id != "None" else None

        if selected_node:
            fb = OptimizationFeatureBuilder()
            feats = fb.build_features(selected_node, workload, history)
            cost_model = RecoveryCostModel()
            cost_breakdown = cost_model.calculate_cost(feats)
            total_cost = cost_breakdown.total_cost
            comm_cost = cost_breakdown.communication_cost
            power_cost = cost_breakdown.power_cost
        else:
            total_cost = 50.0
            comm_cost = 20.0
            power_cost = 15.0

        is_success = event.migration_status.value == "COMPLETED" if event else False
        v_state = event.verification_status if event else VerificationResultState.UNVERIFIED
        was_quarantined = manager.metrics.quarantined_nodes > 0

        reward_breakdown = optimizer.compute_reward(
            is_success=is_success,
            verification_state=v_state,
            was_quarantined=was_quarantined,
            recovery_cost=total_cost,
            is_workload_lost=(not is_success),
        )

        metrics = manager.metrics

        return PolicyComparisonMetrics(
            scenario_name=name,
            policy_name=policy_name,
            recovery_success_rate=1.0 if is_success else 0.0,
            verification_success_rate=metrics.target_selection_success_rate,
            workload_recovery_rate=1.0 if is_success else 0.0,
            workload_loss_rate=0.0 if is_success else 1.0,
            average_migration_time=metrics.average_migration_time,
            average_retries=float(metrics.recovery_retries),
            average_recovery_cost=total_cost,
            average_communication_cost=comm_cost,
            average_power_cost=power_cost,
            target_selection_accuracy=1.0 if is_success else 0.0,
            quarantined_targets_count=metrics.quarantined_nodes,
            total_reward=reward_breakdown.total_reward,
            selected_target_node=selected_node_id,
        )

    def run_full_benchmark_suite(self) -> Tuple[List[PolicyComparisonMetrics], Dict[str, float]]:
        """Run 10 scenarios for both Baseline and Adaptive policies (20 runs) and compute % improvements."""
        results = []
        for sc_id in range(1, 11):
            res_base = self.run_scenario_policy(sc_id, use_adaptive=False)
            res_adapt = self.run_scenario_policy(sc_id, use_adaptive=True)
            results.append(res_base)
            results.append(res_adapt)

        base_runs = [r for r in results if r.policy_name == "Baseline Policy"]
        adapt_runs = [r for r in results if r.policy_name == "Adaptive Policy"]

        avg_base_cost = sum(r.average_recovery_cost for r in base_runs) / len(base_runs)
        avg_adapt_cost = sum(r.average_recovery_cost for r in adapt_runs) / len(adapt_runs)

        avg_base_reward = sum(r.total_reward for r in base_runs) / len(base_runs)
        avg_adapt_reward = sum(r.total_reward for r in adapt_runs) / len(adapt_runs)

        avg_base_comm = sum(r.average_communication_cost for r in base_runs) / len(base_runs)
        avg_adapt_comm = sum(r.average_communication_cost for r in adapt_runs) / len(adapt_runs)

        improvements = {
            "recovery_cost_reduction_pct": round(((avg_base_cost - avg_adapt_cost) / avg_base_cost) * 100.0, 2),
            "communication_cost_reduction_pct": round(((avg_base_comm - avg_adapt_comm) / avg_base_comm) * 100.0, 2),
            "total_reward_improvement_pct": round(((avg_adapt_reward - avg_base_reward) / abs(avg_base_reward)) * 100.0, 2),
        }

        return results, improvements
