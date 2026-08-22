"""Dashboard simulation state manager integrating Phase 1-10 underlying engines."""

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from orbital_ai_resilience.constellation.bandwidth import BandwidthModel
from orbital_ai_resilience.constellation.routing import ConstellationRoute, ConstellationRouter
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.detection.base import BaseDetector
from orbital_ai_resilience.detection.ml_detector import IsolationForestDetector
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.detection.types import DetectionResult
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile, FaultType
from orbital_ai_resilience.optimization.cost_model import RecoveryCostModel
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.optimizer import AdaptiveRecoveryOptimizer
from orbital_ai_resilience.optimization.orbital_benchmark import OrbitalOptimizationBenchmark
from orbital_ai_resilience.optimization.orbital_policy import OrbitalAwareRecoveryPolicy, OrbitalRecoveryCostModel
from orbital_ai_resilience.optimization.policy import AdaptiveRecoveryPolicy, DeterministicBaselinePolicy
from orbital_ai_resilience.orbital.eclipse import EclipseModel, EclipseStatus
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.prediction import OrbitalPredictionModel
from orbital_ai_resilience.orbital.propagation import OrbitalPropagator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.recovery.migration import MigrationManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.utils.logger import StateLogger
from orbital_ai_resilience.validation.controller import AutonomousResilienceController
from orbital_ai_resilience.validation.experiment import ExperimentResult, ExperimentRunner
from orbital_ai_resilience.validation.report import ResearchReportGenerator
from orbital_ai_resilience.validation.scalability import ScalabilityEvaluator, ScalabilityResult
from orbital_ai_resilience.validation.simulation import SimulationConfig, SimulationEngine
from orbital_ai_resilience.verification.verifier import OutputVerifier
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class DashboardState:
    """Central simulation state controller for the Streamlit research dashboard.

    Integrates real underlying Phase 1-10 engines cleanly without duplicating logic.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self.current_tick: int = 0
        self.selected_detector_name: str = "Statistical"
        self.selected_recovery_policy_name: str = "Phase 9 Orbital Policy"
        self.auto_recovery_enabled: bool = True

        self.reset_simulation()

    def reset_simulation(self) -> None:
        """Reset cluster and all subsystem components to initial baseline state."""
        self.current_tick = 0
        self.logger = StateLogger()
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        self.fault_injector = FaultInjector()
        self.stat_detector = StatisticalDetector()
        self.ml_detector = IsolationForestDetector(random_state=self.seed)
        self.quarantine_manager = QuarantineManager(logger=self.logger)
        self.policy = MigrationPolicy()

        # Phase 9 & 10 Orbital & Validation Components
        self.propagator = OrbitalPropagator()
        self.eclipse_model = EclipseModel()
        self.prediction_model = OrbitalPredictionModel(propagator=self.propagator, eclipse_model=self.eclipse_model)
        self.router = ConstellationRouter(eclipse_model=self.eclipse_model)
        self.bandwidth_model = BandwidthModel()

        # Phase 8, 9 & 10 Optimization & Controller
        self.recovery_history = RecoveryHistory()
        self.cost_model = RecoveryCostModel()
        self.orbital_policy = OrbitalAwareRecoveryPolicy(
            policy=self.policy,
            propagator=self.propagator,
            eclipse_model=self.eclipse_model,
            prediction_model=self.prediction_model,
            router=self.router,
            bandwidth_model=self.bandwidth_model,
        )
        self.optimizer = AdaptiveRecoveryOptimizer(
            history=self.recovery_history,
            cost_model=self.cost_model,
        )

        class CustomDashboardSelector(TargetSelector):
            def __init__(daself, outer_state):
                super().__init__(policy=outer_state.policy, quarantine_manager=outer_state.quarantine_manager)
                daself.outer_state = outer_state

            def select_best_target(
                daself,
                cluster: VirtualCluster,
                source_node_id: str,
                workload: Any,
                quarantine_manager: Optional[QuarantineManager] = None,
            ) -> Tuple[Optional[str], float, Dict[str, Any]]:
                if daself.outer_state.selected_recovery_policy_name == "Phase 9 Orbital Policy":
                    tid, route, sc, exp = daself.outer_state.orbital_policy.select_target_and_route(
                        cluster, source_node_id, workload, daself.outer_state.recovery_history, quarantine_manager=quarantine_manager
                    )
                    cand_scores = {tid: sc} if tid else {}
                    return tid, sc, cand_scores
                else:
                    p_name = "baseline" if daself.outer_state.selected_recovery_policy_name == "Baseline Deterministic" else "adaptive"
                    tid, sc, exp = daself.outer_state.optimizer.select_target_node(
                        cluster, source_node_id, workload, policy_name=p_name, quarantine_manager=quarantine_manager
                    )
                    cand_scores = exp.get("candidate_scores", {}) if "candidate_scores" in exp else ({tid: sc} if tid else {})
                    return tid, sc, cand_scores

        self.selector = CustomDashboardSelector(self)
        self.verifier = OutputVerifier()
        self.migration_manager = MigrationManager(
            policy=self.policy,
            selector=self.selector,
            verifier=self.verifier,
            quarantine_manager=self.quarantine_manager,
            logger=self.logger,
        )

        self.execution_logs: List[Dict[str, Any]] = []
        self.detection_results: List[DetectionResult] = []
        self.baseline_trained: bool = False

        self._train_ml_detector()

    def _train_ml_detector(self) -> None:
        """Train ML Isolation Forest detector on clean baseline execution logs."""
        baseline_cluster = VirtualCluster.create_default_cluster(num_nodes=1)
        base_node = baseline_cluster.get_node("node-1")
        baseline_logs = []
        for t in range(15):
            w = SyntheticAIWorkload(name=f"baseline_train_{t}", seed=self.seed + t)
            log = w.execute_on_node(base_node, fault_injector=None, tick=t)
            baseline_logs.append(log)

        self.ml_detector.fit(baseline_logs)
        self.baseline_trained = True

    def get_active_detector(self) -> BaseDetector:
        """Return currently selected detector instance."""
        if self.selected_detector_name == "ML Isolation Forest":
            return self.ml_detector
        return self.stat_detector

    def inject_fault(
        self,
        target_node_id: str,
        fault_type: FaultType,
        intensity: float = 0.10,
        start_tick: Optional[int] = None,
        duration: Optional[int] = 10,
        seed: int = 42,
    ) -> FaultProfile:
        """Inject a controlled software fault profile using the existing FaultInjector."""
        tick = self.current_tick if start_tick is None else start_tick
        profile = FaultProfile(
            fault_type=fault_type,
            target_node_id=target_node_id,
            start_tick=tick,
            duration=duration,
            intensity=intensity,
            seed=seed,
        )
        self.fault_injector.add_profile(profile)
        return profile

    def advance_tick(self) -> List[Dict[str, Any]]:
        """Advance simulation tick by 1, execute workloads, apply faults, and trigger detection/recovery."""
        tick = self.current_tick
        self.fault_injector.advance_tick(tick)

        self.fault_injector.apply_physical_telemetry_faults(self.cluster, tick=tick)
        self.cluster.step_all(timestamp=float(tick))

        tick_logs = []
        detector = self.get_active_detector()

        for node_id, node in list(self.cluster.nodes.items()):
            if node.status in (NodeStatus.ONLINE, NodeStatus.DEGRADED):
                w = SyntheticAIWorkload(name=f"tick_{tick}_{node_id}", seed=self.seed + tick)
                node.assign_workload(w)

                exec_log = w.execute_on_node(node, fault_injector=self.fault_injector, tick=tick)
                self.execution_logs.append(exec_log)
                tick_logs.append(exec_log)

                node_history = [l for l in self.execution_logs if l.get("target_node_id") == node_id]
                det_result = detector.evaluate(exec_log, history_logs=node_history)
                self.detection_results.append(det_result)

                if self.auto_recovery_enabled and self.policy.should_migrate(det_result):
                    event = self.migration_manager.execute_autonomous_recovery(
                        cluster=self.cluster,
                        source_node_id=node_id,
                        workload=w,
                        detection_result=det_result,
                        fault_injector=self.fault_injector,
                    )
                    if event and event.target_node_id:
                        self.optimizer.record_outcome(
                            node_id=event.target_node_id,
                            verification_result=event.verification_status,
                            was_quarantined=(event.verification_status.value == "VERIFICATION_FAILED"),
                            duration_sec=event.details.get("elapsed_time_sec", 0.0),
                            workload_id=w.workload_id,
                        )

        self.current_tick += 1
        return tick_logs

    def run_n_ticks(self, n: int = 5) -> None:
        """Advance simulation by n ticks."""
        for _ in range(n):
            self.advance_tick()

    def run_cascading_failure_experiment(self) -> None:
        """Execute deterministic Phase 6 cascading failure recovery experiment."""
        self.reset_simulation()
        self.selected_recovery_policy_name = "Baseline Deterministic"

        n5 = self.cluster.get_node("node-5")
        n5.set_status(NodeStatus.DEGRADED)
        n5.update_telemetry(temperature=78.0)

        n3 = self.cluster.get_node("node-3")
        w_task = SyntheticAIWorkload(name="cascading_task", seed=42)
        n3.assign_workload(w_task)

        self.fault_injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-3",
                start_tick=0,
                duration=10,
                intensity=0.15,
                seed=42,
            )
        )
        self.fault_injector.add_profile(
            FaultProfile(
                fault_type=FaultType.MEMORY_BIT_FLIP,
                target_node_id="node-1",
                start_tick=0,
                duration=10,
                intensity=0.20,
                seed=99,
            )
        )

        exec_log = w_task.execute_on_node(n3, fault_injector=self.fault_injector, tick=0)
        self.execution_logs.append(exec_log)
        det_result = self.stat_detector.evaluate(exec_log)
        self.detection_results.append(det_result)

        event = self.migration_manager.execute_autonomous_recovery(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=w_task,
            detection_result=det_result,
            fault_injector=self.fault_injector,
        )
        if event and event.target_node_id:
            self.optimizer.record_outcome(
                node_id=event.target_node_id,
                verification_result=event.verification_status,
                was_quarantined=(event.verification_status.value == "VERIFICATION_FAILED"),
                duration_sec=event.details.get("elapsed_time_sec", 0.0),
                workload_id=w_task.workload_id,
            )

        self.current_tick += 1

    def run_phase10_3policy_comparison(self, scenario_id: int = 2, node_count: int = 5, ticks: int = 50) -> List[Dict[str, Any]]:
        """Run Phase 10 3-policy comparison experiment across Baseline, Adaptive, and Orbital-Aware policies."""
        runner = ExperimentRunner(seed=self.seed)
        results = runner.run_3policy_comparison(scenario_id=scenario_id, node_count=node_count, ticks=ticks)
        return [r.to_dict() for r in results]

    def run_phase10_scalability_eval(self) -> List[Dict[str, Any]]:
        """Run Phase 10 constellation scalability evaluation across 5, 10, 25, and 50 nodes."""
        evaluator = ScalabilityEvaluator(seed=self.seed)
        results = evaluator.evaluate_constellation_sizes(node_sizes=[5, 10, 25, 50], ticks_per_test=20)
        return [r.to_dict() for r in results]

    def get_cluster_summary_dict(self) -> Dict[str, Any]:
        """Aggregate current cluster metrics for dashboard cards."""
        nodes = self.cluster.list_nodes()
        total_nodes = len(nodes)
        online_nodes = sum(1 for n in nodes if n.status == NodeStatus.ONLINE)
        degraded_nodes = sum(1 for n in nodes if n.status == NodeStatus.DEGRADED)
        isolated_nodes = sum(1 for n in nodes if n.status == NodeStatus.ISOLATED)
        quarantined_nodes = len(self.quarantine_manager.get_quarantined_node_ids())

        phys_scores = [n.get_health_score() for n in nodes]
        avg_phys_health = sum(phys_scores) / len(phys_scores) if phys_scores else 0.0

        behav_eval = self.stat_detector.behavior_evaluator
        recent_logs = self.execution_logs[-total_nodes:] if self.execution_logs else []
        behav_scores = [
            behav_eval.compute_score(l.get("deviation", {}))[0] for l in recent_logs
        ] if recent_logs else [100.0]
        avg_behav_score = sum(behav_scores) / len(behav_scores) if behav_scores else 100.0

        active_workloads = sum(len(n.workload_queue) for n in nodes)
        metrics = self.migration_manager.metrics

        return {
            "total_nodes": total_nodes,
            "online_nodes": online_nodes,
            "degraded_nodes": degraded_nodes,
            "isolated_nodes": isolated_nodes,
            "quarantined_nodes": quarantined_nodes,
            "avg_physical_health": round(avg_phys_health, 1),
            "avg_behavioral_score": round(avg_behav_score, 1),
            "active_workloads": active_workloads,
            "total_migrations": metrics.total_migrations,
            "successful_migrations": metrics.successful_migrations,
            "verification_failures": metrics.verification_failures,
            "quarantined_count": metrics.quarantined_nodes,
        }

    def get_node_telemetry_dataframe(self, node_id: str) -> pd.DataFrame:
        """Extract historical telemetry for a selected node into a pandas DataFrame."""
        node = self.cluster.get_node(node_id)
        if not node:
            return pd.DataFrame()

        snapshots = node.telemetry_history.get_recent(50)
        data = []
        for s in snapshots:
            d = s.to_dict()
            d["health_score"] = node.get_health_score()
            data.append(d)

        return pd.DataFrame(data)

    def get_ai_behavior_dataframe(self, node_id: str) -> pd.DataFrame:
        """Extract AI behavioral execution statistics for a selected node into a DataFrame."""
        node_logs = [l for l in self.execution_logs if l.get("target_node_id") == node_id]
        if not node_logs:
            return pd.DataFrame()

        rows = []
        evaluator = self.stat_detector.behavior_evaluator

        for l in node_logs:
            dev = l.get("deviation", {})
            b_score, b_state, _ = evaluator.compute_score(dev)
            rows.append(
                {
                    "tick": l.get("tick", 0),
                    "timestamp": l.get("timestamp", 0.0),
                    "mse": dev.get("mse", 0.0),
                    "mae": dev.get("mae", 0.0),
                    "cosine_sim": dev.get("cosine_sim", 1.0),
                    "behavioral_score": b_score,
                    "behavioral_state": b_state.value,
                    "physical_health_score": l.get("health_score", 100.0),
                    "physical_health_state": l.get("health_state", "HEALTHY"),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["rolling_mse"] = df["mse"].rolling(window=5, min_periods=1).mean()
        return df

    def get_events_dataframe(self) -> pd.DataFrame:
        """Extract filterable audit event log into a pandas DataFrame."""
        events = self.logger.logs_history
        if not events:
            return pd.DataFrame(columns=["timestamp", "event_type", "details"])

        data = []
        for e in events:
            details_str = str(e.get("details", {}))
            data.append(
                {
                    "timestamp": pd.to_datetime(e.get("timestamp", 0.0), unit="s"),
                    "event_type": e.get("event_type", "EVENT"),
                    "details": details_str,
                }
            )
        return pd.DataFrame(data)
