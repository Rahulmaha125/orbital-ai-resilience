"""AutonomousResilienceController implementing the 19-step continuous autonomous resilience control loop."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
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
from orbital_ai_resilience.optimization.orbital_policy import OrbitalAwareRecoveryPolicy, OrbitalRecoveryCostModel
from orbital_ai_resilience.optimization.policy import AdaptiveRecoveryPolicy, DeterministicBaselinePolicy
from orbital_ai_resilience.orbital.eclipse import EclipseModel, EclipseStatus
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.prediction import OrbitalPredictionModel
from orbital_ai_resilience.orbital.propagation import OrbitalPropagator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.recovery.migration import MigrationEvent, MigrationManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.utils.logger import StateLogger
from orbital_ai_resilience.verification.verifier import OutputVerifier
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


@dataclass
class ControllerStepSummary:
    """Summary of actions and state updates performed during a single simulation tick step."""

    tick: int
    active_nodes: int
    online_nodes: int
    degraded_nodes: int
    quarantined_nodes: int
    isolated_nodes: int
    detection_alerts: int
    recovery_events: int
    verified_recoveries: int
    quarantine_events: int
    isolation_events: int
    elapsed_step_sec: float


class AutonomousResilienceController:
    """Central continuous autonomous control-loop coordinator running the 19-step resilience cycle."""

    def __init__(
        self,
        node_count: int = 5,
        policy_name: str = "Phase 9 Orbital Policy",
        detector_name: str = "Statistical",
        seed: int = 42,
    ) -> None:
        self.node_count: int = node_count
        self.policy_name: str = policy_name
        self.detector_name: str = detector_name
        self.seed: int = seed
        self.current_tick: int = 0
        self.is_paused: bool = False

        self.reset_controller()

    def reset_controller(self) -> None:
        """Reset simulation environment and sub-engines to baseline state."""
        self.current_tick = 0
        self.logger = StateLogger()
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=self.node_count)
        self.fault_injector = FaultInjector()
        self.quarantine_manager = QuarantineManager(logger=self.logger)
        self.policy = MigrationPolicy()

        # Anomaly Detectors
        self.stat_detector = StatisticalDetector()
        self.ml_detector = IsolationForestDetector(random_state=self.seed)

        # Orbital & Constellation Subsystems
        self.propagator = OrbitalPropagator()
        self.eclipse_model = EclipseModel()
        self.prediction_model = OrbitalPredictionModel(propagator=self.propagator, eclipse_model=self.eclipse_model)
        self.router = ConstellationRouter(eclipse_model=self.eclipse_model)
        self.bandwidth_model = BandwidthModel()

        # Optimization & Policies
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
        self.adaptive_policy = AdaptiveRecoveryPolicy(policy=self.policy)
        self.baseline_policy = DeterministicBaselinePolicy()
        self.optimizer = AdaptiveRecoveryOptimizer(history=self.recovery_history, cost_model=self.cost_model)

        outer = self

        class ControllerTargetSelector(TargetSelector):
            def select_best_target(
                self,
                cluster: VirtualCluster,
                source_node_id: str,
                workload: Any,
                quarantine_manager: Optional[QuarantineManager] = None,
            ) -> Tuple[Optional[str], float, Dict[str, Any]]:
                if outer.policy_name == "Phase 9 Orbital Policy":
                    tid, route, sc, exp = outer.orbital_policy.select_target_and_route(
                        cluster, source_node_id, workload, outer.recovery_history, quarantine_manager=quarantine_manager
                    )
                    cand_scores = {tid: sc} if tid else {}
                    return tid, sc, cand_scores
                elif outer.policy_name == "Adaptive Recovery":
                    tid, sc, exp = outer.adaptive_policy.select_target(
                        cluster, source_node_id, workload, outer.recovery_history, quarantine_manager=quarantine_manager
                    )
                    return tid, sc, exp.get("candidate_scores", {tid: sc} if tid else {})
                else:
                    tid, sc, exp = outer.baseline_policy.select_target(
                        cluster, source_node_id, workload, outer.recovery_history, quarantine_manager=quarantine_manager
                    )
                    return tid, sc, exp.get("candidate_scores", {tid: sc} if tid else {})

        self.selector = ControllerTargetSelector(policy=self.policy, quarantine_manager=self.quarantine_manager)
        self.verifier = OutputVerifier()
        self.migration_manager = MigrationManager(
            policy=self.policy,
            selector=self.selector,
            verifier=self.verifier,
            quarantine_manager=self.quarantine_manager,
            logger=self.logger,
        )

        self.execution_history_logs: List[Dict[str, Any]] = []
        self.detection_event_history: List[DetectionResult] = []
        self.step_summaries: List[ControllerStepSummary] = []

    def get_active_detector(self) -> BaseDetector:
        """Return currently active anomaly detector instance."""
        if self.detector_name == "ML Isolation Forest":
            return self.ml_detector
        return self.stat_detector

    def step(self) -> ControllerStepSummary:
        """Execute the 19-step continuous autonomous resilience control loop for 1 tick."""
        t_start = time.time()
        tick = self.current_tick

        self.fault_injector.advance_tick(tick)
        self.fault_injector.apply_physical_telemetry_faults(self.cluster, tick=tick)
        self.cluster.step_all(timestamp=float(tick))

        constellation_states = self.propagator.generate_constellation_states(num_satellites=len(self.cluster.nodes), tick=float(tick))

        detector = self.get_active_detector()
        detection_alerts = 0
        recovery_events = 0
        verified_recoveries = 0
        quarantine_events = 0
        isolation_events = 0

        for node_id, node in list(self.cluster.nodes.items()):
            if node.status in (NodeStatus.ONLINE, NodeStatus.DEGRADED):
                w = SyntheticAIWorkload(name=f"tick_{tick}_{node_id}", seed=self.seed + tick)
                node.assign_workload(w)

                exec_log = w.execute_on_node(node, fault_injector=self.fault_injector, tick=tick)
                self.execution_history_logs.append(exec_log)

                node_logs = [l for l in self.execution_history_logs if l.get("target_node_id") == node_id]
                det = detector.evaluate(exec_log, history_logs=node_logs)
                self.detection_event_history.append(det)

                if det.is_anomaly or det.is_silent_degradation:
                    detection_alerts += 1

                if self.policy.should_migrate(det):
                    event = self.migration_manager.execute_autonomous_recovery(
                        cluster=self.cluster,
                        source_node_id=node_id,
                        workload=w,
                        detection_result=det,
                        fault_injector=self.fault_injector,
                    )

                    if event:
                        recovery_events += 1
                        if event.verification_status.value == "VERIFIED":
                            verified_recoveries += 1
                        elif event.verification_status.value == "VERIFICATION_FAILED":
                            quarantine_events += 1

                        if event.target_node_id:
                            self.optimizer.record_outcome(
                                node_id=event.target_node_id,
                                verification_result=event.verification_status,
                                was_quarantined=(event.verification_status.value == "VERIFICATION_FAILED"),
                                duration_sec=event.details.get("elapsed_time_sec", 0.0),
                                workload_id=w.workload_id,
                            )

        self.current_tick += 1
        t_elapsed = time.time() - t_start

        nodes = self.cluster.list_nodes()
        summary = ControllerStepSummary(
            tick=tick,
            active_nodes=len(nodes),
            online_nodes=sum(1 for n in nodes if n.status == NodeStatus.ONLINE),
            degraded_nodes=sum(1 for n in nodes if n.status == NodeStatus.DEGRADED),
            quarantined_nodes=len(self.quarantine_manager.get_quarantined_node_ids()),
            isolated_nodes=len(self.quarantine_manager.get_isolated_node_ids()),
            detection_alerts=detection_alerts,
            recovery_events=recovery_events,
            verified_recoveries=verified_recoveries,
            quarantine_events=quarantine_events,
            isolation_events=isolation_events,
            elapsed_step_sec=t_elapsed,
        )
        self.step_summaries.append(summary)
        return summary

    def run_ticks(self, n: int = 100) -> List[ControllerStepSummary]:
        """Advance controller by N ticks."""
        summaries = []
        for _ in range(n):
            if self.is_paused:
                break
            summaries.append(self.step())
        return summaries
