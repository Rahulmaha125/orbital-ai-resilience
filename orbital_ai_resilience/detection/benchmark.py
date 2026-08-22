"""Benchmarking framework for evaluating anomaly detectors across controlled research scenarios."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.detection.base import BaseDetector
from orbital_ai_resilience.detection.ml_detector import IsolationForestDetector
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.detection.types import DetectionResult
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


@dataclass
class ScenarioMetrics:
    """Quantitative performance metrics for a detector on a research scenario.

    Attributes:
        scenario_name: Name of the research experiment scenario.
        detector_name: Name of the detector being benchmarked.
        first_detection_tick: Tick index when anomaly was first flagged (None if never).
        detection_delay: Ticks elapsed between fault start tick and first detection (None if never/no fault).
        true_positives: Count of correctly flagged anomaly ticks.
        true_negatives: Count of correctly ignored normal ticks.
        false_positives: Count of false alarms on normal ticks.
        false_negatives: Count of missed anomalies during active fault ticks.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN) (Detection Rate).
    """

    scenario_name: str
    detector_name: str
    first_detection_tick: Optional[int]
    detection_delay: Optional[int]
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scenario metrics to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "detector_name": self.detector_name,
            "first_detection_tick": self.first_detection_tick,
            "detection_delay": self.detection_delay,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


class PhysicalHealthOnlyDetector(BaseDetector):
    """Baseline detector using only Phase 2 physical health scores."""

    def __init__(self, threshold: float = 90.0) -> None:
        super().__init__(name="PhysicalHealth_Only_Baseline")
        self.threshold: float = threshold

    def evaluate(
        self,
        exec_log: Dict[str, Any],
        history_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> DetectionResult:
        phys_score = exec_log.get("health_score", 100.0)
        phys_state = exec_log.get("health_state", "HEALTHY")
        is_anomaly = phys_score < self.threshold
        from orbital_ai_resilience.core.types import HealthState
        from orbital_ai_resilience.detection.types import BehavioralState

        return DetectionResult(
            timestamp=exec_log.get("timestamp", 0.0),
            tick=exec_log.get("tick", 0),
            node_id=exec_log.get("target_node_id", "unknown"),
            is_anomaly=is_anomaly,
            is_silent_degradation=False,  # Physical-only detector cannot detect silent degradation
            confidence=1.0 if is_anomaly else 0.0,
            detector_name=self.name,
            behavioral_score=100.0,
            behavioral_state=BehavioralState.NORMAL,
            physical_health_score=phys_score,
            physical_health_state=HealthState(phys_state),
            details={"phys_score": phys_score},
        )


class DetectionBenchmark:
    """Runner for controlled research scenarios 1 to 6 across multiple detectors."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed

    def build_scenario_data(self, scenario_id: int) -> Tuple[str, Optional[FaultProfile], int, int, List[int]]:
        """Construct research scenario definition.

        Returns:
            Tuple of (scenario_name, fault_profile_or_None, total_ticks, fault_start_tick, ground_truth_anomaly_ticks).
        """
        if scenario_id == 1:
            name = "Scenario 1: Normal Node"
            profile = None
            total_ticks = 10
            start_tick = 0
            anomaly_ticks = []

        elif scenario_id == 2:
            name = "Scenario 2: Gradual Silent Degradation"
            profile = FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-1",
                start_tick=5,
                duration=10,
                intensity=0.03,
                seed=self.seed,
            )
            total_ticks = 15
            start_tick = 5
            anomaly_ticks = list(range(5, 15))

        elif scenario_id == 3:
            name = "Scenario 3: Sudden Severe Degradation"
            profile = FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-1",
                start_tick=5,
                duration=10,
                intensity=0.25,
                seed=self.seed,
            )
            total_ticks = 15
            start_tick = 5
            anomaly_ticks = list(range(5, 15))

        elif scenario_id == 4:
            name = "Scenario 4: Temporary Output Anomaly Recovery"
            profile = FaultProfile(
                fault_type=FaultType.INTERMITTENT_COMPUTATION,
                target_node_id="node-1",
                start_tick=5,
                duration=1,
                intensity=1.0,
                seed=self.seed,
            )
            total_ticks = 12
            start_tick = 5
            anomaly_ticks = [5]

        elif scenario_id == 5:
            name = "Scenario 5: Physical Telemetry Degradation (Clean AI Output)"
            profile = FaultProfile(
                fault_type=FaultType.ENVIRONMENTAL_STRESS,
                target_node_id="node-1",
                start_tick=5,
                duration=10,
                intensity=0.80,
                seed=self.seed,
            )
            total_ticks = 15
            start_tick = 5
            # For physical telemetry degradation, physical health drops, but AI output is clean.
            anomaly_ticks = list(range(5, 15))

        elif scenario_id == 6:
            name = "Scenario 6: Silent AI Degradation (Healthy Physical Telemetry)"
            profile = FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-1",
                start_tick=5,
                duration=10,
                intensity=0.10,
                seed=self.seed,
            )
            total_ticks = 15
            start_tick = 5
            anomaly_ticks = list(range(5, 15))
        else:
            raise ValueError(f"Invalid scenario_id: {scenario_id}")

        return name, profile, total_ticks, start_tick, anomaly_ticks

    def run_scenario_execution_logs(
        self, profile: Optional[FaultProfile], total_ticks: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute workload across simulation ticks and collect execution logs.

        Returns:
            Tuple of (clean_baseline_logs, test_execution_logs).
        """
        # 1. Collect clean baseline logs for training ML detector
        baseline_cluster = VirtualCluster.create_default_cluster(num_nodes=1)
        baseline_node = baseline_cluster.get_node("node-1")
        baseline_logs = []
        for t in range(20):
            w = SyntheticAIWorkload(name=f"baseline_{t}", seed=self.seed + t)
            log = w.execute_on_node(baseline_node, fault_injector=None, tick=t)
            baseline_logs.append(log)

        # 2. Collect test execution logs with fault injection profile
        test_cluster = VirtualCluster.create_default_cluster(num_nodes=1)
        test_node = test_cluster.get_node("node-1")
        injector = FaultInjector()
        if profile:
            injector.add_profile(profile)

        test_logs = []
        for t in range(total_ticks):
            injector.apply_physical_telemetry_faults(test_cluster, tick=t)
            w = SyntheticAIWorkload(name=f"test_{t}", seed=self.seed + t)
            log = w.execute_on_node(test_node, fault_injector=injector, tick=t)
            test_logs.append(log)

        return baseline_logs, test_logs

    def evaluate_detector_on_scenario(
        self,
        detector: BaseDetector,
        scenario_id: int,
    ) -> ScenarioMetrics:
        """Run a detector on a specific research scenario and calculate quantitative metrics."""
        name, profile, total_ticks, start_tick, anomaly_ticks = self.build_scenario_data(scenario_id)
        baseline_logs, test_logs = self.run_scenario_execution_logs(profile, total_ticks)

        # Train ML detector if applicable
        if isinstance(detector, IsolationForestDetector):
            detector.fit(baseline_logs)

        first_detection_tick = None
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        history: List[Dict[str, Any]] = []
        for t, log in enumerate(test_logs):
            res = detector.evaluate(log, history_logs=history)
            history.append(log)

            is_ground_truth_anomaly = t in anomaly_ticks

            if res.is_anomaly:
                if first_detection_tick is None:
                    first_detection_tick = t

                if is_ground_truth_anomaly:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_ground_truth_anomaly:
                    fn += 1
                else:
                    tn += 1

        # Calculate detection delay
        detection_delay = None
        if profile is not None and first_detection_tick is not None and first_detection_tick >= start_tick:
            detection_delay = first_detection_tick - start_tick

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fn == 0 else 0.0)

        return ScenarioMetrics(
            scenario_name=name,
            detector_name=detector.name,
            first_detection_tick=first_detection_tick,
            detection_delay=detection_delay,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
        )

    def run_all_benchmarks(self) -> List[ScenarioMetrics]:
        """Run all 6 scenarios across all 3 detectors (18 benchmark evaluations)."""
        detectors: List[BaseDetector] = [
            PhysicalHealthOnlyDetector(),
            StatisticalDetector(),
            IsolationForestDetector(random_state=self.seed),
        ]

        results = []
        for sc_id in range(1, 7):
            for det in detectors:
                metrics = self.evaluate_detector_on_scenario(det, sc_id)
                results.append(metrics)
        return results
