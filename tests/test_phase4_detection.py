"""Unit and integration tests for Phase 4: Anomaly & Silent AI Degradation Detection."""

import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.types import HealthState
from orbital_ai_resilience.detection.behavior_score import BehavioralScoreEvaluator
from orbital_ai_resilience.detection.benchmark import DetectionBenchmark, PhysicalHealthOnlyDetector
from orbital_ai_resilience.detection.ml_detector import IsolationForestDetector
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.detection.types import BehavioralState
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class TestPhase4Detection(unittest.TestCase):
    """Test suite for Phase 4 anomaly detection, behavioral scoring, and benchmarks."""

    def setUp(self) -> None:
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=2)
        self.stat_detector = StatisticalDetector()
        self.ml_detector = IsolationForestDetector(random_state=42)
        self.evaluator = BehavioralScoreEvaluator()

    def test_behavioral_score_evaluator_nominal(self) -> None:
        """Verify perfect score (100.0) for zero deviation."""
        dev = {"mse": 0.0, "mae": 0.0, "cosine_sim": 1.0}
        score, state, sub = self.evaluator.compute_score(dev)
        self.assertEqual(score, 100.0)
        self.assertEqual(state, BehavioralState.NORMAL)

    def test_behavioral_score_transitions(self) -> None:
        """Verify behavioral state transitions across NORMAL -> WARNING -> DEGRADED -> CRITICAL."""
        # 1. WARNING (Score ~80.0)
        dev_warn = {"mse": 0.003, "mae": 0.015, "cosine_sim": 0.992}
        score_w, state_w, _ = self.evaluator.compute_score(dev_warn)
        self.assertEqual(state_w, BehavioralState.WARNING)

        # 2. DEGRADED (Score ~67.5)
        dev_deg = {"mse": 0.006, "mae": 0.025, "cosine_sim": 0.982}
        score_d, state_d, _ = self.evaluator.compute_score(dev_deg)
        self.assertEqual(state_d, BehavioralState.DEGRADED)

        # 3. CRITICAL (Score ~20.0)
        dev_crit = {"mse": 0.018, "mae": 0.07, "cosine_sim": 0.92}
        score_c, state_c, _ = self.evaluator.compute_score(dev_crit)
        self.assertEqual(state_c, BehavioralState.CRITICAL)

    def test_statistical_detector_silent_degradation(self) -> None:
        """Verify StatisticalDetector identifies silent AI degradation scenario."""
        node = self.cluster.get_node("node-1")
        injector = FaultInjector()
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-1",
                start_tick=0,
                intensity=0.10,
                seed=42,
            )
        )
        w = SyntheticAIWorkload(name="task_silent", seed=42)
        exec_log = w.execute_on_node(node, fault_injector=injector, tick=0)

        res = self.stat_detector.evaluate(exec_log)

        self.assertTrue(res.is_anomaly)
        self.assertTrue(res.is_silent_degradation)
        self.assertEqual(res.physical_health_score, 100.0)
        self.assertEqual(res.physical_health_state, HealthState.HEALTHY)
        self.assertIn(res.behavioral_state, (BehavioralState.WARNING, BehavioralState.DEGRADED, BehavioralState.CRITICAL))

    def test_ml_isolation_forest_detector(self) -> None:
        """Verify IsolationForestDetector trains on baseline logs and detects silent degradation."""
        # 1. Generate clean baseline logs
        node = self.cluster.get_node("node-1")
        baseline_logs = []
        for t in range(15):
            w = SyntheticAIWorkload(name=f"base_{t}", seed=42 + t)
            baseline_logs.append(w.execute_on_node(node, tick=t))

        self.ml_detector.fit(baseline_logs)

        # 2. Generate silent degradation log
        injector = FaultInjector()
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-1",
                start_tick=0,
                intensity=0.15,
                seed=42,
            )
        )
        w_fault = SyntheticAIWorkload(name="fault_task", seed=42)
        fault_log = w_fault.execute_on_node(node, fault_injector=injector, tick=0)

        res = self.ml_detector.evaluate(fault_log)
        self.assertTrue(res.is_anomaly)
        self.assertTrue(res.is_silent_degradation)

    def test_temporary_anomaly_recovery(self) -> None:
        """Verify detector flags temporary anomaly and returns to normal state after recovery."""
        benchmark = DetectionBenchmark(seed=42)
        metrics_stat = benchmark.evaluate_detector_on_scenario(self.stat_detector, scenario_id=4)

        # Temporary anomaly at tick 5 only -> True Positives = 1, False Positives = 0
        self.assertEqual(metrics_stat.first_detection_tick, 5)
        self.assertEqual(metrics_stat.true_positives, 1)
        self.assertEqual(metrics_stat.false_positives, 0)

    def test_physical_telemetry_only_anomaly(self) -> None:
        """Verify statistical AI detector does not flag silent AI anomaly when only physical stress occurs with clean AI outputs."""
        node = self.cluster.get_node("node-1")
        injector = FaultInjector()
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.ENVIRONMENTAL_STRESS,
                target_node_id="node-1",
                intensity=0.80,
            )
        )
        injector.apply_physical_telemetry_faults(self.cluster, tick=0)

        w = SyntheticAIWorkload(name="clean_ai", seed=42)
        exec_log = w.execute_on_node(node, fault_injector=injector, tick=0)

        res = self.stat_detector.evaluate(exec_log)
        self.assertFalse(res.is_silent_degradation)

    def test_benchmark_suite_execution(self) -> None:
        """Verify full benchmark runner evaluates 6 scenarios across 3 detectors (18 metrics)."""
        benchmark = DetectionBenchmark(seed=42)
        all_metrics = benchmark.run_all_benchmarks()

        self.assertEqual(len(all_metrics), 18)
        
        # Verify physical-only baseline fails on silent degradation (Scenario 6)
        phys_sc6 = [m for m in all_metrics if m.scenario_name.startswith("Scenario 6") and m.detector_name == "PhysicalHealth_Only_Baseline"][0]
        self.assertIsNone(phys_sc6.first_detection_tick)
        self.assertEqual(phys_sc6.recall, 0.0)

        # Verify Statistical AI detector succeeds on silent degradation (Scenario 6)
        stat_sc6 = [m for m in all_metrics if m.scenario_name.startswith("Scenario 6") and m.detector_name == "Statistical_ZScore_Detector"][0]
        self.assertEqual(stat_sc6.first_detection_tick, 5)
        self.assertGreater(stat_sc6.recall, 0.0)


if __name__ == "__main__":
    unittest.main()
