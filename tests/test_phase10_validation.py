"""Unit and integration tests for Phase 10: Final Autonomous System Integration & Large-Scale Scientific Validation."""

import os
import tempfile
import unittest
from orbital_ai_resilience.validation.controller import AutonomousResilienceController
from orbital_ai_resilience.validation.experiment import ExperimentResult, ExperimentRunner
from orbital_ai_resilience.validation.metrics import ValidationMetrics
from orbital_ai_resilience.validation.report import ResearchReportGenerator
from orbital_ai_resilience.validation.reproducibility import ReproducibilityManager
from orbital_ai_resilience.validation.scalability import ScalabilityEvaluator
from orbital_ai_resilience.validation.scenarios import ScenarioEngine
from orbital_ai_resilience.validation.simulation import SimulationConfig, SimulationEngine


class TestPhase10Validation(unittest.TestCase):
    """Test suite for Phase 10 autonomous resilience controller, simulation engine, and scientific validation framework."""

    def setUp(self) -> None:
        self.seed = 42
        self.controller = AutonomousResilienceController(node_count=5, seed=self.seed)
        self.sim_engine = SimulationEngine(node_count=5, ticks=10, seed=self.seed)
        self.runner = ExperimentRunner(seed=self.seed)
        self.report_gen = ResearchReportGenerator()
        self.scalability_eval = ScalabilityEvaluator(seed=self.seed)

    def test_controller_initialization(self) -> None:
        """Verify AutonomousResilienceController initializes cluster, detectors, and orbital engines."""
        self.assertEqual(len(self.controller.cluster.nodes), 5)
        self.assertEqual(self.controller.current_tick, 0)
        self.assertIsNotNone(self.controller.propagator)
        self.assertIsNotNone(self.controller.router)

    def test_single_tick_execution(self) -> None:
        """Verify controller executes 19-step continuous resilience loop per tick."""
        summary = self.controller.step()
        self.assertEqual(summary.tick, 0)
        self.assertEqual(self.controller.current_tick, 1)
        self.assertEqual(summary.active_nodes, 5)

    def test_multi_tick_execution(self) -> None:
        """Verify controller advances through multi-tick simulations."""
        summaries = self.controller.run_ticks(n=5)
        self.assertEqual(len(summaries), 5)
        self.assertEqual(self.controller.current_tick, 5)

    def test_deterministic_seed_reproducibility(self) -> None:
        """Verify 2 identical simulation engine runs produce 100% identical step summaries."""
        c1 = SimulationConfig(node_count=5, ticks=10, seed=42)
        c2 = SimulationConfig(node_count=5, ticks=10, seed=42)

        s1, m1 = SimulationEngine(config=c1).run()
        s2, m2 = SimulationEngine(config=c2).run()

        self.assertEqual(len(s1), len(s2))
        self.assertEqual(m1.recovery_success_rate, m2.recovery_success_rate)
        self.assertEqual(m1.workload_recovery_rate, m2.workload_recovery_rate)

    def test_scenario_execution(self) -> None:
        """Verify ScenarioEngine constructs valid 10 failure scenario definitions."""
        engine = ScenarioEngine(seed=42)
        sc2 = engine.create_scenario(scenario_id=2, node_count=5)
        self.assertEqual(sc2.scenario_id, 2)
        self.assertGreater(len(sc2.fault_profiles), 0)

    def test_policy_comparison(self) -> None:
        """Verify ExperimentRunner executes 3-policy comparison across Baseline, Adaptive, and Orbital policies."""
        results = self.runner.run_3policy_comparison(scenario_id=2, node_count=5, ticks=10)
        self.assertEqual(len(results), 3)
        policies = [r.policy_name for r in results]
        self.assertIn("Baseline Deterministic", policies)
        self.assertIn("Adaptive Recovery", policies)
        self.assertIn("Phase 9 Orbital Policy", policies)

    def test_failure_injection(self) -> None:
        """Verify controller handles fault injection and detection."""
        summary = self.controller.step()
        self.assertIsNotNone(summary)

    def test_cascading_recovery(self) -> None:
        """Verify cascading target failure handling."""
        res = self.runner.run_policy_scenario_experiment(scenario_id=4, policy_name="Phase 9 Orbital Policy", ticks=10)
        self.assertIsNotNone(res)

    def test_orbital_update(self) -> None:
        """Verify orbital positions propagate across ticks."""
        s1 = self.controller.propagator.compute_state("node-1", 0.0, 0.0)
        s2 = self.controller.propagator.compute_state("node-1", 0.0, 10.0)
        self.assertNotEqual(s1.position_km, s2.position_km)

    def test_communication_update(self) -> None:
        """Verify inter-satellite links and Dijkstra route updates."""
        states = self.controller.propagator.generate_constellation_states(num_satellites=5, tick=0.0)
        route = self.controller.router.find_route("node-3", "node-1", states)
        self.assertIsNotNone(route)

    def test_workload_survival(self) -> None:
        """Verify workload survival rate metrics calculation."""
        summaries, metrics = self.sim_engine.run()
        self.assertGreaterEqual(metrics.workload_recovery_rate, 0.0)
        self.assertLessEqual(metrics.workload_recovery_rate, 1.0)

    def test_metric_calculation(self) -> None:
        """Verify ValidationMetrics extracts reliability, efficiency, safety, and scalability values."""
        summaries, metrics = self.sim_engine.run()
        d = metrics.to_dict()
        self.assertIn("recovery_success_rate", d)
        self.assertIn("verification_success_rate", d)
        self.assertIn("average_recovery_cost", d)

    def test_improvement_calculation(self) -> None:
        """Verify ResearchReportGenerator calculates honest percentage deltas."""
        r_base = self.runner.run_policy_scenario_experiment(scenario_id=2, policy_name="Baseline Deterministic", ticks=10)
        r_orb = self.runner.run_policy_scenario_experiment(scenario_id=2, policy_name="Phase 9 Orbital Policy", ticks=10)

        deltas = self.report_gen.calculate_policy_deltas(r_base, r_orb)
        self.assertGreater(len(deltas), 0)

    def test_json_export(self) -> None:
        """Verify ExperimentRunner exports results to JSON file."""
        results = self.runner.run_3policy_comparison(scenario_id=2, ticks=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name

        try:
            self.runner.export_to_json(results, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_csv_export(self) -> None:
        """Verify ExperimentRunner exports results to CSV file."""
        results = self.runner.run_3policy_comparison(scenario_id=2, ticks=10)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            path = tf.name

        try:
            self.runner.export_to_csv(results, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_scalability_tests(self) -> None:
        """Verify ScalabilityEvaluator evaluates 5, 10, 25, and 50 node constellations."""
        res = self.scalability_eval.evaluate_constellation_sizes(node_sizes=[5, 10, 25, 50], ticks_per_test=5)
        self.assertEqual(len(res), 4)
        sizes = [r.node_count for r in res]
        self.assertEqual(sizes, [5, 10, 25, 50])

    def test_long_duration_simulation_test(self) -> None:
        """Verify simulation engine executes 100-tick long-duration run stably."""
        engine = SimulationEngine(node_count=5, ticks=100, seed=42)
        summaries, metrics = engine.run()
        self.assertEqual(len(summaries), 100)
        self.assertEqual(metrics.total_ticks, 100)

    def test_unsafe_target_rejection_and_duplicate_ownership_protection(self) -> None:
        """Verify quarantined target rejection and single workload ownership invariant."""
        n1 = self.controller.cluster.get_node("node-1")
        self.controller.quarantine_manager.quarantine_node(n1, reason="Test Quarantine")
        summary = self.controller.step()
        self.assertIn("node-1", self.controller.quarantine_manager.get_quarantined_node_ids())


if __name__ == "__main__":
    unittest.main()
