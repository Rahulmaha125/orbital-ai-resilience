"""Unit tests for Phase 3: Fault Injection and Silent AI Degradation Simulation."""

import numpy as np
import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import HealthState, NodeStatus
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class TestFaultInjection(unittest.TestCase):
    """Test suite for deterministic fault injection and synthetic AI workload degradation."""

    def setUp(self) -> None:
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=3)
        self.injector = FaultInjector()

    def test_bit_flip_simulation(self) -> None:
        """Verify memory bit flips alter array values deterministically."""
        arr = np.ones((5, 5), dtype=np.float64)
        corrupted = self.injector.inject_memory_bit_flip(arr, intensity=0.20, seed=123)

        self.assertFalse(np.array_equal(arr, corrupted))
        # Reproducibility check: identical seed produces identical corruption
        corrupted_again = self.injector.inject_memory_bit_flip(arr, intensity=0.20, seed=123)
        self.assertTrue(np.array_equal(corrupted, corrupted_again))

    def test_parameter_perturbation_intensity_scaling(self) -> None:
        """Verify parameter perturbation noise scales with intensity."""
        weights = np.random.default_rng(42).standard_normal((10, 10))

        p_low = self.injector.inject_parameter_perturbation(weights, intensity=0.01, seed=42)
        p_high = self.injector.inject_parameter_perturbation(weights, intensity=0.25, seed=42)

        diff_low = np.mean(np.abs(weights - p_low))
        diff_high = np.mean(np.abs(weights - p_high))

        self.assertGreater(diff_high, diff_low)

    def test_output_drift(self) -> None:
        """Verify output drift applies directional baseline change increasing over ticks."""
        output = np.zeros((4, 4))
        drift1 = self.injector.inject_output_drift(output, intensity=0.10, seed=99, elapsed_ticks=0)
        drift2 = self.injector.inject_output_drift(output, intensity=0.10, seed=99, elapsed_ticks=5)

        self.assertGreater(np.max(np.abs(drift2)), np.max(np.abs(drift1)))

    def test_intermittent_computation_faults(self) -> None:
        """Verify intermittent fault triggers intermittently based on seed and probability."""
        profile = FaultProfile(
            fault_type=FaultType.INTERMITTENT_COMPUTATION,
            target_node_id="node-1",
            intensity=0.50,
            seed=42,
        )
        self.injector.add_profile(profile)
        weights = np.ones((4, 4))
        ref_out = np.ones((4, 4))

        # Check over multiple ticks that corruption triggers intermittently
        corrupted_ticks = 0
        for tick in range(10):
            w_aff, o_aff = self.injector.transform_weights_or_output("node-1", weights, ref_out, tick=tick)
            if not np.array_equal(ref_out, o_aff):
                corrupted_ticks += 1

        self.assertGreater(corrupted_ticks, 0)
        self.assertLess(corrupted_ticks, 10)

    def test_silent_ai_degradation_node_remains_online_and_healthy(self) -> None:
        """CRITICAL TEST: Verify Silent AI degradation causes AI output deviation while node status remains ONLINE and physical health remains HEALTHY."""
        node = self.cluster.get_node("node-1")
        self.assertIsNotNone(node)
        self.assertEqual(node.status, NodeStatus.ONLINE)

        # 1. Physical health before fault
        score_before = node.get_health_score()
        self.assertEqual(node.get_health_state(), HealthState.HEALTHY)

        # 2. Add SILENT_MODEL_DEGRADATION profile
        silent_profile = FaultProfile(
            fault_type=FaultType.SILENT_MODEL_DEGRADATION,
            target_node_id="node-1",
            start_tick=0,
            duration=10,
            intensity=0.10,
            seed=42,
        )
        self.injector.add_profile(silent_profile)

        # 3. Execute synthetic AI workload
        workload = SyntheticAIWorkload(name="silent_test_task", seed=42)
        exec_log = workload.execute_on_node(node, fault_injector=self.injector, tick=0)

        # 4. Verify results:
        # A. Node operational status is STILL ONLINE
        self.assertEqual(node.status, NodeStatus.ONLINE)

        # B. Node physical health score is STILL HEALTHY (score >= 90.0)
        self.assertEqual(exec_log["health_state"], "HEALTHY")
        self.assertGreaterEqual(exec_log["health_score"], 90.0)

        # C. AI Output Deviation IS SIGNIFICANT (MSE > 0)
        deviation = exec_log["deviation"]
        self.assertGreater(deviation["mse"], 0.0)
        self.assertGreater(deviation["mae"], 0.0)
        self.assertLess(deviation["cosine_sim"], 1.0)

    def test_degradation_over_time(self) -> None:
        """Verify output deviation grows continuously over simulation ticks under silent degradation."""
        node = self.cluster.get_node("node-2")
        silent_profile = FaultProfile(
            fault_type=FaultType.SILENT_MODEL_DEGRADATION,
            target_node_id="node-2",
            start_tick=0,
            duration=20,
            intensity=0.05,
            seed=100,
        )
        self.injector.add_profile(silent_profile)

        mses = []
        for tick in range(5):
            workload = SyntheticAIWorkload(name=f"tick_{tick}", seed=42)
            log = workload.execute_on_node(node, fault_injector=self.injector, tick=tick)
            mses.append(log["deviation"]["mse"])

        # Check monotonic increase in MSE over ticks
        for i in range(len(mses) - 1):
            self.assertGreater(mses[i + 1], mses[i])

    def test_environmental_stress_versus_silent_degradation(self) -> None:
        """Verify clear distinction between physical stress telemetry and silent AI degradation."""
        node_env = self.cluster.get_node("node-1")
        node_silent = self.cluster.get_node("node-2")

        # Environmental stress profile on node-1
        self.injector.add_profile(
            FaultProfile(
                fault_type=FaultType.ENVIRONMENTAL_STRESS,
                target_node_id="node-1",
                intensity=0.80,
            )
        )
        # Silent AI degradation profile on node-2
        self.injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-2",
                intensity=0.10,
            )
        )

        self.injector.apply_physical_telemetry_faults(self.cluster, tick=0)

        # Node 1 physical telemetry deteriorated
        self.assertGreater(node_env.temperature, 60.0)
        self.assertLess(node_env.get_health_score(), 90.0)

        # Node 2 physical telemetry remained nominal
        self.assertEqual(node_silent.get_health_state(), HealthState.HEALTHY)
        self.assertGreaterEqual(node_silent.get_health_score(), 90.0)


if __name__ == "__main__":
    unittest.main()
