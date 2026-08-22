"""Unit and integration tests for Phase 9: Orbital Propagation, Constellation Routing, and Predictive Policy."""

import math
import unittest
from orbital_ai_resilience.constellation.bandwidth import BandwidthModel
from orbital_ai_resilience.constellation.links import InterSatelliteLink, LinkEvaluator
from orbital_ai_resilience.constellation.routing import ConstellationRoute, ConstellationRouter
from orbital_ai_resilience.constellation.visibility import VisibilityModel, VisibilityResult
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.orbital_benchmark import OrbitalOptimizationBenchmark
from orbital_ai_resilience.optimization.orbital_policy import OrbitalAwareRecoveryPolicy, OrbitalRecoveryCostModel
from orbital_ai_resilience.orbital.eclipse import EclipseModel, EclipseStatus
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.prediction import FutureOrbitalPrediction, OrbitalPredictionModel
from orbital_ai_resilience.orbital.propagation import EARTH_RADIUS_KM, OrbitalPropagator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class TestPhase9OrbitalAndConstellation(unittest.TestCase):
    """Test suite for Phase 9 orbital mechanics, illumination, constellation links, and predictive policies."""

    def setUp(self) -> None:
        self.propagator = OrbitalPropagator(altitude_km=550.0, inclination_deg=53.0)
        self.eclipse_model = EclipseModel()
        self.prediction_model = OrbitalPredictionModel(propagator=self.propagator, eclipse_model=self.eclipse_model)
        self.visibility_model = VisibilityModel()
        self.link_evaluator = LinkEvaluator(visibility_model=self.visibility_model)
        self.bandwidth_model = BandwidthModel()
        self.router = ConstellationRouter(link_evaluator=self.link_evaluator, eclipse_model=self.eclipse_model)

        self.cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        self.history = RecoveryHistory()
        self.quarantine_manager = QuarantineManager()
        self.workload = SyntheticAIWorkload(name="orb_test_task", seed=42)
        self.orbital_policy = OrbitalAwareRecoveryPolicy(
            propagator=self.propagator,
            eclipse_model=self.eclipse_model,
            prediction_model=self.prediction_model,
            router=self.router,
            bandwidth_model=self.bandwidth_model,
        )

    def test_orbital_propagation(self) -> None:
        """Verify OrbitalPropagator computes valid 3D Cartesian position vectors."""
        state = self.propagator.compute_state("node-1", initial_phase_deg=0.0, tick=0.0)
        self.assertEqual(state.satellite_id, "node-1")
        self.assertAlmostEqual(state.radius_km, EARTH_RADIUS_KM + 550.0, delta=1.0)
        self.assertEqual(len(state.position_km), 3)
        self.assertEqual(len(state.velocity_km_s), 3)

    def test_orbital_reproducibility(self) -> None:
        """Verify orbital state propagation is 100% deterministic given identical tick."""
        s1 = self.propagator.compute_state("node-1", initial_phase_deg=45.0, tick=10.0)
        s2 = self.propagator.compute_state("node-1", initial_phase_deg=45.0, tick=10.0)
        self.assertEqual(s1.position_km, s2.position_km)
        self.assertEqual(s1.velocity_km_s, s2.velocity_km_s)

    def test_eclipse_model(self) -> None:
        """Verify EclipseModel detects sunlight vs. Earth shadow eclipse geometry."""
        # Sunlight position (x > 0)
        state_sun = OrbitalState("node-sun", 550.0, 53.0, 0.0, 0.001, (6921.0, 0.0, 0.0), (0.0, 7.6, 0.0), 0.0)
        status_sun = self.eclipse_model.evaluate_illumination(state_sun)
        self.assertFalse(status_sun.is_eclipse)
        self.assertEqual(status_sun.sunlight_fraction, 1.0)

        # Eclipse position (x < 0 behind Earth)
        state_ecl = OrbitalState("node-ecl", 550.0, 53.0, 180.0, 0.001, (-6921.0, 0.0, 0.0), (0.0, 7.6, 0.0), 0.0)
        status_ecl = self.eclipse_model.evaluate_illumination(state_ecl)
        self.assertTrue(status_ecl.is_eclipse)
        self.assertEqual(status_ecl.eclipse_fraction, 1.0)

    def test_future_eclipse_prediction(self) -> None:
        """Verify OrbitalPredictionModel forecasts future eclipse risks over N ticks."""
        pred = self.prediction_model.predict_future_state(
            satellite_id="node-1",
            initial_phase_deg=105.0,
            current_tick=0.0,
            current_battery_level=30.0,
            lookahead_ticks=6,
        )
        self.assertGreater(pred.future_eclipse_risk, 0.0)
        self.assertTrue(pred.is_eclipse_imminent)

    def test_line_of_sight_obstruction(self) -> None:
        """Verify VisibilityModel detects Earth limb obstruction between opposite satellites."""
        state1 = OrbitalState("n1", 550.0, 0.0, 0.0, 0.001, (6921.0, 0.0, 0.0), (0, 7.6, 0), 0.0)
        state2 = OrbitalState("n2", 550.0, 0.0, 180.0, 0.001, (-6921.0, 0.0, 0.0), (0, 7.6, 0), 0.0)
        vis = self.visibility_model.evaluate_visibility(state1, state2)
        self.assertFalse(vis.is_visible)
        self.assertLess(vis.min_ray_height_km, 0.0)

    def test_inter_satellite_link(self) -> None:
        """Verify InterSatelliteLink calculates distance-derived propagation latency."""
        state1 = OrbitalState("n1", 550.0, 0.0, 0.0, 0.001, (6921.0, 0.0, 0.0), (0, 7.6, 0), 0.0)
        state2 = OrbitalState("n2", 550.0, 0.0, 15.0, 0.001, (6685.0, 1791.0, 0.0), (0, 7.6, 0), 0.0)
        link = self.link_evaluator.evaluate_link(state1, state2)
        self.assertTrue(link.available)
        self.assertGreater(link.distance_km, 0.0)
        self.assertGreater(link.latency_ms, 2.0)

    def test_bandwidth_model_estimate(self) -> None:
        """Verify BandwidthModel estimates transfer time and sufficiency."""
        est = self.bandwidth_model.estimate_transfer(workload_memory_mb=1024.0, bottleneck_bandwidth_mbps=1000.0)
        self.assertAlmostEqual(est.transfer_time_sec, 8.192, delta=0.1)
        self.assertTrue(est.is_bandwidth_sufficient)

    def test_multihop_dijkstra_routing(self) -> None:
        """Verify ConstellationRouter finds multi-hop Dijkstra route when direct link is obstructed."""
        # Satellites positioned 30 deg apart (node-3 -> node-2 -> node-4)
        states = {
            "node-3": self.propagator.compute_state("node-3", 0.0, 0.0),
            "node-2": self.propagator.compute_state("node-2", 30.0, 0.0),
            "node-4": self.propagator.compute_state("node-4", 60.0, 0.0),
        }
        route = self.router.find_route("node-3", "node-4", constellation_states=states)
        self.assertTrue(route.is_route_valid)
        self.assertGreaterEqual(route.hop_count, 1)

    def test_orbital_aware_policy_selection(self) -> None:
        """Verify OrbitalAwareRecoveryPolicy selects target satellite and multi-hop route."""
        states = {
            "node-3": self.propagator.compute_state("node-3", 0.0, 0.0),
            "node-1": self.propagator.compute_state("node-1", 30.0, 0.0),
            "node-2": self.propagator.compute_state("node-2", 60.0, 0.0),
            "node-4": self.propagator.compute_state("node-4", 90.0, 0.0),
            "node-5": self.propagator.compute_state("node-5", 120.0, 0.0),
        }
        target_id, route, score, exp = self.orbital_policy.select_target_and_route(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            constellation_states=states,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertIsNotNone(target_id)
        self.assertIsNotNone(route)
        self.assertTrue(route.is_route_valid)

    def test_future_risk_target_avoidance(self) -> None:
        """Verify OrbitalAwareRecoveryPolicy avoids targets with imminent future eclipse risk."""
        states = {
            "node-3": self.propagator.compute_state("node-3", 0.0, 0.0),
            "node-1": self.propagator.compute_state("node-1", 105.0, 0.0),
            "node-2": self.propagator.compute_state("node-2", 30.0, 0.0),
            "node-4": self.propagator.compute_state("node-4", 60.0, 0.0),
            "node-5": self.propagator.compute_state("node-5", 120.0, 0.0),
        }
        self.cluster.get_node("node-1").update_telemetry(power_level=10.0)

        target_id, route, score, exp = self.orbital_policy.select_target_and_route(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            constellation_states=states,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertNotEqual(target_id, "node-1")

    def test_unsafe_node_rejection_phase9(self) -> None:
        """Verify OrbitalAwareRecoveryPolicy rejects QUARANTINED or ISOLATED nodes."""
        states = {
            "node-3": self.propagator.compute_state("node-3", 0.0, 0.0),
            "node-1": self.propagator.compute_state("node-1", 30.0, 0.0),
            "node-2": self.propagator.compute_state("node-2", 60.0, 0.0),
        }
        n1 = self.cluster.get_node("node-1")
        self.quarantine_manager.quarantine_node(n1, reason="Test Quarantine")

        target_id, route, score, exp = self.orbital_policy.select_target_and_route(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            constellation_states=states,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertNotEqual(target_id, "node-1")

    def test_bandwidth_rejection(self) -> None:
        """Verify candidate target is rejected if available bandwidth is insufficient."""
        strict_policy = OrbitalAwareRecoveryPolicy(
            bandwidth_model=BandwidthModel(min_required_bandwidth_mbps=5000.0)
        )
        target_id, route, score, exp = strict_policy.select_target_and_route(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertIsNone(target_id)

    def test_policy_reproducibility(self) -> None:
        """Verify OrbitalAwareRecoveryPolicy is 100% deterministic given identical cluster state."""
        states = {
            "node-3": self.propagator.compute_state("node-3", 0.0, 0.0),
            "node-1": self.propagator.compute_state("node-1", 30.0, 0.0),
            "node-2": self.propagator.compute_state("node-2", 60.0, 0.0),
        }
        r1 = self.orbital_policy.select_target_and_route(self.cluster, "node-3", self.workload, self.history, constellation_states=states, quarantine_manager=self.quarantine_manager)
        r2 = self.orbital_policy.select_target_and_route(self.cluster, "node-3", self.workload, self.history, constellation_states=states, quarantine_manager=self.quarantine_manager)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[2], r2[2])

    def test_orbital_benchmark_suite(self) -> None:
        """Verify OrbitalOptimizationBenchmark executes 10-scenario comparison suite."""
        bm = OrbitalOptimizationBenchmark(seed=42)
        results, improvements = bm.run_full_orbital_benchmark_suite()
        self.assertEqual(len(results), 20)
        self.assertIn("total_cost_reduction_pct", improvements)
        self.assertIn("communication_cost_reduction_pct", improvements)
        self.assertIn("recovery_success_increase_pct", improvements)


if __name__ == "__main__":
    unittest.main()
