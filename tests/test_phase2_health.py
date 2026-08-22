"""Unit tests for Phase 2: Telemetry History and Health/Trust Scoring."""

import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import HealthState, NodeStatus
from orbital_ai_resilience.health.config import HealthConfig
from orbital_ai_resilience.health.evaluator import HealthEvaluator
from orbital_ai_resilience.telemetry.snapshot import TelemetryHistory, TelemetrySnapshot


class TestTelemetryHistory(unittest.TestCase):
    """Test suite for telemetry history recording and rolling window bounds."""

    def test_history_recording_and_max_len(self) -> None:
        """Verify rolling window respects max_length constraint."""
        history = TelemetryHistory(max_length=3)
        self.assertEqual(len(history), 0)

        for i in range(5):
            history.add_snapshot(
                TelemetrySnapshot(
                    timestamp=float(i),
                    power_level=100.0,
                    temperature=40.0 + i,
                    latency=10.0,
                    error_rate=0.0,
                )
            )

        self.assertEqual(len(history), 3)
        latest = history.get_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.timestamp, 4.0)
        self.assertEqual(latest.temperature, 44.0)

        # Check metric series extraction
        temp_series = history.get_series("temperature")
        self.assertEqual(temp_series, [42.0, 43.0, 44.0])

    def test_recent_snapshots(self) -> None:
        """Verify get_recent retrieves correct snapshot slice."""
        history = TelemetryHistory(max_length=10)
        for i in range(5):
            history.add_snapshot(
                TelemetrySnapshot(timestamp=float(i), power_level=100.0, temperature=40.0, latency=10.0, error_rate=0.0)
            )

        recent = history.get_recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual([s.timestamp for s in recent], [2.0, 3.0, 4.0])


class TestSimulatedTimeSteps(unittest.TestCase):
    """Test suite for tick/time-step simulation mechanisms."""

    def test_node_tick_records_snapshot(self) -> None:
        """Verify node.tick() appends timestamped snapshot to history."""
        node = VirtualNode(node_id="node-tick", max_history_len=10)
        self.assertEqual(len(node.telemetry_history), 1)  # 1 initial tick at creation

        node.tick(timestamp=100.0)
        self.assertEqual(len(node.telemetry_history), 2)
        latest = node.telemetry_history.get_latest()
        self.assertEqual(latest.timestamp, 100.0)

    def test_cluster_step_all(self) -> None:
        """Verify cluster.step_all() advances time steps across all nodes."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=3)
        snapshots = cluster.step_all(timestamp=200.0)

        self.assertEqual(len(snapshots), 3)
        for node_id, snapshot in snapshots.items():
            self.assertEqual(snapshot.timestamp, 200.0)
            node = cluster.get_node(node_id)
            self.assertEqual(node.telemetry_history.get_latest().timestamp, 200.0)


class TestHealthScoring(unittest.TestCase):
    """Test suite for deterministic Health/Trust Score calculation."""

    def setUp(self) -> None:
        self.config = HealthConfig()
        self.evaluator = HealthEvaluator(config=self.config)

    def test_healthy_node_score(self) -> None:
        """Verify perfect score (100.0) for nominal healthy telemetry."""
        snapshot = TelemetrySnapshot(
            timestamp=1.0,
            power_level=100.0,
            temperature=45.0,
            latency=10.0,
            error_rate=0.0,
        )
        score, state, breakdown = self.evaluator.evaluate_health(snapshot)
        self.assertEqual(score, 100.0)
        self.assertEqual(state, HealthState.HEALTHY)
        self.assertEqual(breakdown["sub_scores"]["power"], 100.0)
        self.assertEqual(breakdown["sub_scores"]["temp"], 100.0)
        self.assertEqual(breakdown["sub_scores"]["latency"], 100.0)
        self.assertEqual(breakdown["sub_scores"]["error"], 100.0)

    def test_critical_boundary_score(self) -> None:
        """Verify score drops to 0.0 under extreme critical telemetry."""
        snapshot = TelemetrySnapshot(
            timestamp=1.0,
            power_level=10.0,    # below 20.0 -> 0
            temperature=100.0,   # above 95.0 -> 0
            latency=150.0,       # above 100.0 -> 0
            error_rate=0.20,     # above 0.10 -> 0
        )
        score, state, breakdown = self.evaluator.evaluate_health(snapshot)
        self.assertEqual(score, 0.0)
        self.assertEqual(state, HealthState.CRITICAL)

    def test_health_state_transitions(self) -> None:
        """Verify state transitions across HEALTHY -> WARNING -> DEGRADED -> CRITICAL."""
        node = VirtualNode(node_id="node-trans")
        self.assertEqual(node.get_health_state(), HealthState.HEALTHY)

        # 1. Temperature rise to 70°C -> score ~ 89.29 (WARNING)
        node.update_telemetry(temperature=70.0)
        self.assertEqual(node.get_health_state(), HealthState.WARNING)

        # 2. Temperature rise to 80°C + error_rate 0.03 -> score ~ 69.29 (DEGRADED)
        node.update_telemetry(temperature=80.0, error_rate=0.03)
        self.assertEqual(node.get_health_state(), HealthState.DEGRADED)

        # 3. High error rate 0.08 + temp 85°C -> score ~ 30.71 (CRITICAL)
        node.update_telemetry(temperature=85.0, error_rate=0.08)
        self.assertEqual(node.get_health_state(), HealthState.CRITICAL)

    def test_worsening_trend_penalty(self) -> None:
        """Verify a node with a sustained rising temp receives trend penalty compared to steady temp."""
        # Node A: Constant elevated temperature (75°C) across 5 steps
        node_a = VirtualNode(node_id="node-steady", max_history_len=10)
        for i in range(5):
            node_a.update_telemetry(temperature=75.0)

        # Node B: Continuously rising temperature (45°C -> 75°C) over 5 steps
        node_b = VirtualNode(node_id="node-rising", max_history_len=10)
        temps = [45.0, 52.0, 60.0, 68.0, 75.0]
        for t in temps:
            node_b.update_telemetry(temperature=t)

        score_a = node_a.get_health_score()
        score_b = node_b.get_health_score()

        # Node B has same final temp (75°C), but rising trend causes trend penalty, so score_b < score_a
        self.assertLess(score_b, score_a)
        breakdown_b = node_b.get_health_breakdown()
        self.assertGreater(breakdown_b["trend_penalty"], 0.0)

    def test_repeatable_deterministic_scoring(self) -> None:
        """Verify identical telemetry input always produces identical score outputs."""
        snapshot = TelemetrySnapshot(
            timestamp=5.0,
            power_level=85.0,
            temperature=68.0,
            latency=25.0,
            error_rate=0.02,
        )
        score1, state1, _ = self.evaluator.evaluate_health(snapshot)
        score2, state2, _ = self.evaluator.evaluate_health(snapshot)

        self.assertEqual(score1, score2)
        self.assertEqual(state1, state2)


if __name__ == "__main__":
    unittest.main()
