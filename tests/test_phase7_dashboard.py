"""Unit and integration tests for Phase 7: Streamlit Dashboard State & Visual Data Transformers."""

import unittest
import pandas as pd
from orbital_ai_resilience.dashboard.state import DashboardState
from orbital_ai_resilience.faults.types import FaultType


class TestPhase7DashboardState(unittest.TestCase):
    """Test suite for Phase 7 DashboardState controller and data transformers."""

    def setUp(self) -> None:
        self.dash_state = DashboardState(seed=42)

    def test_dashboard_state_initialization(self) -> None:
        """Verify DashboardState initializes cluster, detectors, and baseline training cleanly."""
        self.assertIsNotNone(self.dash_state.cluster)
        self.assertEqual(len(self.dash_state.cluster.nodes), 5)
        self.assertTrue(self.dash_state.baseline_trained)
        self.assertEqual(self.dash_state.current_tick, 0)

    def test_cluster_summary_dict_aggregation(self) -> None:
        """Verify aggregate cluster summary dictionary calculation for metric cards."""
        summary = self.dash_state.get_cluster_summary_dict()
        self.assertEqual(summary["total_nodes"], 5)
        self.assertEqual(summary["online_nodes"], 5)
        self.assertEqual(summary["quarantined_nodes"], 0)
        self.assertEqual(summary["isolated_nodes"], 0)
        self.assertGreaterEqual(summary["avg_physical_health"], 90.0)

    def test_advance_tick_and_dataframes(self) -> None:
        """Verify advance_tick populates execution logs and dataframes."""
        self.dash_state.advance_tick()
        self.assertEqual(self.dash_state.current_tick, 1)

        # Check telemetry dataframe transformation
        df_telem = self.dash_state.get_node_telemetry_dataframe("node-1")
        self.assertIsInstance(df_telem, pd.DataFrame)
        self.assertFalse(df_telem.empty)
        self.assertIn("temperature", df_telem.columns)

        # Check AI behavior dataframe transformation
        df_ai = self.dash_state.get_ai_behavior_dataframe("node-1")
        self.assertIsInstance(df_ai, pd.DataFrame)
        self.assertFalse(df_ai.empty)
        self.assertIn("mse", df_ai.columns)

    def test_fault_injection_via_dashboard(self) -> None:
        """Verify injecting fault via dashboard state registers in FaultInjector."""
        profile = self.dash_state.inject_fault(
            target_node_id="node-3",
            fault_type=FaultType.SILENT_MODEL_DEGRADATION,
            intensity=0.15,
            duration=5,
        )
        self.assertEqual(profile.target_node_id, "node-3")
        self.assertEqual(profile.fault_type, FaultType.SILENT_MODEL_DEGRADATION)

        # Advance tick to trigger detection and recovery
        self.dash_state.advance_tick()
        self.assertGreater(len(self.dash_state.detection_results), 0)

    def test_run_cascading_failure_experiment_via_dashboard(self) -> None:
        """Verify dashboard state can trigger and execute Phase 6 cascading failure experiment."""
        self.dash_state.run_cascading_failure_experiment()

        summary = self.dash_state.get_cluster_summary_dict()
        self.assertEqual(summary["isolated_nodes"], 1)
        self.assertEqual(summary["quarantined_nodes"], 1)
        self.assertGreater(summary["successful_migrations"], 0)


if __name__ == "__main__":
    unittest.main()
