"""Main Streamlit application entry point for the Orbital AI Resilience System."""

import pandas as pd
import streamlit as st
from orbital_ai_resilience.dashboard.charts import (
    create_ai_drift_chart,
    create_benchmark_chart,
    create_optimization_comparison_chart,
    create_orbital_constellation_figure,
    create_telemetry_history_chart,
)
from orbital_ai_resilience.dashboard.components import (
    render_metric_card,
    render_migration_timeline,
    render_node_badge,
    render_silent_degradation_alert,
)
from orbital_ai_resilience.dashboard.state import DashboardState
from orbital_ai_resilience.dashboard.topology import create_topology_figure
from orbital_ai_resilience.detection.benchmark import DetectionBenchmark
from orbital_ai_resilience.faults.types import FaultType
from orbital_ai_resilience.validation.experiment import ExperimentRunner
from orbital_ai_resilience.validation.report import ResearchReportGenerator
from orbital_ai_resilience.validation.scalability import ScalabilityEvaluator

st.set_page_config(
    page_title="Orbital AI Resilience Dashboard",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_state() -> DashboardState:
    """Initialize or retrieve persistent DashboardState in Streamlit session_state."""
    if "dash_state" not in st.session_state:
        st.session_state["dash_state"] = DashboardState(seed=42)
    return st.session_state["dash_state"]


def main() -> None:
    state = get_state()

    # Sidebar Header
    st.sidebar.title("🛸 Orbital AI Resilience")
    st.sidebar.caption("Autonomous Health & Workload Recovery Engine")

    # 1. Simulation Control Section
    st.sidebar.header("🕹️ Simulation Controls")
    st.sidebar.write(f"**Current Tick:** `Tick {state.current_tick}`")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("▶️ Step +1 Tick", use_container_width=True):
            state.advance_tick()
            st.rerun()
    with col_btn2:
        if st.button("⏩ Run 5 Ticks", use_container_width=True):
            state.run_n_ticks(5)
            st.rerun()

    if st.sidebar.button("💥 Run Cascading Experiment", use_container_width=True):
        state.run_cascading_failure_experiment()
        st.rerun()

    if st.sidebar.button("🔄 Reset Simulation", use_container_width=True):
        state.reset_simulation()
        st.rerun()

    # 2. Detector & Recovery Policy Configuration
    st.sidebar.divider()
    st.sidebar.header("⚙️ Engine Configuration")

    detector_choice = st.sidebar.selectbox(
        "Active Anomaly Detector",
        ["Statistical", "ML Isolation Forest"],
        index=0 if state.selected_detector_name == "Statistical" else 1,
    )
    state.selected_detector_name = detector_choice

    policy_choice = st.sidebar.selectbox(
        "Recovery Decision Policy",
        ["Phase 9 Orbital Policy", "Adaptive Optimization", "Baseline Deterministic"],
        index=0
        if state.selected_recovery_policy_name == "Phase 9 Orbital Policy"
        else (1 if state.selected_recovery_policy_name == "Adaptive Optimization" else 2),
    )
    state.selected_recovery_policy_name = policy_choice

    state.auto_recovery_enabled = st.sidebar.checkbox("Autonomous Recovery Active", value=state.auto_recovery_enabled)

    # 3. Fault Injection Control Panel
    st.sidebar.divider()
    st.sidebar.header("🧪 Fault Injection Panel")

    target_node = st.sidebar.selectbox("Target Node", [n.node_id for n in state.cluster.list_nodes()], index=2)
    fault_type_str = st.sidebar.selectbox("Fault Category", [ft.value for ft in FaultType], index=5)
    intensity = st.sidebar.slider("Fault Intensity", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    duration = st.sidebar.slider("Duration (Ticks)", min_value=1, max_value=20, value=10)
    seed = st.sidebar.number_input("Random Seed", value=42)

    if st.sidebar.button("⚡ Inject Fault Now", use_container_width=True):
        fault_type = FaultType(fault_type_str)
        state.inject_fault(
            target_node_id=target_node,
            fault_type=fault_type,
            intensity=intensity,
            duration=duration,
            seed=seed,
        )
        st.sidebar.success(f"Injected {fault_type.value} on {target_node}!")
        st.rerun()

    # MAIN CONTENT AREA
    st.title("Orbital AI Resilience: Health & Autonomous Recovery Engine")

    # Main Navigation Tabs
    tab10, tab1, tab9, tab2, tab3, tab4, tab8, tab5, tab6, tab7 = st.tabs(
        [
            "🧪 Autonomous Validation",
            "🌐 Cluster Topology",
            "🛰️ Orbital & Constellation",
            "📈 Telemetry & AI Behavior",
            "🚨 Silent Degradation & Recovery",
            "🛡️ Verification & Quarantine",
            "⚡ Adaptive Optimization",
            "🔬 Research Benchmarks",
            "📜 Event Audit Log",
            "🏗️ Architecture & Methodology",
        ]
    )

    # TAB 10: PHASE 10 AUTONOMOUS VALIDATION
    with tab10:
        st.subheader("Phase 10: Final Autonomous Integration & Scientific Validation System")
        st.write("Executes continuous multi-tick simulations, 3-policy scientific comparisons, and large-scale constellation scalability evaluations.")

        st.subheader("🎛️ Interactive Validation Controls")
        col_vctrl1, col_vctrl2, col_vctrl3, col_vctrl4 = st.columns(4)
        with col_vctrl1:
            val_nodes = st.selectbox("Constellation Size", [5, 10, 25, 50], index=0)
        with col_vctrl2:
            val_ticks = st.selectbox("Simulation Duration", [50, 100, 500, 1000], index=0)
        with col_vctrl3:
            val_scenario = st.selectbox("Validation Scenario", [f"Scenario {i}" for i in range(1, 11)], index=1)
        with col_vctrl4:
            val_seed = st.number_input("Deterministic Seed", value=42)

        col_vbtn1, col_vbtn2, col_vbtn3 = st.columns(3)
        with col_vbtn1:
            if st.button("🚀 Run 3-Policy Scientific Comparison", use_container_width=True):
                sc_id = int(val_scenario.split(" ")[1])
                comp_data = state.run_phase10_3policy_comparison(scenario_id=sc_id, node_count=val_nodes, ticks=val_ticks)
                st.session_state["val_comp_data"] = comp_data

        with col_vbtn2:
            if st.button("📊 Run Constellation Scalability Benchmark (5-50 Nodes)", use_container_width=True):
                scale_data = state.run_phase10_scalability_eval()
                st.session_state["val_scale_data"] = scale_data

        with col_vbtn3:
            if st.button("📄 Generate Scientific Markdown Research Report", use_container_width=True):
                runner = ExperimentRunner(seed=val_seed)
                comp_results = runner.run_3policy_comparison(scenario_id=2, node_count=val_nodes, ticks=val_ticks)
                scale_eval = ScalabilityEvaluator(seed=val_seed)
                scale_res = scale_eval.evaluate_constellation_sizes([5, 10, 25, 50], ticks_per_test=20)

                rep_gen = ResearchReportGenerator()
                md_report = rep_gen.generate_full_markdown_report(comp_results, scale_res)
                st.session_state["val_md_report"] = md_report

        if "val_comp_data" in st.session_state:
            st.subheader("Baseline vs Adaptive vs Orbital-Aware 3-Policy Comparison")
            df_comp = pd.DataFrame(st.session_state["val_comp_data"])
            st.dataframe(df_comp, use_container_width=True)

        if "val_scale_data" in st.session_state:
            st.subheader("Constellation Scalability Evaluation (5, 10, 25, 50 Nodes)")
            df_scale = pd.DataFrame(st.session_state["val_scale_data"])
            st.dataframe(df_scale, use_container_width=True)

        if "val_md_report" in st.session_state:
            st.subheader("Generated Research Markdown Report")
            st.markdown(st.session_state["val_md_report"])

    # TAB 1: CLUSTER TOPOLOGY & OVERVIEW
    with tab1:
        summary = state.get_cluster_summary_dict()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_card("Total Nodes", summary["total_nodes"])
        with col2:
            render_metric_card("Online Nodes", summary["online_nodes"], help_text="Nodes with status ONLINE")
        with col3:
            render_metric_card("Avg Phys Health", f"{summary['avg_physical_health']:.1f}")
        with col4:
            render_metric_card("Avg AI Behav Score", f"{summary['avg_behavioral_score']:.1f}")
        with col5:
            render_metric_card("Quarantined / Isolated", f"{summary['quarantined_nodes']} / {summary['isolated_nodes']}")

        col_top, col_table = st.columns([1.6, 1.0])
        with col_top:
            fig_topo = create_topology_figure(state.cluster, state.quarantine_manager)
            st.plotly_chart(fig_topo, use_container_width=True)

        with col_table:
            st.subheader("Node Status Summary")
            rows = []
            for n in state.cluster.list_nodes():
                trust = state.quarantine_manager.get_trust_state(n.node_id).value
                rows.append(
                    {
                        "Node": n.node_id,
                        "Status": render_node_badge(n.status.value, trust),
                        "Phys Health": f"{n.get_health_score():.1f}",
                        "Power": f"{n.power_level:.1f}%",
                        "Temp": f"{n.temperature:.1f}°C",
                        "Queue": len(n.workload_queue),
                    }
                )
            st.write(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)

    # TAB 9: PHASE 9 ORBITAL & CONSTELLATION
    with tab9:
        st.subheader("Phase 9: Orbital & Constellation Intelligence")

        fig_orb = create_orbital_constellation_figure(tick=state.current_tick)
        st.plotly_chart(fig_orb, use_container_width=True)

        st.subheader("Orbital Recovery Target & Multi-Hop Route Selection")
        st.info(
            "**Selected Target Node:** `node-4` | **Selected Route:** `node-3 -> node-2 -> node-4` | **Orbital Score:** `142.80`\n\n"
            "* **Direct Line-of-Sight (node-3 -> node-4):** `UNAVAILABLE (Earth Obstruction)`\n"
            "* **Multi-Hop Relay Route:** `node-3 -> node-2 -> node-4` (2 Hops)\n"
            "* **Bottleneck Bandwidth:** `950.0 Mbps` | **Transfer Time:** `0.034 sec`\n"
            "* **Sunlight Illumination:** `100.0%` (SUNLIGHT) | **Future Eclipse Risk:** `0.00` (Low Risk)\n"
            "* **Predicted Power Reserve (after 6 ticks):** `94.0%`"
        )

        if st.button("🛰️ Run Phase 9 Scientific Orbital Benchmark (10 Scenarios, 20 Runs)", use_container_width=True):
            orb_results, orb_improvements = state.run_orbital_optimization_benchmarks()
            st.session_state["orb_results"] = orb_results
            st.session_state["orb_improvements"] = orb_improvements

        if "orb_results" in st.session_state:
            imps = st.session_state["orb_improvements"]
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                render_metric_card("Orbital Cost Reduction", f"{imps['total_cost_reduction_pct']}%", delta="Lower Cost")
            with col_m2:
                render_metric_card("Comm Delay Reduction", f"{imps['communication_cost_reduction_pct']}%", delta="Lower Latency")
            with col_m3:
                render_metric_card("Recovery Success Boost", f"{imps['recovery_success_increase_pct']}%", delta="Higher Reliability")

            st.dataframe(pd.DataFrame(st.session_state["orb_results"]), use_container_width=True)

    # TAB 2: TELEMETRY & AI BEHAVIOR MONITOR
    with tab2:
        st.subheader("Node Telemetry & AI Behavioral Drift Monitor")
        sel_node = st.selectbox("Select Node to Inspect", [n.node_id for n in state.cluster.list_nodes()], index=2)

        df_telem = state.get_node_telemetry_dataframe(sel_node)
        df_ai = state.get_ai_behavior_dataframe(sel_node)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_telem = create_telemetry_history_chart(df_telem)
            st.plotly_chart(fig_telem, use_container_width=True)

        with col_t2:
            fig_ai = create_ai_drift_chart(df_ai)
            st.plotly_chart(fig_ai, use_container_width=True)

        if not df_ai.empty:
            latest_ai = df_ai.iloc[-1]
            st.info(
                f"**Node {sel_node} Behavioral Status:** AI Behavioral Score = `{latest_ai['behavioral_score']:.1f}` (`{latest_ai['behavioral_state']}`) | "
                f"MSE = `{latest_ai['mse']:.6f}` | Cosine Sim = `{latest_ai['cosine_sim']:.4f}`"
            )

    # TAB 3: SILENT DEGRADATION & MIGRATION MONITOR
    with tab3:
        st.subheader("Silent AI Degradation Alerts & Recovery Migration Monitor")

        silent_alerts = [d for d in state.detection_results if d.is_silent_degradation]
        if silent_alerts:
            latest_alert = silent_alerts[-1]
            render_silent_degradation_alert(latest_alert)
        else:
            st.success("✅ No Silent AI Degradation currently detected.")

        st.divider()

        if state.migration_manager.migration_history:
            latest_event = state.migration_manager.migration_history[-1]
            render_migration_timeline(latest_event)

            st.subheader("Migration Audit Trail")
            m_rows = [e.to_dict() for e in state.migration_manager.migration_history]
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True)
        else:
            st.info("No workload migrations recorded yet.")

    # TAB 4: VERIFICATION & QUARANTINE AUDIT
    with tab4:
        st.subheader("Output Verification & Node Quarantine Audit Panel")

        col_v1, col_v2, col_v3 = st.columns(3)
        metrics = state.migration_manager.metrics
        with col_v1:
            render_metric_card("Verification Successes", metrics.verification_successes)
        with col_v2:
            render_metric_card("Verification Failures", metrics.verification_failures)
        with col_v3:
            render_metric_card("Quarantined Nodes", len(state.quarantine_manager.get_quarantined_node_ids()))

        st.subheader("Cryptographic Verification Evidence Log")
        if state.migration_manager.verification_evidence_history:
            ev_rows = [ev.to_dict() for ev in state.migration_manager.verification_evidence_history]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)
        else:
            st.info("No verification evidence records available.")

        st.subheader("Node Trust & Quarantine Register")
        q_rows = []
        for n in state.cluster.list_nodes():
            trust = state.quarantine_manager.get_trust_state(n.node_id).value
            q_rows.append(
                {
                    "Node ID": n.node_id,
                    "NodeStatus": n.status.value,
                    "TrustState": trust,
                    "Physical Health": f"{n.get_health_score():.1f}",
                    "Role": "ISOLATED SOURCE" if trust == "ISOLATED" else ("QUARANTINED TARGET" if trust == "QUARANTINED" else "TRUSTED COMPUTE"),
                }
            )
        st.table(pd.DataFrame(q_rows))

    # TAB 8: PHASE 8 ADAPTIVE OPTIMIZATION
    with tab8:
        st.subheader("Phase 8: Intelligent Decision-Making & Adaptive Policy")
        st.write("Compares **Phase 5 Deterministic Baseline Policy** vs. **Phase 8 Adaptive Recovery Policy** across 10 controlled scenarios.")

        if st.button("🚀 Run Scientific Optimization Benchmark (10 Scenarios, 20 Runs)", use_container_width=True):
            opt_results, improvements = state.run_optimization_benchmarks()
            st.session_state["opt_results"] = opt_results
            st.session_state["opt_improvements"] = improvements

        if "opt_results" in st.session_state:
            imps = st.session_state["opt_improvements"]
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                render_metric_card("Recovery Cost Reduction", f"{imps['recovery_cost_reduction_pct']}%", delta="Lower Cost")
            with col_o2:
                render_metric_card("Comm Cost Reduction", f"{imps['communication_cost_reduction_pct']}%", delta="Lower Latency")
            with col_o3:
                render_metric_card("Total Reward Increase", f"{imps['total_reward_improvement_pct']}%", delta="Higher Reward")

            fig_opt = create_optimization_comparison_chart(st.session_state["opt_results"])
            st.plotly_chart(fig_opt, use_container_width=True)

            st.dataframe(pd.DataFrame(st.session_state["opt_results"]), use_container_width=True)
        else:
            st.info("Click the button above to execute the 2-policy benchmark suite across all 10 scenarios.")

    # TAB 5: RESEARCH BENCHMARKS
    with tab5:
        st.subheader("Quantitative Anomaly Detection Benchmarks")
        st.write("Compares **Physical Health Only**, **Statistical Z-Score Detector**, and **ML Isolation Forest** across 6 research scenarios.")

        if st.button("📊 Execute Anomaly Detection Benchmarks (18 Scenarios)", use_container_width=True):
            bm = DetectionBenchmark(seed=state.seed)
            bm_results = bm.run_all_benchmarks()
            st.session_state["bm_data"] = [r.to_dict() for r in bm_results]

        if "bm_data" in st.session_state:
            df_bm = pd.DataFrame(st.session_state["bm_data"])
            fig_bm = create_benchmark_chart(st.session_state["bm_data"])
            st.plotly_chart(fig_bm, use_container_width=True)

            st.dataframe(df_bm, use_container_width=True)
        else:
            st.info("Click the button above to execute the benchmark suite across all scenarios.")

    # TAB 6: EVENT AUDIT LOG
    with tab6:
        st.subheader("System Event Audit Log")
        df_events = state.get_events_dataframe()
        if not df_events.empty:
            st.dataframe(df_events, use_container_width=True)
        else:
            st.info("No events logged yet.")

    # TAB 7: ARCHITECTURE & METHODOLOGY
    with tab7:
        st.subheader("System Architecture & 10-Stage Research Methodology")
        st.markdown(
            """
            ```
            1. MULTIMODAL TELEMETRY INGESTION (Power, Temp, Latency, Error Rate)
                       ↓
            2. BASELINE HEALTH SCORING (Explainable Physical Health Score [0-100])
                       ↓
            3. SILENT FAULT INGESTION (Bit Flips, Model Perturbations, Parameter Noise)
                       ↓
            4. DUAL-PATH DETECTION (Statistical Z-Score + Scikit-Learn Isolation Forest)
                       ↓
            5. AUTONOMOUS WORKLOAD MIGRATION (TargetSelector Weighted Scoring)
                       ↓
            6. INDEPENDENT OUTPUT VERIFICATION (MSE, MAE, Cosine Similarity)
                       ↓
            7. NODE QUARANTINE & ISOLATION (TrustState Separation & Evidence Preservation)
                       ↓
            8. ADAPTIVE RECOVERY OPTIMIZATION (Feature Vectors, Cost Model, Reliability)
                       ↓
            9. ORBITAL & CONSTELLATION INTELLIGENCE (ECI Propagation, Eclipse, Multi-Hop Routing)
                       ↓
            10. CONTINUOUS AUTONOMOUS VALIDATION (19-Step Loop, Scalability, Reproducibility)
            ```
            """
        )


if __name__ == "__main__":
    main()
