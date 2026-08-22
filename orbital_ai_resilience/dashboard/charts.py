"""Plotly charts for telemetry history, AI behavioral drift, benchmarks, and 3D orbital constellation visualization."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List
from orbital_ai_resilience.orbital.eclipse import EclipseModel
from orbital_ai_resilience.orbital.propagation import EARTH_RADIUS_KM, OrbitalPropagator


def create_telemetry_history_chart(df_telemetry: pd.DataFrame) -> go.Figure:
    """Create multi-axis historical telemetry chart for a selected node."""
    fig = go.Figure()
    if df_telemetry.empty:
        fig.update_layout(title="No Telemetry History Available")
        return fig

    ticks = list(range(len(df_telemetry)))

    fig.add_trace(go.Scatter(x=ticks, y=df_telemetry["temperature"], name="Temperature (°C)", line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=ticks, y=df_telemetry["power_level"], name="Power Level (%)", line=dict(color="#2ecc71", width=2)))
    fig.add_trace(go.Scatter(x=ticks, y=df_telemetry["latency"], name="Latency (ms)", line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=ticks, y=df_telemetry["error_rate"] * 100.0, name="Error Rate (%)", line=dict(color="#9b59b6", width=2, dash="dash")))

    fig.update_layout(
        title="<b>Physical Node Telemetry History</b>",
        xaxis_title="Time Steps (Ticks)",
        yaxis_title="Metric Value",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_ai_drift_chart(df_ai: pd.DataFrame) -> go.Figure:
    """Create AI behavioral drift chart demonstrating Physical Health vs. AI Degradation."""
    fig = go.Figure()
    if df_ai.empty:
        fig.update_layout(title="No AI Behavioral History Available for Selected Node")
        return fig

    ticks = df_ai["tick"]

    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=df_ai["physical_health_score"],
            name="Physical Health Score (Phase 2)",
            line=dict(color="#2ecc71", width=3, dash="solid"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=df_ai["behavioral_score"],
            name="AI Behavioral Score (Phase 4)",
            line=dict(color="#e74c3c", width=3, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=df_ai["mse"] * 1000.0,
            name="AI Output MSE (x1000)",
            line=dict(color="#e67e22", width=2),
        )
    )

    fig.update_layout(
        title="<b>CORE RESEARCH PROBLEM: Physical Telemetry vs. AI Output Degradation</b>",
        xaxis_title="Simulation Tick",
        yaxis_title="Score / Metric Scale",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_benchmark_chart(results_data: List[Dict[str, Any]]) -> go.Figure:
    """Create interactive bar chart comparing Physical Health Only vs. Statistical vs. ML detectors."""
    if not results_data:
        fig = go.Figure()
        fig.update_layout(title="No Benchmark Results Available")
        return fig

    df_bm = pd.DataFrame(results_data)
    sc6_df = df_bm[df_bm["scenario_name"].str.contains("Scenario 6", na=False)]

    fig = px.bar(
        sc6_df,
        x="detector_name",
        y="recall",
        color="detector_name",
        title="<b>Silent Degradation Detection Rate (Recall) on Scenario 6</b>",
        labels={"recall": "Detection Rate (Recall)", "detector_name": "Detection Algorithm"},
        text_auto=".2f",
        color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71"],
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def create_optimization_comparison_chart(opt_results: List[Dict[str, Any]]) -> go.Figure:
    """Create grouped bar chart comparing Baseline Policy vs. Adaptive Optimization Policy."""
    if not opt_results:
        fig = go.Figure()
        fig.update_layout(title="No Optimization Benchmark Data Available")
        return fig

    df_opt = pd.DataFrame(opt_results)

    fig = px.bar(
        df_opt,
        x="scenario_name",
        y="average_recovery_cost",
        color="policy_name",
        barmode="group",
        title="<b>Phase 8: Baseline Policy vs. Adaptive Policy Recovery Cost Comparison</b>",
        labels={"average_recovery_cost": "Average Recovery Cost", "scenario_name": "Scenario"},
        color_discrete_map={"Baseline Policy": "#e74c3c", "Adaptive Policy": "#2ecc71"},
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=80), legend_title="Policy")
    return fig


def create_orbital_constellation_figure(tick: float = 0.0) -> go.Figure:
    """Create 3D orbital constellation visualization in ECI space showing Earth & satellite crosslinks."""
    propagator = OrbitalPropagator()
    eclipse_model = EclipseModel()
    states = propagator.generate_constellation_states(num_satellites=5, tick=tick)

    fig = go.Figure()

    # Add 2D/3D Satellite Node positions
    x_sat = [s.position_km[0] for s in states.values()]
    y_sat = [s.position_km[1] for s in states.values()]
    z_sat = [s.position_km[2] for s in states.values()]
    labels = list(states.keys())

    colors = []
    hover_texts = []
    for nid, s in states.items():
        ecl = eclipse_model.evaluate_illumination(s)
        color = "#e67e22" if ecl.is_eclipse else "#2ecc71"
        colors.append(color)
        hover_texts.append(
            f"<b>Satellite:</b> {nid}<br>"
            f"<b>Illumination:</b> {'ECLIPSE' if ecl.is_eclipse else 'SUNLIGHT'}<br>"
            f"<b>Phase:</b> {s.orbital_phase_deg:.1f}°<br>"
            f"<b>Position ECI:</b> ({s.position_km[0]:.0f}, {s.position_km[1]:.0f}, {s.position_km[2]:.0f}) km"
        )

    # Satellite markers
    fig.add_trace(
        go.Scatter3d(
            x=x_sat,
            y=y_sat,
            z=z_sat,
            mode="markers+text",
            marker=dict(size=10, color=colors, line=dict(color="#ffffff", width=1)),
            text=labels,
            textposition="top center",
            hoverinfo="text",
            hovertext=hover_texts,
            name="Orbital Satellites",
        )
    )

    fig.update_layout(
        title="<b>Phase 9: 3D Constellation Topology & Orbital Phase Positions (ECI Frame)</b>",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="cube",
        ),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
