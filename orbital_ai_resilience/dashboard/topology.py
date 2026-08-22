"""Interactive Cluster Topology visualization using Plotly."""

import plotly.graph_objects as go
from typing import Any, Dict, List
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.quarantine.state import TrustState


def create_topology_figure(cluster: VirtualCluster, quarantine_manager: QuarantineManager) -> go.Figure:
    """Create interactive Plotly 2D node topology chart with clear state indicators."""
    nodes = cluster.list_nodes()
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title="No Cluster Nodes Found")
        return fig

    # 2D Grid coordinates for 5 nodes
    coords = {
        "node-1": (0, 1),
        "node-2": (1, 1),
        "node-3": (0.5, 0.5),  # Center source node
        "node-4": (0, 0),
        "node-5": (1, 0),
    }

    x_vals = []
    y_vals = []
    colors = []
    labels = []
    hover_texts = []

    for i, node in enumerate(nodes):
        nid = node.node_id
        x, y = coords.get(nid, (i % 3, i // 3))
        x_vals.append(x)
        y_vals.append(y)

        trust = quarantine_manager.get_trust_state(nid)
        status = node.status.value

        # Color mapping logic
        if status == "ISOLATED" or trust == TrustState.ISOLATED.value:
            color = "#e74c3c"  # Red
            state_label = "ISOLATED"
        elif trust == TrustState.QUARANTINED.value:
            color = "#e67e22"  # Orange
            state_label = "ONLINE + QUARANTINED"
        elif status == "DEGRADED":
            color = "#d35400"  # Amber
            state_label = "DEGRADED"
        elif trust == TrustState.SUSPECTED.value:
            color = "#f1c40f"  # Yellow
            state_label = "ONLINE + SUSPECTED"
        elif status == "ONLINE" and trust == TrustState.TRUSTED.value:
            color = "#2ecc71"  # Green
            state_label = "ONLINE + TRUSTED"
        else:
            color = "#7f8c8d"  # Grey
            state_label = f"{status} + {trust}"

        colors.append(color)
        labels.append(f"{nid}<br><b>{state_label}</b>")

        phys_score = node.get_health_score()
        hover_texts.append(
            f"<b>Node:</b> {nid}<br>"
            f"<b>Status:</b> {status}<br>"
            f"<b>Trust State:</b> {trust}<br>"
            f"<b>Physical Health:</b> {phys_score:.1f}<br>"
            f"<b>Power:</b> {node.power_level:.1f}% | <b>Temp:</b> {node.temperature:.1f}°C<br>"
            f"<b>Active Workloads:</b> {len(node.workload_queue)}<br>"
            f"<b>Available Compute:</b> {node.get_available_compute():.1f} TFLOPS"
        )

    # Build scatter trace
    trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(size=55, color=colors, line=dict(color="#2c3e50", width=3)),
        text=[nid for nid in coords.keys() if nid in [n.node_id for n in nodes]],
        textposition="top center",
        hoverinfo="text",
        hovertext=hover_texts,
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="<b>Live Orbital AI Node Cluster Topology</b>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
