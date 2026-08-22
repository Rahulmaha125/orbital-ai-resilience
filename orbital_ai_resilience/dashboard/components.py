"""Reusable Streamlit UI components and custom alert panels."""

import streamlit as st
from typing import Any, Dict, Optional
from orbital_ai_resilience.detection.types import DetectionResult
from orbital_ai_resilience.recovery.events import MigrationEvent


def render_metric_card(title: str, value: Any, delta: Optional[str] = None, help_text: Optional[str] = None) -> None:
    """Render a styled metric card."""
    st.metric(label=title, value=value, delta=delta, help=help_text)


def render_silent_degradation_alert(det_result: DetectionResult) -> None:
    """Render prominent banner alert for silent AI degradation."""
    st.error(
        f"🚨 **SILENT AI DEGRADATION DETECTED**\n\n"
        f"* **Source Node:** `{det_result.node_id}`\n"
        f"* **Detection Tick:** `Tick {det_result.tick}`\n"
        f"* **Physical Health Score:** `{det_result.physical_health_score:.1f}` (`{det_result.physical_health_state.value}`)\n"
        f"* **AI Behavioral Integrity Score:** `{det_result.behavioral_score:.1f}` (`{det_result.behavioral_state.value}`)\n"
        f"* **AI Output MSE:** `{det_result.details.get('mse', 0.0):.6f}` (Cosine Sim: `{det_result.details.get('cosine_sim', 1.0):.4f}`)\n"
        f"* **Detection Algorithm:** `{det_result.detector_name}`\n"
        f"* **Operational Status:** `ONLINE` | **Trust State:** `SUSPECTED`"
    )


def render_node_badge(status: str, trust: str) -> str:
    """Return color-coded markdown string for node operational & trust state."""
    if status == "ISOLATED" or trust == "ISOLATED":
        return "🔴 `ISOLATED`"
    if trust == "QUARANTINED":
        return "🟠 `ONLINE + QUARANTINED`"
    if status == "DEGRADED":
        return "🟡 `DEGRADED`"
    if trust == "SUSPECTED":
        return "⚠️ `ONLINE + SUSPECTED`"
    if status == "ONLINE" and trust == "TRUSTED":
        return "🟢 `ONLINE + TRUSTED`"
    return f"⚪ `{status} + {trust}`"


def render_migration_timeline(event: MigrationEvent) -> None:
    """Render transaction timeline for a workload recovery event."""
    st.subheader(f"Migration Transaction Pipeline: `{event.migration_id}`")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("**1. DETECTED**")
        st.caption(f"Source: `{event.source_node_id}`")
    with col2:
        st.markdown("**2. MIGRATION DECISION**")
        st.caption(f"Reason: `{event.migration_reason[:20]}...`")
    with col3:
        st.markdown("**3. TARGET SELECTED**")
        st.caption(f"Target: `{event.target_node_id}`")
    with col4:
        st.markdown("**4. VERIFICATION**")
        v_color = "🟢" if event.verification_status.value == "VERIFIED" else "🔴"
        st.caption(f"{v_color} `{event.verification_status.value}`")
    with col5:
        st.markdown("**5. RECOVERY**")
        m_color = "🟢" if event.migration_status.value == "COMPLETED" else "🔴"
        st.caption(f"{m_color} `{event.migration_status.value}`")
