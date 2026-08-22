"""InterSatelliteLink dataclass and evaluation module for inter-satellite crosslinks."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional
from orbital_ai_resilience.constellation.visibility import VisibilityModel, VisibilityResult
from orbital_ai_resilience.orbital.models import OrbitalState

SPEED_OF_LIGHT_KM_S = 299792.458  # Speed of light in km/s


@dataclass
class InterSatelliteLink:
    """Represents an active or potential Inter-Satellite Crosslink (ISL).

    Attributes:
        source_satellite: Source node ID.
        target_satellite: Target node ID.
        distance_km: 3D distance between satellites in km.
        available: True if link is active and unobstructed by Earth.
        latency_ms: Propagation + processing latency in milliseconds.
        bandwidth_mbps: Available link bandwidth in Mbps.
        signal_quality: Normalized signal quality [0.0, 1.0].
        communication_cost: Normalized communication link cost.
    """

    source_satellite: str
    target_satellite: str
    distance_km: float
    available: bool
    latency_ms: float
    bandwidth_mbps: float
    signal_quality: float
    communication_cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ISL link parameters to dictionary."""
        return {
            "source_satellite": self.source_satellite,
            "target_satellite": self.target_satellite,
            "distance_km": round(self.distance_km, 2),
            "available": self.available,
            "latency_ms": round(self.latency_ms, 2),
            "bandwidth_mbps": round(self.bandwidth_mbps, 1),
            "signal_quality": round(self.signal_quality, 2),
            "communication_cost": round(self.communication_cost, 2),
        }


class LinkEvaluator:
    """Evaluates physical inter-satellite link metrics derived from distance and visibility."""

    def __init__(
        self,
        max_range_km: float = 12000.0,
        base_bandwidth_mbps: float = 1000.0,
        base_processing_delay_ms: float = 2.0,
        visibility_model: Optional[VisibilityModel] = None,
    ) -> None:
        self.max_range_km: float = max_range_km
        self.base_bandwidth_mbps: float = base_bandwidth_mbps
        self.base_processing_delay_ms: float = base_processing_delay_ms
        self.visibility_model: VisibilityModel = visibility_model or VisibilityModel()

    def evaluate_link(self, state1: OrbitalState, state2: OrbitalState) -> InterSatelliteLink:
        """Compute InterSatelliteLink metrics between two orbital states."""
        vis = self.visibility_model.evaluate_visibility(state1, state2)

        is_avail = vis.is_visible and (vis.distance_km <= self.max_range_km)

        if vis.distance_km == 0.0:
            return InterSatelliteLink(
                source_satellite=state1.satellite_id,
                target_satellite=state2.satellite_id,
                distance_km=0.0,
                available=True,
                latency_ms=0.0,
                bandwidth_mbps=self.base_bandwidth_mbps,
                signal_quality=1.0,
                communication_cost=0.0,
            )

        propagation_delay_ms = (vis.distance_km / SPEED_OF_LIGHT_KM_S) * 1000.0
        latency = self.base_processing_delay_ms + propagation_delay_ms

        range_ratio = min(1.0, vis.distance_km / self.max_range_km)
        signal_qual = max(0.1, 1.0 - (0.5 * (range_ratio ** 2)))
        bandwidth = self.base_bandwidth_mbps * signal_qual

        cost = (0.5 * latency) + (100.0 / max(1.0, bandwidth)) + (20.0 * (1.0 - signal_qual))

        return InterSatelliteLink(
            source_satellite=state1.satellite_id,
            target_satellite=state2.satellite_id,
            distance_km=vis.distance_km,
            available=is_avail,
            latency_ms=latency,
            bandwidth_mbps=bandwidth,
            signal_quality=signal_qual,
            communication_cost=round(cost, 2),
        )
