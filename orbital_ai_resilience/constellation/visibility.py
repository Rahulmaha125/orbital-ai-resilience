"""Line-of-sight visibility model evaluating Earth obstruction between orbital satellites."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.propagation import EARTH_RADIUS_KM


@dataclass
class VisibilityResult:
    """Line-of-sight visibility result between two satellites."""

    source_id: str
    target_id: str
    is_visible: bool
    distance_km: float
    min_ray_height_km: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize visibility result to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "is_visible": self.is_visible,
            "distance_km": round(self.distance_km, 2),
            "min_ray_height_km": round(self.min_ray_height_km, 2),
        }


class VisibilityModel:
    """Calculates 3D line-of-sight visibility and Earth limb obstruction between satellites."""

    def __init__(self, earth_radius_km: float = EARTH_RADIUS_KM) -> None:
        self.earth_radius_km: float = earth_radius_km

    def evaluate_visibility(self, state1: OrbitalState, state2: OrbitalState) -> VisibilityResult:
        """Check if 3D line-of-sight between two satellite states is unobstructed by Earth.

        Args:
            state1: Source OrbitalState.
            state2: Target OrbitalState.

        Returns:
            VisibilityResult instance.
        """
        p1 = state1.position_km
        p2 = state2.position_km

        # 3D Euclidean distance d
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 1e-6:
            return VisibilityResult(
                source_id=state1.satellite_id,
                target_id=state2.satellite_id,
                is_visible=True,
                distance_km=0.0,
                min_ray_height_km=state1.altitude_km,
            )

        # Vector segment P(t) = P1 + t * (P2 - P1), t in [0, 1]
        v_dot_v = dx * dx + dy * dy + dz * dz
        p1_dot_v = p1[0] * dx + p1[1] * dy + p1[2] * dz
        t_min = max(0.0, min(1.0, -p1_dot_v / v_dot_v))

        # Closest point on line segment to Earth center (0,0,0)
        cx = p1[0] + t_min * dx
        cy = p1[1] + t_min * dy
        cz = p1[2] + t_min * dz
        min_dist_to_center = math.sqrt(cx * cx + cy * cy + cz * cz)
        min_ray_height = min_dist_to_center - self.earth_radius_km

        is_visible = min_dist_to_center >= self.earth_radius_km

        return VisibilityResult(
            source_id=state1.satellite_id,
            target_id=state2.satellite_id,
            is_visible=is_visible,
            distance_km=dist,
            min_ray_height_km=min_ray_height,
        )
