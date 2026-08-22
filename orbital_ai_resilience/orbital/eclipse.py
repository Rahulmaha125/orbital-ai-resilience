"""Deterministic EclipseModel evaluating solar eclipse conditions for orbital satellites."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.propagation import EARTH_RADIUS_KM


@dataclass
class EclipseStatus:
    """Represents current and predicted solar illumination state for a satellite."""

    satellite_id: str
    is_eclipse: bool
    sunlight_fraction: float
    eclipse_fraction: float
    ticks_until_next_eclipse: float
    ticks_until_sunlight: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize eclipse status to dictionary."""
        return {
            "satellite_id": self.satellite_id,
            "is_eclipse": self.is_eclipse,
            "sunlight_fraction": round(self.sunlight_fraction, 2),
            "eclipse_fraction": round(self.eclipse_fraction, 2),
            "ticks_until_next_eclipse": round(self.ticks_until_next_eclipse, 1),
            "ticks_until_sunlight": round(self.ticks_until_sunlight, 1),
        }


class EclipseModel:
    """Calculates deterministic solar illumination and Earth shadow eclipse geometry.

    Cylindrical Earth Shadow Assumption:
        Sun is located along +X axis (direction [1, 0, 0]).
        Eclipse occurs when x < 0 AND sqrt(y^2 + z^2) <= R_Earth (6371 km).
    """

    def __init__(self, earth_radius_km: float = EARTH_RADIUS_KM) -> None:
        self.earth_radius_km: float = earth_radius_km

    def evaluate_illumination(self, orbital_state: OrbitalState) -> EclipseStatus:
        """Evaluate if satellite is in Earth shadow or direct sunlight."""
        x, y, z = orbital_state.position_km
        rho = math.sqrt(y * y + z * z)

        # Shadow condition: Behind Earth (x < 0) and within Earth shadow cylinder
        in_shadow = (x < 0.0) and (rho <= self.earth_radius_km)

        sunlight_frac = 0.0 if in_shadow else 1.0
        eclipse_frac = 1.0 if in_shadow else 0.0

        # Estimate ticks until transition based on orbital phase angle
        # 1 full orbit = 360 deg. Approx ~15-20 ticks per orbit depending on step size.
        phase = orbital_state.orbital_phase_deg % 360.0
        if in_shadow:
            ticks_until_sun = (270.0 - phase) % 360.0 / 18.0
            ticks_until_ecl = 0.0
        else:
            ticks_until_ecl = (90.0 - phase) % 360.0 / 18.0
            ticks_until_sun = 0.0

        return EclipseStatus(
            satellite_id=orbital_state.satellite_id,
            is_eclipse=in_shadow,
            sunlight_fraction=sunlight_frac,
            eclipse_fraction=eclipse_frac,
            ticks_until_next_eclipse=max(0.0, ticks_until_ecl),
            ticks_until_sunlight=max(0.0, ticks_until_sun),
        )
