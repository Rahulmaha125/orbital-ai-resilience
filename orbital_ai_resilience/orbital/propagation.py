"""Deterministic OrbitalPropagator for circular satellite orbits in ECI frame."""

import math
from typing import Dict, List, Tuple
from orbital_ai_resilience.orbital.models import OrbitalState

EARTH_RADIUS_KM = 6371.0
EARTH_MU = 398600.4418  # Earth gravitational parameter km^3/s^2


class OrbitalPropagator:
    """Propagates satellite orbital positions deterministically using circular Keplerian motion.

    Assumptions:
        - Circular Low Earth Orbit (LEO) at ~550 km altitude.
        - Earth radius R_Earth = 6371.0 km.
        - Simplified Keplerian propagation: theta(t) = theta_0 + omega * dt.
    """

    def __init__(
        self,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        tick_duration_seconds: float = 60.0,
    ) -> None:
        self.altitude_km: float = altitude_km
        self.inclination_deg: float = inclination_deg
        self.tick_duration_seconds: float = tick_duration_seconds

        self.orbit_radius_km: float = EARTH_RADIUS_KM + self.altitude_km
        # Angular velocity omega = sqrt(mu / R^3) rad/sec
        self.mean_motion: float = math.sqrt(EARTH_MU / (self.orbit_radius_km ** 3))

    def compute_state(self, satellite_id: str, initial_phase_deg: float, tick: float) -> OrbitalState:
        """Compute deterministic 3D position and velocity for a satellite at a specific tick."""
        elapsed_sec = tick * self.tick_duration_seconds
        current_phase_rad = math.radians(initial_phase_deg) + (self.mean_motion * elapsed_sec)
        current_phase_rad = current_phase_rad % (2.0 * math.pi)
        current_phase_deg = math.degrees(current_phase_rad)

        inc_rad = math.radians(self.inclination_deg)
        r = self.orbit_radius_km

        # 3D Cartesian coordinates in ECI frame
        x = r * math.cos(current_phase_rad)
        y = r * math.sin(current_phase_rad) * math.cos(inc_rad)
        z = r * math.sin(current_phase_rad) * math.sin(inc_rad)

        # 3D Velocity vector components (km/s)
        v_mag = math.sqrt(EARTH_MU / r)
        vx = -v_mag * math.sin(current_phase_rad)
        vy = v_mag * math.cos(current_phase_rad) * math.cos(inc_rad)
        vz = v_mag * math.cos(current_phase_rad) * math.sin(inc_rad)

        return OrbitalState(
            satellite_id=satellite_id,
            altitude_km=self.altitude_km,
            inclination_deg=self.inclination_deg,
            orbital_phase_deg=current_phase_deg,
            mean_motion_rad_per_sec=self.mean_motion,
            position_km=(x, y, z),
            velocity_km_s=(vx, vy, vz),
            timestamp=tick,
        )

    def generate_constellation_states(self, num_satellites: int = 5, tick: float = 0.0) -> Dict[str, OrbitalState]:
        """Generate deterministic orbital states for a constellation of satellites distributed along orbit."""
        states = {}
        phase_spacing = 360.0 / num_satellites
        for i in range(num_satellites):
            sat_id = f"node-{i + 1}"
            init_phase = (i * phase_spacing) % 360.0
            states[sat_id] = self.compute_state(sat_id, initial_phase_deg=init_phase, tick=tick)
        return states
