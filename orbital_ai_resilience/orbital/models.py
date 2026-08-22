"""Deterministic OrbitalState data structures representing satellite orbital parameters and 3D position vectors."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class OrbitalState:
    """Represents a satellite's deterministic orbital position and kinematics.

    Attributes:
        satellite_id: Node ID of the satellite (e.g., 'node-1').
        altitude_km: Orbit altitude above Earth surface (km), e.g., 550.0 km.
        inclination_deg: Orbital inclination angle in degrees, e.g., 53.0°.
        orbital_phase_deg: Current angular phase position along orbit [0, 360)°.
        mean_motion_rad_per_sec: Angular orbital velocity (radians/sec).
        position_km: (x, y, z) Cartesian position coordinates in ECI frame (km).
        velocity_km_s: (vx, vy, vz) Cartesian velocity vector (km/s).
        timestamp: Simulation timestamp (tick float).
    """

    satellite_id: str
    altitude_km: float
    inclination_deg: float
    orbital_phase_deg: float
    mean_motion_rad_per_sec: float
    position_km: Tuple[float, float, float]
    velocity_km_s: Tuple[float, float, float]
    timestamp: float

    @property
    def radius_km(self) -> float:
        """Distance from Earth center in km (Earth radius 6371.0 + altitude)."""
        x, y, z = self.position_km
        return math.sqrt(x * x + y * y + z * z)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize orbital state to dictionary."""
        return {
            "satellite_id": self.satellite_id,
            "altitude_km": round(self.altitude_km, 2),
            "inclination_deg": round(self.inclination_deg, 2),
            "orbital_phase_deg": round(self.orbital_phase_deg, 2),
            "position_km": [round(c, 2) for c in self.position_km],
            "velocity_km_s": [round(v, 3) for v in self.velocity_km_s],
            "timestamp": self.timestamp,
        }
