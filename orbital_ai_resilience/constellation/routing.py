"""ConstellationRouter implementing multi-hop Dijkstra graph routing across orbital satellites."""

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from orbital_ai_resilience.constellation.links import InterSatelliteLink, LinkEvaluator
from orbital_ai_resilience.orbital.eclipse import EclipseModel
from orbital_ai_resilience.orbital.models import OrbitalState


@dataclass
class ConstellationRoute:
    """Represents an end-to-end multi-hop communication route between satellites.

    Attributes:
        source_id: Source satellite node ID.
        target_id: Destination satellite node ID.
        path: List of satellite IDs along route [Source, Relay1, ..., Target].
        total_latency_ms: Summed end-to-end communication latency (ms).
        bottleneck_bandwidth_mbps: Minimum bandwidth along path (Mbps).
        total_distance_km: Summed 3D Euclidean distance (km).
        hop_count: Number of network hops len(path) - 1.
        total_route_cost: Composite route optimization cost.
        is_route_valid: True if valid path exists and all links are active.
    """

    source_id: str
    target_id: str
    path: List[str]
    total_latency_ms: float
    bottleneck_bandwidth_mbps: float
    total_distance_km: float
    hop_count: int
    total_route_cost: float
    is_route_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize route to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "path": self.path,
            "route_string": " -> ".join(self.path),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "bottleneck_bandwidth_mbps": round(self.bottleneck_bandwidth_mbps, 1),
            "total_distance_km": round(self.total_distance_km, 2),
            "hop_count": self.hop_count,
            "total_route_cost": round(self.total_route_cost, 2),
            "is_route_valid": self.is_route_valid,
        }


class ConstellationRouter:
    """Calculates deterministic shortest/lowest-cost multi-hop communication routes using Dijkstra's algorithm."""

    def __init__(
        self,
        link_evaluator: Optional[LinkEvaluator] = None,
        eclipse_model: Optional[EclipseModel] = None,
    ) -> None:
        self.link_evaluator: LinkEvaluator = link_evaluator or LinkEvaluator()
        self.eclipse_model: EclipseModel = eclipse_model or EclipseModel()

    def find_route(
        self,
        source_id: str,
        target_id: str,
        constellation_states: Dict[str, OrbitalState],
        excluded_nodes: Optional[Set[str]] = None,
    ) -> ConstellationRoute:
        """Find optimal multi-hop path from source_id to target_id using Dijkstra graph search."""
        excluded = excluded_nodes or set()

        # Direct loopback check
        if source_id == target_id:
            return ConstellationRoute(
                source_id=source_id,
                target_id=target_id,
                path=[source_id],
                total_latency_ms=0.0,
                bottleneck_bandwidth_mbps=1000.0,
                total_distance_km=0.0,
                hop_count=0,
                total_route_cost=0.0,
                is_route_valid=True,
            )

        # Build ISL link graph between all non-excluded satellites
        adj_links: Dict[str, Dict[str, InterSatelliteLink]] = {}
        for nid1, state1 in constellation_states.items():
            if nid1 in excluded:
                continue
            adj_links[nid1] = {}
            for nid2, state2 in constellation_states.items():
                if nid2 in excluded or nid1 == nid2:
                    continue
                link = self.link_evaluator.evaluate_link(state1, state2)
                if link.available:
                    adj_links[nid1][nid2] = link

        # Dijkstra Algorithm
        # Priority Queue tuple: (cost, current_node, path, accum_latency, accum_dist, min_bw)
        pq: List[Tuple[float, str, List[str], float, float, float]] = [
            (0.0, source_id, [source_id], 0.0, 0.0, 10000.0)
        ]
        visited: Set[str] = set()

        while pq:
            cost, curr, path, lat, dist, min_bw = heapq.heappop(pq)

            if curr in visited:
                continue
            visited.add(curr)

            if curr == target_id:
                return ConstellationRoute(
                    source_id=source_id,
                    target_id=target_id,
                    path=path,
                    total_latency_ms=lat,
                    bottleneck_bandwidth_mbps=min_bw if min_bw < 10000.0 else 1000.0,
                    total_distance_km=dist,
                    hop_count=len(path) - 1,
                    total_route_cost=cost,
                    is_route_valid=True,
                )

            for neighbor, link in adj_links.get(curr, {}).items():
                if neighbor not in visited:
                    # Hop cost = link.communication_cost + 5.0 hop penalty
                    ecl_status = self.eclipse_model.evaluate_illumination(constellation_states[neighbor])
                    ecl_penalty = 15.0 if ecl_status.is_eclipse else 0.0

                    next_cost = cost + link.communication_cost + 5.0 + ecl_penalty
                    next_lat = lat + link.latency_ms
                    next_dist = dist + link.distance_km
                    next_bw = min(min_bw, link.bandwidth_mbps)

                    heapq.heappush(pq, (next_cost, neighbor, path + [neighbor], next_lat, next_dist, next_bw))

        # No valid route found
        return ConstellationRoute(
            source_id=source_id,
            target_id=target_id,
            path=[],
            total_latency_ms=999.0,
            bottleneck_bandwidth_mbps=0.0,
            total_distance_km=9999.0,
            hop_count=0,
            total_route_cost=999.0,
            is_route_valid=False,
        )
