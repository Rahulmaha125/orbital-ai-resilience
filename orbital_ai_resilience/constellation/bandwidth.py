"""BandwidthModel evaluating workload data transfer time and link utilization."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BandwidthTransferEstimate:
    """Estimates data payload transfer time and bottleneck bandwidth along a route."""

    workload_memory_mb: float
    bottleneck_bandwidth_mbps: float
    transfer_time_sec: float
    is_bandwidth_sufficient: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transfer estimate to dictionary."""
        return {
            "workload_memory_mb": round(self.workload_memory_mb, 2),
            "bottleneck_bandwidth_mbps": round(self.bottleneck_bandwidth_mbps, 1),
            "transfer_time_sec": round(self.transfer_time_sec, 4),
            "is_bandwidth_sufficient": self.is_bandwidth_sufficient,
        }


class BandwidthModel:
    """Calculates payload transfer times and bandwidth bottlenecks across constellation links."""

    def __init__(self, min_required_bandwidth_mbps: float = 10.0) -> None:
        self.min_required_bandwidth_mbps: float = min_required_bandwidth_mbps

    def estimate_transfer(
        self,
        workload_memory_mb: float,
        bottleneck_bandwidth_mbps: float,
        max_transfer_time_sec: float = 30.0,
    ) -> BandwidthTransferEstimate:
        """Estimate transfer time for a workload state snapshot across a network link.

        Transfer Time (sec) = (Memory_MB * 8 bits/byte) / Bandwidth_Mbps
        """
        eff_bw = max(0.1, bottleneck_bandwidth_mbps)
        transfer_time = (workload_memory_mb * 8.0) / eff_bw

        is_sufficient = (
            bottleneck_bandwidth_mbps >= self.min_required_bandwidth_mbps
            and transfer_time <= max_transfer_time_sec
        )

        return BandwidthTransferEstimate(
            workload_memory_mb=workload_memory_mb,
            bottleneck_bandwidth_mbps=bottleneck_bandwidth_mbps,
            transfer_time_sec=transfer_time,
            is_bandwidth_sufficient=is_sufficient,
        )
