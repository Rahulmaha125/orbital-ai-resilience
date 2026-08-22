"""Trust state enumeration separating physical operational status from computational trust."""

from enum import Enum


class TrustState(str, Enum):
    """Computational trust classification for cluster compute nodes.

    Conceptual Separation:
        NodeStatus = ONLINE (Node hardware is powered and responding)
        TrustState = QUARANTINED / ISOLATED (Node output is untrusted by the cluster)
    """

    TRUSTED = "TRUSTED"          # Fully trusted compute node
    SUSPECTED = "SUSPECTED"      # Under observation due to warning indicators
    QUARANTINED = "QUARANTINED"  # Output verification failed; rejected for new workloads
    ISOLATED = "ISOLATED"        # Confirmed silently degraded source node; isolated from cluster
