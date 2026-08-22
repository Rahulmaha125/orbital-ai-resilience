"""Type definitions and enumerations for workload migration and recovery."""

from enum import Enum


class MigrationState(str, Enum):
    """Lifecycle state of a workload migration process."""
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    SNAPSHOTTED = "SNAPSHOTTED"
    TRANSFERRED = "TRANSFERRED"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    FAILED = "FAILED"


class VerificationStatus(str, Enum):
    """Post-migration output verification status."""
    VERIFIED = "VERIFIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNVERIFIED = "UNVERIFIED"
