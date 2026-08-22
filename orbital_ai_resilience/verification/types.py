"""Type definitions and enumerations for output verification."""

from enum import Enum


class VerificationResultState(str, Enum):
    """Result status of an output verification process."""
    VERIFIED = "VERIFIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNVERIFIED = "UNVERIFIED"
