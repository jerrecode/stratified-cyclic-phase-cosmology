"""Parameter-space scan and outcome-classification utilities."""

from .identity import RunIdentity, canonical_run_identity
from .outcomes import OutcomeAssessment, OutcomeClass, assess_solution

__all__ = [
    "OutcomeAssessment",
    "OutcomeClass",
    "RunIdentity",
    "assess_solution",
    "canonical_run_identity",
]
