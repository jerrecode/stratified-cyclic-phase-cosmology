"""Parameter-space scan and outcome-classification utilities."""

from .identity import RunIdentity, canonical_run_identity
from .outcomes import OutcomeAssessment, OutcomeClass, assess_solution
from .records import (
    FailureClass,
    RunRecord,
    RunStatus,
    completed_run_record,
    failed_run_record,
)

__all__ = [
    "FailureClass",
    "OutcomeAssessment",
    "OutcomeClass",
    "RunIdentity",
    "RunRecord",
    "RunStatus",
    "assess_solution",
    "canonical_run_identity",
    "completed_run_record",
    "failed_run_record",
]
