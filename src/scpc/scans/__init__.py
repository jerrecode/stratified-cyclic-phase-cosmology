"""Parameter-space scan and outcome-classification utilities."""

from .config import DEFAULT_SCAN_SCHEMA, validate_scan_config
from .grid import ScanPoint, expand_parameter_grid
from .identity import RunIdentity, canonical_run_identity
from .outcomes import OutcomeAssessment, OutcomeClass, assess_solution
from .records import (
    FailureClass,
    RunRecord,
    RunStatus,
    completed_run_record,
    failed_run_record,
)
from .runner import run_background_scan

__all__ = [
    "DEFAULT_SCAN_SCHEMA",
    "FailureClass",
    "OutcomeAssessment",
    "OutcomeClass",
    "RunIdentity",
    "RunRecord",
    "RunStatus",
    "ScanPoint",
    "assess_solution",
    "canonical_run_identity",
    "completed_run_record",
    "expand_parameter_grid",
    "failed_run_record",
    "run_background_scan",
    "validate_scan_config",
]
