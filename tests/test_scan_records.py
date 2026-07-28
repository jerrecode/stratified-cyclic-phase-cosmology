import json

import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.scans.errors import OutputSerializationError, ResultIntegrityError
from scpc.scans.identity import canonical_run_identity
from scpc.scans.outcomes import OutcomeAssessment, OutcomeClass
from scpc.scans.records import (
    FailureClass,
    RunStatus,
    classify_exception,
    completed_run_record,
    failed_run_record,
)


def _solution() -> SCPCSolution:
    time = np.asarray([0.0, 1.0])
    return SCPCSolution(
        t=time,
        a=np.asarray([1.0, 1.1]),
        H=np.asarray([0.1, 0.1]),
        phi=np.zeros(2),
        phi_dot=np.zeros(2),
        rho_m=np.zeros(2),
        rho_r=np.zeros(2),
        rho_phi=np.ones(2),
        p_phi=-np.ones(2),
        constraint_residual=np.zeros(2),
        turning_times=np.asarray([]),
        turning_kinds=(),
        parameters=SCPCParameters(
            spatial_curvature_k=0,
            rho_m_ref=0.0,
            rho_r_ref=0.0,
            potential=PeriodicPotential(offset=3.0, amplitude=0.0),
        ),
        solver_metadata={"solver_method": "DOP853", "solver_nfev": 17},
        turning_state_vectors=np.empty((0, 4)),
    )


def test_completed_record_preserves_assessment_and_solver_metadata() -> None:
    specification = {"model": {"offset": 3.0}, "run": {"method": "DOP853"}}
    identity = canonical_run_identity(specification)
    assessment = OutcomeAssessment(
        outcome=OutcomeClass.MONOTONIC_EXPANSION,
        reason="H remains positive.",
        numerically_valid=True,
        bounce_count=0,
        turnaround_count=0,
        degenerate_count=0,
        event_sequence=(),
        max_abs_constraint_residual=0.0,
        return_sequence_classifications={},
    )
    record = completed_run_record(
        identity,
        specification,
        assessment,
        _solution(),
        trajectory_path="trajectories/run.nc",
    )

    assert record.status is RunStatus.COMPLETED
    assert record.outcome == "monotonic_expansion"
    assert record.run_sha256 == identity.sha256
    assert record.solver_metadata == {"solver_method": "DOP853", "solver_nfev": 17}
    assert record.trajectory_path == "trajectories/run.nc"

    row = record.to_flat_row()
    assert json.loads(row["specification"]) == specification
    assert json.loads(row["event_sequence"]) == []
    assert json.loads(row["return_sequence_classifications"]) == {}


def test_numerically_invalid_completed_solution_is_rejected_not_failed() -> None:
    specification = {"case": "constraint-violation"}
    identity = canonical_run_identity(specification)
    assessment = OutcomeAssessment(
        outcome=OutcomeClass.CONSTRAINT_VIOLATION,
        reason="Residual exceeded threshold.",
        numerically_valid=False,
        bounce_count=0,
        turnaround_count=0,
        degenerate_count=0,
        event_sequence=(),
        max_abs_constraint_residual=1.0e-3,
        return_sequence_classifications={},
    )
    record = completed_run_record(identity, specification, assessment, _solution())
    assert record.status is RunStatus.REJECTED
    assert record.failure_class is None


def test_initial_constraint_error_is_preserved_without_singularity_claim() -> None:
    error = ValueError("Initial state violates the Friedmann constraint: H^2=-1")
    specification = {"a0": 1.0}
    record = failed_run_record(canonical_run_identity(specification), specification, error)
    assert record.status is RunStatus.FAILED
    assert record.failure_class is FailureClass.INVALID_INITIAL_CONSTRAINT
    assert record.exception_type == "ValueError"
    assert "singular" not in (record.reason or "").lower()


def test_exception_classification_is_conservative_and_phase_specific() -> None:
    assert classify_exception(FloatingPointError("a reached zero")) is FailureClass.PHYSICAL_DOMAIN_FAILURE
    assert classify_exception(RuntimeError("Background integration failed: step size")) is FailureClass.SOLVER_FAILURE
    assert classify_exception(KeyError("missing")) is FailureClass.CONFIGURATION_ERROR
    assert classify_exception(ResultIntegrityError("bad event data")) is FailureClass.RESULT_INTEGRITY_ERROR
    assert classify_exception(OutputSerializationError("disk write")) is FailureClass.OUTPUT_ERROR
    assert classify_exception(OSError("filesystem")) is FailureClass.UNEXPECTED_ERROR
