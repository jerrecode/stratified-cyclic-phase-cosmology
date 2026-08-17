import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scpc.models.phase import (
    ComputationalResourceLimitExceeded,
    PeriodicPotential,
    SCPCParameters,
    integrate_scpc,
)
from scpc.scans.identity import canonical_run_identity, normalize_background_specification
from scpc.scans.records import FailureClass, classify_exception, failed_run_record
from scpc.scans.runner import run_background_scan

RESOURCE_SMOKE_CONFIG = Path("configs/scans/stage1_resource_limit_smoke.yaml")


def _flat_desitter(max_rhs_evaluations=None):
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0, strata_count=1),
    )
    return integrate_scpc(
        parameters,
        t_span=(0.0, 1.0),
        samples=51,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_rhs_evaluations=max_rhs_evaluations,
    )


def _identity_spec(limit):
    return {
        "run": {
            "t_start": 0.0,
            "t_end": 1.0,
            "samples": 11,
            "method": "DOP853",
            "rtol": 1.0e-9,
            "atol": 1.0e-11,
            "resource_limits": {"max_rhs_evaluations": limit},
        }
    }


def _rows(index_path: Path) -> list[dict[str, str]]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_rhs_limit_exact_boundary_and_success_accounting() -> None:
    reference = _flat_desitter()
    required = int(reference.solver_metadata["solver_nfev"])
    assert required > 1
    assert reference.solver_metadata["rhs_evaluations_consumed"] == required
    assert reference.solver_metadata["rhs_evaluation_limit"] == "unbounded"

    at_limit = _flat_desitter(required)
    above_limit = _flat_desitter(required + 1)
    assert int(at_limit.solver_metadata["solver_nfev"]) == required
    assert at_limit.solver_metadata["rhs_evaluations_consumed"] == required
    assert at_limit.solver_metadata["rhs_evaluation_limit"] == required
    assert np.array_equal(at_limit.t, above_limit.t)
    assert np.allclose(at_limit.a, above_limit.a, rtol=0.0, atol=0.0)

    with pytest.raises(ComputationalResourceLimitExceeded) as caught:
        _flat_desitter(required - 1)
    error = caught.value
    assert error.limit_kind == "max_rhs_evaluations"
    assert error.configured_limit == required - 1
    assert error.completed_count == required - 1
    assert error.attempted_count == required
    assert np.isfinite(error.evaluation_time)
    assert error.diagnostic_rhs_evaluations >= 0
    assert not hasattr(error, "state")


@pytest.mark.parametrize("value", [True, 1.5, 0, -1, np.inf, -np.inf, np.nan])
def test_invalid_rhs_limits_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_rhs_evaluations"):
        _flat_desitter(value)


def test_resource_limit_identity_normalization_and_separation() -> None:
    integer = normalize_background_specification(_identity_spec(64))
    float_equivalent = normalize_background_specification(_identity_spec(64.0))
    different = normalize_background_specification(_identity_spec(65))
    assert integer["run"]["resource_limits"]["max_rhs_evaluations"] == 64
    assert integer == float_equivalent
    assert canonical_run_identity(integer) == canonical_run_identity(float_equivalent)
    assert canonical_run_identity(integer).sha256 != canonical_run_identity(different).sha256


@pytest.mark.parametrize("value", [False, 3.25, 0, -3, np.inf, np.nan])
def test_invalid_identity_resource_limits_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_rhs_evaluations"):
        normalize_background_specification(_identity_spec(value))


def test_resource_exception_has_dedicated_failure_record_without_endpoint() -> None:
    specification = normalize_background_specification(_identity_spec(7))
    identity = canonical_run_identity(specification)
    error = ComputationalResourceLimitExceeded(
        configured_limit=7,
        completed_count=7,
        attempted_count=8,
        evaluation_time=0.25,
        diagnostic_rhs_evaluations=3,
    )
    assert classify_exception(error) is FailureClass.RESOURCE_LIMIT_EXCEEDED
    record = failed_run_record(identity, specification, error)
    assert record.failure_class is FailureClass.RESOURCE_LIMIT_EXCEEDED
    assert record.outcome is None
    assert record.termination_kind is None
    assert record.termination_state_vector is None
    assert record.solver_metadata is None
    assert record.resource_limit_kind == "max_rhs_evaluations"
    assert record.resource_limit_configured == 7
    assert record.resource_completed_count == 7
    assert record.resource_attempted_count == 8
    assert record.resource_evaluation_time == 0.25
    assert record.diagnostic_rhs_evaluations == 3


def test_resource_smoke_scan_is_atomic_and_resumable(tmp_path) -> None:
    output = tmp_path / "resource"
    index = run_background_scan(RESOURCE_SMOKE_CONFIG, output)
    first = _rows(index)
    first_bytes = index.read_bytes()
    assert len(first) == 1
    row = first[0]
    assert row["status"] == "failed"
    assert row["failure_class"] == "resource_limit_exceeded"
    assert row["outcome"] == ""
    assert row["termination_kind"] == ""
    assert json.loads(row["termination_state_vector"]) is None
    assert json.loads(row["solver_metadata"]) is None
    assert int(row["resource_limit_configured"]) == 1
    assert int(row["resource_completed_count"]) == 1
    assert int(row["resource_attempted_count"]) == 2

    resumed = run_background_scan(RESOURCE_SMOKE_CONFIG, output)
    assert resumed == index
    assert _rows(resumed) == first
    assert resumed.read_bytes() == first_bytes

    metadata = json.loads((output / "scan_metadata.json").read_text(encoding="utf-8"))
    assert metadata["metadata_schema_version"] == 3
    assert len(metadata["resource_limits_by_run"]) == 1
    stored = next(iter(metadata["resource_limits_by_run"].values()))
    assert stored == {"max_rhs_evaluations": 1}


def test_rerun_of_resource_failure_preserves_one_atomic_row(tmp_path) -> None:
    source_scan = RESOURCE_SMOKE_CONFIG.read_text(encoding="utf-8")
    source_base = Path("configs/scans/stage1_resource_limit_smoke_base.yaml").read_text(
        encoding="utf-8"
    )
    scan = tmp_path / "rerun.yaml"
    base = tmp_path / "stage1_resource_limit_smoke_base.yaml"
    base.write_text(source_base, encoding="utf-8")
    scan.write_text(
        source_scan.replace("resume: true", "resume: true\nrerun_statuses: [failed]"),
        encoding="utf-8",
    )
    output = tmp_path / "rerun-output"
    first_index = run_background_scan(scan, output)
    first_rows = _rows(first_index)
    second_index = run_background_scan(scan, output)
    second_rows = _rows(second_index)
    assert len(first_rows) == len(second_rows) == 1
    assert first_rows[0]["run_id"] == second_rows[0]["run_id"]
    assert second_rows[0]["failure_class"] == "resource_limit_exceeded"


def test_changing_ceiling_changes_identity() -> None:
    one = canonical_run_identity(normalize_background_specification(_identity_spec(5)))
    two = canonical_run_identity(normalize_background_specification(_identity_spec(6)))
    assert one.run_id != two.run_id
    assert one.sha256 != two.sha256
