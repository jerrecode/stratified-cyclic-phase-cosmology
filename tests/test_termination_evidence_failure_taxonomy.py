import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scpc.scans.runner as runner
from scpc.scans.errors import OutputSerializationError, ResultIntegrityError


SCAN_SCHEMA = Path("configs/scans/scan.schema.json")


def _write_domain_protocol(tmp_path: Path) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {
                    "background": {
                        "spatial_curvature_k": 0,
                        "rho_m_ref": 0.0,
                        "rho_r_ref": 0.0,
                        "a_ref": 1.0,
                    },
                    "potential": {
                        "offset": 3.0,
                        "amplitude": 0.0,
                        "strata_count": 1,
                        "field_scale": 1.0,
                        "target_space": "real",
                    },
                },
                "initial_conditions": {
                    "a": 1.0,
                    "phi": 0.0,
                    "phi_dot": 0.0,
                    "branch": -1,
                },
                "run": {
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "samples": 101,
                    "method": "DOP853",
                    "rtol": 1.0e-10,
                    "atol": 1.0e-12,
                    "max_step": 0.025,
                    "domain_check_substeps": 16,
                    "domain": {"min_scale_factor": 0.8},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scan = tmp_path / "scan.yaml"
    scan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_config": "base.yaml",
                "identity_namespace": "scpc-termination-failure-taxonomy-v1",
                "max_runs": 1,
                "resume": True,
                "axes": {},
                "classification": {
                    "constraint_threshold": 1.0e-7,
                    "hubble_zero_tolerance": 1.0e-10,
                    "return_tolerance": 1.0e-3,
                },
                "retention": {"outcomes": [], "max_trajectories": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return scan


def _single_row(index: Path) -> dict[str, str]:
    with index.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]


def _assert_failed_row(output: Path, row: dict[str, str], failure_class: str, exception: str) -> None:
    assert row["status"] == "failed"
    assert row["numerically_valid"] == "False"
    assert row["failure_class"] == failure_class
    assert row["exception_type"] == exception
    assert row["outcome"] == ""
    assert row["termination_record_path"] == ""
    assert row["termination_record_sha256"] == ""
    summary = json.loads((output / "scan_summary.json").read_text(encoding="utf-8"))
    assert summary["failure_class_counts"] == {failure_class: 1}
    assert summary["termination_record_count"] == 0
    records = output / "termination_records"
    assert not records.exists() or not list(records.glob("*.json"))


def test_termination_evidence_io_failure_is_output_error(tmp_path, monkeypatch) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"

    def fail_serialization(*args, **kwargs):
        raise OutputSerializationError("forced termination evidence write failure")

    monkeypatch.setattr(runner, "write_content_addressed_termination_record", fail_serialization)
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    row = _single_row(index)
    _assert_failed_row(output, row, "output_error", "OutputSerializationError")


def test_malformed_postintegration_evidence_is_result_integrity_error(
    tmp_path,
    monkeypatch,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"

    def fail_evidence_construction(**kwargs):
        raise ResultIntegrityError("forced malformed in-memory termination evidence")

    monkeypatch.setattr(runner, "termination_evidence_payload", fail_evidence_construction)
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    row = _single_row(index)
    _assert_failed_row(output, row, "result_integrity_error", "ResultIntegrityError")


@pytest.mark.parametrize(
    "field",
    ["termination_time", "requested_end_time", "termination_state_component"],
)
def test_malformed_termination_scalars_are_integrity_failures(
    tmp_path,
    monkeypatch,
    field,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    original_integration = runner._integrate_point

    def malformed_solution(point):
        solution = original_integration(point)
        if field == "termination_time":
            solution.termination_time = "not-a-time"
        elif field == "requested_end_time":
            solution.requested_end_time = "not-an-endpoint"
        else:
            state = np.asarray(solution.termination_state_vector, dtype=object)
            state[1] = "not-a-hubble-value"
            solution.termination_state_vector = state
        return solution

    monkeypatch.setattr(runner, "_integrate_point", malformed_solution)
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    row = _single_row(index)
    _assert_failed_row(output, row, "result_integrity_error", "ResultIntegrityError")


def test_nonfinite_terminated_solution_does_not_create_durable_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    original_integration = runner._integrate_point

    def nonfinite_solution(point):
        solution = original_integration(point)
        solution.H[-1] = np.nan
        return solution

    monkeypatch.setattr(runner, "_integrate_point", nonfinite_solution)
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    row = _single_row(index)
    _assert_failed_row(output, row, "result_integrity_error", "ResultIntegrityError")
