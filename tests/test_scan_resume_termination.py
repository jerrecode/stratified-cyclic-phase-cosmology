import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scpc.scans.runner as runner
from scpc.scans.termination_records import (
    read_termination_record,
    write_content_addressed_termination_record,
)


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
                "identity_namespace": "scpc-resume-termination-test-v1",
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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recompute_constraint(solution) -> float:
    a = float(solution.a[-1])
    hubble = float(solution.H[-1])
    field = float(solution.phi[-1])
    field_velocity = float(solution.phi_dot[-1])
    parameters = solution.parameters
    matter = float(parameters.matter_density(a))
    radiation = float(parameters.radiation_density(a))
    field_density = 0.5 * field_velocity**2 + float(parameters.potential.value(field))
    lhs = hubble**2 + parameters.spatial_curvature_k / a**2
    rhs = (matter + radiation + field_density) / 3.0
    return float((lhs - rhs) / max(abs(rhs), 1.0e-15))


def test_domain_scan_indexes_independent_exact_termination_evidence(tmp_path) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    row = _rows(index)[0]

    assert row["status"] == "rejected"
    assert row["outcome"] == "physical_domain_termination"
    state = json.loads(row["termination_state_vector"])
    assert len(state) == 4
    assert state[0] == pytest.approx(0.8)
    assert np.isfinite(float(row["termination_constraint_residual"]))
    assert json.loads(row["termination_boundaries"])[0]["kind"] == "minimum_scale_factor"
    record = output / row["termination_record_path"]
    assert record.is_file()
    assert hashlib.sha256(record.read_bytes()).hexdigest() == row["termination_record_sha256"]
    payload = read_termination_record(record, row["termination_record_sha256"])
    assert payload["termination"]["state"][1] == float(state[1]).hex()
    summary = json.loads((output / "scan_summary.json").read_text(encoding="utf-8"))
    assert summary["termination_record_count"] == 1
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert row["termination_record_path"] in {item["path"] for item in provenance["outputs"]}


def test_higher_priority_constraint_rejection_with_termination_is_reintegrated(
    tmp_path,
    monkeypatch,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    original_integration = runner._integrate_point

    def constraint_violating_termination(point):
        solution = original_integration(point)
        solution.H[-1] += 0.01
        solution.termination_state_vector[1] = solution.H[-1]
        solution.constraint_residual[-1] = _recompute_constraint(solution)
        return solution

    monkeypatch.setattr(runner, "_integrate_point", constraint_violating_termination)
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    first_rows = _rows(index)
    assert first_rows[0]["outcome"] == "constraint_violation"
    assert first_rows[0]["termination_kind"] == "minimum_scale_factor"

    executions = 0

    def counted_integration(point):
        nonlocal executions
        executions += 1
        return constraint_violating_termination(point)

    monkeypatch.setattr(runner, "_integrate_point", counted_integration)
    resumed = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    assert _rows(resumed) == first_rows
    assert executions == 1
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["independently_reintegrated_termination_run_ids"] == [
        first_rows[0]["run_id"]
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "primary_kind",
        "boundary_set",
        "state_vector",
        "state_hubble_sign",
        "primary_nan",
        "outcome",
        "record_digest",
        "record_missing",
    ],
)
def test_resume_rejects_corrupted_termination_record_before_execution(
    tmp_path,
    monkeypatch,
    mutation,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    rows = _rows(index)
    row = rows[0]

    if mutation == "primary_kind":
        row["termination_kind"] = "maximum_scale_factor"
    elif mutation == "boundary_set":
        row["termination_boundaries"] = "[]"
    elif mutation == "state_vector":
        state = json.loads(row["termination_state_vector"])
        state[0] += 0.05
        row["termination_state_vector"] = json.dumps(state)
    elif mutation == "state_hubble_sign":
        state = json.loads(row["termination_state_vector"])
        state[1] = -state[1]
        row["termination_state_vector"] = json.dumps(state)
    elif mutation == "primary_nan":
        row["termination_threshold"] = "nan"
    elif mutation == "outcome":
        row["outcome"] = "monotonic_contraction"
    elif mutation == "record_digest":
        row["termination_record_sha256"] = "0" * 64
    elif mutation == "record_missing":
        (output / row["termination_record_path"]).unlink()
    else:  # pragma: no cover
        raise AssertionError(f"Unknown mutation: {mutation}")
    _write_rows(index, rows)

    executed = False

    def forbidden_integration(point):
        nonlocal executed
        executed = True
        raise AssertionError(f"Unexpected integration of {point.identity.run_id}")

    monkeypatch.setattr(runner, "_integrate_point", forbidden_integration)
    with pytest.raises(
        ValueError,
        match="(?i)terminated|termination|domain|outcome|constraint|evidence|checksum|missing",
    ):
        runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    assert executed is False


@pytest.mark.parametrize(
    "forged_outcome",
    ["constraint_violation", "degenerate_turning_event", "unresolved_event_detection"],
)
def test_resume_recomputes_outcome_even_with_matching_forged_artifact(
    tmp_path,
    monkeypatch,
    forged_outcome,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    rows = _rows(index)
    row = rows[0]
    old_record = output / row["termination_record_path"]
    payload = read_termination_record(old_record, row["termination_record_sha256"])
    payload["classification"]["outcome"] = forged_outcome
    forged_record, forged_digest = write_content_addressed_termination_record(
        payload,
        output / "termination_records",
        row["run_id"],
    )
    row["outcome"] = forged_outcome
    row["termination_record_path"] = str(forged_record.relative_to(output))
    row["termination_record_sha256"] = forged_digest
    _write_rows(index, rows)

    monkeypatch.setattr(
        runner,
        "_integrate_point",
        lambda point: (_ for _ in ()).throw(AssertionError(point.identity.run_id)),
    )
    with pytest.raises(ValueError, match="expected 'physical_domain_termination'"):
        runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)


def test_resume_reintegrates_coherently_forged_constraint_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    rows = _rows(index)
    row = rows[0]
    old_record = output / row["termination_record_path"]
    payload = read_termination_record(old_record, row["termination_record_sha256"])

    forged_maximum = 1.0e-3
    row["outcome"] = "constraint_violation"
    row["max_abs_constraint_residual"] = str(forged_maximum)
    payload["classification"]["outcome"] = "constraint_violation"
    payload["classification"]["max_abs_constraint_residual"] = forged_maximum.hex()
    forged_record, forged_digest = write_content_addressed_termination_record(
        payload,
        output / "termination_records",
        row["run_id"],
    )
    row["termination_record_path"] = str(forged_record.relative_to(output))
    row["termination_record_sha256"] = forged_digest
    _write_rows(index, rows)

    original_integration = runner._integrate_point
    executions = 0

    def counted_integration(point):
        nonlocal executions
        executions += 1
        return original_integration(point)

    monkeypatch.setattr(runner, "_integrate_point", counted_integration)
    resumed = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    refreshed = _rows(resumed)[0]
    assert executions == 1
    assert refreshed["outcome"] == "physical_domain_termination"
    assert float(refreshed["max_abs_constraint_residual"]) < forged_maximum
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["independently_reintegrated_termination_run_ids"] == [row["run_id"]]


def test_resume_removes_unreferenced_termination_record(tmp_path) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    orphan = output / "termination_records" / "orphan.json"
    orphan.write_text("{}", encoding="utf-8")
    runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    assert not orphan.exists()
    assert _rows(index)[0]["termination_record_path"]
