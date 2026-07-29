import csv
import json
from pathlib import Path

import pytest
import yaml

import scpc.scans.runner as runner


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


def test_domain_scan_indexes_exact_termination_state(tmp_path) -> None:
    scan = _write_domain_protocol(tmp_path)
    index = runner.run_background_scan(scan, tmp_path / "output", schema_path=SCAN_SCHEMA)
    row = _rows(index)[0]

    assert row["status"] == "rejected"
    assert row["outcome"] == "physical_domain_termination"
    state = json.loads(row["termination_state_vector"])
    assert len(state) == 4
    assert state[0] == pytest.approx(0.8)
    assert json.loads(row["termination_boundaries"])[0]["kind"] == "minimum_scale_factor"


@pytest.mark.parametrize(
    "mutation",
    [
        "primary_kind",
        "boundary_set",
        "state_vector",
        "primary_nan",
        "outcome",
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
    elif mutation == "primary_nan":
        row["termination_threshold"] = "nan"
    elif mutation == "outcome":
        row["outcome"] = "monotonic_contraction"
    else:  # pragma: no cover
        raise AssertionError(f"Unknown mutation: {mutation}")
    _write_rows(index, rows)

    executed = False

    def forbidden_integration(point):
        nonlocal executed
        executed = True
        raise AssertionError(f"Unexpected integration of {point.identity.run_id}")

    monkeypatch.setattr(runner, "_integrate_point", forbidden_integration)
    with pytest.raises(ValueError, match="(?i)terminated|termination|domain|outcome"):
        runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    assert executed is False
