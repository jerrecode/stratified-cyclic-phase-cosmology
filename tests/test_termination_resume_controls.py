import csv
import json
from pathlib import Path

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
                "identity_namespace": "scpc-resume-control-verification-v1",
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


def test_resume_reintegrates_and_restores_dense_detection_controls(
    tmp_path,
    monkeypatch,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    rows = _rows(index)
    row = rows[0]

    metadata = json.loads(row["solver_metadata"])
    metadata["solver_method"] = "RK45"
    metadata["solver_max_step"] = 9.0
    metadata["domain_check_substeps"] = 2
    row["solver_metadata"] = json.dumps(metadata, sort_keys=True)
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
    refreshed_metadata = json.loads(refreshed["solver_metadata"])

    assert executions == 1
    assert refreshed_metadata["solver_method"] == "DOP853"
    assert refreshed_metadata["solver_max_step"] == 0.025
    assert refreshed_metadata["domain_check_substeps"] == 16
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["independently_reintegrated_termination_run_ids"] == [row["run_id"]]
