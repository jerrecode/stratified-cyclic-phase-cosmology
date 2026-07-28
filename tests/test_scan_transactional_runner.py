import csv
import json
from pathlib import Path

import pytest
import yaml

import scpc.scans.runner as runner


SCAN_SCHEMA = Path("configs/scans/scan.schema.json")


def _write_protocol(tmp_path: Path) -> Path:
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
                    "branch": 1,
                },
                "run": {
                    "t_start": 0.0,
                    "t_end": 0.5,
                    "samples": 51,
                    "method": "DOP853",
                    "rtol": 1.0e-9,
                    "atol": 1.0e-11,
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
                "identity_namespace": "scpc-transaction-test-v1",
                "max_runs": 1,
                "resume": True,
                "rerun_statuses": ["completed"],
                "axes": {},
                "classification": {
                    "constraint_threshold": 1.0e-7,
                    "hubble_zero_tolerance": 1.0e-10,
                    "return_tolerance": 1.0e-3,
                },
                "retention": {
                    "outcomes": ["monotonic_expansion"],
                    "max_trajectories": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return scan


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_scan_import_resolves_transactional_runner_package() -> None:
    assert Path(runner.__file__).name == "__init__.py"


def test_resume_metadata_contains_strict_implementation_runtime_fingerprint(tmp_path) -> None:
    scan = _write_protocol(tmp_path)
    output = tmp_path / "output"
    runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)

    metadata = json.loads((output / "scan_metadata.json").read_text(encoding="utf-8"))
    fingerprint = metadata["implementation_runtime_fingerprint"]
    assert metadata["metadata_schema_version"] == 2
    assert len(fingerprint["source_tree_sha256"]) == 64
    assert len(fingerprint["strict_sha256"]) == 64
    assert fingerprint["python_version"]
    assert fingerprint["platform"]
    assert fingerprint["machine"]
    assert "python_executable" not in fingerprint


def test_tampered_runtime_fingerprint_blocks_resume_before_execution(tmp_path, monkeypatch) -> None:
    scan = _write_protocol(tmp_path)
    output = tmp_path / "output"
    runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    metadata_path = output / "scan_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["implementation_runtime_fingerprint"]["strict_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    executed = False

    def forbidden_integration(point):
        nonlocal executed
        executed = True
        raise AssertionError(f"Unexpected integration of {point.identity.run_id}")

    monkeypatch.setattr(runner, "_integrate_point", forbidden_integration)
    with pytest.raises(ValueError, match="configuration or runtime"):
        runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    assert executed is False


def test_interruption_before_index_commit_preserves_old_durable_result(tmp_path, monkeypatch) -> None:
    scan = _write_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    old_row = _rows(index)[0]
    old_trajectory = output / old_row["trajectory_path"]
    assert old_trajectory.is_file()

    original_integrate = runner._integrate_point
    original_replace = runner.replace_index_row_atomic

    def changed_integration(point):
        solution = original_integrate(point)
        solution.phi = solution.phi + 0.125
        return solution

    def interrupt_before_commit(index_path, rows, new_row):
        raise KeyboardInterrupt("simulated hard interruption before atomic index replacement")

    monkeypatch.setattr(runner, "_integrate_point", changed_integration)
    monkeypatch.setattr(runner, "replace_index_row_atomic", interrupt_before_commit)
    with pytest.raises(KeyboardInterrupt, match="simulated hard interruption"):
        runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)

    assert _rows(index) == [old_row]
    assert old_trajectory.is_file()
    orphan_candidates = [
        path for path in (output / "trajectories").glob("*.nc") if path != old_trajectory
    ]
    assert len(orphan_candidates) == 1

    monkeypatch.setattr(runner, "replace_index_row_atomic", original_replace)
    runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    new_row = _rows(index)[0]
    new_trajectory = output / new_row["trajectory_path"]

    assert new_row["run_id"] == old_row["run_id"]
    assert new_trajectory.is_file()
    assert new_trajectory != old_trajectory
    assert not old_trajectory.exists()
    # Recovery removes the orphan before execution. The deterministic rerun then
    # recreates the same content-addressed path and commits it as the new result.
    assert orphan_candidates[0] == new_trajectory
    assert len(list((output / "trajectories").glob("*.nc"))) == 1


def test_failed_replacement_commits_failure_before_old_trajectory_cleanup(tmp_path, monkeypatch) -> None:
    scan = _write_protocol(tmp_path)
    output = tmp_path / "output"
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    old_row = _rows(index)[0]
    old_trajectory = output / old_row["trajectory_path"]

    def failed_integration(point):
        raise RuntimeError(f"Background integration failed for {point.identity.run_id}: injected")

    monkeypatch.setattr(runner, "_integrate_point", failed_integration)
    runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    failed_row = _rows(index)[0]

    assert failed_row["status"] == "failed"
    assert failed_row["failure_class"] == "solver_failure"
    assert failed_row["trajectory_path"] == ""
    assert not old_trajectory.exists()
