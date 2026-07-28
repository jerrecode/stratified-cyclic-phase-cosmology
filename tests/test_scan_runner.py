import csv
import json
from pathlib import Path

from scpc.scans.runner import run_background_scan


SMOKE_CONFIG = Path("configs/scans/stage1_smoke.yaml")


def _rows(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_smoke_scan_preserves_successes_and_expected_failures(tmp_path) -> None:
    output = tmp_path / "scan"
    index_path = run_background_scan(SMOKE_CONFIG, output)
    rows = _rows(index_path)

    assert len(rows) == 4
    assert len({row["run_id"] for row in rows}) == 4
    assert {row["status"] for row in rows} == {"completed", "failed"}
    assert {tuple(sorted(json.loads(row["coordinates"]).items())) for row in rows} == {
        (
            ("model.potential.amplitude", 0.0),
            ("model.potential.offset", 0.5),
        ),
        (
            ("model.potential.amplitude", 0.0),
            ("model.potential.offset", 3.5),
        ),
        (
            ("model.potential.amplitude", 0.08),
            ("model.potential.offset", 0.5),
        ),
        (
            ("model.potential.amplitude", 0.08),
            ("model.potential.offset", 3.5),
        ),
    }

    failures = [row for row in rows if row["status"] == "failed"]
    completed = [row for row in rows if row["status"] == "completed"]
    assert len(failures) == 2
    assert len(completed) == 2
    assert {row["failure_class"] for row in failures} == {"invalid_initial_constraint"}
    assert all(row["outcome"] == "monotonic_expansion" for row in completed)
    assert all("singular" not in row["reason"].lower() for row in failures)

    retained = [row["trajectory_path"] for row in completed if row["trajectory_path"]]
    assert len(retained) == 2
    assert all((output / path).is_file() for path in retained)
    assert (output / "outcome_map.png").stat().st_size > 0

    summary = json.loads((output / "scan_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_runs"] == 4
    assert summary["indexed_runs"] == 4
    assert summary["status_counts"] == {"completed": 2, "failed": 2}
    assert summary["failure_class_counts"] == {"invalid_initial_constraint": 2}
    assert summary["trajectory_count"] == 2

    metadata = json.loads((output / "scan_metadata.json").read_text(encoding="utf-8"))
    assert metadata["metadata_schema_version"] == 2
    assert metadata["base_config_reference"] == "stage1_smoke_base.yaml"
    assert not Path(metadata["scan_config_reference"]).is_absolute()
    assert len(metadata["implementation_runtime_fingerprint"]["strict_sha256"]) == 64

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    inventoried_paths = {item["path"] for item in provenance["outputs"]}
    assert {
        "scan_index.csv",
        "scan_summary.json",
        "scan_metadata.json",
        "outcome_map.png",
    }.issubset(inventoried_paths)
    assert set(retained).issubset(inventoried_paths)


def test_resume_does_not_duplicate_completed_attempts(tmp_path) -> None:
    output = tmp_path / "scan"
    index_path = run_background_scan(SMOKE_CONFIG, output)
    first_rows = _rows(index_path)
    first_index_bytes = index_path.read_bytes()

    resumed_index = run_background_scan(SMOKE_CONFIG, output)
    second_rows = _rows(resumed_index)

    assert resumed_index == index_path
    assert second_rows == first_rows
    assert resumed_index.read_bytes() == first_index_bytes


def test_changed_scan_configuration_requires_new_output_directory(tmp_path) -> None:
    output = tmp_path / "scan"
    run_background_scan(SMOKE_CONFIG, output)

    changed_config = tmp_path / "changed.yaml"
    changed_config.write_text(
        SMOKE_CONFIG.read_text(encoding="utf-8").replace("max_runs: 4", "max_runs: 5"),
        encoding="utf-8",
    )
    base_source = Path("configs/scans/stage1_smoke_base.yaml")
    (tmp_path / "stage1_smoke_base.yaml").write_text(
        base_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    try:
        run_background_scan(changed_config, output)
    except ValueError as error:
        assert "metadata does not match" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Changed scan configuration unexpectedly reused the output directory")
