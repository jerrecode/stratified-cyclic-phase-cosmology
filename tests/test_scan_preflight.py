from pathlib import Path

import pytest
import yaml

import scpc.scans.runner as runner
from scpc.visualization.scans import validate_outcome_map_coordinates


SCAN_SCHEMA = Path("configs/scans/scan.schema.json")
SMOKE_CONFIG = Path("configs/scans/stage1_smoke.yaml")
SMOKE_BASE = Path("configs/scans/stage1_smoke_base.yaml")


def test_outcome_map_requires_exactly_two_axes() -> None:
    coordinates = [
        {"x": 0.0, "y": 0.0, "hidden": 1.0},
        {"x": 1.0, "y": 0.0, "hidden": 1.0},
        {"x": 0.0, "y": 1.0, "hidden": 1.0},
        {"x": 1.0, "y": 1.0, "hidden": 1.0},
    ]
    with pytest.raises(ValueError, match="exactly the two configured scan axes"):
        validate_outcome_map_coordinates(coordinates, x_axis="x", y_axis="y")


def test_outcome_map_rejects_effectively_one_dimensional_grid() -> None:
    coordinates = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
    ]
    with pytest.raises(ValueError, match="genuinely two-dimensional"):
        validate_outcome_map_coordinates(coordinates, x_axis="x", y_axis="y")


def test_invalid_map_axes_fail_before_output_creation_or_integration(tmp_path, monkeypatch) -> None:
    protocol = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    protocol["visualization"]["x_axis"] = "model.potential.misspelled"
    config_path = tmp_path / "scan.yaml"
    config_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    (tmp_path / "stage1_smoke_base.yaml").write_text(
        SMOKE_BASE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    executed = False

    def forbidden_integration(point):
        nonlocal executed
        executed = True
        raise AssertionError(f"Unexpected integration of {point.identity.run_id}")

    monkeypatch.setattr(runner, "_integrate_point", forbidden_integration)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="exactly the two configured scan axes"):
        runner.run_background_scan(config_path, output, schema_path=SCAN_SCHEMA)

    assert executed is False
    assert not output.exists()
