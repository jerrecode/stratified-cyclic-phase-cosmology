import json

import jsonschema
import pytest
import yaml

from scpc.scans.config import validate_scan_config


SCHEMA = "configs/scans/scan.schema.json"
VALID_CONFIG = "configs/scans/stage1_smoke.yaml"


def _load_valid() -> dict:
    with open(VALID_CONFIG, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_committed_smoke_protocol_validates() -> None:
    config = validate_scan_config(VALID_CONFIG, SCHEMA)
    assert config["schema_version"] == 1
    assert config["max_runs"] == 4
    assert set(config["axes"]) == {
        "model.potential.amplitude",
        "model.potential.offset",
    }


def test_schema_itself_is_draft_2020_12_valid() -> None:
    with open(SCHEMA, encoding="utf-8") as handle:
        schema = json.load(handle)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_unknown_protocol_fields_are_rejected(tmp_path) -> None:
    config = _load_valid()
    config["invented_control"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        validate_scan_config(path, SCHEMA)


def test_invalid_axis_path_and_empty_values_are_rejected(tmp_path) -> None:
    config = _load_valid()
    config["axes"] = {"invalid": []}
    path = tmp_path / "invalid-axis.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_scan_config(path, SCHEMA)


def test_negative_retention_limit_is_rejected(tmp_path) -> None:
    config = _load_valid()
    config["retention"]["max_trajectories"] = -1
    path = tmp_path / "negative-retention.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError, match="minimum of 0"):
        validate_scan_config(path, SCHEMA)


def test_unknown_retention_outcome_is_rejected(tmp_path) -> None:
    config = _load_valid()
    config["retention"]["outcomes"] = ["monotonic_expansoin"]
    path = tmp_path / "unknown-outcome.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_scan_config(path, SCHEMA)


def test_nonfinite_protocol_numbers_are_rejected(tmp_path) -> None:
    config = _load_valid()
    config["classification"]["constraint_threshold"] = float("nan")
    path = tmp_path / "nan-threshold.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_scan_config(path, SCHEMA)
