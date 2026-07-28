"""Loading and machine validation of background-scan protocols."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import yaml


DEFAULT_SCAN_SCHEMA = Path("configs/scans/scan.schema.json")


def _reject_nonfinite_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Scan configuration contains a nonfinite number at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_numbers(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_numbers(child, path=f"{path}[{index}]")


def validate_scan_config(
    config_path: str | Path,
    schema_path: str | Path = DEFAULT_SCAN_SCHEMA,
) -> dict[str, Any]:
    config_file = Path(config_path)
    schema_file = Path(schema_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with schema_file.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("Scan configuration root must be a mapping")
    _reject_nonfinite_numbers(config)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(config, schema)
    return config
