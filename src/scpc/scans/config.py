"""Loading and machine validation of background-scan protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml


DEFAULT_SCAN_SCHEMA = Path("configs/scans/scan.schema.json")


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
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(config, schema)
    return config
