"""Canonical identities for parameter-space experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunIdentity:
    """Stable identity of one complete numerical experiment specification."""

    run_id: str
    sha256: str
    canonical_json: str


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Run identity payloads may not contain nonfinite floats")
        return {"__float_hex__": number.hex()}
    if isinstance(value, Path):
        return {"__path__": value.as_posix()}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Run identity mappings require string keys")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"Unsupported run identity value: {type(value).__name__}")


def canonical_run_identity(
    specification: dict[str, Any],
    *,
    namespace: str = "scpc-background-v1",
    prefix_length: int = 24,
) -> RunIdentity:
    """Hash a full run specification with stable float and key encoding."""

    if not namespace:
        raise ValueError("namespace must not be empty")
    if prefix_length < 12 or prefix_length > 64:
        raise ValueError("prefix_length must be between 12 and 64")
    normalized = {
        "namespace": namespace,
        "specification": _normalize(specification),
    }
    canonical_json = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return RunIdentity(
        run_id=f"scpc-{digest[:prefix_length]}",
        sha256=digest,
        canonical_json=canonical_json,
    )
