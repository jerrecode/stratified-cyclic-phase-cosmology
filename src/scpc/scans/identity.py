"""Canonical identities for parameter-space experiments."""

from __future__ import annotations

import copy
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


_EXECUTION_TYPES: dict[str, type] = {
    "model.background.spatial_curvature_k": int,
    "model.background.rho_m_ref": float,
    "model.background.rho_r_ref": float,
    "model.background.a_ref": float,
    "model.potential.offset": float,
    "model.potential.amplitude": float,
    "model.potential.strata_count": int,
    "model.potential.field_scale": float,
    "model.potential.target_space": str,
    "initial_conditions.a": float,
    "initial_conditions.phi": float,
    "initial_conditions.phi_dot": float,
    "initial_conditions.branch": int,
    "run.t_start": float,
    "run.t_end": float,
    "run.samples": int,
    "run.method": str,
    "run.rtol": float,
    "run.atol": float,
    "run.domain.min_scale_factor": float,
    "run.domain.max_scale_factor": float,
    "run.domain.max_total_density": float,
    "run.domain.max_abs_hubble": float,
    "run.domain.max_abs_ricci_scalar": float,
    "run.domain.max_abs_field": float,
    "run.domain.max_abs_field_velocity": float,
}


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(path.split("."))


def _get_existing_path(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in _path_parts(path):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _set_existing_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = _path_parts(path)
    current: dict[str, Any] = document
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise KeyError(path)
        current = child
    if parts[-1] not in current:
        raise KeyError(path)
    current[parts[-1]] = value


def _execution_cast(path: str, value: Any, expected: type) -> Any:
    if expected is int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{path} must be an integer, not boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{path} must be an integer") from error
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{path} must be a finite integer")
        return int(number)
    if expected is float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{path} must be numeric, not boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{path} must be numeric") from error
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite")
        return number
    if expected is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        return value
    raise TypeError(f"Unsupported execution type for {path}: {expected.__name__}")


def normalize_background_specification(specification: dict[str, Any]) -> dict[str, Any]:
    """Return the values exactly as the background executor will interpret them.

    Known numerical and string fields are normalized before run hashing. This
    prevents representations such as ``101`` and ``101.0`` from scheduling two
    experiments that execute identically. Unknown extension fields are retained
    unchanged and remain part of the identity.
    """

    normalized = copy.deepcopy(specification)
    for path, expected in _EXECUTION_TYPES.items():
        try:
            value = _get_existing_path(normalized, path)
        except KeyError:
            continue
        if value is None:
            continue
        _set_existing_path(normalized, path, _execution_cast(path, value, expected))

    run = normalized.get("run")
    if isinstance(run, dict) and "domain" in run:
        domain = run["domain"]
        if domain is None or domain == {}:
            run.pop("domain")
        elif not isinstance(domain, dict):
            raise ValueError("run.domain must be a mapping when configured")
    return normalized


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
    """Hash a full, execution-normalized run specification."""

    if not namespace:
        raise ValueError("namespace must not be empty")
    if prefix_length < 12 or prefix_length > 64:
        raise ValueError("prefix_length must be between 12 and 64")
    normalized = {
        "namespace": namespace,
        "specification": _normalize(normalize_background_specification(specification)),
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
