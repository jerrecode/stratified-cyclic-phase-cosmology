"""Integrity checks for resuming partially completed parameter scans."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scpc.models.phase import (
    DOMAIN_TERMINATION_KINDS,
    PeriodicPotential,
    SCPCIntegrationDomain,
    SCPCParameters,
    evaluate_domain_boundary,
)
from scpc.scans.grid import ScanPoint
from scpc.scans.records import RunStatus


_DOMAIN_FIELD_TO_KIND = {
    "min_scale_factor": "minimum_scale_factor",
    "max_scale_factor": "maximum_scale_factor",
    "max_total_density": "maximum_total_density",
    "max_abs_hubble": "maximum_absolute_hubble",
    "max_abs_ricci_scalar": "maximum_absolute_ricci_scalar",
    "max_abs_field": "maximum_absolute_field",
    "max_abs_field_velocity": "maximum_absolute_field_velocity",
}
_TERMINATED_REJECTION_OUTCOMES = {
    "physical_domain_termination",
    "constraint_violation",
    "degenerate_turning_event",
    "unresolved_event_detection",
}


def _trajectory_file(output: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe trajectory path in scan index: {raw_path!r}")
    resolved = (output / relative).resolve()
    if not resolved.is_relative_to(output.resolve()):
        raise ValueError(f"Trajectory path escapes the scan output directory: {raw_path!r}")
    return resolved


def _json_field(row: dict[str, Any], key: str, run_id: str) -> Any:
    try:
        return json.loads(row[key])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {key} JSON in existing scan row {run_id}") from error


def _boolean_field(row: dict[str, Any], key: str, run_id: str) -> bool | None:
    raw = row.get(key, "")
    if raw == "":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise ValueError(f"Invalid {key} boolean in existing scan row {run_id}: {raw!r}")


def _finite_float(row: dict[str, Any], key: str, run_id: str, *, positive: bool = False) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {key} in existing scan row {run_id}: {raw!r}") from error
    if not np.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{key} must be {qualifier} in existing scan row {run_id}")
    return value


def _domain_tolerance(rtol: float, atol: float, scale: float) -> float:
    magnitude = abs(float(scale))
    roundoff = 64.0 * np.finfo(float).eps * max(magnitude, np.finfo(float).tiny)
    return roundoff + 128.0 * (abs(float(atol)) + abs(float(rtol)) * magnitude)


def _parameters(specification: dict[str, Any]) -> SCPCParameters:
    model = specification["model"]
    return SCPCParameters(
        potential=PeriodicPotential(**model["potential"]),
        **model["background"],
    )


def _configured_domain(specification: dict[str, Any]) -> SCPCIntegrationDomain | None:
    raw = specification["run"].get("domain")
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Planned run.domain must be a mapping")
    domain = SCPCIntegrationDomain(**raw)
    return domain if domain.configured else None


def _canonical_domain(domain: SCPCIntegrationDomain | None) -> str:
    return json.dumps(
        asdict(domain) if domain is not None else {},
        sort_keys=True,
        separators=(",", ":"),
    )


def _constraint_residual(state: np.ndarray, parameters: SCPCParameters) -> float:
    a, hubble, field, field_velocity = (float(value) for value in state)
    matter = float(parameters.matter_density(a))
    radiation = float(parameters.radiation_density(a))
    field_density = 0.5 * field_velocity**2 + float(parameters.potential.value(field))
    lhs = hubble**2 + parameters.spatial_curvature_k / a**2
    rhs = (matter + radiation + field_density) / 3.0
    scale = max(abs(rhs), 1.0e-15)
    return float((lhs - rhs) / scale)


def _expected_boundaries(
    domain: SCPCIntegrationDomain,
    state: np.ndarray,
    parameters: SCPCParameters,
    *,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], ...]:
    expected: list[dict[str, Any]] = []
    for field, raw_threshold in asdict(domain).items():
        if raw_threshold is None:
            continue
        kind = _DOMAIN_FIELD_TO_KIND[field]
        threshold = float(raw_threshold)
        observed, units = evaluate_domain_boundary(kind, state, parameters)
        tolerance = _domain_tolerance(rtol, atol, max(abs(observed), abs(threshold)))
        if abs(observed - threshold) <= tolerance:
            expected.append(
                {
                    "kind": kind,
                    "threshold": threshold,
                    "observed": observed,
                    "units": units,
                }
            )
    if not expected:
        raise ValueError("Existing terminated row has no configured boundary at its exact state")
    return tuple(sorted(expected, key=lambda item: str(item["kind"])))


def _validated_supplied_boundaries(
    raw_boundaries: Any,
    state: np.ndarray,
    parameters: SCPCParameters,
    *,
    rtol: float,
    atol: float,
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise ValueError(f"Terminated existing row {run_id} requires a nonempty boundary list")
    validated: list[dict[str, Any]] = []
    for raw in raw_boundaries:
        if not isinstance(raw, dict) or set(raw) != {"kind", "threshold", "observed", "units"}:
            raise ValueError(f"Malformed termination boundary in existing row {run_id}")
        kind = str(raw["kind"])
        if kind not in DOMAIN_TERMINATION_KINDS:
            raise ValueError(f"Unknown termination boundary {kind!r} in existing row {run_id}")
        try:
            threshold = float(raw["threshold"])
            recorded_observed = float(raw["observed"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Nonnumeric termination boundary in existing row {run_id}") from error
        if (
            not np.isfinite(threshold)
            or threshold <= 0.0
            or not np.isfinite(recorded_observed)
            or recorded_observed <= 0.0
        ):
            raise ValueError(f"Nonfinite termination boundary in existing row {run_id}")
        observed, units = evaluate_domain_boundary(kind, state, parameters)
        tolerance = _domain_tolerance(rtol, atol, max(abs(observed), abs(threshold)))
        if str(raw["units"]) != units:
            raise ValueError(f"Termination units mismatch in existing row {run_id}")
        if abs(recorded_observed - observed) > tolerance:
            raise ValueError(f"Termination observation mismatch in existing row {run_id}")
        if abs(observed - threshold) > tolerance:
            raise ValueError(f"Termination threshold is not reached in existing row {run_id}")
        validated.append(
            {
                "kind": kind,
                "threshold": threshold,
                "observed": recorded_observed,
                "units": units,
            }
        )
    kinds = tuple(item["kind"] for item in validated)
    if kinds != tuple(sorted(set(kinds))):
        raise ValueError(f"Termination boundaries are not unique and ordered in row {run_id}")
    return tuple(validated)


def _validate_termination_row(row: dict[str, Any], point: ScanPoint) -> None:
    run_id = point.identity.run_id
    boundaries_raw = _json_field(row, "termination_boundaries", run_id)
    state_raw = _json_field(row, "termination_state_vector", run_id)
    scalar_keys = (
        "termination_kind",
        "termination_time",
        "termination_constraint_residual",
        "termination_threshold",
        "termination_observed",
        "termination_units",
    )
    has_scalar = any(row.get(key, "") != "" for key in scalar_keys)
    is_terminated = bool(boundaries_raw) or state_raw is not None or has_scalar

    if not is_terminated:
        if boundaries_raw != [] or state_raw is not None:
            raise ValueError(f"Partial termination JSON in existing row {run_id}")
        if row.get("outcome") == "physical_domain_termination":
            raise ValueError(f"Existing row {run_id} has termination outcome without metadata")
        if has_scalar:
            raise ValueError(f"Existing row {run_id} has partial termination scalars")
        return

    if row.get("status") != "rejected":
        raise ValueError(f"Domain-terminated existing row {run_id} must be rejected")
    if row.get("outcome") not in _TERMINATED_REJECTION_OUTCOMES:
        raise ValueError(
            f"Terminated existing row {run_id} has incompatible rejection outcome"
        )
    if _boolean_field(row, "numerically_valid", run_id) is not False:
        raise ValueError(f"Domain-terminated existing row {run_id} must be numerically invalid")
    if _boolean_field(row, "completed_to_requested_end", run_id) is not False:
        raise ValueError(f"Domain-terminated existing row {run_id} cannot be endpoint-complete")
    if row.get("failure_class", ""):
        raise ValueError(f"Domain-terminated existing row {run_id} cannot have a failure class")
    if row.get("trajectory_path", ""):
        raise ValueError(f"Domain-terminated existing row {run_id} cannot retain a candidate trajectory")

    if not isinstance(state_raw, list) or len(state_raw) != 4:
        raise ValueError(f"Termination state must have four components in existing row {run_id}")
    try:
        state = np.asarray([float(value) for value in state_raw], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid termination state in existing row {run_id}") from error
    if np.any(~np.isfinite(state)):
        raise ValueError(f"Nonfinite termination state in existing row {run_id}")

    termination_time = _finite_float(row, "termination_time", run_id)
    termination_constraint = _finite_float(
        row,
        "termination_constraint_residual",
        run_id,
    )
    primary_threshold = _finite_float(row, "termination_threshold", run_id, positive=True)
    primary_observed = _finite_float(row, "termination_observed", run_id, positive=True)
    primary_kind = row.get("termination_kind", "")
    primary_units = row.get("termination_units", "")
    if not primary_kind or not primary_units:
        raise ValueError(f"Existing row {run_id} has incomplete primary termination labels")

    run = point.specification["run"]
    t_start = float(run["t_start"])
    t_end = float(run["t_end"])
    interval_slack = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(t_start),
        abs(t_end),
        abs(termination_time),
    )
    if (
        termination_time < min(t_start, t_end) - interval_slack
        or termination_time > max(t_start, t_end) + interval_slack
    ):
        raise ValueError(f"Termination time lies outside the planned interval for row {run_id}")

    metadata = _json_field(row, "solver_metadata", run_id)
    if not isinstance(metadata, dict):
        raise ValueError(f"Terminated existing row {run_id} requires solver metadata")
    domain = _configured_domain(point.specification)
    if domain is None:
        raise ValueError(f"Terminated existing row {run_id} has no planned domain")
    parameters = _parameters(point.specification)
    domain.validate_for(parameters)
    if metadata.get("integration_domain") != _canonical_domain(domain):
        raise ValueError(f"Integration-domain metadata mismatch in existing row {run_id}")

    try:
        rtol = float(metadata["solver_rtol"])
        atol = float(metadata["solver_atol"])
        requested_end = float(metadata["requested_end_time"])
        reached_end = float(metadata["reached_end_time"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Incomplete solver metadata in existing row {run_id}") from error
    if (
        not np.isfinite(rtol)
        or rtol <= 0.0
        or not np.isfinite(atol)
        or atol <= 0.0
        or not np.isfinite(requested_end)
        or not np.isfinite(reached_end)
    ):
        raise ValueError(f"Nonfinite solver metadata in existing row {run_id}")
    if rtol != float(run["rtol"]) or atol != float(run["atol"]):
        raise ValueError(f"Solver tolerance mismatch in existing row {run_id}")
    if requested_end != t_end:
        raise ValueError(f"Requested endpoint mismatch in existing row {run_id}")
    time_tolerance = _domain_tolerance(rtol, atol, max(abs(reached_end), abs(termination_time)))
    if abs(reached_end - termination_time) > time_tolerance:
        raise ValueError(f"Reached endpoint mismatch in existing row {run_id}")

    recomputed_constraint = _constraint_residual(state, parameters)
    constraint_tolerance = _domain_tolerance(
        rtol,
        atol,
        max(abs(recomputed_constraint), abs(termination_constraint)),
    )
    if abs(recomputed_constraint - termination_constraint) > constraint_tolerance:
        raise ValueError(f"Termination constraint residual mismatch in existing row {run_id}")
    max_constraint = _finite_float(row, "max_abs_constraint_residual", run_id)
    if max_constraint < 0.0:
        raise ValueError(f"Negative maximum constraint residual in existing row {run_id}")
    if abs(recomputed_constraint) > max_constraint + constraint_tolerance:
        raise ValueError(f"Terminal constraint exceeds durable maximum in existing row {run_id}")

    supplied = _validated_supplied_boundaries(
        boundaries_raw,
        state,
        parameters,
        rtol=rtol,
        atol=atol,
        run_id=run_id,
    )
    expected = _expected_boundaries(
        domain,
        state,
        parameters,
        rtol=rtol,
        atol=atol,
    )
    if tuple(item["kind"] for item in supplied) != tuple(item["kind"] for item in expected):
        raise ValueError(f"Incomplete configured termination boundary set in existing row {run_id}")
    for existing, configured in zip(supplied, expected, strict=True):
        tolerance = _domain_tolerance(
            rtol,
            atol,
            max(abs(configured["observed"]), abs(configured["threshold"])),
        )
        if existing["units"] != configured["units"]:
            raise ValueError(f"Configured termination units mismatch in existing row {run_id}")
        if abs(existing["threshold"] - configured["threshold"]) > tolerance:
            raise ValueError(f"Configured termination threshold mismatch in existing row {run_id}")

    primary = supplied[0]
    primary_tolerance = _domain_tolerance(
        rtol,
        atol,
        max(abs(primary["observed"]), abs(primary["threshold"])),
    )
    if primary_kind != primary["kind"] or primary_units != primary["units"]:
        raise ValueError(f"Primary termination labels mismatch in existing row {run_id}")
    if abs(primary_threshold - primary["threshold"]) > primary_tolerance:
        raise ValueError(f"Primary termination threshold mismatch in existing row {run_id}")
    if abs(primary_observed - primary["observed"]) > primary_tolerance:
        raise ValueError(f"Primary termination observation mismatch in existing row {run_id}")


def load_existing_scan_rows(
    index_path: Path,
    output: Path,
    points: tuple[ScanPoint, ...],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Load and validate an existing index against the planned experiments."""

    if not index_path.exists():
        return [], {}, 0
    if not resume:
        raise ValueError("The output directory already contains a scan index and resume is disabled")

    with index_path.open("r", newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    points_by_id = {point.identity.run_id: point for point in points}
    rows_by_id: dict[str, dict[str, Any]] = {}
    saved_trajectories = 0
    valid_statuses = {status.value for status in RunStatus}

    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id or run_id in rows_by_id:
            raise ValueError(
                f"Existing scan index contains a missing or duplicate run ID: {run_id!r}"
            )
        point = points_by_id.get(run_id)
        if point is None:
            raise ValueError(f"Existing scan index contains an unplanned run ID: {run_id}")
        if row.get("run_sha256") != point.identity.sha256:
            raise ValueError(f"Run hash mismatch for existing scan row {run_id}")
        if row.get("status") not in valid_statuses:
            raise ValueError(
                f"Unknown run status in existing scan row {run_id}: {row.get('status')!r}"
            )

        try:
            coordinates = json.loads(row["coordinates"])
            specification = json.loads(row["specification"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSON fields in existing scan row {run_id}") from error
        if coordinates != point.coordinates:
            raise ValueError(f"Scan-coordinate mismatch for existing row {run_id}")
        if specification != point.specification:
            raise ValueError(f"Run-specification mismatch for existing row {run_id}")

        _validate_termination_row(row, point)

        trajectory_path = row.get("trajectory_path", "")
        if trajectory_path:
            trajectory = _trajectory_file(output, trajectory_path)
            if not trajectory.is_file():
                raise ValueError(
                    f"Existing scan row {run_id} references a missing trajectory: "
                    f"{trajectory_path}"
                )
            saved_trajectories += 1
        rows_by_id[run_id] = row

    return rows, rows_by_id, saved_trajectories
