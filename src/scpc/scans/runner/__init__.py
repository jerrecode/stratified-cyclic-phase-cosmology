"""Transactional deterministic background-scan execution.

This package preserves the previous durable result during reruns, binds resume
to the exact SCPC source and numerical runtime, and treats domain termination
and orphan cleanup as analysis records rather than proof of a physical outcome.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from scpc.models.phase import (
    PeriodicPotential,
    SCPCIntegrationDomain,
    SCPCParameters,
    integrate_scpc,
)
from scpc.numerics.provenance import (
    build_output_inventory,
    build_provenance,
    sha256_file,
    write_provenance,
)
from scpc.scans.config import DEFAULT_SCAN_SCHEMA, validate_scan_config
from scpc.scans.fingerprint import implementation_runtime_fingerprint
from scpc.scans.grid import ScanPoint, expand_parameter_grid
from scpc.scans.outcomes import assess_solution
from scpc.scans.records import completed_run_record, failed_run_record
from scpc.scans.resume import load_existing_scan_rows
from scpc.scans.transactions import (
    cleanup_replaced_trajectory,
    cleanup_unreferenced_transaction_files,
    replace_index_row_atomic,
    write_content_addressed_netcdf,
)
from scpc.visualization.scans import (
    plot_scan_outcome_map,
    validate_outcome_map_coordinates,
)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def _resolve_relative(owner: Path, referenced: str | Path) -> Path:
    path = Path(referenced)
    return path if path.is_absolute() else (owner.parent / path).resolve()


def _integrate_point(point: ScanPoint):
    specification = point.specification
    potential = PeriodicPotential(**specification["model"]["potential"])
    parameters = SCPCParameters(
        potential=potential,
        **specification["model"]["background"],
    )
    initial = specification["initial_conditions"]
    run = specification["run"]
    domain_config = run.get("domain")
    if domain_config is None:
        domain = None
    elif not isinstance(domain_config, dict):
        raise TypeError("run.domain must be a mapping when configured")
    else:
        domain = SCPCIntegrationDomain(**domain_config)
    return integrate_scpc(
        parameters,
        t_span=(float(run["t_start"]), float(run["t_end"])),
        samples=int(run["samples"]),
        method=str(run["method"]),
        rtol=float(run["rtol"]),
        atol=float(run["atol"]),
        a0=float(initial["a"]),
        phi0=float(initial["phi"]),
        phi_dot0=float(initial["phi_dot"]),
        branch=int(initial["branch"]),
        domain=domain,
    )


def _summary(rows: list[dict[str, Any]], planned_runs: int) -> dict[str, Any]:
    return {
        "planned_runs": planned_runs,
        "indexed_runs": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "outcome_counts": dict(
            sorted(Counter(row["outcome"] for row in rows if row.get("outcome")).items())
        ),
        "failure_class_counts": dict(
            sorted(
                Counter(
                    row["failure_class"]
                    for row in rows
                    if row.get("failure_class")
                ).items()
            )
        ),
        "termination_kind_counts": dict(
            sorted(
                Counter(
                    row["termination_kind"]
                    for row in rows
                    if row.get("termination_kind")
                ).items()
            )
        ),
        "trajectory_count": sum(bool(row.get("trajectory_path")) for row in rows),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _outcome_map_filename(scan: dict[str, Any]) -> Path | None:
    visualization = scan.get("visualization")
    if visualization is None:
        return None
    filename = Path(str(visualization.get("outcome_map", "outcome_map.png")))
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("visualization.outcome_map must remain inside the scan output directory")
    return filename


def _preflight_outcome_map(scan: dict[str, Any], points: tuple[ScanPoint, ...]) -> None:
    visualization = scan.get("visualization")
    if visualization is None:
        return
    _outcome_map_filename(scan)
    validate_outcome_map_coordinates(
        [point.coordinates for point in points],
        x_axis=str(visualization["x_axis"]),
        y_axis=str(visualization["y_axis"]),
    )


def _configured_outcome_map(scan: dict[str, Any], output: Path, index_path: Path) -> Path | None:
    visualization = scan.get("visualization")
    filename = _outcome_map_filename(scan)
    if visualization is None or filename is None:
        return None
    return plot_scan_outcome_map(
        index_path,
        output / filename,
        x_axis=str(visualization["x_axis"]),
        y_axis=str(visualization["y_axis"]),
        annotate=bool(visualization.get("annotate", True)),
    )


def _strict_fingerprint() -> tuple[dict[str, Any], dict[str, Any]]:
    complete = implementation_runtime_fingerprint()
    metadata = {key: value for key, value in complete.items() if key != "python_executable"}
    return metadata, complete


def run_background_scan(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCAN_SCHEMA,
) -> Path:
    """Execute or resume one strict, transactional serial background scan."""

    scan_path = Path(config_path).resolve()
    schema_file = Path(schema_path).resolve()
    scan = validate_scan_config(scan_path, schema_file)
    base_reference = str(scan["base_config"])
    base_path = _resolve_relative(scan_path, base_reference)
    base = _load_mapping(base_path)
    points = expand_parameter_grid(
        base,
        scan.get("axes", {}),
        max_runs=int(scan.get("max_runs", 10000)),
        identity_namespace=str(scan.get("identity_namespace", "scpc-background-v1")),
    )
    _preflight_outcome_map(scan, points)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "scan_index.csv"
    summary_path = output / "scan_summary.json"
    metadata_path = output / "scan_metadata.json"
    trajectories_dir = output / "trajectories"

    strict_fingerprint, complete_fingerprint = _strict_fingerprint()
    planned_hashes = [point.identity.sha256 for point in points]
    metadata = {
        "metadata_schema_version": 2,
        "scan_schema_version": 1,
        "scan_schema_reference": schema_file.name,
        "scan_schema_sha256": sha256_file(schema_file),
        "scan_config_reference": scan_path.name,
        "scan_config_sha256": sha256_file(scan_path),
        "base_config_reference": base_reference,
        "base_config_sha256": sha256_file(base_path),
        "implementation_runtime_fingerprint": strict_fingerprint,
        "planned_run_sha256": planned_hashes,
    }
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError(
                "Existing scan metadata does not match this configuration or runtime; "
                "use a new output directory"
            )
    else:
        if index_path.exists():
            raise ValueError("Existing scan index is missing its scan_metadata.json integrity record")
        _write_json_atomic(metadata_path, metadata)

    resume = bool(scan.get("resume", True))
    rows, rows_by_id, saved_trajectories = load_existing_scan_rows(
        index_path,
        output,
        points,
        resume=resume,
    )
    removed_recovery_files = cleanup_unreferenced_transaction_files(output, rows)
    rerun_statuses = set(str(status) for status in scan.get("rerun_statuses", []))

    classification = scan.get("classification", {})
    constraint_threshold = float(classification.get("constraint_threshold", 1.0e-7))
    hubble_zero_tolerance = float(classification.get("hubble_zero_tolerance", 1.0e-10))
    return_tolerance = float(classification.get("return_tolerance", 1.0e-3))

    retention = scan.get("retention", {})
    retained_outcomes = set(str(value) for value in retention.get("outcomes", []))
    max_trajectories = int(retention.get("max_trajectories", 0))

    for point in points:
        old_row = rows_by_id.get(point.identity.run_id)
        if old_row is not None and old_row["status"] not in rerun_statuses:
            continue

        old_has_trajectory = bool(old_row and old_row.get("trajectory_path"))
        effective_saved_trajectories = saved_trajectories - int(old_has_trajectory)
        candidate_trajectory: Path | None = None
        try:
            solution = _integrate_point(point)
            assessment = assess_solution(
                solution,
                constraint_threshold=constraint_threshold,
                hubble_zero_tolerance=hubble_zero_tolerance,
                return_tolerance=return_tolerance,
            )
            if (
                assessment.numerically_valid
                and assessment.outcome.value in retained_outcomes
                and effective_saved_trajectories < max_trajectories
            ):
                candidate_trajectory = write_content_addressed_netcdf(
                    solution.to_xarray(),
                    trajectories_dir,
                    point.identity.run_id,
                )
            record = completed_run_record(
                point.identity,
                point.specification,
                assessment,
                solution,
                coordinates=point.coordinates,
                trajectory_path=(
                    candidate_trajectory.relative_to(output)
                    if candidate_trajectory is not None
                    else None
                ),
            )
        except Exception as error:  # every planned run must remain represented
            if candidate_trajectory is not None and (
                old_row is None
                or str(candidate_trajectory.relative_to(output)) != old_row.get("trajectory_path", "")
            ):
                candidate_trajectory.unlink(missing_ok=True)
            record = failed_run_record(
                point.identity,
                point.specification,
                error,
                coordinates=point.coordinates,
            )

        new_row = record.to_flat_row()
        rows = replace_index_row_atomic(index_path, rows, new_row)
        rows_by_id[record.run_id] = new_row
        cleanup_replaced_trajectory(output, old_row, new_row)
        saved_trajectories = sum(bool(row.get("trajectory_path")) for row in rows)
        _write_json_atomic(summary_path, _summary(rows, len(points)))

    if not rows:
        raise RuntimeError("The scan produced no index records")
    _write_json_atomic(summary_path, _summary(rows, len(points)))

    outcome_map_path = _configured_outcome_map(scan, output, index_path)
    output_files = [index_path, summary_path, metadata_path]
    if outcome_map_path is not None:
        output_files.append(outcome_map_path)
    if trajectories_dir.exists():
        output_files.extend(sorted(trajectories_dir.glob("*.nc")))
    provenance = build_provenance(
        scan_path,
        {
            "workflow": "run_background_scan",
            "scan_schema": str(schema_file),
            "scan_schema_sha256": sha256_file(schema_file),
            "base_config": str(base_path),
            "base_config_sha256": sha256_file(base_path),
            "implementation_runtime_fingerprint": complete_fingerprint,
            "recovered_unreferenced_files": removed_recovery_files,
        },
    )
    provenance["outputs"] = build_output_inventory(output_files, relative_to=output)
    write_provenance(output / "provenance.json", provenance)
    return index_path


__all__ = ["run_background_scan"]
