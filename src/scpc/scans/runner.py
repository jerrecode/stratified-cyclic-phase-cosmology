"""Configuration-driven serial parameter scans for the SCPC background.

This first Stage 1 runner prioritizes determinism, auditability, and complete
failure records over parallel throughput. Every planned experiment is assigned
a stable identity before execution. The index is atomically rewritten after
each completed attempt so interrupted scans retain prior outcomes.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from scpc.models.phase import PeriodicPotential, SCPCParameters, integrate_scpc
from scpc.numerics.provenance import (
    build_output_inventory,
    build_provenance,
    sha256_file,
    write_provenance,
)
from scpc.scans.config import DEFAULT_SCAN_SCHEMA, validate_scan_config
from scpc.scans.grid import ScanPoint, expand_parameter_grid
from scpc.scans.outcomes import assess_solution
from scpc.scans.records import completed_run_record, failed_run_record
from scpc.visualization.scans import plot_scan_outcome_map


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
    )


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty scan index")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("All scan-index rows must have identical ordered fields")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_netcdf_atomic(dataset, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        dataset.to_netcdf(temporary, engine="scipy")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        "trajectory_count": sum(bool(row.get("trajectory_path")) for row in rows),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _configured_outcome_map(scan: dict[str, Any], output: Path, index_path: Path) -> Path | None:
    visualization = scan.get("visualization")
    if visualization is None:
        return None
    filename = Path(str(visualization.get("outcome_map", "outcome_map.png")))
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("visualization.outcome_map must remain inside the scan output directory")
    destination = output / filename
    return plot_scan_outcome_map(
        index_path,
        destination,
        x_axis=str(visualization["x_axis"]),
        y_axis=str(visualization["y_axis"]),
        annotate=bool(visualization.get("annotate", True)),
    )


def run_background_scan(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCAN_SCHEMA,
) -> Path:
    """Execute or resume one deterministic serial background scan."""

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

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "scan_index.csv"
    summary_path = output / "scan_summary.json"
    metadata_path = output / "scan_metadata.json"
    trajectories_dir = output / "trajectories"

    planned_hashes = [point.identity.sha256 for point in points]
    metadata = {
        "metadata_schema_version": 1,
        "scan_schema_version": 1,
        "scan_schema_reference": schema_file.name,
        "scan_schema_sha256": sha256_file(schema_file),
        "scan_config_reference": scan_path.name,
        "scan_config_sha256": sha256_file(scan_path),
        "base_config_reference": base_reference,
        "base_config_sha256": sha256_file(base_path),
        "planned_run_sha256": planned_hashes,
    }
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError(
                "Existing scan metadata does not match this configuration; use a new output directory"
            )
    else:
        _write_json_atomic(metadata_path, metadata)

    rows = _read_existing_rows(index_path)
    rows_by_id = {row["run_id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("Existing scan index contains duplicate run IDs")

    resume = bool(scan.get("resume", True))
    rerun_statuses = set(str(status) for status in scan.get("rerun_statuses", []))
    if rows and not resume:
        raise ValueError("The output directory already contains a scan index and resume is disabled")

    classification = scan.get("classification", {})
    constraint_threshold = float(classification.get("constraint_threshold", 1.0e-7))
    hubble_zero_tolerance = float(classification.get("hubble_zero_tolerance", 1.0e-10))
    return_tolerance = float(classification.get("return_tolerance", 1.0e-3))

    retention = scan.get("retention", {})
    retained_outcomes = set(str(value) for value in retention.get("outcomes", []))
    max_trajectories = int(retention.get("max_trajectories", 0))
    saved_trajectories = sum(bool(row.get("trajectory_path")) for row in rows)

    for point in points:
        existing = rows_by_id.get(point.identity.run_id)
        if existing is not None and existing["status"] not in rerun_statuses:
            continue
        if existing is not None:
            rows.remove(existing)
            rows_by_id.pop(point.identity.run_id)

        trajectory_path: Path | None = None
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
                and saved_trajectories < max_trajectories
            ):
                trajectories_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = trajectories_dir / f"{point.identity.run_id}.nc"
                _write_netcdf_atomic(solution.to_xarray(), trajectory_path)
                saved_trajectories += 1
            record = completed_run_record(
                point.identity,
                point.specification,
                assessment,
                solution,
                coordinates=point.coordinates,
                trajectory_path=(
                    trajectory_path.relative_to(output) if trajectory_path is not None else None
                ),
            )
        except Exception as error:  # every planned run must remain represented
            if trajectory_path is not None:
                trajectory_path.unlink(missing_ok=True)
            record = failed_run_record(
                point.identity,
                point.specification,
                error,
                coordinates=point.coordinates,
            )

        row = record.to_flat_row()
        rows.append(row)
        rows.sort(key=lambda item: item["run_id"])
        rows_by_id[record.run_id] = row
        _write_csv_atomic(index_path, rows)
        _write_json_atomic(summary_path, _summary(rows, len(points)))

    if not rows:
        raise RuntimeError("The scan produced no index records")
    if not summary_path.exists():
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
        },
    )
    provenance["outputs"] = build_output_inventory(output_files, relative_to=output)
    write_provenance(output / "provenance.json", provenance)
    return index_path
