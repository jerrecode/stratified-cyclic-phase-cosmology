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
from scpc.scans.grid import ScanPoint, expand_parameter_grid
from scpc.scans.outcomes import assess_solution
from scpc.scans.records import RunRecord, completed_run_record, failed_run_record


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


def run_background_scan(config_path: str | Path, output_dir: str | Path) -> Path:
    """Execute or resume one deterministic serial background scan."""

    scan_path = Path(config_path).resolve()
    scan = _load_mapping(scan_path)
    base_path = _resolve_relative(scan_path, scan["base_config"])
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
        "scan_config": str(scan_path),
        "scan_config_sha256": sha256_file(scan_path),
        "base_config": str(base_path),
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
    if max_trajectories < 0:
        raise ValueError("retention.max_trajectories must be nonnegative")
    saved_trajectories = sum(bool(row.get("trajectory_path")) for row in rows)

    for point in points:
        existing = rows_by_id.get(point.identity.run_id)
        if existing is not None and existing["status"] not in rerun_statuses:
            continue
        if existing is not None:
            rows.remove(existing)
            rows_by_id.pop(point.identity.run_id)

        try:
            solution = _integrate_point(point)
            assessment = assess_solution(
                solution,
                constraint_threshold=constraint_threshold,
                hubble_zero_tolerance=hubble_zero_tolerance,
                return_tolerance=return_tolerance,
            )
            trajectory_path: Path | None = None
            if (
                assessment.numerically_valid
                and assessment.outcome.value in retained_outcomes
                and saved_trajectories < max_trajectories
            ):
                trajectories_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = trajectories_dir / f"{point.identity.run_id}.nc"
                solution.to_xarray().to_netcdf(trajectory_path, engine="scipy")
                saved_trajectories += 1
            record = completed_run_record(
                point.identity,
                point.specification,
                assessment,
                solution,
                trajectory_path=(
                    trajectory_path.relative_to(output) if trajectory_path is not None else None
                ),
            )
        except Exception as error:  # every planned run must remain represented
            record = failed_run_record(point.identity, point.specification, error)

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

    output_files = [index_path, summary_path, metadata_path]
    if trajectories_dir.exists():
        output_files.extend(sorted(trajectories_dir.glob("*.nc")))
    provenance = build_provenance(
        scan_path,
        {
            "workflow": "run_background_scan",
            "base_config": str(base_path),
            "base_config_sha256": sha256_file(base_path),
        },
    )
    provenance["outputs"] = build_output_inventory(output_files, relative_to=output)
    write_provenance(output / "provenance.json", provenance)
    return index_path
