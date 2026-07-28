"""Integrity checks for resuming partially completed parameter scans."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scpc.scans.grid import ScanPoint
from scpc.scans.records import RunStatus


def _trajectory_file(output: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe trajectory path in scan index: {raw_path!r}")
    resolved = (output / relative).resolve()
    if not resolved.is_relative_to(output.resolve()):
        raise ValueError(f"Trajectory path escapes the scan output directory: {raw_path!r}")
    return resolved


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
            raise ValueError(f"Existing scan index contains a missing or duplicate run ID: {run_id!r}")
        point = points_by_id.get(run_id)
        if point is None:
            raise ValueError(f"Existing scan index contains an unplanned run ID: {run_id}")
        if row.get("run_sha256") != point.identity.sha256:
            raise ValueError(f"Run hash mismatch for existing scan row {run_id}")
        if row.get("status") not in valid_statuses:
            raise ValueError(f"Unknown run status in existing scan row {run_id}: {row.get('status')!r}")

        try:
            coordinates = json.loads(row["coordinates"])
            specification = json.loads(row["specification"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSON fields in existing scan row {run_id}") from error
        if coordinates != point.coordinates:
            raise ValueError(f"Scan-coordinate mismatch for existing row {run_id}")
        if specification != point.specification:
            raise ValueError(f"Run-specification mismatch for existing row {run_id}")

        trajectory_path = row.get("trajectory_path", "")
        if trajectory_path:
            trajectory = _trajectory_file(output, trajectory_path)
            if not trajectory.is_file():
                raise ValueError(
                    f"Existing scan row {run_id} references a missing trajectory: {trajectory_path}"
                )
            saved_trajectories += 1
        rows_by_id[run_id] = row

    return rows, rows_by_id, saved_trajectories


def remove_existing_run_for_rerun(
    run_id: str,
    rows: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    output: Path,
) -> bool:
    """Remove one old record and any retained trajectory before a rerun."""

    existing = rows_by_id.pop(run_id)
    rows.remove(existing)
    trajectory_path = existing.get("trajectory_path", "")
    if not trajectory_path:
        return False
    trajectory = _trajectory_file(output, trajectory_path)
    trajectory.unlink()
    return True
