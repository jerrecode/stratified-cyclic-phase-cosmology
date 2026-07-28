"""Crash-safe primitives for replacing durable scan results."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from scpc.numerics.provenance import sha256_file


def safe_relative_output_path(output: Path, raw_path: str) -> Path:
    """Resolve one index path without permitting output-directory escape."""

    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe scan output path: {raw_path!r}")
    resolved = (output.resolve() / relative).resolve()
    if not resolved.is_relative_to(output.resolve()):
        raise ValueError(f"Scan output path escapes its root: {raw_path!r}")
    return resolved


def write_content_addressed_netcdf(dataset: Any, directory: Path, run_id: str) -> Path:
    """Write a candidate trajectory without replacing a currently indexed file.

    The final filename contains a prefix of the file checksum. A process crash
    before index replacement can therefore leave only an unreferenced file,
    never a missing file behind the still-durable old index row.
    """

    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id contains unsafe filename characters")
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / f".{run_id}.pending-{os.getpid()}.nc"
    try:
        dataset.to_netcdf(pending, engine="scipy")
        checksum = sha256_file(pending)
        destination = directory / f"{run_id}-{checksum[:20]}.nc"
        if destination.exists():
            if sha256_file(destination) != checksum:
                raise RuntimeError(f"Content-address collision for {destination.name}")
            pending.unlink()
        else:
            pending.replace(destination)
        return destination
    except Exception:
        pending.unlink(missing_ok=True)
        raise


def replace_index_row_atomic(
    index_path: Path,
    rows: list[dict[str, Any]],
    new_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Atomically replace or append one run row and return the committed rows."""

    run_id = str(new_row.get("run_id", ""))
    if not run_id:
        raise ValueError("new_row must contain a nonempty run_id")
    fieldnames = list(new_row)
    retained = [row for row in rows if row.get("run_id") != run_id]
    if len(retained) != len(rows) - sum(row.get("run_id") == run_id for row in rows):
        raise RuntimeError("Unexpected index filtering result")
    committed = sorted([*retained, new_row], key=lambda row: str(row["run_id"]))
    if any(list(row) != fieldnames for row in committed):
        raise ValueError("All scan-index rows must have identical ordered fields")

    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(committed)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(index_path)
    return committed


def cleanup_replaced_trajectory(
    output: Path,
    old_row: dict[str, Any] | None,
    new_row: dict[str, Any],
) -> None:
    """Delete the old trajectory only after the new row is durably indexed."""

    if old_row is None:
        return
    old_raw = str(old_row.get("trajectory_path", ""))
    new_raw = str(new_row.get("trajectory_path", ""))
    if not old_raw or old_raw == new_raw:
        return
    safe_relative_output_path(output, old_raw).unlink(missing_ok=True)


def cleanup_unreferenced_transaction_files(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    trajectories_subdirectory: str = "trajectories",
) -> list[str]:
    """Remove pending or orphaned scan-owned trajectory files after validation."""

    directory = safe_relative_output_path(output, trajectories_subdirectory)
    if not directory.exists():
        return []
    referenced = {
        safe_relative_output_path(output, str(row["trajectory_path"]))
        for row in rows
        if row.get("trajectory_path")
    }
    removed: list[str] = []
    for path in sorted(directory.glob("*.nc")):
        if path.resolve() not in referenced:
            path.unlink()
            removed.append(str(path.relative_to(output)))
    for path in sorted(directory.glob(".*.pending-*.nc")):
        path.unlink()
        removed.append(str(path.relative_to(output)))
    return removed
