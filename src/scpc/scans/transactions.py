"""Crash-safe primitives for replacing durable scan results."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from scpc.numerics.provenance import sha256_file
from scpc.scans.durability import fsync_directory
from scpc.scans.errors import OutputSerializationError


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
    """Write a candidate trajectory without replacing a currently indexed file."""

    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not run_id or any(character not in safe_characters for character in run_id):
        raise ValueError("run_id contains unsafe filename characters")
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / f".{run_id}.pending-{os.getpid()}.nc"
    try:
        dataset.to_netcdf(pending, engine="scipy")
        with pending.open("rb") as handle:
            os.fsync(handle.fileno())
        checksum = sha256_file(pending)
        destination = directory / f"{run_id}-{checksum[:20]}.nc"
        if destination.exists():
            if sha256_file(destination) != checksum:
                raise OutputSerializationError(
                    f"Content-address collision for {destination.name}"
                )
            pending.unlink()
        else:
            pending.replace(destination)
        fsync_directory(directory)
        return destination
    except OutputSerializationError:
        pending.unlink(missing_ok=True)
        raise
    except Exception as error:
        pending.unlink(missing_ok=True)
        raise OutputSerializationError(
            f"Could not serialize trajectory for {run_id}: {error}"
        ) from error


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
    expected_count = len(rows) - sum(row.get("run_id") == run_id for row in rows)
    if len(retained) != expected_count:
        raise RuntimeError("Unexpected index filtering result")
    committed = sorted([*retained, new_row], key=lambda row: str(row["run_id"]))
    if any(list(row) != fieldnames for row in committed):
        raise ValueError("All scan-index rows must have identical ordered fields")

    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(committed)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(index_path)
        fsync_directory(index_path.parent)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if isinstance(error, OutputSerializationError):
            raise
        raise OutputSerializationError(
            f"Could not durably replace scan index {index_path}: {error}"
        ) from error
    return committed


def _cleanup_replaced_artifact(
    output: Path,
    old_row: dict[str, Any] | None,
    new_row: dict[str, Any],
    field: str,
) -> None:
    if old_row is None:
        return
    old_raw = str(old_row.get(field, ""))
    new_raw = str(new_row.get(field, ""))
    if not old_raw or old_raw == new_raw:
        return
    safe_relative_output_path(output, old_raw).unlink(missing_ok=True)


def cleanup_replaced_trajectory(
    output: Path,
    old_row: dict[str, Any] | None,
    new_row: dict[str, Any],
) -> None:
    """Delete the old trajectory only after the new row is durably indexed."""

    _cleanup_replaced_artifact(output, old_row, new_row, "trajectory_path")


def cleanup_replaced_termination_record(
    output: Path,
    old_row: dict[str, Any] | None,
    new_row: dict[str, Any],
) -> None:
    """Delete old termination evidence only after replacement is durably indexed."""

    _cleanup_replaced_artifact(output, old_row, new_row, "termination_record_path")


def _cleanup_unreferenced_directory(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    subdirectory: str,
    field: str,
    suffix: str,
) -> list[str]:
    directory = safe_relative_output_path(output, subdirectory)
    if not directory.exists():
        return []
    referenced = {
        safe_relative_output_path(output, str(row[field]))
        for row in rows
        if row.get(field)
    }
    removed: list[str] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        if path.resolve() not in referenced:
            path.unlink()
            removed.append(str(path.relative_to(output)))
    for path in sorted(directory.glob(f".*.pending-*{suffix}")):
        path.unlink()
        removed.append(str(path.relative_to(output)))
    return removed


def cleanup_unreferenced_transaction_files(
    output: Path,
    rows: list[dict[str, Any]],
) -> list[str]:
    """Remove pending or orphaned scan-owned artifacts after index validation."""

    removed = _cleanup_unreferenced_directory(
        output,
        rows,
        subdirectory="trajectories",
        field="trajectory_path",
        suffix=".nc",
    )
    removed.extend(
        _cleanup_unreferenced_directory(
            output,
            rows,
            subdirectory="termination_records",
            field="termination_record_path",
            suffix=".json",
        )
    )
    return sorted(removed)
