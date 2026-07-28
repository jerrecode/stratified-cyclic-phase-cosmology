import csv
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scpc.numerics.provenance import sha256_file
from scpc.scans.transactions import (
    cleanup_replaced_trajectory,
    cleanup_unreferenced_transaction_files,
    replace_index_row_atomic,
    safe_relative_output_path,
    write_content_addressed_netcdf,
)


FIELDNAMES = ["run_id", "status", "trajectory_path"]


def _row(run_id: str, status: str, trajectory_path: str = "") -> dict[str, str]:
    return {"run_id": run_id, "status": status, "trajectory_path": trajectory_path}


def _dataset(value: float) -> xr.Dataset:
    return xr.Dataset({"value": ("time", np.asarray([value, value + 1.0]))})


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_content_addressed_trajectory_does_not_replace_old_file(tmp_path) -> None:
    trajectories = tmp_path / "trajectories"
    old = write_content_addressed_netcdf(_dataset(1.0), trajectories, "scpc-old")
    old_hash = sha256_file(old)
    new = write_content_addressed_netcdf(_dataset(9.0), trajectories, "scpc-old")

    assert old != new
    assert old.is_file() and new.is_file()
    assert sha256_file(old) == old_hash
    assert sha256_file(new) != old_hash


def test_identical_trajectory_reuses_content_addressed_file(tmp_path) -> None:
    trajectories = tmp_path / "trajectories"
    first = write_content_addressed_netcdf(_dataset(1.0), trajectories, "scpc-run")
    second = write_content_addressed_netcdf(_dataset(1.0), trajectories, "scpc-run")
    assert first == second
    assert len(list(trajectories.glob("*.nc"))) == 1


def test_old_index_and_trajectory_survive_until_atomic_row_commit(tmp_path) -> None:
    trajectories = tmp_path / "trajectories"
    old_trajectory = write_content_addressed_netcdf(_dataset(1.0), trajectories, "scpc-run")
    new_trajectory = write_content_addressed_netcdf(_dataset(2.0), trajectories, "scpc-run")
    index = tmp_path / "scan_index.csv"
    old_row = _row("scpc-run", "completed", str(old_trajectory.relative_to(tmp_path)))
    replace_index_row_atomic(index, [], old_row)

    # A crash here would leave the durable old row and old trajectory valid;
    # the new content-addressed trajectory is only an orphan.
    assert _read_rows(index) == [old_row]
    assert old_trajectory.is_file()
    assert new_trajectory.is_file()

    new_row = _row("scpc-run", "completed", str(new_trajectory.relative_to(tmp_path)))
    committed = replace_index_row_atomic(index, [old_row], new_row)
    assert committed == [new_row]
    assert _read_rows(index) == [new_row]
    assert old_trajectory.is_file()

    cleanup_replaced_trajectory(tmp_path, old_row, new_row)
    assert not old_trajectory.exists()
    assert new_trajectory.is_file()


def test_failed_replacement_switches_index_before_old_cleanup(tmp_path) -> None:
    trajectories = tmp_path / "trajectories"
    old_trajectory = write_content_addressed_netcdf(_dataset(1.0), trajectories, "scpc-run")
    index = tmp_path / "scan_index.csv"
    old_row = _row("scpc-run", "completed", str(old_trajectory.relative_to(tmp_path)))
    replace_index_row_atomic(index, [], old_row)

    failed_row = _row("scpc-run", "failed", "")
    replace_index_row_atomic(index, [old_row], failed_row)
    assert _read_rows(index) == [failed_row]
    assert old_trajectory.is_file()

    cleanup_replaced_trajectory(tmp_path, old_row, failed_row)
    assert not old_trajectory.exists()


def test_recovery_removes_only_unreferenced_transaction_files(tmp_path) -> None:
    trajectories = tmp_path / "trajectories"
    referenced = write_content_addressed_netcdf(_dataset(1.0), trajectories, "scpc-a")
    orphan = write_content_addressed_netcdf(_dataset(2.0), trajectories, "scpc-b")
    pending = trajectories / ".scpc-c.pending-123.nc"
    pending.write_bytes(b"partial")
    row = _row("scpc-a", "completed", str(referenced.relative_to(tmp_path)))

    removed = cleanup_unreferenced_transaction_files(tmp_path, [row])

    assert referenced.is_file()
    assert not orphan.exists()
    assert not pending.exists()
    assert set(removed) == {
        str(orphan.relative_to(tmp_path)),
        str(pending.relative_to(tmp_path)),
    }


def test_atomic_row_replacement_preserves_other_runs(tmp_path) -> None:
    index = tmp_path / "scan_index.csv"
    first = _row("scpc-a", "completed")
    second = _row("scpc-b", "failed")
    replace_index_row_atomic(index, [], first)
    rows = replace_index_row_atomic(index, [first], second)
    replacement = _row("scpc-a", "failed")
    rows = replace_index_row_atomic(index, rows, replacement)
    assert rows == [replacement, second]
    assert _read_rows(index) == [replacement, second]


def test_unsafe_output_paths_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        safe_relative_output_path(tmp_path, "../escape.nc")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_relative_output_path(tmp_path, str((tmp_path / "absolute.nc").resolve()))
