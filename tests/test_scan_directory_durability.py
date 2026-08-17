from pathlib import Path

import pytest

import scpc.scans.termination_records as termination_records
import scpc.scans.transactions as transactions
from scpc.scans.errors import OutputSerializationError


class _MinimalDataset:
    def to_netcdf(self, path: Path, *, engine: str) -> None:
        assert engine == "scipy"
        Path(path).write_bytes(b"deterministic-netcdf-placeholder")


def test_trajectory_syncs_file_before_directory(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "trajectories"
    calls: list[str] = []
    monkeypatch.setattr(transactions.os, "fsync", lambda descriptor: calls.append("file"))
    monkeypatch.setattr(
        transactions,
        "fsync_directory",
        lambda path: calls.append("directory"),
    )

    destination = transactions.write_content_addressed_netcdf(
        _MinimalDataset(),
        directory,
        "scpc-test-run",
    )

    assert destination.is_file()
    assert calls == ["file", "directory"]


def test_termination_record_syncs_directory_before_return(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "termination_records"
    calls: list[Path] = []
    monkeypatch.setattr(
        termination_records,
        "fsync_directory",
        lambda path: calls.append(Path(path)),
    )

    destination, digest = termination_records.write_content_addressed_termination_record(
        {"evidence_schema_version": 1, "value": "stable"},
        directory,
        "scpc-test-run",
    )

    assert destination.is_file()
    assert len(digest) == 64
    assert calls == [directory]


def test_index_replacement_syncs_parent_before_return(tmp_path, monkeypatch) -> None:
    index = tmp_path / "scan_index.csv"
    calls: list[Path] = []
    monkeypatch.setattr(
        transactions,
        "fsync_directory",
        lambda path: calls.append(Path(path)),
    )

    committed = transactions.replace_index_row_atomic(
        index,
        [],
        {"run_id": "scpc-test-run", "status": "completed"},
    )

    assert committed == [{"run_id": "scpc-test-run", "status": "completed"}]
    assert index.is_file()
    assert calls == [tmp_path]


def test_index_directory_sync_failure_is_output_error(tmp_path, monkeypatch) -> None:
    index = tmp_path / "scan_index.csv"

    def fail_sync(path: Path) -> None:
        raise OSError(f"forced directory sync failure for {path}")

    monkeypatch.setattr(transactions, "fsync_directory", fail_sync)
    with pytest.raises(OutputSerializationError, match="durably replace scan index"):
        transactions.replace_index_row_atomic(
            index,
            [],
            {"run_id": "scpc-test-run", "status": "completed"},
        )
    assert not index.with_suffix(".csv.tmp").exists()
