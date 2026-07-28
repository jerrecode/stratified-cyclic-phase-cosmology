"""Run provenance and checksum helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_output_inventory(
    paths: Iterable[str | Path],
    *,
    relative_to: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Describe generated files after they have been written.

    The provenance file itself should not be included, because its checksum
    would be self-referential. Paths are sorted for deterministic records.
    """

    root = Path(relative_to).resolve() if relative_to is not None else None
    inventory: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot inventory missing output file: {raw_path}")
        display_path = str(path.relative_to(root)) if root is not None else str(raw_path)
        inventory.append(
            {
                "path": display_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_provenance(config_path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "scipy", "xarray", "PyYAML", "jsonschema", "matplotlib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    record: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    if extra:
        record.update(extra)
    return record


def write_provenance(path: str | Path, record: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
