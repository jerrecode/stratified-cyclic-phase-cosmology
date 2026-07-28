"""Run provenance and checksum helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
