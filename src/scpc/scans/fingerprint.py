"""Deterministic implementation and numerical-runtime fingerprints for scans."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


NUMERICAL_PACKAGES = (
    "numpy",
    "scipy",
    "xarray",
    "PyYAML",
    "jsonschema",
    "matplotlib",
)


def source_tree_sha256(package_root: str | Path | None = None) -> str:
    """Hash every Python source file in the installed ``scpc`` package tree."""

    root = (
        Path(package_root).resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"No Python source files found below {root}")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in NUMERICAL_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def implementation_runtime_fingerprint(
    *,
    package_root: str | Path | None = None,
    package_versions: dict[str, str | None] | None = None,
    python_version: str | None = None,
    python_implementation: str | None = None,
    platform_string: str | None = None,
    machine: str | None = None,
) -> dict[str, Any]:
    """Build the strict compatibility fingerprint for scan resume.

    Platform and architecture are intentionally strict. A scan copied to a
    different numerical environment must use a new output directory rather
    than silently mixing rows produced by different implementations.
    """

    strict_payload: dict[str, Any] = {
        "fingerprint_schema_version": 1,
        "source_tree_sha256": source_tree_sha256(package_root),
        "python_implementation": python_implementation or platform.python_implementation(),
        "python_version": python_version or platform.python_version(),
        "platform": platform_string or platform.platform(),
        "machine": machine or platform.machine(),
        "packages": dict(sorted((package_versions or _installed_versions()).items())),
    }
    canonical = json.dumps(
        strict_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **strict_payload,
        "strict_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "python_executable": sys.executable,
    }
