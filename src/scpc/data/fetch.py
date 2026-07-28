"""Acquisition functions for machine-resolvable manifest products."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


class AccessInstructionRequired(RuntimeError):
    """Raised when a product requires an archive query or interactive request."""


def _download_http(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return destination


def _clone_git(url: str, destination: Path, ref: str | None) -> Path:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    command = ["git", "clone", "--filter=blob:none"]
    if ref:
        command.extend(["--branch", ref])
    command.extend([url, str(destination)])
    subprocess.run(command, check=True)
    return destination


def fetch_product(product: dict[str, Any], destination_root: str | Path) -> Path:
    access = product["access"]
    method = access["method"]
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    target_name = access.get("target_name") or product["id"]
    target = destination_root / target_name

    if method == "direct_http":
        return _download_http(access["url"], target)
    if method == "git":
        return _clone_git(access["url"], target, access.get("ref"))

    instructions = access.get("instructions", "Consult the authoritative release documentation.")
    raise AccessInstructionRequired(
        f"Product {product['id']} uses access method {method!r}. {instructions}"
    )
