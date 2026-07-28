"""Conservative retrieval planning for manifest products.

This module prints or executes only explicit, stable command templates recorded
in the manifest. Query-based, basket, TAP, and Globus products remain manual or
provider-tool workflows and are never guessed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def retrieval_plan(release: dict[str, Any], destination: str | Path) -> list[str]:
    destination = Path(destination)
    commands = release.get("access", {}).get("commands", [])
    return [command.format(destination=str(destination), release_id=release["id"]) for command in commands]


def execute_plan(commands: list[str], *, dry_run: bool = True) -> None:
    for command in commands:
        print(command)
        if not dry_run:
            subprocess.run(command, shell=True, check=True)  # noqa: S602
