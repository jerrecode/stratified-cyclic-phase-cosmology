"""Filesystem durability helpers for atomic scan transactions."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes before a transaction proceeds.

    POSIX requires syncing the containing directory after rename/unlink operations
    when crash durability matters. Windows does not expose the same directory-fd
    contract through ``os.open``, so atomic replacement remains the available
    guarantee there.
    """

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
