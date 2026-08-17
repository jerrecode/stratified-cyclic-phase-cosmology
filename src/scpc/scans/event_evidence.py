"""Structured evidence derived from conservative outcome diagnostics."""

from __future__ import annotations

from typing import Any

from scpc.models.phase import SCPCSolution
from scpc.scans.outcomes import _unmatched_sampled_crossings


def unmatched_hubble_crossing_records(
    solution: SCPCSolution,
    tolerance: float,
) -> tuple[dict[str, Any], ...]:
    """Return the exact unmatched-crossing evidence used by outcome classification."""

    return tuple(
        {
            "kind": crossing.kind,
            "start_time": crossing.start_time,
            "end_time": crossing.end_time,
        }
        for crossing in _unmatched_sampled_crossings(solution, tolerance)
    )
