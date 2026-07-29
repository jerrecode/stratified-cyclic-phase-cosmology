"""Canonical, content-addressed evidence for terminated scan attempts."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scpc.scans.errors import OutputSerializationError, ResultIntegrityError


def _finite_hex(value: Any, name: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number.hex()


def _canonical_crossings(crossings: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for index, crossing in enumerate(crossings):
        try:
            kind = str(crossing["kind"])
            start = crossing["start_time"]
            end = crossing["end_time"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed unmatched crossing at index {index}") from error
        if kind not in {"bounce", "turnaround"}:
            raise ValueError(f"Unknown unmatched crossing kind: {kind!r}")
        start_hex = _finite_hex(start, f"crossing[{index}].start_time")
        end_hex = _finite_hex(end, f"crossing[{index}].end_time")
        if float.fromhex(end_hex) < float.fromhex(start_hex):
            raise ValueError("Unmatched crossing end time precedes its start time")
        canonical.append({"kind": kind, "start_time": start_hex, "end_time": end_hex})
    return canonical


def _canonical_boundaries(boundaries: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for index, boundary in enumerate(boundaries):
        try:
            kind = str(boundary["kind"])
            threshold = boundary["threshold"]
            observed = boundary["observed"]
            units = str(boundary["units"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed termination boundary at index {index}") from error
        if not kind or not units:
            raise ValueError("Termination boundary kind and units must be nonempty")
        canonical.append(
            {
                "kind": kind,
                "threshold": _finite_hex(threshold, f"boundary[{index}].threshold"),
                "observed": _finite_hex(observed, f"boundary[{index}].observed"),
                "units": units,
            }
        )
    kinds = [item["kind"] for item in canonical]
    if kinds != sorted(set(kinds)):
        raise ValueError("Termination boundaries must be unique and lexically ordered")
    return canonical


def _termination_evidence_payload_impl(
    *,
    run_id: str,
    run_sha256: str,
    outcome: str,
    numerically_valid: bool,
    max_abs_constraint_residual: Any,
    bounce_count: int,
    turnaround_count: int,
    degenerate_count: int,
    event_sequence: Sequence[str],
    unmatched_hubble_crossings: Iterable[Mapping[str, Any]],
    termination_time: Any,
    termination_state_vector: Sequence[Any],
    termination_constraint_residual: Any,
    termination_boundaries: Iterable[Mapping[str, Any]],
    requested_end_time: Any,
    solver_rtol: Any,
    solver_atol: Any,
    integration_domain: str,
) -> dict[str, Any]:
    if not run_id or not run_sha256 or not outcome:
        raise ValueError("Run identity and outcome fields must be nonempty")
    if not isinstance(numerically_valid, bool):
        raise TypeError("numerically_valid must be boolean")
    counts = (bounce_count, turnaround_count, degenerate_count)
    if any(isinstance(value, bool) or int(value) != value or int(value) < 0 for value in counts):
        raise ValueError("Event counts must be nonnegative integers")
    sequence = [str(kind) for kind in event_sequence]
    if sequence.count("bounce") != int(bounce_count):
        raise ValueError("bounce_count disagrees with event_sequence")
    if sequence.count("turnaround") != int(turnaround_count):
        raise ValueError("turnaround_count disagrees with event_sequence")
    if sequence.count("degenerate") != int(degenerate_count):
        raise ValueError("degenerate_count disagrees with event_sequence")
    if set(sequence) - {"bounce", "turnaround", "degenerate"}:
        raise ValueError("event_sequence contains an unknown turning kind")
    if len(termination_state_vector) != 4:
        raise ValueError("termination_state_vector must contain four components")
    if not isinstance(integration_domain, str) or not integration_domain:
        raise ValueError("integration_domain must be a nonempty canonical JSON string")
    try:
        decoded_domain = json.loads(integration_domain)
    except json.JSONDecodeError as error:
        raise ValueError("integration_domain is not valid JSON") from error
    if not isinstance(decoded_domain, dict) or not decoded_domain:
        raise ValueError("integration_domain must describe at least one configured boundary")

    return {
        "evidence_schema_version": 1,
        "run_id": run_id,
        "run_sha256": run_sha256,
        "classification": {
            "outcome": outcome,
            "numerically_valid": numerically_valid,
            "max_abs_constraint_residual": _finite_hex(
                max_abs_constraint_residual,
                "max_abs_constraint_residual",
            ),
            "bounce_count": int(bounce_count),
            "turnaround_count": int(turnaround_count),
            "degenerate_count": int(degenerate_count),
            "event_sequence": sequence,
            "unmatched_hubble_crossings": _canonical_crossings(
                unmatched_hubble_crossings
            ),
        },
        "termination": {
            "time": _finite_hex(termination_time, "termination_time"),
            "state": [
                _finite_hex(value, f"termination_state_vector[{index}]")
                for index, value in enumerate(termination_state_vector)
            ],
            "constraint_residual": _finite_hex(
                termination_constraint_residual,
                "termination_constraint_residual",
            ),
            "boundaries": _canonical_boundaries(termination_boundaries),
            "requested_end_time": _finite_hex(requested_end_time, "requested_end_time"),
        },
        "solver": {
            "rtol": _finite_hex(solver_rtol, "solver_rtol"),
            "atol": _finite_hex(solver_atol, "solver_atol"),
            "integration_domain": integration_domain,
        },
    }


def termination_evidence_payload(
    *,
    run_id: str,
    run_sha256: str,
    outcome: str,
    numerically_valid: bool,
    max_abs_constraint_residual: Any,
    bounce_count: int,
    turnaround_count: int,
    degenerate_count: int,
    event_sequence: Sequence[str],
    unmatched_hubble_crossings: Iterable[Mapping[str, Any]],
    termination_time: Any,
    termination_state_vector: Sequence[Any],
    termination_constraint_residual: Any,
    termination_boundaries: Iterable[Mapping[str, Any]],
    requested_end_time: Any,
    solver_rtol: Any,
    solver_atol: Any,
    integration_domain: str,
) -> dict[str, Any]:
    """Build exact evidence, classifying malformed in-memory results as integrity errors."""

    try:
        return _termination_evidence_payload_impl(
            run_id=run_id,
            run_sha256=run_sha256,
            outcome=outcome,
            numerically_valid=numerically_valid,
            max_abs_constraint_residual=max_abs_constraint_residual,
            bounce_count=bounce_count,
            turnaround_count=turnaround_count,
            degenerate_count=degenerate_count,
            event_sequence=event_sequence,
            unmatched_hubble_crossings=unmatched_hubble_crossings,
            termination_time=termination_time,
            termination_state_vector=termination_state_vector,
            termination_constraint_residual=termination_constraint_residual,
            termination_boundaries=termination_boundaries,
            requested_end_time=requested_end_time,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            integration_domain=integration_domain,
        )
    except ResultIntegrityError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ResultIntegrityError(
            f"Could not construct canonical termination evidence: {error}"
        ) from error


def canonical_evidence_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def evidence_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_evidence_bytes(payload)).hexdigest()


def _remove_pending(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


def write_content_addressed_termination_record(
    payload: Mapping[str, Any],
    directory: Path,
    run_id: str,
) -> tuple[Path, str]:
    """Atomically store canonical evidence or raise an output-serialization error."""

    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not run_id or any(character not in safe_characters for character in run_id):
        raise ResultIntegrityError("run_id contains unsafe filename characters")
    try:
        content = canonical_evidence_bytes(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResultIntegrityError(
            f"Could not canonicalize termination evidence for {run_id}: {error}"
        ) from error
    digest = hashlib.sha256(content).hexdigest()
    pending = directory / f".{run_id}.pending-{os.getpid()}.json"
    destination = directory / f"{run_id}-{digest[:20]}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with pending.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            if destination.read_bytes() != content:
                raise OutputSerializationError(
                    f"Content-address collision for {destination.name}"
                )
            pending.unlink()
        else:
            pending.replace(destination)
        return destination, digest
    except OutputSerializationError:
        _remove_pending(pending)
        raise
    except Exception as error:
        _remove_pending(pending)
        raise OutputSerializationError(
            f"Could not serialize termination evidence for {run_id}: {error}"
        ) from error


def read_termination_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Read canonical evidence and verify both bytes and the full digest."""

    if len(expected_sha256) != 64:
        raise ValueError("termination_record_sha256 must contain 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("termination_record_sha256 is not hexadecimal") from error
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Termination record checksum mismatch for {path.name}")
    if not path.name.endswith(f"-{digest[:20]}.json"):
        raise ValueError(f"Termination record filename is not content-addressed: {path.name}")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Termination record is not valid JSON: {path.name}") from error
    if canonical_evidence_bytes(payload) != content:
        raise ValueError(f"Termination record is not canonically serialized: {path.name}")
    return payload
