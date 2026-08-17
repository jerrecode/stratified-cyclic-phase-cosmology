"""Machine validation for the dependency-ordered SCPC scientific gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ALLOWED_STATUSES = {"passed", "open", "blocked"}


def load_and_validate_gate_ledger(path: str | Path) -> dict[str, Any]:
    """Load a gate ledger and reject dependency/status inconsistencies.

    A passed gate may depend only on passed gates. An open gate must have all
    prerequisites passed. A blocked gate must have at least one prerequisite
    that is not passed. These rules make it impossible to mark a late-stage
    claim as passed while an earlier evidential dependency remains unresolved.
    """

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("scientific gate ledger must be a schema_version 1 mapping")
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("scientific gate ledger must contain a non-empty gates list")

    by_id: dict[str, dict[str, Any]] = {}
    for raw in gates:
        if not isinstance(raw, dict):
            raise ValueError("every scientific gate must be a mapping")
        gate_id = raw.get("id")
        status = raw.get("status")
        dependencies = raw.get("depends_on", [])
        if not isinstance(gate_id, str) or not gate_id:
            raise ValueError("every scientific gate needs a non-empty id")
        if gate_id in by_id:
            raise ValueError(f"duplicate scientific gate id {gate_id!r}")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"invalid status {status!r} for gate {gate_id}")
        if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise ValueError(f"depends_on must be a string list for gate {gate_id}")
        if gate_id in dependencies:
            raise ValueError(f"gate {gate_id} cannot depend on itself")
        if status == "passed":
            evidence = raw.get("evidence")
            if not isinstance(evidence, list) or not evidence or any(
                not isinstance(item, str) or not item.strip() for item in evidence
            ):
                raise ValueError(f"passed gate {gate_id} requires non-empty evidence")
        by_id[gate_id] = raw

    for gate_id, gate in by_id.items():
        for dependency in gate.get("depends_on", []):
            if dependency not in by_id:
                raise ValueError(f"gate {gate_id} depends on unknown gate {dependency}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(gate_id: str) -> None:
        if gate_id in visiting:
            raise ValueError(f"scientific gate dependency cycle includes {gate_id}")
        if gate_id in visited:
            return
        visiting.add(gate_id)
        for dependency in by_id[gate_id].get("depends_on", []):
            visit(dependency)
        visiting.remove(gate_id)
        visited.add(gate_id)

    for gate_id in by_id:
        visit(gate_id)

    for gate_id, gate in by_id.items():
        dependency_statuses = [by_id[item]["status"] for item in gate.get("depends_on", [])]
        status = gate["status"]
        if status == "passed" and any(item != "passed" for item in dependency_statuses):
            raise ValueError(f"passed gate {gate_id} has an unpassed dependency")
        if status == "open" and any(item != "passed" for item in dependency_statuses):
            raise ValueError(f"open gate {gate_id} is still dependency-blocked")
        if status == "blocked" and dependency_statuses and all(
            item == "passed" for item in dependency_statuses
        ):
            raise ValueError(f"blocked gate {gate_id} has no unpassed dependency")
        if status == "blocked" and not dependency_statuses:
            raise ValueError(f"root gate {gate_id} cannot be blocked without a dependency")

    payload["gate_count"] = len(by_id)
    payload["passed_gates"] = [gate_id for gate_id, gate in by_id.items() if gate["status"] == "passed"]
    payload["open_gates"] = [gate_id for gate_id, gate in by_id.items() if gate["status"] == "open"]
    payload["blocked_gates"] = [gate_id for gate_id, gate in by_id.items() if gate["status"] == "blocked"]
    return payload
