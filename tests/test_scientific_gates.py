from __future__ import annotations

from pathlib import Path

import pytest

from scpc.scientific_gates import load_and_validate_gate_ledger


def test_repository_gate_ledger_is_dependency_consistent() -> None:
    ledger = load_and_validate_gate_ledger("configs/scientific_gates.yaml")
    assert ledger["passed_gates"] == ["stage0_numerical_consistency"]
    assert ledger["open_gates"] == ["stage1_reproduced_background_candidate"]
    assert "stage2_homogeneous_stability" in ledger["blocked_gates"]
    assert "stage5_scpc_statistical_inference" in ledger["blocked_gates"]


def test_passed_gate_cannot_skip_an_unpassed_dependency(tmp_path: Path) -> None:
    path = tmp_path / "gates.yaml"
    path.write_text(
        """schema_version: 1
gates:
  - id: stage0
    status: open
    depends_on: []
  - id: stage1
    status: passed
    depends_on: [stage0]
    evidence: [forged]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unpassed dependency"):
        load_and_validate_gate_ledger(path)


def test_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gates.yaml"
    path.write_text(
        """schema_version: 1
gates:
  - id: a
    status: blocked
    depends_on: [b]
  - id: b
    status: blocked
    depends_on: [a]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dependency cycle"):
        load_and_validate_gate_ledger(path)
