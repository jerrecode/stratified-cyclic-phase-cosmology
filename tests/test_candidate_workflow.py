from __future__ import annotations

import json
from pathlib import Path

from scpc.workflows import audit_scpc_candidate


def test_canonical_candidate_audit_short_circuits_on_exact_exclusion(tmp_path: Path) -> None:
    report_path = audit_scpc_candidate(
        "configs/scpc_candidate_verification.yaml",
        tmp_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "analytically_excluded_future_turnaround"
    assert report["candidate_gate_passed"] is False
    assert report["stage2_input_eligible"] is False
    assert report["analytic_preflight_short_circuit"] is True
    assert report["runs"] == {}
    certificate = report["turning_feasibility_certificate"]
    assert certificate["future_turnaround_excluded"] is True
    assert certificate["turning_scale_factor_upper_bound"] < certificate["initial_scale_factor"]
