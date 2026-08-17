from __future__ import annotations

import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.numerics.candidate_verification import compare_event_topology, verify_recurrence_candidate


def _event_solution(*, time: float = 0.5, kind: str = "bounce") -> SCPCSolution:
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0),
    )
    stored_time = np.asarray([0.0, 0.5, 1.0])
    return SCPCSolution(
        t=stored_time,
        a=np.ones_like(stored_time),
        H=np.asarray([-0.1, 0.0, 0.1]),
        phi=np.zeros_like(stored_time),
        phi_dot=np.zeros_like(stored_time),
        rho_m=np.zeros_like(stored_time),
        rho_r=np.zeros_like(stored_time),
        rho_phi=np.full_like(stored_time, 3.0),
        p_phi=np.full_like(stored_time, -3.0),
        constraint_residual=np.zeros_like(stored_time),
        turning_times=np.asarray([time]),
        turning_kinds=(kind,),
        parameters=parameters,
        solver_metadata={},
        turning_state_vectors=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        requested_end_time=1.0,
    )


def test_event_topology_detects_time_and_kind_changes() -> None:
    reference = _event_solution()
    shifted = _event_solution(time=0.5001)
    difference = compare_event_topology(reference, shifted)
    assert difference.event_sequence_match
    assert difference.max_normalized_event_time_error is not None
    assert difference.max_normalized_event_time_error > 0.0

    wrong_kind = _event_solution(kind="turnaround")
    wrong = compare_event_topology(reference, wrong_kind)
    assert not wrong.event_sequence_match
    assert np.isinf(wrong.maximum)


def test_default_like_short_run_is_a_clean_negative_candidate_result() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.03,
        rho_r_ref=0.0001,
        potential=PeriodicPotential(offset=3.5, amplitude=0.08, strata_count=4),
    )
    report = verify_recurrence_candidate(
        parameters,
        integration_options={
            "t_span": (0.0, 0.1),
            "samples": 21,
            "a0": 1.0,
            "phi0": 0.15,
            "phi_dot0": 0.0,
            "branch": 1,
        },
        tolerance_levels=(("medium", 1.0e-7, 1.0e-9), ("reference", 1.0e-9, 1.0e-11)),
        primary_method="DOP853",
        independent_method="Radau",
        independent_rtol=1.0e-8,
        independent_atol=1.0e-10,
        max_constraint_residual=1.0e-6,
        max_solution_error=1.0e-5,
        max_event_time_error=1.0e-6,
        max_event_state_error=1.0e-5,
        max_return_error=1.0e-3,
        minimum_same_kind_return_metrics=2,
    )
    assert report["status"] == "not_candidate_insufficient_or_nonclosing_returns"
    assert report["candidate_gate_passed"] is False
    assert report["stage2_input_eligible"] is False
