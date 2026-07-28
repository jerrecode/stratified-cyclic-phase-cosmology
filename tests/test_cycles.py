import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.numerics.cycles import (
    CycleReturnMetric,
    classify_recurrence,
    cycle_return_metrics,
    turning_states,
    wrapped_phase_difference,
)


def _synthetic_solution(*, exact_event_states: bool = False) -> SCPCSolution:
    potential = PeriodicPotential(offset=1.0, amplitude=0.1, strata_count=4, field_scale=2.0)
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=potential,
    )
    field_period = 2.0 * np.pi * potential.field_scale / potential.strata_count
    time = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    event_states = None
    if exact_event_states:
        event_states = np.asarray(
            [
                [2.0, 0.0, 0.2, 0.4],
                [2.0, 0.0, 0.2 + field_period, 0.4],
            ]
        )
    return SCPCSolution(
        t=time,
        a=np.asarray([1.8, 2.0, 2.2, 2.0, 1.8]),
        H=np.asarray([0.2, 0.0, -0.2, 0.0, 0.2]),
        phi=np.asarray([0.0, 0.2, 0.7, 0.2 + field_period, 0.4 + field_period]),
        phi_dot=np.asarray([0.0, 0.4, 0.0, 0.4, 0.0]),
        rho_m=np.zeros_like(time),
        rho_r=np.zeros_like(time),
        rho_phi=np.ones_like(time),
        p_phi=-np.ones_like(time),
        constraint_residual=np.zeros_like(time),
        turning_times=np.asarray([1.0, 3.0]),
        turning_kinds=("bounce", "bounce"),
        parameters=parameters,
        solver_metadata={},
        turning_state_vectors=event_states,
    )


def _coarse_misleading_solution() -> SCPCSolution:
    potential = PeriodicPotential(offset=1.0, amplitude=0.1, strata_count=4, field_scale=2.0)
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=potential,
    )
    field_period = 2.0 * np.pi * potential.field_scale / potential.strata_count
    time = np.asarray([0.0, 2.0, 4.0])
    return SCPCSolution(
        t=time,
        a=np.asarray([1.0, 8.0, 3.0]),
        H=np.asarray([0.4, -0.3, 0.2]),
        phi=np.asarray([-2.0, 4.0, 9.0]),
        phi_dot=np.asarray([-3.0, 2.0, 5.0]),
        rho_m=np.zeros_like(time),
        rho_r=np.zeros_like(time),
        rho_phi=np.ones_like(time),
        p_phi=-np.ones_like(time),
        constraint_residual=np.zeros_like(time),
        turning_times=np.asarray([1.0, 3.0]),
        turning_kinds=("bounce", "bounce"),
        parameters=parameters,
        solver_metadata={},
        turning_state_vectors=np.asarray(
            [
                [2.0, 0.0, 0.25, 0.5],
                [2.0, 0.0, 0.25 + field_period, 0.5],
            ]
        ),
    )


def test_wrapped_phase_difference_respects_periodicity() -> None:
    assert np.isclose(wrapped_phase_difference(0.2, 0.2 + 2.0 * np.pi, 2.0 * np.pi), 0.0)


def test_equivalent_turning_states_are_recurrent_candidate() -> None:
    metrics = cycle_return_metrics(_synthetic_solution(exact_event_states=True), kind="bounce")
    assert len(metrics) == 1
    assert metrics[0].maximum_error < 1.0e-12
    assert classify_recurrence(metrics, tolerance=1.0e-9) == "recurrent_candidate"


def test_exact_event_states_override_misleading_output_grid() -> None:
    solution = _coarse_misleading_solution()
    states = turning_states(solution)
    assert states[0].scale_factor == states[1].scale_factor == 2.0
    metrics = cycle_return_metrics(solution, kind="bounce")
    assert metrics[0].maximum_error < 1.0e-12


def test_event_states_are_serialized_with_coordinates() -> None:
    dataset = _synthetic_solution(exact_event_states=True).to_xarray()
    assert dataset.sizes["turning_event"] == 2
    assert np.allclose(dataset["turning_hubble"].values, 0.0)
    assert dataset["turning_kind_code"].values.tolist() == [1, 1]


def test_large_return_error_is_not_recurrent() -> None:
    metric = CycleReturnMetric(
        kind="bounce",
        start_time=1.0,
        end_time=3.0,
        period=2.0,
        relative_scale_factor_error=0.1,
        wrapped_phase_error=0.0,
        relative_field_velocity_error=0.0,
        maximum_error=0.1,
    )
    assert classify_recurrence((metric,), tolerance=1.0e-3) == "nonrecurrent_or_drifting"
