import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.numerics.cycles import (
    CycleReturnMetric,
    classify_recurrence,
    cycle_return_metrics,
    turning_states,
    wrapped_phase_difference,
)


def _synthetic_solution(
    *,
    target_space: str = "real",
    second_field_shift: float = 0.0,
    misleading_grid: bool = False,
) -> SCPCSolution:
    potential = PeriodicPotential(
        offset=1.0,
        amplitude=0.1,
        strata_count=4,
        field_scale=2.0,
        target_space=target_space,
    )
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=potential,
    )
    if misleading_grid:
        time = np.asarray([0.0, 2.0, 4.0])
        a = np.asarray([1.0, 8.0, 3.0])
        H = np.asarray([0.4, -0.3, 0.2])
        phi = np.asarray([-2.0, 4.0, 9.0])
        phi_dot = np.asarray([-3.0, 2.0, 5.0])
    else:
        time = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        a = np.asarray([1.8, 2.0, 2.2, 2.0, 1.8])
        H = np.asarray([0.2, 0.0, -0.2, 0.0, 0.2])
        phi = np.asarray([0.0, 0.2, 0.7, 0.2, 0.4])
        phi_dot = np.asarray([0.0, 0.4, 0.0, 0.4, 0.0])
    return SCPCSolution(
        t=time,
        a=a,
        H=H,
        phi=phi,
        phi_dot=phi_dot,
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
                [2.0, 0.0, 0.25 + second_field_shift, 0.5],
            ]
        ),
    )


def _return_metric(*, error: float, winding: int = 0) -> CycleReturnMetric:
    return CycleReturnMetric(
        kind="bounce",
        start_time=1.0,
        end_time=3.0,
        period=2.0,
        target_space="circle" if winding else "real",
        relative_scale_factor_error=error,
        field_error=0.0,
        field_winding=winding,
        relative_field_velocity_error=0.0,
        maximum_error=error,
    )


def test_wrapped_phase_difference_respects_circumference() -> None:
    assert np.isclose(wrapped_phase_difference(0.2, 0.2 + 2.0 * np.pi, 2.0 * np.pi), 0.0)


def test_one_close_return_is_insufficient_for_positive_classification() -> None:
    metrics = cycle_return_metrics(_synthetic_solution(), kind="bounce")
    assert len(metrics) == 1
    assert metrics[0].field_winding == 0
    assert metrics[0].maximum_error < 1.0e-12
    assert classify_recurrence(metrics, tolerance=1.0e-9) == "insufficient_repeated_returns"


def test_two_close_returns_receive_only_a_return_sequence_summary() -> None:
    metrics = (_return_metric(error=1.0e-10), _return_metric(error=2.0e-10))
    assert classify_recurrence(metrics, tolerance=1.0e-9) == "repeated_close_returns"


def test_real_field_shift_by_potential_period_is_not_a_close_return() -> None:
    solution = _synthetic_solution()
    shifted = _synthetic_solution(second_field_shift=solution.parameters.potential.potential_period)
    metric = cycle_return_metrics(shifted, kind="bounce")[0]
    assert np.isclose(metric.field_error, 1.0)
    assert metric.field_winding == 0


def test_adjacent_circle_strata_are_distinct() -> None:
    solution = _synthetic_solution(target_space="circle")
    shifted = _synthetic_solution(
        target_space="circle",
        second_field_shift=solution.parameters.potential.potential_period,
    )
    metric = cycle_return_metrics(shifted, kind="bounce")[0]
    assert np.isclose(metric.field_error, 1.0)
    assert metric.field_winding == 0


def test_full_circle_return_reports_winding_separately() -> None:
    solution = _synthetic_solution(target_space="circle")
    circumference = solution.parameters.potential.target_circumference
    assert circumference is not None
    winding_solution = _synthetic_solution(
        target_space="circle",
        second_field_shift=circumference,
    )
    metric = cycle_return_metrics(winding_solution, kind="bounce")[0]
    assert metric.field_error < 1.0e-12
    assert metric.field_winding == 1
    repeated = (metric, metric)
    assert classify_recurrence(repeated, tolerance=1.0e-9) == "repeated_close_winding_returns"


def test_exact_event_states_override_misleading_output_grid() -> None:
    solution = _synthetic_solution(misleading_grid=True)
    states = turning_states(solution)
    assert states[0].scale_factor == states[1].scale_factor == 2.0
    metrics = cycle_return_metrics(solution, kind="bounce")
    assert metrics[0].maximum_error < 1.0e-12


def test_event_states_are_serialized_with_coordinates() -> None:
    dataset = _synthetic_solution().to_xarray()
    assert dataset.sizes["turning_event"] == 2
    assert np.allclose(dataset["turning_hubble"].values, 0.0)
    assert dataset["turning_kind_code"].values.tolist() == [1, 1]


def test_decreasing_errors_above_tolerance_remain_unresolved() -> None:
    metrics = (_return_metric(error=9.5e-3), _return_metric(error=9.0e-3))
    assert (
        classify_recurrence(metrics, tolerance=1.0e-3)
        == "return_errors_decreasing_but_unresolved"
    )


def test_large_repeated_return_errors_are_nonclosing() -> None:
    metrics = (_return_metric(error=0.1), _return_metric(error=0.2))
    assert classify_recurrence(metrics, tolerance=1.0e-3) == "nonclosing_or_drifting_returns"
