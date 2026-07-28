import numpy as np
import pytest

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.scans.errors import ResultIntegrityError
from scpc.scans.outcomes import OutcomeClass, assess_solution


def _solution(
    *,
    hubble: list[float],
    turning_kinds: tuple[str, ...] = (),
    turning_times: list[float] | None = None,
    scale_factor: list[float] | None = None,
    constraint: list[float] | None = None,
) -> SCPCSolution:
    count = len(hubble)
    time = np.linspace(0.0, float(count - 1), count)
    a = np.asarray(scale_factor if scale_factor is not None else np.linspace(1.0, 2.0, count))
    residual = np.asarray(constraint if constraint is not None else np.zeros(count))
    event_times = np.asarray(turning_times if turning_times is not None else [], dtype=float)
    event_states = np.empty((0, 4), dtype=float)
    if turning_kinds:
        if turning_times is None or len(turning_times) != len(turning_kinds):
            raise ValueError("turning_times must match turning_kinds in the test fixture")
        event_states = np.asarray(
            [
                [
                    float(np.interp(event_time, time, a)),
                    0.0,
                    0.1 * index,
                    0.0,
                ]
                for index, event_time in enumerate(event_times)
            ],
            dtype=float,
        )
    return SCPCSolution(
        t=time,
        a=a,
        H=np.asarray(hubble, dtype=float),
        phi=np.zeros(count),
        phi_dot=np.zeros(count),
        rho_m=np.zeros(count),
        rho_r=np.zeros(count),
        rho_phi=np.ones(count),
        p_phi=-np.ones(count),
        constraint_residual=residual,
        turning_times=event_times,
        turning_kinds=turning_kinds,
        parameters=SCPCParameters(
            spatial_curvature_k=0,
            rho_m_ref=0.0,
            rho_r_ref=0.0,
            potential=PeriodicPotential(offset=3.0, amplitude=0.0),
        ),
        solver_metadata={},
        turning_state_vectors=event_states,
    )


def test_monotonic_expansion_is_classified_without_cyclic_language() -> None:
    assessment = assess_solution(_solution(hubble=[0.4, 0.3, 0.2]))
    assert assessment.outcome is OutcomeClass.MONOTONIC_EXPANSION
    assert assessment.numerically_valid is True
    assert assessment.event_sequence == ()


def test_monotonic_contraction_is_classified() -> None:
    assessment = assess_solution(_solution(hubble=[-0.2, -0.3, -0.4]))
    assert assessment.outcome is OutcomeClass.MONOTONIC_CONTRACTION


def test_unrecorded_sign_change_is_numerical_event_failure() -> None:
    assessment = assess_solution(_solution(hubble=[0.3, 0.1, -0.2]))
    assert assessment.outcome is OutcomeClass.UNRESOLVED_EVENT_DETECTION
    assert assessment.numerically_valid is False


def test_zero_plateau_sign_change_is_numerical_event_failure() -> None:
    assessment = assess_solution(_solution(hubble=[0.3, 0.0, -0.2]))
    assert assessment.outcome is OutcomeClass.UNRESOLVED_EVENT_DETECTION
    assert assessment.numerically_valid is False


def test_partial_event_record_does_not_hide_later_missed_crossings() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[-0.2, 0.0, 0.2, 0.0, -0.2, 0.0, 0.2],
            turning_kinds=("bounce",),
            turning_times=[1.0],
        )
    )
    assert assessment.outcome is OutcomeClass.UNRESOLVED_EVENT_DETECTION
    assert assessment.numerically_valid is False


def test_constraint_violation_overrides_trajectory_morphology() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[0.2, 0.0, -0.2],
            turning_kinds=("turnaround",),
            turning_times=[1.0],
            constraint=[0.0, 2.0e-4, 0.0],
        ),
        constraint_threshold=1.0e-7,
    )
    assert assessment.outcome is OutcomeClass.CONSTRAINT_VIOLATION
    assert assessment.numerically_valid is False


def test_nonfinite_state_is_not_called_singular() -> None:
    assessment = assess_solution(_solution(hubble=[0.2, np.nan, -0.2]))
    assert assessment.outcome is OutcomeClass.NONFINITE_STATE
    assert "singular" not in assessment.reason.lower()


def test_nonpositive_scale_factor_is_separate_invalidity() -> None:
    assessment = assess_solution(
        _solution(hubble=[0.2, 0.1, 0.05], scale_factor=[1.0, 0.5, 0.0])
    )
    assert assessment.outcome is OutcomeClass.NONPOSITIVE_SCALE_FACTOR


def test_one_turnaround_without_bounce_is_recollapse() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[0.2, 0.0, -0.2],
            turning_kinds=("turnaround",),
            turning_times=[1.0],
        )
    )
    assert assessment.outcome is OutcomeClass.RECOLLAPSE_WITHOUT_BOUNCE
    assert assessment.turnaround_count == 1


def test_one_bounce_is_not_called_a_cycle() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[-0.2, 0.0, 0.2],
            turning_kinds=("bounce",),
            turning_times=[1.0],
        )
    )
    assert assessment.outcome is OutcomeClass.ONE_OFF_BOUNCE
    assert "cycle" not in assessment.reason.lower()


def test_one_bounce_turnaround_pair_is_not_repeated() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[-0.2, 0.0, 0.2, 0.0, -0.2],
            turning_kinds=("bounce", "turnaround"),
            turning_times=[1.0, 3.0],
        )
    )
    assert assessment.outcome is OutcomeClass.SINGLE_BOUNCE_TURNAROUND_PAIR


def test_more_than_two_events_are_only_repeated_turning_points() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[-0.2, 0.0, 0.2, 0.0, -0.2, 0.0, 0.2],
            turning_kinds=("bounce", "turnaround", "bounce"),
            turning_times=[1.0, 3.0, 5.0],
        )
    )
    assert assessment.outcome is OutcomeClass.REPEATED_TURNING_POINTS
    assert assessment.return_sequence_classifications == {
        "bounce": "insufficient_repeated_returns"
    }


def test_degenerate_event_is_not_treated_as_bounce_or_turnaround() -> None:
    assessment = assess_solution(
        _solution(
            hubble=[0.1, 0.0, 0.1],
            turning_kinds=("degenerate",),
            turning_times=[1.0],
        )
    )
    assert assessment.outcome is OutcomeClass.DEGENERATE_TURNING_EVENT
    assert assessment.numerically_valid is False


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_invalid_thresholds_are_rejected(value: float) -> None:
    solution = _solution(hubble=[0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="finite and positive"):
        assess_solution(solution, constraint_threshold=value)


def test_inconsistent_event_data_raises_result_integrity_error() -> None:
    solution = _solution(hubble=[0.3, 0.2, 0.1])
    solution.turning_kinds = ("bounce",)
    with pytest.raises(ResultIntegrityError, match="equal lengths"):
        assess_solution(solution)
