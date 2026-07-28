import json

import numpy as np
import pytest
import xarray as xr

from scpc.models.phase import (
    PeriodicPotential,
    SCPCIntegrationDomain,
    SCPCParameters,
    _DomainEventDefinition,
    _first_dense_domain_exit,
    integrate_scpc,
)
from scpc.scans.errors import ResultIntegrityError
from scpc.scans.outcomes import assess_solution


def _de_sitter_parameters(*, target_space: str = "real") -> SCPCParameters:
    return SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(
            offset=3.0,
            amplitude=0.0,
            strata_count=1,
            target_space=target_space,
        ),
    )


def _integrate_de_sitter(
    *,
    branch: int,
    domain: SCPCIntegrationDomain,
    samples: int = 11,
):
    return integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 1.0),
        samples=samples,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=branch,
        rtol=1.0e-11,
        atol=1.0e-13,
        domain=domain,
        max_step=0.025,
        domain_check_substeps=16,
    )


def test_contracting_de_sitter_stops_at_exact_minimum_scale_factor() -> None:
    solution = _integrate_de_sitter(
        branch=-1,
        domain=SCPCIntegrationDomain(min_scale_factor=0.8),
    )

    expected_time = -np.log(0.8)
    assert solution.termination_kind == "minimum_scale_factor"
    assert np.isclose(solution.termination_time, expected_time, rtol=1.0e-9, atol=1.0e-11)
    assert np.isclose(solution.t[-1], expected_time, rtol=1.0e-9, atol=1.0e-11)
    assert np.isclose(solution.a[-1], 0.8, rtol=1.0e-10, atol=1.0e-12)
    assert solution.termination_boundaries == (
        {
            "kind": "minimum_scale_factor",
            "threshold": 0.8,
            "observed": pytest.approx(0.8, rel=1.0e-10),
            "units": "1",
        },
    )
    assert solution.completed_to_requested_end is False


def test_expanding_de_sitter_stops_at_exact_maximum_scale_factor() -> None:
    solution = _integrate_de_sitter(
        branch=1,
        domain=SCPCIntegrationDomain(max_scale_factor=1.25),
        samples=7,
    )

    assert solution.termination_kind == "maximum_scale_factor"
    assert np.isclose(solution.termination_time, np.log(1.25), rtol=1.0e-8)
    assert np.isclose(solution.a[-1], 1.25, rtol=1.0e-9)


def test_unbounded_reference_run_reaches_requested_end() -> None:
    solution = integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 0.5),
        samples=21,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
    )
    assert solution.termination_kind is None
    assert solution.completed_to_requested_end is True
    assert solution.t[-1] == pytest.approx(0.5)


def test_termination_metadata_and_boundary_set_are_scipy_netcdf_serializable(tmp_path) -> None:
    solution = _integrate_de_sitter(
        branch=-1,
        domain=SCPCIntegrationDomain(min_scale_factor=0.85),
        samples=9,
    )
    dataset = solution.to_xarray()

    assert dataset.attrs["termination_kind"] == "minimum_scale_factor"
    assert dataset.attrs["completed_to_requested_end"] == 0
    assert json.loads(dataset.attrs["termination_boundaries"])[0]["kind"] == "minimum_scale_factor"
    assert dataset.sizes["termination_boundary"] == 1
    assert float(dataset["termination_time"]) == pytest.approx(solution.termination_time)
    assert float(dataset["termination_scale_factor"]) == pytest.approx(0.85)

    destination = tmp_path / "terminated.nc"
    dataset.to_netcdf(destination, engine="scipy")
    restored = xr.load_dataset(destination, engine="scipy")
    assert restored.attrs["termination_kind"] == "minimum_scale_factor"
    assert restored.attrs["completed_to_requested_end"] == 0
    assert restored.sizes["termination_boundary"] == 1


def test_coincident_boundaries_are_all_recorded_with_deterministic_primary() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=3.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=0.0, amplitude=0.0, strata_count=1),
    )
    scale_threshold = 0.8
    hubble_threshold = scale_threshold ** (-1.5)
    solution = integrate_scpc(
        parameters,
        t_span=(0.0, 1.0),
        samples=31,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=-1,
        rtol=1.0e-10,
        atol=1.0e-12,
        domain=SCPCIntegrationDomain(
            min_scale_factor=scale_threshold,
            max_abs_hubble=hubble_threshold,
        ),
        max_step=0.01,
        domain_check_substeps=16,
    )

    kinds = tuple(boundary["kind"] for boundary in solution.termination_boundaries)
    assert kinds == ("maximum_absolute_hubble", "minimum_scale_factor")
    assert solution.termination_kind == "maximum_absolute_hubble"
    assert np.isclose(solution.a[-1], scale_threshold, rtol=1.0e-8)
    assert np.isclose(abs(solution.H[-1]), hubble_threshold, rtol=1.0e-8)
    assert solution.to_xarray().sizes["termination_boundary"] == 2


def test_dense_checker_detects_inside_outside_inside_excursion() -> None:
    def dense_solution(time):
        values = np.asarray(time, dtype=float)
        field = np.sin(2.0 * np.pi * values)
        if values.ndim == 0:
            return np.asarray([1.0, 0.0, field.item(), 0.0])
        return np.vstack((np.ones_like(values), np.zeros_like(values), field, np.zeros_like(values)))

    definition = _DomainEventDefinition(
        kind="maximum_absolute_field",
        threshold=0.5,
        units="M_pl",
        observed=lambda state: abs(float(state[2])),
        residual=lambda state: 0.5 - abs(float(state[2])),
    )
    assert definition.residual(dense_solution(0.0)) > 0.0
    assert definition.residual(dense_solution(1.0)) > 0.0

    exit_record = _first_dense_domain_exit(
        np.asarray([0.0, 1.0]),
        dense_solution,
        (definition,),
        substeps=32,
    )
    assert exit_record is not None
    exit_time, kind, state = exit_record
    assert kind == "maximum_absolute_field"
    assert exit_time == pytest.approx(1.0 / 12.0, rel=1.0e-8)
    assert abs(state[2]) == pytest.approx(0.5, rel=1.0e-8)


def test_corrupted_observed_value_is_result_integrity_error() -> None:
    solution = _integrate_de_sitter(
        branch=-1,
        domain=SCPCIntegrationDomain(min_scale_factor=0.8),
    )
    boundary = dict(solution.termination_boundaries[0])
    boundary["observed"] = 0.9
    solution.termination_boundaries = (boundary,)
    solution.termination_observed = 0.9
    with pytest.raises(ResultIntegrityError, match="disagrees with the final state"):
        assess_solution(solution)


def test_initial_state_outside_domain_is_rejected_before_solver_execution() -> None:
    with pytest.raises(ValueError, match="outside integration domain minimum_scale_factor"):
        integrate_scpc(
            _de_sitter_parameters(),
            t_span=(0.0, 1.0),
            samples=11,
            a0=1.0,
            phi0=0.0,
            phi_dot0=0.0,
            domain=SCPCIntegrationDomain(min_scale_factor=1.0),
            max_step=0.05,
            domain_check_substeps=8,
        )


def test_domain_requires_finite_max_step() -> None:
    with pytest.raises(ValueError, match="finite positive max_step"):
        integrate_scpc(
            _de_sitter_parameters(),
            t_span=(0.0, 1.0),
            samples=11,
            a0=1.0,
            phi0=0.0,
            phi_dot0=0.0,
            domain=SCPCIntegrationDomain(max_scale_factor=2.0),
        )


def test_coordinate_field_bound_is_rejected_for_circular_target() -> None:
    with pytest.raises(ValueError, match="coordinate-dependent"):
        integrate_scpc(
            _de_sitter_parameters(target_space="circle"),
            t_span=(0.0, 1.0),
            samples=11,
            a0=1.0,
            phi0=0.0,
            phi_dot0=0.0,
            domain=SCPCIntegrationDomain(max_abs_field=1.0),
            max_step=0.05,
            domain_check_substeps=8,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_domain_thresholds_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        SCPCIntegrationDomain(max_total_density=value)
