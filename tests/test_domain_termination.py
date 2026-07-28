import numpy as np
import pytest
import xarray as xr

from scpc.models.phase import (
    PeriodicPotential,
    SCPCIntegrationDomain,
    SCPCParameters,
    integrate_scpc,
)


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


def test_contracting_de_sitter_stops_at_exact_minimum_scale_factor() -> None:
    solution = integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 1.0),
        samples=11,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=-1,
        rtol=1.0e-11,
        atol=1.0e-13,
        domain=SCPCIntegrationDomain(min_scale_factor=0.8),
    )

    expected_time = -np.log(0.8)
    assert solution.termination_kind == "minimum_scale_factor"
    assert np.isclose(solution.termination_time, expected_time, rtol=1.0e-9, atol=1.0e-11)
    assert np.isclose(solution.t[-1], expected_time, rtol=1.0e-9, atol=1.0e-11)
    assert np.isclose(solution.a[-1], 0.8, rtol=1.0e-10, atol=1.0e-12)
    assert np.isclose(solution.termination_observed, 0.8, rtol=1.0e-10)
    assert solution.termination_threshold == 0.8
    assert solution.termination_units == "1"
    assert solution.completed_to_requested_end is False


def test_expanding_de_sitter_stops_at_exact_maximum_scale_factor() -> None:
    solution = integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 1.0),
        samples=7,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=1,
        domain=SCPCIntegrationDomain(max_scale_factor=1.25),
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


def test_termination_metadata_and_exact_state_are_scipy_netcdf_serializable(tmp_path) -> None:
    solution = integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 1.0),
        samples=9,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=-1,
        domain=SCPCIntegrationDomain(min_scale_factor=0.85),
    )
    dataset = solution.to_xarray()

    assert dataset.attrs["termination_kind"] == "minimum_scale_factor"
    assert dataset.attrs["completed_to_requested_end"] == 0
    assert float(dataset["termination_time"]) == pytest.approx(solution.termination_time)
    assert float(dataset["termination_scale_factor"]) == pytest.approx(0.85)
    assert float(dataset["termination_hubble"]) == pytest.approx(-1.0)

    destination = tmp_path / "terminated.nc"
    dataset.to_netcdf(destination, engine="scipy")
    restored = xr.load_dataset(destination, engine="scipy")
    assert restored.attrs["termination_kind"] == "minimum_scale_factor"
    assert restored.attrs["completed_to_requested_end"] == 0
    assert float(restored["termination_scale_factor"]) == pytest.approx(0.85)


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
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_domain_thresholds_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        SCPCIntegrationDomain(max_total_density=value)
