from types import SimpleNamespace

import numpy as np
import pytest

import scpc.models.phase as phase_module
from scpc.models.phase import (
    PeriodicPotential,
    SCPCIntegrationDomain,
    SCPCParameters,
    integrate_scpc,
)
from scpc.scans.errors import ResultIntegrityError
from scpc.scans.outcomes import OutcomeClass, assess_solution


def _de_sitter_parameters() -> SCPCParameters:
    return SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0, strata_count=1),
    )


def _contracting_de_sitter_solution():
    return integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 1.0),
        samples=17,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=-1,
        rtol=1.0e-11,
        atol=1.0e-13,
        domain=SCPCIntegrationDomain(min_scale_factor=0.8),
        max_step=0.025,
        domain_check_substeps=16,
    )


def _coincident_flat_dust_solution():
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=3.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=0.0, amplitude=0.0, strata_count=1),
    )
    scale_threshold = 0.8
    hubble_threshold = scale_threshold ** (-1.5)
    return integrate_scpc(
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


def test_dense_termination_discards_later_solver_turning_event(monkeypatch) -> None:
    later_turning_time = 0.500001

    def dense_solution(time):
        values = np.asarray(time, dtype=float)
        scale_factor = np.exp(values)
        if values.ndim == 0:
            return np.asarray([float(scale_factor), 1.0, 0.0, 0.0])
        return np.vstack(
            (
                scale_factor,
                np.ones_like(values),
                np.zeros_like(values),
                np.zeros_like(values),
            )
        )

    def fake_solve_ivp(*_args, **_kwargs):
        return SimpleNamespace(
            success=True,
            message="success",
            sol=dense_solution,
            t=np.asarray([0.0, 1.0]),
            t_events=[
                np.asarray([later_turning_time]),
                np.asarray([], dtype=float),
            ],
            y_events=[
                np.asarray([[np.exp(later_turning_time), 0.0, 0.0, 0.0]]),
                np.empty((0, 4), dtype=float),
            ],
            nfev=8,
            status=0,
        )

    monkeypatch.setattr(phase_module, "solve_ivp", fake_solve_ivp)
    solution = integrate_scpc(
        _de_sitter_parameters(),
        t_span=(0.0, 1.0),
        samples=21,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        domain=SCPCIntegrationDomain(max_scale_factor=float(np.exp(0.5))),
        max_step=1.0,
        domain_check_substeps=16,
    )

    assert solution.termination_time == pytest.approx(0.5)
    assert solution.turning_times.size == 0
    assert solution.turning_kinds == ()
    assessment = assess_solution(solution)
    assert assessment.outcome is OutcomeClass.PHYSICAL_DOMAIN_TERMINATION


def test_missing_nonprimary_coincident_boundary_is_integrity_error() -> None:
    solution = _coincident_flat_dust_solution()
    assert len(solution.termination_boundaries) == 2
    solution.termination_boundaries = (solution.termination_boundaries[0],)

    with pytest.raises(ResultIntegrityError, match="configured coincident surfaces"):
        assess_solution(solution)


def test_undeclared_boundary_substitution_is_integrity_error() -> None:
    solution = _contracting_de_sitter_solution()
    substituted = {
        "kind": "maximum_scale_factor",
        "threshold": 0.8,
        "observed": 0.8,
        "units": "1",
    }
    solution.termination_boundaries = (substituted,)
    solution.termination_kind = "maximum_scale_factor"
    solution.termination_threshold = 0.8
    solution.termination_observed = 0.8
    solution.termination_units = "1"

    with pytest.raises(ResultIntegrityError, match="configured coincident surfaces"):
        assess_solution(solution)


@pytest.mark.parametrize("metadata", [None, "{not-json", "{}"])
def test_terminated_solution_requires_valid_configured_domain_metadata(metadata) -> None:
    solution = _contracting_de_sitter_solution()
    if metadata is None:
        solution.solver_metadata.pop("integration_domain")
    else:
        solution.solver_metadata["integration_domain"] = metadata

    with pytest.raises(ResultIntegrityError, match="integration-domain|configured"):
        assess_solution(solution)
