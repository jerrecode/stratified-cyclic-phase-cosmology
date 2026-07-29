"""Canonical covariant stratification-field baseline in FLRW spacetime.

Natural units c = hbar = M_pl = 1 are used internally. This module is a
background-theory baseline, not a claim that stable cyclic solutions exist.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Callable, Literal

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


DOMAIN_TERMINATION_KINDS = (
    "maximum_absolute_field",
    "maximum_absolute_field_velocity",
    "maximum_absolute_hubble",
    "maximum_absolute_ricci_scalar",
    "maximum_scale_factor",
    "maximum_total_density",
    "minimum_scale_factor",
)
DOMAIN_TERMINATION_KIND_CODES = {
    kind: index for index, kind in enumerate(DOMAIN_TERMINATION_KINDS, start=1)
}


@dataclass(frozen=True)
class PeriodicPotential:
    """Periodic scalar potential with an explicitly declared target-space topology."""

    offset: float = 3.0
    amplitude: float = 0.1
    strata_count: int = 4
    field_scale: float = 1.0
    target_space: Literal["real", "circle"] = "real"

    def __post_init__(self) -> None:
        if isinstance(self.strata_count, bool) or not isinstance(self.strata_count, Integral):
            raise ValueError("strata_count must be an integer")
        if self.amplitude < 0 or self.field_scale <= 0 or self.strata_count < 1:
            raise ValueError("Potential parameters are outside their allowed domain")
        if self.target_space not in ("real", "circle"):
            raise ValueError("target_space must be 'real' or 'circle'")

    @property
    def potential_period(self) -> float:
        return float(2.0 * np.pi * self.field_scale / self.strata_count)

    @property
    def target_circumference(self) -> float | None:
        if self.target_space == "circle":
            return float(2.0 * np.pi * self.field_scale)
        return None

    def value(self, phi: np.ndarray | float) -> np.ndarray:
        x = self.strata_count * np.asarray(phi, dtype=float) / self.field_scale
        return self.offset + self.amplitude * (1.0 - np.cos(x))

    def derivative(self, phi: np.ndarray | float) -> np.ndarray:
        x = self.strata_count * np.asarray(phi, dtype=float) / self.field_scale
        return self.amplitude * self.strata_count * np.sin(x) / self.field_scale


@dataclass(frozen=True)
class SCPCParameters:
    spatial_curvature_k: int = 1
    rho_m_ref: float = 0.03
    rho_r_ref: float = 0.0
    a_ref: float = 1.0
    potential: PeriodicPotential = PeriodicPotential()

    def __post_init__(self) -> None:
        if self.spatial_curvature_k not in (-1, 0, 1):
            raise ValueError("spatial_curvature_k must be -1, 0, or 1")
        if self.rho_m_ref < 0 or self.rho_r_ref < 0 or self.a_ref <= 0:
            raise ValueError("Reference densities must be non-negative and a_ref positive")

    def matter_density(self, a: np.ndarray | float) -> np.ndarray:
        return self.rho_m_ref * (self.a_ref / np.asarray(a, dtype=float)) ** 3

    def radiation_density(self, a: np.ndarray | float) -> np.ndarray:
        return self.rho_r_ref * (self.a_ref / np.asarray(a, dtype=float)) ** 4


@dataclass(frozen=True)
class SCPCIntegrationDomain:
    """Declared physical-analysis boundaries for one background run.

    Event detection is backed by a finite solver ``max_step`` and dense
    substep checks. These limits are analysis boundaries, not singularity
    claims.
    """

    min_scale_factor: float | None = None
    max_scale_factor: float | None = None
    max_total_density: float | None = None
    max_abs_hubble: float | None = None
    max_abs_ricci_scalar: float | None = None
    max_abs_field: float | None = None
    max_abs_field_velocity: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when configured")
        if (
            self.min_scale_factor is not None
            and self.max_scale_factor is not None
            and self.min_scale_factor >= self.max_scale_factor
        ):
            raise ValueError("min_scale_factor must be smaller than max_scale_factor")

    @property
    def configured(self) -> bool:
        return any(value is not None for value in asdict(self).values())

    def validate_for(self, parameters: SCPCParameters) -> None:
        if self.max_abs_field is not None and parameters.potential.target_space == "circle":
            raise ValueError(
                "max_abs_field is coordinate-dependent on a circular target; "
                "use an invariant compact-target criterion instead"
            )


@dataclass
class SCPCSolution:
    t: np.ndarray
    a: np.ndarray
    H: np.ndarray
    phi: np.ndarray
    phi_dot: np.ndarray
    rho_m: np.ndarray
    rho_r: np.ndarray
    rho_phi: np.ndarray
    p_phi: np.ndarray
    constraint_residual: np.ndarray
    turning_times: np.ndarray
    turning_kinds: tuple[str, ...]
    parameters: SCPCParameters
    solver_metadata: dict[str, Any]
    turning_state_vectors: np.ndarray | None = None
    termination_kind: str | None = None
    termination_time: float | None = None
    termination_state_vector: np.ndarray | None = None
    termination_threshold: float | None = None
    termination_observed: float | None = None
    termination_units: str | None = None
    termination_boundaries: tuple[dict[str, Any], ...] = ()
    requested_end_time: float | None = None

    @property
    def completed_to_requested_end(self) -> bool:
        if self.requested_end_time is None or self.t.size == 0:
            return self.termination_kind is None
        scale = max(1.0, abs(float(self.requested_end_time)))
        return bool(
            self.termination_kind is None
            and np.isclose(
                float(self.t[-1]),
                float(self.requested_end_time),
                rtol=0.0,
                atol=64.0 * np.finfo(float).eps * scale,
            )
        )

    def _termination_state(self) -> np.ndarray | None:
        if self.termination_kind is None:
            fields = (
                self.termination_time,
                self.termination_state_vector,
                self.termination_threshold,
                self.termination_observed,
                self.termination_units,
            )
            if any(value is not None for value in fields) or self.termination_boundaries:
                raise ValueError("Nonterminated solution contains partial termination metadata")
            return None
        if self.termination_kind not in DOMAIN_TERMINATION_KINDS:
            raise ValueError(f"Unknown termination kind: {self.termination_kind}")
        if (
            self.termination_time is None
            or self.termination_state_vector is None
            or self.termination_threshold is None
            or self.termination_observed is None
            or not self.termination_units
            or not self.termination_boundaries
        ):
            raise ValueError("Terminated solutions require complete termination metadata")
        state = np.asarray(self.termination_state_vector, dtype=float)
        if state.shape != (4,) or np.any(~np.isfinite(state)):
            raise ValueError("termination_state_vector must be a finite array with shape (4,)")
        kinds = tuple(str(boundary["kind"]) for boundary in self.termination_boundaries)
        if kinds != tuple(sorted(set(kinds))):
            raise ValueError("termination_boundaries must be unique and lexically ordered")
        primary = self.termination_boundaries[0]
        if (
            primary["kind"] != self.termination_kind
            or float(primary["threshold"]) != float(self.termination_threshold)
            or float(primary["observed"]) != float(self.termination_observed)
            or primary["units"] != self.termination_units
        ):
            raise ValueError("Primary termination fields must match the first boundary record")
        return state

    def to_xarray(self) -> xr.Dataset:
        termination_state = self._termination_state()
        attrs: dict[str, Any] = {
            "model": "canonical_scpc_phase_baseline",
            "unit_system": "natural units c=hbar=M_pl=1",
            "max_abs_constraint_residual": float(np.max(np.abs(self.constraint_residual))),
            "parameters": str(asdict(self.parameters)),
            "completed_to_requested_end": int(self.completed_to_requested_end),
            **self.solver_metadata,
        }
        if self.requested_end_time is not None:
            attrs["requested_end_time"] = float(self.requested_end_time)
        if self.termination_kind is not None:
            attrs.update(
                {
                    "termination_kind": self.termination_kind,
                    "termination_threshold": float(self.termination_threshold),
                    "termination_observed": float(self.termination_observed),
                    "termination_units": str(self.termination_units),
                    "termination_boundaries": json.dumps(
                        list(self.termination_boundaries),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

        dataset = xr.Dataset(
            data_vars={
                "scale_factor": ("time", self.a, {"units": "1"}),
                "hubble": ("time", self.H, {"units": "M_pl"}),
                "stratification_field": ("time", self.phi, {"units": "M_pl"}),
                "field_velocity": ("time", self.phi_dot, {"units": "M_pl^2"}),
                "matter_density": ("time", self.rho_m, {"units": "M_pl^4"}),
                "radiation_density": ("time", self.rho_r, {"units": "M_pl^4"}),
                "field_density": ("time", self.rho_phi, {"units": "M_pl^4"}),
                "field_pressure": ("time", self.p_phi, {"units": "M_pl^4"}),
                "friedmann_constraint_residual": (
                    "time",
                    self.constraint_residual,
                    {"units": "1"},
                ),
            },
            coords={"time": ("time", self.t, {"units": "M_pl^-1"})},
            attrs=attrs,
        )

        if self.turning_state_vectors is not None:
            states = np.asarray(self.turning_state_vectors, dtype=float)
            if states.ndim != 2 or states.shape[1] != 4:
                raise ValueError("turning_state_vectors must have shape (events, 4)")
            if states.shape[0] != len(self.turning_times) or states.shape[0] != len(self.turning_kinds):
                raise ValueError("Turning times, kinds, and state vectors must have equal lengths")
            if states.shape[0] > 0:
                kind_codes = np.asarray(
                    [
                        1 if kind == "bounce" else -1 if kind == "turnaround" else 0
                        for kind in self.turning_kinds
                    ],
                    dtype=np.int8,
                )
                dataset = dataset.assign_coords(turning_event=np.arange(states.shape[0], dtype=int))
                dataset["turning_time"] = (
                    "turning_event",
                    self.turning_times,
                    {"units": "M_pl^-1"},
                )
                dataset["turning_scale_factor"] = (
                    "turning_event",
                    states[:, 0],
                    {"units": "1"},
                )
                dataset["turning_hubble"] = (
                    "turning_event",
                    states[:, 1],
                    {"units": "M_pl"},
                )
                dataset["turning_field"] = (
                    "turning_event",
                    states[:, 2],
                    {"units": "M_pl"},
                )
                dataset["turning_field_velocity"] = (
                    "turning_event",
                    states[:, 3],
                    {"units": "M_pl^2"},
                )
                dataset["turning_kind_code"] = (
                    "turning_event",
                    kind_codes,
                    {"codes": "1=bounce,-1=turnaround,0=degenerate"},
                )

        if termination_state is not None:
            dataset["termination_time"] = xr.DataArray(
                float(self.termination_time),
                attrs={"units": "M_pl^-1"},
            )
            dataset["termination_scale_factor"] = xr.DataArray(
                termination_state[0],
                attrs={"units": "1"},
            )
            dataset["termination_hubble"] = xr.DataArray(
                termination_state[1],
                attrs={"units": "M_pl"},
            )
            dataset["termination_field"] = xr.DataArray(
                termination_state[2],
                attrs={"units": "M_pl"},
            )
            dataset["termination_field_velocity"] = xr.DataArray(
                termination_state[3],
                attrs={"units": "M_pl^2"},
            )
            boundaries = self.termination_boundaries
            dataset = dataset.assign_coords(
                termination_boundary=np.arange(len(boundaries), dtype=int)
            )
            dataset["termination_boundary_kind_code"] = (
                "termination_boundary",
                np.asarray(
                    [DOMAIN_TERMINATION_KIND_CODES[str(item["kind"])] for item in boundaries],
                    dtype=np.int16,
                ),
                {"codes": json.dumps(DOMAIN_TERMINATION_KIND_CODES, sort_keys=True)},
            )
            dataset["termination_boundary_threshold"] = (
                "termination_boundary",
                np.asarray([float(item["threshold"]) for item in boundaries]),
            )
            dataset["termination_boundary_observed"] = (
                "termination_boundary",
                np.asarray([float(item["observed"]) for item in boundaries]),
            )
            dataset["termination_boundary_threshold"].attrs["units_by_boundary"] = json.dumps(
                [str(item["units"]) for item in boundaries]
            )
            dataset["termination_boundary_observed"].attrs["units_by_boundary"] = json.dumps(
                [str(item["units"]) for item in boundaries]
            )
        return dataset


def _components(a: np.ndarray, phi: np.ndarray, v: np.ndarray, p: SCPCParameters):
    rho_m = p.matter_density(a)
    rho_r = p.radiation_density(a)
    potential = p.potential.value(phi)
    rho_phi = 0.5 * v**2 + potential
    p_phi = 0.5 * v**2 - potential
    return rho_m, rho_r, rho_phi, p_phi


def initial_hubble(
    a0: float,
    phi0: float,
    phi_dot0: float,
    parameters: SCPCParameters,
    branch: int = 1,
) -> float:
    if a0 <= 0 or branch not in (-1, 1):
        raise ValueError("a0 must be positive and branch must be +1 or -1")
    rho_m, rho_r, rho_phi, _ = _components(
        np.asarray(a0), np.asarray(phi0), np.asarray(phi_dot0), parameters
    )
    h2 = (rho_m + rho_r + rho_phi) / 3.0 - parameters.spatial_curvature_k / a0**2
    if h2 < 0:
        raise ValueError(f"Initial state violates the Friedmann constraint: H^2={float(h2)}")
    return float(branch * np.sqrt(h2))


def _rhs(_t: float, y: np.ndarray, p: SCPCParameters) -> np.ndarray:
    a, H, phi, v = y
    if a <= 0:
        raise FloatingPointError("Scale factor reached a non-positive value")
    rho_m = float(p.matter_density(a))
    rho_r = float(p.radiation_density(a))
    return np.asarray(
        [
            a * H,
            p.spatial_curvature_k / a**2 - 0.5 * (rho_m + 4.0 * rho_r / 3.0 + v**2),
            v,
            -3.0 * H * v - float(p.potential.derivative(phi)),
        ]
    )


def _total_density(state: np.ndarray, parameters: SCPCParameters) -> float:
    a, _H, phi, v = state
    rho_m, rho_r, rho_phi, _ = _components(
        np.asarray(a), np.asarray(phi), np.asarray(v), parameters
    )
    return float(rho_m + rho_r + rho_phi)


def _ricci_scalar(state: np.ndarray, parameters: SCPCParameters) -> float:
    a, H, _phi, _v = state
    hdot = float(_rhs(0.0, state, parameters)[1])
    return float(6.0 * (hdot + 2.0 * H**2 + parameters.spatial_curvature_k / a**2))


def evaluate_domain_boundary(
    kind: str,
    state: np.ndarray,
    parameters: SCPCParameters,
) -> tuple[float, str]:
    """Recompute one declared boundary observable from a state vector."""

    if kind == "minimum_scale_factor" or kind == "maximum_scale_factor":
        return float(state[0]), "1"
    if kind == "maximum_total_density":
        return _total_density(state, parameters), "M_pl^4"
    if kind == "maximum_absolute_hubble":
        return abs(float(state[1])), "M_pl"
    if kind == "maximum_absolute_ricci_scalar":
        return abs(_ricci_scalar(state, parameters)), "M_pl^2"
    if kind == "maximum_absolute_field":
        return abs(float(state[2])), "M_pl"
    if kind == "maximum_absolute_field_velocity":
        return abs(float(state[3])), "M_pl^2"
    raise ValueError(f"Unknown domain boundary kind: {kind}")


@dataclass(frozen=True)
class _DomainEventDefinition:
    kind: str
    threshold: float
    units: str
    observed: Callable[[np.ndarray], float]
    residual: Callable[[np.ndarray], float]


def _domain_event_definitions(
    domain: SCPCIntegrationDomain,
    parameters: SCPCParameters,
) -> tuple[_DomainEventDefinition, ...]:
    domain.validate_for(parameters)
    definitions: list[_DomainEventDefinition] = []

    def add(
        kind: str,
        threshold: float | None,
        units: str,
        observed: Callable[[np.ndarray], float],
        residual: Callable[[np.ndarray], float],
    ) -> None:
        if threshold is not None:
            definitions.append(
                _DomainEventDefinition(
                    kind=kind,
                    threshold=float(threshold),
                    units=units,
                    observed=observed,
                    residual=residual,
                )
            )

    add(
        "minimum_scale_factor",
        domain.min_scale_factor,
        "1",
        lambda state: float(state[0]),
        lambda state: float(state[0] - domain.min_scale_factor),
    )
    add(
        "maximum_scale_factor",
        domain.max_scale_factor,
        "1",
        lambda state: float(state[0]),
        lambda state: float(domain.max_scale_factor - state[0]),
    )
    add(
        "maximum_total_density",
        domain.max_total_density,
        "M_pl^4",
        lambda state: _total_density(state, parameters),
        lambda state: float(domain.max_total_density - _total_density(state, parameters)),
    )
    add(
        "maximum_absolute_hubble",
        domain.max_abs_hubble,
        "M_pl",
        lambda state: abs(float(state[1])),
        lambda state: float(domain.max_abs_hubble - abs(state[1])),
    )
    add(
        "maximum_absolute_ricci_scalar",
        domain.max_abs_ricci_scalar,
        "M_pl^2",
        lambda state: abs(_ricci_scalar(state, parameters)),
        lambda state: float(domain.max_abs_ricci_scalar - abs(_ricci_scalar(state, parameters))),
    )
    add(
        "maximum_absolute_field",
        domain.max_abs_field,
        "M_pl",
        lambda state: abs(float(state[2])),
        lambda state: float(domain.max_abs_field - abs(state[2])),
    )
    add(
        "maximum_absolute_field_velocity",
        domain.max_abs_field_velocity,
        "M_pl^2",
        lambda state: abs(float(state[3])),
        lambda state: float(domain.max_abs_field_velocity - abs(state[3])),
    )
    return tuple(sorted(definitions, key=lambda definition: definition.kind))


def _make_terminal_event(definition: _DomainEventDefinition):
    def event(_t: float, state: np.ndarray) -> float:
        return definition.residual(state)

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1  # type: ignore[attr-defined]
    return event


def _event_tolerance(rtol: float, atol: float, scale: float) -> float:
    magnitude = abs(float(scale))
    roundoff = 64.0 * np.finfo(float).eps * max(magnitude, np.finfo(float).tiny)
    return roundoff + 128.0 * (abs(float(atol)) + abs(float(rtol)) * magnitude)


def _first_dense_domain_exit(
    internal_times: np.ndarray,
    dense_solution: Callable[[np.ndarray | float], np.ndarray],
    definitions: tuple[_DomainEventDefinition, ...],
    *,
    substeps: int,
) -> tuple[float, str, np.ndarray] | None:
    """Find the first sampled dense-output exit within accepted solver steps.

    A finite solver max step and explicit substeps bound the unresolved time
    scale. Convergence in both controls is required for production scans.
    """

    earliest: tuple[float, str, np.ndarray] | None = None
    for left, right in zip(internal_times[:-1], internal_times[1:], strict=True):
        sample_times = np.linspace(float(left), float(right), substeps + 1)
        states = np.asarray(dense_solution(sample_times), dtype=float)
        for definition in definitions:
            residuals = np.asarray(
                [definition.residual(states[:, index]) for index in range(states.shape[1])]
            )
            for index in range(substeps):
                left_value = float(residuals[index])
                right_value = float(residuals[index + 1])
                if left_value <= 0.0:
                    candidate_time = float(sample_times[index])
                    candidate_state = states[:, index]
                elif right_value <= 0.0:
                    left_time = float(sample_times[index])
                    right_time = float(sample_times[index + 1])
                    candidate_time = float(
                        brentq(
                            lambda time: definition.residual(
                                np.asarray(dense_solution(time), dtype=float)
                            ),
                            left_time,
                            right_time,
                        )
                    )
                    candidate_state = np.asarray(dense_solution(candidate_time), dtype=float)
                else:
                    continue
                candidate = (candidate_time, definition.kind, candidate_state)
                if earliest is None or candidate_time < earliest[0]:
                    earliest = candidate
                break
        if earliest is not None and earliest[0] <= right:
            break
    return earliest


def _coincident_boundary_records(
    state: np.ndarray,
    definitions: tuple[_DomainEventDefinition, ...],
    *,
    forced_kind: str,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for definition in definitions:
        observed = float(definition.observed(state))
        tolerance = _event_tolerance(rtol, atol, max(abs(observed), definition.threshold))
        if definition.kind == forced_kind or abs(observed - definition.threshold) <= tolerance:
            records.append(
                {
                    "kind": definition.kind,
                    "threshold": definition.threshold,
                    "observed": observed,
                    "units": definition.units,
                }
            )
    return tuple(sorted(records, key=lambda record: str(record["kind"])))


def _evaluation_grid(
    t_span: tuple[float, float],
    samples: int,
    end_time: float,
) -> np.ndarray:
    requested = np.linspace(t_span[0], t_span[1], samples)
    scale = max(1.0, abs(end_time))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if np.isclose(end_time, t_span[1], rtol=0.0, atol=tolerance):
        return requested
    before = requested[requested < end_time - tolerance]
    return np.concatenate((before, np.asarray([end_time])))


def integrate_scpc(
    parameters: SCPCParameters,
    *,
    t_span: tuple[float, float],
    a0: float,
    phi0: float,
    phi_dot0: float,
    branch: int = 1,
    samples: int = 2001,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    method: str = "DOP853",
    domain: SCPCIntegrationDomain | None = None,
    max_step: float | None = None,
    domain_check_substeps: int = 16,
) -> SCPCSolution:
    """Integrate the homogeneous baseline with turning and checked domain events."""

    if samples < 2:
        raise ValueError("samples must be at least 2")
    if domain_check_substeps < 2 or isinstance(domain_check_substeps, bool):
        raise ValueError("domain_check_substeps must be an integer of at least 2")
    definitions = _domain_event_definitions(domain, parameters) if domain is not None else ()
    if definitions:
        if max_step is None or not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError(
                "A finite positive max_step is required when integration-domain boundaries are configured"
            )
        solver_max_step = float(max_step)
    else:
        solver_max_step = float(max_step) if max_step is not None else np.inf
        if not np.isfinite(solver_max_step) and max_step is not None:
            raise ValueError("max_step must be finite and positive when configured")
        if solver_max_step <= 0.0:
            raise ValueError("max_step must be finite and positive when configured")

    H0 = initial_hubble(a0, phi0, phi_dot0, parameters, branch)
    initial_state = np.asarray([a0, H0, phi0, phi_dot0], dtype=float)

    def turning_event(_t: float, state: np.ndarray) -> float:
        return float(state[1])

    turning_event.terminal = False  # type: ignore[attr-defined]
    turning_event.direction = 0  # type: ignore[attr-defined]

    for definition in definitions:
        initial_residual = definition.residual(initial_state)
        if not np.isfinite(initial_residual) or initial_residual <= 0.0:
            raise ValueError(
                f"Initial state is outside integration domain {definition.kind}: "
                f"observed={definition.observed(initial_state)}, threshold={definition.threshold}"
            )
    events = [turning_event, *[_make_terminal_event(definition) for definition in definitions]]

    sol = solve_ivp(
        lambda t, y: _rhs(t, y, parameters),
        t_span,
        initial_state,
        method=method,
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=True,
        max_step=solver_max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Background integration failed: {sol.message}")
    if sol.sol is None:
        raise RuntimeError("Background integration did not return dense output")

    reported_candidates: list[tuple[float, str, np.ndarray]] = []
    for event_index, definition in enumerate(definitions, start=1):
        if len(sol.t_events[event_index]):
            reported_candidates.append(
                (
                    float(sol.t_events[event_index][0]),
                    definition.kind,
                    np.asarray(sol.y_events[event_index][0], dtype=float),
                )
            )
    reported = min(reported_candidates, key=lambda candidate: (candidate[0], candidate[1])) if reported_candidates else None
    dense_exit = (
        _first_dense_domain_exit(
            np.asarray(sol.t, dtype=float),
            sol.sol,
            definitions,
            substeps=domain_check_substeps,
        )
        if definitions
        else None
    )
    candidates = [candidate for candidate in (reported, dense_exit) if candidate is not None]
    termination = min(candidates, key=lambda candidate: (candidate[0], candidate[1])) if candidates else None

    termination_kind: str | None = None
    termination_time: float | None = None
    termination_state: np.ndarray | None = None
    termination_threshold: float | None = None
    termination_observed: float | None = None
    termination_units: str | None = None
    termination_boundaries: tuple[dict[str, Any], ...] = ()
    if termination is not None:
        termination_time, forced_kind, termination_state = termination
        termination_state = np.asarray(sol.sol(termination_time), dtype=float)
        termination_boundaries = _coincident_boundary_records(
            termination_state,
            definitions,
            forced_kind=forced_kind,
            rtol=rtol,
            atol=atol,
        )
        primary = termination_boundaries[0]
        termination_kind = str(primary["kind"])
        termination_threshold = float(primary["threshold"])
        termination_observed = float(primary["observed"])
        termination_units = str(primary["units"])

    effective_end = termination_time if termination_time is not None else float(t_span[1])
    times = _evaluation_grid(t_span, samples, effective_end)
    states = np.asarray(sol.sol(times), dtype=float)
    if termination_state is not None:
        states[:, -1] = termination_state

    a, H, phi, v = states
    rho_m, rho_r, rho_phi, p_phi = _components(a, phi, v, parameters)
    lhs = H**2 + parameters.spatial_curvature_k / a**2
    rhs = (rho_m + rho_r + rho_phi) / 3.0
    scale = np.maximum(np.abs(rhs), 1e-15)
    residual = (lhs - rhs) / scale

    all_turning_times = np.asarray(sol.t_events[0] if sol.t_events else [], dtype=float)
    if sol.y_events and len(sol.y_events[0]) > 0:
        all_turning_states = np.asarray(sol.y_events[0], dtype=float)
    else:
        all_turning_states = np.empty((0, 4), dtype=float)
    if all_turning_states.shape != (all_turning_times.size, 4):
        raise RuntimeError("Solver returned inconsistent turning-event state data")
    endpoint_slack = 64.0 * np.finfo(float).eps * max(1.0, abs(effective_end))
    keep = all_turning_times <= effective_end + endpoint_slack
    event_times = all_turning_times[keep].copy()
    event_times[event_times > effective_end] = effective_end
    event_states = all_turning_states[keep]

    kinds: list[str] = []
    for event_time, state in zip(event_times, event_states, strict=True):
        hdot = _rhs(float(event_time), state, parameters)[1]
        kinds.append("bounce" if hdot > 0 else "turnaround" if hdot < 0 else "degenerate")

    return SCPCSolution(
        t=times,
        a=a,
        H=H,
        phi=phi,
        phi_dot=v,
        rho_m=rho_m,
        rho_r=rho_r,
        rho_phi=rho_phi,
        p_phi=p_phi,
        constraint_residual=residual,
        turning_times=event_times,
        turning_kinds=tuple(kinds),
        parameters=parameters,
        solver_metadata={
            "solver_method": method,
            "solver_rtol": rtol,
            "solver_atol": atol,
            "solver_nfev": int(sol.nfev),
            "solver_status": int(sol.status),
            "solver_max_step": solver_max_step if np.isfinite(solver_max_step) else "unbounded",
            "domain_check_substeps": int(domain_check_substeps),
            "requested_end_time": float(t_span[1]),
            "reached_end_time": float(times[-1]),
            "integration_domain": json.dumps(
                asdict(domain) if domain is not None else {},
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        turning_state_vectors=event_states,
        termination_kind=termination_kind,
        termination_time=termination_time,
        termination_state_vector=termination_state,
        termination_threshold=termination_threshold,
        termination_observed=termination_observed,
        termination_units=termination_units,
        termination_boundaries=termination_boundaries,
        requested_end_time=float(t_span[1]),
    )
