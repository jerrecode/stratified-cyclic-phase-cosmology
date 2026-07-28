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


DOMAIN_TERMINATION_KINDS = (
    "minimum_scale_factor",
    "maximum_scale_factor",
    "maximum_total_density",
    "maximum_absolute_hubble",
    "maximum_absolute_ricci_scalar",
    "maximum_absolute_field",
    "maximum_absolute_field_velocity",
)


@dataclass(frozen=True)
class PeriodicPotential:
    """Periodic scalar potential with an explicitly declared target-space topology.

    A periodic potential on the real line does not identify field values that
    differ by a potential period. When ``target_space`` is ``"circle"``, only
    shifts by the full target circumference ``2*pi*field_scale`` identify the
    same point; adjacent potential minima remain distinct strata.
    """

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
        """Field displacement between adjacent equivalent potential cells."""

        return float(2.0 * np.pi * self.field_scale / self.strata_count)

    @property
    def target_circumference(self) -> float | None:
        """Full compact target circumference, or ``None`` for a real scalar."""

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
    """Declared numerical/physical domain boundaries for one background run.

    Reaching one of these limits terminates the integration at a root-localized
    event. A limit is a declared analysis boundary, not a singularity claim.
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
            if any(value is not None for value in fields):
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
        ):
            raise ValueError("Terminated solutions require complete termination metadata")
        state = np.asarray(self.termination_state_vector, dtype=float)
        if state.shape != (4,) or np.any(~np.isfinite(state)):
            raise ValueError("termination_state_vector must be a finite array with shape (4,)")
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
                "friedmann_constraint_residual": ("time", self.constraint_residual, {"units": "1"}),
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
                    [1 if kind == "bounce" else -1 if kind == "turnaround" else 0 for kind in self.turning_kinds],
                    dtype=np.int8,
                )
                dataset = dataset.assign_coords(turning_event=np.arange(states.shape[0], dtype=int))
                dataset["turning_time"] = ("turning_event", self.turning_times, {"units": "M_pl^-1"})
                dataset["turning_scale_factor"] = ("turning_event", states[:, 0], {"units": "1"})
                dataset["turning_hubble"] = ("turning_event", states[:, 1], {"units": "M_pl"})
                dataset["turning_field"] = ("turning_event", states[:, 2], {"units": "M_pl"})
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
    return tuple(definitions)


def _make_terminal_event(definition: _DomainEventDefinition):
    def event(_t: float, state: np.ndarray) -> float:
        return definition.residual(state)

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1  # type: ignore[attr-defined]
    return event


def _append_exact_endpoint(
    times: np.ndarray,
    states: np.ndarray,
    event_time: float,
    event_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = max(1.0, abs(event_time))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if times.size and np.isclose(times[-1], event_time, rtol=0.0, atol=tolerance):
        exact_times = times.copy()
        exact_states = states.copy()
        exact_times[-1] = event_time
        exact_states[:, -1] = event_state
        return exact_times, exact_states
    return (
        np.concatenate((times, np.asarray([event_time]))),
        np.concatenate((states, np.asarray(event_state, dtype=float)[:, None]), axis=1),
    )


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
) -> SCPCSolution:
    """Integrate the homogeneous baseline with turning and domain events."""

    H0 = initial_hubble(a0, phi0, phi_dot0, parameters, branch)
    initial_state = np.asarray([a0, H0, phi0, phi_dot0], dtype=float)
    t_eval = np.linspace(t_span[0], t_span[1], samples)

    def turning_event(_t: float, state: np.ndarray) -> float:
        return float(state[1])

    turning_event.terminal = False  # type: ignore[attr-defined]
    turning_event.direction = 0  # type: ignore[attr-defined]

    definitions = _domain_event_definitions(domain, parameters) if domain is not None else ()
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
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Background integration failed: {sol.message}")

    termination_kind: str | None = None
    termination_time: float | None = None
    termination_state: np.ndarray | None = None
    termination_threshold: float | None = None
    termination_observed: float | None = None
    termination_units: str | None = None
    terminated_events: list[tuple[float, int, np.ndarray]] = []
    for event_index, definition in enumerate(definitions, start=1):
        event_times = sol.t_events[event_index]
        event_states = sol.y_events[event_index]
        if len(event_times):
            terminated_events.append((float(event_times[0]), event_index - 1, np.asarray(event_states[0])))
    if terminated_events:
        termination_time, definition_index, termination_state = min(
            terminated_events,
            key=lambda item: item[0],
        )
        definition = definitions[definition_index]
        termination_kind = definition.kind
        termination_threshold = definition.threshold
        termination_observed = definition.observed(termination_state)
        termination_units = definition.units

    times = np.asarray(sol.t, dtype=float)
    states = np.asarray(sol.y, dtype=float)
    if termination_time is not None and termination_state is not None:
        times, states = _append_exact_endpoint(times, states, termination_time, termination_state)

    a, H, phi, v = states
    rho_m, rho_r, rho_phi, p_phi = _components(a, phi, v, parameters)
    lhs = H**2 + parameters.spatial_curvature_k / a**2
    rhs = (rho_m + rho_r + rho_phi) / 3.0
    scale = np.maximum(np.abs(rhs), 1e-15)
    residual = (lhs - rhs) / scale

    event_times = np.asarray(sol.t_events[0] if sol.t_events else [], dtype=float)
    if sol.y_events and len(sol.y_events[0]) > 0:
        event_states = np.asarray(sol.y_events[0], dtype=float)
    else:
        event_states = np.empty((0, 4), dtype=float)
    if event_states.shape != (event_times.size, 4):
        raise RuntimeError("Solver returned inconsistent turning-event state data")

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
        requested_end_time=float(t_span[1]),
    )
