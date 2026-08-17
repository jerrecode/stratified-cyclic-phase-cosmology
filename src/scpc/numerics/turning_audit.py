"""Independent physics audit of exact homogeneous turning-point evidence.

The audit recomputes the load-bearing H=0 identities from the serialized event
state.  It is deliberately diagnostic: it can falsify an inconsistent event
record, but a consistent turning point is not evidence for recurrence or
stability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from scpc.models.phase import SCPCSolution


@dataclass(frozen=True)
class TurningPointAudit:
    """Recomputed local physics at one exact Hubble-zero event."""

    index: int
    time: float
    recorded_kind: str
    theorem_kind: str
    scale_factor: float
    hubble: float
    field: float
    field_velocity: float
    potential: float
    rho_m: float
    rho_r: float
    rho_phi: float
    p_phi: float
    rho_total: float
    p_total: float
    nec_density: float
    sec_density: float
    friedmann_surface_residual: float
    normalized_friedmann_surface_residual: float
    hdot_raychaudhuri: float
    hdot_turning_identity: float
    hdot_identity_difference: float
    bounce_margin: float
    classification_tolerance: float
    kind_consistent: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _classification_tolerance(*terms: float) -> float:
    """Roundoff-scale tolerance for a density-valued turning-point identity.

    Solver tolerances have state-dependent dimensions and therefore are not
    silently reused as density tolerances here.  This audit reports the local
    identity rather than replacing convergence tests.
    """

    scale = max(*(abs(float(term)) for term in terms), np.finfo(float).tiny)
    return float(512.0 * np.finfo(float).eps * scale)


def audit_turning_points(solution: SCPCSolution) -> tuple[TurningPointAudit, ...]:
    """Recompute local turning-point identities from exact event states.

    Exact ``turning_state_vectors`` are mandatory.  Interpolating the plotting
    trajectory would defeat the purpose of authenticating the root-localized
    event evidence.
    """

    times = np.asarray(solution.turning_times, dtype=float)
    kinds = tuple(solution.turning_kinds)
    if times.ndim != 1 or times.size != len(kinds):
        raise ValueError("turning times and kinds must be one-dimensional and equal length")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("turning times must be finite and strictly increasing")
    if solution.turning_state_vectors is None:
        if times.size:
            raise ValueError("exact turning_state_vectors are required for the physics audit")
        return ()
    states = np.asarray(solution.turning_state_vectors, dtype=float)
    if states.shape != (times.size, 4) or np.any(~np.isfinite(states)):
        raise ValueError("turning_state_vectors must be finite with shape (events, 4)")

    p = solution.parameters
    audits: list[TurningPointAudit] = []
    for index, (time, kind, state) in enumerate(zip(times, kinds, states, strict=True)):
        a, hubble, phi, velocity = (float(value) for value in state)
        if a <= 0.0:
            raise ValueError("turning-point scale factor must be positive")
        if kind not in {"bounce", "turnaround", "degenerate"}:
            raise ValueError(f"unknown turning-point kind {kind!r}")

        rho_m = float(p.matter_density(a))
        rho_r = float(p.radiation_density(a))
        potential = float(p.potential.value(phi))
        rho_phi = 0.5 * velocity**2 + potential
        p_phi = 0.5 * velocity**2 - potential
        rho_total = rho_m + rho_r + rho_phi
        p_total = rho_r / 3.0 + p_phi

        friedmann_surface_residual = (
            hubble**2 + p.spatial_curvature_k / a**2 - rho_total / 3.0
        )
        friedmann_scale = max(
            abs(hubble**2),
            abs(p.spatial_curvature_k / a**2),
            abs(rho_total / 3.0),
            np.finfo(float).tiny,
        )
        normalized_friedmann = friedmann_surface_residual / friedmann_scale

        nec_density = rho_total + p_total
        sec_density = rho_total + 3.0 * p_total
        hdot_raychaudhuri = p.spatial_curvature_k / a**2 - 0.5 * (
            rho_m + 4.0 * rho_r / 3.0 + velocity**2
        )
        hdot_turning_identity = -sec_density / 6.0
        bounce_margin = potential - 0.5 * rho_m - rho_r - velocity**2
        tolerance = _classification_tolerance(
            potential,
            0.5 * rho_m,
            rho_r,
            velocity**2,
            bounce_margin,
        )
        if bounce_margin > tolerance:
            theorem_kind = "bounce"
        elif bounce_margin < -tolerance:
            theorem_kind = "turnaround"
        else:
            theorem_kind = "degenerate"

        audits.append(
            TurningPointAudit(
                index=index,
                time=float(time),
                recorded_kind=str(kind),
                theorem_kind=theorem_kind,
                scale_factor=a,
                hubble=hubble,
                field=phi,
                field_velocity=velocity,
                potential=potential,
                rho_m=rho_m,
                rho_r=rho_r,
                rho_phi=rho_phi,
                p_phi=p_phi,
                rho_total=rho_total,
                p_total=p_total,
                nec_density=nec_density,
                sec_density=sec_density,
                friedmann_surface_residual=float(friedmann_surface_residual),
                normalized_friedmann_surface_residual=float(normalized_friedmann),
                hdot_raychaudhuri=float(hdot_raychaudhuri),
                hdot_turning_identity=float(hdot_turning_identity),
                hdot_identity_difference=float(hdot_raychaudhuri - hdot_turning_identity),
                bounce_margin=float(bounce_margin),
                classification_tolerance=tolerance,
                kind_consistent=bool(theorem_kind == kind),
            )
        )
    return tuple(audits)
