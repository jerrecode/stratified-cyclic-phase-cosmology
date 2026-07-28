"""Adaptive background integration and standardized result serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp

from scpc.numerics.diagnostics import ConstraintSummary, summarize_friedmann_constraint
from scpc.numerics.events import bounce_event, invalid_scale_factor_event, turnaround_event
from scpc.theory.background import BackgroundParameters, BackgroundState, background_rhs
from scpc.theory.potentials import ScalarPotential

VARIABLE_NAMES = (
    "scale_factor",
    "hubble",
    "phi",
    "phi_dot",
    "rho_matter",
    "rho_radiation",
)

VARIABLE_UNITS = {
    "scale_factor": "1",
    "hubble": "natural_time^-1",
    "phi": "natural_field",
    "phi_dot": "natural_field natural_time^-1",
    "rho_matter": "natural_energy_density",
    "rho_radiation": "natural_energy_density",
}


@dataclass(frozen=True)
class IntegrationSettings:
    time_start: float
    time_end: float
    samples: int = 2000
    method: str = "DOP853"
    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-12
    max_step: float = np.inf

    def __post_init__(self) -> None:
        if self.time_end <= self.time_start:
            raise ValueError("time_end must be greater than time_start")
        if self.samples < 2:
            raise ValueError("samples must be at least two")
        if self.relative_tolerance <= 0.0 or self.absolute_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")


@dataclass(frozen=True)
class BackgroundSolution:
    time: np.ndarray
    values: np.ndarray
    bounce_times: tuple[float, ...]
    turnaround_times: tuple[float, ...]
    solver_message: str
    function_evaluations: int
    constraint: ConstraintSummary

    def to_dataset(self) -> xr.Dataset:
        data_vars: dict[str, tuple[tuple[str], np.ndarray, dict[str, str]]] = {}
        for index, name in enumerate(VARIABLE_NAMES):
            data_vars[name] = (("time",), self.values[index], {"units": VARIABLE_UNITS[name]})
        dataset = xr.Dataset(data_vars=data_vars, coords={"time": self.time})
        dataset["time"].attrs["units"] = "natural_time"
        dataset.attrs.update(
            {
                "model": "canonical closed-FLRW SCPC baseline",
                "scientific_status": "exploratory baseline",
                "bounce_count": len(self.bounce_times),
                "turnaround_count": len(self.turnaround_times),
                "maximum_absolute_relative_friedmann_residual": self.constraint.maximum_absolute_relative_residual,
            }
        )
        return dataset


def integrate_background(
    initial_state: BackgroundState,
    parameters: BackgroundParameters,
    potential: ScalarPotential,
    settings: IntegrationSettings,
) -> BackgroundSolution:
    evaluation_times = np.linspace(settings.time_start, settings.time_end, settings.samples)

    def rhs(time: float, values: np.ndarray) -> np.ndarray:
        return background_rhs(time, values, parameters, potential)

    result = solve_ivp(
        rhs,
        (settings.time_start, settings.time_end),
        initial_state.as_array(),
        method=settings.method,
        t_eval=evaluation_times,
        rtol=settings.relative_tolerance,
        atol=settings.absolute_tolerance,
        max_step=settings.max_step,
        events=(bounce_event, turnaround_event, invalid_scale_factor_event),
        dense_output=False,
    )
    if not result.success:
        raise RuntimeError(f"Background integration failed: {result.message}")
    if not np.all(np.isfinite(result.y)):
        raise FloatingPointError("Background integration produced non-finite values")

    constraint = summarize_friedmann_constraint(result.y, parameters, potential)
    event_times = result.t_events or (np.asarray([]), np.asarray([]), np.asarray([]))
    return BackgroundSolution(
        time=result.t,
        values=result.y,
        bounce_times=tuple(float(value) for value in event_times[0]),
        turnaround_times=tuple(float(value) for value in event_times[1]),
        solver_message=result.message,
        function_evaluations=int(result.nfev),
        constraint=constraint,
    )


def write_solution(
    solution: BackgroundSolution,
    output_directory: str | Path,
    provenance: dict[str, Any],
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    dataset = solution.to_dataset()
    dataset.to_netcdf(output / "trajectory.nc", engine="scipy")

    table = np.column_stack([solution.time, solution.values.T])
    header = "time," + ",".join(VARIABLE_NAMES)
    np.savetxt(output / "trajectory.csv", table, delimiter=",", header=header, comments="")

    diagnostics = {
        "bounce_times": solution.bounce_times,
        "turnaround_times": solution.turnaround_times,
        "solver_message": solution.solver_message,
        "function_evaluations": solution.function_evaluations,
        "constraint": asdict(solution.constraint),
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
