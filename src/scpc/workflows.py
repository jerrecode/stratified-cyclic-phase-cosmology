"""Configuration-driven scientific workflows."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scpc.models.phase import PeriodicPotential, SCPCParameters, integrate_scpc
from scpc.models.standard import ExpansionParameters, FLRWExpansion
from scpc.numerics.candidate_verification import verify_recurrence_candidate
from scpc.numerics.convergence import (
    run_cross_solver_comparison,
    run_tolerance_ladder,
)
from scpc.numerics.cycles import classify_return_sequences, cycle_return_metrics
from scpc.numerics.provenance import (
    build_output_inventory,
    build_provenance,
    sha256_file,
    write_provenance,
)
from scpc.numerics.turning_audit import audit_turning_points
from scpc.visualization.backgrounds import plot_expansion_comparison, plot_scpc_background


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def _finite_float(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _positive_finite_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _strict_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    converted = int(value)
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _background_inputs(config: dict[str, Any]) -> tuple[SCPCParameters, dict[str, Any]]:
    potential = PeriodicPotential(**config["model"]["potential"])
    parameters = SCPCParameters(potential=potential, **config["model"]["background"])
    finite_parameters = {
        "potential.offset": potential.offset,
        "potential.amplitude": potential.amplitude,
        "potential.field_scale": potential.field_scale,
        "rho_m_ref": parameters.rho_m_ref,
        "rho_r_ref": parameters.rho_r_ref,
        "a_ref": parameters.a_ref,
    }
    for name, value in finite_parameters.items():
        _finite_float(value, name)

    run = config["run"]
    initial = config["initial_conditions"]
    start = _finite_float(run["t_start"], "run.t_start")
    end = _finite_float(run["t_end"], "run.t_end")
    if end <= start:
        raise ValueError("run.t_end must be greater than run.t_start")
    branch = _strict_int(initial["branch"], "initial_conditions.branch")
    if branch not in (-1, 1):
        raise ValueError("initial_conditions.branch must be +1 or -1")
    options: dict[str, Any] = {
        "t_span": (start, end),
        "samples": _strict_int(run["samples"], "run.samples", minimum=2),
        "a0": _positive_finite_float(initial["a"], "initial_conditions.a"),
        "phi0": _finite_float(initial["phi"], "initial_conditions.phi"),
        "phi_dot0": _finite_float(initial["phi_dot"], "initial_conditions.phi_dot"),
        "branch": branch,
    }
    return parameters, options


def compare_models(config_path: str | Path, output_dir: str | Path) -> Path:
    config = _load_yaml(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid = config["grid"]
    z = np.linspace(float(grid["z_min"]), float(grid["z_max"]), int(grid["samples"]))
    tables: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | str]] = []
    for model in config["models"]:
        params = ExpansionParameters(name=model["id"], **model["parameters"])
        table = FLRWExpansion(params).distance_table(z)
        tables[model["label"]] = table
        for i in range(z.size):
            rows.append({"model": model["id"], **{key: float(value[i]) for key, value in table.items()}})

    csv_path = output / "background_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_path = output / "hubble_comparison.png"
    plot_expansion_comparison(tables, figure_path)
    provenance = build_provenance(config_path, {"workflow": "compare_models"})
    provenance["outputs"] = build_output_inventory(
        [csv_path, figure_path],
        relative_to=output,
    )
    write_provenance(output / "provenance.json", provenance)
    return csv_path


def run_scpc_background(config_path: str | Path, output_dir: str | Path) -> Path:
    config = _load_yaml(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters, integration_options = _background_inputs(config)
    run = config["run"]
    solution = integrate_scpc(
        parameters,
        rtol=_positive_finite_float(run["rtol"], "run.rtol"),
        atol=_positive_finite_float(run["atol"], "run.atol"),
        method=str(run["method"]),
        **integration_options,
    )
    dataset_path = output / "trajectory.nc"
    solution.to_xarray().to_netcdf(dataset_path, engine="scipy")

    return_metrics = cycle_return_metrics(solution)
    turning_audit = audit_turning_points(solution)
    acceptance = config.get("acceptance", {})
    return_tolerance = float(
        acceptance.get(
            "max_return_error",
            acceptance.get("max_cycle_return_error", 1.0e-3),
        )
    )
    diagnostics = {
        "max_abs_friedmann_constraint_residual": float(
            np.max(np.abs(solution.constraint_residual))
        ),
        "turning_times": solution.turning_times.tolist(),
        "turning_kinds": list(solution.turning_kinds),
        "turning_point_physics_audit": [item.to_dict() for item in turning_audit],
        "turning_point_physics_consistent": all(item.kind_consistent for item in turning_audit),
        "cycle_return_metrics": [asdict(metric) for metric in return_metrics],
        "return_sequence_classifications": classify_return_sequences(
            return_metrics,
            tolerance=return_tolerance,
        ),
        "return_tolerance": return_tolerance,
        "recurrence_assessment": "not_performed_requires_repeated_converged_multi_solver_returns",
    }
    diagnostics_path = output / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    figure_path = output / "background.png"
    plot_scpc_background(solution, figure_path)
    provenance = build_provenance(config_path, {"workflow": "run_scpc_background"})
    provenance["outputs"] = build_output_inventory(
        [dataset_path, diagnostics_path, figure_path],
        relative_to=output,
    )
    write_provenance(output / "provenance.json", provenance)
    return dataset_path


def verify_scpc_background(config_path: str | Path, output_dir: str | Path) -> Path:
    """Execute the declared tolerance and cross-solver verification protocol."""

    verification = _load_yaml(config_path)
    baseline_path = Path(verification["baseline_config"])
    baseline = _load_yaml(baseline_path)
    parameters, integration_options = _background_inputs(baseline)

    ladder_config = verification["tolerance_ladder"]
    levels = tuple(
        (str(level["label"]), float(level["rtol"]), float(level["atol"]))
        for level in ladder_config["levels"]
    )
    ladder = run_tolerance_ladder(
        parameters,
        integration_options=integration_options,
        tolerances=levels,
        method=str(ladder_config["method"]),
    )

    solver_config = verification["cross_solver"]
    solver_results = run_cross_solver_comparison(
        parameters,
        integration_options=integration_options,
        methods=tuple(str(method) for method in solver_config["methods"]),
        reference_method=str(solver_config["reference_method"]),
        rtol=float(solver_config["rtol"]),
        atol=float(solver_config["atol"]),
    )

    acceptance = verification["acceptance"]
    medium = next(result for result in ladder if result.label == "medium")
    if medium.difference_to_reference is None:
        raise ValueError("The verification protocol must define a non-reference medium level")
    solver_errors = [
        result.difference_to_reference.maximum
        for result in solver_results
        if result.difference_to_reference is not None
    ]
    maximum_constraint = max(
        result.max_abs_constraint_residual for result in (*ladder, *solver_results)
    )
    maximum_solver_error = max(solver_errors, default=0.0)
    checks = {
        "constraint_residual": maximum_constraint
        <= float(acceptance["max_constraint_residual"]),
        "medium_to_reference": medium.difference_to_reference.maximum
        <= float(acceptance["max_medium_to_reference_error"]),
        "cross_solver": maximum_solver_error
        <= float(acceptance["max_cross_solver_error"]),
    }

    report = {
        "verification_config": str(config_path),
        "verification_config_sha256": sha256_file(config_path),
        "baseline_config": str(baseline_path),
        "baseline_config_sha256": sha256_file(baseline_path),
        "tolerance_ladder": [asdict(result) for result in ladder],
        "cross_solver": [asdict(result) for result in solver_results],
        "acceptance": acceptance,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    provenance = build_provenance(
        config_path,
        {
            "workflow": "verify_scpc_background",
            "baseline_config": str(baseline_path),
            "baseline_config_sha256": sha256_file(baseline_path),
        },
    )
    provenance["outputs"] = build_output_inventory([report_path], relative_to=output)
    write_provenance(output / "provenance.json", provenance)
    return report_path


def audit_scpc_candidate(config_path: str | Path, output_dir: str | Path) -> Path:
    """Run the non-promotional Stage-1 recurrence-candidate verification gate."""

    config = _load_yaml(config_path)
    baseline_path = Path(config["baseline_config"])
    baseline = _load_yaml(baseline_path)
    parameters, integration_options = _background_inputs(baseline)

    tolerance_levels = tuple(
        (
            str(level["label"]),
            _positive_finite_float(level["rtol"], f"{level['label']}.rtol"),
            _positive_finite_float(level["atol"], f"{level['label']}.atol"),
        )
        for level in config["tolerance_levels"]
    )
    independent = config["independent_solver"]
    acceptance = config["acceptance"]
    report = verify_recurrence_candidate(
        parameters,
        integration_options=integration_options,
        tolerance_levels=tolerance_levels,
        primary_method=str(config["primary_method"]),
        independent_method=str(independent["method"]),
        independent_rtol=_positive_finite_float(independent["rtol"], "independent_solver.rtol"),
        independent_atol=_positive_finite_float(independent["atol"], "independent_solver.atol"),
        max_constraint_residual=_positive_finite_float(
            acceptance["max_constraint_residual"], "acceptance.max_constraint_residual"
        ),
        max_solution_error=_positive_finite_float(
            acceptance["max_solution_error"], "acceptance.max_solution_error"
        ),
        max_event_time_error=_positive_finite_float(
            acceptance["max_event_time_error"], "acceptance.max_event_time_error"
        ),
        max_event_state_error=_positive_finite_float(
            acceptance["max_event_state_error"], "acceptance.max_event_state_error"
        ),
        max_return_error=_positive_finite_float(
            acceptance["max_return_error"], "acceptance.max_return_error"
        ),
        minimum_same_kind_return_metrics=_strict_int(
            acceptance["minimum_same_kind_return_metrics"],
            "acceptance.minimum_same_kind_return_metrics",
            minimum=2,
        ),
    )
    report.update(
        {
            "verification_config": str(config_path),
            "verification_config_sha256": sha256_file(config_path),
            "baseline_config": str(baseline_path),
            "baseline_config_sha256": sha256_file(baseline_path),
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "candidate_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    provenance = build_provenance(
        config_path,
        {
            "workflow": "audit_scpc_candidate",
            "baseline_config": str(baseline_path),
            "baseline_config_sha256": sha256_file(baseline_path),
            "candidate_gate_passed": bool(report["candidate_gate_passed"]),
            "candidate_status": str(report["status"]),
        },
    )
    provenance["outputs"] = build_output_inventory([report_path], relative_to=output)
    write_provenance(output / "provenance.json", provenance)
    return report_path
