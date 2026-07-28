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
from scpc.numerics.convergence import (
    run_cross_solver_comparison,
    run_tolerance_ladder,
)
from scpc.numerics.cycles import classify_recurrence, cycle_return_metrics
from scpc.numerics.provenance import build_provenance, sha256_file, write_provenance
from scpc.visualization.backgrounds import plot_expansion_comparison, plot_scpc_background


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def _background_inputs(config: dict[str, Any]) -> tuple[SCPCParameters, dict[str, Any]]:
    potential = PeriodicPotential(**config["model"]["potential"])
    parameters = SCPCParameters(potential=potential, **config["model"]["background"])
    run = config["run"]
    initial = config["initial_conditions"]
    options: dict[str, Any] = {
        "t_span": (float(run["t_start"]), float(run["t_end"])),
        "samples": int(run["samples"]),
        "a0": float(initial["a"]),
        "phi0": float(initial["phi"]),
        "phi_dot0": float(initial["phi_dot"]),
        "branch": int(initial["branch"]),
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
    plot_expansion_comparison(tables, output / "hubble_comparison.png")
    write_provenance(output / "provenance.json", build_provenance(config_path, {"workflow": "compare_models"}))
    return csv_path


def run_scpc_background(config_path: str | Path, output_dir: str | Path) -> Path:
    config = _load_yaml(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters, integration_options = _background_inputs(config)
    run = config["run"]
    solution = integrate_scpc(
        parameters,
        rtol=float(run["rtol"]),
        atol=float(run["atol"]),
        method=str(run["method"]),
        **integration_options,
    )
    dataset_path = output / "trajectory.nc"
    solution.to_xarray().to_netcdf(dataset_path, engine="scipy")

    return_metrics = cycle_return_metrics(solution)
    recurrence_tolerance = float(
        config.get("acceptance", {}).get("max_cycle_return_error", 1.0e-3)
    )
    diagnostics = {
        "max_abs_friedmann_constraint_residual": float(
            np.max(np.abs(solution.constraint_residual))
        ),
        "turning_times": solution.turning_times.tolist(),
        "turning_kinds": list(solution.turning_kinds),
        "cycle_return_metrics": [asdict(metric) for metric in return_metrics],
        "recurrence_classification": classify_recurrence(
            return_metrics,
            tolerance=recurrence_tolerance,
        ),
        "recurrence_tolerance": recurrence_tolerance,
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plot_scpc_background(solution, output / "background.png")
    write_provenance(
        output / "provenance.json",
        build_provenance(config_path, {"workflow": "run_scpc_background"}),
    )
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
    write_provenance(
        output / "provenance.json",
        build_provenance(
            config_path,
            {
                "workflow": "verify_scpc_background",
                "baseline_config": str(baseline_path),
                "baseline_config_sha256": sha256_file(baseline_path),
            },
        ),
    )
    return report_path
