"""Configuration-driven scientific workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scpc.models.phase import PeriodicPotential, SCPCParameters, integrate_scpc
from scpc.models.standard import ExpansionParameters, FLRWExpansion
from scpc.numerics.provenance import build_provenance, write_provenance
from scpc.visualization.backgrounds import plot_expansion_comparison, plot_scpc_background


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


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
    potential = PeriodicPotential(**config["model"]["potential"])
    params = SCPCParameters(potential=potential, **config["model"]["background"])
    run = config["run"]
    initial = config["initial_conditions"]
    solution = integrate_scpc(
        params,
        t_span=(float(run["t_start"]), float(run["t_end"])),
        samples=int(run["samples"]),
        rtol=float(run["rtol"]),
        atol=float(run["atol"]),
        method=str(run["method"]),
        a0=float(initial["a"]),
        phi0=float(initial["phi"]),
        phi_dot0=float(initial["phi_dot"]),
        branch=int(initial["branch"]),
    )
    dataset_path = output / "trajectory.nc"
    solution.to_xarray().to_netcdf(dataset_path, engine="scipy")
    diagnostics = {
        "max_abs_friedmann_constraint_residual": float(np.max(np.abs(solution.constraint_residual))),
        "turning_times": solution.turning_times.tolist(),
        "turning_kinds": list(solution.turning_kinds),
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plot_scpc_background(solution, output / "background.png")
    write_provenance(output / "provenance.json", build_provenance(config_path, {"workflow": "run_scpc_background"}))
    return dataset_path
