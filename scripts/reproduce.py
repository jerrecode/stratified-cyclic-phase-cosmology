#!/usr/bin/env python3
"""Regenerate the paper's baseline, verification, topology, and layered test-field products."""

import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from scpc.models.layered import LayeredScalarParameters, S4RadialGrid, integrate_layered_pulse
from scpc.visualization.topology import plot_layered_propagation, plot_s4_foliation
from scpc.workflows import compare_models, run_scpc_background, verify_scpc_background


def _run_layered(output: Path) -> tuple[Path, Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    grid = S4RadialGrid(cells=96, radius=1.0)
    parameters = LayeredScalarParameters(mass=0.35, coupling=0.05, damping=0.0)
    result = integrate_layered_pulse(grid=grid, parameters=parameters)
    dataset = xr.Dataset(
        data_vars={
            "field": (("time", "chi"), np.asarray(result["field"])),
            "velocity": (("time", "chi"), np.asarray(result["velocity"])),
            "energy": (("time",), np.asarray(result["energy"])),
            "relative_energy_error": (("time",), np.asarray(result["relative_energy_error"])),
        },
        coords={"time": np.asarray(result["time"]), "chi": np.asarray(result["chi"])},
        attrs={
            "geometry": "fixed R x S4; S3-invariant scalar test field; no gravitational backreaction",
            "cells": int(result["cells"]),
            "radius": float(result["radius"]),
            "mass": parameters.mass,
            "coupling": parameters.coupling,
            "damping": parameters.damping,
        },
    )
    data_path = output / "layered_test_field.nc"
    dataset.to_netcdf(data_path, engine="scipy")
    diagnostics = {
        "geometry": "fixed_R_times_S4",
        "gravitational_backreaction": False,
        "cells": int(result["cells"]),
        "stored_times": int(np.asarray(result["time"]).size),
        "nfev": int(result["nfev"]),
        "max_abs_relative_energy_drift": float(result["max_abs_relative_energy_drift"]),
        "energy_definition": "finite-volume face-gradient quadratic form matched to the radial Laplacian",
    }
    diagnostics_path = output / "layered_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    figure_path = output / "layered_propagation.png"
    plot_layered_propagation(result, figure_path)
    return data_path, diagnostics_path, figure_path


def main() -> None:
    model_dir = Path("results/model_comparison")
    scpc_dir = Path("results/scpc_baseline")
    verification_dir = Path("results/scpc_verification")
    layered_dir = Path("results/layered_test")
    topology_dir = Path("results/topology")
    paper_dir = Path("paper/generated")
    paper_dir.mkdir(parents=True, exist_ok=True)
    topology_dir.mkdir(parents=True, exist_ok=True)

    compare_models("configs/model_comparisons.yaml", model_dir)
    run_scpc_background("configs/scpc_baseline.yaml", scpc_dir)
    verification_path = verify_scpc_background("configs/scpc_verification.yaml", verification_dir)
    layered_data, layered_diagnostics, layered_figure = _run_layered(layered_dir)
    topology_figure = topology_dir / "s4_foliation.png"
    plot_s4_foliation(topology_figure)

    for source, target in [
        (model_dir / "background_comparison.csv", paper_dir / "background_comparison.csv"),
        (model_dir / "hubble_comparison.png", paper_dir / "hubble_comparison.png"),
        (scpc_dir / "trajectory.nc", paper_dir / "scpc_baseline_trajectory.nc"),
        (scpc_dir / "background.png", paper_dir / "scpc_baseline_background.png"),
        (scpc_dir / "diagnostics.json", paper_dir / "scpc_baseline_diagnostics.json"),
        (verification_path, paper_dir / "scpc_verification.json"),
        (layered_data, paper_dir / "layered_test_field.nc"),
        (layered_diagnostics, paper_dir / "layered_diagnostics.json"),
        (layered_figure, paper_dir / "layered_propagation.png"),
        (topology_figure, paper_dir / "s4_foliation.png"),
    ]:
        shutil.copy2(source, target)


if __name__ == "__main__":
    main()
