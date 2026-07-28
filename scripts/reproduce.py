#!/usr/bin/env python3
"""Regenerate the initial paper data products and figures."""

from pathlib import Path
import shutil

from scpc.workflows import compare_models, run_scpc_background


def main() -> None:
    model_dir = Path("results/model_comparison")
    scpc_dir = Path("results/scpc_baseline")
    paper_dir = Path("paper/generated")
    paper_dir.mkdir(parents=True, exist_ok=True)

    compare_models("configs/model_comparisons.yaml", model_dir)
    run_scpc_background("configs/scpc_baseline.yaml", scpc_dir)

    for source, target in [
        (model_dir / "background_comparison.csv", paper_dir / "background_comparison.csv"),
        (model_dir / "hubble_comparison.png", paper_dir / "hubble_comparison.png"),
        (scpc_dir / "trajectory.nc", paper_dir / "scpc_baseline_trajectory.nc"),
        (scpc_dir / "background.png", paper_dir / "scpc_baseline_background.png"),
        (scpc_dir / "diagnostics.json", paper_dir / "scpc_baseline_diagnostics.json"),
    ]:
        shutil.copy2(source, target)


if __name__ == "__main__":
    main()
