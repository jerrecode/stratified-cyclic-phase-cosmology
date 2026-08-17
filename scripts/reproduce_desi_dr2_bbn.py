#!/usr/bin/env python3
"""Reproduce publication-oriented DESI DR2 BAO + BBN inference diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scpc.inference.desi_chains import ensure_official_chain_set, load_cobaya_chains
from scpc.inference.desi_likelihood import DESIDR2BAOLikelihood, fit_flat_lcdm_bbn_map
from scpc.inference.figures import make_aubourg_validation_figure, make_kinematic_figure, make_main_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/desi_dr2_bbn"))
    parser.add_argument("--chain-cache", type=Path, default=Path("data/external/desi_dr2_cosmology_chains"))
    parser.add_argument("--download-official-chains", action="store_true")
    parser.add_argument("--burn-fraction", type=float, default=0.3)
    parser.add_argument("--aubourg-samples", type=int, default=32)
    parser.add_argument("--skip-map", action="store_true", help="Skip the independent CAMB MAP regression check")
    parser.add_argument(
        "--skip-aubourg",
        action="store_true",
        help="Skip the CAMB-vs-Aubourg posterior-draw validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bao_paths, bao_provenance = ensure_official_chain_set(
        args.chain_cache,
        model="base",
        dataset="desi-bao-all",
        download=args.download_official_chains,
    )
    bbn_paths, bbn_provenance = ensure_official_chain_set(
        args.chain_cache,
        model="base",
        dataset="desi-bao-all_schoneberg2024-bbn",
        download=args.download_official_chains,
    )
    bao_chain = load_cobaya_chains(bao_paths, burn_fraction=args.burn_fraction)
    bbn_chain = load_cobaya_chains(bbn_paths, burn_fraction=args.burn_fraction)

    map_result = None
    if not args.skip_map:
        likelihood = DESIDR2BAOLikelihood(
            Path("data/analysis_ready/desi_dr2_bao/mean.txt"),
            Path("data/analysis_ready/desi_dr2_bao/covariance.txt"),
        )
        map_result = fit_flat_lcdm_bbn_map(likelihood)

    main_summary = make_main_figure(
        bao_chain,
        bbn_chain,
        args.output_dir / "desi_dr2_bao_bbn_posteriors.png",
        map_result=map_result,
    )
    kinematics = make_kinematic_figure(bbn_chain, args.output_dir / "kinematic_diagnostics.png")
    aubourg = None
    if not args.skip_aubourg:
        aubourg = make_aubourg_validation_figure(
            bbn_chain,
            args.output_dir / "sound_horizon_backend_validation.png",
            samples=args.aubourg_samples,
        )

    report = {
        "schema_version": 1,
        "analysis": "DESI DR2 2025 BAO likelihood + Schoneberg 2024 BBN prior, flat LambdaCDM",
        "posterior_source": "Official DESI DR2 released Cobaya MCMC chains",
        "posterior_chain_burn_fraction": args.burn_fraction,
        "late_2026_lya_update_combined": False,
        "late_2026_lya_update_note": (
            "DESI DR2 Results IV (2026) is tracked separately and is not multiplied into the pinned "
            "2025 Results I/II likelihood without a release-native joint covariance/likelihood."
        ),
        "bao_chain_provenance": bao_provenance,
        "bbn_chain_provenance": bbn_provenance,
        "main_figure": main_summary,
        "map_crosscheck": map_result,
        "kinematics": kinematics,
        "aubourg_crosscheck": aubourg,
        "interpretation": (
            "The upper panels are marginalized posteriors from the official DESI chain release. "
            "The lower panels are model-implied flat-LambdaCDM histories. BBN calibrates omega_b h^2; "
            "it is not plotted as a direct r_d measurement. CAMB supplies r_d and the high-redshift "
            "massive-neutrino background."
        ),
    }
    summary_path = args.output_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    paper_dir = Path("paper/generated")
    paper_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "desi_dr2_bao_bbn_posteriors.png",
        "kinematic_diagnostics.png",
        "sound_horizon_backend_validation.png",
        "inference_summary.json",
    ):
        source = args.output_dir / filename
        if source.exists():
            shutil.copy2(source, paper_dir / filename)


if __name__ == "__main__":
    main()
