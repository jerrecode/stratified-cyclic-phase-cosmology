#!/usr/bin/env python3
"""Reproduce publication-oriented DESI DR2 BAO + BBN inference diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from scpc.inference.desi_chains import (
    ensure_official_chain_metadata,
    ensure_official_chain_set,
    load_cobaya_chains,
    load_getdist_samples,
    select_converged_burn_fraction,
    sha256_file,
)
from scpc.inference.desi_likelihood import DESIDR2BAOLikelihood, fit_flat_lcdm_bbn_map
from scpc.inference.figures import make_aubourg_validation_figure, make_kinematic_figure, make_main_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/desi_dr2_bbn"))
    parser.add_argument("--chain-cache", type=Path, default=Path("data/external/desi_dr2_cosmology_chains"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference/desi_dr2_bbn.yaml"))
    parser.add_argument("--download-official-chains", action="store_true")
    parser.add_argument(
        "--burn-fraction",
        type=float,
        default=None,
        help="Explicit override. By default select the earliest predeclared cut satisfying GetDist R-1.",
    )
    parser.add_argument("--max-rminus1", type=float, default=0.01)
    parser.add_argument("--aubourg-samples", type=int, default=128)
    parser.add_argument("--skip-map", action="store_true", help="Skip the independent CAMB MAP regression check")
    parser.add_argument("--skip-aubourg", action="store_true", help="Skip CAMB-vs-Aubourg validation")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "matplotlib", "camb", "cobaya", "getdist", "PyYAML")
    result = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _resolve_chain(
    cache: Path,
    *,
    dataset: str,
    download: bool,
    explicit_burn: float | None,
    max_rminus1: float,
):
    paths, samples_provenance = ensure_official_chain_set(
        cache,
        model="base",
        dataset=dataset,
        download=download,
    )
    metadata_provenance = ensure_official_chain_metadata(
        cache,
        model="base",
        dataset=dataset,
        download=download,
    )
    directory = cache / "base" / dataset
    if explicit_burn is None:
        burn, getdist_samples, convergence = select_converged_burn_fraction(
            directory,
            max_rminus1=max_rminus1,
        )
    else:
        burn = float(explicit_burn)
        getdist_samples = load_getdist_samples(directory, burn_fraction=burn)
        rminus1 = float(getdist_samples.getGelmanRubin())
        convergence = {
            "selection_rule": "explicit command-line override",
            "max_rminus1": float(max_rminus1),
            "selected_burn_fraction": burn,
            "trials": [{"burn_fraction": burn, "getdist_rminus1": rminus1}],
            "getdist_convergence_summary": getdist_samples.getConvergeTests(
                what=("MeanVar", "GelmanRubin", "SplitTest", "CorrLengths")
            ),
        }
        if not rminus1 <= max_rminus1:
            raise RuntimeError(
                f"Explicit burn fraction {burn} leaves GetDist R-1={rminus1:.5g} above {max_rminus1}"
            )
    chain = load_cobaya_chains(paths, burn_fraction=burn)
    if chain.weights.size != getdist_samples.numrows:
        raise RuntimeError(
            "GetDist and direct Cobaya readers retained different row counts: "
            f"{getdist_samples.numrows} != {chain.weights.size}"
        )
    return chain, getdist_samples, burn, convergence, samples_provenance, metadata_provenance


def _uncertainty(summary: dict[str, float]) -> float:
    return 0.5 * (float(summary["q84"]) - float(summary["q16"]))


def _write_latex_tables(report: dict[str, object], output_dir: Path) -> list[Path]:
    main = report["main_figure"]
    bao = main["bao_only"]
    bbn = main["bao_bbn"]
    kinematics = report["kinematics"]
    convergence = report["chain_convergence"]
    aubourg = report.get("aubourg_crosscheck")

    reproduction = output_dir / "desi_reproduction_table.tex"
    reproduction.write_text(
        """\\begin{tabular}{lcc}\n"
        "\\toprule\nQuantity & DESI DR2 published & This pipeline reproduction \\\\\n"
        "\\midrule\n"
        f"$\\Omega_m$ (BAO only) & $0.2975\\pm0.0086$ & ${bao['omega_m']['mean']:.4f}\\pm{_uncertainty(bao['omega_m']):.4f}$ \\\\\n"
        f"$h\\,r_d$ [Mpc] & $101.54\\pm0.73$ & ${bao['h_r_drag_Mpc']['mean']:.2f}\\pm{_uncertainty(bao['h_r_drag_Mpc']):.2f}$ \\\\\n"
        f"$\\rho(\\Omega_m,h r_d)$ & $-0.92$ & ${bao['correlation_omega_m_h_r_drag']:.3f}$ \\\\\n"
        f"$\\Omega_m$ (BAO+BBN) & $0.2977\\pm0.0086$ & ${bbn['omega_m']['mean']:.4f}\\pm{_uncertainty(bbn['omega_m']):.4f}$ \\\\\n"
        f"$H_0$ [km s$^{{-1}}$ Mpc$^{{-1}}$] & $68.51\\pm0.58$ & ${bbn['H0']['mean']:.2f}\\pm{_uncertainty(bbn['H0']):.2f}$ \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )

    convergence_table = output_dir / "desi_chain_convergence_table.tex"
    rows = []
    for label, details in (("BAO only", convergence["bao_only"]), ("BAO+BBN", convergence["bao_bbn"])):
        selected = details["trials"][-1]
        rows.append(
            f"{label} & {details['selected_burn_fraction']:.2f} & {selected['getdist_rminus1']:.4g} & "
            f"{int(selected.get('rows', 0))} & {selected.get('weighted_ess', float('nan')):.0f} \\\\"
        )
    convergence_table.write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Chain set & Burn fraction & GetDist $R-1$ & Rows & Weighted ESS \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )

    kinematic_table = output_dir / "kinematic_summary_table.tex"
    kinematic_table.write_text(
        "\\begin{tabular}{lcc}\n\\toprule\nQuantity & Median & 68\\% interval \\\\\n\\midrule\n"
        f"$q_0$ & ${kinematics['q0']['median']:.4f}$ & "
        f"$[{kinematics['q0']['q16']:.4f},{kinematics['q0']['q84']:.4f}]$ \\\\\n"
        f"$w_{{\\rm tot},0}$ & ${kinematics['w_tot0']['median']:.4f}$ & "
        f"$[{kinematics['w_tot0']['q16']:.4f},{kinematics['w_tot0']['q84']:.4f}]$ \\\\\n"
        f"$z_{{\\rm acc}}$ & ${kinematics['z_acc']['median']:.4f}$ & "
        f"$[{kinematics['z_acc']['q16']:.4f},{kinematics['z_acc']['q84']:.4f}]$ \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )

    paths = [reproduction, convergence_table, kinematic_table]
    if aubourg is not None:
        aubourg_table = output_dir / "aubourg_validation_table.tex"
        aubourg_table.write_text(
            "\\begin{tabular}{lr}\n\\toprule\nDiagnostic & Value \\\\\n\\midrule\n"
            f"Posterior-spanning evaluations & {aubourg['samples']} \\\\\n"
            f"RMS fractional difference & ${aubourg['rms_fractional_difference']:.3e}$ \\\\\n"
            f"Maximum absolute fractional difference & ${aubourg['max_abs_fractional_difference']:.3e}$ \\\\\n"
            f"CAMB $r_d$ range [Mpc] & ${aubourg['r_drag_camb_min_Mpc']:.3f}--{aubourg['r_drag_camb_max_Mpc']:.3f}$ \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        paths.append(aubourg_table)
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bao_chain, bao_getdist, bao_burn, bao_conv, bao_prov, bao_meta = _resolve_chain(
        args.chain_cache,
        dataset="desi-bao-all",
        download=args.download_official_chains,
        explicit_burn=args.burn_fraction,
        max_rminus1=args.max_rminus1,
    )
    bbn_chain, bbn_getdist, bbn_burn, bbn_conv, bbn_prov, bbn_meta = _resolve_chain(
        args.chain_cache,
        dataset="desi-bao-all_schoneberg2024-bbn",
        download=args.download_official_chains,
        explicit_burn=args.burn_fraction,
        max_rminus1=args.max_rminus1,
    )

    likelihood = DESIDR2BAOLikelihood(
        Path("data/analysis_ready/desi_dr2_bao/mean.txt"),
        Path("data/analysis_ready/desi_dr2_bao/covariance.txt"),
    )
    map_result = None if args.skip_map else fit_flat_lcdm_bbn_map(likelihood)

    main_summary = make_main_figure(
        bao_chain,
        bbn_chain,
        bao_getdist,
        bbn_getdist,
        likelihood,
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

    report: dict[str, object] = {
        "schema_version": 2,
        "analysis": "DESI DR2 2025 BAO likelihood + Schoneberg 2024 BBN prior, flat LambdaCDM",
        "epistemic_status": "standard-cosmology reproduction benchmark; not an SCPC observational fit",
        "software_commit": _git_commit(),
        "inference_config": str(args.config),
        "inference_config_sha256": sha256_file(args.config),
        "runtime_versions": _versions(),
        "posterior_source": "Official DESI DR2 released Cobaya MCMC chains, analysed with GetDist",
        "posterior_chain_burn_fraction": {"bao_only": bao_burn, "bao_bbn": bbn_burn},
        "chain_convergence": {"bao_only": bao_conv, "bao_bbn": bbn_conv},
        "late_2026_lya_update_combined": False,
        "late_2026_lya_update_note": (
            "DESI DR2 Results IV (2026) is tracked separately and is not multiplied into the pinned "
            "2025 Results I/II likelihood without a release-native joint covariance/likelihood."
        ),
        "bao_chain_provenance": bao_prov,
        "bao_chain_metadata_provenance": bao_meta,
        "bbn_chain_provenance": bbn_prov,
        "bbn_chain_metadata_provenance": bbn_meta,
        "main_figure": main_summary,
        "map_crosscheck": map_result,
        "kinematics": kinematics,
        "aubourg_crosscheck": aubourg,
        "published_benchmark": {
            "source": "DESI DR2 Results II, Table 5 / Eqs. 17 and 19",
            "bao_only": {"omega_m": [0.2975, 0.0086], "h_r_drag_Mpc": [101.54, 0.73], "correlation": -0.92},
            "bao_bbn": {"omega_m": [0.2977, 0.0086], "H0": [68.51, 0.58]},
        },
        "interpretation": (
            "Upper panels reproduce marginalized DESI posteriors with GetDist. The lower-left panel "
            "is a full-covariance posterior-predictive BAO check. The lower-right panel is a CAMB "
            "standard-cosmology background extrapolation. BBN calibrates omega_b h^2, not r_d directly."
        ),
    }
    table_paths = _write_latex_tables(report, args.output_dir)
    report["latex_tables"] = [str(path) for path in table_paths]

    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paper_dir = Path("paper/generated")
    paper_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "desi_dr2_bao_bbn_posteriors.png",
        args.output_dir / "kinematic_diagnostics.png",
        args.output_dir / "sound_horizon_backend_validation.png",
        summary_path,
        *table_paths,
    ]
    for source in outputs:
        if source.exists():
            shutil.copy2(source, paper_dir / source.name)


if __name__ == "__main__":
    main()
