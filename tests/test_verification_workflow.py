import json

import yaml

from scpc.cli import main
from scpc.numerics.provenance import sha256_file


def test_over_strict_verification_fails_and_records_output_checksum(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {
                    "background": {
                        "spatial_curvature_k": 0,
                        "rho_m_ref": 0.0,
                        "rho_r_ref": 0.0,
                        "a_ref": 1.0,
                    },
                    "potential": {
                        "offset": 3.0,
                        "amplitude": 0.0,
                        "strata_count": 1,
                        "field_scale": 1.0,
                        "target_space": "real",
                    },
                },
                "initial_conditions": {
                    "a": 1.0,
                    "phi": 0.0,
                    "phi_dot": 0.0,
                    "branch": 1,
                },
                "run": {
                    "t_start": 0.0,
                    "t_end": 0.5,
                    "samples": 51,
                    "method": "DOP853",
                    "rtol": 1.0e-9,
                    "atol": 1.0e-11,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    verification_path = tmp_path / "verification.yaml"
    verification_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "baseline_config": str(baseline_path),
                "tolerance_ladder": {
                    "method": "DOP853",
                    "levels": [
                        {"label": "coarse", "rtol": 1.0e-6, "atol": 1.0e-8},
                        {"label": "medium", "rtol": 1.0e-8, "atol": 1.0e-10},
                        {"label": "reference", "rtol": 1.0e-10, "atol": 1.0e-12},
                    ],
                },
                "cross_solver": {
                    "methods": ["DOP853", "RK45"],
                    "reference_method": "DOP853",
                    "rtol": 1.0e-9,
                    "atol": 1.0e-11,
                },
                "acceptance": {
                    "max_constraint_residual": 0.0,
                    "max_medium_to_reference_error": 0.0,
                    "max_cross_solver_error": 0.0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "verification-output"
    exit_code = main(
        [
            "verify-background",
            "--config",
            str(verification_path),
            "--output",
            str(output_dir),
        ]
    )

    report_path = output_dir / "verification.json"
    provenance_path = output_dir / "provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["passed"] is False
    assert not all(report["checks"].values())
    assert provenance["outputs"] == [
        {
            "path": "verification.json",
            "size_bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        }
    ]
