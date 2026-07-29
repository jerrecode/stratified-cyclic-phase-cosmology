import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scpc.models.phase import PeriodicPotential, SCPCIntegrationDomain, SCPCParameters, integrate_scpc
from scpc.scans.errors import ResultIntegrityError
from scpc.scans.grid import expand_parameter_grid
from scpc.scans.identity import canonical_run_identity
from scpc.scans.outcomes import OutcomeClass, assess_solution
from scpc.scans.runner import run_background_scan


SCAN_SCHEMA = Path("configs/scans/scan.schema.json")


def _base_specification(*, threshold: float | int = 0.8) -> dict:
    return {
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
            "branch": -1,
        },
        "run": {
            "t_start": 0.0,
            "t_end": 1.0,
            "samples": 21,
            "method": "DOP853",
            "rtol": 1.0e-10,
            "atol": 1.0e-12,
            "max_step": 0.025,
            "domain_check_substeps": 16,
            "domain": {"min_scale_factor": threshold},
        },
    }


def _write_scan(tmp_path: Path) -> Path:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_specification(), sort_keys=False), encoding="utf-8")
    scan_path = tmp_path / "scan.yaml"
    scan_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "description": "Single exact domain-termination regression run.",
                "base_config": "base.yaml",
                "identity_namespace": "scpc-domain-test-v1",
                "max_runs": 1,
                "resume": True,
                "rerun_statuses": [],
                "axes": {},
                "classification": {
                    "constraint_threshold": 1.0e-7,
                    "hubble_zero_tolerance": 1.0e-10,
                    "return_tolerance": 1.0e-3,
                },
                "retention": {"outcomes": [], "max_trajectories": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return scan_path


def _single_row(index_path: Path) -> dict[str, str]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]


def test_domain_and_check_controls_are_execution_normalized_in_run_identity() -> None:
    integer = _base_specification(threshold=1)
    integer["run"]["max_step"] = 1
    integer["run"]["domain_check_substeps"] = 16.0
    floating = _base_specification(threshold=1.0)
    floating["run"]["max_step"] = 1.0
    floating["run"]["domain_check_substeps"] = 16
    assert canonical_run_identity(integer) == canonical_run_identity(floating)


def test_empty_domain_mapping_is_identity_equivalent_to_omission() -> None:
    empty_domain = _base_specification()
    empty_domain["run"]["domain"] = {}
    omitted = _base_specification()
    omitted["run"].pop("domain")
    assert canonical_run_identity(empty_domain) == canonical_run_identity(omitted)


def test_execution_equivalent_domain_axis_values_are_duplicate_runs() -> None:
    with pytest.raises(ValueError, match="duplicate complete specifications"):
        expand_parameter_grid(
            _base_specification(),
            {"run.domain.min_scale_factor": [1, 1.0]},
        )


def test_domain_termination_is_rejected_outcome_not_failed_run(tmp_path) -> None:
    scan_path = _write_scan(tmp_path)
    output = tmp_path / "output"
    row = _single_row(run_background_scan(scan_path, output, schema_path=SCAN_SCHEMA))

    assert row["status"] == "rejected"
    assert row["outcome"] == "physical_domain_termination"
    assert row["failure_class"] == ""
    assert row["numerically_valid"] == "False"
    assert row["completed_to_requested_end"] == "False"
    assert row["termination_kind"] == "minimum_scale_factor"
    assert float(row["termination_time"]) == pytest.approx(-np.log(0.8), rel=1.0e-8)
    assert float(row["termination_threshold"]) == pytest.approx(0.8)
    assert float(row["termination_observed"]) == pytest.approx(0.8)
    assert row["termination_units"] == "1"
    boundaries = json.loads(row["termination_boundaries"])
    assert boundaries == [
        {
            "kind": "minimum_scale_factor",
            "observed": pytest.approx(0.8, rel=1.0e-8),
            "threshold": 0.8,
            "units": "1",
        }
    ]
    assert "not a spacetime singularity claim" in row["reason"]
    assert row["trajectory_path"] == ""

    specification = json.loads(row["specification"])
    assert specification["run"]["domain"] == {"min_scale_factor": 0.8}
    assert specification["run"]["max_step"] == 0.025
    assert specification["run"]["domain_check_substeps"] == 16
    summary = json.loads((output / "scan_summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"] == {"rejected": 1}
    assert summary["outcome_counts"] == {"physical_domain_termination": 1}
    assert summary["termination_kind_counts"] == {"minimum_scale_factor": 1}
    assert (output / "provenance.json").is_file()


def test_terminated_solution_is_not_full_interval_morphology() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0, strata_count=1),
    )
    solution = integrate_scpc(
        parameters,
        t_span=(0.0, 1.0),
        samples=21,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=-1,
        domain=SCPCIntegrationDomain(min_scale_factor=0.8),
        max_step=0.025,
        domain_check_substeps=16,
    )
    assessment = assess_solution(solution)
    assert assessment.outcome is OutcomeClass.PHYSICAL_DOMAIN_TERMINATION
    assert assessment.numerically_valid is False


def test_inconsistent_termination_endpoint_is_result_integrity_error() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0, strata_count=1),
    )
    solution = integrate_scpc(
        parameters,
        t_span=(0.0, 1.0),
        samples=21,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=-1,
        domain=SCPCIntegrationDomain(min_scale_factor=0.8),
        max_step=0.025,
        domain_check_substeps=16,
    )
    solution.termination_time = float(solution.termination_time) + 0.1
    with pytest.raises(ResultIntegrityError, match="final stored time"):
        assess_solution(solution)
