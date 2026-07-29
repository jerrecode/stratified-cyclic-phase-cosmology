import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

import scpc.scans.runner as runner
from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.scans.errors import ResultIntegrityError
from scpc.scans.outcomes import assess_solution


SCAN_SCHEMA = Path("configs/scans/scan.schema.json")


def _simple_solution() -> SCPCSolution:
    time = np.asarray([0.0, 1.0, 2.0])
    return SCPCSolution(
        t=time,
        a=np.asarray([1.0, 1.2, 1.4]),
        H=np.asarray([0.2, 0.0, -0.2]),
        phi=np.zeros(3),
        phi_dot=np.zeros(3),
        rho_m=np.zeros(3),
        rho_r=np.zeros(3),
        rho_phi=np.ones(3),
        p_phi=-np.ones(3),
        constraint_residual=np.zeros(3),
        turning_times=np.asarray([1.0]),
        turning_kinds=("turnaround",),
        parameters=SCPCParameters(
            spatial_curvature_k=0,
            rho_m_ref=0.0,
            rho_r_ref=0.0,
            potential=PeriodicPotential(offset=3.0, amplitude=0.0),
        ),
        solver_metadata={},
        turning_state_vectors=np.asarray([[1.2, 0.0, 0.0, 0.0]]),
        requested_end_time=2.0,
    )


def _write_domain_protocol(tmp_path: Path) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(
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
                    "branch": -1,
                },
                "run": {
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "samples": 101,
                    "method": "DOP853",
                    "rtol": 1.0e-10,
                    "atol": 1.0e-12,
                    "max_step": 0.025,
                    "domain_check_substeps": 16,
                    "domain": {"min_scale_factor": 0.8},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scan = tmp_path / "scan.yaml"
    scan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_config": "base.yaml",
                "identity_namespace": "scpc-exact-event-integrity-v1",
                "max_runs": 1,
                "resume": True,
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
    return scan


def _single_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(
    "states, message",
    [
        (np.asarray([[1.2, "bad", 0.0, 0.0]], dtype=object), "numeric"),
        (np.asarray([[1.2, np.nan, 0.0, 0.0]]), "finite"),
        (np.asarray([[1.2, 0.0, 0.0]]), "shape"),
    ],
)
def test_exact_turning_state_vectors_are_validated_before_classification(
    states,
    message,
) -> None:
    solution = _simple_solution()
    solution.turning_state_vectors = states
    with pytest.raises(ResultIntegrityError, match=message):
        assess_solution(solution)


def test_nonnumeric_nonterminated_requested_endpoint_is_result_integrity_error() -> None:
    solution = _simple_solution()
    solution.requested_end_time = "not-an-endpoint"
    with pytest.raises(ResultIntegrityError, match="endpoint must be numeric"):
        assess_solution(solution)


@pytest.mark.parametrize("corruption", ["event_state", "requested_endpoint"])
def test_runner_persists_exact_event_corruption_as_result_integrity_error(
    tmp_path,
    monkeypatch,
    corruption,
) -> None:
    scan = _write_domain_protocol(tmp_path)
    output = tmp_path / "output"
    original_integration = runner._integrate_point

    def corrupted_solution(point):
        solution = original_integration(point)
        if corruption == "event_state":
            solution.turning_times = np.asarray([float(solution.t[-1])])
            solution.turning_kinds = ("turnaround",)
            solution.turning_state_vectors = np.asarray(
                [[float(solution.a[-1]), "bad", 0.0, 0.0]],
                dtype=object,
            )
        else:
            solution.termination_kind = None
            solution.termination_time = None
            solution.termination_state_vector = None
            solution.termination_threshold = None
            solution.termination_observed = None
            solution.termination_units = None
            solution.termination_boundaries = ()
            solution.requested_end_time = "not-an-endpoint"
        return solution

    monkeypatch.setattr(runner, "_integrate_point", corrupted_solution)
    index = runner.run_background_scan(scan, output, schema_path=SCAN_SCHEMA)
    row = _single_row(index)

    assert row["status"] == "failed"
    assert row["failure_class"] == "result_integrity_error"
    assert row["exception_type"] == "ResultIntegrityError"
    assert row["termination_record_path"] == ""
    records = output / "termination_records"
    assert not records.exists() or not list(records.glob("*.json"))
