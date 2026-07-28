"""Reproduce the baseline simulation and standardized model comparison."""

from __future__ import annotations

from scpc.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "simulate",
                "configs/baseline/scpc_closed.yaml",
                "--output",
                "results/baseline",
            ]
        )
    )
