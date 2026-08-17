# Changelog

All notable changes follow Keep a Changelog conventions.

## [Unreleased]

### Added

- Same-kind turning-point return metrics using exact solver event states.
- Explicit real-line or circular stratification-field target topology.
- Separate potential period, compact target circumference, and winding diagnostics.
- Conservative recurrence classification that avoids stability overclaims.
- Tolerance-ladder and cross-solver background verification.
- Machine-readable verification configuration and archived CI reports.
- Gated research roadmap and staged implementation issues.
- Independent exact turning-point physics audits that recompute the local Friedmann surface, Raychaudhuri derivative, energy-condition combinations, and bounce/turnaround theorem.
- A non-promotional Stage 1 recurrence-candidate gate requiring repeated close same-kind returns, tolerance convergence, an algorithmically distinct Radau cross-check, and exact event-topology agreement.
- A machine-validated scientific dependency ledger exposed through `scpc validate-gates`.
- An analytic turning-feasibility certificate that can rule out future regular turnaround before a long integration; the canonical expanding baseline is certified excluded because `a0=1 > sqrt(3/3.5)`.
- Separate registry tracking for the 2026 DESI DR2 Lyman-alpha full-shape/AP result and its companion mock-validation study.
- Regression tests for exact-event physics, event-topology disagreement, scientific-gate dependency integrity, nonfinite solver/model inputs, and analytic turning exclusions.

### Changed

- Background diagnostics now serialize return metrics, topology, winding, recurrence classification, exact turning-point audits, and an analytic turning-feasibility certificate.
- Solver verification compares the unwrapped scalar and rejects hidden phase slips.
- Direct background APIs now reject nonfinite model parameters, initial conditions, time ranges, tolerances, and nonintegral discrete controls before solver entry.
- CI now captures Ruff diagnostics on failure, validates the scientific gate ledger, runs the recurrence-candidate audit, and archives its machine-readable evidence.
- The manuscript and README document the implemented verification protocol, exact no-turnaround bound for the canonical baseline, and the distinction between the 2025 DESI production likelihood and the separately tracked 2026 Lyman-alpha result.
- The canonical baseline conclusion is strengthened from finite-horizon non-recurrence to an analytic exclusion of every future regular turnaround on its configured expanding branch.

## [0.1.0] - 2026-07-28

### Added

- Clean-room SCPC repository and model constitution.
- Canonical covariant stratification-field background in FLRW spacetime.
- Standard and analytic comparison models on common grids.
- Constraint and event diagnostics with NetCDF result products.
- Versioned public cosmology data manifest, schema, and release-specific metadata.
- Reproducible workflows, provenance capture, tests, figures, and modular LaTeX paper.
- Continuous scientific verification and paper-build workflows.
