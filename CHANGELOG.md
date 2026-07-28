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

### Changed

- Background diagnostics now serialize return metrics, topology, winding, and recurrence classification.
- Solver verification compares the unwrapped scalar and rejects hidden phase slips.
- CI now captures Ruff diagnostics on failure and requires numerical verification.
- The manuscript and README document the implemented verification protocol.

## [0.1.0] - 2026-07-28

### Added

- Clean-room SCPC repository and model constitution.
- Canonical covariant stratification-field background in FLRW spacetime.
- Standard and analytic comparison models on common grids.
- Constraint and event diagnostics with NetCDF result products.
- Versioned public cosmology data manifest, schema, and release-specific metadata.
- Reproducible workflows, provenance capture, tests, figures, and modular LaTeX paper.
- Continuous scientific verification and paper-build workflows.
