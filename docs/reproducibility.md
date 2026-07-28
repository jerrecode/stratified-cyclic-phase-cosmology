# Reproducibility standard

Every scientific run is configuration-driven and must record:

- source Git commit and dirty-tree state;
- configuration file and SHA-256 digest;
- Python, package, and operating-system versions;
- solver, tolerances, evaluation grid, and event settings;
- random seeds where applicable;
- external release identifiers and local checksums;
- wall-clock duration and output checksums.

## Numerical acceptance

A result intended for the paper must have:

1. an analytical-limit or regression test;
2. a tolerance or resolution-convergence result;
3. a constraint-residual diagnostic;
4. an independent implementation or solver cross-check for load-bearing claims;
5. a serialized data product from which the figure is regenerated.

## Data integrity

Downloaded products should be treated as immutable. When a provider publishes checksums, record and verify them. When checksums are absent, compute a local SHA-256 checksum and store it in a run-specific lock file without claiming it is an upstream checksum.
