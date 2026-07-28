# Deterministic background parameter scans

The Stage 1 scan system classifies homogeneous background integrations over an explicit parameter set. Its purpose is to map numerical and trajectory outcomes without selecting only visually interesting runs and without inferring cyclicity from repeated oscillations.

## Command

```bash
scpc scan-background \
  --config configs/scans/stage1_smoke.yaml \
  --schema configs/scans/scan.schema.json \
  --output results/stage1_smoke_scan
```

The initial runner is deliberately serial. Deterministic correctness, complete failure records, transactional replacement, and resumability are prerequisites for parallel execution.

## Protocol structure

A scan protocol declares a complete background base configuration, explicit dotted paths to vary, a maximum Cartesian run count, numerical classification thresholds, trajectory-retention policy, optional two-axis visualization, and resume policy. Unknown controls, empty axes, nonfinite numbers, unsupported statuses, negative limits, and unknown retention outcomes are rejected before integration.

Outcome-map preflight requires exactly two numeric axes, at least two distinct values on each axis, unique coordinate pairs, and a complete Cartesian grid. Invalid visualization plans fail before an output directory or numerical run is created.

## Run identities

Every complete numerical specification receives a SHA-256 identity before execution. Values are normalized to the types consumed by the executor, keys are sorted, and floating-point values are encoded exactly. Solver method, tolerances, domain thresholds, `max_step`, dense-check subdivision count, model parameters, and initial conditions are part of the identity. Execution-equivalent duplicate declarations are rejected.

## Declared integration domain

A run may declare analysis boundaries and the temporal resolution used to verify them:

```yaml
run:
  t_start: 0.0
  t_end: 1.0
  samples: 201
  method: DOP853
  rtol: 1.0e-10
  atol: 1.0e-12
  max_step: 0.025
  domain_check_substeps: 16
  domain:
    min_scale_factor: 0.8
    max_scale_factor: 10.0
    max_total_density: 100.0
    max_abs_hubble: 10.0
    max_abs_ricci_scalar: 100.0
    max_abs_field: 20.0
    max_abs_field_velocity: 20.0
```

Every threshold and `max_step` must be finite and positive. `domain_check_substeps` must be an integer of at least two. `min_scale_factor` must be smaller than `max_scale_factor`. A coordinate field bound is rejected for a circular target because it is not invariant under the compact identification.

### Root events and dense-step verification

SciPy root events localize endpoint sign changes, but a nonmonotonic observable can leave and re-enter a domain within one accepted step. SCPC therefore:

1. requires a finite solver `max_step` whenever domain surfaces are configured;
2. requests dense solver output on the internal accepted-step grid;
3. evaluates every configured residual at `domain_check_substeps + 1` points in every accepted step;
4. root-localizes the first sampled positive-to-nonpositive residual bracket;
5. uses the earliest of the solver event and dense-step detection;
6. truncates the uniformly sampled output at that exact state.

Production results must demonstrate convergence under smaller `max_step` and larger `domain_check_substeps`. These controls bound the unresolved temporal scale; they do not constitute a theorem that arbitrary infinitely rapid excursions are impossible.

### Coincident surfaces

At the first terminal state, every configured observable is recomputed. All surfaces agreeing with their thresholds within solver-derived event tolerance are serialized as a unique, lexically ordered boundary set. The first lexical kind supplies backward-compatible scalar fields, while the full set remains authoritative. This prevents correlated limits from being lost through solver event ordering.

The exact termination time and state are appended even when they lie between plotting samples. The record contains the complete coincident boundary set, primary kind, exact time and state, thresholds, observed values, units, requested endpoint, and completion flag.

A domain-terminated run is returned successfully by the solver but classified as `physical_domain_termination` and rejected for full-interval morphology. It is not a solver failure and is not a spacetime-singularity claim. Ending early without complete declared termination metadata is a result-integrity error.

The reproducible example is `configs/scans/stage1_domain_example.yaml`.

## Outcome hierarchy

Returned trajectories are classified only after validating array shape and finiteness, time ordering, turning-event consistency, exact termination metadata, positivity of the scale factor, Friedmann residual, degenerate roots, and complete sampled Hubble-crossing reconciliation. Every recorded termination observable is recomputed from the final state and model parameters; corrupted kind, value, threshold, or units are rejected as result-integrity failures.

Morphology classes include monotonic expansion or contraction, quasi-static behavior, recollapse, one-off bounce, one bounce-turnaround pair, and repeated turning points. Repeated turning points are not a cyclicity claim.

## Failure and rejection records

Every planned run remains represented. Distinct records cover invalid initial constraints, configuration errors, undeclared physical-domain failure, solver failure, result-integrity failure, output-serialization failure, unexpected errors, and returned-but-rejected classes such as declared domain termination or constraint violation. None of these labels alone proves a spacetime singularity.

## Output contract

A scan output directory contains:

```text
scan_index.csv
scan_summary.json
scan_metadata.json
outcome_map.png                         # when configured
provenance.json
trajectories/
  scpc-<run-id>-<content-hash>.nc       # selected valid outcomes only
```

The CSV records run hashes, coordinates, normalized specification, status, outcome or failure, event diagnostics, constraint residual, return summaries, completion state, primary termination fields, canonical JSON for the full coincident boundary set, solver metadata, and retained trajectory path. The summary counts every boundary kind, including coincident surfaces.

NetCDF termination products contain the shared exact terminal state and a `termination_boundary` dimension with kind codes, thresholds, observed values, and per-boundary units. JSON attributes preserve the full named representation.

## Resume and transactions

Resume metadata binds outputs to scan/schema/base hashes, planned run hashes, SCPC source hash, Python, numerical dependency versions, platform, and architecture. Existing rows, complete specifications, statuses, hashes, and retained files are validated before any skip.

Reruns retain the old durable row and trajectory until a replacement trajectory is content-addressed and the new row is atomically committed. Only then is an obsolete prior trajectory removed. Startup recovery removes only unreferenced scan-owned transaction files.

## Scientific boundary

This layer classifies numerical experiments. It does not establish geodesic completeness, a singularity theorem, recurrence across numerical methods, a stable periodic orbit, perturbative viability, an observable, or an empirical fit. Those claims remain gated by `docs/research_roadmap.md` and issues #3–#8.
