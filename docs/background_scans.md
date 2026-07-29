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

Turning roots materially later than a dense termination are discarded. Roots differing only by endpoint roundoff are timestamp-clamped to the exact terminal time. Production results must demonstrate convergence under smaller `max_step` and larger `domain_check_substeps`. These controls bound the unresolved temporal scale; they do not constitute a theorem that arbitrary infinitely rapid excursions are impossible.

### Coincident surfaces

At the first terminal state, every configured observable is recomputed. All surfaces agreeing with their thresholds within a tolerance scaled to the actual observable, threshold, and solver tolerances are serialized as a unique, lexically ordered boundary set. The first lexical kind supplies compatibility scalar fields, while the full set remains authoritative. This prevents correlated limits from being lost through solver event ordering and avoids promoting nearby small thresholds through a unit-scale tolerance floor.

The exact termination time and four-component state are appended even when they lie between plotting samples. The index records the boundary set, primary fields, exact state, exact endpoint Friedmann residual, requested endpoint, completion flag, structured unmatched-crossing evidence, and a reference to independent content-addressed evidence.

A domain-terminated integration is returned successfully by the solver but rejected for full-interval morphology. Its outcome is `physical_domain_termination` unless a documented higher-priority numerical rejection applies: `constraint_violation`, `degenerate_turning_event`, or `unresolved_event_detection`. It is not a solver failure and is not a spacetime-singularity claim. Ending early without complete declared termination metadata is a result-integrity error.

The reproducible example is `configs/scans/stage1_domain_example.yaml`.

## Outcome hierarchy

Returned trajectories are classified only after validating array shape and finiteness, time ordering, turning-event consistency, exact termination metadata, positivity of the scale factor, Friedmann residual, degenerate roots, and complete sampled Hubble-crossing reconciliation. Every recorded termination observable is recomputed from the final state and model parameters; corrupted kind, value, threshold, units, timestamp, state, or outcome evidence is rejected as a result-integrity failure.

For a terminated attempt, resume reconstructs the rejection precedence from the scan's immutable classification threshold and durable event evidence. A constraint violation outranks a degenerate event, which outranks an unresolved sampled crossing, which outranks plain physical-domain termination.

Morphology classes include monotonic expansion or contraction, quasi-static behavior, recollapse, one-off bounce, one bounce-turnaround pair, and repeated turning points. Repeated turning points are not a cyclicity claim.

## Failure and rejection records

Every planned run remains represented. Distinct records cover invalid initial constraints, configuration errors, undeclared physical-domain failure, solver failure, result-integrity failure, output-serialization failure, unexpected errors, and returned-but-rejected classes such as declared domain termination or constraint violation. None of these labels alone proves a spacetime singularity.

## Output contract

A scan output directory contains:

```text
scan_index.csv
scan_summary.json
scan_metadata.json
outcome_map.png                              # when configured
provenance.json
trajectories/
  scpc-<run-id>-<content-hash>.nc            # selected valid outcomes only
termination_records/
  scpc-<run-id>-<content-hash>.json          # exact evidence for terminated attempts
```

The CSV records run hashes, coordinates, normalized specification, status, outcome or failure, event diagnostics, unmatched sampled crossings, constraint residuals, return summaries, completion state, exact termination state, full coincident boundary set, evidence path and checksum, solver metadata, and retained trajectory path. The summary counts every boundary kind, including coincident surfaces, plus retained trajectory and termination-record counts.

A termination-record artifact is canonical JSON addressed by its SHA-256 digest. It independently authenticates the signed state components, exact time and residual, complete boundary set, event sequence, unmatched-crossing evidence, outcome, numerical-validity flag, solver tolerances, requested endpoint, and integration-domain declaration. Direct `SCPCSolution.to_xarray()` serialization additionally supports a `termination_boundary` dimension for scientific products that intentionally retain the returned solution.

## Resume and transactions

Resume metadata binds outputs to scan/schema/base hashes, planned run hashes, SCPC source hash, Python, numerical dependency versions, platform, and architecture. Existing rows, complete specifications, statuses, hashes, retained files, and termination artifacts are validated before any skip.

For a terminated row, resume:

1. reconstructs the model and domain from the planned immutable specification;
2. recomputes the endpoint Friedmann residual and complete coincident boundary set;
3. verifies event counts, unmatched-crossing evidence, and rejection precedence;
4. verifies the content-addressed artifact checksum and canonical bytes;
5. requires the artifact and CSV evidence to agree exactly.

Reruns retain the old durable row, trajectory, and termination artifact until replacement artifacts are content-addressed and the new row is atomically committed. Only then are obsolete prior artifacts removed. Startup recovery removes only unreferenced scan-owned transaction files.

## Scientific boundary

This layer classifies numerical experiments. It does not establish geodesic completeness, a singularity theorem, recurrence across numerical methods, a stable periodic orbit, perturbative viability, an observable, or an empirical fit. Those claims remain gated by `docs/research_roadmap.md` and issues #3–#8.
