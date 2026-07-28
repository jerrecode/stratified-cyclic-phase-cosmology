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

A scan protocol declares:

- one complete background base configuration;
- explicit dotted paths to vary;
- finite value lists for each axis;
- a maximum Cartesian run count;
- numerical classification thresholds;
- trajectory-retention outcomes and a hard retention limit;
- optional two-axis outcome-map settings;
- resume and rerun policy.

The protocol is validated against `configs/scans/scan.schema.json`. Unknown controls, empty axes, nonfinite numbers, unsupported rerun statuses, negative limits, and unknown retention outcomes are rejected before any integration begins.

When an outcome map is requested, preflight additionally requires exactly two numeric scan axes, two or more distinct values on each axis, unique coordinate pairs, and a complete Cartesian grid. Misspelled axes, singleton dimensions, and hidden third dimensions fail before the output directory or any integration is created.

## Run identities

Every complete numerical specification receives a SHA-256 identity before execution. Values are first normalized to the types consumed by the executor. Mapping keys are sorted and floating-point values are encoded by their exact hexadecimal representation. Consequently:

- mapping insertion order cannot change a run identity;
- adjacent representable floating-point values remain distinct;
- execution-equivalent declarations such as an integer sample count written as `101` or `101.0` share one identity;
- nonintegral values for integer execution fields are rejected;
- solver method, tolerances, domain thresholds, model parameters, and initial conditions are part of the experiment identity;
- duplicate complete executable specifications are rejected.

The short `run_id` is a 24-hex-character prefix for filenames and tables. The full SHA-256 digest remains in the index and resume metadata.

## Declared integration domain

A background run may declare root-localized analysis boundaries under `run.domain`:

```yaml
run:
  t_start: 0.0
  t_end: 1.0
  samples: 201
  method: DOP853
  rtol: 1.0e-10
  atol: 1.0e-12
  domain:
    min_scale_factor: 0.8
    max_scale_factor: 10.0
    max_total_density: 100.0
    max_abs_hubble: 10.0
    max_abs_ricci_scalar: 100.0
    max_abs_field: 20.0
    max_abs_field_velocity: 20.0
```

Every configured threshold must be finite and positive. `min_scale_factor` must be smaller than `max_scale_factor`. A coordinate bound on the field is rejected for a circular target space because it is not invariant under the compact identification.

Reaching a configured surface terminates the integration at the solver-localized root. The exact termination time and state are appended to the stored trajectory even when the root lies between plotting samples. The record contains:

- termination kind;
- exact time;
- threshold;
- observed boundary value;
- units;
- whether the requested endpoint was reached.

A domain-terminated run is returned successfully by the solver but is classified as `physical_domain_termination` and rejected for full-interval morphology. It is not a solver failure and is not a spacetime-singularity claim. Ending before the requested endpoint without a declared termination event is a result-integrity error.

The reproducible one-axis example is `configs/scans/stage1_domain_example.yaml`.

## Outcome hierarchy

Returned trajectories are classified only after checking:

1. all stored arrays are nonempty, one-dimensional, equal-length, and finite;
2. stored times are strictly increasing;
3. event times and event kinds are internally consistent;
4. termination metadata is complete and agrees with the exact final state;
5. the scale factor remains positive;
6. the Friedmann residual remains below the declared threshold;
7. no degenerate turning root is present;
8. every detectable sampled Hubble sign transition, including transitions across zero plateaus, is represented by a matching root-localized event.

The current classes include:

- physical-domain termination before the requested endpoint;
- monotonic expansion;
- monotonic contraction;
- quasi-static or ambiguous behavior without a recorded crossing;
- recollapse without a bounce in the integrated interval;
- one-off bounce;
- one bounce and one turnaround;
- repeated turning points;
- explicitly rejected numerical-integrity classes.

Repeated turning points are not labelled a cyclic solution. Same-kind return diagnostics remain separate and do not establish recurrence or stability.

## Failure records

Every planned run remains represented even when no accepted solution is returned. Exceptions and rejected trajectories are classified conservatively as:

- invalid initial Friedmann constraint;
- configuration error;
- physical-domain failure outside the declared event system;
- solver failure;
- result-integrity error;
- output-serialization error;
- unexpected error;
- returned but rejected classes such as declared domain termination, constraint violation, or nonfinite state.

A solver, integrity, output, or domain-boundary record is not called a spacetime singularity. The exact exception type and message are preserved for audit and later reclassification.

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

`scan_index.csv` contains one row per attempted experiment, including:

- short and full run hashes;
- status and outcome or failure class;
- first-class scan coordinates;
- the complete normalized experiment specification;
- event counts and event sequence;
- maximum constraint residual;
- return-sequence summaries;
- completion and domain-termination fields;
- solver metadata;
- retained trajectory path, when applicable.

Nested fields are canonical JSON strings so the CSV remains interoperable without discarding structure. `scan_summary.json` includes counts by status, outcome, failure class, and termination kind.

## Resume integrity

Resume metadata records:

- scan-schema version and hash;
- scan-protocol hash;
- base-configuration hash;
- ordered full hashes of every planned run;
- an SCPC source-tree hash;
- Python implementation and version;
- numerical dependency versions;
- operating-system platform and machine architecture.

On resume, the runner verifies every indexed run ID, full hash, coordinate mapping, complete specification, status, and retained trajectory file before skipping any point. A copied output directory can move to another filesystem location, but it cannot be resumed under a changed implementation or declared numerical runtime. Such changes require a new output directory.

## Transactional reruns and recovery

A rerun does not delete the previous durable result before replacement. The sequence is:

1. retain the existing index row and retained trajectory;
2. compute the replacement attempt;
3. write a candidate trajectory under a content-addressed filename;
4. atomically replace the index row;
5. only then remove an obsolete previously referenced trajectory;
6. atomically refresh the summary and final provenance products.

If execution stops before the row replacement, the previous row and trajectory remain valid. On the next resume, pending and unreferenced scan-owned transaction files are removed after the existing index has been validated.

## Trajectory retention

The index always stores every result, but full trajectories are retained only when:

- the completed solution reaches the requested endpoint;
- it passes numerical-validity checks;
- its outcome is listed in the protocol retention policy;
- the hard trajectory-count limit has not been reached.

Domain-terminated runs remain represented by exact scalar termination fields in the index. They are not retained as full candidate trajectories by the current policy because they did not complete the declared interval.

## Outcome maps

A configured map represents exactly two numeric scan axes and is preflight-validated before execution. Coordinate pairs must be unique, both axes must contain at least two distinct values, and the plan must be a complete Cartesian grid. Cell boundaries are derived from the actual parameter values, so nonuniform parameter spacing is not visually converted into uniform spacing. Failed and rejected runs remain visible as separate categorical labels.

## Scientific boundary

This scan layer classifies numerical experiments. It does not establish:

- geodesic completeness or a spacetime singularity theorem;
- recurrence across solver methods and tolerances;
- a stable periodic orbit;
- perturbative viability;
- a cosmological observable;
- an empirical fit.

Those conclusions remain gated by the later stages in `docs/research_roadmap.md`.
