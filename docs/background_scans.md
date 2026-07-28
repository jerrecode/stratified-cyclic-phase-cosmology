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
- solver method, tolerances, sample count, model parameters, and initial conditions are part of the experiment identity;
- duplicate complete executable specifications are rejected.

The short `run_id` is a 24-hex-character prefix for filenames and tables. The full SHA-256 digest remains in the index and resume metadata.

## Outcome hierarchy

Completed trajectories are classified only after checking:

1. all stored arrays are nonempty, one-dimensional, equal-length, and finite;
2. stored times are strictly increasing;
3. event times and event kinds are internally consistent;
4. the scale factor remains positive;
5. the Friedmann residual remains below the declared threshold;
6. no degenerate turning root is present;
7. every detectable sampled Hubble sign transition, including transitions across zero plateaus, is represented by a matching root-localized event.

The current morphology classes are:

- monotonic expansion;
- monotonic contraction;
- quasi-static or ambiguous behavior without a recorded crossing;
- recollapse without a bounce in the integrated interval;
- one-off bounce;
- one bounce and one turnaround;
- repeated turning points.

Repeated turning points are not labelled a cyclic solution. Same-kind return diagnostics remain separate and do not establish recurrence or stability.

## Failure records

Every planned run remains represented even when no accepted solution is returned. Exceptions and rejected trajectories are classified conservatively as:

- invalid initial Friedmann constraint;
- configuration error;
- physical-domain failure;
- solver failure;
- result-integrity error;
- output-serialization error;
- unexpected error;
- returned but numerically rejected trajectory classes such as a constraint violation or nonfinite state.

A solver, integrity, or output exception is not called a spacetime singularity. The exact exception type and message are preserved for audit and later reclassification.

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
- solver metadata;
- retained trajectory path, when applicable.

Nested fields are canonical JSON strings so the CSV remains interoperable without discarding structure.

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

If execution stops before the row replacement, the previous row and trajectory remain valid. On the next resume, pending and unreferenced scan-owned transaction files are removed after the existing index has been validated. A deterministic rerun may recreate the same content-addressed filename, in which case that file becomes the newly referenced result rather than remaining an orphan.

## Trajectory retention

The index always stores every result, but full trajectories are retained only when:

- the completed solution passes numerical-validity checks;
- its outcome is listed in the protocol retention policy;
- the hard trajectory-count limit has not been reached.

NetCDF trajectories use content-addressed filenames and atomic finalization. Serialization failures are recorded as output failures without replacing a previous durable result.

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
