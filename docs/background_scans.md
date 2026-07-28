# Deterministic background parameter scans

The Stage 1 scan system classifies homogeneous background integrations over an explicit Cartesian parameter grid. Its purpose is to map numerical and trajectory outcomes without selecting only visually interesting runs and without inferring cyclicity from repeated oscillations.

## Command

```bash
scpc scan-background \
  --config configs/scans/stage1_smoke.yaml \
  --schema configs/scans/scan.schema.json \
  --output results/stage1_smoke_scan
```

The initial runner is deliberately serial. Deterministic correctness, complete failure records, and resumability are treated as prerequisites for parallel execution.

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

The protocol is validated against `configs/scans/scan.schema.json`. Unknown controls, empty axes, unsupported rerun statuses, negative limits, and unknown retention outcomes are rejected before any integration begins.

## Run identities

Every complete numerical specification receives a SHA-256 identity before execution. Mapping keys are sorted and floating-point values are encoded by their exact hexadecimal representation. Consequently:

- mapping insertion order cannot change a run identity;
- adjacent representable floating-point values remain distinct;
- solver method, tolerances, sample count, model parameters, and initial conditions are part of the experiment identity;
- duplicate complete specifications are rejected.

The short `run_id` is a 24-hex-character prefix for filenames and tables. The full SHA-256 digest remains in the index and resume metadata.

## Outcome hierarchy

Completed trajectories are classified only after checking:

1. all stored arrays are nonempty, one-dimensional, equal-length, and finite;
2. the scale factor remains positive;
3. the Friedmann residual remains below the declared threshold;
4. no degenerate turning root is present;
5. sampled Hubble sign changes are represented by root-localized events.

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

Every planned run remains represented even when no solution is returned. Exceptions are classified conservatively as:

- invalid initial Friedmann constraint;
- configuration error;
- physical-domain failure;
- solver failure;
- unexpected error.

A solver exception is not called a spacetime singularity. The exact exception type and message are preserved for audit and later reclassification.

## Output contract

A scan output directory contains:

```text
scan_index.csv
scan_summary.json
scan_metadata.json
outcome_map.png              # when configured
provenance.json
trajectories/
  scpc-<run-id>.nc           # selected valid outcomes only
```

`scan_index.csv` contains one row per attempted experiment, including:

- short and full run hashes;
- status and outcome or failure class;
- first-class scan coordinates;
- the complete experiment specification;
- event counts and event sequence;
- maximum constraint residual;
- return-sequence summaries;
- solver metadata;
- retained trajectory path, when applicable.

Nested fields are canonical JSON strings so the CSV remains interoperable without discarding structure.

## Resume integrity

The index is atomically replaced after each attempted run. Resume metadata records:

- scan-schema version and hash;
- scan-protocol hash;
- base-configuration hash;
- ordered full hashes of every planned run.

On resume, the runner verifies every indexed run ID, full hash, coordinate mapping, complete specification, status, and retained trajectory file. A copied output directory can be resumed from another filesystem location because integrity relies on content hashes and configured references rather than absolute path equality.

Changing the protocol or base configuration requires a new output directory. Configured reruns remove the old row and retained trajectory before executing the identical experiment again.

## Trajectory retention

The index always stores every result, but full trajectories are retained only when:

- the completed solution passes numerical-validity checks;
- its outcome is listed in the protocol retention policy;
- the hard trajectory-count limit has not been reached.

NetCDF trajectories are written through a temporary file and atomically renamed. A partial trajectory is removed if serialization fails.

## Outcome maps

A configured two-axis map is generated only for a complete Cartesian index. Coordinate pairs must be unique and numeric. Cell boundaries are derived from the actual parameter values, so nonuniform parameter spacing is not visually converted into uniform spacing. Failed and rejected runs remain visible as separate categorical labels.

## Scientific boundary

This scan layer classifies numerical experiments. It does not establish:

- geodesic completeness or a spacetime singularity theorem;
- recurrence across solver methods and tolerances;
- a stable periodic orbit;
- perturbative viability;
- a cosmological observable;
- an empirical fit.

Those conclusions remain gated by the later stages in `docs/research_roadmap.md`.
