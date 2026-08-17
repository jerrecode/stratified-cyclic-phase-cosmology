# Deterministic computational resource limits

Stage 1 supports the execution-normalized run field
`run.resource_limits.max_rhs_evaluations`. It is a finite positive integer and
limits only RHS calls made by SciPy's ODE solver. RHS evaluations used by
diagnostics or event observables are counted separately and do not consume the
solver budget.

Exhaustion raises `ComputationalResourceLimitExceeded` and is serialized as the
failure class `resource_limit_exceeded`. The record stores the limit kind,
configured limit, completed and attempted solver-RHS counts, the attempted
evaluation time, and the diagnostic-RHS count. It deliberately stores no
attempted state as a physical endpoint and assigns no morphology, recurrence,
viability, singularity, or cosmological outcome.

Successful solutions record `rhs_evaluations_consumed`, `solver_nfev`,
`diagnostic_rhs_evaluations`, and `rhs_evaluation_limit` in solver metadata.
Resume validation requires the solver counter to agree with `solver_nfev` and
requires the configured ceiling to match the immutable run specification.

Resource limits participate in canonical run identity, scan metadata, and
provenance. Changing the ceiling therefore defines a different numerical
experiment rather than silently reusing a prior row. Wall-clock deadlines are
intentionally excluded because machine load would make them nonreproducible.
