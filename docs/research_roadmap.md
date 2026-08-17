# SCPC gated research roadmap

This roadmap orders work by evidential dependency. Later stages must not be used to compensate for a failure at an earlier stage. In particular, parameter fitting cannot make an inconsistent action viable, and visually periodic trajectories cannot substitute for recurrence or stability analysis.

The machine-readable source of truth for current gate status is `configs/scientific_gates.yaml`. `scpc validate-gates` rejects dependency/status combinations that would let a later claim outrun an unresolved prerequisite.

## Current status

The repository currently provides:

- a canonical stratification field in homogeneous FLRW spacetime;
- standard late-time FLRW comparison backgrounds;
- exact event-based detection of candidate bounces and turnarounds;
- checked finite/physical-domain termination with exact terminal evidence and durable failure classification;
- an independent local turning-point physics audit that recomputes the Friedmann H=0 surface, Raychaudhuri derivative, energy-condition combinations, and exact bounce/turnaround theorem from root-localized event states;
- Friedmann-constraint monitoring;
- explicit same-kind turning-point return metrics;
- tolerance-ladder and cross-solver verification machinery;
- deterministic/stochastic background-scan infrastructure with retained rejected outcomes;
- an executable Stage 1 recurrence-candidate audit that additionally requires exact event-topology agreement and an algorithmically distinct solver;
- a versioned public-data release registry, now including the 2026 DESI DR2 Lyman-alpha full-shape/AP result as a separate comparator rather than silently combining it with the pinned 2025 BAO covariance;
- an official DESI DR2 2025 BAO-only and BAO+BBN released-chain reproduction benchmark with GetDist convergence checks, exact-CAMB diagnostics, full-covariance residual treatment, and cryptographically pinned external products;
- a modular manuscript and reproducibility workflow exercised by dedicated CI, inference, and paper-build jobs.

The canonical scalar baseline is a reference implementation. It is not yet a demonstrated cyclic theory. The DESI DR2 calculation validates the observational/inference machinery under standard flat-LambdaCDM assumptions; it is not an SCPC fit and does not advance SCPC directly to the inference gate.

## Current gate assessment

- **Stage 0 numerical baseline:** satisfied for the currently implemented baseline workflows. This is a statement about implementation consistency, not physical viability.
- **Stage 1 parameter-space classification:** open. Infrastructure and physical-domain/failure semantics are substantially implemented, and candidate verification is now executable, but no converged parameter region has yet passed the repeated-return, exact-event, multi-tolerance, multi-solver gate.
- **Stage 2 homogeneous recurrence/stability:** blocked. No stable limit cycle has been demonstrated with a Poincare map, converged monodromy/Floquet analysis, or equivalent recurrence/stability evidence.
- **Stage 3 perturbative/EFT health:** blocked. A complete perturbative no-ghost, gradient-stability, and strong-coupling analysis for a viable cyclic background is not available.
- **Stage 4 SCPC observable transfer functions:** blocked. The standard-cosmology CAMB/BAO machinery is a comparator and validation backend; it is not a derived SCPC Boltzmann/transfer calculation.
- **Stage 5 SCPC statistical inference:** blocked. The official DESI posterior reproduction demonstrates statistical plumbing for future inference, but no release likelihood is currently constraining an independently derived SCPC observable prediction.
- **Stage 6 optional thermodynamic/topological extensions:** blocked downstream and must not be used to bypass the physical viability gates.

## Stage 0: reproducible numerical baseline

**Objective:** establish that the implemented equations are solved consistently.

Required evidence:

1. CI, tests, manifest validation, reproduction, and paper builds pass.
2. The Friedmann residual remains below a declared threshold.
3. A tolerance ladder converges toward a declared reference run.
4. At least two independent integrators agree on a common stored grid.
5. Every run records configuration hashes, software versions, solver settings, and output checksums.
6. Root-localized turning states reproduce the local H=0 Friedmann and Raychaudhuri identities and agree with their recorded event classification.

**Current state:** passed for the implemented baseline and verification workflows. This does not establish recurrence or physical viability.

**Gate:** no parameter scan is scientifically interpretable until Stage 0 passes.

## Stage 1: background parameter-space classification

**Objective:** map which parameter and initial-condition regions produce expansion, recollapse, physical-domain termination, one-off bounces, repeated turning points, or candidate recurrence without interpreting numerical analysis boundaries as physical singularities.

Implementation requirements:

- deterministic parameter-grid and separately identified seeded stochastic sampling modes;
- future low-discrepancy/refined sampling near outcome boundaries without silently changing an original experiment set;
- physical-domain and finite-state termination rules;
- event-complete integration with checked zero-crossing detection of the Hubble parameter;
- classification independent of plotting resolution;
- exact terminal-state and boundary evidence for terminated runs;
- storage of failures and rejected runs, not only successful examples;
- durable tabular/content-addressed evidence sufficient to reconstruct why a run was retained or rejected;
- a candidate protocol separate from ordinary outcome classification.

The implemented candidate protocol in `configs/scpc_candidate_verification.yaml` requires all of the following before promotion to Stage 2 input:

1. at least two return metrics of one same-kind event, requiring at least three root-localized turning points of that kind;
2. every required return error below the declared threshold in the reference run;
3. reproduction of those close returns under tighter primary-solver tolerances;
4. reproduction with a different solver algorithm (`Radau` versus the primary `DOP853` in the current protocol);
5. agreement of the complete exact event sequence, event count, event times, and event states;
6. agreement of the full stored trajectory including the unwrapped scalar lift;
7. completion of the declared integration interval without physical-domain termination;
8. Friedmann residuals below threshold; and
9. local turning-point identity audits consistent with the recorded event kind.

The candidate audit is deliberately non-promotional: a clean `not_candidate_*` result is successful execution and valid negative scientific evidence. It does not make CI fail merely because the model did not become cyclic.

Primary outputs:

- phase diagram of outcome class versus parameters;
- constraint-residual and solver/failure maps;
- distributions of bounce and turnaround scales where they exist;
- turning-point return-error maps;
- sensitivity to initial conditions;
- explicit maps of numerical-domain exits separated from physical/model classifications;
- machine-readable candidate-verification reports for selected points.

**Current state:** open. The baseline has not passed the complete recurrence-candidate gate. Richer phase diagrams, low-discrepancy/refined sampling, larger restart tests, and deterministic parallel equivalence remain Stage 1 work. Deterministic computational resource ceilings remain a separate issue and must not be conflated with physical-domain boundaries.

**Gate:** a candidate cyclic region requires repeated close same-kind returns reproduced by the full multi-tolerance, multi-solver, exact-event protocol. A one-run close return is insufficient.

## Stage 2: homogeneous recurrence and stability

**Objective:** distinguish periodic, quasiperiodic, drifting, transient, and chaotic homogeneous trajectories.

Required methods:

- Poincare sections at a declared crossing condition;
- return maps and cycle-period convergence;
- variational equations around the background trajectory;
- monodromy matrix and Floquet multipliers for periodic candidates;
- Lyapunov diagnostics for nonperiodic recurrent candidates;
- basin-of-attraction and fine-tuning analysis.

**Current state:** blocked by Stage 1. Floquet calculations must not be run merely because a trajectory looks oscillatory.

**Gate:** a background may be called a stable limit-cycle candidate only when its nontrivial Floquet multipliers satisfy the declared stability criterion and the result is numerically converged.

## Stage 3: covariant perturbations and effective-theory health

**Objective:** determine whether the background is physically admissible beyond homogeneity.

Required derivations and checks:

- gauge-invariant scalar, vector, and tensor perturbation equations;
- quadratic action where applicable;
- no-ghost kinetic coefficients;
- positive gradient terms or declared controlled exceptions;
- finite propagation speeds compatible with the effective theory;
- strong-coupling scale above every scale used in the calculation;
- regular matching or evolution through every bounce and turnaround.

**Current state:** blocked by Stage 2.

**Gate:** unstable or strongly coupled parameter regions are rejected before observable calculations.

## Stage 4: observable transfer functions

**Objective:** map viable primordial or background dynamics to quantities measured by cosmological surveys.

Required observables may include:

- background distances and expansion-rate combinations;
- primordial scalar and tensor spectra;
- CMB temperature, polarization, and lensing spectra;
- matter power spectrum and growth observables;
- BAO distance combinations;
- supernova distance moduli;
- stochastic gravitational-wave spectra.

Implementation should extend or interface with validated Boltzmann software rather than silently reimplementing it. Every mapping from an SCPC field variable to an observable must be derived and dimensionally documented. Standard-LambdaCDM CAMB calculations used for regression/reproduction do not count as SCPC transfer functions.

**Current state:** blocked by Stage 3.

**Gate:** apparent frequencies or features must survive duration, cadence, window, resolution, transfer-function, and look-elsewhere tests.

## Stage 5: statistical inference and model comparison

**Objective:** constrain rather than merely illustrate the model.

Required practices:

- declared priors and parameter transformations;
- release-native likelihoods and nuisance parameters where available;
- synthetic parameter-recovery tests;
- sampler convergence diagnostics;
- posterior-predictive checks;
- comparison against matched standard and cyclic baselines;
- information criteria or Bayesian evidence with complexity penalties;
- publication of chains/configs and exact immutable data-product identifiers or cryptographic local pins.

The current DESI DR2 released-chain reproduction demonstrates these practices for a standard flat-LambdaCDM benchmark. It is a pipeline qualification step only. SCPC enters this stage only after Stage 4 supplies validated SCPC predictions for the measured observables.

**Current state:** blocked by Stage 4 for SCPC.

**Gate:** empirical support is not claimed from a better visual overlay, an uncorrected maximum-likelihood improvement, a standard-cosmology reproduction, or any fit performed before the model-to-observable map exists.

## Stage 6: thermodynamic and topological extensions

Thermodynamic state-space geometry, entropy production, compact-field defects, winding, or vorticity should be added only after the baseline spacetime and field theory are consistent. Each extension requires its own degrees of freedom, action or declared effective equations, conserved or produced currents, dimensional conventions, invariants, limiting cases, stability analysis, and falsification criteria.

The term "vortex" should remain absent from the model name unless a nonzero vorticity, circulation, winding number, or other invariant is explicitly defined and realized by solutions.

**Current state:** blocked by the preceding scientific gates.

## Release gates

| Release class | Minimum evidence |
|---|---|
| `0.x framework` | executable equations, tests, manifests, reproducibility |
| `0.x background candidate` | converged repeated turning points, exact-event reproduction, and outcome maps |
| `0.x stable-background candidate` | homogeneous Floquet or equivalent stability evidence |
| `0.x perturbatively viable candidate` | ghost, gradient, and strong-coupling checks |
| `0.x predictive candidate` | derived observables and validated transfer pipeline |
| `1.0 scientific model` | reproducible SCPC inference, limitations, and explicit rejection domain |
