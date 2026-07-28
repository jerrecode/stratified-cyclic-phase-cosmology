# Scientific scope and nonclaims

## Baseline ontology

The baseline theory contains a Lorentzian metric g_mu_nu, ordinary matter and radiation fluids, and a canonical scalar stratification field phi. Distinguishable phase strata are minima or sectors of the field potential. The integer labeling of those sectors is not assumed to be physical discrete time.

## Structures kept distinct

- **Spacetime curvature:** R[g], R_mu_nu[g], and curvature invariants derived from the metric.
- **Field-space structure:** the geometry or topology of the stratification-field target space.
- **Thermodynamic state-space geometry:** a fluctuation metric derived from a declared thermodynamic potential and ensemble.
- **Numerical grid:** a computational approximation with convergence requirements.

## Baseline nonclaims

The initial code does not claim:

- a proven stable cosmological limit cycle;
- fundamental discrete time;
- a physical vortex without a winding or vorticity invariant;
- an entropy reset between cycles;
- a CMB or gravitational-wave detection;
- equivalence between thermodynamic and spacetime curvature;
- validity beyond the declared effective theory.

## Mandatory rejection criteria

A model instance is rejected if it violates its constraint tolerance, develops non-finite states, requires negative physical densities without an explicit interpretation, lacks a controlled standard-cosmology limit, or exhibits ghost/gradient instability once perturbations are implemented. A claimed spectral feature is rejected if it is not stable against sampling, window, duration, solver, and resolution changes.
