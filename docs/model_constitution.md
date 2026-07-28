# SCPC model constitution

This document fixes terminology that equations and code must obey.

## Spacetime

The baseline theory uses a continuous four-dimensional Lorentzian manifold \((\mathcal M,g_{\mu\nu})\). Numerical time steps are discretization devices, not physical temporal atoms.

## Stratification field

The scalar \(\phi(x^\mu)\) is the **stratification field**. A periodic or multi-minimum potential may partition field space into distinguishable phase strata. The field is not automatically entropy, physical time, or information.

## Phase stratum

A phase stratum is a connected region, basin, or vacuum sector in field or state space. A label \(n\) identifies a stratum and does not by itself define a spacetime coordinate.

## Cosmological cycle

A candidate cycle contains an expansion phase, a turnaround, a contraction phase, and a bounce. A bounce at \(t_b\) requires

\[
H(t_b)=0,\qquad \dot H(t_b)>0,\qquad a(t_b)>0,
\]

while a turnaround requires \(H=0\) and \(\dot H<0\). A stable cycle additionally requires recurrence under a Poincaré map and acceptable Floquet multipliers.

## Curvatures

- \(R[g]\): spacetime Ricci scalar.
- \(\mathcal R_{\rm th}\): thermodynamic state-space curvature.
- \(\mathcal R_{\rm field}\): field-space curvature.
- finite differences: numerical operators, never physical curvature without derivation.

## Vorticity and topology

The project does not use “vortex” as a model-wide claim. A future vortex sector must define nonzero vorticity, phase winding, or another invariant topological charge.

## Falsification

A concrete SCPC model fails if it lacks a controlled GR limit, violates a constraint without convergence, requires a ghost or gradient instability, produces only discretization-dependent signatures, or is excluded throughout its stable parameter domain by declared likelihoods.
