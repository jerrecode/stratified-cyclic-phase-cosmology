# Contributing

Scientific changes must state whether they alter ontology, action, equations of motion, numerical method, observable mapping, likelihood, or presentation only.

Every physical model change should include:

1. assumptions and degrees of freedom;
2. units and parameter dimensions;
3. variational or effective-equation derivation;
4. conservation and constraint relations;
5. GR or benchmark limiting behavior;
6. stability conditions;
7. analytical or manufactured-solution tests;
8. convergence evidence;
9. falsification conditions;
10. regenerated paper assets where applicable.

Do not infer empirical support from visual resemblance. Do not label a finite difference, thermodynamic Hessian, or field-space curvature as spacetime curvature without a derivation.

Generated outputs belong under `results/` and must be reproducible from a committed configuration. Large external data must not be committed; record release identifiers, checksums, and access instructions in the manifest or a run provenance file.
