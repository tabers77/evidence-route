"""Research outputs: tables, figures and the static report.

Evallab produces episode-level and aggregate evaluation reports. This package
adds the research artifacts specific to EvidenceRoute's claims:

- workflow comparison table with paired uncertainty intervals
- routing-policy comparison across all declared reward profiles
- cost-quality Pareto plots
- calibration plots and risk-coverage curves
- OPE bias and variance plots against known policy values
- failure-category distribution
- statistical comparison appendix

Static reports come first; a dashboard is a stretch goal. A UI-heavy project
that never finishes its experiments has failed at the thing it was for
(spec section 27).

Every published number must be traceable to a run manifest. The reporting layer
can therefore refuse to emit a claim backed by a dirty worktree — see
``evidence_route.evaluation.reproducibility.RunManifest.is_publishable``.
"""
