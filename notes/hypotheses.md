# Hypothesis Registry

Updated 17 September 2026. All thresholds below are prospective design choices, not measured outcomes. Complete controls and gates are in the [current proposal](../documents/md/current/research_proposal_and_roadmap_2026_09_17.md). Existing misconfigured pilots do not support or contradict the scientific hypotheses.

## H001 — Normal supervision (retained, unvalidated)

- Mechanism: normals may constrain pose directions poorly constrained by keypoints.
- Control / variable: identical keypoint+mask+prior fitting, normal weight zero versus nonzero.
- Metrics: paired MPJPE, PA-MPJPE, PVE and worsening rate; angular error only with genuine normal ground truth.
- Prospective gate: >=2 mm mean PA gain with sequence-clustered 95% interval excluding zero; no >1 mm mean MPJPE regression.
- Falsification: a valid, adequately precise experiment rules out the 2 mm target; otherwise inconclusive.
- Confounders: clothing, predictor errors, camera mismatch, visibility, initialization.
- Status: unvalidated; not a claim that monocular ambiguity is resolved.

## H002 — Gaussian overlap collision (deferred)

- Mechanism: proxy overlap penalty may reduce mesh intersections.
- Control / variable: matched optimizer with penalty off/on and covariance assignment fixed.
- Metrics: independently measured mesh penetration volume and contact preservation, plus pose error. Gaussian overlap is not physical volume.
- Prospective gate: >=10% relative reduction in independently measured penetration without >1 mm mean MPJPE regression on a predeclared contact set.
- Falsification: valid comparison excludes this reduction or damages legitimate contacts.
- Confounders: proxy shrinkage, sample density, containment, adjacency and mesh-volume estimator validity.
- Status: not tested by six-way pilot; no collision-free guarantee.

## H003 — Capacity scheduling (corrected meaning, exploratory)

- Actual mechanism: freeze Gaussian deformation first, then release bounded offsets. This is NOT normal-loss gating on keypoint convergence; normals remain active in stage 1.
- Control / variable: same controller, bounds, learning-rate schedule and objectives; vary only release timing. Separate LR-only and bounds-only controls.
- Metrics: PA-MPJPE, MPJPE, PVE, worsening rate, runtime.
- Prospective gate: >=2 mm PA gain and >=5 percentage-point lower worsening than simultaneous release, with cluster intervals excluding zero.
- Falsification: valid adequately precise comparison excludes these targets; misconfigured F results are inconclusive.
- Confounders: F currently changes LR, bounds and timing together; unequal effective compute.

## H004 — Nuisance-aware refinement (primary)

- Mechanism: estimate which pose evidence remains after Gaussian nuisance adaptation; limit unreliable pose updates.
- Control / variable: frozen initializer, mesh and fixed-Gaussian refinement, free Gaussian fitting, static detachment, generic damping/prior; vary only sensitivity-dependent control in the primary comparison.
- Metrics: paired PA-MPJPE, MPJPE, PVE, >0.1 mm PA worsening rate, tail error, update coverage and runtime.
- Prospective gate: >=2 mm mean PA gain over A and >=5 percentage-point lower worsening than tuned free-Gaussian control; cluster intervals exclude zero; mean MPJPE must not worsen by >1 mm. Beat matched generic damping to attribute benefit to the diagnostic.
- Falsification: valid controlled evidence rules out meaningful incremental benefit or shows generic damping fully explains it. Wide intervals mean inconclusive, not failure.
- Confounders: oracle shape, wrong camera, synthetic model, fallback initializer, Adam scale cancellation, unequal losses, test-set tuning, noisy normals.
- Status: proposed; current implementation and historical pilots do not establish the mechanism.
