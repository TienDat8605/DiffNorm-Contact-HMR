# DiffNorm-Contact HMR: Current Research State

Updated 17 September 2026 following source-level and artifact audit. Authoritative details: [proposal](documents/md/current/research_proposal_and_roadmap_2026_09_17.md), [evidence audit](documents/md/current/progress_and_evidence_audit_2026_09_17.md).

## Goal
Determine whether accounting for Gaussian nuisance flexibility can make **single-image, fixed-predicted-shape, test-time pose refinement** more accurate and less harmful than matched conventional refinement. Normals are an input cue; collision is deferred, not the main novelty.

## Current Baseline
No valid HMR2/SMPL benchmark baseline is established. The historical six-way JSON records A=336.04 mm MPJPE / 215.44 mm PA-MPJPE for 24 examples across 20 sequence names. The inspected runner selects synthetic geometry and permits a fixed-pose initializer fallback; the result is not an authenticated HMR2 baseline.

## Current Hypothesis
H004: nuisance-aware sensitivity can guide pose updates better than ordinary joint fitting, static routing, and equally conservative generic damping at matched compute and update coverage. Candidate contribution only; the current controller is a heuristic, not an implemented Schur-complement solver.

## Evidence So Far
- Implemented: mesh/Gaussian renderers, six-condition runner, heuristic sensitivity filter, two-stage Gaussian capacity scheduler, partial numerical repairs.
- CPU tests: 33 passed, 2 skipped, 12 warnings (4.40 s on 17 September 2026).
- Artifact arithmetic verified: four-way file has 2 records / 1 sequence; five-way 24 / 19; six-way 24 / 20.
- Six-way E: PA-MPJPE -0.503 mm versus A, but MPJPE +17.561 mm; E and D both worsen 8/24 examples under the saved >0.1 mm PA threshold.
- F: PA-MPJPE +1.582 mm and MPJPE +13.193 mm versus A. No validated gain; all scientific pilot verdicts remain **Inconclusive** because of protocol/provenance defects.

## Main Uncertainty
Does nuisance-aware control help after authentic initialization, official geometry, camera/subject alignment, consistent shape, matched objectives and correct optimizer-step control are enforced?

## Next Candidate Experiment
V001: fail-closed evaluation and geometry/metric provenance gate (CPU tests followed by a separately authorized tiny visual smoke test). Then M001 synthetic mechanism/derivative checks; only then P001 held-out paired real-image pilot. **Do not proceed directly to multi-epoch training or promote E/F as validated defaults.**

## Research authority
Codex lead owns decisions and final verification; Gemini Antigravity handles bounded assistant tasks. See AGENTS.md. Previous state preserved in documents/archive/md/state_before_2026_09_17/RESEARCH.md.
