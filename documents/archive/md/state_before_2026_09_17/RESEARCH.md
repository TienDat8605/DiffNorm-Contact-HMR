# DiffNorm-Contact HMR: Research State

## Goal
Improve Human Mesh Recovery (HMR) pose accuracy, reduce out-of-plane rotation ambiguity, and eliminate unnatural self-penetrations by integrating differentiable surface normal supervision (DSINE) with continuous Gaussian contact and collision regularizers on 3DPW.

## Current Baseline
- **Model**: HMR 2.0 / 4D-Humans initialized baseline
- **Dataset**: 3DPW (`downtown_arguing_00`, `courtyard_range_of_motions_00`, etc.)
- **Primary Metrics**: MPJPE (mm), PA-MPJPE (mm), PVE (mm), Worsening Rate (%)
- **Baseline Result**:
  - 100-frame neutral evaluation: MPJPE = 196.6 mm, PA-MPJPE = 214.2 mm, PVE = 632.5 mm
  - 24-frame 6-way pilot (Condition A - Frozen): MPJPE = 336.0 mm, PA-MPJPE = 215.4 mm

## Current Hypotheses
- **H001 (Differentiable Normal Cosine Alignment)**: Supervising predicted SMPL surface normals against DSINE pseudo-ground truth normal fields reduces depth/rotation ambiguity compared to 2D keypoint reprojection alone.
- **H002 (Gaussian Collision Surrogate)**: Modeling SMPL body segments with anisotropic 3D Gaussians provides a piecewise-smooth repulsive potential that penalizes mesh self-penetration without gradient explosion or vertex-triangle culling singularities.
- **H003 (Stage-Gated Normal Routing - Condition F)**: Gating the normal loss until root rotation and 2D keypoint alignment converge prevents gradient corruption of initial kinematic estimates and reduces the worsening rate.

## Evidence So Far
- Pilot 6-way evaluation (`results/pilot_6way_experiment_results.json`):
  - Condition D (unconstrained 3DGS) and E (nuisance-controlled 3DGS) achieve lower worsening rates (33.3%) than naive mesh optimization (50.0%).
  - Condition F (stage-gated) achieves lower overall MPJPE (349.2 mm) than naive mesh optimization (367.6 mm).
- Mathematical audit (`documents/md/reviews/diffnorm_mathematical_review.md`):
  - Verified fixed-covariance center gradient formulation.
  - Rectified Gaussian integral normalization factor and established piecewise-smooth surrogate bounds.

## Main Uncertainty
- Under full multi-epoch training on 3DPW (rather than short test-time optimization of 15 iterations), does dual-stage normal routing reliably lower PA-MPJPE across dynamic motion sequences without destabilizing SMPL joint limits?

## Next Candidate Experiment
- Execute multi-epoch training on Tesla T4 via Google Colab CLI orchestration (`code/scripts/train_colab_cli.sh`), evaluating checkpoints on the full 3DPW test split with `code/scripts/eval_3dpw.py`.
