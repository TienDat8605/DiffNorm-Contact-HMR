# Research Hypotheses: DiffNorm-Contact HMR

## H001: Surface Normal Alignment Resolves Out-of-Plane Rotation
- **Hypothesis**: Differentiable cosine distance supervision between predicted SMPL surface normals and high-fidelity estimated normal maps (DSINE) provides direct 3D orientation gradients that reduce the Bas-relief depth/rotation ambiguity inherent in 2D joint reprojection.
- **Baseline**: 2D joint reprojection + silhouette loss alone.
- **Independent Variable**: Normal loss weight $\lambda_{\text{norm}}$ and cosine vs angle metric formulation.
- **Metrics**: PA-MPJPE (mm), MPJPE (mm), surface normal error (degrees).
- **Falsification Condition**: Normal supervision fails to reduce 3D joint error or increases out-of-plane rotation error compared to the baseline under identical camera calibrations.
- **Status**: Partially supported in pilot tests; requires full epoch training verification.

## H002: Gaussian Overlap Collision Surrogate Penalizes Self-Penetration
- **Hypothesis**: Replacing non-differentiable discrete triangle mesh intersection checks with an analytical Gaussian product integral over proxy ellipsoids yields smooth repulsive forces that suppress limb interpenetration during complex poses.
- **Baseline**: No collision penalty, or naive vertex-to-surface distance penalty.
- **Independent Variable**: Gaussian covariance assignment and repulsive scale $\lambda_{\text{collision}}$.
- **Metrics**: Mesh penetration volume ($\text{cm}^3$), interpenetration frequency, MPJPE.
- **Falsification Condition**: Repulsive forces distort plausible SMPL joint configurations without significantly decreasing physical intersection volume.
- **Status**: Formulated and audited; piecewise smooth in practice due to active-pair thresholding.

## H003: Stage-Gated Normal Gradient Routing Prevents Initial Drift
- **Hypothesis**: Normal loss gradients initially pull bone rotations prematurely if global root orientation and translation have not yet converged. Gating normal gradients until keypoint reprojection error falls below a threshold $\tau$ prevents early divergence.
- **Baseline**: Continuous un-gated normal loss throughout all optimization steps.
- **Independent Variable**: Gating schedule (iteration cutoff or residual threshold).
- **Metrics**: Per-frame worsening rate (percentage of frames where optimization degrades error), final PA-MPJPE.
- **Falsification Condition**: Gated optimization performs worse than or identically to un-gated optimization across diverse 3DPW sequences.
- **Status**: Pilot 6-way experiment indicates 33.3% worsening rate for nuisance-controlled conditions vs 50.0% for naive mesh.
