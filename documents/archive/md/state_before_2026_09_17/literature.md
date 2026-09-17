# Literature Notes: DiffNorm-Contact HMR

## DSINE: Deep Surface Normal Estimation
- **Reference**: Bae et al., "Deep Surface Normal Estimation with Learned Ray Representation", NeurIPS.
- **Key Insight**: Employs pixel ray direction encoding and von Mises-Fisher distribution modeling to output per-pixel unit normal vectors with sharp boundary preservation.
- **Role in Project**: Provides pseudo-ground-truth normal fields for single-view images in 3DPW, which guide differentiable SMPL mesh rasterization.

## 4D-Humans / HMR 2.0
- **Reference**: Goel et al., "Humans in 4D: Reconstructing and Tracking Humans with Transformers", CVPR 2023.
- **Key Insight**: ViT-based backbone trained on massive in-the-wild datasets yielding robust initial SMPL pose and shape parameters $\theta \in \mathbb{R}^{24 \times 3}, \beta \in \mathbb{R}^{10}$.
- **Role in Project**: Serves as the initialization baseline. Optimization adjusts latent pose parameters to match DSINE normal maps while maintaining plausible joint limits.

## Continuous Collision Detection & Contact Modeling
- **Reference**: Müller et al. / SMPL-X collision penalties.
- **Key Insight**: Exact triangle mesh intersection tests are piecewise non-differentiable and computationally expensive on GPU. Proxy collision volumes (such as anisotropic 3D Gaussians or capsule sweeps) provide differentiable repulsive gradients.
- **Role in Project**: Analytical Gaussian product integrals compute overlap surrogates that penalize self-intersection without discrete culling discontinuities.
