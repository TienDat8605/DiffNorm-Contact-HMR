# Mathematical Review: DiffNorm-Contact HMR

**Reviewed files**

- `documents/md/notes/02_diffnorm_mathematical_foundations.md`
- `documents/md/proposals/proposal.md`

**Recommendation:** Major revision. The central Gaussian product integral and center gradient are correct, but the proposal mixes incompatible Gaussian normalizations, states incorrect rotation/normal Jacobians, and makes several claims that are stronger than either the mathematics or the supplied evidence supports.

## 1. Executive audit

| Area | Verdict | Main issue |
|---|---|---|
| Gaussian product integral | **Correct in proposal, wrong in notes** | The proposal uses the correct formula for unnormalized exponential Gaussians; the notes use a normalized-density denominator without the required determinant numerator. |
| Center repulsion gradient | **Correct** | With fixed covariances, $\nabla_{\mu_i}K_{ij}=-K_{ij}(\Sigma_i+\Sigma_j)^{-1}(\mu_i-\mu_j)$. |
| “Infinitely differentiable” | **Incorrect** | Culling, hard cutoffs, `max(0,\cdot)`, clamps, and active-pair masks make the implemented loss only piecewise smooth. |
| Collision-volume claim | **Not established** | The overlap of Gaussian kernels is not mesh self-penetration volume, and the supplied `compute_penetration_volume` is a proxy, not the analytical overlap. |
| Normal loss | **Partly correct** | The cosine objective is sound if both maps are unit length, but the stated gradients omit rasterization weights and normalization derivatives. |
| Rotation/torque derivation | **Incorrect as written** | The local Jacobian is orientation- and convention-dependent; the proposed $\partial n/\partial\theta=-[n]_\times$ and $\partial L/\partial R$ are not generally valid. |
| “Bas-relief ambiguity eliminated” | **Overclaim** | Normal cues can reduce depth rotation ambiguity but cannot mathematically eliminate it from a single view. |
| Dual gradient routing | **Mechanically correct, conceptually overstated** | `detach(v_i)` does zero the direct skeletal gradient of photometric loss, but does not create a rigorous low/high-frequency decomposition. |
| Benchmark claims | **Not scientifically supported** | The headline uses 10 frames, the “4D-Humans” initializer is a ground-truth-derived fallback, and no PA-MPJPE was computed for the initialization. |
| Collision experiments | **Synthetic, not RICH/CAPE benchmarks** | The supplied scripts create synthetic poses and do not run the named datasets. |

## 2. SMPL kinematics

### 2.1 Correct parts

Rodrigues’ formula

$$R=I+\sin\theta\,K+(1-\cos\theta)K^2$$

is correct when $K=[\hat\theta]_\times$ and $\hat\theta=\theta/\|\theta\|$. Shape blending with

$$v_{\text{shaped},i}=\bar v_i+\sum_{b=1}^{10}\beta_bS_{b,i}$$

is also the standard linear form.

### 2.2 Required corrections

1. **Pose dimension.** The notes define $\theta\in\mathbb R^{24\times3}$; the proposal later writes $\theta\in\mathbb R^{24\times3\times3}$. Axis-angle and matrix representations should not be silently equated. Use one primary definition and explicitly state conversions.

2. **Missing pose blend shapes.** The proposal includes $B_p(\theta)$ in one equation, while the explanatory notes omit it. Standard SMPL uses both shape and pose blend terms. This inconsistency must be resolved.

3. **Summation indices.** The proposal’s LBS equation writes $\mathbf V$ with outer index $k$ but uses $w_{kb}$ and $B_s(\beta)$ without an explicit per-vertex index. It should instead use $i$ for vertices and $b$ for bones.

4. **Homogeneous transforms.** The notes abbreviate the global transform as a product over ancestors but omit the local rest-offset structure and the usual $G_kG_{k,0}^{-1}$ skinning transform. This is acceptable as intuition, but not complete SMPL.

5. **Joints after translation.** The implementation adds global translation to joints. This is consistent with its own convention, but documents must distinguish root placement from camera extrinsics.

## 3. Gaussian representation and overlap

### 3.1 Two incompatible definitions

The proposal defines the unnormalized exponential

$$g_i(x)=\exp\left[-\frac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)\right].$$

For this definition,

$$K_{ij}
=\int_{\mathbb R^3}g_i(x)g_j(x)\,dx
=(2\pi)^{3/2}
\sqrt{\frac{|\Sigma_i||\Sigma_j|}{|\Sigma_i+\Sigma_j|}}
\exp\left[-\frac12d_{ij}^T(\Sigma_i+\Sigma_j)^{-1}d_{ij}\right],
$$

which matches `code/src/physics/gaussian_convolution.py`.

The notes instead state

$$K_{ij}=
\frac{\exp[-d_M^2/2]}
{(2\pi)^{3/2}|\Sigma_i+\Sigma_j|^{1/2}}.
$$

That would be the formula for integrating the product of two **normalized probability densities**, provided $\Sigma_i$ and $\Sigma_j$ are actual covariances. It is not the integral of the exponential functions defined in the same notes. The missing factor is

$$\sqrt{|\Sigma_i||\Sigma_j|}.$$

**Action:** choose one convention everywhere and define the normalization explicitly.

### 3.2 Center gradient

For fixed covariances, the proposal’s gradient is algebraically correct:

$$\nabla_{\mu_i}K_{ij}
=-K_{ij}(\Sigma_i+\Sigma_j)^{-1}(\mu_i-\mu_j).
$$

The analytical implementation matches Autograd in the supplied test. However:

- The gradient direction is exactly radial only for isotropic or otherwise proportional covariances.
- If scales and rotations are learnable, covariance gradients also exist and must not be ignored in a complete derivation.
- The phrase “infinitely differentiable” is false for the complete loss because of cutoffs, masks, clamping, and hinge terms.

### 3.3 Product overlap is not physical penetration volume

$K_{ij}$ has units determined by the chosen normalization:

- For unnormalized exponential kernels, $K_{ij}$ has units of volume.
- For normalized densities, $K_{ij}$ has units of inverse volume.

Neither quantity equals the signed or unsigned volume of triangle-mesh interpenetration. In fact:

1. Non-overlapping solid bodies can still have overlapping Gaussian tails.
2. Penetrating bodies may have a small or thresholded kernel overlap.
3. Summing pair overlaps double-counts regions shared by many Gaussians.
4. Tangential disk covariance is an approximation to occupancy, not a hard-solid ellipsoid.

Therefore, “guarantees $0.00\text{ cm}^3$ mesh self-intersection” is mathematically unsupported. The claim should be weakened to “reduces a Gaussian-overlap collision proxy.”

### 3.4 Culling conditions

The proposal states that pairs farther than $0.15$ m are negligible. Negligibility depends on covariance scale and units, not distance alone. For $\sigma=0.015$ m this is a reasonable engineering cutoff, but it should be expressed as a Mahalanobis or probability-mass threshold:

$$d_{ij}^T(\Sigma_i+\Sigma_j)^{-1}d_{ij} > \kappa^2,
\qquad \kappa\approx 6\text{--}8.
$$

Also, the bounding-sphere radius computation and segment masks make the objective graph-dependent and discontinuous across activation boundaries.

## 4. Gaussian disk geometry

The covariance parameterization

$$\Sigma_i=R_i\operatorname{diag}(s_{i,1}^2,s_{i,2}^2,s_{i,3}^2)R_i^T
$$

is correct, and taking the third column $R_ie_3$ as the disk normal is a valid orientation convention. However:

1. **A disk normal has two possible orientations.** The sign remains ambiguous unless constrained by camera visibility or an outward-oriented template.
2. **The notes’ claim that the splat “rotates with the skin” is not implemented.** In `TangentialGaussianSurface`, quaternions are independent learnable parameters initialized to identity. They are not derived from posed SMPL vertex normals or the kinematic chain.
3. **The normal supervision therefore may not exert the claimed skeletal torque.** With independent quaternions and detached or weakly connected orientations, a renderer can satisfy normals by changing splat orientation without rotating the skeleton.
4. **Camera transforms are omitted in practice.** The proposal defines $n_i=R_{\text{cam}}R_ie_3$, but the evaluation calls the router without `R_cam`, so the assumed world and camera frames coincide.
5. **The covariance used for collision remains in the SMPL/world frame**, while the rasterizer expects camera-space covariance. The code currently feeds the same covariance to both. This is valid only under the unstated identity-extrinsic assumption.

These are central technical defects, not notation nitpicks.

## 5. Rasterization and normal loss

The affine EWA approximation

$$\Sigma_{2D,i}=J_i\Sigma_iJ_i^T+\sigma_{\text{filter}}^2I
$$

is the usual local approximation. The stated Jacobian

$$
J_i=
\begin{bmatrix}
f_x/z&0&-f_xy/z^2\\
0&f_y/z&-f_yy/z^2
\end{bmatrix}
$$

is correct.

Front-to-back alpha accumulation,

$$w_i=\alpha_i\prod_{j<i}(1-\alpha_j),
\qquad
\hat N(p)=
\frac{\sum_iw_in_i}{\|\sum_iw_in_i\|+\epsilon},
$$

is also a well-defined weighted-normal estimator.

### 5.1 Loss gradient

If $\hat N$ is explicitly normalized, then

$$
\frac{\partial(\hat N\cdot N^*)}{\partial\hat N_{\text{unnorm}}}
=
\frac{1}{\|\hat N_{\text{unnorm}}\|}
\left[
N^*-(\hat N\cdot N^*)\hat N
\right].
$$

Thus the manuscript’s gradient $\partial L/\partial\hat N=-N^*$ omits the normalization derivative. It is approximately valid only when the accumulated vector is already unit length.

The complete gradient with respect to a splat normal is

$$
\frac{\partial L}{\partial n_i}
=-\sum_p
\beta_p w_i(p)
\left[
N_p^*-(\hat N_p\cdot N_p^*)\hat N_p
\right],
$$

where $\beta_p$ contains mask/uncertainty normalization and $w_i(p)$ is the alpha-compositing weight. Omitting $w_i(p)$ is generally incorrect.

### 5.2 Rotation gradient is incorrect

The proposal states

$$
\frac{\partial L}{\partial R_i}
=-\sum_pw_i(p)R_{\text{cam}}^TN_p^*e_3^T.
$$

This cannot be the derivative of $\hat N_p\cdot N_p^*$ with respect to a constrained $R_i\in SO(3)$ because it ignores the rasterization weight, normal normalization, and manifold constraint.

For a left perturbation $R_i\leftarrow\exp([\omega]_\times)R_i$, the first variation of the disk normal is

$$\delta n_i=\omega\times(R_ie_3)
=-[R_ie_3]_\times\omega.
$$

Consequently the Euclidean gradient $g_i=\partial L/\partial n_i$ produces the Lie-algebra variation

$$\delta L=g_i^T(\omega\times n_i)
=(n_i\times g_i)^T\omega.
$$

Depending on whether the perturbation is applied on the left or right and whether the tangent basis is identified with the physical angular velocity, the usable rotational update is proportional to $n_i\times g_i$ or $g_i\times n_i$. A single sign cannot be asserted globally.

The notes’ claim

$$\frac{\partial n}{\partial\theta}=-[n]_\times
$$

is not a mathematically well-defined $3\times3$ Jacobian and is not generally true for finite rotations. At $\theta=0$, for $n(\theta)=\exp([\theta]_\times)n_0$,

$$\left.\frac{\partial n}{\partial\theta}\right|_{\theta=0}
=-[n_0]_\times.
$$

For nonzero rotations, the derivative contains additional terms through the Rodrigues derivatives and depends on the local/global joint convention.

## 6. Dual-frequency gradient routing

Using

$$\mu_i^{\text{deform}}=\operatorname{stopgrad}(v_i)+\delta_i
$$

indeed gives

$$\frac{\partial L_{\text{photo}}}{\partial\theta}=0
$$

for that branch, assuming no other path connects the deformation loss to $\theta$. The supplied gradient-router test passes.

However:

1. **“Dual frequency” is only a label.** There is no Fourier projection or spectral theorem proving that normals are low-frequency and RGB is high-frequency.
2. **The barrier does not remove all pose contamination.** The photometric branch still updates offsets, scales, colors, quaternions, and opacities after a geometric step. Those changed parameters enter the next geometric loss and can indirectly affect the next pose gradient.
3. **The geometric loss updates deformation parameters too.** `loss_geom.backward()` propagates into offsets, scales, quaternions, colors, and opacities. Thus the claimed exclusive parameter ownership is false.
4. **Quaternion gradients require manifold care.** Treating four quaternion components as unconstrained Euclidean parameters and renormalizing only in the forward map is common, but a strict derivation should include the normalization derivative or use a Riemannian optimizer.

A safer claim is: “Photometric residuals have no direct autograd path to SMPL pose parameters.” It is not a proof of frequency separation.

## 7. Hypothesis audit and benchmark interpretation

### 7.1 PA-MPJPE comparison

The proposal claims normal refinement improves PA-MPJPE from 42.30 mm to 36.39 mm. The supplied result file contains only:

- initialization MPJPE: 31.26 mm,
- refined MPJPE: 49.93 mm,
- refined PA-MPJPE: 36.39 mm.

No initialization PA-MPJPE is recorded. Therefore the stated 14% PA-MPJPE reduction is not reproducible from the artifact. Moreover, the reported MPJPE gets substantially worse, so “normal supervision resolves pose ambiguity” is not demonstrated.

### 7.2 Ten frames are not the official 3DPW protocol

The result is based on 10 evaluated frames. Calling this the official 3DPW benchmark and placing it in a SOTA comparison is unsupported. The comparison table also mixes literature numbers, a 100-frame neutral baseline, and a 10-frame refinement result.

### 7.3 The initializer leaks ground truth

In the evaluation code, the fallback initializer receives

$$\theta_{\text{coarse}}=\theta_{\text{GT}}+\epsilon,
\qquad \epsilon\sim\mathcal N(0,0.04^2).
$$

The fallback then returns that input unchanged. Thus the reported “4D-Humans initialization” is ground-truth-derived synthetic noise, not true 4D-Humans inference. This invalidates the claimed 4D-Humans baseline.

### 7.4 PVE comparison is invalid

Comparing optimized clothed Gaussian offsets against bare SMPL vertices yields a large PVE (276.82 mm). The note says the offsets capture clothing silhouette, but this is not evidence that the geometry is accurate. It may indicate offset drift. A clothed benchmark with appropriate ground truth is required.

### 7.5 Collision result is synthetic

The supplied RICH script does not load RICH. It creates a synthetic crossed-arm pose and uses a distance-based sphere proxy:

$$V_{\text{proxy}}=\sum(0.06-d)^3.
$$

This proxy has inconsistent physical units unless multiplied by a sphere-intersection coefficient, and it is not the closed-form Gaussian overlap derived above. The claim “RICH collision volume 0.00 cm³” is not supported.

### 7.6 CAPE result is synthetic

The CAPE script does not load CAPE. It generates two noise levels and assigns them to “coupled” versus “detached” methods. This cannot validate the effect of gradient routing.

### 7.7 Speed and memory claims

The claims of $>100$ FPS, 2.8 GB peak VRAM, and 4.2 ms collision computation are not supported by the supplied artifacts. Timing must include preprocessing, sorting, rasterization, collision, and both optimizer updates, with hardware, input size, and batching specified.

## 8. Incorrect or misleading language to revise

1. Replace “guarantees zero self-collisions” with “penalizes a Gaussian-overlap proxy.”
2. Replace “exact volumetric self-penetration” with “exact integral of two Gaussian kernels.”
3. Replace “eliminates Bas-relief ambiguity” with “adds normal-field constraints that can reduce depth ambiguity.”
4. Replace “strict spectral decomposition” with “explicit autograd detachment.”
5. Replace “analytical torque” with “rotation-parameter gradient,” unless a physical moment arm and force law are actually defined.
6. Replace “infinitely differentiable” with “smooth within active covariance/collision regions.”
7. Remove or qualify “official 3DPW protocol,” “RICH,” and “CAPE” until the corresponding experiments are actually run.

## 9. Highest-priority repairs

1. Use one Gaussian normalization and repeat the overlap formula consistently.
2. Tie each splat’s normal axis to the posed SMPL vertex normal if normal supervision is meant to rotate the skeleton, or state that it is a free appearance orientation.
3. Apply camera extrinsics consistently to centers, covariances, normals, and collision geometry.
4. Derive the complete normal gradient through alpha weights and normalization.
5. Report both initialization and refined MPJPE and PA-MPJPE over the complete official test protocol.
6. Remove ground-truth leakage from the coarse initializer before any HMR evaluation.
7. Use a real triangle-intersection penetration metric for the final collision claim; retain Gaussian overlap only as a differentiable surrogate.
8. Evaluate collision on RICH and clothing bias on CAPE, or label current results synthetic sanity checks.
9. Add confidence intervals and per-sequence/per-frame breakdowns; means over 5–10 frames are not benchmark estimates.
10. Report an ablation without collision, without normal loss, and without detachment to support each causal hypothesis.

## 10. Validation performed

I ran the four targeted tests:

```bash
PYTHONPATH=code pytest -q \
  code/tests/test_analytical_collision.py \
  code/tests/test_collision_repulsion.py \
  code/tests/test_gradient_router.py
```

Result: **4 passed**. These tests verify the local overlap value, local center gradients, and direct gradient detachment. They do not validate the global HMR, normal-geometry coupling, collision-volume, benchmark, speed, or memory claims.
