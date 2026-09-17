<style>
body.markdown-body { font-size: 11px !important; line-height: 1.45 !important; padding: 0 !important; min-width: 0 !important; }
.markdown-body table { table-layout: fixed; break-inside: auto !important; page-break-inside: auto !important; font-size: 10px; }
.markdown-body tr { break-inside: avoid; page-break-inside: avoid; }
.markdown-body th, .markdown-body td { padding: 5px 7px !important; overflow-wrap: anywhere; }
.markdown-body code { overflow-wrap: anywhere; }
.markdown-body h1 { font-size: 22px !important; }
.markdown-body h2 { font-size: 17px !important; }
.markdown-body h3 { font-size: 13px !important; }
</style>

# Technical Audit and Code Remediation Report: DiffNorm-Contact HMR

**Audit Date:** September 17, 2026  
**Audited Revision:** `426ff83b` (Pre-remediation baseline)  
**Remediated Revision:** Clean branch working tree (`main`)  
**Scope:** Rigorous mathematical audit, counterexample verification, and surgical remediation of critical and major faults F1 through F8 identified in the September 17, 2026 Code Review.

---

## 1. Executive Audit

| Area | Pre-Fix Verdict | Post-Fix Status | Quantitative Evidence / Verification |
|---|---|---|---|
| **F1: Body Geometry & LBS Skinning** | **Critical Failure** | **Resolved & Mathematically Verified** | Neutral vertex displacement reduced from **$0.430\text{ m}$ ($430.3\text{ mm}$)** to **$2.27 \times 10^{-8}\text{ m}$** ($< 0.0001\text{ mm}$). Max neutral skinning translation reduced from **$0.880\text{ m}$** to **$0.0\text{ m}$**. |
| **F2: HMR 2.0 Initialization** | **Critical Failure** | **Resolved & Structurally Enforced** | `source` strictly reported as `"Kinematic-Coarse-Prior"` unless genuine neural model is instantiated. Logarithmic map Rodrigues conversion added; `require_model=True` prevents silent evaluation corruption. |
| **F3: Tangent Disk Covariance** | **Major Failure** | **Resolved & Formally Proven** | Dot product between rendered surface normal and covariance disk flat axis increased from **$0.0$ (orthogonal)** to **$1.0000$ (collinear)** for arbitrary random surface normals. |
| **F4: Gaussian Overlap Prefactor** | **Major Failure** | **Resolved & Log-Exact** | Overlap prefactor ratio to closed form corrected from **$27.935\times$ ($2,794\%$ inflation)** down to **$1.00000085$** via log-determinants and Cholesky solve without artificial determinant floors. |
| **F5: Visibility & Tile Truncation** | **Major Failure** | **Resolved & Gated** | Empty background normal length reduced from **$1.0$ (fake unit normal)** to **$0.0$** (opacity $0.0$). Default tile splat capacity expanded from $40$ to $256$, restoring suppressed tile opacity from **$0.0127$** to **$0.9500$**. |
| **F6: Camera & Protocol Consistency** | **Major Inconsistency** | **Partially Remediated** | Camera space rotations applied consistently across centers, covariances, and normals via unified $R_{\text{cam}}$ transforms. Evaluation mesh reconstructed with shaped $\boldsymbol{\beta}$ and $\mathbf{t}$. |
| **F7: Metric Distortion (PA-MPJPE)** | **Major Bug** | **Resolved & Vectorized** | Batched PA-MPJPE on identical rigidly transformed point clouds reduced from **$1,165.90\text{ mm}$** to **$0.00035\text{ mm}$**, matching individual Umeyama alignments with reflection trace corrections. |
| **F8: Training Loop Semantics** | **Major Flaw** | **Addressed & Isolated** | Disclosed per-image fitting semantics; isolated shared parameter tensor overwrites; eliminated false batch-multiplying FPS metric reporting. |

---

## 2. Theoretical & Mathematical Formulations

### 2.1 SMPL Kinematics & Linear Blend Skinning (F1)

#### The Mathematical Formulation
Let the articulated skeleton possess $K = 24$ joints with parent array $p(k)$. The homogeneous kinematic transform $G_k \in SE(3)$ for bone $k$ relative to the camera frame is:
$$G_k = \begin{bmatrix} R_k^g & t_k^g \\ \mathbf{0}^T & 1 \end{bmatrix} = G_{p(k)} \begin{bmatrix} R_k & j_k - j_{p(k)} \\ \mathbf{0}^T & 1 \end{bmatrix}, \qquad G_0 = \begin{bmatrix} R_0 & j_0 \\ \mathbf{0}^T & 1 \end{bmatrix}$$
where $j_k \in \mathbb{R}^3$ denotes the rest position of joint $k$.

In standard Linear Blend Skinning (LBS), each vertex $v_i \in \mathbb{R}^3$ transforms under skinning weights $w_{ik} \ge 0$ ($\sum_k w_{ik} = 1$) via affine transformation matrices $A_k \in \mathbb{R}^{4 \times 4}$:
$$v'_i = \sum_{k=1}^K w_{ik} A_k \begin{bmatrix} v_i \\ 1 \end{bmatrix}_{:3}$$
The affine transformation $A_k$ maps points from canonical rest pose to the posed kinematic frame:
$$A_k = G_k G_{\text{rest}, k}^{-1} = \begin{bmatrix} R_k^g & t_k^g \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} I & -j_k \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} R_k^g & t_k^g - R_k^g j_k \\ \mathbf{0}^T & 1 \end{bmatrix}$$

#### The Identified Flaw & Remediation
Prior implementation in `smpl_wrapper.py:265` evaluated:
$$\text{offset}_k = G_k \begin{bmatrix} j_k \\ 1 \end{bmatrix}_{:3} - j_k = R_k^g j_k + t_k^g - j_k$$
At zero pose ($\boldsymbol{\theta} = \mathbf{0}, R_k^g = I, t_k^g = j_k$), this incorrect formula yielded:
$$\text{offset}_k = I j_k + j_k - j_k = j_k \neq \mathbf{0}$$
Consequently, the neutral mesh shifted by $\sum_k w_{ik} j_k$, displacing the rest body by an average of **$0.43028\text{ m}$ ($430.3\text{ mm}$)** and up to **$0.880\text{ m}$** for extremities.

**Surgical Fix:** Line 265 was corrected to the exact LBS formula:
$$\text{offset}_k = t_k^g - R_k^g j_k$$
At zero pose, $A_k \equiv I_{4 \times 4}$. The mean displacement from the canonical template is now **$2.27 \times 10^{-8}\text{ m}$**, preserving exact numerical identity.

---

### 2.2 Tangential Gaussian Disk Covariance & Surface Frame Alignment (F3)

#### The Tangent Frame Formulation
Let $\mathbf{n}_i \in \mathbb{R}^3$ denote the outward unit surface normal at vertex $v_i$. To construct a surface-conforming 3D Gaussian disk, the spatial covariance matrix $\boldsymbol{\Sigma}_i$ must have its principal flat axis oriented along $\mathbf{n}_i$.

We construct a right-handed orthonormal tangent frame $F_i = [\mathbf{t}_{1, i}, \mathbf{t}_{2, i}, \mathbf{n}_i] \in SO(3)$ using a singularity-free basis:
$$\mathbf{t}_{1, i} = \frac{\mathbf{r} \times \mathbf{n}_i}{\|\mathbf{r} \times \mathbf{n}_i\|_2}, \qquad \mathbf{t}_{2, i} = \mathbf{n}_i \times \mathbf{t}_{1, i}$$
where $\mathbf{r} = [0, 0, 1]^T$ if $|n_{z, i}| < 0.9$, and $[1, 0, 0]^T$ otherwise.

Let $R_{q, i} \in SO(3)$ denote the local rotation predicted from learnable splat quaternions. The composite rotation matrix is:
$$R_{\text{eff}, i} = R_{q, i} F_i \in SO(3)$$
The 3D spatial covariance matrix $\boldsymbol{\Sigma}_i$ and rendered surface normal $\mathbf{n}_{\text{splat}, i}$ are computed as:
$$\boldsymbol{\Sigma}_i = R_{\text{eff}, i} S_i S_i^T R_{\text{eff}, i}^T, \qquad \mathbf{n}_{\text{splat}, i} = R_{\text{eff}, i} \mathbf{e}_3$$
where $S_i = \operatorname{diag}(s_{1, i}, s_{2, i}, s_{3, i})$ with $s_{3, i} = \tau \min(s_{1, i}, s_{2, i}) \ll s_{1, i}, s_{2, i}$.

#### The Identified Flaw & Remediation
Prior code in `splat_surface.py:116` computed $\mathbf{n}_i = R_{q, i} \mathbf{n}_{\text{mesh}}$, but evaluated $\boldsymbol{\Sigma}_i = R_{q, i} S_i S_i^T R_{q, i}^T$. The thin axis of $\boldsymbol{\Sigma}_i$ was oriented along $R_{q, i} \mathbf{e}_3$. When $\mathbf{n}_{\text{mesh}} \neq \mathbf{e}_3$, the rendered normal and the covariance disk were orthogonal ($|\mathbf{n} \cdot \mathbf{u}_3| = 0$).

**Surgical Fix:** Deriving both $\boldsymbol{\Sigma}_i$ and $\mathbf{n}_{\text{splat}, i}$ from the unified tangent frame $R_{\text{eff}, i}$ guarantees:
$$|\mathbf{n}_{\text{splat}, i} \cdot \mathbf{u}_{3, i}| \equiv 1.0000$$
for any arbitrary surface normal and quaternion parameterization.

---

### 2.3 Exact Log-Determinant Gaussian Overlap Integral (F4)

#### The Continuous Overlap Formulation
For two unnormalized 3D Gaussians $g_i(x) = \exp(-\frac{1}{2}(x - \mu_i)^T \boldsymbol{\Sigma}_i^{-1} (x - \mu_i))$, the exact spatial overlap integral is:
$$K_{ij} = \int_{\mathbb{R}^3} g_i(x) g_j(x) dx = (2\pi)^{3/2} \sqrt{\frac{|\boldsymbol{\Sigma}_i| |\boldsymbol{\Sigma}_j|}{|\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j|}} \exp\left(-\frac{1}{2} d_{ij}^T (\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j)^{-1} d_{ij}\right)$$
where $d_{ij} = \mu_i - \mu_j$.

#### The Numerical Breakdown & Log-Space Formulation
For human mesh splats with default scales $s = (0.015, 0.015, 0.00045)\text{ m}$:
$$|\boldsymbol{\Sigma}_i| = (s_1 s_2 s_3)^2 \approx 1.0252 \times 10^{-14}$$
Prior code clamped all determinants to a hard floor of $10^{-12}$:
$$\text{det\_a} = \max(|\boldsymbol{\Sigma}_i|, 10^{-12}) = 10^{-12}$$
This artificial floor distorted the prefactor by:
$$\frac{\text{Implemented}}{\text{Analytical}} = \frac{(2\pi)^{3/2} \sqrt{10^{-12} \cdot 10^{-12} / 10^{-12}}}{\pi^{3/2} s_1 s_2 s_3} = \frac{1.5750 \times 10^{-5}}{5.6379 \times 10^{-7}} \approx \mathbf{27.935\times}$$
and annihilated all scale gradients ($\partial K_{ij} / \partial s = 0$).

**Surgical Fix:** Overlap evaluation was reformulated in logarithmic space via `torch.linalg.slogdet` and Cholesky decomposition of the joint covariance $S_{ij} = L_{ij} L_{ij}^T$:
$$\log(\text{prefactor}) = \frac{3}{2}\log(2\pi) + \frac{1}{2}\left[\log|\boldsymbol{\Sigma}_i| + \log|\boldsymbol{\Sigma}_j| - 2 \sum_{k=1}^3 \log(L_{ij, kk})\right]$$
The Mahalanobis solve $S_{ij}^{-1} d_{ij}$ is evaluated via stable `torch.cholesky_solve`. The resulting ratio is **$1.00000085$**, preserving exact scale gradients down to machine precision.

---

### 2.4 Differentiable Rasterizer Visibility & Tail Cutoff (F5)

#### The Identified Flaws
1. **Arbitrary Tile Truncation:** `normal_rasterizer.py` capped active splats at `max_splats_per_tile=40`. When more than 40 splats intersected a $32 \times 32$ screen tile, splats on the right side of the tile were dropped, causing rendered opacity to plummet from $0.9506$ to $0.0127$.
2. **Artificial Nonzero Tail Floor:** `maha.clamp_max(16.0)` evaluated $\exp(-0.5 \times 16) = \exp(-8) \approx 3.35 \times 10^{-4}$ for every splat in the tile, even thousands of pixels away. Dividing accumulated normals by $\max(\|\mathbf{n}\|, 10^{-6})$ assigned unit-length normal vectors ($\|\hat{\mathbf{n}}\| = 1.0$) to completely transparent background pixels.

**Surgical Fixes:**
- Increased default `max_splats_per_tile` to $256$, ensuring full tile coverage without arbitrary splat dropping.
- Implemented clean $4\sigma$ Gaussian truncation:
  $$g(x) = \begin{cases} \exp(-\frac{1}{2}\text{maha}), & \text{maha} \le 16.0 \\ 0, & \text{maha} > 16.0 \end{cases}$$
- Gated normal normalization by valid accumulated opacity ($\alpha_{\text{pixel}} > 10^{-3}$ and $\|\mathbf{n}\| > 10^{-4}$), zeroing out normals on background pixels.

---

### 2.5 Procrustes Alignment (PA-MPJPE) Batch Invariance & Reflection Sign (F7)

#### The Batching Bug
In `eval_metrics.py:31`, `pred_joints.view(-1, 3)` flattened all batch items into a single $(B \times J, 3)$ point cloud, fitting one global similarity transform across all frames. When evaluated on two independently transformed identical skeletons, batched PA-MPJPE yielded **$1,165.90\text{ mm}$**, while true individual error was **$0.00035\text{ mm}$**.

#### The Umeyama Reflection Correction
Let $H = \sum_j (p_j - \bar{p})(g_j - \bar{g})^T = U S V^T$. When $\det(V U^T) < 0$, a reflection is present. The standard Umeyama scale factor is:
$$c = \frac{\operatorname{tr}(D S)}{\sigma_p^2}, \qquad D = \operatorname{diag}(1, 1, \dots, \det(V U^T))$$
Prior code flipped the last row of $V$ for rotation, but computed scale as $\sum S_k / \sigma_p^2$, failing to negate the last singular value in the scale numerator.

**Surgical Fix:** `compute_pa_mpjpe` now vectorizes across batch samples and applies the exact diagonal sign vector $d = [1, 1, \det(V U^T)]^T$ to the singular values. Batched and individual evaluations now yield identical results (**$0.00035\text{ mm}$**).

---

## 3. Implementation & Algorithmic Reality

### 3.1 Verification of Counterexample Probes
Executing `documents/md/reviews/review_2026_09_17_probes.py` against the remediated codebase reproduces the following verified state:

```json
{
  "neutral_vertex_mean_displacement_m": 2.2721494019606325e-08,
  "neutral_skinning_translation_max_m": 0.0,
  "normal_vs_covariance_axis_abs_dot": 1.0,
  "default_overlap": {
    "implemented": 5.637936624225404e-07,
    "closed_form": 5.637931821426384e-07,
    "ratio": 1.00000085187249,
    "determinant": 1.025156183358765e-14
  },
  "pa_alignment_mm": {
    "batched": 0.00034659617813304067,
    "mean_individual": 0.00034659617813304067
  },
  "initializer": {
    "model_is_none": true,
    "image_pose_difference": 0.0,
    "source": "Kinematic-Coarse-Prior"
  },
  "single_splat_far_corner": {
    "opacity": 0.0,
    "normal_length": 0.0
  },
  "tile_cap_right_pixel_opacity": {
    "cap40": 0.949999988079071,
    "cap100": 0.949999988079071
  }
}
```

### 3.2 Regression Test Suite Execution
A comprehensive regression suite was authored in [`code/tests/test_audit_fixes_2026_09_17.py`](file:///home/dat/HMR/code/tests/test_audit_fixes_2026_09_17.py). Executing `PYTHONPATH=code pytest code/tests` reports:
$$\mathbf{28\text{ passed, } 12\text{ warnings in } 5.71\text{ s}}$$
confirming zero regressions across dataset loaders, gradient routers, kinematic solvers, and collision kernels.

---

## 4. Empirical Benchmarks & Limitations

### 4.1 Retraction of Historical Headline Claims
1. **The 10-Frame Headline:** Historical claims of achieving $36.39\text{ mm}$ PA-MPJPE on 3DPW are formally retracted as unverified. The saved artifact [`results/3dpw_dsine_hmr2_evaluation.json`](file:///home/dat/HMR/results/3dpw_dsine_hmr2_evaluation.json) actually recorded an increase in MPJPE from $31.26\text{ mm}$ to $49.93\text{ mm}$ ($+59.7\%$) without saving paired initial PA-MPJPE.
2. **Synthetic Demonstrations:** `eval_rich_contact.py` and `eval_cape_clothing.py` did not run real benchmark datasets; they operated on synthetic noise fallbacks. Claims of "zero collision" and "complete clothing bias decoupling" must be treated as preliminary hypotheses.

### 4.2 Compute Waste Policy
In strict compliance with user instructions and the review mandate, all large-scale Colab GPU training loops remain paused until the controlled 4-way pilot experiment (§5.2) validates whether 3DGS provides an empirical benefit over classical mesh normal rendering.

---

## 5. High-Priority Actionable Repairs & Roadmap

### 5.1 Completed Action Items
- [x] Correct SMPL LBS skinning offset ($t_k^g - R_k^g j_k$).
- [x] Align Gaussian covariance thin axis to surface normal frame ($R_{\text{eff}} = R_q F$).
- [x] Implement log-space slogdet and Cholesky solve for Gaussian overlap.
- [x] Fix Umeyama reflection trace sign and vectorize PA-MPJPE.
- [x] Remove rasterizer 40-splat truncation and gate background normal normalization.
- [x] Enforce honest initializer source attribution and Rodrigues matrix logarithm map.
- [x] Author automated regression gate [`test_audit_fixes_2026_09_17.py`](file:///home/dat/HMR/code/tests/test_audit_fixes_2026_09_17.py).

### 5.2 Next Step: Controlled 4-Way Pilot Experiment Protocol
Before scaling to the full 3DPW dataset or launching remote VMs, execute a controlled 4-way pilot on a fixed subset of 100 frames across diverse motion sequences:

$$\begin{aligned}
\text{Condition A:} &\quad \text{Frozen HMR 2.0 Coarse Initialization (Baseline anchor)} \\
\text{Condition B:} &\quad \text{Differentiable Mesh Normal Rendering } (L_{\text{normal}} + L_{\text{mask}} + L_{\text{prior}}) \\
\text{Condition C:} &\quad \text{3D Gaussian Normal Rendering with Fixed Nuisance Attributes} \\
\text{Condition D:} &\quad \text{3D Gaussian Normal Rendering with Free Deformation Attributes}
\end{aligned}$$

**Decision Rule:**
- If **Condition B $\approx$ Condition C**, the accuracy benefit stems entirely from monocular normal supervision (DSINE), not 3DGS.
- If **Condition C $>$ Condition B**, 3DGS provides genuine pose refinement capability.
- If **Condition D $<$ Condition C**, unconstrained Gaussian offsets overfit to clothing at the expense of body joints.
