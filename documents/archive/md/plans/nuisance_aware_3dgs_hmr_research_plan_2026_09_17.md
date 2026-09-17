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

# Nuisance-Aware Articulated Pose Refinement in 3D Gaussian Splatting: Research & Experimental Plan

**Date:** September 17, 2026  
**Status:** Approved for Execution  
**Authors:** AI Research & Engineering Team (DiffNorm-Contact HMR)  
**Primary References:**
- Code Review & Repository Audit: [`3dgs_hmr_research_code_review_2026_09_17.md`](file:///home/dat/HMR/documents/md/reviews/3dgs_hmr_research_code_review_2026_09_17.md)
- Novelty Directions Monograph: [`3dgs_hmr_novelty_directions_2026_09_17.md`](file:///home/dat/HMR/documents/md/reviews/3dgs_hmr_novelty_directions_2026_09_17.md)
- Code Remediation Audit: [`diffnorm_code_audit_and_remediation_report_2026_09_17.md`](file:///home/dat/HMR/documents/md/reviews/diffnorm_code_audit_and_remediation_report_2026_09_17.md)

---

## 1. Executive Audit & Strategic Alignment

| Strategic Pillar | Prior Flawed Paradigm | Remediated Research Direction | Scientific Rationale |
|---|---|---|---|
| **Core Scientific Question** | *"Can 3DGS achieve SOTA HMR through normal torque & overlap repulsion?"* | **"When Should Gaussians Move the Skeleton? Does accounting for Gaussian nuisance flexibility prevent pose corruption?"** | Dense normals improve pose, but unconstrained 3DGS deformation allows the optimizer to explain the image with an incorrect skeleton and distorted splats. |
| **Novelty Claim** | Claimed novelty for Gaussian product integrals and monocular normal supervision. | **Nuisance-Aware Articulated Step Control via Local Sensitivity & Schur Curvature Attenuation.** | Normal supervision (ICON) and Gaussian collision proxies (SMPLify) predate 3DGS. Novelty lies in controlling the trade-off between skeletal pose and splat deformation. |
| **Mathematical Soundness** | Broken LBS translation ($430\text{ mm}$ shift); orthogonal disk covariances; $28\times$ overlap clamp distortion. | **Mathematically verified zero-pose identity, collinear tangent frames, exact log-space Cholesky overlap.** | Fixed and audited in [`diffnorm_code_audit_and_remediation_report_2026_09_17.md`](file:///home/dat/HMR/documents/md/reviews/diffnorm_code_audit_and_remediation_report_2026_09_17.md); all 28 project tests passing. |
| **Empirical Evaluation** | Unpaired 10-frame artifact with $+59.7\%$ worsening MPJPE; synthetic CAPE/RICH fallbacks. | **Paired initial vs. refined benchmark protocol (MPJPE, PA-MPJPE, PVE, Worsening Rate $W\%$) across authentic 3DPW.** | Replaces unverified historical claims with rigorous statistical paired evaluations and sequence-clustered error bounds. |
| **Compute Execution Policy** | Immediate scaling to 4,800-frame Colab VM training before isolating mechanisms. | **Fail-fast 4-way pilot experiment (100 frames) on local hardware before any large-scale GPU allocation.** | Eliminates compute waste by validating the core scientific hypothesis on a small, controlled dataset first. |

---

## 2. Theoretical & Mathematical Formulations

### 2.1 Articulated Kinematics and Gaussian Nuisance Parameters
Let $\boldsymbol{\theta} = \{R_k\}_{k=1}^{24} \in SO(3)^{24}$ denote articulated SMPL joint rotations, and let $\mathbf{t} \in \mathbb{R}^3$ denote global translation. We parameterize local kinematic perturbations $\delta \mathbf{x} \in \mathbb{R}^{75}$ via the Lie algebra $\mathfrak{so}(3)$:
$$R'_k = \exp([\delta \boldsymbol{\omega}_k]_\times) R_k, \qquad \mathbf{t}' = \mathbf{t} + \delta \mathbf{t}$$
where $[\delta \boldsymbol{\omega}_k]_\times \in \mathfrak{so}(3)$ is the skew-symmetric matrix of axis-angle perturbation $\delta \boldsymbol{\omega}_k \in \mathbb{R}^3$.

Let $\mathbf{a} \in \mathbb{R}^q$ collect all non-skeletal Gaussian parameters (nuisance parameters), including:
- Local vertex displacement offsets: $\boldsymbol{\delta}_i \in \mathbb{R}^3$ ($3N$ parameters)
- Tangential Gaussian scales: $\mathbf{s}_i \in \mathbb{R}^2$ ($2N$ parameters)
- Splat quaternions: $\mathbf{q}_i \in \mathbb{R}^4$ ($4N$ parameters)
- Opacities and diffuse colors: $\alpha_i \in \mathbb{R}, \mathbf{c}_i \in \mathbb{R}^3$ ($4N$ parameters)

### 2.2 Local Sensitivity & Curvature Attenuation (Schur Complement)
Let $\mathbf{r}(\mathbf{x}, \mathbf{a}) \in \mathbb{R}^m$ represent the concatenated image observation residual, combining normal cosine residuals $\mathbf{r}_N$ and silhouette coverage residuals $\mathbf{r}_M$. For a small perturbation $(\delta \mathbf{x}, \delta \mathbf{a})$ around current state $(\mathbf{x}_0, \mathbf{a}_0)$, the first-order Taylor expansion is:
$$\mathbf{r}(\mathbf{x} \oplus \delta \mathbf{x}, \mathbf{a} + \delta \mathbf{a}) \approx \mathbf{r} + J_x \delta \mathbf{x} + J_a \delta \mathbf{a}$$
where $J_x = \frac{\partial \mathbf{r}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times 75}$ and $J_a = \frac{\partial \mathbf{r}}{\partial \mathbf{a}} \in \mathbb{R}^{m \times q}$.

The regularized joint optimization objective is:
$$\min_{\delta \mathbf{x}, \delta \mathbf{a}} \frac{1}{2} \|\mathbf{r} + J_x \delta \mathbf{x} + J_a \delta \mathbf{a}\|_W^2 + \frac{1}{2} \delta \mathbf{a}^T \Lambda_a \delta \mathbf{a} + \frac{1}{2} \delta \mathbf{x}^T \Lambda_x \delta \mathbf{x}$$
where $W \succeq 0$ is a diagonal pixel confidence weight matrix (derived from DSINE normal uncertainty), $\Lambda_a = \lambda_a I_q \succ 0$ penalizes spline/splat deviation, and $\Lambda_x \succ 0$ is an anatomical pose prior Hessian.

Eliminating the nuisance perturbation $\delta \mathbf{a}^* = -(J_a^T W J_a + \Lambda_a)^{-1} J_a^T W (\mathbf{r} + J_x \delta \mathbf{x})$ yields the reduced skeletal system:
$$\left(S + \Lambda_x\right) \delta \mathbf{x} = -J_x^T W \left[I - J_a (J_a^T W J_a + \Lambda_a)^{-1} J_a^T W\right] \mathbf{r}$$
where $S \in \mathbb{R}^{75 \times 75}$ is the **Schur complement pose-curvature matrix**:
$$S = J_x^T W J_x - J_x^T W J_a (J_a^T W J_a + \Lambda_a)^{-1} J_a^T W J_x$$

### 2.3 Mathematical Proof of Pose Curvature Dilution
**Theorem 1 (Curvature Monotonicity):** *For any positive semi-definite observation weighting $W \succeq 0$ and non-zero nuisance damping $\Lambda_a \succ 0$, the effective pose curvature under Gaussian deformation is strictly upper-bounded by the rigid mesh curvature:*
$$S \preceq J_x^T W J_x$$
*Proof:* Let $M = J_a^T W J_a + \Lambda_a$. Since $W \succeq 0$ and $\Lambda_a \succ 0$, $M$ is symmetric positive definite ($M \succ 0$), and its inverse $M^{-1}$ is symmetric positive definite ($M^{-1} \succ 0$).  
For any non-zero skeletal direction $\mathbf{v} \in \mathbb{R}^{75}$, define $\mathbf{u} = J_a^T W J_x \mathbf{v} \in \mathbb{R}^q$. Then:
$$\mathbf{v}^T (J_x^T W J_x - S) \mathbf{v} = \mathbf{v}^T \left[J_x^T W J_a M^{-1} J_a^T W J_x\right] \mathbf{v} = \mathbf{u}^T M^{-1} \mathbf{u} \ge 0$$
Thus, $J_x^T W J_x - S \succeq 0$, establishing $S \preceq J_x^T W J_x$. $\blacksquare$

**Physical Meaning:** Releasing Gaussian degrees of freedom ($\mathbf{a}$) cannot increase skeletal pose certainty; it strictly absorbs residual energy, attenuating the curvature available to guide joint angles. Directions where $\mathbf{u} = J_a^T W J_x \mathbf{v}$ is large represent **nuisance-entangled pose modes** where splat deformations masquerade as skeletal motion.

### 2.4 Nuisance-Aware Adaptive Damping Rule
Rather than taking naive unconstrained Adam steps, the skeletal update is modulated by the spectral sensitivity of $S$:
1. Perform eigendecomposition of the dimensionless pose curvature: $S = V_S \operatorname{diag}(\sigma_1, \dots, \sigma_{75}) V_S^T$.
2. Compute the nuisance attenuation ratio along each mode $k$:
   $$\gamma_k = \frac{\mathbf{v}_k^T S \mathbf{v}_k}{\mathbf{v}_k^T (J_x^T W J_x) \mathbf{v}_k + \epsilon} \in [0, 1]$$
   - $\gamma_k \approx 1$: Pure skeletal mode; normal cues directly inform joint angles without splat interference.
   - $\gamma_k \ll 1$: Nuisance-dominated mode; splat deformations explain the image, creating high risk of skeletal drift.
3. Apply mode-adaptive trust-region damping:
   $$\delta \mathbf{x}^* = \sum_{k=1}^{75} \left[\frac{\gamma_k}{\sigma_k + \lambda_{\text{prior}}}\right] (\mathbf{v}_k^T \mathbf{g}_x) \mathbf{v}_k$$
   where $\mathbf{g}_x$ is the skeletal gradient. Modes corrupted by Gaussian flexibility are automatically frozen or anchored to the HMR 2.0 initializer.

---

## 3. Experimental Architecture & Pilot Protocol

Before performing multi-epoch training or processing thousands of frames, we execute a controlled **4-Way Pilot Experiment** across a fixed validation slice of $100$ frames from 3DPW (`downtown_arguing_00`, `downtown_walking_00`, `office_phoneCall_00`).

```
                              [Input Monocular RGB Image]
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
             [DSINE Normal Estimator]             [HMR 2.0 Coarse Predictor]
             (Dense Target Normals N*)            (Initial Pose θ_0, Trans t_0)
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────────┐
               ▼                           ▼                           ▼
        [Condition B]               [Condition C]               [Condition D]
    (Classical Mesh Normal      (Constrained Gaussian       (Unconstrained 3DGS
          Rendering)              Normal Rendering)          Gaussian Splatting)
    - Mesh triangles only       - Fixed offsets (δ = 0)     - Free offsets (δ_i)
    - Posed vertex normals      - Fixed scales (s_0)        - Free scales (s_i)
    - No splat deformation      - Identity quats (q_0)      - Free quats (q_i)
               │                           │                           │
               └───────────────────────────┼───────────────────────────┘
                                           │
                                           ▼
                       [Condition E: Proposed Controller]
                   (Nuisance-Aware Spectral Step Damping)
                                           │
                                           ▼
                     [Paired Metric Evaluation vs. Ground Truth]
                     - MPJPE (mm)        - PA-MPJPE (mm)
                     - PVE (mm)          - Worsening Rate W (%)
```

### 3.1 The 4 Experimental Conditions

| Condition | Surface Representation | Active Parameters | Scientific Hypothesis Tested |
|---|---|---|---|
| **A: Frozen Initializer** | Parametric SMPL Mesh | None (Frozen $\boldsymbol{\theta}_0, \mathbf{t}_0$) | Establishes the exact paired baseline error of raw HMR 2.0 feedforward inference. |
| **B: Mesh Normal Rendering** | Triangular Mesh | $\boldsymbol{\theta}, \mathbf{t}$ only | Tests whether monocular normal supervision ($L_N + L_{\text{mask}}$) alone explains pose improvements without 3DGS. |
| **C: Constrained Gaussian** | Tangential Gaussian Disks | $\boldsymbol{\theta}, \mathbf{t}$ (Gaussians fixed: $\boldsymbol{\delta}=\mathbf{0}, \mathbf{s}=\mathbf{s}_0, \mathbf{q}=\mathbf{q}_0$) | Isolates the differential rendering representation of 3DGS from Gaussian parameter flexibility. |
| **D: Unconstrained Gaussian** | Tangential Gaussian Disks | $\boldsymbol{\theta}, \mathbf{t}$ AND $\boldsymbol{\delta}_i, \mathbf{s}_i, \mathbf{q}_i, \mathbf{c}_i, \alpha_i$ | Tests whether releasing Gaussian nuisance flexibility degrades skeletal pose accuracy (the nuisance interference hypothesis). |
| **E: Nuisance-Controlled** | Tangential Gaussian Disks | $\boldsymbol{\theta}, \mathbf{t}, \mathbf{a}$ with Spectral Attenuation Damping | Demonstrates whether the proposed sensitivity controller prevents pose corruption while improving appearance fidelity. |

### 3.2 Quantitative Falsification Criteria & Decision Rules
1. **Falsification of 3DGS Accuracy Advantage:** If Condition B (Mesh) achieves equal or superior MPJPE/PA-MPJPE to Condition C (Gaussian) at matched compute, 3DGS does not provide an accuracy benefit over classical mesh rendering.
2. **Confirmation of Nuisance Interference:** If Condition D achieves lower photometric/normal rendering loss than Condition C, but significantly worsens MPJPE ($W_{\text{rate}} > 30\%$), the hypothesis that unconstrained Gaussians corrupt skeletal pose is empirically proven.
3. **Validation of Proposed Method:** Condition E must achieve:
   - Lower PA-MPJPE than Condition A (paired improvement).
   - Statistically significant reduction in Worsening Rate: $W(E) < W(D)$ with $p < 0.01$.
   - Superior joint accuracy compared to naive static detachment.

---

## 4. Benchmark Execution & Statistical Validation Protocol

### 4.1 Evaluation Dataset & Provenance
- **Dataset:** 3DPW Official Test Split ($24$ sequences, diverse motions, outdoor lighting, natural clothing).
- **Paired Protocol:** For every frame $t$, record initial metrics $M_0(t)$ and refined metrics $M_{\text{ref}}(t)$. Never compare against unpaired published numbers from differing splits.
- **Reporting Metrics:**
  1. **MPJPE (mm):** Pelvis-aligned Mean Per Joint Position Error across 24 joints.
  2. **PA-MPJPE (mm):** Procrustes-aligned joint error via sample-vectorized Umeyama alignment.
  3. **PVE (mm):** Per-Vertex Error against ground truth SMPL mesh (shaped with $\boldsymbol{\beta}$ and translated with $\mathbf{t}$).
  4. **Worsening Rate ($W\%$):** Percentage of frames where $\text{PA-MPJPE}_{\text{ref}} > \text{PA-MPJPE}_0$.
  5. **Tail Error ($P_{95}$ mm):** 95th percentile error measuring robustness against catastrophic optimization divergence.
  6. **Throughput (FPS):** Actual optimization iterations per second and fully refined images per second.

### 4.2 Sequence-Clustered Statistical Inference
Because consecutive video frames within a 3DPW sequence are temporally correlated, standard i.i.d. student's $t$-tests underestimate standard errors. We report clustered standard errors grouped by sequence $s \in \{1, \dots, 24\}$:
$$\widehat{\operatorname{Var}}(\bar{\Delta}) = \frac{1}{N^2} \sum_{s=1}^{S} \left(\sum_{i \in \mathcal{I}_s} (\Delta_i - \bar{\Delta})\right)^2$$
where $\Delta_i = M_{\text{ref}, i} - M_{0, i}$. All reported gains must achieve $95\%$ confidence under sequence clustering.

---

## 5. Phased Work Breakdown & Actionable Task List

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Controlled 4-Way Pilot Experiment Implementation (Immediate)       │
│ - Implement Condition B (Differentiable Mesh Normal Rasterizer)             │
│ - Implement Condition C & D in pilot evaluation script                     │
│ - Execute 100-frame paired benchmark on local RTX 3050 Ti GPU               │
│ - Decision Gate 1: Verify whether unconstrained 3DGS degrades pose (D vs C) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Nuisance Sensitivity Diagnostic & Step Controller (Sprint 2)       │
│ - Implement JVP-based Schur complement curvature block S in PyTorch         │
│ - Implement spectral attenuation damping rule for optimizer steps           │
│ - Benchmark Condition E on pilot set; verify reduction in worsening rate   │
│ - Decision Gate 2: Confirm Condition E beats Condition D and Condition B    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Benchmark Scaling, Ablation Suite & Paper Deliverables             │
│ - Scale validated controller to 4,800-frame 3DPW cache & test set           │
│ - Generate versioned result JSONs and paired progression scatter plots      │
│ - Compile publication-ready manuscript with complete mathematical proofs    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Actionable Check-Off List

- [ ] **Task 1.1**: Author `code/src/rendering/mesh_normal_rasterizer.py` implementing classical differentiable mesh normal rendering for Condition B.
- [ ] **Task 1.2**: Author `code/scripts/pilot_4way_experiment.py` orchestrating Conditions A, B, C, D on 100 3DPW frames with paired logging.
- [ ] **Task 1.3**: Run local GPU benchmark and generate initial empirical comparison table.
- [ ] **Task 2.1**: Implement `NuisanceSensitivityEstimator` in `code/src/optimization/nuisance_sensitivity.py`.
- [ ] **Task 2.2**: Integrate adaptive spectral step damping into `dual_frequency_router.py`.
- [ ] **Task 2.3**: Evaluate Condition E on pilot frames and measure worsening rate suppression.
- [ ] **Task 3.1**: Execute full 3DPW test evaluation and compile verified master results.
- [ ] **Task 3.2**: Render final technical monograph and empirical audit to PDF deliverable.
