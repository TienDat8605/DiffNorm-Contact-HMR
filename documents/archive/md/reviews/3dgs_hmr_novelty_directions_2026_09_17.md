<style>
body.markdown-body { font-size: 11px !important; line-height: 1.45 !important; padding: 0 !important; min-width: 0 !important; }
.markdown-body table { table-layout: fixed; break-inside: auto !important; page-break-inside: auto !important; font-size: 10px; }
.markdown-body tr { break-inside: avoid; page-break-inside: avoid; }
.markdown-body th, .markdown-body td { padding: 5px 7px !important; overflow-wrap: anywhere; }
.markdown-body h1 { font-size: 23px !important; }
.markdown-body h2 { font-size: 18px !important; }
.markdown-body h3 { font-size: 14px !important; }
</style>

# Research Direction: When Should Gaussians Move the Skeleton?

Candidate research questions, 17 September 2026. AI-generated starting points, not established novelty claims or experimental results. Based on the preceding repository audit and a targeted primary-source search; this is not an exhaustive literature review.

## 1. Executive Audit

| Direction | Verdict | Reason |
|---|---|---|
| Add normal supervision to Gaussian HMR | Weak standalone novelty | Normal-driven body refinement has precedent; Gaussian HMR already exists. |
| Add analytical Gaussian collision | Weak standalone novelty | Gaussian overlap penalties for body fitting predate 3DGS; a surface overlap remains a surrogate. |
| Nuisance-aware articulated-pose refinement | Recommended candidate | Ask which pose evidence survives after Gaussian appearance and deformation can explain the image. |
| Pose-aware Gaussian capacity allocation | Higher-risk alternative | Control where splats may gain flexibility according to their interference with skeletal estimation. |
| Clothing-aware normal-to-body supervision | Useful alternative, crowded | Infer when an observed clothed normal is valid evidence about the underlying body. |

**Primary research question:** At matched image supervision and computation, does accounting for Gaussian nuisance flexibility reduce the frequency with which single-image refinement worsens human pose relative to its HMR initialization?

The scientific target is mesh accuracy and refinement reliability, not attractive renders. A system can explain an image using a wrong skeleton and compensating splat deformation. This is a hypothesis about the mechanism to test, not an explanation established by the currently flawed evaluation pipeline.

## 2. Theoretical & Mathematical Formulations (Correct Parts vs. Required Corrections)

### Recommended mechanism: residual attribution through local sensitivity

Start with fixed body shape and fixed camera intrinsics. Let $R_k\in SO(3)$, $k=1,\ldots,24$, be SMPL joint rotations and let $t\in\mathbb R^3$ be root translation. Define local increments $\delta x\in\mathbb R^{75}$ using $R_k'=\exp([\delta\omega_k]_\times)R_k$ and $t'=t+\delta t$. This is a local rotation update, not a finite axis-angle Jacobian identity.

Let $a\in\mathbb R^q$ collect permitted Gaussian appearance and bounded surface-deformation parameters. For a residual $r\in\mathbb R^m$ with fixed active visibility and locally fixed weights $W\succeq0$, write

$$r(x\oplus\delta x,a+\delta a)\approx r+J_x\delta x+J_a\delta a,$$

where $J_x\in\mathbb R^{m\times75}$ and $J_a\in\mathbb R^{m\times q}$. A local regularized nuisance-elimination model is

$$\min_{\delta x,\delta a}\ \tfrac12\|r+J_x\delta x+J_a\delta a\|_W^2+\tfrac12\delta a^T\Lambda_a\delta a+\text{pose-prior term},$$

with $\Lambda_a\succ0$ and $\|u\|_W^2=u^TWu$. Eliminating $\delta a$ yields the pose data-curvature block

$$S=J_x^TWJ_x-J_x^TWJ_a(J_a^TWJ_a+\Lambda_a)^{-1}J_a^TWJ_x.$$

For this local model, $S\preceq J_x^TWJ_x$: allowing nuisance adaptation cannot increase the residual's remaining pose curvature. This follows because the subtracted matrix is positive semidefinite. Directions poorly constrained by $S$ should receive smaller updates or stronger anchoring to the initializer. Compare eigenvalues only in a declared, dimensionless parameter metric; radians and metres cannot be combined arbitrarily.

**What is not new:** Schur complements, variable elimination, and nuisance-parameter optimization are established techniques [4]. Eliminating variables alone is algebraically equivalent to solving the same joint local system; it does not create a new objective or guarantee better recovery.

**Candidate contribution:** an efficient articulated-body-specific approximation to this sensitivity, paired with a justified capacity/step-control rule that predicts and reduces harmful updates. Demonstrate improvement over ordinary joint optimization, alternating optimization, static detachment, confidence weighting, and a generic reduced-system solver. The weighting and control rule, not the displayed equation alone, must earn the contribution.

**Important limitations:** This is regularized local sensitivity, not calibrated posterior uncertainty or global identifiability. Flexible Gaussians may remove nearly all pose evidence. Strong regularization may merely freeze the initializer. Occlusion, clothing, illumination and normal-predictor bias remain unresolved. No method can recover information absent from the observation without additional assumptions.

## 3. Implementation & Algorithmic Reality

Use the existing project as a prototype scaffold, not as empirical support. The preceding audit identified invalid SMPL/HMR initialization and benchmark paths. Local source files have subsequently changed; this note does not re-audit those edits.

For a minimum experiment, use authentic SMPL and HMR initialization, a small fixed set of surface-bound Gaussians, bounded deformation and no densification. Compute a low-dimensional nuisance approximation using regional bases or Jacobian-vector products; measure the approximation against a tiny dense reference. Do not form a pixel-by-pixel projector or invert a full splat-parameter matrix.

Keep the current normal cue as one residual channel, not the headline novelty. Retain visibility and confidence masks, with the complete renderer and unit-normal derivatives. Begin without collision to isolate the mechanism. Fix body shape initially and describe the output as pose refinement; adding shape estimation requires additional ambiguity controls and evaluation.

Two alternatives worth keeping:

1. **Pose-aware capacity allocation:** Can releasing Gaussian deformation only in low-interference regions improve HMR at a fixed splat budget? This targets representation design rather than merely optimizer choice. Risk: withholding deformation can incorrectly push clothing mismatch into the skeleton. Compare to uniform capacity, simple delayed release and generic regularization.
2. **Clothing-aware normal supervision:** Can a model distinguish normals explained by body pose from normals explained by clothing, and thereby avoid pulling the body toward garments? Risk: ordinary confidence weighting and normal-guided reconstruction already cover much of this territory [3]. It needs explicit clothing/body mismatch modeling and suitable evidence, not just another scalar loss weight.

Initial FINER judgments (subjective 1–5 feasibility assessments, not validated novelty scores): recommended direction F=3, I=4, N=3, E=4, R=4; capacity allocation F=3, I=4, N=3, E=4, R=4; clothing-aware normals F=3, I=4, N=2, E=4, R=4. Feasibility depends on repairing the evaluation; novelty remains provisional; ethical acceptability depends on dataset permissions and responsible handling of identifiable imagery. The first direction is preferred for its direct connection to the existing gradient-routing proposal, not a numerical ranking advantage.

## 4. Empirical Benchmarks & Limitations

**Decisive first test:** sweep Gaussian flexibility while keeping initial body predictions, observations and compute fixed. Record whether image residual decreases while actual joint/vertex error increases. Then test whether the proposed sensitivity predicts these failures and whether its control rule reduces them.

All three subquestions inherit the same scope: single-image, single-person, fixed-shape SMPL pose refinement from authentic HMR predictions; no ground-truth initialization in real-image evaluation.

1. Does increasing permitted nuisance flexibility increase pose–render disagreement?
2. Does nuisance-aware sensitivity predict harmful pose updates better than residual magnitude or normal confidence?
3. Does sensitivity-controlled refinement improve the accuracy–coverage tradeoff over matched controls?

Use synthetic known-geometry perturbations only for mechanism validation. Follow with a declared real-image benchmark split, consistent coordinate transforms, genuine observations, and per-example paired initial/refined MPJPE, PA-MPJPE and PVE. Report worsening rate, tail errors, runtime and the fraction of examples actually updated. For correlated video frames, uncertainty estimates should respect sequence grouping.

Essential controls: unchanged initializer; normals-only mesh refinement; Gaussian refinement with static detachment; ordinary joint and alternating optimization; generic regularized reduced-system optimization; proposed adaptive control. Match priors, supervision, initial conditions and compute where possible. Include a mesh-renderer version with comparable nuisance capacity to determine whether the benefit is genuinely Gaussian-specific.

**Falsification:** If improvement disappears after matching regularization and compute, the proposed mechanism is unsupported. If it simply rejects nearly all updates, compare against equally conservative baselines at matched coverage. If it works equally well with a mesh renderer, frame the contribution as general inverse-rendering HMR optimization, not a unique property of 3DGS.

## 5. High-Priority Actionable Repairs / To-Do List

1. Establish trustworthy SMPL, image-dependent HMR initialization and paired evaluation before adding modules.
2. Reproduce the capacity-versus-pose-error phenomenon with controls; abandon this mechanism if it does not occur.
3. Implement the smallest sensitivity diagnostic and compare it to cheap confidence/gradient baselines.
4. Add adaptive control only after demonstrating predictive value; validate improvement at matched coverage.
5. Audit the closest literature at full-method level before claiming novelty. Do not claim “first uncertainty-aware 3DGS refinement.”

**Prior-work boundary and source trail.** GST already connects single-image human-body prediction and Gaussian rendering [1]. HumanSplatHMR already closes the pose/avatar optimization loop [2]. ICON already uses normal-related feedback for human reconstruction [3]. Variable elimination has a long history [4]. UGS-Loc addresses camera-pose and geometric uncertainty in 3DGS through sampling and Fisher-information-based PnP [5]; its abstract concerns camera localization, not the proposed articulated pose-versus-adaptive-avatar mechanism. The distinction needs full-method checking, not an assertion of absence.

- [1] [GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers](https://arxiv.org/abs/2409.04196).
- [2] [HumanSplatHMR: Closing the Loop Between Human Mesh Recovery and Gaussian Splatting Avatar](https://arxiv.org/abs/2605.02784).
- [3] [ICON: Implicit Clothed humans Obtained from Normals](https://openaccess.thecvf.com/content/CVPR2022/html/Xiu_ICON_Implicit_Clothed_Humans_Obtained_From_Normals_CVPR_2022_paper.html).
- [4] [Aravkin and van Leeuwen, Estimating Nuisance Parameters in Inverse Problems](https://arxiv.org/abs/1206.6532).
- [5] [Kong et al., Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](https://arxiv.org/abs/2603.16538), CVPR 2026.
- Collision precedent: [Bogo et al., Keep it SMPL](https://files.is.tue.mpg.de/black/papers/BogoECCV2016old.pdf), Gaussian body-interpenetration penalty.

Search scope: targeted web queries on 17 September 2026 combining human mesh recovery, Gaussian splatting, pose/appearance ambiguity, normals, geometric uncertainty, Schur complement and variable projection; primary papers, author repositories and official proceedings were preferred. Earlier full-text review supports the project-specific comparisons; the newly found UGS-Loc comparison is abstract-level. No systematic coverage or novelty certification is implied. No new training or performance measurements were performed for this note.
