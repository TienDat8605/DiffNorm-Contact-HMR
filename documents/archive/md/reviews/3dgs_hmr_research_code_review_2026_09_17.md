<style>
body.markdown-body { font-size: 11px !important; line-height: 1.45 !important; padding: 0 !important; min-width: 0 !important; }
.markdown-body table { table-layout: fixed; break-inside: auto !important; page-break-inside: auto !important; font-size: 10px; }
.markdown-body tr { break-inside: avoid; page-break-inside: avoid; }
.markdown-body th, .markdown-body td { padding: 5px 7px !important; overflow-wrap: anywhere; }
.markdown-body code { overflow-wrap: anywhere; }
.markdown-body h1 { font-size: 23px !important; }
.markdown-body h2 { font-size: 18px !important; }
.markdown-body h3 { font-size: 14px !important; }
</style>

# 3DGS for Human Mesh Recovery: Research and Code Review

## 1. Executive Audit

| Area | Verdict | Root cause / evidence |
|---|---|---|
| Research direction | Worth testing | Dense geometric supervision may improve pose; the specific benefit of Gaussian rendering is unmeasured. |
| Claimed HMR improvement | Unsupported | No paired initial PA-MPJPE in the saved ten-frame artifact; recorded MPJPE worsens. |
| Body geometry | Critical implementation failure | Default synthetic humanoid; incorrect rest-pose skinning transform; missing SMPL components. |
| HMR 2.0 initialization | Not implemented | Checkpoint loading never constructs the prediction network. |
| Gaussian normal geometry | Inconsistent | Rendered normal and covariance’s thin axis are different vectors. |
| Gaussian overlap | Correct ideal formula, incorrect default numerical evaluation | Determinant flooring changes the default overlap by approximately 28 times. |
| Contact / collision claims | Unsupported | Surface-kernel repulsion is neither solid occupancy nor a contact constraint. |
| Gradient routing | Partial isolation | RGB has no direct pose gradient, but geometric losses still update free Gaussian parameters. |
| Evaluation and training | Not a valid research pipeline yet | Protocol errors, synthetic evaluations, shared per-frame parameters, unused evaluation checkpoint. |
| Alternative UV-attention proposal | Testable idea, invalid proof | GST’s group decoder is not the asserted rank-26 broadcast operator. |

**Recommendation:** major redesign and revalidation before scaling experiments. As a research proposal, this contains a worthwhile question. As an empirical paper, the current evidence cannot support its conclusions. I would not submit the present claims or spend more GPU time on the present training loop.

The strongest possible contribution is a demonstrated improvement in *body pose or shape attributable to the Gaussian representation*, with matched supervision and initialization. Faster data loading, a correct Gaussian identity, and a nonzero autograd path do not establish that contribution.

**Scope and provenance.** Review date: 17 September 2026. Code revision: `426ff83b653e9c2706cf19a7d371eb6cbcfdcde0`. Examined the main proposal, comparative proposals, three active technical notes, prior mathematical review, README, core geometry/rendering/optimization/evaluation/training/cache code, tests, and saved result JSONs. Read the two PDF-only proposal documents and relevant GST/HumanSplatHMR material using PyMuPDF4LLM. PDF extraction drops some embedded equations; no criticism here relies on reconstructing those missing symbols. This is a single-assistant technical assessment informed by the academic-review skill’s evidence criteria, not a completed five-reviewer panel or venue-calibrated acceptance prediction. Calibration: `NOT_CALIBRATED`; `criteria_binding_unavailable`. No training, GPU benchmark, or proposal/code repair was performed.

### 1.1 What deserves to be retained

1. **A concrete inverse-rendering hypothesis.** Surface orientation is useful information that sparse keypoints omit. Testing whether it corrects out-of-plane pose errors is reasonable. Evidence: `proposal.md`, §2.2.
2. **Explicit separation of body and appearance objectives.** The detached RGB branch is a real, testable mechanism. Evidence: `dual_frequency_router.py:164` and `test_gradient_router.py`.
3. **An analytically tractable regularizer.** The stated unnormalized Gaussian product integral is correct. Evidence: `proposal.md`, §2.3; its production-scale numerical implementation needs repair.
4. **Useful module boundaries.** Geometry, rendering, loss routing, and metrics are separable enough to test and replace individually. Existing tests provide a starting point, although they mostly check local consistency rather than scientific validity.

## 2. Theoretical & Mathematical Formulations (Correct Parts vs. Required Corrections)

### 2.1 Define the actual problem before choosing the representation

Use $\theta\in\mathbb R^{24\times3}$ for axis-angle pose, $\beta\in\mathbb R^{10}$ for shape, and $t\in\mathbb R^3$ for translation. Distinguish the underlying body mesh $V(\theta,\beta)$ from a clothed surface or Gaussian avatar. Better PSNR or cloth detail does not imply better body joints. Conversely, better body joints do not establish an accurate animatable garment.

The documents alternate between single-image test-time refinement, a learned feedforward estimator, multi-frame avatar fitting, and cloth physics. These require different training variables, evidence, and comparisons. The present `optimize_frame` is a per-image fitting procedure. It has no learned image-to-pose predictor of its own and does not optimize shape. Describe that scope accurately.

### 2.2 Normal supervision is plausible, but neither unique to 3DGS nor sufficient

For depth-sorted splats $i=1,\ldots,N$ and pixel $p$, define

$$w_{ip}=\alpha_{ip}\prod_{j<i}(1-\alpha_{jp}),\qquad a_p=\sum_iw_{ip}n_i,\qquad \hat n_p=\frac{a_p}{\|a_p\|}.$$

With fixed nonnegative confidence weights $c_p$, $C=\sum_pc_p>0$, and unit targets $n_p^*$,

$$L_N=\frac1C\sum_pc_p(1-\hat n_p^Tn_p^*).$$

Away from zero accumulation and discrete visibility boundaries,

$$\frac{\partial L_N}{\partial a_p}=-\frac{c_p}{C\|a_p\|}(I-\hat n_p\hat n_p^T)n_p^*,$$

$$\frac{\partial a_p}{\partial\theta}=\sum_i\left[w_{ip}\frac{\partial n_i}{\partial\theta}+n_i\frac{\partial w_{ip}}{\partial\theta}\right].$$

The second term includes projected centers, projected covariances, opacity/transmittance, and visibility dependence. The technical notes’ normal-only Jacobian does not describe the complete rendered objective. PyTorch can differentiate implemented continuous paths, but that does not validate an incomplete derivation or make sorting differentiable.

For a left rotation perturbation, $\delta n=\delta\omega\times n$, so $\delta L=(n\times g)^T\delta\omega$. This is a local rotation gradient; the descent direction has the opposite sign. It is not a physical torque guarantee or the finite axis-angle Jacobian. For $n(\theta)=\operatorname{Exp}([\theta]_\times)n_0$, that Jacobian includes the appropriate $SO(3)$ left Jacobian: $- [n]_\times J_l(\theta)$. Skinning and mesh-normal differentiation introduce further dependencies.

**Major conceptual limitation.** Monocular normal estimates describe visible clothing and exposed surfaces, not necessarily the underlying naked body. A coat normal can bias a body joint even when RGB gradients are detached. Global scale, occluded geometry, symmetries, and correspondence remain ambiguous. Normals are invariant to uniform geometric scaling, but their pixel correspondence and their estimation can depend on camera intrinsics. The comparison document’s claim to bypass focal-length problems entirely is therefore too strong.

Normal-guided clothed reconstruction also has substantial precedent: ICON explicitly combines SMPL and inferred normals in an iterative feedback process. Establish novelty against that family, not only against RGB-only Gaussian avatars. [ICON, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xiu_ICON_Implicit_Clothed_Humans_Obtained_From_Normals_CVPR_2022_paper.html).

### 2.3 A Gaussian overlap integral is not a collision barrier

Let $\Sigma_i\succ0$ have units $\mathrm m^2$, $\mu_i\in\mathbb R^3$ have units m, and

$$g_i(x)=\exp[-\tfrac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)].$$

Writing $S=\Sigma_i+\Sigma_j$ and $d=\mu_i-\mu_j$, the correct integral is

$$K_{ij}=(2\pi)^{3/2}\sqrt{\frac{|\Sigma_i||\Sigma_j|}{|S|}}\exp(-\tfrac12d^TS^{-1}d).$$

It has units $\mathrm m^3$, but it is not the intersection volume of solids. With fixed covariances, $\nabla_{\mu_i}K_{ij}=-K_{ij}S^{-1}d$ is correct. Several consequences defeat stronger claims:

- At coincident centers, $d=0$ and the center gradient is zero, even at maximum pair overlap. This is a bounded penalty, not a barrier preventing entry into an invalid state.
- Thin kernels placed on a surface do not fill its interior. A body part deeply inside another body can have little nearby surface-kernel overlap. Conversely, valid near-contact can incur a penalty.
- Learnable widths and offsets can reduce overlap without fixing the body mesh. For identical coincident kernels with scales multiplied by $\epsilon$, the ideal overlap scales as $\epsilon^3$.
- A raw pair sum depends on sampling density and counts overlapping regions repeatedly. Fix the discretization or use justified weights before comparing magnitudes.
- Hard distance and segment-sum thresholds can create discontinuities, not merely derivative discontinuities. The notes’ blanket "$C^0$ across activation boundaries" statement is false for this implementation.

**Required interpretation:** a local repulsion surrogate whose relationship to mesh penetration must be measured. Contact preservation requires an additional formulation or explicit evaluation; repulsion alone favors separation. "Contact HMR" currently promises more than the implemented objective.

Gaussian body representations predate modern 3DGS; Stoll et al. used a sums-of-Gaussians body model for articulated tracking in 2011. The product identity alone cannot carry a novelty claim. [Original paper](https://people.mpi-inf.mpg.de/~theobalt/sog.pdf). Also, BVH candidate selection can coexist with differentiable collision penalties; the claim that mesh collision optimization is categorically impossible is contradicted by existing implementations. [SMPLify-X fitting code](https://github.com/vchoutas/smplify-x/blob/master/smplifyx/fitting.py).

More directly, **SMPLify already uses a scaled integral of products of 3D Gaussian proxies for incompatible body parts**, and explicitly avoids using that term to optimize shape because it would encourage thin bodies. This anticipates both the core collision mechanism and the shrinkage problem. An anisotropic surface-splat variant may still be useful, but it needs a specific distinction and comparison rather than presenting differentiable Gaussian collision as a new solution. [SMPLify, §3.2](https://files.is.tue.mpg.de/black/papers/BogoECCV2016old.pdf).

### 2.4 Detachment does not solve identifiability

The implemented relationship is approximately

$$\nabla_\theta L_{\rm deform}=0,\qquad \nabla_\phi L_{\rm geom}\ne0,$$

where $\phi$ collects offsets, scales, quaternions, and opacity. The deformation optimizer receives both losses’ accumulated gradients. Thus geometry can still be explained by changing Gaussian attributes. A free per-vertex quaternion can align a normal without correcting the skeleton; free offsets can satisfy a silhouette without correcting joints.

Calling this a frequency decomposition is unjustified: there is no spectral projection. Nor is graph Dirichlet smoothing a calibrated cloth elasticity model. A useful first experiment freezes nuisance geometry during pose fitting and compares this with joint optimization and the current one-way detachment. This is a suggested control, not a theorem that freezing always wins.

### 2.5 Review of the other proposals and comparison claims

**UV/deformable-attention proposal:** the one-variable experimental intent is better scoped, but the claimed GST rank proof is not grounded. GST describes $5K+1$ tokens and a linear readout producing distinct parameters for groups of 265 Gaussians, with $K=26$. That does not imply a shared feature broadcast $PF$ with rank at most 26. Separate output rows can predict distinct within-group attributes. Sparse vertex queries remain a valid empirical hypothesis, but "rank-6890" and guaranteed wrinkle recovery do not follow. Verify the actual decoder and compare against larger grouped decoders with matched compute. The proposal’s exact memory and training-time numbers need profiling. [GST, §§3.1 and 4.2](https://arxiv.org/html/2409.04196v2).

**Phys-HMR-GS / OmniHMR-GS:** adaptive margins, dynamics, temporal recurrence, and feature sampling each introduce a separate hypothesis. One-view appearance does not identify garment stiffness or inertia without additional motion/excitation assumptions. Bidirectional recurrence requires future frames, and smooth latent recurrence alone does not guarantee temporal consistency. These are later research directions, not prerequisites for testing whether 3DGS improves HMR.

**HumanSplatHMR comparison:** the local comparison overstates both our success and the competitor’s failure. Its local paper describes joint pose/avatar fitting with depth and segmentation, and it studies alternative Gaussian-to-body bindings. Its modest pose improvement does not prove that improvements come exclusively from global scale or translation. That causal explanation requires an ablation. Our ten-frame result cannot be ranked against its different protocol. [HumanSplatHMR](https://arxiv.org/abs/2605.02784).

GST’s multi-view *training* does not mean its *inference* requires a camera rig. Avoid conflating them when defining the monocular advantage. The strongest counterargument to DiffNorm is that any gain might come entirely from pretrained normal supervision plus stronger regularization, with 3DGS adding unnecessary degrees of freedom.

## 3. Implementation & Algorithmic Reality

Findings below distinguish directly reproduced failures from static-code conclusions. Critical means the issue independently prevents the present central empirical claim from being accepted; Major means substantial repair is needed. Confidence is high for reproduced/code-local findings, moderate for empirical consequences not yet measured.

### F1 — Critical: default body model is synthetic, and LBS is incorrect

**Evidence:** `code/src/geometry/smpl_wrapper.py:183`, `:162`, `:265`, `:300`; `optimize_single_image.py:49`; `eval_3dpw.py:264`.

Evaluation and single-image fitting instantiate `SMPLWrapper` without a model path. It creates random shell points, sequential-index triangles, synthetic blend weights, and random shape directions. Having 6,890 vertices and 24 joints does not make this SMPL. Applying dataset SMPL coefficients to this template is not evaluation against dataset body geometry.

There is also a separate LBS bug affecting the official-file branch. For $G_k=[R_k^g,t_k^g;0,1]$, the correct skinning translation is $t_k^g-R_k^gJ_k$. The code uses $R_k^gJ_k+t_k^g-J_k$. At zero pose it should produce identity skinning transforms, but returns translations as large as 0.88 m. The neutral mesh moves **0.430285 m on average** from its template in the included probe.

The official loader additionally omits pose blend directions and keeps joints independent of shape. The standard implementation recomputes joints from shaped vertices and applies pose offsets. [SMPL-X LBS reference](https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py).

**Minimum repair:** use a validated SMPL implementation, require licensed model assets explicitly, and test template identity, single-joint motion, shape-dependent joints, and reference-output parity. Confidence: 5/5, reproduced and algebraically verified.

### F2 — Critical: the advertised initializer does not run HMR 2.0

**Evidence:** `coarse_pose_hmr2.py:39`, `:55`, `:89`, `:120`.

`self.model` remains `None`; loading weights only stores a dictionary. Inputs with all-black and all-white pixels yield identical fallback poses. If a checkpoint loads, the fallback can nevertheless label its source "4D-Humans-HMR2.0". Merely setting `self.model` later also leaves a wrong output conversion: official body pose is a set of rotation matrices with global orientation separate, not a tensor that can be reshaped into 24 axis-angle vectors. [Official prediction head](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/heads/smpl_head.py).

**Minimum repair:** integrate the official inference path, crop/normalize correctly, convert rotation representations explicitly, preserve predicted shape and camera convention, and fail evaluation when unavailable. Confidence: 5/5, static and reproduced.

### F3 — Major: disk covariance and rendered normal disagree

**Evidence:** `splat_surface.py:116` and `:129`.

The normal is $R_qn_{\rm mesh}$ but the covariance’s shortest axis is $R_qe_3$. These coincide only for a special mesh normal. With identity quaternion and $n_{\rm mesh}=e_1$, their absolute dot product is **0**, rather than 1. Supplying mesh normals restored a real pose gradient since the older audit, but did not establish tangent-aligned covariance or prevent quaternion compensation.

**Minimum repair:** derive both covariance and normal from the same posed surface frame. Apply camera rotation consistently to centers, normals, and covariances; the current optional `R_cam` rotates only normals. Confidence: 5/5, reproduced.

### F4 — Major: "exact" overlap is numerically altered at default scales

**Evidence:** `gaussian_convolution.py:60` and `:74`; `collision_loss.py:100`.

Default scales are $(0.015,0.015,0.00045)$ m. Their covariance determinant is approximately $1.0252\times10^{-14}$, below the hard $10^{-12}$ floor. Two identical coincident defaults should integrate to $\pi^{3/2}s_1s_2s_3=5.6379\times10^{-7}$; code returns $1.5750\times10^{-5}$, **27.94 times larger**. Flooring also removes the prefactor’s correct scale derivative in the clamped regime.

The Mahalanobis upper clamp imposes a positive tail floor and zero center gradient beyond that threshold. The analytical-gradient helper has a different active threshold, so it is not a general derivative of the clamped function. The router uses autograd, not that helper. Segment sums below $10^{-6}$ are entirely omitted, making zero diagnostics particularly weak evidence.

**Minimum repair:** use stable log-determinants/Cholesky evaluation with declared covariance conditioning, a justified tail cutoff, production-scale tests, and explicit loss units. Confidence: 5/5, reproduced.

### F5 — Major: renderer truncation and tail flooring corrupt visibility

**Evidence:** `normal_rasterizer.py:163`, `:189`.

Only the nearest 40 splats are retained for an entire tile. A farther splat can be the only contributor at another pixel in that tile. The probe yields opacity **0.01267 versus 0.95063** at such a pixel when the cap changes from 40 to 100. This is not valid transmittance-based early termination.

`exp(-0.5 * maha.clamp_max(16))` gives every selected splat a nonzero tail everywhere in the tile. A far-corner pixel with opacity only **0.000319** nevertheless receives a unit-length rendered normal. The normal loss lacks rendered-coverage gating, so almost invisible support can act as full directional evidence.

**Minimum repair:** benchmark against a trusted Gaussian or mesh renderer, remove arbitrary tile truncation, and handle unsupported pixels explicitly without allowing the model to hide hard pixels by reducing coverage. Confidence: 5/5, reproduced.

### F6 — Major: supervision is inconsistent with the documented protocol

**Evidence:** `eval_3dpw.py:203`; `dataset_3dpw.py:68` and `:121`; `extract_dsine_normals.py:115`; `build_3dpw_cache.py`, mask construction.

The fitting mask is a dilated prediction of the initial body, not an observed silhouette. It cannot locate a missed limb and may anchor the optimization to a wrong initial projection. Cache construction uses a central rectangle; another cache path uses $|n_z|<0.98$, which is not person segmentation. Image-derived subject masks are needed.

The dataset returns camera extrinsics, but the evaluator does not use them to reconcile dataset-world and predicted-camera geometry. Each actor receives the same resized full image, with no actor-specific crop in this path. Missing images become gray placeholders. DSINE receives a fixed 60-degree field of view instead of the available sample intrinsics, and fitting passes no uncertainty map. DSINE inference was not successfully validated in this review’s environment.

**Minimum repair:** make image/crop/intrinsics/extrinsics/normal conventions explicit; verify reprojection overlays; reject missing observations and missing models in benchmark mode. Confidence: 5/5 for code paths; effect size unmeasured.

### F7 — Major: metric implementation and output definitions are unsuitable for the claimed benchmark

**Evidence:** `eval_metrics.py:31`, `:52`, `:71`; `eval_3dpw.py:220`.

PA-MPJPE flattens frames into one point cloud, fitting one similarity transform across the batch. Two individually perfect similarity-transformed poses produce **1165.90 mm** batched PA-MPJPE but approximately **0.00035 mm** when aligned separately. The ten-frame evaluator calls this per frame, so this particular batching bug does **not** explain its saved value. The reflection correction also fails to adjust the scale numerator’s singular-value signs.

The evaluator uses the wrapper’s 24 kinematic joints without a validated benchmark joint regressor/mapping. Refined vertices are reconstructed without shape or translation; `optimize_frame` exports the body without Gaussian offsets. Consequently, attributing the saved PVE to cloth offsets is not supported by the current evaluation path. `compute_penetration_volume` ignores its covariance argument and sums cubed deficits of vertex-pair distances below 6 cm. It is neither an SDF metric nor the stated Gaussian overlap.

**Minimum repair:** use a published evaluation protocol with explicit body model, joint mapping, camera frame, alignment, gender/shape handling, and per-person frame IDs. Test per-frame similarity invariance and reflections. Confidence: 5/5, reproduced/static.

### F8 — Major: the training loop does not learn the advertised estimator

**Evidence:** `train_colab.py:79`, `:98`, `:102`, `:123`; `eval_3dpw.py:255`; `optimize_single_image.py:49`.

Only the first example of each batch contributes to the step, while FPS counts the configured batch size. Raw/cached dataset pose and translation overwrite one shared parameter tensor with ground truth. One Gaussian attribute set is reused across shuffled actors/frames; Adam moments are also carried across pose resets. There is no image-conditioned predictor or per-subject latent organization that makes this a generalizable HMR training procedure.

The evaluation checkpoint argument is printed but never loaded. Single-image optimization constructs fresh Gaussian parameters. Thus the reported checkpoint is not an evaluated learned component. Checkpoints also omit learned colors and opacity, preventing exact appearance-state restoration.

**Minimum repair:** choose per-image fitting or a genuine learned predictor. For fitting, initialize/reset per sample and count actual processed samples and iterations. For learning, define shared network parameters and per-frame/per-subject state explicitly. Confidence: 5/5, static.

### F9 — Additional bounded engineering findings

- `optimize_frame` retains render tensors in every history entry; detach diagnostic snapshots to avoid retaining unnecessary autograd graph references.
- The photometric loss called SSIM contains only a local-mean similarity term, omitting variance/covariance components; its structural term is unmasked. Name or implement the actual objective. Mesh regularizers use means while proposal equations use sums, so listed weights are not directly transferable.
- Partial DSINE cache injection replaces full arrays, leaves unprocessed rows at defaults, then marks the cache computed. Cache manifests need row-level completion and model/camera provenance.
- Colab orchestration has session reuse and `pipefail`, but `status` errors are swallowed into a string test, many stderr streams bypass `tee`, and there is no explicit numerical-failure watchdog in the training loop. A 20-minute scheduler may exist outside the script; it was not inspected or started. Static infrastructure review only.

These should follow scientific-path repairs; they are not reasons to add more model components.

## 4. Empirical Benchmarks & Limitations

### 4.1 Saved artifacts do not establish the headline gain

`results/3dpw_dsine_hmr2_evaluation.json` contains ten frames, initial MPJPE **31.2597 mm**, refined MPJPE **49.9304 mm**, refined PA-MPJPE **36.3917 mm**, and PVE **276.8238 mm**. The recorded MPJPE increases by **18.6707 mm (59.7%)**. Initial PA-MPJPE is absent. Subtracting a published 42.30 mm baseline from this different experiment is not a paired improvement.

The present source has removed the older ground-truth-noise initializer and now records initial PA-MPJPE, but that does not retroactively validate old artifacts. The prior audit reports historical leakage; this review independently confirms a current missing predictor and a result/source-schema mismatch. Raw execution provenance is needed to establish exactly how historical values were produced.

There is a further document-level inconsistency. The ten rows in `01_diffnorm_master_technical_report.md`, §5, do not reproduce its mean row:

| Metric | Arithmetic mean of displayed rows | Claimed mean |
|---|---:|---:|
| Initial MPJPE | 31.801 | 31.80 |
| Refined MPJPE | 42.986 | 49.93 |
| Refined PA-MPJPE | 29.368 | 36.39 |
| Refined PVE | 226.889 | 276.82 |

This exceeds rounding error. Regenerate tables from versioned per-frame outputs; do not treat the displayed rows or their aggregate as verified observations. This is a provenance failure, not an inference about author intent.

### 4.2 Synthetic RICH/CAPE demonstrations are not dataset evidence

`eval_rich_contact.py` constructs one pose and reports a distance proxy; it does not load RICH or optimize a before/after pair. `eval_cape_clothing.py` directly generates larger noise for "coupled" and smaller noise for "detached" poses; it does not run either optimizer on CAPE. That builds the desired comparison into the inputs. The benchmark JSON’s zero-penetration and clothing-bias conclusions are unsupported.

### 4.3 What was actually verified today

CPU command: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=code pytest -q code/tests`.

Result: **18 passed, 2 skipped, 12 warnings, 3.92 s**. The two DSINE-dependent tests skip when model loading fails. Tests establish selected local behaviors, not successful HMR 2.0 inference, SMPL equivalence, accurate clothing, or physical nonpenetration.

The comparative note claims quadrature over 1,000 random Gaussian pairs. The checked test instead has one isotropic closed-form value comparison and one center-gradient comparison. It does not contain that quadrature experiment.

The separate reproducible CPU counterexamples are in `documents/md/reviews/review_2026_09_17_probes.py`. They diagnose deterministic implementation errors. They are not new HMR accuracy results. No end-to-end runtime or peak GPU memory was measured; the 100 FPS, 2.8 GB and 4.2 ms claims remain unverified. Data throughput, optimization steps per second, and fully refined images per second must be reported separately.

## 5. High-Priority Actionable Repairs / To-Do List

### 5.1 Restore a trustworthy measurement path first

1. Mark existing accuracy/speed/contact claims unverified and regenerate their source tables. Preserve old artifacts as historical records.
2. Integrate official SMPL and real HMR 2.0; make synthetic mode explicit and unavailable to benchmark execution. Preserve predicted shape even if shape refinement is initially disabled.
3. Validate image-to-mesh overlays, actor association, normal signs, camera transforms, joint mapping, and metric invariances. Save sample IDs, input/model hashes, commit, configuration, initial/final predictions, and both sets of metrics.
4. Correct tangent frames, overlap numerics, renderer visibility, and gradient ownership. Re-run the included counterexamples plus reference parity checks.
5. Resolve fitting versus training semantics and checkpoint use before another Colab run.

### 5.2 The smallest experiment that answers the research question

Suggested hypothesis: **with identical image-derived supervision and initialization, constrained Gaussian normal rendering improves body-pose refinement relative to mesh normal rendering, without degrading body shape or increasing failures.** This is a proposal for the next experiment, not an established claim or a change made to your manuscript.

| Condition | Purpose |
|---|---|
| A. Frozen HMR 2.0 prediction | Paired starting point for every sample. |
| B. Mesh normal renderer + identical normal/mask/pose-prior losses | Tests whether normal supervision alone explains the gain. |
| C. Gaussian normal renderer + the same losses, fixed nuisance attributes | Isolates the rendering representation. |
| D. C with free Gaussian attributes and current routing | Measures whether extra freedom helps appearance while harming pose. |

Use one fixed pilot set spanning multiple sequences, pose difficulty, and occlusion; a few hundred person-frame instances is a suggested debugging scale, not an official protocol. Freeze selection before examining improvements. Use a disjoint validation set for loss weights. Report paired MPJPE, PA-MPJPE, body PVE, failure fraction, and runtime per fully refined image. Include per-sequence distributions and uncertainty respecting sequence correlation. Report both equal-iteration and equal-time comparisons. Save visual overlays and side views, including failures.

Normal-estimator confidence is not automatically calibrated for clothing/body mismatch. Test oracle rendered normals separately from estimated normals to distinguish an optimizer failure from a supervision failure. Add a pose prior/trust region and image-derived keypoints as controls if needed, using them consistently across B and C.

**Decision rule:** if B and C are indistinguishable, the evidence supports normal-guided HMR but not a 3DGS-specific accuracy contribution. If C only improves speed, frame the contribution accordingly. If both degrade A, repair or reject the supervision/optimization hypothesis before scaling. If C consistently improves body metrics beyond B, then evaluate collision and clothing mechanisms separately.

### 5.3 Treat collision and clothing as subsequent controlled questions

For collision, first test separated surfaces, correct touching contact, shallow crossings, deep containment, and exactly coincident kernels. Compare fixed Gaussian proxies against a simple capsule or mesh/SDF penalty under matched compute. Measure mesh penetration independently and report contact preservation; do not optimize the sole reported metric and call its zero value physical validation.

For clothing, compare frozen offsets, coupled fitting, one-way detachment, and strict body/appearance parameter ownership on the same observations. Evaluate the body and clothed surface separately. Independent offset vectors attached to posed vertices do not, by themselves, establish canonical garment transport or correct novel-pose behavior.

### 5.4 Literature and writing corrections

The priority is accurate positioning, not collecting more citations. Compare with normal-guided body refinement, mesh rendering, prior Gaussian body tracking, and Gaussian-avatar binding methods. The alternative vertex-attention study is a separate avenue if feedforward clothed reconstruction is the intended goal; profile it after verifying GST’s actual implementation.

Correct the DSINE reference: *Rethinking Inductive Biases for Surface Normal Estimation* is the 2024 work; *Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation* is the separate 2021 work. [Official DSINE repository](https://github.com/baegwangbin/DSINE). *Humans in 4D* is ICCV 2023, not CVPR 2023/2024. [Official paper record](https://arxiv.org/abs/2305.20091).

Replace "confirmed," "eradication," "guaranteed convergence," "physically grounded," and "zero collisions" with the precise measured property or a pending hypothesis. Calling a result preliminary does not repair a wrong baseline, synthetic target geometry, or inconsistent arithmetic. The immediate objective should be one trustworthy paired improvement on a validated body model, followed by proof that 3DGS caused it.
