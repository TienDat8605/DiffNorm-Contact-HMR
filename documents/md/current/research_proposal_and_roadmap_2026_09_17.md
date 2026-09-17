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

# Current Research Proposal and Gated Roadmap
## Nuisance-Aware Test-Time Human Pose Refinement with Surface-Bound Gaussians

17 September 2026. Replaces earlier broad DiffNorm-Contact proposals. **Status: meaningful, falsifiable research direction; not a validated method.** The companion progress audit details implementation defects and historical results.

## 1. Executive Audit

| Area | Verdict | Root cause / decision |
|---|---|---|
| Research question | Viable candidate | Rendering quality and skeletal accuracy can disagree; attribute residuals before moving the body. |
| Novelty | Incomplete | Joint Gaussian/HMR fitting and nuisance elimination have precedent. Novelty must lie in useful articulated sensitivity and control. |
| Current controller | Incomplete | Gradient/stiffness heuristic; no rendered-residual Schur matrix is computed. |
| Six-way pilot | Inconclusive | Synthetic body, unverified initializer, unmatched objectives and missing run provenance. |
| Immediate priority | Evaluation repair | Establish trustworthy geometry and initialization before testing the mechanism. |
| Multi-epoch training | Premature | First contribution is per-image refinement; current training does not establish a learned cross-person model. |

**Proposal in one paragraph.** Given a person image, a genuine HMR initializer and estimated surface normals, refine pose using a differentiable surface-bound Gaussian representation. Test whether overlap between pose-induced and Gaussian-induced image changes can identify unreliable skeletal updates. Use that diagnostic to control the actual pose step or nuisance flexibility. Compare against matched mesh refinement, joint Gaussian fitting, static detachment and generic damping. Success means improved joint/mesh accuracy and fewer harmful updates—not merely lower image loss.

**Primary question:** Does nuisance-aware control improve single-image pose refinement beyond equally conservative conventional optimization at matched observations, priors and computation?

**Scope.** Single person, single image; official SMPL topology; 24 joint rotations and root translation; fixed *predicted* shape and declared camera intrinsics. Ground-truth shape is reserved for a labeled oracle ablation. Video, trainable regressors, collision/contact and detailed clothing reconstruction are outside the first contribution.

## 2. Theoretical & Mathematical Formulations

### 2.1 Local model and established result

Let $R_k\in SO(3)$, $k=1,\ldots,24$, and $t\in\mathbb R^3$ denote pose and translation. Local increments $\delta x\in\mathbb R^{75}$ give
$R'_k=\exp([\delta\omega_k]_\times)R_k$, $t'=t+\delta t$.
Fix predicted shape $\beta_0\in\mathbb R^{10}$. Let $a\in\mathbb R^q$ collect selected nuisance coordinates; quaternion orientation, if included, requires a constrained/local parameterization.

For $m$ residual components, $r\in\mathbb R^m$, $J_x\in\mathbb R^{m\times75}$ and $J_a\in\mathbb R^{m\times q}$:
$$r(x\oplus\delta x,a+\delta a)\simeq r+J_x\delta x+J_a\delta a.$$
With locally fixed $W\succeq0$ and nuisance increment damping $\Lambda_a\succ0$, eliminate $\delta a$ from
$$\tfrac12\|r+J_x\delta x+J_a\delta a\|_W^2+\tfrac12\delta a^T\Lambda_a\delta a.$$
Writing $H_a=J_a^TWJ_a+\Lambda_a$, the reduced data curvature and gradient are
$$S=J_x^TWJ_x-J_x^TWJ_aH_a^{-1}J_a^TWJ_x,$$
$$g_{\rm red}=J_x^TWr-J_x^TWJ_aH_a^{-1}J_a^TWr.$$
A damped descent step solves $(S+\Lambda_x)\delta x=-g_{\rm red}$ before adding any nonzero pose-prior gradient consistently. This simplified model uses increment damping; a nonzero nuisance-prior gradient must also be included for a general prior.

For $v\in\mathbb R^{75}$, setting $u=J_a^TWJ_xv$ gives
$$v^T(J_x^TWJ_x-S)v=u^TH_a^{-1}u\ge0.$$
Hence $S\preceq J_x^TWJ_x$. This is a **non-strict local curvature inequality**, not a novel theorem, calibrated confidence, or guarantee of worse accuracy. Equality can occur in uncoupled directions. Merely assuming $\Lambda_a\succeq0$ does not ensure the ordinary inverse exists.

At vanishing damping, only pose effects in the nuisance image-Jacobian span can disappear. Complete loss of pose information is not automatic: pose-dependent normals and covariance can provide paths not reproduced by center offsets.

### 2.2 Derivative and optimizer requirements

Use a dimensionless parameter metric before comparing rotation and translation sensitivities. For camera coordinates $(X,Y,Z)$, $Z>0$, the perspective Jacobian is
$$D\pi=\begin{bmatrix}f_x/Z&0&-f_xX/Z^2\\0&f_y/Z&-f_yY/Z^2\end{bmatrix}.$$
For accumulated normal $n$, $\|n\|>\epsilon$,
$$D(n/\|n\|)=\frac{I-\hat n\hat n^T}{\|n\|},\qquad
\nabla_n(1-\hat n^Tn^*)=-\frac{I-\hat n\hat n^T}{\|n\|}n^*.$$
Include projection, splat weights, covariance and normal normalization in the chain rule. The renderer is piecewise smooth across culling/threshold changes; test smooth-region derivatives and boundary behavior separately.

The current field $w_{ik}(\omega_k\times(v_i-j_k))$ is a heuristic, not a verified finite-angle SMPL derivative: it omits complete hierarchical rotation transport and descendant contributions. Use actual local-rotation JVPs and finite differences.

A Schur solve alone is equivalent to solving the same joint local system; it is a baseline, not sufficient novelty. Scaling an Adam gradient is also not equivalent to scaling its parameter step: moment normalization can cancel positive scaling. Control or log actual displacement and moment behavior.

### 2.3 Candidate contribution

First show that a cheap diagnostic approximates a small exact reference and predicts harmful updates better than gradient magnitude, normal confidence and residual size. Then design step/capacity control around it. Start with fixed splat count and low-dimensional bounded nuisance bases.

If equally effective with a mesh nuisance representation, frame the result as general inverse-rendering HMR optimization. If a stronger prior explains the improvement, do not attribute it to nuisance awareness.

## 3. Implementation & Algorithmic Reality

**Present:** six-condition runner; triangle/Gaussian renderers; detach-based appearance routing; heuristic sensitivity filter; fixed-iteration capacity release; partial LBS, covariance, overlap and PA repairs.

**Not validated:** authentic fail-closed HMR2/SMPL evaluation, observation transforms, matched objectives, rendered-Jacobian sensitivity, actual step damping and held-out performance.

Gemini implements bounded repairs/tests against lead-defined acceptance criteria. Codex checks diffs, equations, provenance and final results. Worker summaries are not evidence.

Scheduler F freezes Gaussian deformation at iterations 0–7, releases it at 8–14, ramps deformation LR, bounds offsets to 0.015 m and reduces kinematic LR. It does **not** wait for keypoint convergence or disable normal supervision. Release timing, bounds and LR changes need separate ablations.

## 4. Empirical Benchmarks & Limitations

| Proposed control | Purpose |
|---|---|
| A: frozen genuine initializer | Paired reference, fixed predicted shape. |
| B: mesh normal refinement | Tests whether normals alone explain improvement. |
| C: fixed Gaussian refinement | Representation without adaptive nuisance. |
| D: free/bounded Gaussian fitting | Same objectives, prior and optimizer as E. |
| E: sensitivity control | Changes only proposed control relative to D. |
| F: capacity scheduling | Separate timing from LR and bounds. |
| G: static detachment / generic damping | Tests ordinary conservative optimization. |

Equal iterations do not mean equal compute. Report fixed-step and matched-wall-time comparisons, normal/mask losses, actual pose displacement and update coverage. Disable collision in the initial matched mechanism comparison.

**Prospective H004 target:** at least **2 mm mean PA-MPJPE gain over A**, at least **5 percentage-point lower worsening than tuned D**, and no more than **1 mm mean MPJPE regression versus A**. Require sequence-clustered 95% intervals excluding zero for the two claimed improvements and incremental benefit over tuned G at matched coverage. These are proposed practical thresholds—not achieved scores or a power calculation.

Define $\Delta_i=M_i^{\rm refined}-M_i^{\rm init}$ and worsening as PA $\Delta_i>0.1$ mm. Save MPJPE, PA-MPJPE, PVE with declared alignment/joint conventions, tail error, coverage, runtime and peak GPU allocation. Physical penetration is **not applicable** until independently validated mesh-intersection evaluation exists.

A valid, adequately precise experiment ruling out the target gain contradicts the practical hypothesis. Wide intervals are inconclusive. Missing models, camera bugs, synthetic substitutions, crashes and unmatched controls invalidate the experiment, not the hypothesis.

Use a frozen sequence/actor/frame manifest. Tune on validation subjects/sequences; historical test examples already inspected are development data, not untouched confirmation. Cluster inference by sequence. Choose confirmation size from valid pilot variance; an arbitrary 4,800-frame cache is not an official split.

## 5. High-Priority Actionable Repairs / To-Do List

| Gate | Work and completion evidence | Status |
|---|---|---|
| V001: evaluation | Official body-model parity; mandatory real HMR; image-dependent predictions; correct person crop/camera transform; predicted beta consistent in all branches; fail on missing assets. | Pending. |
| V001: numerical validity | Known-pose overlays; official joint mapping; batched PA; near-zero/near-pi rotations; tile-cap convergence; mesh silhouette gradients; remove caller-frame introspection. | Pending. |
| M001: diagnostic | Tiny exact Jacobian/Schur reference; finite differences; candidate-reference agreement; Gram-correct projection if used; actual Adam step behavior. | CPU-first, pending. |
| M001: perturbation test | Known pose error versus clothing/normal noise; locked seed; nuisance-capacity sweep; save image and 3D errors each step. | Synthetic sanity test, pending. |
| P001: paired real-image pilot | Frozen manifest; genuine A–G; matched losses/priors/shape; no GT inputs except evaluation; full telemetry. | Blocked on V001/M001. |
| P002: confirmation | Untouched sequences; cluster intervals; multi-seed robustness; efficiency and damping controls. | Conditional on meaningful P001. |
| C001: contact | Physical mesh-intersection/contact metrics and penalty ablations. | Deferred. |
| Learned training / paper | Define a genuine learned model only if needed; claims follow valid results and novelty review. | Not ready. |

Each run must store experiment ID, Git commit plus dirty patch/hash, frozen config/sample manifest, model/data hashes, seed, command, log location, environment, per-example metrics, failures, verdict and next action under results/&lt;exp_id&gt;/. Future authorized GPU runs follow the Colab orchestration/session rules. No GPU experiment was run for this document.

**Self-review:** Contribution—candidate; clarity—theory separated from code; empirical strength—needs valid data; evaluation—controls planned, not executed; soundness—derivative/optimizer questions unresolved. Claim-evidence map: components exist (source-supported); benchmark gain established (unsupported, removed); direction worth testing (research judgment).

### Prior-work boundary and disclosure

[HumanSplatHMR](https://arxiv.org/abs/2605.02784) is a pose/avatar optimization precedent; [GST](https://arxiv.org/abs/2409.04196) connects Gaussian rendering and human-body prediction. [Nuisance elimination](https://arxiv.org/abs/1206.6532) is established. [UGS-Loc](https://arxiv.org/abs/2603.16538) addresses Gaussian camera-pose/geometric uncertainty. [DSINE](https://github.com/baegwangbin/DSINE) and [4D-Humans](https://shubham-goel.github.io/4dhumans/) are intended upstream sources. Full-method novelty comparison remains pending. AI-assisted drafting and a bounded Gemini inventory were used; scientific judgments and verification remained with the lead.
