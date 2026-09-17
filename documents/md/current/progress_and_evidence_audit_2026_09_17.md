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

# Progress and Evidence Audit
## What We Have Built, What the Pilot Shows, and What Is Still Missing

17 September 2026. Scope: updated AGENTS.md and RESEARCH.md, six-way report, current implementation/diffs, saved pilot JSONs and CPU tests. Historical base commit: 426ff83b653e9c2706cf19a7d371eb6cbcfdcde0; inspected working tree contains later uncommitted research changes. **A snapshot commit does not establish the source revision that produced old results.**

## 1. Executive Audit

| Area | Verdict | Root cause |
|---|---|---|
| Research direction | Retain, narrow | Investigate nuisance-aware test-time pose refinement; defer collision and learned training. |
| Six-way arithmetic | Correct | Saved means/worsening rates reproduce from 24 records. |
| “24 authentic sequences” | Incorrect | JSON contains 20 distinct sequence names, not 24. |
| “HMR2/SMPL benchmark” | Not established | Runner selects synthetic body geometry and does not require a genuine HMR model. |
| “State-of-the-art / validated E” | Overclaim | No valid benchmark comparison; E ties D worsening and worsens MPJPE versus A. |
| “Schur spectral controller” | Incorrect description | Implementation uses gradient magnitude, hand-built fields and stiffness heuristics. |
| “Normal gating after keypoints converge” | Incorrect description | F freezes Gaussian capacity on a fixed schedule; normal loss remains active; no keypoint gate. |
| “All audit faults resolved” | Overclaim | Core benchmark, renderer, training and integration defects remain. |
| Software verification | Limited pass | 33 CPU tests pass, 2 skip; tests do not establish model authenticity or efficacy. |

**Bottom line:** meaningful engineering progress exists, but no defensible HMR accuracy improvement is established. The right next action is a fail-closed evaluation and mechanism-validation gate, not a larger training run or camera-ready paper.

## 2. Theoretical & Mathematical Formulations

### What remains valid

For $J_x\in\mathbb R^{m\times75}$, $J_a\in\mathbb R^{m\times q}$, $W\succeq0$ and $\Lambda_a\succ0$:
$$S=J_x^TWJ_x-J_x^TWJ_a(J_a^TWJ_a+\Lambda_a)^{-1}J_a^TWJ_x
\preceq J_x^TWJ_x.$$
This follows from positive semidefiniteness of the subtracted matrix. It is established block-elimination algebra, not something empirically “proven” by a 24-example pilot. The inequality is not necessarily strict. Nuisance freedom may also absorb observation errors, so curvature reduction alone does not establish harmful fitting.

### Required corrections to the historical claims

1. **Heuristic is not the Schur matrix.** The estimator uses squared joint-gradient magnitude as “render curvature,” normalized skinning-weighted cross products as kinematic fields, and graph energy as stiffness. It evaluates neither the rendered-residual Jacobians nor their Schur complement/eigendecomposition. Label it a gradient/stiffness heuristic until comparison against an exact small reference is provided.
2. **Normalization is not orthogonalization.** Let $U$ contain the normalized kinematic fields. Subtracting $U\operatorname{diag}(\gamma)U^Tg$ is not generally an orthogonal projection when columns overlap. For unweighted projection onto their span, the appropriate expression is $U(U^TU)^\dagger U^Tg$; soft weighting needs its own defined operator. The current subtraction can over-remove shared directions.
3. **Gradient scaling is not step damping.** AdamW normalizes first and second moments. With a constant positive scalar multiplier and negligible epsilon, both moments scale so the normalized update can remain almost unchanged. Log actual parameter displacement and isolate extra prior terms before claiming a trust-region effect.
4. **Direct-normal pathways matter.** The equality $J_x=J_aJ_v$ applies only to residual paths explained through the modeled offsets. Here posed normals and covariance also depend on pose; center-only arguments do not prove complete curvature collapse.
5. **Zero nuisance does not force “100% of noise” into pose.** Least-squares fitting responds to the component coupled to the pose Jacobian, not every residual component. Existing records do not identify whether noise, occlusion, camera error or model mismatch caused a change.
6. **No physical collision result.** Gaussian overlap and the current distance-cube metric are surrogate quantities. Neither demonstrates mesh intersection volume or zero self-collision.

The companion proposal supplies the local rotation convention, projection/normal derivatives, reduced gradient and prospective experiment gates.

## 3. Implementation & Algorithmic Reality

### 3.1 What was actually done

- Implemented LBS rest-offset correction and shape-dependent joint regression when an official model is provided.
- Added covariance/normal frame consistency, log-determinant overlap evaluation and per-example PA alignment with reflection correction.
- Removed artificial rasterizer tail floors; added background-normal masking and larger default tile capacity.
- Added triangle-normal renderer, six-condition runner, nuisance heuristic, capacity scheduler and associated tests.
- Added no-grad sensitivity computations and CPU staging in the runner. This supports reduced graph retention; historical peak-memory reduction percentages are **not independently reproduced** here.
- Produced four-/five-/six-way result artifacts and reports. These are historical prototypes, not valid confirmatory HMR experiments.
- This consolidation adds an arithmetic audit, corrected proposal/state/registries, recoverable archives and document navigation. It does not repair the scientific code paths below or run GPU workloads.

### 3.2 Blocking source findings

| Priority | Evidence in current source | Consequence / repair |
|---|---|---|
| P0 | pilot_4way_experiment.py constructs SMPLWrapper(device=device), without model_path; wrapper selects the random synthetic humanoid. | Both predicted and “GT” geometry are synthetic reconstructions. Require official model and parity against reference SMPL, including pose blend shapes. |
| P0 | Pilot calls CoarsePoseHMR2(auto_download=False) without require_model=True. Source string/checkpoint identity is not saved. | Genuine image-dependent HMR initialization is unproven. Make benchmark execution fail closed and verify official preprocessing/inference. |
| P0 | Dataset returns campose but pilot never uses it; full-frame images are reused for every actor. Actor ID is not emitted. | Camera and person correspondence are not established. Apply verified transforms and matched person crops. |
| P0 | A/B/C use GT beta; D/E/F optimize without beta and evaluate using GT beta. | Oracle leakage and optimization/evaluation shape mismatch. Use the same fixed predicted beta everywhere. |
| P0 | Target mask is a dilation of the initializer render. | It is a prior-derived mask, not an independent foreground observation. Label it and add observed segmentation/keypoints when claiming those constraints. |
| P1 | B/C use mask MSE plus pose prior; D/E/F use IoU, collision, photometric and deformation losses with differing prior behavior. | Representation comparisons are confounded; match objectives before attributing effects to nuisance freedom. |
| P1 | Mesh mask is inside.any(...).float(); tile cap is 64. Gaussian pilot cap is 96, despite a default of 256. | Mesh silhouette lacks a useful continuous coverage gradient; both caps can drop visible geometry. Test convergence and use a trustworthy mesh control. |
| P1 | Router detaches covariance during geometric rendering; deformation centers omit R_cam while normals/covariance can use it. | Modified gradients and potential frame inconsistency must be explicit; these are not the full joint residual derivatives. |
| P1 | Gaussian _resolve_mesh_normals inspects caller stack local variables. | Behavior can depend on unrelated caller names; passing a probe does not prove the explicit API works. Remove introspection and pass normals explicitly. |
| P1 | eval_3dpw.py references undefined sample when assigning sample_betas; uses betas instead of dataset beta. | The test-time path can fail at runtime; tests currently miss this branch. |
| P1 | train_colab.py still takes the first sample, copies GT theta/trans into shared parameters, and multiplies throughput by batch size. | Earlier claim that training semantics were repaired is false. Do not interpret this as training a general HMR model. |
| P1 | eval_3dpw.py checks/prints checkpoint presence without loading its learned state. | “Checkpoint evaluation” does not establish checkpoint impact. |

Relevant sources: [pilot runner](../../../code/scripts/pilot_4way_experiment.py), [SMPL](../../../code/src/geometry/smpl_wrapper.py), [HMR wrapper](../../../code/src/pipeline/coarse_pose_hmr2.py), [dataset](../../../code/src/pipeline/dataset_3dpw.py), [controller](../../../code/src/optimization/nuisance_sensitivity.py), [router](../../../code/src/optimization/dual_frequency_router.py), [scheduler](../../../code/src/optimization/capacity_scheduler.py), [mesh renderer](../../../code/src/rendering/mesh_normal_rasterizer.py), [evaluation](../../../code/scripts/eval_3dpw.py), [training](../../../code/scripts/train_colab.py).

### 3.3 F is not the hypothesis recorded in the old state file

F performs fixed-step Gaussian capacity locking, then bounded detail release and a kinematic LR reduction. It does not inspect 2D keypoint convergence or disable the early normal loss. Its combined changes cannot isolate whether capacity, bounds or LR caused any effect. The old H003 wording has been corrected, not silently treated as a successful test of a different hypothesis.

## 4. Empirical Benchmarks & Limitations

### 4.1 Recomputed six-way artifact numbers

These are **saved prototype artifact values, not official HMR benchmark results**. Lower errors are better. Worsening uses the runner's PA delta >0.1 mm threshold.

| Condition | MPJPE mm | PA-MPJPE mm | PA delta vs A mm | Worsening |
|---|---:|---:|---:|---:|
| A: initialization | 336.04 | 215.44 | 0.00 | 0/24 |
| B: mesh | 367.64 | 220.67 | +5.24 | 12/24 (50.0%) |
| C: fixed Gaussian | 356.41 | 221.30 | +5.87 | 15/24 (62.5%) |
| D: adaptive Gaussian | 352.08 | 215.90 | +0.46 | 8/24 (33.3%) |
| E: heuristic control | 353.60 | 214.93 | -0.50 | 8/24 (33.3%) |
| F: capacity schedule | 349.23 | 217.02 | +1.58 | 11/24 (45.8%) |

The report's -0.51 mm arose from subtracting rounded means. The unrounded E–A PA difference is **-0.502807 mm**; E–D is **-0.964226 mm**. E is not better than D on saved worsening frequency and increases MPJPE by **17.561149 mm** versus A. F has the lowest MPJPE among refined B–F, but still worsens A by **13.193390 mm**. Calling F the overall best without A is misleading.

Saved raw PVE means are A=3925.72, B=3945.96, C=3786.59, D=3776.26, E=3779.48, F=3816.13 mm. These metre-scale values, raw translation-sensitive metric and unverified camera frame make a body-surface accuracy interpretation unsafe. Do not conceal them while reporting PA alone.

### 4.2 Experiment ledger and provenance

| Retrospective ID | Artifact / scope | Provenance and verdict |
|---|---|---|
| LEGACY-P4 | pilot_4way_experiment_results.json: 2 records, 1 sequence name. | Original commit, command, full config, log and model hashes absent. Inconclusive. |
| LEGACY-P5 | pilot_5way_experiment_results.json: 24 records, 19 sequence names. | Same missing provenance. Different sample identities from six-way run; not a paired before/after controller comparison. Inconclusive. |
| LEGACY-P6 | pilot_6way_experiment_results.json: 24 records, 20 sequence names, 15 iterations, device=cuda. | Aggregate consistency passes; historical model/run identity absent. Inconclusive for scientific hypotheses. |
| DOC-AUDIT-20260917 | CPU recomputation of P4/P5/P6 plus current source hashes. | Base commit 426ff83b; dirty tree recorded. Arithmetic supported; benchmark validity not established. |

The three LEGACY IDs are assigned retrospectively for tracking, not recovered original run identifiers. Original files remain unchanged. Raw six-way elapsed time is 148.584 s, but the runner starts timing after neural pre-extraction; it is not end-to-end throughput and does not establish equal runtime per condition.

Reproduction command:
**python code/scripts/audit_pilot_artifact.py --out results/DOC-AUDIT-20260917/result.json**

The [audit JSON](../../../results/DOC-AUDIT-20260917/result.json) records source-artifact SHA-256 values, reconstructed means/deltas, sequence counts, missing historical provenance and hashes of the **current** Python sources. Current hashes are not retroactive run provenance. [Verification transcript](../../../results/DOC-AUDIT-20260917/verification.txt) records this consolidation's checks.

### 4.3 Tests and limits of the audit

CPU command: **CUDA_VISIBLE_DEVICES='' PYTHONPATH=code pytest code/tests/ -q**.
Observed: **33 passed, 2 skipped, 12 warnings in 4.40 s** on the first check of this turn. DSINE-dependent tests skip in this environment. The final transcript may record a different duration.

Tests cover limited invariants and integration, not authentic HMR2 inference, official SMPL parity, real camera alignment, accurate occlusion or improvement on held-out images. In particular, controller tests check shapes/bounds and constructed collinearity, not agreement with a rendered-residual Schur complement. No GPU experiment, memory benchmark, formal confidence interval or statistical significance claim was generated here.

## 5. High-Priority Actionable Repairs / To-Do List

1. **V001 before scale:** official geometry; authentic HMR initialization; fail-closed images/assets; camera/actor/crop correspondence; consistent predicted shape; independent observations and verified metrics.
2. **Clean controls:** identical priors/losses for causal comparisons; correct silhouette gradients; visibility/candidate-cap tests; remove caller-stack behavior; add smoke coverage for the evaluator.
3. **M001 before naming the method:** exact tiny Jacobian/Schur reference, finite-difference rotation checks, proper projection operator and actual optimizer-step measurements. Keep the current E as a heuristic baseline.
4. **P001 before conclusions:** fixed manifest, matched A–G conditions, logs and provenance, residual trajectories, all 3D metrics, worsening/coverage and sequence-aware uncertainty.
5. **P002 before a publication claim:** untouched confirmation sequences, meaningful practical effect, robust controls and full-method novelty comparison. Collision and learned training remain separate later projects.

**Archive decisions.** Old proposals, notes, pilot claims, broad remediation/framework reports and earlier exploratory plans moved under documents/archive/md and documents/archive/pdf. Original bytes and old root-state snapshots were preserved; a machine-readable relocation manifest explains destinations. The canonical mathematical writing-standard source remains where AGENTS.md references it. Papers remain under documents/pdf/papers; probe Python code moved into code/scripts. Root state files and notes/ remain deliberate operational exceptions.

**Assistant use and final judgment.** The requested Gemini model was invoked through agy. Its initial headless repository read was denied; a subsequent text-only inventory completed without file edits. It spotted layout anomalies but could not establish document validity. The lead rejected suggestions to retain overclaimed reports as active and independently inspected current code and JSONs. No scientific judgment was delegated.

**Self-review and claim map.** Components exist—source-supported; saved arithmetic matches—recomputed; genuine benchmark gain—unsupported; controller is exact Schur—refuted by implementation; proposed mechanism is worth testing—judgment, with explicit falsification gates. Unresolved issues are documented rather than presented as repaired.
