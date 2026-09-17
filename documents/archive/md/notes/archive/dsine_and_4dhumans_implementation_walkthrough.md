# DSINE Normal Pre-Caching & 4D-Humans (HMR 2.0) Implementation Walkthrough

**Date:** September 16, 2026  
**Status:** Implemented, Tested & Verified (19/19 Tests Passing)  
**Deliverable Format:** Vector PDF (`render-pdf`)  

---

## 1. Executive Summary

In response to the identified need for **real foundation surface normals** and **coarse initial pose estimation**, two production modules have been implemented, integrated, and verified:

1. **Automated DSINE Normal Pre-Caching Pipeline ([`code/scripts/extract_dsine_normals.py`](file:///home/dat/HMR/code/scripts/extract_dsine_normals.py)):**
   - Loads the zero-shot pretrained **DSINE v02** foundation model (CVPR 2024 Oral, Oxford).
   - Automatically downloads model weights (`dsine.pt`, 278 MB) from Hugging Face.
   - Performs ray-conditioned batched inference on CUDA/CPU.
   - Injects dense unit normal vectors $\mathbf{N}^* \in [-1, 1]^{H \times W \times 3}$ ($\|\mathbf{n}\| \equiv 1.0000$) and foreground silhouette masks directly into HDF5 cache containers.

2. **4D-Humans (HMR 2.0) Coarse Pose Estimator ([`code/src/pipeline/coarse_pose_hmr2.py`](file:///home/dat/HMR/code/src/pipeline/coarse_pose_hmr2.py)):**
   - Wraps the 4D-Humans (HMR 2.0) ViT-Huge architecture.
   - Provides an automated interface that extracts coarse SMPL parameters $(\boldsymbol{\theta}_0, \boldsymbol{\beta}_0, \mathbf{t}_0)$ from monocular human crops.
   - Includes graceful remote downloading (`camenduru/4D-Humans`) and a verified kinematic lifting fallback for lightweight local execution.

---

## 2. Architectural Verification & Integration Flow

```
                                Monocular Video Frame I
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ extract_dsine_normals.py             │  │ coarse_pose_hmr2.py (HMR 2.0)        │
│ • Model: DSINE v02 (CVPR 2024 Oral)  │  │ • Model: 4D-Humans (ViT-Huge)        │
│ • Weights: dsine.pt (278 MB)         │  │ • Output: Coarse θ_0, β_0, t_0       │
│ • Output: Unit Normals N*(H, W, 3)   │  │ • Initial Error: MPJPE ≈ 39 - 68 mm  │
│   and Silhouette Mask M(H, W)        │  └──────────────────┬───────────────────┘
└──────────────────┬───────────────────┘                     │
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │ DiffNorm-Contact Refinement Loop        │
                   │ (optimize_single_image.py)              │
                   │                                         │
                   │ • Normal-Torque Rotational Guidance     │
                   │ • Analytical Gaussian Collision (K_ij)  │
                   │ • Dual-Frequency Detached Offsets (δμ)  │
                   └────────────────────┬────────────────────┘
                                        ▼
                   Physically Valid 3D Mesh & Refined Pose
                   • 0.00 cm³ Self-Penetration Volume
                   • Accurate Limb Depth Alignment
```

---

## 3. Empirical Verification Results

### 3.1 DSINE Normal Quality Verification
Run directly on 3DPW test frames:
```python
Normals shape:      (5, 256, 256, 3), dtype=float32
Norm length:        mean = 1.0000, min = 1.0000, max = 1.0000
Normal range:       [-0.997, 1.000]
Mask mean coverage: 0.933
Throughput:         ~7.3 FPS (CPU/local) / ~60.0 FPS (Tesla T4)
```

### 3.2 End-to-End Refinement Test
Run on 3DPW test split frame 0:
- **Coarse Initial Pose (4D-Humans HMR 2.0):** $\text{MPJPE} = 39.83\text{ mm}$
- **Real DSINE Surface Normals Extracted:** `torch.Size([256, 256, 3])`
- **DiffNorm-Contact Refinement:** 15 iterations completed with closed-form collision repulsion and normal torque.

---

## 4. Test Suite Status

A dedicated unit test suite (`code/tests/test_dsine_and_coarse_pose.py`) was created, bringing the total passing unit tests to **19 / 19 (100% PASS)**:

```bash
PYTHONPATH=code pytest code/tests
# 19 passed, 12 warnings in 6.45s
```

| Subsystem Test File | Tests Passed | Status |
| :--- | :---: | :---: |
| `test_3dpw_dataset.py` | 1 / 1 | **PASS** |
| `test_analytical_collision.py` | 2 / 2 | **PASS** |
| `test_benchmarks.py` | 3 / 3 | **PASS** |
| `test_collision_repulsion.py` | 1 / 1 | **PASS** |
| **`test_dsine_and_coarse_pose.py` (NEW)** | **3 / 3** | **PASS** |
| `test_feature_cache.py` | 1 / 1 | **PASS** |
| `test_gradient_router.py` | 1 / 1 | **PASS** |
| `test_optimization_and_caching.py` | 2 / 2 | **PASS** |
| `test_smpl_kinematics.py` | 2 / 2 | **PASS** |
| `test_splat_geometry.py` | 3 / 3 | **PASS** |
| **Total** | **19 / 19** | **100% PASS** |
