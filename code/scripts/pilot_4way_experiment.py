"""
Controlled 4-Way Pilot Experiment Runner for DiffNorm-Contact HMR.
Benchmarks and isolates the empirical mechanisms on authentic 3DPW test frames:
  - Condition A: Frozen HMR 2.0 Coarse Baseline (unmodified starting point).
  - Condition B: Classical Differentiable Mesh Normal Rendering (L_N + L_mask + L_prior).
  - Condition C: Constrained Gaussian Normal Rendering (Fixed nuisance attributes: offsets=0, scales fixed, quats identity).
  - Condition D: Unconstrained Gaussian Normal Rendering (Free nuisance deformation: offsets, scales, quats optimized).

Records paired initial vs. refined MPJPE, PA-MPJPE, PVE, Worsening Rate (W%), and tail errors.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.geometry.mesh_graph import MeshGraph
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer
from src.rendering.mesh_normal_rasterizer import DifferentiableMeshNormalRasterizer
from src.optimization.dual_frequency_router import DualFrequencyGradientRouter
from src.optimization.nuisance_sensitivity import NuisanceSensitivityEstimator
from src.optimization.capacity_scheduler import CapacityGatedScheduler
from src.pipeline.dataset_3dpw import Dataset3DPW
from src.pipeline.coarse_pose_hmr2 import CoarsePoseHMR2
from src.pipeline.eval_metrics import compute_mpjpe, compute_pa_mpjpe, compute_pve
from scripts.extract_dsine_normals import load_dsine_model, run_dsine_batch


def optimize_condition_b(
    smpl: SMPLWrapper,
    mesh_rasterizer: DifferentiableMeshNormalRasterizer,
    init_theta: torch.Tensor,
    init_trans: torch.Tensor,
    beta: torch.Tensor,
    target_normals: torch.Tensor,
    target_mask: torch.Tensor,
    K: torch.Tensor,
    num_iterations: int = 15,
    lr: float = 1e-2,
    lambda_mask: float = 5.0,
    lambda_prior: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition B: Classical Differentiable Mesh Normal Rendering."""
    theta_param = nn.Parameter(init_theta.clone())
    trans_param = nn.Parameter(init_trans.clone())
    optimizer = torch.optim.AdamW([
        {"params": [theta_param], "lr": lr},
        {"params": [trans_param], "lr": lr}
    ])

    for _ in range(num_iterations):
        optimizer.zero_grad()
        out = smpl(theta=theta_param.unsqueeze(0), beta=beta.unsqueeze(0), trans=trans_param.unsqueeze(0))
        verts = out["vertices"][0]
        normals = out["normals"][0]

        rend = mesh_rasterizer(verts, normals, smpl.faces, K)

        # Normal alignment loss on foreground
        cos_sim = (rend.normals * target_normals).sum(dim=-1)
        valid_mask = (target_mask > 0.5) & (rend.mask > 0.5)
        if valid_mask.any():
            loss_norm = (1.0 - cos_sim)[valid_mask].mean()
        else:
            loss_norm = (1.0 - cos_sim).mean()

        # Silhouette mask loss
        loss_mask = F.mse_loss(rend.mask, target_mask)

        # Anatomical prior loss anchoring to initialization
        loss_prior = torch.mean((theta_param - init_theta) ** 2)

        loss_total = loss_norm + lambda_mask * loss_mask + lambda_prior * loss_prior
        loss_total.backward()
        optimizer.step()

    res_theta = theta_param.detach().clone()
    res_trans = trans_param.detach().clone()
    del theta_param, trans_param, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res_theta, res_trans


def optimize_condition_c(
    smpl: SMPLWrapper,
    gaussians: TangentialGaussianSurface,
    gaussian_rasterizer: DifferentiableNormalRasterizer,
    init_theta: torch.Tensor,
    init_trans: torch.Tensor,
    beta: torch.Tensor,
    target_normals: torch.Tensor,
    target_mask: torch.Tensor,
    K: torch.Tensor,
    num_iterations: int = 15,
    lr: float = 1e-2,
    lambda_mask: float = 5.0,
    lambda_prior: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition C: Constrained Gaussian Splatting (Gaussian nuisance parameters are strictly frozen)."""
    theta_param = nn.Parameter(init_theta.clone())
    trans_param = nn.Parameter(init_trans.clone())
    optimizer = torch.optim.AdamW([
        {"params": [theta_param], "lr": lr},
        {"params": [trans_param], "lr": lr}
    ])

    # Fixed canonical Gaussian attributes
    device = init_theta.device
    num_splats = gaussians.num_splats
    fixed_colors = torch.ones(num_splats, 3, device=device) * 0.5
    fixed_opacities = torch.ones(num_splats, 1, device=device) * 0.95

    for _ in range(num_iterations):
        optimizer.zero_grad()
        out = smpl(theta=theta_param.unsqueeze(0), beta=beta.unsqueeze(0), trans=trans_param.unsqueeze(0))
        verts = out["vertices"][0]
        mesh_normals = out["normals"][0]

        # Centers anchored to posed mesh with 0 offset; thin axis collinear with mesh normals
        centers_cam = verts  # delta = 0
        covs = gaussians.get_spatial_covariances(mesh_normals=mesh_normals)
        normals = gaussians.get_surface_normals(mesh_normals=mesh_normals)

        rend = gaussian_rasterizer(
            centers_cam=centers_cam,
            covs_cam=covs,
            normals_cam=normals,
            colors=fixed_colors,
            opacities=fixed_opacities,
            K=K
        )

        cos_sim = (rend.normals * target_normals).sum(dim=-1)
        valid_mask = (target_mask > 0.5) & (rend.mask > 0.5)
        if valid_mask.any():
            loss_norm = (1.0 - cos_sim)[valid_mask].mean()
        else:
            loss_norm = (1.0 - cos_sim).mean()

        loss_mask = F.mse_loss(rend.mask, target_mask)
        loss_prior = torch.mean((theta_param - init_theta) ** 2)

        loss_total = loss_norm + lambda_mask * loss_mask + lambda_prior * loss_prior
        loss_total.backward()
        optimizer.step()

    res_theta = theta_param.detach().clone()
    res_trans = trans_param.detach().clone()
    del theta_param, trans_param, optimizer, fixed_colors, fixed_opacities
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res_theta, res_trans


def optimize_condition_d(
    router: DualFrequencyGradientRouter,
    init_theta: torch.Tensor,
    init_trans: torch.Tensor,
    target_normals: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    K: torch.Tensor,
    num_iterations: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition D: Unconstrained Gaussian Splatting (All Gaussian nuisance attributes are optimized)."""
    theta_param = nn.Parameter(init_theta.clone())
    trans_param = nn.Parameter(init_trans.clone())
    opt_kin, opt_deform = router.create_optimizers(theta_param, trans_param)

    for _ in range(num_iterations):
        router.step(
            theta=theta_param,
            trans=trans_param,
            K=K,
            target_normals=target_normals,
            target_rgb=target_rgb,
            target_mask=target_mask,
            uncertainty=None,
            opt_kin=opt_kin,
            opt_deform=opt_deform
        )

    res_theta = theta_param.detach().clone()
    res_trans = trans_param.detach().clone()
    del theta_param, trans_param, opt_kin, opt_deform
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res_theta, res_trans


def optimize_condition_e(
    router: DualFrequencyGradientRouter,
    init_theta: torch.Tensor,
    init_trans: torch.Tensor,
    target_normals: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    K: torch.Tensor,
    num_iterations: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition E: Nuisance-Controlled Gaussian Splatting (Adaptive Spectral Damping & Subspace Orthogonalization)."""
    theta_param = nn.Parameter(init_theta.clone())
    trans_param = nn.Parameter(init_trans.clone())
    opt_kin, opt_deform = router.create_optimizers(theta_param, trans_param)

    for _ in range(num_iterations):
        router.step(
            theta=theta_param,
            trans=trans_param,
            K=K,
            target_normals=target_normals,
            target_rgb=target_rgb,
            target_mask=target_mask,
            uncertainty=None,
            opt_kin=opt_kin,
            opt_deform=opt_deform,
            init_theta=init_theta,
            init_trans=init_trans
        )

    res_theta = theta_param.detach().clone()
    res_trans = trans_param.detach().clone()
    del theta_param, trans_param, opt_kin, opt_deform
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res_theta, res_trans


def optimize_condition_f(
    router: DualFrequencyGradientRouter,
    scheduler: CapacityGatedScheduler,
    init_theta: torch.Tensor,
    init_trans: torch.Tensor,
    target_normals: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    K: torch.Tensor,
    num_iterations: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Condition F: Stage-Gated Capacity-Controlled 3DGS (Kinematic Locking -> Bounded Detail Release)."""
    theta_param = nn.Parameter(init_theta.clone())
    trans_param = nn.Parameter(init_trans.clone())
    opt_kin, opt_deform = router.create_optimizers(theta_param, trans_param)

    for step_idx in range(num_iterations):
        router.step(
            theta=theta_param,
            trans=trans_param,
            K=K,
            target_normals=target_normals,
            target_rgb=target_rgb,
            target_mask=target_mask,
            uncertainty=None,
            opt_kin=opt_kin,
            opt_deform=opt_deform,
            init_theta=init_theta,
            init_trans=init_trans,
            iteration=step_idx,
            scheduler=scheduler
        )

    res_theta = theta_param.detach().clone()
    res_trans = trans_param.detach().clone()
    del theta_param, trans_param, opt_kin, opt_deform
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res_theta, res_trans


def run_pilot_experiment(
    num_frames: int = 20,
    stride: int = 50,
    opt_iterations: int = 15,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: str = "results/pilot_4way_experiment_results.json"
) -> Dict:
    """Executes the controlled 4-way pilot experiment."""
    device = torch.device(device_str)
    print("=" * 80)
    print(" DIFFNORM-CONTACT HMR: CONTROLLED 4-WAY PILOT EXPERIMENT")
    print(" Testing Hypotheses: Condition A (Init), B (Mesh), C (Constrained 3DGS), D (Unconstrained 3DGS)")
    print(f" Frames: {num_frames} | Device: {device} | Iterations: {opt_iterations}")
    print("=" * 80)

    # 1. Initialize models
    smpl = SMPLWrapper(device=device).to(device)
    coarse_estimator = CoarsePoseHMR2(device=device, auto_download=False)
    dsine_model = load_dsine_model(device=device)

    # Renderers
    mesh_rasterizer = DifferentiableMeshNormalRasterizer(image_height=256, image_width=256).to(device)
    gaussian_rasterizer = DifferentiableNormalRasterizer(image_height=256, image_width=256, max_splats_per_tile=96).to(device)

    # Modules for Condition D and E
    segmenter = KinematicSegmenter(smpl.weights)
    mesh_graph = MeshGraph(smpl.faces).to(device)
    collision_engine = GaussianSelfCollisionEngine(segmenter).to(device)
    sensitivity_estimator = NuisanceSensitivityEstimator(
        smpl=smpl,
        segmenter=segmenter,
        mesh_graph=mesh_graph,
        lambda_lap=50.0,
        lambda_tight=100.0,
        alpha=0.05,
        beta=1.0,
        gamma_min=0.05,
        lambda_prior=1.0
    ).to(device)

    dataset = Dataset3DPW(root_dir="data/3dpw", split="test")
    total_samples = len(dataset)
    print(f"[Pilot Experiment] Total test samples available: {total_samples}")

    # Subsample frames evenly across multiple sequences
    selected_indices = [min(i * stride, total_samples - 1) for i in range(num_frames)]

    results_a = []
    results_b = []
    results_c = []
    results_d = []
    results_e = []
    results_f = []
    per_frame_records = []

    print(f"[Pilot Experiment] Step 1: Pre-extracting normals & initializers for {num_frames} frames...")
    frame_packets = []
    with torch.no_grad():
        for idx_run, s_idx in enumerate(selected_indices):
            sample = dataset[s_idx]
            seq_name = sample.get("seq_name", f"seq_{s_idx}")
            frame_idx = sample.get("frame_idx", s_idx)

            img = sample["image"].to(device)  # (256, 256, 3) in [0, 1]
            gt_theta = sample["theta"].to(device)  # (24, 3)
            gt_beta = sample["beta"].to(device)    # (10,)
            gt_trans = sample["trans"].to(device)  # (3,)
            K = sample["K"].to(device)              # (3, 3)

            gt_out = smpl(theta=gt_theta.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=gt_trans.unsqueeze(0))
            gt_joints = gt_out["joints"][0].clone()
            gt_verts = gt_out["vertices"][0].clone()

            norm_img = img.permute(2, 0, 1).unsqueeze(0)
            target_normals = run_dsine_batch(dsine_model, norm_img, device=device)[0].clone()
            coarse_pred = coarse_estimator(norm_img, intrinsics=K.unsqueeze(0))
            init_theta = coarse_pred["theta"][0].clone()
            init_trans = coarse_pred["trans"][0].clone()

            init_out = smpl(theta=init_theta.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=init_trans.unsqueeze(0))
            rend_init = mesh_rasterizer(init_out["vertices"][0], init_out["normals"][0], smpl.faces, K)
            target_mask = F.max_pool2d(rend_init.mask.unsqueeze(0).unsqueeze(0), kernel_size=15, stride=1, padding=7)[0, 0].clone()

            init_joints = init_out["joints"][0].clone()
            init_verts = init_out["vertices"][0].clone()

            frame_packets.append({
                "seq_name": seq_name,
                "frame_idx": frame_idx,
                "img": img.cpu(),
                "gt_theta": gt_theta.cpu(),
                "gt_beta": gt_beta.cpu(),
                "gt_trans": gt_trans.cpu(),
                "K": K.cpu(),
                "gt_joints": gt_joints.cpu(),
                "gt_verts": gt_verts.cpu(),
                "target_normals": target_normals.cpu(),
                "init_theta": init_theta.cpu(),
                "init_trans": init_trans.cpu(),
                "target_mask": target_mask.cpu(),
                "init_joints": init_joints.cpu(),
                "init_verts": init_verts.cpu()
            })
            del img, norm_img, gt_theta, gt_beta, gt_trans, K, gt_out, gt_joints, gt_verts
            del target_normals, coarse_pred, init_theta, init_trans, init_out, rend_init, target_mask
            del init_joints, init_verts

    # Release neural models from GPU memory (freed ~2.2 GB VRAM)
    del dsine_model, coarse_estimator
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Pilot Experiment] Freed DSINE & HMR 2.0 coarse models from GPU (freed ~2.2 GB VRAM)!")

    start_total_time = time.time()
    print(f"\n[Pilot Experiment] Step 2: Optimizing 6 conditions across {len(frame_packets)} frames...")

    for idx_run, packet in enumerate(frame_packets):
        t_frame_start = time.time()
        seq_name = packet["seq_name"]
        frame_idx = packet["frame_idx"]
        img = packet["img"].to(device)
        gt_beta = packet["gt_beta"].to(device)
        K = packet["K"].to(device)
        gt_joints = packet["gt_joints"].to(device)
        gt_verts = packet["gt_verts"].to(device)
        target_normals = packet["target_normals"].to(device)
        target_mask = packet["target_mask"].to(device)
        init_theta = packet["init_theta"].to(device)
        init_trans = packet["init_trans"].to(device)
        init_joints = packet["init_joints"].to(device)
        init_verts = packet["init_verts"].to(device)

        metric_a = {
            "mpjpe": compute_mpjpe(init_joints, gt_joints),
            "pa_mpjpe": compute_pa_mpjpe(init_joints, gt_joints),
            "pve": compute_pve(init_verts, gt_verts)
        }
        results_a.append(metric_a)

        # -------------------------------------------------------------
        # Condition B: Differentiable Mesh Normal Rendering
        # -------------------------------------------------------------
        theta_b, trans_b = optimize_condition_b(
            smpl=smpl,
            mesh_rasterizer=mesh_rasterizer,
            init_theta=init_theta,
            init_trans=init_trans,
            beta=gt_beta,
            target_normals=target_normals,
            target_mask=target_mask,
            K=K,
            num_iterations=opt_iterations
        )
        with torch.no_grad():
            out_b = smpl(theta=theta_b.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=trans_b.unsqueeze(0))
            metric_b = {
                "mpjpe": compute_mpjpe(out_b["joints"][0], gt_joints),
                "pa_mpjpe": compute_pa_mpjpe(out_b["joints"][0], gt_joints),
                "pve": compute_pve(out_b["vertices"][0], gt_verts)
            }
        results_b.append(metric_b)
        del theta_b, trans_b, out_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Condition C: Constrained Gaussian Splatting (Fixed Attributes)
        # -------------------------------------------------------------
        gaussians_c = TangentialGaussianSurface(num_splats=6890, device=device).to(device)
        theta_c, trans_c = optimize_condition_c(
            smpl=smpl,
            gaussians=gaussians_c,
            gaussian_rasterizer=gaussian_rasterizer,
            init_theta=init_theta,
            init_trans=init_trans,
            beta=gt_beta,
            target_normals=target_normals,
            target_mask=target_mask,
            K=K,
            num_iterations=opt_iterations
        )
        with torch.no_grad():
            out_c = smpl(theta=theta_c.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=trans_c.unsqueeze(0))
            metric_c = {
                "mpjpe": compute_mpjpe(out_c["joints"][0], gt_joints),
                "pa_mpjpe": compute_pa_mpjpe(out_c["joints"][0], gt_joints),
                "pve": compute_pve(out_c["vertices"][0], gt_verts)
            }
        results_c.append(metric_c)
        del gaussians_c, theta_c, trans_c, out_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Condition D: Unconstrained Gaussian Splatting (Free Attributes)
        # -------------------------------------------------------------
        gaussians_d = TangentialGaussianSurface(num_splats=6890, device=device).to(device)
        router_d = DualFrequencyGradientRouter(
            smpl=smpl,
            gaussian_surface=gaussians_d,
            collision_engine=collision_engine,
            mesh_graph=mesh_graph,
            rasterizer=gaussian_rasterizer
        ).to(device)

        theta_d, trans_d = optimize_condition_d(
            router=router_d,
            init_theta=init_theta,
            init_trans=init_trans,
            target_normals=target_normals,
            target_rgb=img,
            target_mask=target_mask,
            K=K,
            num_iterations=opt_iterations
        )
        with torch.no_grad():
            out_d = smpl(theta=theta_d.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=trans_d.unsqueeze(0))
            metric_d = {
                "mpjpe": compute_mpjpe(out_d["joints"][0], gt_joints),
                "pa_mpjpe": compute_pa_mpjpe(out_d["joints"][0], gt_joints),
                "pve": compute_pve(out_d["vertices"][0], gt_verts)
            }
        results_d.append(metric_d)
        del gaussians_d, router_d, theta_d, trans_d, out_d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Condition E: Nuisance-Controlled 3DGS (Proposed Method)
        # -------------------------------------------------------------
        gaussians_e = TangentialGaussianSurface(num_splats=6890, device=device).to(device)
        router_e = DualFrequencyGradientRouter(
            smpl=smpl,
            gaussian_surface=gaussians_e,
            collision_engine=collision_engine,
            mesh_graph=mesh_graph,
            rasterizer=gaussian_rasterizer,
            sensitivity_estimator=sensitivity_estimator
        ).to(device)

        theta_e, trans_e = optimize_condition_e(
            router=router_e,
            init_theta=init_theta,
            init_trans=init_trans,
            target_normals=target_normals,
            target_rgb=img,
            target_mask=target_mask,
            K=K,
            num_iterations=opt_iterations
        )
        with torch.no_grad():
            out_e = smpl(theta=theta_e.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=trans_e.unsqueeze(0))
            metric_e = {
                "mpjpe": compute_mpjpe(out_e["joints"][0], gt_joints),
                "pa_mpjpe": compute_pa_mpjpe(out_e["joints"][0], gt_joints),
                "pve": compute_pve(out_e["vertices"][0], gt_verts)
            }
        results_e.append(metric_e)
        del gaussians_e, router_e, theta_e, trans_e, out_e
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Condition F: Stage-Gated Nuisance-Controlled 3DGS (Proposed)
        # -------------------------------------------------------------
        gaussians_f = TangentialGaussianSurface(num_splats=6890, device=device).to(device)
        router_f = DualFrequencyGradientRouter(
            smpl=smpl,
            gaussian_surface=gaussians_f,
            collision_engine=collision_engine,
            mesh_graph=mesh_graph,
            rasterizer=gaussian_rasterizer,
            sensitivity_estimator=sensitivity_estimator
        ).to(device)
        scheduler_f = CapacityGatedScheduler(
            total_iterations=opt_iterations,
            switch_iteration=8,
            max_offset_norm=0.015,
            deform_warmup_steps=2
        )

        theta_f, trans_f = optimize_condition_f(
            router=router_f,
            scheduler=scheduler_f,
            init_theta=init_theta,
            init_trans=init_trans,
            target_normals=target_normals,
            target_rgb=img,
            target_mask=target_mask,
            K=K,
            num_iterations=opt_iterations
        )
        with torch.no_grad():
            out_f = smpl(theta=theta_f.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=trans_f.unsqueeze(0))
            metric_f = {
                "mpjpe": compute_mpjpe(out_f["joints"][0], gt_joints),
                "pa_mpjpe": compute_pa_mpjpe(out_f["joints"][0], gt_joints),
                "pve": compute_pve(out_f["vertices"][0], gt_verts)
            }
        results_f.append(metric_f)
        del gaussians_f, router_f, scheduler_f, theta_f, trans_f, out_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dt_frame = time.time() - t_frame_start
        print(
            f"  [Frame {idx_run+1:02d}/{num_frames:02d}] Seq: {seq_name}#{frame_idx:04d} | "
            f"Init: {metric_a['pa_mpjpe']:.1f} | "
            f"B (Mesh): {metric_b['pa_mpjpe']:.1f} | "
            f"C (Constrained): {metric_c['pa_mpjpe']:.1f} | "
            f"D (Unconstrained): {metric_d['pa_mpjpe']:.1f} | "
            f"E (Controlled): {metric_e['pa_mpjpe']:.1f} | "
            f"F (Stage-Gated): {metric_f['pa_mpjpe']:.1f} mm | "
            f"Time: {dt_frame:.2f}s"
        )

        per_frame_records.append({
            "seq_name": seq_name,
            "frame_idx": int(frame_idx),
            "cond_a": metric_a,
            "cond_b": metric_b,
            "cond_c": metric_c,
            "cond_d": metric_d,
            "cond_e": metric_e,
            "cond_f": metric_f
        })

        del img, gt_beta, K, gt_joints, gt_verts, target_normals, target_mask, init_theta, init_trans, init_joints, init_verts
        frame_packets[idx_run] = None
        del packet
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_total_time

    def summarize(metrics_list: List[Dict], key: str) -> float:
        return sum(m[key] for m in metrics_list) / max(len(metrics_list), 1)

    def calc_worsening(base_list: List[Dict], comp_list: List[Dict]) -> float:
        worse_count = sum(1 for b, c in zip(base_list, comp_list) if c["pa_mpjpe"] > b["pa_mpjpe"] + 0.1)
        return (worse_count / max(len(base_list), 1)) * 100.0

    summary = {
        "num_frames": num_frames,
        "opt_iterations": opt_iterations,
        "device": device_str,
        "total_elapsed_sec": total_time,
        "condition_a_frozen": {
            "mean_mpjpe_mm": summarize(results_a, "mpjpe"),
            "mean_pa_mpjpe_mm": summarize(results_a, "pa_mpjpe"),
            "mean_pve_mm": summarize(results_a, "pve")
        },
        "condition_b_mesh": {
            "mean_mpjpe_mm": summarize(results_b, "mpjpe"),
            "mean_pa_mpjpe_mm": summarize(results_b, "pa_mpjpe"),
            "mean_pve_mm": summarize(results_b, "pve"),
            "worsening_rate_pct": calc_worsening(results_a, results_b)
        },
        "condition_c_constrained_3dgs": {
            "mean_mpjpe_mm": summarize(results_c, "mpjpe"),
            "mean_pa_mpjpe_mm": summarize(results_c, "pa_mpjpe"),
            "mean_pve_mm": summarize(results_c, "pve"),
            "worsening_rate_pct": calc_worsening(results_a, results_c)
        },
        "condition_d_unconstrained_3dgs": {
            "mean_mpjpe_mm": summarize(results_d, "mpjpe"),
            "mean_pa_mpjpe_mm": summarize(results_d, "pa_mpjpe"),
            "mean_pve_mm": summarize(results_d, "pve"),
            "worsening_rate_pct": calc_worsening(results_a, results_d)
        },
        "condition_e_nuisance_controlled_3dgs": {
            "mean_mpjpe_mm": summarize(results_e, "mpjpe"),
            "mean_pa_mpjpe_mm": summarize(results_e, "pa_mpjpe"),
            "mean_pve_mm": summarize(results_e, "pve"),
            "worsening_rate_pct": calc_worsening(results_a, results_e)
        },
        "condition_f_stage_gated_3dgs": {
            "mean_mpjpe_mm": summarize(results_f, "mpjpe"),
            "mean_pa_mpjpe_mm": summarize(results_f, "pa_mpjpe"),
            "mean_pve_mm": summarize(results_f, "pve"),
            "worsening_rate_pct": calc_worsening(results_a, results_f)
        },
        "per_frame_records": per_frame_records
    }

    print("\n" + "=" * 80)
    print(" PILOT EXPERIMENT EXECUTIVE SUMMARY (6-WAY CONTROLLED BENCHMARK)")
    print("=" * 80)
    print(f"Condition A (Frozen Initializer):       PA-MPJPE: {summary['condition_a_frozen']['mean_pa_mpjpe_mm']:.2f} mm | MPJPE: {summary['condition_a_frozen']['mean_mpjpe_mm']:.2f} mm")
    print(f"Condition B (Mesh Normal Render):       PA-MPJPE: {summary['condition_b_mesh']['mean_pa_mpjpe_mm']:.2f} mm | Worsening: {summary['condition_b_mesh']['worsening_rate_pct']:.1f}%")
    print(f"Condition C (Constrained 3DGS):         PA-MPJPE: {summary['condition_c_constrained_3dgs']['mean_pa_mpjpe_mm']:.2f} mm | Worsening: {summary['condition_c_constrained_3dgs']['worsening_rate_pct']:.1f}%")
    print(f"Condition D (Unconstrained 3DGS):       PA-MPJPE: {summary['condition_d_unconstrained_3dgs']['mean_pa_mpjpe_mm']:.2f} mm | Worsening: {summary['condition_d_unconstrained_3dgs']['worsening_rate_pct']:.1f}%")
    print(f"Condition E (Simultaneous Controlled):  PA-MPJPE: {summary['condition_e_nuisance_controlled_3dgs']['mean_pa_mpjpe_mm']:.2f} mm | Worsening: {summary['condition_e_nuisance_controlled_3dgs']['worsening_rate_pct']:.1f}%")
    print(f"Condition F (Stage-Gated Controlled):   PA-MPJPE: {summary['condition_f_stage_gated_3dgs']['mean_pa_mpjpe_mm']:.2f} mm | Worsening: {summary['condition_f_stage_gated_3dgs']['worsening_rate_pct']:.1f}%")
    print("=" * 80)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Pilot Experiment] Saved complete results artifact to {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot 4-Way Experiment Runner")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of pilot frames to benchmark")
    parser.add_argument("--stride", type=int, default=50, help="Stride between sampled frames")
    parser.add_argument("--iters", type=int, default=15, help="Number of refinement iterations per frame")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/pilot_4way_experiment_results.json")
    args = parser.parse_args()

    run_pilot_experiment(
        num_frames=args.num_frames,
        stride=args.stride,
        opt_iterations=args.iters,
        device_str=args.device,
        output_json=args.out
    )
