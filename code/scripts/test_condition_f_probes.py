"""
Probe test comparing Stage-Gated Capacity Scheduling variations on 3 distinct 3DPW frames.
"""

import os
import sys
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
from src.pipeline.eval_metrics import compute_pa_mpjpe
from scripts.extract_dsine_normals import load_dsine_model, run_dsine_batch


def run_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Probe] Using device: {device}")

    smpl = SMPLWrapper(device=device).to(device)
    coarse_estimator = CoarsePoseHMR2(device=device, auto_download=False)
    dsine_model = load_dsine_model(device=device)

    mesh_rasterizer = DifferentiableMeshNormalRasterizer(image_height=256, image_width=256).to(device)
    gaussian_rasterizer = DifferentiableNormalRasterizer(image_height=256, image_width=256, max_splats_per_tile=256).to(device)

    segmenter = KinematicSegmenter(smpl.weights)
    mesh_graph = MeshGraph(smpl.faces).to(device)
    collision_engine = GaussianSelfCollisionEngine(segmenter).to(device)
    sensitivity_estimator = NuisanceSensitivityEstimator(
        smpl=smpl,
        segmenter=segmenter,
        mesh_graph=mesh_graph,
        lambda_lap=50.0,
        lambda_tight=100.0,
        gamma_min=0.05
    ).to(device)

    dataset = Dataset3DPW(root_dir="data/3dpw", split="test")

    # Pick 3 known frames: 0 (arguing), 2 (cafe), 4 (sitOnStairs)
    test_indices = [0, 2 * 1423, 4 * 1423]

    for s_idx in test_indices:
        s_idx = min(s_idx, len(dataset) - 1)
        sample = dataset[s_idx]
        seq_name = sample.get("seq_name", f"seq_{s_idx}")
        frame_idx = sample.get("frame_idx", s_idx)
        print(f"\n--- Testing on {seq_name} #{frame_idx} ---")

        img = sample["image"].to(device)
        gt_theta = sample["theta"].to(device)
        gt_beta = sample["beta"].to(device)
        gt_trans = sample["trans"].to(device)
        K = sample["K"].to(device)

        with torch.no_grad():
            gt_joints = smpl(theta=gt_theta.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=gt_trans.unsqueeze(0))["joints"][0]
            norm_img = img.permute(2, 0, 1).unsqueeze(0)
            target_normals = run_dsine_batch(dsine_model, norm_img, device=device)[0]
            coarse_pred = coarse_estimator(norm_img, intrinsics=K.unsqueeze(0))
            init_theta = coarse_pred["theta"][0]
            init_trans = coarse_pred["trans"][0]

            init_out = smpl(theta=init_theta.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=init_trans.unsqueeze(0))
            init_joints = init_out["joints"][0]
            rend_init = mesh_rasterizer(init_out["vertices"][0], init_out["normals"][0], smpl.faces, K)
            target_mask = F.max_pool2d(rend_init.mask.unsqueeze(0).unsqueeze(0), kernel_size=15, stride=1, padding=7)[0, 0]

            pa_init = compute_pa_mpjpe(init_joints, gt_joints)
        print(f"Condition A (Init): {pa_init:.2f} mm")

        # Condition B (Mesh normal)
        th_b = nn.Parameter(init_theta.clone())
        tr_b = nn.Parameter(init_trans.clone())
        opt_b = torch.optim.AdamW([{"params": [th_b, tr_b], "lr": 1e-2}])
        for _ in range(15):
            opt_b.zero_grad()
            out = smpl(theta=th_b.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=tr_b.unsqueeze(0))
            r = mesh_rasterizer(out["vertices"][0], out["normals"][0], smpl.faces, K)
            cos = (r.normals * target_normals).sum(dim=-1)
            vm = (target_mask > 0.5) & (r.mask > 0.5)
            ln = (1.0 - cos)[vm].mean() if vm.any() else (1.0 - cos).mean()
            lm = F.mse_loss(r.mask, target_mask)
            lp = torch.mean((th_b - init_theta) ** 2)
            (ln + 5.0 * lm + 1.0 * lp).backward()
            opt_b.step()
        with torch.no_grad():
            out_b = smpl(theta=th_b.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=tr_b.unsqueeze(0))
            pa_b = compute_pa_mpjpe(out_b["joints"][0], gt_joints)
        print(f"Condition B (Mesh Normal): {pa_b:.2f} mm (diff: {pa_b - pa_init:+.2f} mm)")

        # Condition F (Pure 3DGS Stage-Gated)
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
            total_iterations=15,
            switch_iteration=8,
            max_offset_norm=0.015,
            deform_warmup_steps=2
        )
        th_f = nn.Parameter(init_theta.clone())
        tr_f = nn.Parameter(init_trans.clone())
        opt_kin_f, opt_def_f = router_f.create_optimizers(th_f, tr_f)

        for step_idx in range(15):
            router_f.step(
                theta=th_f,
                trans=tr_f,
                K=K,
                target_normals=target_normals,
                target_rgb=img,
                target_mask=target_mask,
                uncertainty=None,
                opt_kin=opt_kin_f,
                opt_deform=opt_def_f,
                init_theta=init_theta,
                init_trans=init_trans,
                iteration=step_idx,
                scheduler=scheduler_f
            )

        with torch.no_grad():
            out_f = smpl(theta=th_f.unsqueeze(0), beta=gt_beta.unsqueeze(0), trans=tr_f.unsqueeze(0))
            pa_f = compute_pa_mpjpe(out_f["joints"][0], gt_joints)
        print(f"Condition F (Stage-Gated 3DGS): {pa_f:.2f} mm (diff: {pa_f - pa_init:+.2f} mm)")


if __name__ == "__main__":
    run_probe()
