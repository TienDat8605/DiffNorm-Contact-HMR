"""
Test-Time Single-Image Pose and Avatar Optimization Pipeline.
Runs test-time joint optimization on a single in-the-wild image or video frame.
"""

from typing import Dict, Optional, Tuple
import os
import argparse
import torch
import torch.nn as nn

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.geometry.mesh_graph import MeshGraph
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer
from src.optimization.dual_frequency_router import DualFrequencyGradientRouter


def export_mesh_to_obj(vertices: torch.Tensor, faces: torch.Tensor, out_path: str):
    """Exports triangular mesh to standard .obj file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    v_np = vertices.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()

    with open(out_path, "w") as f:
        for v in v_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in f_np:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def optimize_frame(
    target_normals: torch.Tensor,            # (H, W, 3)
    target_rgb: torch.Tensor,                # (H, W, 3)
    target_mask: torch.Tensor,               # (H, W)
    K: torch.Tensor,                         # (3, 3)
    init_theta: Optional[torch.Tensor] = None, # (24, 3)
    init_trans: Optional[torch.Tensor] = None, # (3,)
    num_iterations: int = 50,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Runs DiffNorm-Contact test-time optimization on a single frame.
    """
    # 1. Initialize modules
    smpl = SMPLWrapper(device=device).to(device)
    segmenter = KinematicSegmenter(smpl.weights)
    mesh_graph = MeshGraph(smpl.faces).to(device)
    gaussians = TangentialGaussianSurface(num_splats=6890, device=device).to(device)
    collision_engine = GaussianSelfCollisionEngine(segmenter).to(device)
    rasterizer = DifferentiableNormalRasterizer(
        image_height=target_normals.shape[0],
        image_width=target_normals.shape[1]
    ).to(device)

    router = DualFrequencyGradientRouter(
        smpl=smpl,
        gaussian_surface=gaussians,
        collision_engine=collision_engine,
        mesh_graph=mesh_graph,
        rasterizer=rasterizer,
        lr_theta=1e-2,
        lr_trans=1e-2,
        lr_offsets=5e-3
    ).to(device)

    # 2. Parameters to optimize
    if init_theta is None:
        init_theta = torch.zeros(24, 3, device=device, dtype=torch.float32)
    if init_trans is None:
        init_trans = torch.tensor([0.0, 0.0, 2.5], device=device, dtype=torch.float32)

    theta_param = nn.Parameter(init_theta.clone())
    trans_param = nn.Parameter(init_trans.clone())

    opt_kin, opt_deform = router.create_optimizers(theta_param, trans_param)

    history = []
    if verbose:
        print(f"Starting DiffNorm-Contact optimization ({num_iterations} iterations)...")

    for it in range(num_iterations):
        step_out = router.step(
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
        history.append(step_out)

        if verbose and (it % 10 == 0 or it == num_iterations - 1):
            print(
                f"Iter {it:03d} | Total: {step_out.loss_total:.4f} | "
                f"Normal: {step_out.loss_normal:.4f} | "
                f"Coll: {step_out.loss_collision:.4f} | "
                f"Mask: {step_out.loss_mask:.4f} | "
                f"Photo: {step_out.loss_photo:.4f}"
            )

    # Final posed mesh
    final_out = smpl(theta=theta_param.unsqueeze(0), trans=trans_param.unsqueeze(0))
    final_verts = final_out["vertices"][0]

    return {
        "theta": theta_param.detach(),
        "trans": trans_param.detach(),
        "vertices": final_verts.detach(),
        "faces": smpl.faces,
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffNorm-Contact Single Image Optimizer")
    parser.add_argument("--iters", type=int, default=30, help="Number of optimization iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/mesh_refined.obj")
    args = parser.parse_args()

    dev = torch.device(args.device)
    H, W = 256, 256
    # Synthetic target normal, color, and mask for dry-run
    tgt_norm = torch.zeros(H, W, 3, device=dev)
    tgt_norm[..., 2] = 1.0  # Facing camera
    tgt_rgb = torch.ones(H, W, 3, device=dev) * 0.7
    tgt_mask = torch.zeros(H, W, device=dev)
    tgt_mask[64:192, 64:192] = 1.0
    K = torch.tensor([[300.0, 0.0, 128.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]], device=dev)

    res = optimize_frame(tgt_norm, tgt_rgb, tgt_mask, K, num_iterations=args.iters, device=dev)
    export_mesh_to_obj(res["vertices"], res["faces"], args.out)
    print(f"Exported refined mesh to {args.out}")
