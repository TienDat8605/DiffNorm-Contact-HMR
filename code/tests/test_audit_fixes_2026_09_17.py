"""
Regression tests for September 17, 2026 Audit Fixes.
Verifies SMPL zero-pose identity, tangent frame disk-normal alignment,
exact log-space Gaussian overlap, batched PA-MPJPE invariance,
rasterizer tail cutoff gating, and initializer source provenance.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from src.geometry.smpl_wrapper import SMPLWrapper, rodrigues
from src.gaussian.splat_surface import TangentialGaussianSurface, build_tangent_frame
from src.physics.gaussian_convolution import compute_gaussian_overlap_matrix
from src.pipeline.eval_metrics import compute_pa_mpjpe, compute_single_pa_mpjpe
from src.pipeline.coarse_pose_hmr2 import CoarsePoseHMR2, rotation_matrix_to_axis_angle
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer


def test_smpl_zero_pose_identity():
    """F1: Skinning transform at zero pose must be the identity with < 1e-6 m vertex error."""
    smpl = SMPLWrapper()
    zero_theta = torch.zeros(1, 24, 3)
    out = smpl(zero_theta)
    
    # 1. Neutral vertex displacement from canonical template
    disp = (out["vertices"][0] - smpl.v_template).norm(dim=-1).max().item()
    assert disp < 1e-6, f"Neutral mesh moved {disp * 1000.0:.4f} mm from template at zero pose!"

    # 2. Skinning translation offset must be zero
    _, rel_transforms = smpl.forward_kinematics(torch.eye(3).repeat(1, 24, 1, 1), smpl.joints_template[None])
    max_translation = rel_transforms[0, :, :3, 3].abs().max().item()
    assert max_translation < 1e-6, f"Zero-pose skinning translation is {max_translation:.4f} m, expected 0.0"


def test_smpl_single_joint_rotation():
    """F1: Articulating a child joint (e.g. elbow) must not move vertices outside its kinematic branch."""
    smpl = SMPLWrapper()
    theta = torch.zeros(1, 24, 3)
    # Rotate left elbow (joint 18) by 45 degrees around Z
    theta[0, 18, 2] = 0.785398
    out = smpl(theta)
    neutral = smpl(torch.zeros(1, 24, 3))

    # Vertices unattached to the left arm branch (joints 18, 20, 22) must experience ~0 motion
    arm_branch = [18, 20, 22]
    branch_weight = smpl.weights[:, arm_branch].sum(dim=-1)
    unaffected_verts = torch.where(branch_weight < 1e-5)[0]
    assert len(unaffected_verts) > 1000, "Should have plenty of unaffected vertices"
    motion = (out["vertices"][0, unaffected_verts] - neutral["vertices"][0, unaffected_verts]).norm(dim=-1).max().item()
    assert motion < 1e-4, f"Unaffected vertices moved {motion * 1000.0:.4f} mm during elbow rotation!"


def test_tangent_covariance_normal_alignment():
    """F3: Gaussian covariance disk thin axis must be collinear with surface normal for arbitrary normals."""
    torch.manual_seed(42)
    normals = F.normalize(torch.randn(10, 3), dim=-1)
    
    for i in range(10):
        n = normals[i:i+1]
        g = TangentialGaussianSurface(num_splats=1)
        cov = g.get_spatial_covariances(mesh_normals=n)
        
        # Smallest eigenvalue corresponds to thin axis
        eigvals, eigvecs = torch.linalg.eigh(cov)
        thin_axis = eigvecs[0, :, 0]
        
        surf_normal = g.get_surface_normals(mesh_normals=n)[0]
        dot = (surf_normal * thin_axis).abs().sum().item()
        assert dot > 0.9999, f"Normal and thin covariance axis misaligned: dot={dot:.6f}"


def test_gaussian_overlap_closed_form_exact():
    """F4: Gaussian overlap prefactor must match closed form at production scale without 28x clamp inflation."""
    g = TangentialGaussianSurface(num_splats=1)
    cov = g.get_spatial_covariances()
    mu = torch.zeros(1, 3)
    
    actual = compute_gaussian_overlap_matrix(mu, cov, mu, cov).item()
    expected = math.pi ** 1.5 * g.get_scales().prod().item()
    ratio = actual / expected
    
    assert abs(ratio - 1.0) < 1e-4, f"Gaussian overlap ratio {ratio:.6f} diverges from 1.0 (clamp inflation detected!)"


def test_pa_mpjpe_batch_invariance():
    """F7: Batched PA-MPJPE must equal the mean of individual Umeyama alignments."""
    torch.manual_seed(123)
    points = torch.randn(14, 3)
    R1 = rodrigues(torch.tensor([[0.3, -0.2, 0.5]]))[0]
    R2 = rodrigues(torch.tensor([[-0.4, 0.1, 0.8]]))[0]
    
    pred = torch.stack([
        points @ R1.T + torch.tensor([1.0, -2.0, 0.5]),
        points @ R2.T + torch.tensor([-3.0, 1.5, 2.0])
    ])
    gt = torch.stack([points, points])
    
    batched_err = compute_pa_mpjpe(pred, gt)
    indiv_err = sum(compute_single_pa_mpjpe(p, g) for p, g in zip(pred, gt)) / 2.0
    
    assert abs(batched_err - indiv_err) < 1e-5, f"Batched PA-MPJPE ({batched_err}) != mean individual ({indiv_err})"
    assert batched_err < 1e-3, f"PA-MPJPE between rigidly transformed identical points was {batched_err} mm, expected 0"


def test_rasterizer_background_normal_gating():
    """F5: Pixels with near-zero opacity must not be assigned unit-length normal vectors."""
    renderer = DifferentiableNormalRasterizer(image_height=32, image_width=32)
    center = torch.tensor([[0.0, 0.0, 2.0]])
    K = torch.tensor([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]])
    cov = torch.eye(3).unsqueeze(0) * 0.0001
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    color = torch.ones(1, 3)
    opacity = torch.ones(1, 1) * 0.95
    
    rend = renderer(center, cov, normal, color, opacity, K)
    # Corner pixel (0, 0) is > 20 pixels away from splat center
    assert rend.mask[0, 0].item() == 0.0, "Background corner pixel has nonzero opacity!"
    assert rend.normals[0, 0].norm().item() == 0.0, "Background corner pixel has nonzero normal length!"


def test_coarse_pose_source_integrity():
    """F2: Initializer must never claim '4D-Humans-HMR2.0' when running kinematic fallback."""
    coarse = CoarsePoseHMR2(
        checkpoint_path="/tmp/nonexistent_ckpt.pt",
        device=torch.device("cpu"),
        auto_download=False,
        require_model=False
    )
    out = coarse(torch.zeros(1, 3, 32, 32))
    assert out["source"] == "Kinematic-Coarse-Prior", f"False source claim: {out['source']}"
    assert coarse.model is None, "Model should be None when checkpoint does not exist"


def test_rotation_matrix_to_axis_angle():
    """F2: Rodrigues logarithmic map roundtrip must recover original axis-angle."""
    theta_orig = torch.tensor([[0.1, -0.4, 0.8], [0.0, 0.0, 0.0], [1.2, 0.0, -0.5]])
    R = rodrigues(theta_orig.unsqueeze(0))[0]
    theta_rec = rotation_matrix_to_axis_angle(R)
    diff = (theta_orig - theta_rec).abs().max().item()
    assert diff < 1e-6, f"Axis-angle recovery error {diff:.6e} exceeds 1e-6"


def test_mesh_normal_rasterizer_differentiability():
    """Condition B: DifferentiableMeshNormalRasterizer must produce non-zero gradients to SMPL parameters."""
    from src.rendering.mesh_normal_rasterizer import DifferentiableMeshNormalRasterizer
    smpl = SMPLWrapper()
    rasterizer = DifferentiableMeshNormalRasterizer(image_height=64, image_width=64, tile_size=16)

    theta = torch.zeros(1, 24, 3, requires_grad=True)
    trans = torch.tensor([[0.0, 0.0, 2.5]], requires_grad=True)
    K = torch.tensor([[150.0, 0.0, 32.0], [0.0, 150.0, 32.0], [0.0, 0.0, 1.0]])

    out = smpl(theta=theta, trans=trans)
    rend = rasterizer(out["vertices"][0], out["normals"][0], smpl.faces, K)

    assert rend.normals.shape == (64, 64, 3)
    assert rend.mask.shape == (64, 64)
    assert (rend.mask > 0.5).sum().item() > 0

    loss = (1.0 - (rend.normals * torch.tensor([0.0, 0.0, 1.0])).sum(dim=-1)).mean()
    loss.backward()
    assert theta.grad is not None
    assert theta.grad.norm().item() > 0.0

