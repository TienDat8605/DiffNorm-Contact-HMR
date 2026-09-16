"""
Tests for Dual-Frequency Gradient Router and Gradient Detachment Integrity.
"""

import pytest
import torch
import torch.nn as nn

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.geometry.mesh_graph import MeshGraph
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer
from src.optimization.dual_frequency_router import DualFrequencyGradientRouter


def test_gradient_detachment_integrity():
    device = torch.device("cpu")
    smpl = SMPLWrapper(device=device)
    segmenter = KinematicSegmenter(smpl.weights)
    mesh_graph = MeshGraph(smpl.faces)
    gaussians = TangentialGaussianSurface(num_splats=6890, device=device)
    collision_engine = GaussianSelfCollisionEngine(segmenter)
    rasterizer = DifferentiableNormalRasterizer(image_height=64, image_width=64)

    router = DualFrequencyGradientRouter(
        smpl=smpl,
        gaussian_surface=gaussians,
        collision_engine=collision_engine,
        mesh_graph=mesh_graph,
        rasterizer=rasterizer,
    )

    theta = torch.zeros(1, 24, 3, requires_grad=True, device=device)
    trans = torch.tensor([[0.0, 0.0, 2.5]], requires_grad=True, device=device)

    # 1. Forward deformation stream alone with detached vertices
    verts = smpl(theta=theta, trans=trans)["vertices"][0]
    verts_detached = verts.detach()
    centers_detached = gaussians.get_centers(verts_detached, detach_mesh=True)
    covs = gaussians.get_spatial_covariances()
    normals = gaussians.get_surface_normals()

    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], device=device)
    render_deform = rasterizer(
        centers_cam=centers_detached,
        covs_cam=covs,
        normals_cam=normals,
        colors=gaussians.colors,
        opacities=gaussians.get_opacities(),
        K=K
    )

    target_rgb = torch.ones(64, 64, 3, device=device)
    loss_deform = router.loss_photo_fn(render_deform.rgb, target_rgb)
    loss_deform.backward()

    # CRITICAL INVARIANT: theta.grad MUST BE None (or zero) from deformation loss
    assert theta.grad is None, "Deformation loss leaked gradients into skeletal joint parameters theta!"

    # 2. Geometric stream: forward with connected graph (recomputed fresh)
    verts_fresh = smpl(theta=theta, trans=trans)["vertices"][0]
    centers_connected = gaussians.get_centers(verts_fresh, detach_mesh=False)
    covs_geom = gaussians.get_spatial_covariances()
    normals_geom = gaussians.get_surface_normals()

    render_geom = rasterizer(
        centers_cam=centers_connected,
        covs_cam=covs_geom,
        normals_cam=normals_geom,
        colors=gaussians.colors,
        opacities=gaussians.get_opacities(),
        K=K
    )

    # Use silhouette mask and varying normal target
    target_mask = torch.ones(64, 64, device=device)
    loss_mask = router.loss_mask_fn(render_geom.mask, target_mask)

    target_normals = torch.zeros(64, 64, 3, device=device)
    target_normals[..., 0] = 1.0  # Misaligned target: rendered is [0, 0, 1], target is [1, 0, 0]
    loss_norm = router.loss_normal_fn(render_geom.normals, target_normals)

    loss_geom = loss_norm + loss_mask
    loss_geom.backward()

    # Geometric stream MUST produce non-zero gradient in theta or trans!
    assert trans.grad is not None and not torch.all(trans.grad == 0), "Geometric loss failed to backpropagate into trans!"
    assert theta.grad is not None and not torch.all(theta.grad == 0), "Geometric loss failed to backpropagate into theta!"
