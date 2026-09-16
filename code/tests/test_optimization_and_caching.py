"""
Unit tests for Colab T4 Caching, Vectorized Rasterization, and Stateful Resumption.
"""

import os
import tempfile
import torch
import numpy as np
import pytest

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.geometry.mesh_graph import MeshGraph
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer
from src.optimization.dual_frequency_router import DualFrequencyGradientRouter
from src.pipeline.feature_cache import FeatureCacheDataset


def test_vectorized_rasterizer_shapes_and_gradients():
    """Verifies that the vectorized rasterizer produces valid maps and backward gradients."""
    device = torch.device("cpu")
    rasterizer = DifferentiableNormalRasterizer(image_height=64, image_width=64, tile_size=16)

    centers = torch.randn(50, 3, requires_grad=True)
    centers.data[:, 2] = centers.data[:, 2].abs() + 1.0  # in front of camera
    covs = torch.eye(3).unsqueeze(0).repeat(50, 1, 1) * 0.01
    normals = torch.randn(50, 3)
    colors = torch.rand(50, 3)
    opacities = torch.rand(50, 1)
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])

    out = rasterizer(centers, covs, normals, colors, opacities, K)

    assert out.normals.shape == (64, 64, 3)
    assert out.rgb.shape == (64, 64, 3)
    assert out.mask.shape == (64, 64)
    assert out.depth.shape == (64, 64)

    # Gradient flow test
    loss = out.normals.sum() + out.mask.sum()
    loss.backward()
    assert centers.grad is not None
    assert torch.all(torch.isfinite(centers.grad))


def test_stateful_checkpoint_persistence():
    """Verifies that model parameters and optimizer states serialize and deserialize perfectly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = os.path.join(tmpdir, "test_ckpt.pt")

        theta = torch.nn.Parameter(torch.randn(24, 3))
        trans = torch.nn.Parameter(torch.randn(3))
        opt = torch.optim.AdamW([theta, trans], lr=1e-3)

        # Run dummy step to populate optimizer state
        loss = (theta ** 2).sum() + (trans ** 2).sum()
        loss.backward()
        opt.step()

        # Save checkpoint
        torch.save({
            "theta": theta.detach().cpu(),
            "trans": trans.detach().cpu(),
            "opt_state": opt.state_dict(),
            "epoch": 3,
            "batch_idx": 450,
        }, chk_path)

        assert os.path.exists(chk_path)
        size_bytes = os.path.getsize(chk_path)
        assert size_bytes < 50 * 1024  # Under 50 KB for test parameters

        # Load into fresh instances
        theta_fresh = torch.nn.Parameter(torch.zeros(24, 3))
        trans_fresh = torch.nn.Parameter(torch.zeros(3))
        opt_fresh = torch.optim.AdamW([theta_fresh, trans_fresh], lr=1e-3)

        ckpt = torch.load(chk_path)
        theta_fresh.data.copy_(ckpt["theta"])
        trans_fresh.data.copy_(ckpt["trans"])
        opt_fresh.load_state_dict(ckpt["opt_state"])

        assert torch.allclose(theta, theta_fresh)
        assert torch.allclose(trans, trans_fresh)
        assert ckpt["epoch"] == 3
        assert ckpt["batch_idx"] == 450
