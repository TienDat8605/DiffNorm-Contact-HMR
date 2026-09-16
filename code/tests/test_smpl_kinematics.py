"""
Tests for SMPLWrapper, forward kinematics, and LBS skinning.
"""

import pytest
import torch
from src.geometry.smpl_wrapper import SMPLWrapper, rodrigues


def test_rodrigues_conversion():
    # Identity rotation
    r_zero = torch.zeros(1, 3)
    R_ident = rodrigues(r_zero)
    assert torch.allclose(R_ident, torch.eye(3).unsqueeze(0), atol=1e-5)

    # 90 degrees around Z axis: [0, 0, pi/2]
    r_z = torch.tensor([[0.0, 0.0, 3.14159265 / 2.0]])
    R_z = rodrigues(r_z)
    expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    assert torch.allclose(R_z, expected, atol=1e-4)


def test_smpl_forward_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPLWrapper(device=device).to(device)

    B = 2
    theta = torch.zeros(B, 24, 3, device=device, requires_grad=True)
    beta = torch.zeros(B, 10, device=device, requires_grad=True)
    trans = torch.zeros(B, 3, device=device, requires_grad=True)

    out = smpl(theta=theta, beta=beta, trans=trans)
    assert out["vertices"].shape == (B, 6890, 3)
    assert out["joints"].shape == (B, 24, 3)
    assert out["normals"].shape == (B, 6890, 3)

    # Check normal vectors are unit length
    norm_lens = torch.norm(out["normals"], dim=-1)
    assert torch.allclose(norm_lens, torch.ones_like(norm_lens), atol=1e-4)

    # Check differentiability
    loss = torch.sum(out["vertices"] ** 2)
    loss.backward()
    assert theta.grad is not None and not torch.all(theta.grad == 0)
    assert beta.grad is not None and not torch.all(beta.grad == 0)
    assert trans.grad is not None and not torch.all(trans.grad == 0)


def test_normal_gradient_flow_into_theta():
    """Verifies that surface normal supervision backpropagates directly into skeletal joint rotations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPLWrapper(device=device).to(device)

    theta = torch.zeros(1, 24, 3, device=device, requires_grad=True)
    out = smpl(theta=theta)
    mesh_normals = out["normals"][0]  # (6890, 3)

    # Simulated target normal pointing towards camera [0, 0, 1]
    target_normals = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).repeat(6890, 1)
    loss_normal = 1.0 - torch.mean(torch.sum(mesh_normals * target_normals, dim=-1))

    loss_normal.backward()
    assert theta.grad is not None
    # Must produce non-zero rotational gradients on skeletal joints
    assert torch.norm(theta.grad) > 1e-4
