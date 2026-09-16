"""
Tests for TangentialGaussianSurface, disk constraint, and normal vectors.
"""

import pytest
import torch
from src.gaussian.splat_surface import TangentialGaussianSurface, quaternion_to_rotation_matrix


def test_tangential_disk_constraint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = 0.03
    surface = TangentialGaussianSurface(num_splats=100, tau=tau, device=device).to(device)

    scales = surface.get_scales()  # (100, 3)
    s1, s2, s3 = scales[:, 0], scales[:, 1], scales[:, 2]

    # Verify s3 is strictly tau * min(s1, s2)
    expected_s3 = tau * torch.minimum(s1, s2)
    assert torch.allclose(s3, expected_s3, atol=1e-5)
    # Verify s3 is much smaller than tangential axes
    assert torch.all(s3 < s1 * 0.1)
    assert torch.all(s3 < s2 * 0.1)


def test_quaternion_rotation_and_normals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surface = TangentialGaussianSurface(num_splats=10, device=device).to(device)

    R = surface.get_rotation_matrices()  # (10, 3, 3)
    # Check orthogonality: R @ R^T == I
    eye = torch.eye(3, device=device).unsqueeze(0).repeat(10, 1, 1)
    RtR = torch.matmul(R, R.transpose(-1, -2))
    assert torch.allclose(RtR, eye, atol=1e-5)

    # Check surface normals are unit length
    normals = surface.get_surface_normals()
    lens = torch.norm(normals, dim=-1)
    assert torch.allclose(lens, torch.ones_like(lens), atol=1e-5)


def test_spatial_covariance_positive_definite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surface = TangentialGaussianSurface(num_splats=10, device=device).to(device)

    covs = surface.get_spatial_covariances()  # (10, 3, 3)
    # Check symmetric: cov == cov^T
    assert torch.allclose(covs, covs.transpose(-1, -2), atol=1e-6)
    # Check positive eigenvalues
    eigvals = torch.linalg.eigvalsh(covs)
    assert torch.all(eigvals > 0)
