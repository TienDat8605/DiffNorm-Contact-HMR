"""
Tests for Closed-Form Analytical Gaussian Overlap Integral and Repulsion Gradients.
"""

import pytest
import torch
from src.physics.gaussian_convolution import (
    compute_gaussian_overlap_matrix,
    compute_analytical_repulsion_gradients,
)


def test_analytical_overlap_value():
    device = torch.device("cpu")
    # Two identical isotropic Gaussians at same position: mu1 = mu2 = 0, Sigma1 = Sigma2 = sigma^2 * I
    sigma = 0.05
    mu_A = torch.zeros(1, 3, device=device)
    cov_A = (sigma ** 2) * torch.eye(3, device=device).unsqueeze(0)
    mu_B = torch.zeros(1, 3, device=device)
    cov_B = (sigma ** 2) * torch.eye(3, device=device).unsqueeze(0)

    K = compute_gaussian_overlap_matrix(mu_A, cov_A, mu_B, cov_B, distance_cutoff=0.5)
    # Expected integral: int N(x; 0, s^2) * N(x; 0, s^2) dx = (pi * s^2)^(3/2) * (2*pi)^1.5 ...
    # K_11 = (2*pi)^1.5 * sqrt((s^6 * s^6) / (2s^2)^3) = (2*pi)^1.5 * sqrt(s^12 / 8s^6) = (2*pi)^1.5 * (s^3 / sqrt(8)) = (pi)^1.5 * s^3
    import math
    expected = (math.pi ** 1.5) * (sigma ** 3)
    assert math.isclose(K[0, 0].item(), expected, rel_tol=1e-4)


def test_analytical_vs_autograd_gradients():
    device = torch.device("cpu")
    # Set up interacting Gaussians with non-zero distance
    mu_A = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device, requires_grad=True)
    cov_A = (0.04 ** 2) * torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0)
    
    mu_B = torch.tensor([[0.02, 0.01, -0.01]], dtype=torch.float64, device=device, requires_grad=True)
    cov_B = (0.04 ** 2) * torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0)

    # Autograd computation
    K_matrix = compute_gaussian_overlap_matrix(mu_A, cov_A, mu_B, cov_B, distance_cutoff=0.2)
    loss = torch.sum(K_matrix)
    loss.backward()
    autograd_grad_A = mu_A.grad.clone()
    autograd_grad_B = mu_B.grad.clone()

    # Analytical computation
    grad_A_anal, grad_B_anal = compute_analytical_repulsion_gradients(
        mu_A.detach(), cov_A, mu_B.detach(), cov_B, K_matrix.detach()
    )

    # Compare autograd vs analytical
    assert torch.allclose(autograd_grad_A, grad_A_anal, rtol=1e-4, atol=1e-6)
    assert torch.allclose(autograd_grad_B, grad_B_anal, rtol=1e-4, atol=1e-6)
    # Assert repulsion: gradients push A and B in opposite directions
    assert torch.allclose(grad_A_anal, -grad_B_anal, rtol=1e-5)
