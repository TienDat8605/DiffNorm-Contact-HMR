"""
Tests for Gaussian Collision Repulsion Dynamics.
Simulates two interpenetrating Gaussian clusters and verifies that gradient descent
pushes them apart, monotonically reducing penetration.
"""

import pytest
import torch
import torch.optim as optim

from src.physics.gaussian_convolution import compute_gaussian_overlap_matrix


def test_collision_repulsion_dynamics():
    device = torch.device("cpu")
    # Cluster A at origin, Cluster B overlapping with A
    mu_A = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], device=device, requires_grad=True)
    cov_A = (0.03 ** 2) * torch.eye(3, device=device).unsqueeze(0).repeat(2, 1, 1)

    mu_B = torch.tensor([[0.01, 0.01, 0.0], [0.0, 0.01, 0.0]], device=device, requires_grad=True)
    cov_B = (0.03 ** 2) * torch.eye(3, device=device).unsqueeze(0).repeat(2, 1, 1)

    # Adam optimizer with appropriate step size for repulsion
    optimizer = optim.Adam([mu_A, mu_B], lr=0.01)

    initial_overlap = compute_gaussian_overlap_matrix(mu_A, cov_A, mu_B, cov_B).sum().item()
    assert initial_overlap > 1e-4

    prev_overlap = initial_overlap
    for step in range(25):
        optimizer.zero_grad()
        K = compute_gaussian_overlap_matrix(mu_A, cov_A, mu_B, cov_B)
        loss = torch.sum(K) * 1000.0  # Scale loss to provide active gradient steps
        loss.backward()
        optimizer.step()

        curr_overlap = (loss / 1000.0).item()
        assert curr_overlap <= prev_overlap + 1e-5
        prev_overlap = curr_overlap

    # Final overlap should be significantly lower than initial
    assert prev_overlap < initial_overlap * 0.4
    # Distance between cluster centers should increase
    dist_after = torch.norm(mu_A.mean(dim=0) - mu_B.mean(dim=0)).item()
    assert dist_after > 0.015
