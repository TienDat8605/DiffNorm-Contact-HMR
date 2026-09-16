"""
Exact Closed-Form Volumetric Gaussian Overlap Integral Engine.
Computes the analytical cross-correlation integral K_ij between 3D Gaussians
and its exact repulsive gradients.
"""

from typing import Tuple
import math
import torch
import torch.nn.functional as F

SQRT_2PI_CUBE = (2.0 * math.pi) ** 1.5


def compute_gaussian_overlap_matrix(
    mu_A: torch.Tensor,   # (M, 3) centers of segment A
    cov_A: torch.Tensor,  # (M, 3, 3) covariances of segment A
    mu_B: torch.Tensor,   # (P, 3) centers of segment B
    cov_B: torch.Tensor,  # (P, 3, 3) covariances of segment B
    distance_cutoff: float = 0.25  # In meters; beyond this, overlap is negligible (< 1e-7)
) -> torch.Tensor:
    """
    Computes the exact volumetric overlap matrix K_ij between Gaussians in segment A and segment B:
    K_ij = (2*pi)^(3/2) * sqrt(|Sigma_i| * |Sigma_j| / |Sigma_i + Sigma_j|) * exp(-0.5 * d^T (Sigma_i + Sigma_j)^(-1) d)
    Returns: (M, P) matrix of analytical overlap integrals.
    """
    M = mu_A.shape[0]
    P = mu_B.shape[0]
    device = mu_A.device
    
    if M == 0 or P == 0:
        return torch.zeros(M, P, device=device, dtype=mu_A.dtype)

    # 1. Compute pairwise center displacement d_ij = mu_A_i - mu_B_j
    # (M, 1, 3) - (1, P, 3) -> (M, P, 3)
    diff = mu_A.unsqueeze(1) - mu_B.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (M, P)

    # Fast distance culling mask
    mask = dist_sq < (distance_cutoff ** 2)
    if not torch.any(mask):
        return torch.zeros(M, P, device=device, dtype=mu_A.dtype)

    # Filter active candidate pairs to avoid computing full M x P 3x3 matrix inverses
    active_i, active_j = torch.where(mask)
    if len(active_i) == 0:
        return torch.zeros(M, P, device=device, dtype=mu_A.dtype)

    # Gather active Gaussians in float32 for exact 3x3 matrix determinants and solves
    mu_a_act = mu_A[active_i].float()       # (K, 3)
    mu_b_act = mu_B[active_j].float()       # (K, 3)
    cov_a_act = cov_A[active_i].float()     # (K, 3, 3)
    cov_b_act = cov_B[active_j].float()     # (K, 3, 3)
    diff_act = mu_a_act - mu_b_act          # (K, 3)

    # 2. Joint covariance Sigma_ij = Sigma_i + Sigma_j
    joint_cov = cov_a_act + cov_b_act       # (K, 3, 3)

    # 3. Determinants
    det_a = torch.linalg.det(cov_a_act).clamp_min(1e-12)     # (K,)
    det_b = torch.linalg.det(cov_b_act).clamp_min(1e-12)     # (K,)
    det_joint = torch.linalg.det(joint_cov).clamp_min(1e-12) # (K,)

    # Prefactor: (2*pi)^(3/2) * sqrt(det_a * det_b / det_joint)
    prefactor = SQRT_2PI_CUBE * torch.sqrt((det_a * det_b) / det_joint)  # (K,)

    # 4. Mahalanobis term: diff^T * (joint_cov)^(-1) * diff
    # Solves (joint_cov) * x = diff -> x = joint_cov^(-1) * diff
    # Use torch.linalg.solve for speed and numerical stability
    sol = torch.linalg.solve(joint_cov, diff_act.unsqueeze(-1)).squeeze(-1)  # (K, 3)
    mahalanobis = torch.sum(diff_act * sol, dim=-1)  # (K,)

    # 5. Overlap value K_act = prefactor * exp(-0.5 * mahalanobis)
    k_act = prefactor * torch.exp(-0.5 * mahalanobis.clamp_max(50.0))  # (K,)

    # Scatter back into (M, P) matrix
    K_matrix = torch.zeros(M, P, device=device, dtype=mu_A.dtype)
    K_matrix[active_i, active_j] = k_act.to(mu_A.dtype)

    return K_matrix


def compute_analytical_repulsion_gradients(
    mu_A: torch.Tensor,
    cov_A: torch.Tensor,
    mu_B: torch.Tensor,
    cov_B: torch.Tensor,
    K_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes analytical gradients of the overlap sum w.r.t. mu_A and mu_B:
    grad_mu_A = - K_ij * (Sigma_i + Sigma_j)^(-1) * (mu_i - mu_j)
    grad_mu_B = - grad_mu_A
    Returns:
        grad_A: (M, 3)
        grad_B: (P, 3)
    """
    active_i, active_j = torch.where(K_matrix > 1e-8)
    M, P = K_matrix.shape
    device = mu_A.device

    grad_A = torch.zeros_like(mu_A)
    grad_B = torch.zeros_like(mu_B)

    if len(active_i) == 0:
        return grad_A, grad_B

    k_vals = K_matrix[active_i, active_j].float()  # (K,)
    diff_act = (mu_A[active_i] - mu_B[active_j]).float()  # (K, 3)
    joint_cov = (cov_A[active_i] + cov_B[active_j]).float()  # (K, 3, 3)

    sol = torch.linalg.solve(joint_cov, diff_act.unsqueeze(-1)).squeeze(-1)  # (K, 3)
    pair_grad = (- k_vals.unsqueeze(-1) * sol).to(mu_A.dtype)  # (K, 3)

    grad_A.index_add_(0, active_i, pair_grad)
    grad_B.index_add_(0, active_j, -pair_grad)

    return grad_A, grad_B
