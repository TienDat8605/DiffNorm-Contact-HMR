"""
Evaluation Metrics for Human Mesh Recovery and Contact Physics.
Calculates MPJPE, PA-MPJPE (Procrustes-aligned), PVE, and Volumetric Penetration (V_pen).
"""

from typing import Tuple
import torch


def compute_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """
    Mean Per Joint Position Error (MPJPE) after root-joint alignment.
    pred_joints: (..., J, 3) in meters
    gt_joints: (..., J, 3) in meters
    Returns: error in millimeters (mm)
    """
    # Root joint is index 0 (Pelvis)
    pred_aligned = pred_joints - pred_joints[..., :1, :]
    gt_aligned = gt_joints - gt_joints[..., :1, :]
    dist = torch.norm(pred_aligned - gt_aligned, dim=-1)  # (..., J)
    return (dist.mean() * 1000.0).item()


def compute_single_pa_mpjpe(pred_j: torch.Tensor, gt_j: torch.Tensor) -> float:
    """
    Computes Umeyama Procrustes alignment for a single sample of shape (J, 3).
    """
    p = pred_j.view(-1, 3)
    g = gt_j.view(-1, 3)

    mu_p = p.mean(dim=0, keepdim=True)
    mu_g = g.mean(dim=0, keepdim=True)

    p_c = p - mu_p
    g_c = g - mu_g

    # Covariance matrix H = p_c^T * g_c
    H = torch.matmul(p_c.t(), g_c)
    U, S, Vt = torch.linalg.svd(H)

    R = torch.matmul(Vt.t(), U.t())
    d = torch.ones(3, device=p.device, dtype=p.dtype)

    # Reflection correction: adjust sign of last singular value in trace
    if torch.linalg.det(R) < 0:
        d[2] = -1.0
        Vt_mod = Vt.clone()
        Vt_mod[-1, :] *= -1
        R = torch.matmul(Vt_mod.t(), U.t())

    var_p = torch.sum(p_c ** 2)
    scale = torch.sum(d * S) / var_p.clamp_min(1e-8)

    p_transformed = scale * torch.matmul(p_c, R.t()) + mu_g
    dist = torch.norm(p_transformed - g, dim=-1)
    return (dist.mean() * 1000.0).item()


def compute_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """
    Procrustes-Aligned Mean Per Joint Position Error (PA-MPJPE).
    Applies optimal rigid rotation, translation, and scale (Umeyama algorithm) per sample.
    pred_joints: (..., J, 3)
    gt_joints: (..., J, 3)
    Returns: mean error in millimeters (mm)
    """
    if pred_joints.dim() == 2:
        return compute_single_pa_mpjpe(pred_joints, gt_joints)

    pred_flat = pred_joints.view(-1, pred_joints.shape[-2], pred_joints.shape[-1])
    gt_flat = gt_joints.view(-1, gt_joints.shape[-2], gt_joints.shape[-1])

    errors = [
        compute_single_pa_mpjpe(p, g)
        for p, g in zip(pred_flat, gt_flat)
    ]
    return sum(errors) / len(errors)


def compute_pve(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> float:
    """
    Per-Vertex Error (PVE) in millimeters.
    """
    dist = torch.norm(pred_verts - gt_verts, dim=-1)
    return (dist.mean() * 1000.0).item()


def compute_penetration_volume(
    centers: torch.Tensor,
    covariances: torch.Tensor,
    non_adjacent_pairs: list,
    segmenter
) -> float:
    """
    Approximates volumetric self-penetration in cm^3 from Gaussian overlap.
    1 cubic meter = 1,000,000 cm^3.
    """
    total_vol = 0.0
    for seg_a, seg_b in non_adjacent_pairs:
        idx_a = segmenter.get_segment_indices(seg_a)
        idx_b = segmenter.get_segment_indices(seg_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        pts_a = centers[idx_a]
        pts_b = centers[idx_b]
        # Check pairwise distances below 6cm (0.06m)
        diff = pts_a.unsqueeze(1) - pts_b.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        interpenetrating = dist < 0.06
        if torch.any(interpenetrating):
            # Volume of intersecting spheres of radius r = 0.03m
            overlap_depth = (0.06 - dist[interpenetrating]).clamp_min(0.0)
            vol_m3 = torch.sum(overlap_depth ** 3)
            total_vol += (vol_m3.item() * 1e6)  # Convert to cm^3
    return total_vol
