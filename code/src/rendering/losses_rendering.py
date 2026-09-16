"""
Rendering Loss Functions for DiffNorm-Contact HMR.
Includes Cosine Surface Normal Loss with aleatoric uncertainty weighting,
Differentiable Silhouette IoU Loss, and Photometric L1/SSIM Loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceNormalLoss(nn.Module):
    """
    Computes cosine similarity loss between rendered surface normals N_hat
    and pseudo-ground-truth normals N_star, optionally weighted by uncertainty map C.
    L_normal = 1 - sum(C * (N_hat . N_star)) / (sum(C) + eps)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        n_rendered: torch.Tensor,               # (H, W, 3)
        n_target: torch.Tensor,                 # (H, W, 3)
        mask: Optional[torch.Tensor] = None,    # (H, W) in [0, 1]
        uncertainty: Optional[torch.Tensor] = None # (H, W) in [0, 1]
    ) -> torch.Tensor:
        # Dot product between unit normal vectors: (H, W)
        cos_sim = torch.sum(n_rendered * n_target, dim=-1)

        # Confidence weight: mask * (1 - uncertainty)
        weight = torch.ones_like(cos_sim)
        if mask is not None:
            weight = weight * mask
        if uncertainty is not None:
            weight = weight * (1.0 - uncertainty.clamp(0.0, 1.0))

        sum_weight = torch.sum(weight).clamp_min(self.eps)
        loss = 1.0 - (torch.sum(weight * cos_sim) / sum_weight)
        return loss


class SilhouetteIoULoss(nn.Module):
    """
    Computes differentiable Soft-IoU loss between rendered mask and target mask:
    L_mask = 1 - (intersection / (union + eps))
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, m_rendered: torch.Tensor, m_target: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(m_rendered * m_target)
        union = torch.sum(m_rendered) + torch.sum(m_target) - intersection
        iou = intersection / union.clamp_min(self.eps)
        return 1.0 - iou


class PhotometricLoss(nn.Module):
    """
    Computes masked L1 photometric loss + SSIM perceptual loss.
    """
    def __init__(self, ssim_weight: float = 0.2):
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(
        self,
        rgb_rendered: torch.Tensor,
        rgb_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = torch.abs(rgb_rendered - rgb_target)  # (H, W, 3)
        if mask is not None:
            mask_3d = mask.unsqueeze(-1)
            l1_loss = torch.sum(diff * mask_3d) / torch.sum(mask_3d).clamp_min(1.0)
        else:
            l1_loss = torch.mean(diff)

        # Simplified local structural patch similarity
        mu_x = F.avg_pool2d(rgb_rendered.permute(2, 0, 1).unsqueeze(0), kernel_size=7, stride=1, padding=3)
        mu_y = F.avg_pool2d(rgb_target.permute(2, 0, 1).unsqueeze(0), kernel_size=7, stride=1, padding=3)
        ssim_sim = (2 * mu_x * mu_y + 1e-4) / (mu_x ** 2 + mu_y ** 2 + 1e-4)
        ssim_loss = 1.0 - torch.mean(ssim_sim)

        return l1_loss + self.ssim_weight * ssim_loss
