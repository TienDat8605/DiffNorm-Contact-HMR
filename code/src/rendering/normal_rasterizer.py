"""
Differentiable Screen-Space Normal and Color Tile Rasterizer.
Projects 3D Gaussian disks into screen space and renders continuous
surface normal maps, color images, and silhouette masks via tiled front-to-back alpha-compositing.
"""

from typing import Dict, NamedTuple, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RenderOutput(NamedTuple):
    normals: torch.Tensor       # (H, W, 3) unit normal map in camera space
    rgb: torch.Tensor           # (H, W, 3) rendered color in [0, 1]
    mask: torch.Tensor          # (H, W) silhouette accumulation in [0, 1]
    depth: torch.Tensor         # (H, W) expected depth map


class DifferentiableNormalRasterizer(nn.Module):
    """
    Differentiable tile-based rasterizer for 3D Gaussian disks rendering surface normal maps,
    photometric images, and silhouette masks without in-place slice mutation.
    """
    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 256,
        tile_size: int = 32,
        anti_aliasing_filter: float = 0.3,
        max_splats_per_tile: int = 40
    ):
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.tile_size = tile_size
        self.filter_var = anti_aliasing_filter
        self.max_splats = max_splats_per_tile
        
        # Precompute coordinate grid (H, W, 2) where coords[..., 0] is X (col) and coords[..., 1] is Y (row)
        y, x = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij"
        )
        self.register_buffer("coords", torch.stack([x, y], dim=-1))

    def project_gaussians_to_2d(
        self,
        centers_cam: torch.Tensor,       # (N, 3) in camera coordinates
        covs_cam: torch.Tensor,          # (N, 3, 3) 3D covariances in camera coordinates
        K: torch.Tensor                  # (3, 3) camera intrinsics
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects 3D Gaussians to 2D screen space:
        mu_2d = [fx * x / z + cx, fy * y / z + cy]
        Sigma_2d = J * Sigma_3d * J^T + filter_var * I
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x, y, z = centers_cam[:, 0], centers_cam[:, 1], centers_cam[:, 2]
        z_safe = z.clamp_min(0.1)
        z_sq = z_safe ** 2

        u = fx * x / z_safe + cx
        v = fy * y / z_safe + cy
        mu_2d = torch.stack([u, v], dim=-1)  # (N, 2)

        # Jacobian J of perspective projection
        zeros = torch.zeros_like(x)
        J = torch.stack([
            fx / z_safe, zeros, -fx * x / z_sq,
            zeros, fy / z_safe, -fy * y / z_sq
        ], dim=-1).view(-1, 2, 3)  # (N, 2, 3)

        # 2D screen covariance Sigma_2d = J * Sigma_3d * J^T
        covs_2d = torch.matmul(J, torch.matmul(covs_cam, J.transpose(-1, -2)))  # (N, 2, 2)

        # Add low-pass anti-aliasing filter
        eye = torch.eye(2, device=centers_cam.device, dtype=centers_cam.dtype).unsqueeze(0)
        covs_2d = covs_2d + self.filter_var * eye

        # Precompute 2D inverse and determinant
        det = (covs_2d[:, 0, 0] * covs_2d[:, 1, 1] - covs_2d[:, 0, 1] * covs_2d[:, 1, 0]).clamp_min(1e-6)
        inv_cov = torch.zeros_like(covs_2d)
        inv_cov[:, 0, 0] = covs_2d[:, 1, 1] / det
        inv_cov[:, 1, 1] = covs_2d[:, 0, 0] / det
        inv_cov[:, 0, 1] = -covs_2d[:, 0, 1] / det
        inv_cov[:, 1, 0] = -covs_2d[:, 1, 0] / det

        # Conservative 3-sigma bounding radius in pixels
        radii = torch.sqrt(torch.maximum(covs_2d[:, 0, 0], covs_2d[:, 1, 1])).clamp_min(1.0) * 3.0

        return mu_2d, inv_cov, radii, z_safe

    def forward(
        self,
        centers_cam: torch.Tensor,       # (N, 3) in camera coordinates
        covs_cam: torch.Tensor,          # (N, 3, 3) in camera coordinates
        normals_cam: torch.Tensor,       # (N, 3) unit normals in camera space
        colors: torch.Tensor,            # (N, 3) RGB colors in [0, 1]
        opacities: torch.Tensor,         # (N, 1) opacities in [0, 1]
        K: torch.Tensor                  # (3, 3) camera intrinsics
    ) -> RenderOutput:
        """
        Renders normal map, RGB image, and silhouette mask via tiled front-to-back compositing.
        """
        device = centers_cam.device
        mu_2d, inv_cov, radii, depths = self.project_gaussians_to_2d(centers_cam, covs_cam, K)

        # Filter splats strictly behind camera
        valid = (depths > 0.1) & (mu_2d[:, 0] >= -self.W * 0.2) & (mu_2d[:, 0] <= self.W * 1.2) & (mu_2d[:, 1] >= -self.H * 0.2) & (mu_2d[:, 1] <= self.H * 1.2)
        valid_idx = torch.where(valid)[0]

        if len(valid_idx) == 0:
            zeros_3d = torch.zeros(self.H, self.W, 3, device=device)
            zeros_2d = torch.zeros(self.H, self.W, device=device)
            return RenderOutput(normals=zeros_3d, rgb=zeros_3d, mask=zeros_2d, depth=zeros_2d)

        # Pre-sort active splats front-to-back by camera depth
        valid_depths = depths[valid_idx]
        sorted_order = torch.argsort(valid_depths)
        ordered_splat_indices = valid_idx[sorted_order]

        # Splat attributes in sorted order
        mu_s = mu_2d[ordered_splat_indices]          # (S, 2)
        inv_cov_s = inv_cov[ordered_splat_indices]    # (S, 2, 2)
        radii_s = radii[ordered_splat_indices]        # (S,)
        normals_s = normals_cam[ordered_splat_indices]# (S, 3)
        colors_s = colors[ordered_splat_indices]      # (S, 3)
        opacities_s = opacities[ordered_splat_indices]# (S, 1)
        depths_s = depths[ordered_splat_indices]      # (S,)

        # Tiled accumulation without in-place tensor mutation
        row_blocks_rgb = []
        row_blocks_norm = []
        row_blocks_mask = []
        row_blocks_depth = []

        ts = self.tile_size
        for r in range(0, self.H, ts):
            r_end = min(r + ts, self.H)
            col_blocks_rgb = []
            col_blocks_norm = []
            col_blocks_mask = []
            col_blocks_depth = []

            for c in range(0, self.W, ts):
                c_end = min(c + ts, self.W)
                th = r_end - r
                tw = c_end - c
                tile_coords = self.coords[r:r_end, c:c_end]  # (th, tw, 2)

                # Tile bounding box test
                in_tile = (
                    (mu_s[:, 0] + radii_s >= c) &
                    (mu_s[:, 0] - radii_s <= c_end) &
                    (mu_s[:, 1] + radii_s >= r) &
                    (mu_s[:, 1] - radii_s <= r_end)
                )
                tile_splat_idx = torch.where(in_tile)[0][:self.max_splats]

                if len(tile_splat_idx) == 0:
                    col_blocks_rgb.append(torch.zeros(th, tw, 3, device=device))
                    col_blocks_norm.append(torch.zeros(th, tw, 3, device=device))
                    col_blocks_mask.append(torch.zeros(th, tw, device=device))
                    col_blocks_depth.append(torch.zeros(th, tw, device=device))
                    continue

                # Vectorized accumulation across all active splats in the tile simultaneously
                K_cnt = len(tile_splat_idx)
                sub_mu = mu_s[tile_splat_idx]            # (K, 2)
                sub_inv = inv_cov_s[tile_splat_idx]      # (K, 2, 2)
                sub_opa = opacities_s[tile_splat_idx]    # (K, 1)
                sub_col = colors_s[tile_splat_idx]       # (K, 3)
                sub_nor = normals_s[tile_splat_idx]      # (K, 3)
                sub_dep = depths_s[tile_splat_idx]       # (K,)

                # (K, th, tw, 2)
                diff = tile_coords.unsqueeze(0) - sub_mu.unsqueeze(1).unsqueeze(2)
                maha = (
                    diff[..., 0] ** 2 * sub_inv[:, 0, 0].view(K_cnt, 1, 1) +
                    diff[..., 1] ** 2 * sub_inv[:, 1, 1].view(K_cnt, 1, 1) +
                    2.0 * diff[..., 0] * diff[..., 1] * sub_inv[:, 0, 1].view(K_cnt, 1, 1)
                )  # (K, th, tw)

                g_val = torch.exp(-0.5 * maha.clamp_max(16.0)).unsqueeze(-1)  # (K, th, tw, 1)
                alpha = sub_opa.view(K_cnt, 1, 1, 1) * g_val                   # (K, th, tw, 1)

                # Front-to-back parallel transmittance using cumprod
                one_minus_alpha = (1.0 - alpha).clamp_min(0.0)
                pad = torch.ones(1, th, tw, 1, device=device, dtype=alpha.dtype)
                one_minus_alpha_shifted = torch.cat([pad, one_minus_alpha[:-1]], dim=0)
                t_tile = torch.cumprod(one_minus_alpha_shifted, dim=0)          # (K, th, tw, 1)
                w = alpha * t_tile                                             # (K, th, tw, 1)

                c_tile = torch.sum(w * sub_col.view(K_cnt, 1, 1, 3), dim=0)
                n_tile = torch.sum(w * sub_nor.view(K_cnt, 1, 1, 3), dim=0)
                m_tile = torch.sum(w, dim=0).squeeze(-1)
                d_tile = torch.sum(w * sub_dep.view(K_cnt, 1, 1, 1), dim=0).squeeze(-1)

                col_blocks_rgb.append(c_tile)
                col_blocks_norm.append(n_tile)
                col_blocks_mask.append(m_tile)
                col_blocks_depth.append(d_tile)

            row_blocks_rgb.append(torch.cat(col_blocks_rgb, dim=1))
            row_blocks_norm.append(torch.cat(col_blocks_norm, dim=1))
            row_blocks_mask.append(torch.cat(col_blocks_mask, dim=1))
            row_blocks_depth.append(torch.cat(col_blocks_depth, dim=1))

        full_rgb = torch.cat(row_blocks_rgb, dim=0)
        full_norm = torch.cat(row_blocks_norm, dim=0)
        full_mask = torch.cat(row_blocks_mask, dim=0)
        full_depth = torch.cat(row_blocks_depth, dim=0)

        # Normalize accumulated surface normals
        norm_len = torch.norm(full_norm, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        full_norm = full_norm / norm_len

        return RenderOutput(
            normals=full_norm,
            rgb=full_rgb.clamp(0.0, 1.0),
            mask=full_mask.clamp(0.0, 1.0),
            depth=full_depth
        )
