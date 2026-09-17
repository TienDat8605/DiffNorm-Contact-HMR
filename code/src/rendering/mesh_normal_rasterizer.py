"""
Differentiable Screen-Space Mesh Normal Rasterizer.
Provides classical triangular mesh surface normal rendering for Condition B baseline comparisons.
Projects SMPL faces to screen space with barycentric interpolation and differentiable soft depth compositing.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rendering.normal_rasterizer import RenderOutput


class DifferentiableMeshNormalRasterizer(nn.Module):
    """
    Differentiable tile-based triangle mesh normal rasterizer.
    Renders continuous surface normal maps, silhouette masks, and depth maps
    directly from 3D mesh vertices, vertex normals, and triangle topology.
    """
    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 256,
        tile_size: int = 64,
        depth_sharpness: float = 20.0,
        max_triangles_per_tile: int = 64
    ):
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.tile_size = tile_size
        self.depth_sharpness = depth_sharpness
        self.max_triangles_per_tile = max_triangles_per_tile

        # Precompute coordinate grid (H, W, 2)
        y, x = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij"
        )
        self.register_buffer("coords_x", x)
        self.register_buffer("coords_y", y)

    def forward(
        self,
        vertices_cam: torch.Tensor,       # (N, 3) mesh vertices in camera coordinates
        normals_cam: torch.Tensor,        # (N, 3) unit vertex normals in camera coordinates
        faces: torch.Tensor,               # (M, 3) triangle face vertex indices
        K: torch.Tensor,                   # (3, 3) camera intrinsics
        colors: Optional[torch.Tensor] = None  # (N, 3) optional vertex colors
    ) -> RenderOutput:
        """
        Renders normal map, silhouette mask, and depth map from posed 3D mesh.
        """
        device = vertices_cam.device
        z = vertices_cam[:, 2]

        # 1. Perspective projection
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        z_safe = z.clamp_min(0.1)

        u = fx * (vertices_cam[:, 0] / z_safe) + cx
        v = fy * (vertices_cam[:, 1] / z_safe) + cy
        pts_2d = torch.stack([u, v], dim=-1)  # (N, 2)

        # 2. Gather per-face attributes: (M, 3, ...)
        f_pts = pts_2d[faces]       # (M, 3, 2)
        f_z = z[faces]              # (M, 3)
        f_norm = normals_cam[faces]  # (M, 3, 3)
        f_col = colors[faces] if colors is not None else None  # (M, 3, 3)

        # 3. Backface culling in screen space (cross product of 2D screen edges)
        e1 = f_pts[:, 1] - f_pts[:, 0]
        e2 = f_pts[:, 2] - f_pts[:, 0]
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        front = (cross > 0.0) & (f_z.min(dim=-1).values > 0.1)
        front_idx = torch.where(front)[0]

        if len(front_idx) == 0:
            return RenderOutput(
                normals=torch.zeros(self.H, self.W, 3, device=device),
                rgb=torch.zeros(self.H, self.W, 3, device=device),
                mask=torch.zeros(self.H, self.W, device=device),
                depth=torch.zeros(self.H, self.W, device=device)
            )

        # Sort front-facing triangles front-to-back by camera depth
        f_z_mean = f_z[front_idx].mean(dim=-1)
        depth_order = torch.argsort(f_z_mean)
        front_sorted_idx = front_idx[depth_order]

        sub_pts = f_pts[front_sorted_idx]
        sub_z = f_z[front_sorted_idx]
        sub_norm = f_norm[front_sorted_idx]
        sub_col = f_col[front_sorted_idx] if f_col is not None else None

        # 2D Bounding boxes: (K, 2)
        bbox_min = sub_pts.min(dim=1).values
        bbox_max = sub_pts.max(dim=1).values

        # 4. Tile-based parallel rasterization
        full_normals = torch.zeros(self.H, self.W, 3, device=device)
        full_rgb = torch.zeros(self.H, self.W, 3, device=device)
        full_mask = torch.zeros(self.H, self.W, device=device)
        full_depth = torch.zeros(self.H, self.W, device=device)

        ts = self.tile_size
        for r in range(0, self.H, ts):
            r_end = min(r + ts, self.H)
            for c in range(0, self.W, ts):
                c_end = min(c + ts, self.W)

                # Find candidate triangles overlapping this screen tile
                in_tile = (
                    (bbox_max[:, 0] >= c) &
                    (bbox_min[:, 0] <= c_end) &
                    (bbox_max[:, 1] >= r) &
                    (bbox_min[:, 1] <= r_end)
                )
                t_idx = torch.where(in_tile)[0]
                if len(t_idx) == 0:
                    continue

                if len(t_idx) > self.max_triangles_per_tile:
                    t_idx = t_idx[:self.max_triangles_per_tile]

                t_pts = sub_pts[t_idx]    # (K_t, 3, 2)
                t_z = sub_z[t_idx]        # (K_t, 3)
                t_n = sub_norm[t_idx]     # (K_t, 3, 3)

                tile_x = self.coords_x[r:r_end, c:c_end]  # (th, tw)
                tile_y = self.coords_y[r:r_end, c:c_end]  # (th, tw)

                # Vectorized Barycentric Coordinates for all K_t triangles in tile
                p0 = t_pts[:, 0].view(-1, 1, 1, 2)
                p1 = t_pts[:, 1].view(-1, 1, 1, 2)
                p2 = t_pts[:, 2].view(-1, 1, 1, 2)

                denom = (
                    (p1[..., 1] - p2[..., 1]) * (p0[..., 0] - p2[..., 0]) +
                    (p2[..., 0] - p1[..., 0]) * (p0[..., 1] - p2[..., 1])
                ).clamp_min(1e-6)

                px = tile_x.unsqueeze(0)  # (1, th, tw)
                py = tile_y.unsqueeze(0)

                w0 = (
                    (p1[..., 1] - p2[..., 1]) * (px - p2[..., 0]) +
                    (p2[..., 0] - p1[..., 0]) * (py - p2[..., 1])
                ) / denom
                w1 = (
                    (p2[..., 1] - p0[..., 1]) * (px - p2[..., 0]) +
                    (p0[..., 0] - p2[..., 0]) * (py - p2[..., 1])
                ) / denom
                w2 = 1.0 - w0 - w1

                inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)  # (K_t, th, tw)
                if not torch.any(inside):
                    continue

                # Depth interpolation
                tri_z = (
                    w0 * t_z[:, 0].view(-1, 1, 1) +
                    w1 * t_z[:, 1].view(-1, 1, 1) +
                    w2 * t_z[:, 2].view(-1, 1, 1)
                )  # (K_t, th, tw)
                tri_z = torch.where(inside, tri_z, torch.full_like(tri_z, 1e4))

                # Differentiable softmin depth weighting over visible overlapping triangles
                z_min = tri_z.min(dim=0, keepdim=True).values
                weights = torch.where(
                    inside,
                    torch.exp(-self.depth_sharpness * (tri_z - z_min)),
                    torch.zeros_like(tri_z)
                )
                w_sum = weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
                norm_weights = weights / w_sum

                # Normal interpolation: sum over K_t
                tri_n = (
                    w0.unsqueeze(-1) * t_n[:, 0].view(-1, 1, 1, 3) +
                    w1.unsqueeze(-1) * t_n[:, 1].view(-1, 1, 1, 3) +
                    w2.unsqueeze(-1) * t_n[:, 2].view(-1, 1, 1, 3)
                )  # (K_t, th, tw, 3)

                n_tile = torch.sum(norm_weights.unsqueeze(-1) * tri_n, dim=0)
                m_tile = inside.any(dim=0).float()
                d_tile = torch.sum(norm_weights * tri_z, dim=0)

                if sub_col is not None:
                    t_c = sub_col[t_idx]
                    tri_c = (
                        w0.unsqueeze(-1) * t_c[:, 0].view(-1, 1, 1, 3) +
                        w1.unsqueeze(-1) * t_c[:, 1].view(-1, 1, 1, 3) +
                        w2.unsqueeze(-1) * t_c[:, 2].view(-1, 1, 1, 3)
                    )
                    c_tile = torch.sum(norm_weights.unsqueeze(-1) * tri_c, dim=0)
                    full_rgb[r:r_end, c:c_end] = c_tile

                full_normals[r:r_end, c:c_end] = n_tile
                full_mask[r:r_end, c:c_end] = m_tile
                full_depth[r:r_end, c:c_end] = d_tile

        # Normalize accumulated surface normals and gate by mask coverage
        norm_len = torch.norm(full_normals, p=2, dim=-1, keepdim=True)
        valid = (full_mask.unsqueeze(-1) > 0.5) & (norm_len > 1e-4)
        full_normals = torch.where(valid, full_normals / norm_len.clamp_min(1e-6), torch.zeros_like(full_normals))

        return RenderOutput(
            normals=full_normals,
            rgb=full_rgb.clamp(0.0, 1.0),
            mask=full_mask.clamp(0.0, 1.0),
            depth=full_depth
        )
