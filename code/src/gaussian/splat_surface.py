"""
Tangential Gaussian Disk Representation and Parametric Surface Anchoring.
Constrains 3D Gaussians to flat elliptical disks aligned tangentially to the human body mesh,
computing 3D spatial covariances and camera-space surface normal vectors.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized unit quaternions [w, x, y, z] to 3x3 rotation matrices.
    q: (..., 4)
    Returns: (..., 3, 3)
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    r00 = 1 - 2 * (y ** 2 + z ** 2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x ** 2 + z ** 2)
    r12 = 2 * (y * z - x * w)
    
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x ** 2 + y ** 2)
    
    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).view(q.shape[:-1] + (3, 3))
    return R


class TangentialGaussianSurface(nn.Module):
    """
    Parametric 3D Gaussian Splats anchored to human mesh vertices with strict
    tangential disk constraints (s3 = tau * min(s1, s2)).
    """
    def __init__(
        self,
        num_splats: int = 6890,
        tau: float = 0.03,
        init_scale: float = 0.015,
        init_opacity: float = 0.95,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.num_splats = num_splats
        self.tau = tau
        self.device = device
        
        # Learnable local displacement offsets from mesh vertices: delta_i in R^3
        self.offsets = nn.Parameter(torch.zeros(num_splats, 3, device=device))
        
        # Learnable tangential scales: log(s1), log(s2)
        init_log_scale = torch.log(torch.tensor(init_scale, device=device))
        self.log_tangent_scales = nn.Parameter(
            init_log_scale.unsqueeze(0).repeat(num_splats, 2)
        )
        
        # Quaternions: initialized to identity [1, 0, 0, 0]
        init_quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_splats, 1)
        self.quats = nn.Parameter(init_quats)
        
        # Opacities: raw logits
        init_logit = torch.log(torch.tensor(init_opacity / (1.0 - init_opacity), device=device))
        self.opacity_logits = nn.Parameter(init_logit.unsqueeze(0).repeat(num_splats, 1))
        
        # Colors: spherical harmonics degree 0 (RGB diffuse)
        self.colors = nn.Parameter(torch.ones(num_splats, 3, device=device) * 0.5)

    def get_scales(self) -> torch.Tensor:
        """
        Returns full 3D scales [s1, s2, s3] where s3 = tau * min(s1, s2).
        Returns: (N, 3)
        """
        st = torch.exp(self.log_tangent_scales)  # (N, 2)
        s1, s2 = st[:, 0], st[:, 1]
        s3 = self.tau * torch.minimum(s1, s2)   # (N,)
        scales = torch.stack([s1, s2, s3], dim=-1)  # (N, 3)
        return scales

    def get_rotation_matrices(self) -> torch.Tensor:
        """
        Returns 3x3 rotation matrices R_i for each splat.
        Returns: (N, 3, 3)
        """
        return quaternion_to_rotation_matrix(self.quats)

    def get_surface_normals(
        self,
        mesh_normals: Optional[torch.Tensor] = None,
        R_cam: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the unit surface normal vector of each Gaussian disk.
        If mesh_normals (N, 3) are provided from SMPLWrapper, the base splat
        orientation is anchored directly to the posed vertex normals, ensuring
        that normal gradients backpropagate directly through face cross-products
        into skeletal joint angles theta.
        If mesh_normals is None, falls back to the third column of R_i (R_i * [0, 0, 1]^T).
        If R_cam is provided (3, 3), transforms normals into camera coordinates.
        Returns: (N, 3)
        """
        R = self.get_rotation_matrices()  # (N, 3, 3)
        if mesh_normals is not None:
            # Rotate base vertex normal by local splat rotation offset
            normals_world = torch.einsum("n i j, n j -> n i", R, mesh_normals)
            normals_world = F.normalize(normals_world, p=2, dim=-1, eps=1e-8)
        else:
            # Third column: R[:, :, 2]
            normals_world = R[:, :, 2]  # (N, 3)
            normals_world = F.normalize(normals_world, p=2, dim=-1, eps=1e-8)
        
        if R_cam is not None:
            # normals_cam = normals_world @ R_cam^T
            normals_cam = torch.matmul(normals_world, R_cam.t())
            return F.normalize(normals_cam, p=2, dim=-1, eps=1e-8)
        return normals_world

    def get_spatial_covariances(self) -> torch.Tensor:
        """
        Computes the 3D spatial covariance matrix Sigma_i = R S S^T R^T.
        Returns: (N, 3, 3)
        """
        R = self.get_rotation_matrices()  # (N, 3, 3)
        S = torch.diag_embed(self.get_scales())  # (N, 3, 3)
        M = torch.matmul(R, S)  # (N, 3, 3)
        covs = torch.matmul(M, M.transpose(-1, -2))  # (N, 3, 3)
        return covs

    def get_opacities(self) -> torch.Tensor:
        """Returns splat opacities in [0, 1]. Shape: (N, 1)"""
        return torch.sigmoid(self.opacity_logits)

    def get_centers(self, mesh_vertices: torch.Tensor, detach_mesh: bool = False) -> torch.Tensor:
        """
        Computes 3D Gaussian centers mu_i = v_i + delta_i.
        If detach_mesh is True, detaches mesh_vertices from autograd graph.
        mesh_vertices: (B, N, 3) or (N, 3)
        Returns: Same shape as mesh_vertices
        """
        v = mesh_vertices.detach() if detach_mesh else mesh_vertices
        return v + self.offsets.unsqueeze(0) if v.dim() == 3 else v + self.offsets
