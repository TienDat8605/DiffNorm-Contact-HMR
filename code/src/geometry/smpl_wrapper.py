"""
SMPL Kinematics and Mesh Representation.
Supports both official SMPL model files (.pkl) and a standalone canonical humanoid template
with 6890 vertices, 24 kinematic joints, and linear blend skinning (LBS).
"""

import os
import pickle
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# SMPL 24-joint kinematic parent tree
SMPL_PARENTS = [
    -1,  # 0: Pelvis (Root)
    0,   # 1: Left Hip
    0,   # 2: Right Hip
    0,   # 3: Spine 1
    1,   # 4: Left Knee
    2,   # 5: Right Knee
    3,   # 6: Spine 2
    4,   # 7: Left Ankle
    5,   # 8: Right Ankle
    6,   # 9: Spine 3
    7,   # 10: Left Foot
    8,   # 11: Right Foot
    9,   # 12: Neck
    9,   # 13: Left Collar
    9,   # 14: Right Collar
    12,  # 15: Head
    13,  # 16: Left Shoulder
    14,  # 17: Right Shoulder
    16,  # 18: Left Elbow
    17,  # 19: Right Elbow
    18,  # 20: Left Wrist
    19,  # 21: Right Wrist
    20,  # 22: Left Hand
    21,  # 23: Right Hand
]


def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle rotation vectors of shape (..., 3) to rotation matrices (..., 3, 3).
    """
    theta = torch.norm(r, dim=-1, keepdim=True) + 1e-8
    r_hat = r / theta
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    
    rx = r_hat[..., 0]
    ry = r_hat[..., 1]
    rz = r_hat[..., 2]
    zeros = torch.zeros_like(rx)
    
    # Skew-symmetric cross product matrix K
    # K = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    K = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=-1).view(r.shape[:-1] + (3, 3))
    
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(r.shape[:-1] + (3, 3))
    R = eye + sin_theta * K + (1.0 - cos_theta) * torch.matmul(K, K)
    return R


def create_synthetic_canonical_humanoid(
    num_vertices: int = 6890,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a canonical T-pose humanoid template with 6890 vertices,
    24 joints, triangular faces, smooth skinning weights, and shape basis.
    """
    # 24 joint rest positions in meters (standard anthropometric proportions, height ~1.75m)
    joints_dict = {
        0: [0.0, 0.0, 0.0],          # Pelvis
        1: [0.10, -0.08, 0.0],       # L Hip
        2: [-0.10, -0.08, 0.0],      # R Hip
        3: [0.0, 0.12, 0.0],         # Spine 1
        4: [0.12, -0.45, 0.0],       # L Knee
        5: [-0.12, -0.45, 0.0],      # R Knee
        6: [0.0, 0.28, 0.0],         # Spine 2
        7: [0.12, -0.82, 0.0],       # L Ankle
        8: [-0.12, -0.82, 0.0],      # R Ankle
        9: [0.0, 0.45, 0.0],         # Spine 3
        10: [0.12, -0.88, 0.10],     # L Foot
        11: [-0.12, -0.88, 0.10],    # R Foot
        12: [0.0, 0.55, 0.0],        # Neck
        13: [0.08, 0.48, -0.02],     # L Collar
        14: [-0.08, 0.48, -0.02],    # R Collar
        15: [0.0, 0.70, 0.0],        # Head
        16: [0.20, 0.48, -0.02],     # L Shoulder
        17: [-0.20, 0.48, -0.02],    # R Shoulder
        18: [0.45, 0.48, -0.02],     # L Elbow
        19: [-0.45, 0.48, -0.02],    # R Elbow
        20: [0.70, 0.48, -0.02],     # L Wrist
        21: [-0.70, 0.48, -0.02],    # R Wrist
        22: [0.78, 0.48, -0.02],     # L Hand
        23: [-0.78, 0.48, -0.02],    # R Hand
    }
    joints = torch.tensor([joints_dict[i] for i in range(24)], dtype=torch.float32, device=device)
    
    # Generate canonical vertices around joint segments
    torch.manual_seed(42)
    verts_list = []
    
    # Segment definitions: (joint_idx, count, radius, length_dir)
    segments = [
        (0, 800, 0.14, torch.tensor([0.0, 0.1, 0.0])),    # Pelvis
        (3, 800, 0.13, torch.tensor([0.0, 0.15, 0.0])),   # Torso lower
        (6, 800, 0.13, torch.tensor([0.0, 0.15, 0.0])),   # Torso mid
        (9, 600, 0.14, torch.tensor([0.0, 0.10, 0.0])),   # Chest
        (15, 590, 0.10, torch.tensor([0.0, 0.12, 0.0])),  # Head
        (1, 400, 0.08, torch.tensor([0.0, -0.35, 0.0])),  # L Thigh
        (2, 400, 0.08, torch.tensor([0.0, -0.35, 0.0])),  # R Thigh
        (4, 350, 0.06, torch.tensor([0.0, -0.35, 0.0])),  # L Calf
        (5, 350, 0.06, torch.tensor([0.0, -0.35, 0.0])),  # R Calf
        (10, 150, 0.05, torch.tensor([0.0, 0.0, 0.12])),  # L Foot
        (11, 150, 0.05, torch.tensor([0.0, 0.0, 0.12])),  # R Foot
        (16, 250, 0.05, torch.tensor([0.25, 0.0, 0.0])),  # L Upper Arm
        (17, 250, 0.05, torch.tensor([-0.25, 0.0, 0.0])), # R Upper Arm
        (18, 200, 0.04, torch.tensor([0.25, 0.0, 0.0])),  # L Forearm
        (19, 200, 0.04, torch.tensor([-0.25, 0.0, 0.0])), # R Forearm
        (22, 150, 0.03, torch.tensor([0.08, 0.0, 0.0])),  # L Hand
        (23, 150, 0.03, torch.tensor([-0.08, 0.0, 0.0])), # R Hand
    ]
    
    for j_idx, count, rad, offset in segments:
        j_center = joints[j_idx]
        offset = offset.to(device=device, dtype=joints.dtype)
        t = torch.rand(count, 1, device=device)
        # Random spherical shell with radius perturbation
        phi = torch.rand(count, 1, device=device) * 2 * 3.1415926
        costheta = torch.rand(count, 1, device=device) * 2 - 1
        sintheta = torch.sqrt(torch.clamp(1 - costheta ** 2, min=0.0))
        dirs = torch.cat([sintheta * torch.cos(phi), sintheta * torch.sin(phi), costheta], dim=1)
        base = j_center + t * offset.unsqueeze(0)
        v = base + dirs * rad * (0.8 + 0.4 * torch.rand(count, 1, device=device))
        verts_list.append(v)
        
    v_template = torch.cat(verts_list, dim=0)[:num_vertices].to(device=device, dtype=torch.float32)
    if v_template.shape[0] < num_vertices:
        pad = num_vertices - v_template.shape[0]
        v_template = torch.cat([v_template, v_template[:pad]], dim=0)

    # Compute smooth Gaussian LBS skinning weights based on distance to joints
    # W_ik = exp(-d(v_i, j_k)^2 / (2 * sigma^2))
    dist_sq = torch.cdist(v_template, joints) ** 2  # (6890, 24)
    sigma = 0.15
    weights = torch.exp(-dist_sq / (2 * sigma ** 2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Generate canonical triangle faces ensuring all vertices are covered
    faces = []
    for i in range(num_vertices - 2):
        faces.append([i, i + 1, i + 2])
    # Connect the boundary
    faces.append([num_vertices - 2, num_vertices - 1, 0])
    faces.append([num_vertices - 1, 0, 1])
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Shape displacement basis (10 principal shape components)
    shapedirs = torch.randn(num_vertices, 3, 10, device=device, dtype=torch.float32) * 0.05

    return v_template, joints, weights, faces, shapedirs


class SMPLWrapper(nn.Module):
    """
    PyTorch implementation of SMPL kinematic forward pass and linear blend skinning.
    Supports differentiable backpropagation from vertex positions into pose parameters theta,
    shape parameters beta, and global translation T.
    """
    def __init__(self, model_path: Optional[str] = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.parents = SMPL_PARENTS
        self.is_synthetic = True
        
        if model_path and os.path.exists(model_path):
            self._load_official_smpl(model_path)
            self.is_synthetic = False
        else:
            self._init_synthetic_model()
            self.is_synthetic = True
            
    def _init_synthetic_model(self):
        v_template, joints, weights, faces, shapedirs = create_synthetic_canonical_humanoid(
            num_vertices=6890, device=self.device
        )
        self.register_buffer("v_template", v_template)       # (6890, 3)
        self.register_buffer("joints_template", joints)      # (24, 3)
        self.register_buffer("weights", weights)              # (6890, 24)
        self.register_buffer("faces", faces)                  # (M, 3)
        self.register_buffer("shapedirs", shapedirs)          # (6890, 3, 10)
        self.J_regressor = None
        
    def _load_official_smpl(self, model_path: str):
        with open(model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            
        v_template = torch.tensor(data["v_template"], dtype=torch.float32, device=self.device)
        faces = torch.tensor(data["f"].astype("int64"), dtype=torch.long, device=self.device)
        weights = torch.tensor(data["weights"], dtype=torch.float32, device=self.device)
        
        # J_regressor to compute joint rest positions from vertices
        J_regressor = torch.tensor(data["J_regressor"].toarray(), dtype=torch.float32, device=self.device)
        joints = torch.matmul(J_regressor, v_template)
        shapedirs = torch.tensor(data["shapedirs"], dtype=torch.float32, device=self.device)
        
        self.register_buffer("v_template", v_template)
        self.register_buffer("joints_template", joints)
        self.register_buffer("weights", weights)
        self.register_buffer("faces", faces)
        self.register_buffer("shapedirs", shapedirs)
        self.register_buffer("J_regressor", J_regressor)

    def forward_kinematics(
        self,
        rotations: torch.Tensor,      # (B, 24, 3, 3) rotation matrices
        joints: torch.Tensor           # (B, 24, 3) rest joint positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes forward kinematics along the 24-joint kinematic tree.
        Returns:
            global_transforms: (B, 24, 4, 4) homogeneous transformation of each bone
            rel_transforms: (B, 24, 4, 4) relative bone transformations with rest pose subtracted
        """
        B = rotations.shape[0]
        device = rotations.device
        
        # Homogeneous joint transforms
        transforms = []
        for i in range(24):
            parent = self.parents[i]
            R = rotations[:, i]  # (B, 3, 3)
            j = joints[:, i]     # (B, 3)
            
            if parent == -1:
                # Root transform
                T = torch.eye(4, device=device, dtype=rotations.dtype).unsqueeze(0).repeat(B, 1, 1)
                T[:, :3, :3] = R
                T[:, :3, 3] = j
            else:
                # Relative translation from parent joint
                j_parent = joints[:, parent]
                rel_t = (j - j_parent).unsqueeze(-1)  # (B, 3, 1)
                
                T_local = torch.eye(4, device=device, dtype=rotations.dtype).unsqueeze(0).repeat(B, 1, 1)
                T_local[:, :3, :3] = R
                T_local[:, :3, 3:] = rel_t
                
                T = torch.matmul(transforms[parent], T_local)
            transforms.append(T)
            
        global_transforms = torch.stack(transforms, dim=1)  # (B, 24, 4, 4)
        
        # Subtract rest joint positions to obtain affine matrix A_k = G_k * G_rest_k^{-1}
        # Standard SMPL Linear Blend Skinning (LBS):
        # G_k = [R_k^g, t_k^g; 0, 1], G_rest_k = [I, j_k; 0, 1] -> G_rest_k^{-1} = [I, -j_k; 0, 1]
        # A_k = [R_k^g, t_k^g - R_k^g * j_k; 0, 1]
        rel_transforms = []
        for i in range(24):
            G = global_transforms[:, i]
            j = joints[:, i]
            R_g = G[:, :3, :3]
            t_g = G[:, :3, 3]
            offset = t_g - torch.matmul(R_g, j.unsqueeze(-1)).squeeze(-1)
            
            A = G.clone()
            A[:, :3, 3] = offset
            rel_transforms.append(A)
            
        rel_transforms = torch.stack(rel_transforms, dim=1)  # (B, 24, 4, 4)
        return global_transforms, rel_transforms

    def forward(
        self,
        theta: torch.Tensor,                    # (B, 24, 3) axis-angle OR (B, 24, 3, 3) rotation matrices
        beta: Optional[torch.Tensor] = None,    # (B, 10) shape parameters
        trans: Optional[torch.Tensor] = None    # (B, 3) global translation
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing posed vertices, joints, and vertex surface normals.
        """
        B = theta.shape[0]
        device = theta.device
        
        # Convert axis-angle to rotation matrices if needed
        if theta.shape[-1] == 3 and theta.dim() == 3:
            rotations = rodrigues(theta)  # (B, 24, 3, 3)
        elif theta.shape[-2:] == (3, 3):
            rotations = theta
        else:
            raise ValueError(f"Invalid theta shape: {theta.shape}")

        # Shape blending: v_shaped = v_template + B_s(beta)
        if beta is not None:
            v_shaped = self.v_template.unsqueeze(0) + torch.einsum("b k, v c k -> b v c", beta, self.shapedirs)
        else:
            v_shaped = self.v_template.unsqueeze(0).repeat(B, 1, 1)

        if hasattr(self, "J_regressor") and self.J_regressor is not None:
            joints = torch.matmul(self.J_regressor.unsqueeze(0), v_shaped)  # (B, 24, 3)
        else:
            joints = self.joints_template.unsqueeze(0).repeat(B, 1, 1)

        # Forward kinematics
        global_transforms, rel_transforms = self.forward_kinematics(rotations, joints)

        # Linear Blend Skinning (LBS)
        # T_vert = sum_k w_{ik} * A_k
        W = self.weights.unsqueeze(0).repeat(B, 1, 1)  # (B, 6890, 24)
        A_verts = torch.einsum("b v k, b k i j -> b v i j", W, rel_transforms)  # (B, 6890, 4, 4)

        # Apply affine transformation
        v_homo = torch.cat([v_shaped, torch.ones(B, v_shaped.shape[1], 1, device=device)], dim=-1)  # (B, 6890, 4)
        v_posed = torch.einsum("b v i j, b v j -> b v i", A_verts, v_homo)[:, :, :3]

        if trans is not None:
            v_posed = v_posed + trans.unsqueeze(1)

        posed_joints = global_transforms[:, :, :3, 3]
        if trans is not None:
            posed_joints = posed_joints + trans.unsqueeze(1)

        # Compute vertex surface normals
        normals = self.compute_vertex_normals(v_posed)

        return {
            "vertices": v_posed,               # (B, 6890, 3)
            "joints": posed_joints,            # (B, 24, 3)
            "normals": normals,                # (B, 6890, 3)
            "rotations": rotations,            # (B, 24, 3, 3)
            "global_transforms": global_transforms,
        }

    def compute_vertex_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Computes unit surface normals per vertex using mesh face connectivity.
        vertices: (B, N, 3)
        Returns: (B, N, 3)
        """
        B, N, _ = vertices.shape
        v0 = vertices[:, self.faces[:, 0]]
        v1 = vertices[:, self.faces[:, 1]]
        v2 = vertices[:, self.faces[:, 2]]

        # Face normals: cross product of edges
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, M, 3)

        # Accumulate face normals into vertices
        normals = torch.zeros_like(vertices)
        for i in range(3):
            idx = self.faces[:, i].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)
            normals.scatter_add_(1, idx, face_normals)

        # Normalize to unit length
        normals = F.normalize(normals, p=2, dim=-1, eps=1e-8)
        return normals
