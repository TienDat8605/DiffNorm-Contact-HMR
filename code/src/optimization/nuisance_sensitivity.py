"""
Nuisance-Aware Sensitivity Estimator and Adaptive Step Controller.
Implements the Schur Complement Curvature Attenuation formulation from:
  nuisance_aware_3dgs_hmr_research_plan_2026_09_17.md

Evaluates the physical coupling between skeletal joint rotations theta and
Gaussian nuisance flexibility (offsets delta, scales, orientations).
Detects modes where splat deformations absorb visual residuals, computes
the mode retention ratio gamma_k in [0, 1], and damps contaminated kinematic steps
while anchoring ambiguous joints to the HMR 2.0 initializer.
"""

from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter, SEGMENT_TO_BONES
from src.geometry.mesh_graph import MeshGraph


class SensitivityOutput(NamedTuple):
    gamma_theta: torch.Tensor          # (24,) mode retention factors in [gamma_min, 1.0]
    gamma_trans: torch.Tensor          # (3,) translation retention factor
    rho_theta: torch.Tensor            # (24,) kinematic-nuisance collinearity in [0, 1]
    rho_trans: torch.Tensor            # (3,) translation-nuisance collinearity
    kappa_stiff: torch.Tensor          # (24,) deformation stiffness energy
    kappa_render: torch.Tensor         # (24,) rendering driving force energy
    filtered_grad_theta: torch.Tensor  # (24, 3) nuisance-damped joint gradients
    filtered_grad_trans: torch.Tensor  # (3,) nuisance-damped translation gradients
    filtered_grad_offsets: torch.Tensor# (N, 3) orthogonalized Gaussian offset gradients


class NuisanceSensitivityEstimator(nn.Module):
    """
    Computes local sensitivity metrics and applies mode-adaptive spectral damping
    to prevent Gaussian nuisance flexibility from corrupting skeletal pose.
    """
    def __init__(
        self,
        smpl: SMPLWrapper,
        segmenter: KinematicSegmenter,
        mesh_graph: MeshGraph,
        lambda_lap: float = 50.0,
        lambda_tight: float = 100.0,
        alpha: float = 0.05,
        beta: float = 1.0,
        gamma_min: float = 0.05,
        lambda_prior: float = 1.0,
    ):
        super().__init__()
        self.smpl = smpl
        self.segmenter = segmenter
        self.mesh_graph = mesh_graph
        self.lambda_lap = lambda_lap
        self.lambda_tight = lambda_tight
        self.alpha = alpha
        self.beta = beta
        self.gamma_min = gamma_min
        self.lambda_prior = lambda_prior

        # Map each of the 24 bones to its segment index (0 to 13)
        bone_to_segment = torch.zeros(24, dtype=torch.long)
        for seg_id, bones in SEGMENT_TO_BONES.items():
            for b in bones:
                bone_to_segment[b] = seg_id
        self.register_buffer("bone_to_segment", bone_to_segment)

    @torch.no_grad()
    def forward(
        self,
        theta: torch.Tensor,               # (24, 3) current pose
        trans: torch.Tensor,               # (3,) current translation
        init_theta: torch.Tensor,          # (24, 3) HMR 2.0 initializer
        init_trans: torch.Tensor,          # (3,) HMR 2.0 initial translation
        verts: torch.Tensor,               # (N, 3) posed mesh vertices in camera frame
        joints: torch.Tensor,              # (24, 3) posed joint locations in camera frame
        grad_theta: torch.Tensor,          # (24, 3) raw skeletal gradients
        grad_trans: torch.Tensor,          # (3,) raw translation gradients
        grad_offsets: torch.Tensor,        # (N, 3) raw Gaussian offset gradients
    ) -> SensitivityOutput:
        """
        Calculates the Schur complement attenuation ratio and filters the gradients.
        """
        device = theta.device
        N = verts.shape[0]
        K = 24
        weights = self.smpl.weights.to(device)  # (N, 24)

        # -------------------------------------------------------------
        # 1. Kinematic Vector Field per Joint U_k (N, 24, 3)
        # -------------------------------------------------------------
        # Compute unit rotation axis for each joint
        gnorm_theta = torch.norm(grad_theta, dim=-1, keepdim=True).clamp_min(1e-8)  # (24, 1)
        omega = grad_theta / gnorm_theta  # (24, 3)

        # Relative lever arm from joint center to vertices: (N, 24, 3)
        diff = verts.unsqueeze(1) - joints.unsqueeze(0)  # (N, 24, 3)
        omega_exp = omega.unsqueeze(0).expand(N, K, 3)

        # Tangential displacement under joint rotation: v_step = omega x (v - j)
        cross_disp = torch.linalg.cross(omega_exp, diff, dim=-1)  # (N, 24, 3)

        # Weight displacement by LBS blend weights
        U = weights.unsqueeze(-1) * cross_disp  # (N, 24, 3)
        U_norms = torch.norm(U, dim=(0, 2)).clamp_min(1e-8)  # (24,)
        hat_U = U / U_norms.view(1, K, 1)  # Normalized kinematic displacement fields (N, 24, 3)

        # -------------------------------------------------------------
        # 2. Deformation Stiffness Energy kappa_stiff (24,)
        # -------------------------------------------------------------
        # Evaluate Laplacian and tether penalty if Gaussians deform along hat_U
        U_src = hat_U[self.mesh_graph.edge_src]  # (E, 24, 3)
        U_dst = hat_U[self.mesh_graph.edge_dst]  # (E, 24, 3)
        lap_energy = torch.sum((U_src - U_dst) ** 2, dim=-1).mean(dim=0)  # (24,)
        tether_energy = torch.sum(hat_U ** 2, dim=-1).mean(dim=0)  # (24,)
        kappa_stiff = self.lambda_lap * lap_energy + self.lambda_tight * tether_energy  # (24,)

        # -------------------------------------------------------------
        # 3. Visual Driving Force Energy kappa_render (24,)
        # -------------------------------------------------------------
        kappa_render = (gnorm_theta.squeeze(-1) ** 2)  # (24,)

        # -------------------------------------------------------------
        # 4. Kinematic-Nuisance Gradient Collinearity rho_theta (24,)
        # -------------------------------------------------------------
        # Inner product between Gaussian offset gradient and kinematic vector field
        inner_prod = (hat_U * grad_offsets.unsqueeze(1)).sum(dim=(0, 2))  # (24,)
        grad_offsets_norm = torch.norm(grad_offsets).clamp_min(1e-8)
        rho_theta = (torch.abs(inner_prod) / grad_offsets_norm).clamp(0.0, 1.0)  # (24,)

        # -------------------------------------------------------------
        # 5. Schur Mode Retention Ratio gamma_theta (24,)
        # -------------------------------------------------------------
        # Harmonic attenuation between stiffness and rendering force
        gamma_schur = kappa_stiff / (kappa_stiff + self.alpha * kappa_render + 1e-8)  # (24,)
        
        # Penalize modes with high collinearity interference
        gamma_theta = torch.clamp(
            gamma_schur * (1.0 - self.beta * (rho_theta ** 2)),
            min=self.gamma_min,
            max=1.0
        )

        # -------------------------------------------------------------
        # 6. Global Translation Sensitivity
        # -------------------------------------------------------------
        gnorm_trans = torch.norm(grad_trans).clamp_min(1e-8)
        dir_trans = grad_trans / gnorm_trans  # (3,)
        kappa_render_trans = gnorm_trans ** 2
        kappa_stiff_trans = torch.tensor(self.lambda_tight, device=device)

        # Translation collinearity with offset gradients
        inner_trans = (grad_offsets * dir_trans.unsqueeze(0)).sum()
        rho_trans_val = (torch.abs(inner_trans) / (grad_offsets_norm * (N ** 0.5) + 1e-8)).clamp(0.0, 1.0)
        gamma_trans_val = torch.clamp(
            (kappa_stiff_trans / (kappa_stiff_trans + self.alpha * kappa_render_trans + 1e-8)) *
            (1.0 - self.beta * (rho_trans_val ** 2)),
            min=self.gamma_min,
            max=1.0
        )
        gamma_trans = gamma_trans_val.expand(3)
        rho_trans = rho_trans_val.expand(3)

        # -------------------------------------------------------------
        # 7. Gradient Filtering and Subspace Orthogonalization
        # -------------------------------------------------------------
        # A. Filtered Joint Gradients: mode-damped + anchored to initializer on contaminated modes
        # g_theta^* = gamma * g_theta + (1 - gamma) * lambda_prior * (theta - init_theta)
        prior_pull_theta = self.lambda_prior * (theta - init_theta)
        filtered_grad_theta = (
            gamma_theta.unsqueeze(-1) * grad_theta +
            (1.0 - gamma_theta.unsqueeze(-1)) * prior_pull_theta
        )

        # B. Filtered Translation Gradients
        prior_pull_trans = self.lambda_prior * (trans - init_trans)
        filtered_grad_trans = (
            gamma_trans * grad_trans +
            (1.0 - gamma_trans) * prior_pull_trans
        )

        # C. Filtered Gaussian Offset Gradients:
        # Subtract the collinear component corresponding to confident joints (gamma_theta > threshold)
        # to prevent Gaussians from stealing confident skeletal motion.
        subspace_coeffs = (gamma_theta * inner_prod).view(1, K, 1)  # (1, 24, 1)
        subspace_drift = (subspace_coeffs * hat_U).sum(dim=1)      # (N, 3)
        filtered_grad_offsets = grad_offsets - subspace_drift

        return SensitivityOutput(
            gamma_theta=gamma_theta.detach(),
            gamma_trans=gamma_trans.detach(),
            rho_theta=rho_theta.detach(),
            rho_trans=rho_trans.detach(),
            kappa_stiff=kappa_stiff.detach(),
            kappa_render=kappa_render.detach(),
            filtered_grad_theta=filtered_grad_theta,
            filtered_grad_trans=filtered_grad_trans,
            filtered_grad_offsets=filtered_grad_offsets,
        )
