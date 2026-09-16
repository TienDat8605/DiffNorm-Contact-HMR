"""
Dual-Frequency Kinematic-Deformation Optimizer & Gradient Router.
Enforces strict spectral isolation: skeletal joint parameters theta are updated
exclusively by geometric normals and collision potentials, while local splat offsets
absorb high-frequency clothing folds with strictly detached joint gradients (d(L_deform)/d(theta) == 0).
"""

from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.mesh_graph import MeshGraph
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer, RenderOutput
from src.rendering.losses_rendering import SurfaceNormalLoss, SilhouetteIoULoss, PhotometricLoss


class StepOutput(NamedTuple):
    loss_total: float
    loss_normal: float
    loss_collision: float
    loss_mask: float
    loss_photo: float
    loss_lap: float
    diagnostics: Dict[str, float]
    rendered: RenderOutput


class DualFrequencyGradientRouter(nn.Module):
    """
    Manages the coupled kinematic and deformation optimization loops with strict
    gradient detachment of high-frequency photometric loss from skeletal joint angles.
    """
    def __init__(
        self,
        smpl: SMPLWrapper,
        gaussian_surface: TangentialGaussianSurface,
        collision_engine: GaussianSelfCollisionEngine,
        mesh_graph: MeshGraph,
        rasterizer: DifferentiableNormalRasterizer,
        lr_theta: float = 1e-2,
        lr_trans: float = 1e-2,
        lr_offsets: float = 5e-3,
        lr_scales: float = 5e-3,
        lr_colors: float = 1e-2,
        lambda_coll: float = 10.0,
        lambda_mask: float = 5.0,
        lambda_lap: float = 50.0,
        lambda_tight: float = 100.0,
    ):
        super().__init__()
        self.smpl = smpl
        self.gaussians = gaussian_surface
        self.collision_engine = collision_engine
        self.mesh_graph = mesh_graph
        self.rasterizer = rasterizer

        self.lambda_coll = lambda_coll
        self.lambda_mask = lambda_mask
        self.lambda_lap = lambda_lap
        self.lambda_tight = lambda_tight

        self.loss_normal_fn = SurfaceNormalLoss()
        self.loss_mask_fn = SilhouetteIoULoss()
        self.loss_photo_fn = PhotometricLoss()

        # Optimizer for Kinematics: theta, trans
        self.lr_theta = lr_theta
        self.lr_trans = lr_trans
        self.lr_offsets = lr_offsets
        self.lr_scales = lr_scales
        self.lr_colors = lr_colors

    def create_optimizers(
        self,
        theta_param: torch.Tensor,
        trans_param: torch.Tensor
    ) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """
        Creates independent AdamW optimizers for kinematic and deformation streams.
        """
        opt_kin = optim.AdamW([
            {"params": [theta_param], "lr": self.lr_theta},
            {"params": [trans_param], "lr": self.lr_trans},
        ])

        opt_deform = optim.AdamW([
            {"params": [self.gaussians.offsets], "lr": self.lr_offsets},
            {"params": [self.gaussians.log_tangent_scales, self.gaussians.quats], "lr": self.lr_scales},
            {"params": [self.gaussians.colors, self.gaussians.opacity_logits], "lr": self.lr_colors},
        ])
        return opt_kin, opt_deform

    def step(
        self,
        theta: torch.Tensor,                    # (1, 24, 3) or (24, 3)
        trans: torch.Tensor,                    # (1, 3) or (3,)
        K: torch.Tensor,                        # (3, 3) camera intrinsics
        target_normals: torch.Tensor,           # (H, W, 3) pseudo-GT normal map
        target_rgb: torch.Tensor,               # (H, W, 3) target color image
        target_mask: torch.Tensor,              # (H, W) target silhouette mask
        uncertainty: Optional[torch.Tensor],    # (H, W) normal uncertainty
        opt_kin: optim.Optimizer,
        opt_deform: optim.Optimizer,
        R_cam: Optional[torch.Tensor] = None
    ) -> StepOutput:
        """
        Executes one step of dual-frequency optimization.
        """
        if theta.dim() == 2:
            theta = theta.unsqueeze(0)
        if trans.dim() == 1:
            trans = trans.unsqueeze(0)

        opt_kin.zero_grad()
        opt_deform.zero_grad()

        # -------------------------------------------------------------
        # 1. Kinematic Stream: Forward Pass with Connected Graph
        # -------------------------------------------------------------
        smpl_out = self.smpl(theta=theta, trans=trans)
        verts = smpl_out["vertices"][0]  # (N, 3)

        # Centers with full autograd history to theta
        centers_cam = self.gaussians.get_centers(verts, detach_mesh=False)  # (N, 3)
        covs = self.gaussians.get_spatial_covariances()                     # (N, 3, 3)
        normals = self.gaussians.get_surface_normals(R_cam=R_cam)           # (N, 3)
        colors = self.gaussians.colors                                      # (N, 3)
        opacities = self.gaussians.get_opacities()                         # (N, 1)

        # 1a. Collision Loss (Analytical Overlap K_ij)
        loss_coll, coll_diag = self.collision_engine(centers_cam, covs)

        # 1b. Differentiable Normal Rasterization
        render_out = self.rasterizer(
            centers_cam=centers_cam,
            covs_cam=covs,
            normals_cam=normals,
            colors=colors,
            opacities=opacities,
            K=K
        )

        loss_norm = self.loss_normal_fn(
            render_out.normals, target_normals, mask=target_mask, uncertainty=uncertainty
        )
        loss_mask = self.loss_mask_fn(render_out.mask, target_mask)

        # Geometric Low-Frequency Loss (Drives theta, trans)
        loss_geom = loss_norm + self.lambda_coll * loss_coll + self.lambda_mask * loss_mask

        # Backpropagate geometric loss into both kinematics and offsets
        if loss_geom.requires_grad:
            loss_geom.backward(retain_graph=True)

        # -------------------------------------------------------------
        # 2. Deformation Stream: Forward Pass with Detached Mesh Vertices
        # -------------------------------------------------------------
        # Detach mesh vertices so d(loss_deform) / d(theta) == 0
        centers_detached = self.gaussians.get_centers(verts, detach_mesh=True)

        render_deform = self.rasterizer(
            centers_cam=centers_detached,
            covs_cam=covs,
            normals_cam=normals,
            colors=colors,
            opacities=opacities,
            K=K
        )

        loss_photo = self.loss_photo_fn(render_deform.rgb, target_rgb, mask=target_mask)
        loss_lap = self.mesh_graph.laplacian_loss(self.gaussians.offsets)
        loss_tight = self.mesh_graph.elastic_tether_loss(self.gaussians.offsets)

        loss_deform = loss_photo + self.lambda_lap * loss_lap + self.lambda_tight * loss_tight

        # Backpropagate deformation loss (affects ONLY offsets, colors, scales)
        if loss_deform.requires_grad:
            loss_deform.backward()

        # Step optimizers
        opt_kin.step()
        opt_deform.step()

        loss_total = (loss_geom + loss_deform).item()

        return StepOutput(
            loss_total=loss_total,
            loss_normal=loss_norm.item(),
            loss_collision=loss_coll.item(),
            loss_mask=loss_mask.item(),
            loss_photo=loss_photo.item(),
            loss_lap=loss_lap.item(),
            diagnostics=coll_diag,
            rendered=render_out
        )
