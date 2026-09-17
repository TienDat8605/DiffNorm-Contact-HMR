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
from src.optimization.nuisance_sensitivity import NuisanceSensitivityEstimator, SensitivityOutput
from src.optimization.capacity_scheduler import CapacityGatedScheduler, StageState


class StepOutput(NamedTuple):
    loss_total: float
    loss_normal: float
    loss_collision: float
    loss_mask: float
    loss_photo: float
    loss_lap: float
    diagnostics: Dict[str, float]
    rendered: RenderOutput
    sensitivity: Optional[SensitivityOutput] = None
    stage_state: Optional[StageState] = None


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
        sensitivity_estimator: Optional[NuisanceSensitivityEstimator] = None,
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
        self.sensitivity_estimator = sensitivity_estimator

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
        init_theta: Optional[torch.Tensor] = None,
        init_trans: Optional[torch.Tensor] = None,
        R_cam: Optional[torch.Tensor] = None,
        iteration: Optional[int] = None,
        scheduler: Optional[CapacityGatedScheduler] = None
    ) -> StepOutput:
        """
        Executes one step of dual-frequency optimization with optional capacity gating.
        """
        stage_state: Optional[StageState] = None
        is_locked: bool = False
        if scheduler is not None and iteration is not None:
            stage_state = scheduler.pre_step(iteration, opt_kin, opt_deform, self.gaussians)
            is_locked = stage_state.is_kinematic_locked

        theta_leaf = theta
        trans_leaf = trans
        theta_in = theta.unsqueeze(0) if theta.dim() == 2 else theta
        trans_in = trans.unsqueeze(0) if trans.dim() == 1 else trans

        opt_kin.zero_grad()
        opt_deform.zero_grad()

        # -------------------------------------------------------------
        # 1. Kinematic Stream: Forward Pass with Connected Graph
        # -------------------------------------------------------------
        smpl_out = self.smpl(theta=theta_in, trans=trans_in)
        verts = smpl_out["vertices"][0]  # (N, 3)
        mesh_normals = smpl_out["normals"][0]  # (N, 3)

        # Centers, covariances, and normals transformed consistently to camera frame
        # If locked, detach offsets so geometric normal loss only differentiates w.r.t. kinematics
        centers_cam = self.gaussians.get_centers(
            verts, detach_mesh=False, detach_offsets=is_locked, R_cam=R_cam
        )
        covs = self.gaussians.get_spatial_covariances(mesh_normals=mesh_normals, R_cam=R_cam)       # (N, 3, 3)
        normals = self.gaussians.get_surface_normals(mesh_normals=mesh_normals, R_cam=R_cam)       # (N, 3)
        colors_detached = self.gaussians.colors.detach()                                             # (N, 3)
        opacities_detached = self.gaussians.get_opacities().detach()                                # (N, 1)

        # 1a. Collision Loss (Analytical Overlap K_ij)
        loss_coll, coll_diag = self.collision_engine(centers_cam, covs.detach())

        # 1b. Differentiable Normal Rasterization
        render_out = self.rasterizer(
            centers_cam=centers_cam,
            covs_cam=covs.detach(),
            normals_cam=normals,
            colors=colors_detached,
            opacities=opacities_detached,
            K=K
        )

        loss_norm = self.loss_normal_fn(
            render_out.normals, target_normals, mask=target_mask, uncertainty=uncertainty
        )
        loss_mask = self.loss_mask_fn(render_out.mask, target_mask)

        # Geometric Low-Frequency Loss (Drives theta, trans)
        loss_geom = loss_norm + self.lambda_coll * loss_coll + self.lambda_mask * loss_mask

        # Backpropagate geometric loss into kinematics (and offsets if not locked)
        if loss_geom.requires_grad:
            loss_geom.backward()

        if is_locked and self.gaussians.offsets.grad is not None:
            self.gaussians.offsets.grad.zero_()

        # 1c. Nuisance-Aware Spectral Step Damping & Subspace Orthogonalization
        sens_out: Optional[SensitivityOutput] = None
        if not is_locked and self.sensitivity_estimator is not None and init_theta is not None and init_trans is not None:
            th_in = theta_leaf.squeeze(0) if theta_leaf.dim() == 3 else theta_leaf
            tr_in = trans_leaf.squeeze(0) if trans_leaf.dim() == 2 else trans_leaf
            init_th = init_theta.squeeze(0) if init_theta.dim() == 3 else init_theta
            init_tr = init_trans.squeeze(0) if init_trans.dim() == 2 else init_trans
            g_th = theta_leaf.grad.squeeze(0) if (theta_leaf.grad is not None and theta_leaf.grad.dim() == 3) else theta_leaf.grad
            g_tr = trans_leaf.grad.squeeze(0) if (trans_leaf.grad is not None and trans_leaf.grad.dim() == 2) else trans_leaf.grad

            if g_th is not None and g_tr is not None and self.gaussians.offsets.grad is not None:
                sens_out = self.sensitivity_estimator(
                    theta=th_in,
                    trans=tr_in,
                    init_theta=init_th,
                    init_trans=init_tr,
                    verts=verts,
                    joints=smpl_out["joints"][0],
                    grad_theta=g_th,
                    grad_trans=g_tr,
                    grad_offsets=self.gaussians.offsets.grad
                )

                theta_leaf.grad.copy_(sens_out.filtered_grad_theta.view_as(theta_leaf.grad))
                trans_leaf.grad.copy_(sens_out.filtered_grad_trans.view_as(trans_leaf.grad))
                self.gaussians.offsets.grad.copy_(sens_out.filtered_grad_offsets)

        # -------------------------------------------------------------
        # 2. Deformation Stream: Forward Pass with Detached Mesh Vertices & Normals
        # -------------------------------------------------------------
        if not is_locked:
            centers_detached = self.gaussians.get_centers(verts, detach_mesh=True)
            covs_detached = self.gaussians.get_spatial_covariances(mesh_normals=mesh_normals.detach(), R_cam=R_cam)
            normals_detached = self.gaussians.get_surface_normals(mesh_normals=mesh_normals.detach(), R_cam=R_cam)

            render_deform = self.rasterizer(
                centers_cam=centers_detached,
                covs_cam=covs_detached,
                normals_cam=normals_detached,
                colors=self.gaussians.colors,
                opacities=self.gaussians.get_opacities(),
                K=K
            )

            loss_photo = self.loss_photo_fn(render_deform.rgb, target_rgb, mask=target_mask)
            loss_lap = self.mesh_graph.laplacian_loss(self.gaussians.offsets)
            loss_tight = self.mesh_graph.elastic_tether_loss(self.gaussians.offsets)

            loss_deform = loss_photo + self.lambda_lap * loss_lap + self.lambda_tight * loss_tight

            # Backpropagate deformation loss (affects ONLY offsets, colors, scales)
            if loss_deform.requires_grad:
                loss_deform.backward()
        else:
            loss_photo = torch.tensor(0.0, device=verts.device)
            loss_lap = torch.tensor(0.0, device=verts.device)
            loss_deform = torch.tensor(0.0, device=verts.device)

        # Step optimizers
        opt_kin.step()
        if not is_locked:
            opt_deform.step()

        # Enforce physical bounding box on offsets if scheduler is present
        if scheduler is not None and iteration is not None:
            stage_state = scheduler.post_step(iteration, self.gaussians)

        loss_total = (loss_geom + loss_deform).item()

        return StepOutput(
            loss_total=loss_total,
            loss_normal=loss_norm.item(),
            loss_collision=loss_coll.item(),
            loss_mask=loss_mask.item(),
            loss_photo=loss_photo.item(),
            loss_lap=loss_lap.item(),
            diagnostics=coll_diag,
            rendered=RenderOutput(
                normals=render_out.normals.detach(),
                rgb=render_out.rgb.detach(),
                mask=render_out.mask.detach(),
                depth=render_out.depth.detach()
            ),
            sensitivity=sens_out,
            stage_state=stage_state
        )
