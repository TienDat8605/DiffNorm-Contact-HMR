"""
Capacity-Gated Stage Scheduler for Nuisance-Aware 3DGS Articulated Pose Refinement.

Formalizes the two-stage refinement protocol derived from the Curvature Dilution Theorem:
  Stage 1 (Kinematic Locking, t < T_switch):
    Gaussian nuisance offsets are strictly locked (delta = 0, J_a = 0).
    The Hessian reduces to S = J_x^T W J_x, allowing normal cues to exert full
    unattenuated torque on the skeletal kinematic chain (achieving classical mesh performance).
  Stage 2 (Nuisance-Controlled Detail Release, t >= T_switch):
    Gaussian offsets are released under strict physical bounding (||delta_i|| <= delta_max),
    spectral Laplacian smoothing, and Schur complement sensitivity attenuation (gamma_k).
    High-frequency clothing wrinkles are absorbed without corrupting joint angles.
"""

from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.gaussian.splat_surface import TangentialGaussianSurface
from src.optimization.nuisance_sensitivity import NuisanceSensitivityEstimator, SensitivityOutput


class StageState(NamedTuple):
    iteration: int
    stage_name: str           # "kinematic_rigid" or "detail_release"
    is_kinematic_locked: bool # True if delta is locked to 0
    deform_scale: float       # Multiplier in [0.0, 1.0] for deformation updates
    max_offset_norm: float    # Clamping radius delta_max (m)
    active_splats_ratio: float # Fraction of splats at boundary


class CapacityGatedScheduler:
    """
    Orchestrates capacity-gated scheduling between skeletal kinematics and local splat deformations.
    """
    def __init__(
        self,
        total_iterations: int = 15,
        switch_iteration: int = 8,
        max_offset_norm: float = 0.015,       # 1.5 cm physical garment displacement limit
        deform_warmup_steps: int = 2,        # Smooth ramp-up of deformation learning rate
        decay_kinematics_stage2: bool = True, # Slightly decay kinematic lr in stage 2 for stability
        stage2_kinematic_lr_scale: float = 0.5
    ):
        """
        Args:
            total_iterations: Total optimization iterations per frame (default: 15).
            switch_iteration: Step at which deformation degrees of freedom are released (default: 8).
            max_offset_norm: Maximum L2 displacement radius ||delta_i||_2 in meters (default: 0.015 = 1.5 cm).
            deform_warmup_steps: Number of steps to linearly ramp deformation learning rate (default: 2).
            decay_kinematics_stage2: Whether to reduce kinematic step size once details are released.
            stage2_kinematic_lr_scale: Multiplier for kinematic lr during Stage 2.
        """
        assert switch_iteration < total_iterations, (
            f"switch_iteration ({switch_iteration}) must be strictly less than total_iterations ({total_iterations})"
        )
        self.total_iterations = total_iterations
        self.switch_iteration = switch_iteration
        self.max_offset_norm = max_offset_norm
        self.deform_warmup_steps = max(1, deform_warmup_steps)
        self.decay_kinematics_stage2 = decay_kinematics_stage2
        self.stage2_kinematic_lr_scale = stage2_kinematic_lr_scale

        # Base learning rates recorded upon optimizer attachment
        self._base_kin_lrs: Dict[int, float] = {}
        self._base_deform_lrs: Dict[int, float] = {}
        self._optimizers_attached: bool = False

    def attach_optimizers(
        self,
        opt_kin: optim.Optimizer,
        opt_deform: optim.Optimizer
    ) -> None:
        """
        Records initial learning rates for kinematic and deformation optimizer param groups.
        """
        self._base_kin_lrs = {
            i: pg.get("lr", 1e-2) for i, pg in enumerate(opt_kin.param_groups)
        }
        self._base_deform_lrs = {
            i: pg.get("lr", 5e-3) for i, pg in enumerate(opt_deform.param_groups)
        }
        self._optimizers_attached = True

    def get_stage_state(self, iteration: int) -> StageState:
        """
        Computes stage state and deformation learning rate multiplier for step `iteration` (0-indexed).
        """
        if iteration < self.switch_iteration:
            return StageState(
                iteration=iteration,
                stage_name="kinematic_rigid",
                is_kinematic_locked=True,
                deform_scale=0.0,
                max_offset_norm=self.max_offset_norm,
                active_splats_ratio=0.0
            )
        else:
            steps_into_stage2 = iteration - self.switch_iteration
            if steps_into_stage2 < self.deform_warmup_steps:
                deform_scale = float(steps_into_stage2 + 1) / float(self.deform_warmup_steps)
            else:
                deform_scale = 1.0

            return StageState(
                iteration=iteration,
                stage_name="detail_release",
                is_kinematic_locked=False,
                deform_scale=deform_scale,
                max_offset_norm=self.max_offset_norm,
                active_splats_ratio=0.0  # updated dynamically in post_step
            )

    def pre_step(
        self,
        iteration: int,
        opt_kin: optim.Optimizer,
        opt_deform: optim.Optimizer,
        gaussians: TangentialGaussianSurface
    ) -> StageState:
        """
        Configures optimizer learning rates and splat gradients prior to the forward/backward step.
        """
        if not self._optimizers_attached:
            self.attach_optimizers(opt_kin, opt_deform)

        state = self.get_stage_state(iteration)

        # 1. Kinematic learning rates
        kin_scale = (
            self.stage2_kinematic_lr_scale
            if (not state.is_kinematic_locked and self.decay_kinematics_stage2)
            else 1.0
        )
        for i, pg in enumerate(opt_kin.param_groups):
            pg["lr"] = self._base_kin_lrs[i] * kin_scale

        # 2. Deformation learning rates
        for i, pg in enumerate(opt_deform.param_groups):
            pg["lr"] = self._base_deform_lrs[i] * state.deform_scale

        # 3. If locked, enforce zero offsets
        if state.is_kinematic_locked:
            with torch.no_grad():
                gaussians.offsets.zero_()

        return state

    def post_step(
        self,
        iteration: int,
        gaussians: TangentialGaussianSurface
    ) -> StageState:
        """
        Projects splat offsets onto the physical bounding ball ||delta_i|| <= delta_max
        and returns updated diagnostic telemetry.
        """
        state = self.get_stage_state(iteration)

        if state.is_kinematic_locked:
            with torch.no_grad():
                gaussians.offsets.zero_()
            return state

        # Project splat offsets: ||delta_i|| <= max_offset_norm
        with torch.no_grad():
            offset_norms = torch.norm(gaussians.offsets, dim=-1, keepdim=True)  # (N, 1)
            exceeding = offset_norms > self.max_offset_norm
            num_exceeding = exceeding.sum().item()
            active_ratio = float(num_exceeding) / float(gaussians.num_splats)

            clamping_factor = torch.clamp(self.max_offset_norm / (offset_norms + 1e-8), max=1.0)
            gaussians.offsets.mul_(clamping_factor)

        return StageState(
            iteration=iteration,
            stage_name=state.stage_name,
            is_kinematic_locked=False,
            deform_scale=state.deform_scale,
            max_offset_norm=self.max_offset_norm,
            active_splats_ratio=active_ratio
        )
