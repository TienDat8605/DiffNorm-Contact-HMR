"""
Unit tests for CapacityGatedScheduler.
Validates two-stage capacity scheduling, learning rate scaling, and Euclidean projection bounds.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.gaussian.splat_surface import TangentialGaussianSurface
from src.optimization.capacity_scheduler import CapacityGatedScheduler, StageState


def test_scheduler_initialization_and_validation():
    # Valid setup
    sched = CapacityGatedScheduler(total_iterations=15, switch_iteration=8, max_offset_norm=0.015)
    assert sched.total_iterations == 15
    assert sched.switch_iteration == 8
    assert sched.max_offset_norm == 0.015

    # Invalid setup: switch >= total
    with pytest.raises(AssertionError):
        CapacityGatedScheduler(total_iterations=10, switch_iteration=10)


def test_stage_state_transitions():
    sched = CapacityGatedScheduler(
        total_iterations=15,
        switch_iteration=8,
        max_offset_norm=0.015,
        deform_warmup_steps=2
    )

    # Stage 1: iterations 0..7
    for t in range(8):
        state = sched.get_stage_state(t)
        assert state.stage_name == "kinematic_rigid"
        assert state.is_kinematic_locked is True
        assert state.deform_scale == 0.0

    # Stage 2: iteration 8 (step 1 of warmup: scale = 0.5)
    s8 = sched.get_stage_state(8)
    assert s8.stage_name == "detail_release"
    assert s8.is_kinematic_locked is False
    assert abs(s8.deform_scale - 0.5) < 1e-5

    # Stage 2: iteration 9 (step 2 of warmup: scale = 1.0)
    s9 = sched.get_stage_state(9)
    assert s9.stage_name == "detail_release"
    assert s9.is_kinematic_locked is False
    assert abs(s9.deform_scale - 1.0) < 1e-5

    # Stage 2: iteration 10..14 (full scale = 1.0)
    for t in range(10, 15):
        st = sched.get_stage_state(t)
        assert st.stage_name == "detail_release"
        assert abs(st.deform_scale - 1.0) < 1e-5


def test_pre_and_post_step_execution():
    device = torch.device("cpu")
    gaussians = TangentialGaussianSurface(num_splats=100, device=device)

    theta = nn.Parameter(torch.randn(24, 3, device=device))
    trans = nn.Parameter(torch.randn(3, device=device))

    opt_kin = optim.AdamW([{"params": [theta, trans], "lr": 0.01}])
    opt_deform = optim.AdamW([{"params": [gaussians.offsets], "lr": 0.005}])

    sched = CapacityGatedScheduler(
        total_iterations=10,
        switch_iteration=5,
        max_offset_norm=0.015,
        deform_warmup_steps=2,
        decay_kinematics_stage2=True,
        stage2_kinematic_lr_scale=0.5
    )

    # In Stage 1 (t=2), simulate non-zero offsets
    with torch.no_grad():
        gaussians.offsets.add_(torch.randn_like(gaussians.offsets))

    state_pre = sched.pre_step(2, opt_kin, opt_deform, gaussians)
    assert state_pre.is_kinematic_locked is True
    assert opt_deform.param_groups[0]["lr"] == 0.0
    assert opt_kin.param_groups[0]["lr"] == 0.01
    assert torch.all(gaussians.offsets == 0.0)

    state_post = sched.post_step(2, gaussians)
    assert state_post.is_kinematic_locked is True
    assert torch.all(gaussians.offsets == 0.0)

    # In Stage 2 (t=6), check bounding projection
    sched.pre_step(6, opt_kin, opt_deform, gaussians)
    assert opt_kin.param_groups[0]["lr"] == 0.005  # decayed by 0.5
    assert opt_deform.param_groups[0]["lr"] == 0.005  # warmup complete

    # Inject large offsets exceeding 1.5 cm (0.015 m)
    with torch.no_grad():
        gaussians.offsets.fill_(0.10)  # 10 cm, far exceeding 1.5 cm

    state_post_6 = sched.post_step(6, gaussians)
    assert state_post_6.is_kinematic_locked is False
    assert state_post_6.active_splats_ratio == 1.0

    # Max norm must be <= 0.015 + epsilon
    norms = torch.norm(gaussians.offsets, dim=-1)
    assert torch.all(norms <= 0.015001)
    assert torch.allclose(norms, torch.tensor(0.015), atol=1e-4)
