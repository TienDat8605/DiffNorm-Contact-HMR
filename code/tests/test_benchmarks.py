"""
Tests for benchmark evaluation runners (3DPW, RICH, CAPE).
"""

import pytest
import torch
from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from scripts.eval_3dpw import evaluate_3dpw_sequence
from scripts.eval_rich_contact import evaluate_contact_penetration
from scripts.eval_cape_clothing import evaluate_clothing_bias


def test_3dpw_evaluator():
    device = torch.device("cpu")
    smpl = SMPLWrapper(device=device)

    theta = torch.zeros(2, 24, 3, device=device)
    beta = torch.zeros(2, 10, device=device)

    # Perfect prediction should have virtually zero error (< 0.01 mm)
    metrics = evaluate_3dpw_sequence(theta, beta, theta, beta, smpl)
    assert metrics["MPJPE_mm"] < 1e-2
    assert metrics["PA_MPJPE_mm"] < 1e-2
    assert metrics["PVE_mm"] < 1e-2


def test_rich_contact_evaluator():
    device = torch.device("cpu")
    smpl = SMPLWrapper(device=device)
    segmenter = KinematicSegmenter(smpl.weights)
    gaussians = TangentialGaussianSurface(device=device)
    collision_engine = GaussianSelfCollisionEngine(segmenter)

    theta = torch.zeros(24, 3, device=device)
    trans = torch.tensor([0.0, 0.0, 2.5], device=device)

    metrics = evaluate_contact_penetration(theta, trans, smpl, segmenter, gaussians, collision_engine)
    assert "V_pen_cm3" in metrics
    assert "collision_loss" in metrics
    assert isinstance(metrics["V_pen_cm3"], float)


def test_cape_clothing_evaluator():
    device = torch.device("cpu")
    smpl = SMPLWrapper(device=device)

    bare_theta = torch.zeros(2, 24, 3, device=device)
    clothed_coupled = bare_theta + 0.05
    clothed_detached = bare_theta + 0.01

    metrics = evaluate_clothing_bias(bare_theta, clothed_coupled, clothed_detached, smpl)
    assert metrics["Detached_MPJPE_mm"] < metrics["Coupled_MPJPE_mm"]
    assert metrics["Bias_Reduction_mm"] > 0
