"""
Unit and Regression Tests for Nuisance Sensitivity Estimator & Spectral Step Damping.
Tests the mathematical invariants of Schur complement curvature attenuation,
kinematic-nuisance collinearity detection, and subspace orthogonalization.
"""

import pytest
import torch
import torch.nn as nn

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.geometry.mesh_graph import MeshGraph
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.optimization.nuisance_sensitivity import NuisanceSensitivityEstimator, SensitivityOutput
from src.optimization.dual_frequency_router import DualFrequencyGradientRouter


@pytest.fixture
def sensitivity_setup():
    device = torch.device("cpu")
    smpl = SMPLWrapper(device=device).to(device)
    segmenter = KinematicSegmenter(smpl.weights)
    mesh_graph = MeshGraph(smpl.faces).to(device)
    estimator = NuisanceSensitivityEstimator(
        smpl=smpl,
        segmenter=segmenter,
        mesh_graph=mesh_graph,
        lambda_lap=50.0,
        lambda_tight=100.0,
        alpha=0.05,
        beta=1.0,
        gamma_min=0.05,
        lambda_prior=1.0,
    ).to(device)
    return {
        "device": device,
        "smpl": smpl,
        "segmenter": segmenter,
        "mesh_graph": mesh_graph,
        "estimator": estimator
    }


def test_sensitivity_output_invariants(sensitivity_setup):
    """Verifies bounds and dimensions of the sensitivity estimator output."""
    setup = sensitivity_setup
    device = setup["device"]
    estimator = setup["estimator"]
    smpl = setup["smpl"]

    theta = torch.zeros(24, 3, device=device)
    trans = torch.zeros(3, device=device)
    init_theta = torch.zeros(24, 3, device=device)
    init_trans = torch.zeros(3, device=device)

    out = smpl(theta=theta.unsqueeze(0), trans=trans.unsqueeze(0))
    verts = out["vertices"][0]
    joints = out["joints"][0]

    grad_theta = torch.randn(24, 3, device=device) * 0.1
    grad_trans = torch.randn(3, device=device) * 0.1
    grad_offsets = torch.randn(6890, 3, device=device) * 0.01

    res: SensitivityOutput = estimator(
        theta=theta,
        trans=trans,
        init_theta=init_theta,
        init_trans=init_trans,
        verts=verts,
        joints=joints,
        grad_theta=grad_theta,
        grad_trans=grad_trans,
        grad_offsets=grad_offsets
    )

    assert isinstance(res, SensitivityOutput)
    assert res.gamma_theta.shape == (24,)
    assert (res.gamma_theta >= 0.05).all() and (res.gamma_theta <= 1.0).all()

    assert res.rho_theta.shape == (24,)
    assert (res.rho_theta >= 0.0).all() and (res.rho_theta <= 1.0).all()

    assert res.filtered_grad_theta.shape == (24, 3)
    assert res.filtered_grad_trans.shape == (3,)
    assert res.filtered_grad_offsets.shape == (6890, 3)


def test_collinear_interference_penalization(sensitivity_setup):
    """Verifies that high collinearity between splat offsets and joint motion depresses gamma."""
    setup = sensitivity_setup
    device = setup["device"]
    estimator = setup["estimator"]
    smpl = setup["smpl"]

    theta = torch.zeros(24, 3, device=device)
    trans = torch.zeros(3, device=device)
    out = smpl(theta=theta.unsqueeze(0), trans=trans.unsqueeze(0))
    verts = out["vertices"][0]
    joints = out["joints"][0]

    # Test joint 4 (left knee)
    grad_theta = torch.zeros(24, 3, device=device)
    grad_theta[4, 0] = 1.0  # Rotation around x-axis

    # Construct offset gradient that is artificially identical to joint 4's kinematic vector field
    diff = verts - joints[4]
    omega = torch.tensor([1.0, 0.0, 0.0], device=device)
    cross = torch.linalg.cross(omega.expand_as(diff), diff, dim=-1)
    collinear_offsets_grad = smpl.weights[:, 4].unsqueeze(1) * cross

    res_collinear: SensitivityOutput = estimator(
        theta=theta,
        trans=trans,
        init_theta=theta,
        init_trans=trans,
        verts=verts,
        joints=joints,
        grad_theta=grad_theta,
        grad_trans=torch.zeros(3, device=device),
        grad_offsets=collinear_offsets_grad
    )

    # Now construct orthogonal offset gradient
    ortho_offsets_grad = torch.randn(6890, 3, device=device) * 0.001

    res_ortho: SensitivityOutput = estimator(
        theta=theta,
        trans=trans,
        init_theta=theta,
        init_trans=trans,
        verts=verts,
        joints=joints,
        grad_theta=grad_theta,
        grad_trans=torch.zeros(3, device=device),
        grad_offsets=ortho_offsets_grad
    )

    # Collinearity must be higher for the collinear gradient
    assert res_collinear.rho_theta[4] > res_ortho.rho_theta[4]
    # Mode retention gamma must be depressed by collinear interference
    assert res_collinear.gamma_theta[4] < res_ortho.gamma_theta[4]


def test_dual_frequency_router_condition_e_integration(sensitivity_setup):
    """Verifies that DualFrequencyGradientRouter executes Condition E with sensitivity controller."""
    setup = sensitivity_setup
    device = setup["device"]
    smpl = setup["smpl"]
    segmenter = setup["segmenter"]
    mesh_graph = setup["mesh_graph"]
    estimator = setup["estimator"]

    gaussians = TangentialGaussianSurface(num_splats=6890, device=device)
    collision = GaussianSelfCollisionEngine(segmenter)
    rasterizer = DifferentiableNormalRasterizer(image_height=64, image_width=64, max_splats_per_tile=64)

    router = DualFrequencyGradientRouter(
        smpl=smpl,
        gaussian_surface=gaussians,
        collision_engine=collision,
        mesh_graph=mesh_graph,
        rasterizer=rasterizer,
        sensitivity_estimator=estimator
    )

    theta = nn.Parameter(torch.zeros(24, 3, device=device))
    trans = nn.Parameter(torch.zeros(3, device=device))
    init_theta = theta.clone().detach()
    init_trans = trans.clone().detach()

    opt_kin, opt_deform = router.create_optimizers(theta, trans)

    K = torch.eye(3, device=device)
    K[0, 0] = 50.0
    K[1, 1] = 50.0
    K[0, 2] = 32.0
    K[1, 2] = 32.0

    target_normals = torch.zeros(64, 64, 3, device=device)
    target_normals[:, :, 2] = 1.0
    target_rgb = torch.ones(64, 64, 3, device=device) * 0.5
    target_mask = torch.ones(64, 64, device=device)

    step_out = router.step(
        theta=theta,
        trans=trans,
        K=K,
        target_normals=target_normals,
        target_rgb=target_rgb,
        target_mask=target_mask,
        uncertainty=None,
        opt_kin=opt_kin,
        opt_deform=opt_deform,
        init_theta=init_theta,
        init_trans=init_trans
    )

    assert step_out.loss_total > 0
    assert step_out.sensitivity is not None
    assert isinstance(step_out.sensitivity, SensitivityOutput)
    assert step_out.sensitivity.gamma_theta.shape == (24,)
