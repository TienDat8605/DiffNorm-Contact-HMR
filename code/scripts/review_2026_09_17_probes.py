"""Read-only CPU counterexamples for the September 17 proposal/code review.

Run from repository root: PYTHONPATH=code python code/scripts/review_2026_09_17_probes.py
These diagnose existing code; they are not benchmark evaluations.
"""
import json
import math
import torch
from src.geometry.smpl_wrapper import SMPLWrapper, rodrigues
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.gaussian_convolution import compute_gaussian_overlap_matrix
from src.pipeline.eval_metrics import compute_pa_mpjpe
from src.pipeline.coarse_pose_hmr2 import CoarsePoseHMR2
from src.rendering.normal_rasterizer import DifferentiableNormalRasterizer

torch.set_num_threads(1)
torch.manual_seed(17)
out = {}
smpl = SMPLWrapper()
neutral = smpl(torch.zeros(1, 24, 3))
out['neutral_vertex_mean_displacement_m'] = (neutral['vertices'][0] - smpl.v_template).norm(dim=-1).mean().item()
_, skinning = smpl.forward_kinematics(torch.eye(3).repeat(1, 24, 1, 1), smpl.joints_template[None])
out['neutral_skinning_translation_max_m'] = skinning[0, :, :3, 3].abs().max().item()

g = TangentialGaussianSurface(num_splats=1)
normal = torch.tensor([[1., 0., 0.]])
cov = g.get_spatial_covariances()
disk_axis = torch.linalg.eigh(cov).eigenvectors[:, :, 0]
out['normal_vs_covariance_axis_abs_dot'] = (g.get_surface_normals(normal) * disk_axis).sum().abs().item()
mu = torch.zeros(1, 3)
actual = compute_gaussian_overlap_matrix(mu, cov, mu, cov).item()
expected = math.pi ** 1.5 * g.get_scales().prod().item()
out['default_overlap'] = {'implemented': actual, 'closed_form': expected, 'ratio': actual / expected, 'determinant': torch.linalg.det(cov).item()}

points = torch.randn(12, 3)
R = rodrigues(torch.tensor([[0., 0., 1.1]]))[0]
pred = torch.stack([points, points @ R.T + torch.tensor([2., -1., 3.])])
gt = torch.stack([points, points])
out['pa_alignment_mm'] = {'batched': compute_pa_mpjpe(pred, gt), 'mean_individual': sum(compute_pa_mpjpe(p, t) for p, t in zip(pred, gt)) / 2}

coarse = CoarsePoseHMR2(checkpoint_path='/tmp/nonexistent_hmr_review_checkpoint.pt', device=torch.device('cpu'), auto_download=False)
a = coarse(torch.zeros(1, 3, 32, 32))
b = coarse(torch.ones(1, 3, 32, 32))
out['initializer'] = {'model_is_none': coarse.model is None, 'image_pose_difference': (a['theta'] - b['theta']).abs().max().item(), 'source': a['source']}

renderer = DifferentiableNormalRasterizer(image_height=32, image_width=32)
center = torch.tensor([[0., 0., 2.]])
K = torch.tensor([[100., 0., 16.], [0., 100., 16.], [0., 0., 1.]])
rend = renderer(center, cov.detach(), torch.tensor([[0., 0., 1.]]), torch.ones(1, 3), torch.ones(1, 1)*.95, K)
out['single_splat_far_corner'] = {'opacity': rend.mask[0, 0].item(), 'normal_length': rend.normals[0, 0].norm().item()}

# Forty nearer splats on the left can suppress an otherwise visible splat on the right.
centers = torch.tensor([[-.24, 0., 2.]] * 40 + [[.252, 0., 2.1]])
covs = torch.eye(3).repeat(41, 1, 1) * .0001
normals = torch.tensor([[0., 0., 1.]]).repeat(41, 1)
colors = torch.ones(41, 3)
opacity = torch.ones(41, 1) * .95
limited = renderer(centers, covs, normals, colors, opacity, K)
uncapped = DifferentiableNormalRasterizer(image_height=32, image_width=32, max_splats_per_tile=100)(centers, covs, normals, colors, opacity, K)
out['tile_cap_right_pixel_opacity'] = {'cap40': limited.mask[16, 28].item(), 'cap100': uncapped.mask[16, 28].item()}
print(json.dumps(out, indent=2))
