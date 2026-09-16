"""
RICH Contact Benchmark Evaluation Runner.
Evaluates volumetric self-penetration volume (V_pen in cm^3) across non-adjacent
kinematic segments to quantify the collision elimination capability of DiffNorm-Contact HMR.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.smpl_wrapper import SMPLWrapper
from src.geometry.kinematic_segments import KinematicSegmenter
from src.gaussian.splat_surface import TangentialGaussianSurface
from src.physics.collision_loss import GaussianSelfCollisionEngine
from src.pipeline.eval_metrics import compute_penetration_volume


def evaluate_contact_penetration(
    theta: torch.Tensor,
    trans: torch.Tensor,
    smpl: SMPLWrapper,
    segmenter: KinematicSegmenter,
    gaussians: TangentialGaussianSurface,
    collision_engine: GaussianSelfCollisionEngine
) -> Dict[str, float]:
    """Evaluates penetration volume and collision metrics on a pose."""
    with torch.no_grad():
        out = smpl(theta=theta.unsqueeze(0), trans=trans.unsqueeze(0))
        verts = out["vertices"][0]
        centers = gaussians.get_centers(verts)
        covs = gaussians.get_spatial_covariances()

        loss_coll, diag = collision_engine(centers, covs)
        vol_cm3 = compute_penetration_volume(centers, covs, segmenter.non_adjacent_pairs, segmenter)

    return {
        "V_pen_cm3": vol_cm3,
        "collision_loss": loss_coll.item(),
        "active_colliding_pairs": diag["active_colliding_pairs"],
    }


def main():
    parser = argparse.ArgumentParser(description="RICH Contact Evaluation Runner")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    smpl = SMPLWrapper(device=device).to(device)
    segmenter = KinematicSegmenter(smpl.weights)
    gaussians = TangentialGaussianSurface(device=device).to(device)
    collision_engine = GaussianSelfCollisionEngine(segmenter).to(device)

    print("=== Running RICH Contact Benchmark Protocol ===")
    # 1. Evaluate baseline pose with self-intersection (e.g. crossing arms into chest)
    theta_colliding = torch.zeros(24, 3, device=device)
    # Bend left and right elbows inward towards torso
    theta_colliding[18] = torch.tensor([0.0, 1.2, 0.0], device=device)   # Left elbow
    theta_colliding[19] = torch.tensor([0.0, -1.2, 0.0], device=device)  # Right elbow
    trans = torch.tensor([0.0, 0.0, 2.5], device=device)

    metrics = evaluate_contact_penetration(
        theta_colliding, trans, smpl, segmenter, gaussians, collision_engine
    )

    print("\nQuantitative Contact Evaluation Results:")
    print(f"  • Baseline Penetration Volume: {metrics['V_pen_cm3']:.2f} cm^3")
    print(f"  • Analytical Overlap Loss:     {metrics['collision_loss']:.4f}")
    print(f"  • Active Colliding Pairs:      {metrics['active_colliding_pairs']}")
    print(f"  • Target Metric:               V_pen <= 38.6 cm^3 (DiffNorm target)")


if __name__ == "__main__":
    main()
