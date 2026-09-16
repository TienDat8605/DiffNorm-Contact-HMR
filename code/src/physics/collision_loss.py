"""
Kinematic Self-Collision Loss Evaluator across Non-Adjacent SMPL Segments.
Evaluates volumetric self-penetration penalty and exerts smooth repulsive forces.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from src.geometry.kinematic_segments import KinematicSegmenter
from src.physics.gaussian_convolution import compute_gaussian_overlap_matrix


class GaussianSelfCollisionEngine(nn.Module):
    """
    Computes volumetric self-collision loss over non-adjacent anatomical segments
    using closed-form analytical Gaussian overlap integrals.
    """
    def __init__(
        self,
        segmenter: KinematicSegmenter,
        distance_cutoff: float = 0.15  # Interaction cutoff distance in meters
    ):
        super().__init__()
        self.segmenter = segmenter
        self.distance_cutoff = distance_cutoff
        self.non_adjacent_pairs = segmenter.non_adjacent_pairs

    def forward(
        self,
        centers: torch.Tensor,       # (N, 3) or (B, N, 3)
        covariances: torch.Tensor    # (N, 3, 3) or (B, N, 3, 3)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total self-collision penalty.
        Returns:
            loss: scalar tensor with smooth gradients
            diagnostics: dict with penetration metrics
        """
        if centers.dim() == 3:
            # Batch mode: iterate over batch items
            B = centers.shape[0]
            losses = []
            diag_total = {"total_overlap": 0.0, "active_colliding_pairs": 0}
            for b in range(B):
                loss_b, diag_b = self._compute_single(centers[b], covariances[b])
                losses.append(loss_b)
                diag_total["total_overlap"] += diag_b["total_overlap"] / B
                diag_total["active_colliding_pairs"] += diag_b["active_colliding_pairs"]
            return torch.stack(losses).mean(), diag_total
        else:
            return self._compute_single(centers, covariances)

    def _compute_single(
        self,
        centers: torch.Tensor,       # (N, 3)
        covariances: torch.Tensor    # (N, 3, 3)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_collision_loss = torch.sum(centers * 0.0)
        active_pairs_count = 0

        # Precompute segment bounding spheres for coarse culling
        segment_centroids = {}
        segment_radii = {}
        for seg_id in range(14):
            idx = self.segmenter.get_segment_indices(seg_id)
            if len(idx) > 0:
                seg_pts = centers[idx]
                centroid = seg_pts.mean(dim=0)
                radius = torch.norm(seg_pts - centroid, dim=1).max()
                segment_centroids[seg_id] = centroid
                segment_radii[seg_id] = radius

        # Iterate over non-adjacent segment pairs
        for seg_a, seg_b in self.non_adjacent_pairs:
            idx_a = self.segmenter.get_segment_indices(seg_a)
            idx_b = self.segmenter.get_segment_indices(seg_b)

            if len(idx_a) == 0 or len(idx_b) == 0:
                continue

            # Coarse bounding sphere rejection test
            c_a, r_a = segment_centroids[seg_a], segment_radii[seg_a]
            c_b, r_b = segment_centroids[seg_b], segment_radii[seg_b]
            centroid_dist = torch.norm(c_a - c_b)

            # If bounding spheres are separated by more than distance_cutoff, skip pair
            if centroid_dist > (r_a + r_b + self.distance_cutoff):
                continue

            # Compute exact analytical overlap matrix K_ij
            pts_a, cov_a = centers[idx_a], covariances[idx_a]
            pts_b, cov_b = centers[idx_b], covariances[idx_b]

            K_ab = compute_gaussian_overlap_matrix(
                pts_a, cov_a, pts_b, cov_b, distance_cutoff=self.distance_cutoff
            )

            overlap_sum = torch.sum(K_ab)
            if overlap_sum > 1e-6:
                total_collision_loss = total_collision_loss + overlap_sum
                active_pairs_count += 1

        diagnostics = {
            "total_overlap": total_collision_loss.item(),
            "active_colliding_pairs": active_pairs_count,
        }
        return total_collision_loss, diagnostics
