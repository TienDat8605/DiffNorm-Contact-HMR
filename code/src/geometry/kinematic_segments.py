"""
Kinematic Segmentation and Non-Adjacent Segment Pair Computation for SMPL.
Partitions the 6,890 vertices into 14 anatomically distinct body segments and
precomputes the non-adjacent collision candidate pairs.
"""

from typing import Dict, List, Tuple
import torch


# 14 Kinematic Segments mapped to joint bone indices
SEGMENT_NAMES = [
    "Torso",            # 0
    "Head",             # 1
    "Left_Upper_Arm",   # 2
    "Left_Forearm",     # 3
    "Left_Hand",        # 4
    "Right_Upper_Arm",  # 5
    "Right_Forearm",    # 6
    "Right_Hand",       # 7
    "Left_Thigh",       # 8
    "Left_Calf",        # 9
    "Left_Foot",        # 10
    "Right_Thigh",      # 11
    "Right_Calf",       # 12
    "Right_Foot",       # 13
]

SEGMENT_TO_BONES: Dict[int, List[int]] = {
    0: [0, 3, 6, 9],    # Pelvis, Spine 1, Spine 2, Spine 3
    1: [12, 15],        # Neck, Head
    2: [13, 16],        # Left Collar, Left Shoulder
    3: [18],            # Left Elbow / Forearm
    4: [20, 22],        # Left Wrist, Left Hand
    5: [14, 17],        # Right Collar, Right Shoulder
    6: [19],            # Right Elbow / Forearm
    7: [21, 23],        # Right Wrist, Right Hand
    8: [1],             # Left Hip / Thigh
    9: [4],             # Left Knee / Calf
    10: [7, 10],        # Left Ankle, Left Foot
    11: [2],            # Right Hip / Thigh
    12: [5],            # Right Knee / Calf
    13: [8, 11],        # Right Ankle, Right Foot
}

# Adjacent pairs that physically connect and flex naturally (excluded from collision penalty)
ADJACENT_PAIRS: List[Tuple[int, int]] = [
    (0, 1),   # Torso - Head
    (0, 2),   # Torso - L Upper Arm
    (0, 5),   # Torso - R Upper Arm
    (0, 8),   # Torso - L Thigh
    (0, 11),  # Torso - R Thigh
    (2, 3),   # L Upper Arm - L Forearm
    (3, 4),   # L Forearm - L Hand
    (5, 6),   # R Upper Arm - R Forearm
    (6, 7),   # R Forearm - R Hand
    (8, 9),   # L Thigh - L Calf
    (9, 10),  # L Calf - L Foot
    (11, 12), # R Thigh - R Calf
    (12, 13), # R Calf - R Foot
]


class KinematicSegmenter:
    """
    Partitions vertices of an SMPL mesh into 14 anatomical body parts and
    determines non-adjacent collision candidate pairs.
    """
    def __init__(self, weights: torch.Tensor):
        """
        weights: (N, 24) SMPL blend skinning weights
        """
        self.num_vertices = weights.shape[0]
        self.device = weights.device
        
        # Aggregate skinning weights per segment
        # (N, 14)
        segment_weights = torch.zeros(self.num_vertices, 14, device=self.device)
        for seg_id, bones in SEGMENT_TO_BONES.items():
            segment_weights[:, seg_id] = weights[:, bones].sum(dim=1)
            
        # Assign each vertex to the segment with maximum influence
        self.vertex_to_segment = torch.argmax(segment_weights, dim=1)  # (N,)
        
        # Build index list per segment
        self.segment_indices: Dict[int, torch.Tensor] = {}
        for seg_id in range(14):
            idx = torch.where(self.vertex_to_segment == seg_id)[0]
            self.segment_indices[seg_id] = idx
            
        # Compute list of all non-adjacent pairs (A, B) with A < B
        adjacent_set = set(tuple(sorted(p)) for p in ADJACENT_PAIRS)
        self.non_adjacent_pairs: List[Tuple[int, int]] = []
        for i in range(14):
            for j in range(i + 1, 14):
                if (i, j) not in adjacent_set:
                    self.non_adjacent_pairs.append((i, j))

    def get_segment_mask(self, seg_id: int) -> torch.Tensor:
        """Returns boolean mask of vertices belonging to segment seg_id."""
        return self.vertex_to_segment == seg_id

    def get_segment_indices(self, seg_id: int) -> torch.Tensor:
        """Returns vertex indices belonging to segment seg_id."""
        return self.segment_indices[seg_id]
