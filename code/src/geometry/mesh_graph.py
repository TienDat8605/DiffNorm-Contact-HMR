"""
Canonical Mesh Graph and Laplacian Smoothness Operator for SMPL.
Builds the edge connectivity from mesh faces and computes the graph Laplacian
loss for elastic cloth and Gaussian displacement regularization.
"""

from typing import Tuple
import torch
import torch.nn as nn


class MeshGraph(nn.Module):
    """
    Constructs the canonical SMPL edge graph and computes the graph Laplacian
    smoothness loss on displacement offsets delta.
    """
    def __init__(self, faces: torch.Tensor, num_vertices: int = 6890):
        super().__init__()
        self.num_vertices = num_vertices
        self.device = faces.device
        
        # Extract unique undirected edges from triangular faces
        # faces: (M, 3)
        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ], dim=0)  # (3M, 2)
        
        # Sort each edge so that i < j
        edges_sorted, _ = torch.sort(edges, dim=1)
        # Remove duplicate edges
        edges_unique = torch.unique(edges_sorted, dim=0)  # (E, 2)
        
        self.register_buffer("edges", edges_unique)
        self.register_buffer("edge_src", edges_unique[:, 0])
        self.register_buffer("edge_dst", edges_unique[:, 1])

    def laplacian_loss(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dirichlet energy / Graph Laplacian smoothness penalty:
        L_lap = (1 / |E|) * sum_{(i,j) in E} ||delta_i - delta_j||_2^2
        offsets: (B, N, 3) or (N, 3)
        """
        if offsets.dim() == 2:
            offsets = offsets.unsqueeze(0)
            
        src_offsets = offsets[:, self.edge_src]  # (B, E, 3)
        dst_offsets = offsets[:, self.edge_dst]  # (B, E, 3)
        
        diff_sq = torch.sum((src_offsets - dst_offsets) ** 2, dim=-1)  # (B, E)
        return torch.mean(diff_sq)

    def elastic_tether_loss(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        L_tight = (1 / N) * sum_i ||delta_i||_2^2
        offsets: (B, N, 3) or (N, 3)
        """
        return torch.mean(torch.sum(offsets ** 2, dim=-1))
