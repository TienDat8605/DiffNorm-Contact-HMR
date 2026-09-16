"""
Foundation Feature Cache Manager (HDF5).
Pre-extracts and stores zero-shot surface normals (DSINE), segmentation masks (SAMv2),
and camera intrinsics to accelerate Google Colab T4 training by eliminating redundant
foundation model forward passes during backpropagation.
"""

from typing import Dict, Optional, Tuple
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureCacheDataset(Dataset):
    """
    Reads precomputed foundation features from an HDF5 archive:
    - rgb: (H, W, 3) in [0, 1]
    - normals: (H, W, 3) unit surface normals in camera space
    - mask: (H, W) binary silhouette mask
    - K: (3, 3) camera intrinsics
    """
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Feature cache file not found at: {h5_path}")
            
        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f["rgb"])
        self.h5_file = None
        self.rgb_ds = None
        self.normals_ds = None
        self.mask_ds = None
        self.K_ds = None

    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", libver="latest")
            self.rgb_ds = self.h5_file["rgb"]
            self.normals_ds = self.h5_file["normals"]
            self.mask_ds = self.h5_file["mask"]
            self.K_ds = self.h5_file["K"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._open_file()
        raw_rgb = self.rgb_ds[idx]
        rgb = torch.from_numpy(raw_rgb).float()
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
            
        normals = torch.from_numpy(self.normals_ds[idx]).float()
        mask = torch.from_numpy(self.mask_ds[idx]).float()
        K = torch.from_numpy(self.K_ds[idx]).float()

        sample = {
            "rgb": rgb,
            "normals": normals,
            "mask": mask,
            "K": K,
        }
        if "theta" in self.h5_file:
            sample["theta"] = torch.from_numpy(self.h5_file["theta"][idx]).float()
        if "beta" in self.h5_file:
            sample["beta"] = torch.from_numpy(self.h5_file["beta"][idx]).float()
        if "trans" in self.h5_file:
            sample["trans"] = torch.from_numpy(self.h5_file["trans"][idx]).float()

        return sample

    @staticmethod
    def create_cache(
        out_path: str,
        rgb_list: np.ndarray,       # (N, H, W, 3)
        normals_list: np.ndarray,   # (N, H, W, 3)
        mask_list: np.ndarray,      # (N, H, W)
        K_list: np.ndarray          # (N, 3, 3)
    ):
        """Creates a compressed HDF5 feature cache archive."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("rgb", data=rgb_list, compression="gzip", chunks=True)
            f.create_dataset("normals", data=normals_list, compression="gzip", chunks=True)
            f.create_dataset("mask", data=mask_list, compression="gzip", chunks=True)
            f.create_dataset("K", data=K_list, compression="gzip", chunks=True)
        print(f"Created feature cache archive at {out_path} ({len(rgb_list)} samples)")
