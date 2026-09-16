"""
Tests for FeatureCacheDataset HDF5 reader and writer.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
from src.pipeline.feature_cache import FeatureCacheDataset


def test_hdf5_feature_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.h5")
        N, H, W = 4, 32, 32
        rgb = np.random.rand(N, H, W, 3).astype(np.float32)
        normals = np.random.rand(N, H, W, 3).astype(np.float32)
        mask = (np.random.rand(N, H, W) > 0.5).astype(np.float32)
        K = np.repeat(np.eye(3)[np.newaxis, ...], N, axis=0).astype(np.float32)

        # 1. Create cache
        FeatureCacheDataset.create_cache(cache_path, rgb, normals, mask, K)
        assert os.path.exists(cache_path)

        # 2. Read cache
        dataset = FeatureCacheDataset(cache_path)
        assert len(dataset) == N

        sample = dataset[0]
        assert sample["rgb"].shape == (H, W, 3)
        assert sample["normals"].shape == (H, W, 3)
        assert sample["mask"].shape == (H, W)
        assert sample["K"].shape == (3, 3)
        assert torch.allclose(sample["rgb"], torch.from_numpy(rgb[0]))
