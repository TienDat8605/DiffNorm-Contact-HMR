"""
Tests for 3DPW Primary Training Dataset Loader.
"""

import os
import tempfile
import pytest
import torch
from src.pipeline.dataset_3dpw import Dataset3DPW, create_mock_3dpw_sample


def test_3dpw_dataset_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock 3DPW sequence
        pkl_path = create_mock_3dpw_sample(root_dir=tmpdir, split="train")
        assert os.path.exists(pkl_path)

        # Load dataset
        dataset = Dataset3DPW(root_dir=tmpdir, split="train")
        assert len(dataset) == 10

        sample = dataset[0]
        assert "image" in sample
        assert "theta" in sample
        assert "beta" in sample
        assert "trans" in sample
        assert "K" in sample

        assert sample["image"].shape == (256, 256, 3)
        assert sample["theta"].shape == (24, 3)
        assert sample["beta"].shape == (10,)
        assert sample["trans"].shape == (3,)
        assert sample["K"].shape == (3, 3)
