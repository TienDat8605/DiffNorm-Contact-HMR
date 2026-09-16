"""
Unit tests for DSINE normal pre-caching and 4D-Humans coarse pose estimation.
"""

import os
import sys
import tempfile
import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.coarse_pose_hmr2 import CoarsePoseHMR2
from scripts.extract_dsine_normals import load_dsine_model, run_dsine_batch, precompute_dsine_normals_in_cache


def test_coarse_pose_hmr2():
    device = torch.device("cpu")
    hmr2 = CoarsePoseHMR2(device=device, auto_download=False)

    B = 2
    dummy_imgs = torch.rand(B, 3, 256, 256, device=device)
    out = hmr2(dummy_imgs)

    assert "theta" in out
    assert "beta" in out
    assert "trans" in out
    assert "joints" in out

    assert out["theta"].shape == (B, 24, 3)
    assert out["beta"].shape == (B, 10)
    assert out["trans"].shape == (B, 3)
    assert out["joints"].shape == (B, 24, 3)
    assert out["is_coarse"] is True


def test_dsine_loading_and_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dsine_model(device=device)
    if model is None:
        pytest.skip("DSINE model not available in current environment")

    # Run forward pass on 1 dummy image
    dummy_img = torch.rand(1, 3, 256, 256, device=device)
    normals = run_dsine_batch(model, dummy_img, device=device)

    assert normals.shape == (1, 256, 256, 3)
    # Unit vector magnitude check
    lengths = torch.norm(normals, dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4)


def test_dsine_cache_injection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dsine_model(device=device)
    if model is None:
        pytest.skip("DSINE model not available in current environment")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_h5 = tmp.name

    try:
        # Create minimal mock cache with 2 RGB images
        with h5py.File(tmp_h5, "w") as f:
            f.create_dataset("rgb", data=np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8))

        # Run precompute
        count = precompute_dsine_normals_in_cache(tmp_h5, batch_size=2, max_samples=2, device=device)
        assert count == 2

        with h5py.File(tmp_h5, "r") as f:
            assert "normals" in f
            assert "mask" in f
            assert f["normals"].shape == (2, 256, 256, 3)
            assert f["mask"].shape == (2, 256, 256)

            norms = f["normals"][:]
            lengths = np.linalg.norm(norms, axis=-1)
            assert np.allclose(lengths, np.ones_like(lengths), atol=1e-4)
    finally:
        if os.path.exists(tmp_h5):
            os.remove(tmp_h5)
