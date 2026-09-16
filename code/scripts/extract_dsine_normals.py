"""
Automated DSINE Surface Normal Pre-Caching Pipeline.
Loads zero-shot pretrained DSINE foundation model (CVPR 2024 Oral),
runs high-speed batched inference over human mesh recovery frames,
and stores dense surface normals and masks directly into HDF5 cache containers.
"""

import os
import sys
import time
import argparse
import types
import subprocess
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DEFAULT_CKPT_PATH = os.path.expanduser("~/.cache/torch/hub/checkpoints/dsine.pt")
DSINE_HF_URL = "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
DSINE_REPO_DIR = os.path.expanduser("~/.cache/torch/hub/baegwangbin_DSINE_main")


def load_dsine_model(
    ckpt_path: str = DEFAULT_CKPT_PATH,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Optional[nn.Module]:
    """
    Initializes and loads the DSINE v02 model with pretrained weights.
    """
    hub_dir = torch.hub.get_dir()
    candidate_dirs = [
        DSINE_REPO_DIR,
        os.path.join(hub_dir, "baegwangbin_DSINE_main"),
        os.path.join(hub_dir, "baegwangbin_DSINE_master"),
    ]
    repo_path = next((d for d in candidate_dirs if os.path.exists(d)), None)

    if repo_path is None:
        target_dir = os.path.join(hub_dir, "baegwangbin_DSINE_main")
        print(f"[DSINE] Cloning repository to {target_dir}...")
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/baegwangbin/DSINE.git", target_dir],
                check=True
            )
            repo_path = target_dir
        except Exception as e:
            print(f"[DSINE] Git clone failed: {e}. Falling back to torch.hub.load...")
            try:
                torch.hub.load("baegwangbin/DSINE", "DSINE", trust_repo=True)
                repo_path = target_dir
            except Exception as e2:
                print(f"[DSINE] torch.hub.load error: {e2}")

    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    if not os.path.exists(ckpt_path):
        print(f"[DSINE] Downloading weights from {DSINE_HF_URL}...")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.hub.download_url_to_file(DSINE_HF_URL, ckpt_path)

    try:
        from models.dsine.v02 import DSINE_v02

        args = types.SimpleNamespace(
            NNET_architecture="v02",
            NNET_encoder_B=5,
            NNET_decoder_NF=2048,
            NNET_decoder_down=8,
            NNET_learned_upsampling=True,
            NNET_decoder_BN=False,
            NRN_prop_ps=5,
            NRN_num_iter_train=5,
            NRN_num_iter_test=5,
            NRN_ray_relu=True,
            NNET_output_dim=3,
            NNET_feature_dim=64,
            NNET_hidden_dim=64
        )

        model = DSINE_v02(args).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=True)
        model.eval()
        print(f"[DSINE] Model successfully loaded on {device} from {ckpt_path}!")
        return model
    except Exception as e:
        print(f"[DSINE] Error initializing model: {e}")
        return None


def run_dsine_batch(
    model: nn.Module,
    images: torch.Tensor,       # (B, 3, H, W) normalized
    device: torch.device,
    fov: float = 60.0
) -> torch.Tensor:
    """
    Runs batched inference through DSINE.
    Returns: (B, H, W, 3) unit normal vectors in camera coordinates.
    """
    from utils.projection import intrins_from_fov
    import utils.utils as utils

    B, _, H, W = images.shape
    pad_lrtb = utils.get_padding(H, W)
    img_padded = F.pad(images, pad_lrtb, mode="constant", value=0.0)

    intrins = intrins_from_fov(new_fov=fov, H=H, W=W, device=device).unsqueeze(0).repeat(B, 1, 1)
    intrins[:, 0, 2] += pad_lrtb[0]
    intrins[:, 1, 2] += pad_lrtb[2]

    with torch.no_grad():
        pred = model(img_padded, intrins=intrins)[-1]  # (B, 3, H_pad, W_pad)
        pred_cropped = pred[:, :, pad_lrtb[2]:pad_lrtb[2] + H, pad_lrtb[0]:pad_lrtb[0] + W]

    # Normalize to exact unit vectors
    norm = torch.norm(pred_cropped, dim=1, keepdim=True).clamp_min(1e-6)
    pred_unit = pred_cropped / norm

    # Rearrange to (B, H, W, 3)
    return pred_unit.permute(0, 2, 3, 1)


def precompute_dsine_normals_in_cache(
    cache_h5_path: str,
    ckpt_path: str = DEFAULT_CKPT_PATH,
    batch_size: int = 16,
    max_samples: int = 0,
    device: Optional[torch.device] = None
) -> int:
    """
    Processes an existing HDF5 cache, computes DSINE normals for all RGB images,
    and saves the 'normals' and 'mask' datasets directly into the HDF5 file.
    """
    if not os.path.exists(cache_h5_path):
        raise FileNotFoundError(f"Target HDF5 cache not found: {cache_h5_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting DSINE Normal Pre-Caching for {cache_h5_path} on {device} ===")
    model = load_dsine_model(ckpt_path=ckpt_path, device=device)
    if model is None:
        raise RuntimeError("Failed to load DSINE model.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    t0 = time.time()
    with h5py.File(cache_h5_path, "a") as f:
        if "rgb" not in f:
            raise KeyError("HDF5 container must contain 'rgb' dataset.")

        total_frames = f["rgb"].shape[0]
        N = min(total_frames, max_samples) if max_samples > 0 else total_frames
        H, W = f["rgb"].shape[1], f["rgb"].shape[2]

        print(f"[CacheManager] Processing {N} frames ({H}x{W}) with batch size {batch_size}...")

        # Create or replace 'normals' and 'mask' datasets
        if "normals" in f:
            del f["normals"]
        if "mask" in f:
            del f["mask"]

        d_norm = f.create_dataset(
            "normals",
            shape=(total_frames, H, W, 3),
            dtype="float32",
            chunks=(1, H, W, 3)
        )
        d_mask = f.create_dataset(
            "mask",
            shape=(total_frames, H, W),
            dtype="float32",
            chunks=(1, H, W)
        )

        processed = 0
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            cur_bs = end_idx - i

            # Load raw RGB (uint8) -> Float [0, 1]
            raw_rgb = f["rgb"][i:end_idx].astype(np.float32) / 255.0  # (B, H, W, 3)
            tensor_rgb = torch.from_numpy(raw_rgb).permute(0, 3, 1, 2).to(device)  # (B, 3, H, W)

            # Apply ImageNet normalization per image
            norm_rgb = torch.stack([normalize(img) for img in tensor_rgb])

            # Run DSINE inference
            batch_normals = run_dsine_batch(model, norm_rgb, device=device)  # (B, H, W, 3)
            normals_np = batch_normals.cpu().numpy()

            # Compute foreground mask from normal variation (silhouette extraction)
            # Background regions typically have flat constant normals (nz ~ 1)
            nz = np.abs(normals_np[..., 2])
            mask_np = (nz < 0.98).astype(np.float32)

            d_norm[i:end_idx] = normals_np
            d_mask[i:end_idx] = mask_np
            processed += cur_bs

            elapsed = time.time() - t0
            fps = processed / max(elapsed, 1e-4)
            if (processed % 50 == 0) or (processed == N):
                print(f"  [Progress {processed:04d}/{N:04d}] Throughput: {fps:.1f} FPS | Elapsed: {elapsed:.1f}s")

        f.attrs["dsine_computed"] = True

    print(f"\n[✔] DSINE Normal Pre-Caching Complete! {processed} frames updated in {cache_h5_path}.")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Automated DSINE Normal Pre-Caching")
    parser.add_argument("--cache_h5", type=str, default="data/cache/3dpw_train_cache.h5")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    precompute_dsine_normals_in_cache(
        cache_h5_path=args.cache_h5,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=device
    )


if __name__ == "__main__":
    main()
