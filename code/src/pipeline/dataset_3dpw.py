"""
3DPW (3D Poses in the Wild) Dataset Loader for Training and Evaluation.
Parses official 3DPW sequence .pkl files, extracts ground-truth SMPL parameters (theta, beta, trans),
camera intrinsics (K), camera extrinsics (campose), and aligns with frame images.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Dataset3DPW(Dataset):
    """
    3DPW Dataset loader.
    Expected directory layout:
      root_dir/
        sequenceFiles/
          train/*.pkl
          validation/*.pkl
          test/*.pkl
        imageFiles/
          <sequence_name>/image_00000.jpg ...
    """
    def __init__(
        self,
        root_dir: str = "data/3dpw",
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        use_cache: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.H, self.W = image_size
        self.use_cache = use_cache

        self.seq_dir = os.path.join(root_dir, "sequenceFiles", split)
        self.img_dir = os.path.join(root_dir, "imageFiles")

        self.samples: List[Dict] = []
        if os.path.exists(self.seq_dir):
            self._load_sequences()
        else:
            print(f"[Dataset3DPW] Directory '{self.seq_dir}' not found.")

    def _load_sequences(self):
        pkl_files = sorted(glob.glob(os.path.join(self.seq_dir, "*.pkl")))
        print(f"[Dataset3DPW] Found {len(pkl_files)} sequences in split '{self.split}'.")

        for pkl_path in pkl_files:
            seq_name = os.path.splitext(os.path.basename(pkl_path))[0]
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
            except Exception as e:
                print(f"[Dataset3DPW] Warning: Failed to load {pkl_path}: {e}")
                continue

            poses_list = data["poses"]            # List of arrays per actor: each is (T, 72)
            trans_list = data["trans"]            # List of arrays per actor: each is (T, 3)
            betas_list = data["betas"]            # List of arrays per actor
            K = np.array(data["cam_intrinsics"], dtype=np.float32)  # (3, 3)
            
            raw_campose = data.get("cam_poses", data.get("campose", None))
            campose = np.array(raw_campose, dtype=np.float32) if raw_campose is not None else None
            campose_valid = data.get("campose_valid", None)
            num_actors = len(poses_list)

            # Frame indices
            frame_ids = data.get("img_frame_ids", np.arange(len(poses_list[0])))
            T = len(frame_ids)

            for t_idx in range(T):
                f_id = int(frame_ids[t_idx])
                # Image filename pattern: image_00000.jpg or image_00000.png
                img_path = os.path.join(self.img_dir, seq_name, f"image_{f_id:05d}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.img_dir, seq_name, f"image_{f_id:05d}.png")

                for a_idx in range(num_actors):
                    if campose_valid is not None and a_idx < len(campose_valid):
                        if t_idx < len(campose_valid[a_idx]) and not campose_valid[a_idx][t_idx]:
                            continue

                    # SMPL pose (72) -> 24 joints x 3 axis-angle
                    theta_72 = poses_list[a_idx][t_idx]
                    theta_24x3 = theta_72.reshape(24, 3)
                    trans_3 = trans_list[a_idx][t_idx]
                    
                    beta_actor = betas_list[a_idx]
                    if beta_actor.ndim == 2:
                        beta_10 = beta_actor[0][:10]
                    else:
                        beta_10 = beta_actor[:10]

                    c_mat = torch.eye(4)
                    if campose is not None and t_idx < len(campose):
                        c_mat = torch.from_numpy(campose[t_idx]).float()

                    self.samples.append({
                        "sequence_name": seq_name,
                        "frame_idx": f_id,
                        "actor_idx": a_idx,
                        "image_path": img_path,
                        "theta": torch.from_numpy(theta_24x3).float(),
                        "beta": torch.from_numpy(beta_10).float(),
                        "trans": torch.from_numpy(trans_3).float(),
                        "K": torch.from_numpy(K).float(),
                        "campose": c_mat,
                    })

        print(f"[Dataset3DPW] Loaded {len(self.samples)} total frame instances for split '{self.split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample["image_path"]

        # Load image if it exists on disk, otherwise generate neutral placeholder
        K = sample["K"].clone()
        if os.path.exists(img_path):
            orig_img = Image.open(img_path)
            orig_w, orig_h = orig_img.size
            img = orig_img.convert("RGB").resize((self.W, self.H))
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            if orig_w > 0 and orig_h > 0:
                K[0, :] = K[0, :] * (float(self.W) / float(orig_w))
                K[1, :] = K[1, :] * (float(self.H) / float(orig_h))
        else:
            img_tensor = torch.ones(self.H, self.W, 3, dtype=torch.float32) * 0.5
            # Scale intrinsics assuming 3DPW standard optical center
            if K[0, 2] > self.W:
                orig_w = float(K[0, 2] * 2.0)
                orig_h = float(K[1, 2] * 2.0)
                K[0, :] = K[0, :] * (float(self.W) / orig_w)
                K[1, :] = K[1, :] * (float(self.H) / orig_h)

        return {
            "image": img_tensor,            # (H, W, 3) in [0, 1]
            "theta": sample["theta"],       # (24, 3) axis-angle
            "beta": sample["beta"],         # (10,) shape parameters
            "trans": sample["trans"],       # (3,) translation in meters
            "K": K,                         # (3, 3) camera intrinsics scaled to (H, W)
            "campose": sample["campose"],   # (4, 4) camera extrinsics
            "seq_name": sample["sequence_name"],
            "frame_idx": sample["frame_idx"],
        }


def create_mock_3dpw_sample(root_dir: str = "data/3dpw", split: str = "train") -> str:
    """
    Creates a synthetic 3DPW sequence for testing and validation without full dataset download.
    Returns path to created .pkl file.
    """
    seq_dir = os.path.join(root_dir, "sequenceFiles", split)
    img_dir = os.path.join(root_dir, "imageFiles", "mock_outdoor_walk")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    T = 10
    # Create mock frames
    for i in range(T):
        frame_path = os.path.join(img_dir, f"image_{i:05d}.jpg")
        dummy_img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
        dummy_img.save(frame_path)

    # Mock sequence dictionary matching official 3DPW format
    mock_data = {
        "sequence": "mock_outdoor_walk",
        "img_frame_ids": np.arange(T),
        "poses": [np.random.randn(T, 72).astype(np.float32) * 0.1],
        "trans": [np.zeros((T, 3), dtype=np.float32)],
        "betas": [np.zeros(10, dtype=np.float32)],
        "cam_intrinsics": np.array([[300.0, 0.0, 128.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "campose": np.repeat(np.eye(4)[np.newaxis, ...], T, axis=0).astype(np.float32),
    }

    pkl_path = os.path.join(seq_dir, "mock_outdoor_walk.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(mock_data, f, protocol=2)

    return pkl_path
