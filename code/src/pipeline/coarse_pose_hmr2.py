"""
4D-Humans (HMR 2.0) Coarse Pose Estimator Module.
Provides feed-forward initialization for DiffNorm-Contact HMR test-time optimization.
Extracts coarse SMPL parameters (theta, beta, trans) from monocular human crops.
"""

import os
import sys
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.geometry.smpl_wrapper import SMPLWrapper


DEFAULT_HMR2_URL = "https://huggingface.co/camenduru/4D-Humans/resolve/main/HMR2/logs/train/multiruns/hmr2/0/checkpoints/epoch%3D35-step%3D1000000.ckpt"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/4DHumans")


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 rotation matrices to axis-angle vectors via the matrix logarithm map.
    R: (..., 3, 3)
    Returns: (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    sin_angle = torch.sin(angle)
    small_mask = angle.abs() < 1e-5
    safe_sin = torch.where(small_mask, torch.ones_like(sin_angle), sin_angle)
    scale = torch.where(small_mask, 0.5 + angle ** 2 / 12.0, angle / (2.0 * safe_sin))

    rx = (R[..., 2, 1] - R[..., 1, 2]) * scale
    ry = (R[..., 0, 2] - R[..., 2, 0]) * scale
    rz = (R[..., 1, 0] - R[..., 0, 1]) * scale
    return torch.stack([rx, ry, rz], dim=-1)


class CoarsePoseHMR2(nn.Module):
    """
    Wrapper for 4D-Humans (HMR 2.0) feed-forward pose prediction.
    Outputs initial SMPL pose parameters (theta_0, beta_0, trans_0) to seed
    DiffNorm-Contact's normal-torque and collision-physics optimization.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        auto_download: bool = True,
        require_model: bool = False
    ):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path or os.path.join(DEFAULT_CACHE_DIR, "epoch=35-step=1000000.ckpt")
        self.has_weights = False
        self.model = None
        self.require_model = require_model

        self.smpl = SMPLWrapper(device=device).to(device)

        if os.path.exists(self.checkpoint_path):
            self._load_weights(self.checkpoint_path)
        elif auto_download:
            self._try_download_or_fallback()
        else:
            print(f"[4D-Humans] Checkpoint not found at {self.checkpoint_path}. Operating in fallback mode.")

        if self.require_model and self.model is None:
            raise RuntimeError(
                f"[4D-Humans] Execution halted: require_model=True but valid HMR 2.0 model could not be instantiated "
                f"from '{self.checkpoint_path}'. Synthetic or kinematic fallbacks are strictly prohibited in benchmark evaluation."
            )

    def _load_weights(self, ckpt_path: str):
        """Loads state_dict and initializes HMR 2.0 architecture."""
        try:
            print(f"[4D-Humans] Loading HMR 2.0 weights from {ckpt_path}...")
            # Try importing official 4D-Humans package if installed
            try:
                from hmr2.models import HMR2
                self.model = HMR2.load_from_checkpoint(ckpt_path).to(self.device).eval()
                self.has_weights = True
                print("[4D-Humans] Official 4D-Humans HMR2 model loaded successfully!")
                return
            except ImportError:
                pass

            state = torch.load(ckpt_path, map_location=self.device)
            self.state_dict_raw = state
            # If architecture cannot be instantiated, do not claim active model
            self.model = None
            self.has_weights = False
            print("[4D-Humans] Checkpoint loaded into memory, but hmr2 architecture is not installed. Fallback mode active.")
        except Exception as e:
            print(f"[4D-Humans] Warning: Failed to load checkpoint: {e}. Fallback mode active.")
            self.has_weights = False
            self.model = None

    def _try_download_or_fallback(self):
        """Attempts to download checkpoint or gracefully activates kinematic fallback."""
        os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            print(f"[4D-Humans] Pretrained weights not cached at {self.checkpoint_path}.")
            print(f"[4D-Humans] Remote download URL: {DEFAULT_HMR2_URL}")
            print("[4D-Humans] Activating fast kinematic pose initializer for local execution.")
            self.has_weights = False

    def forward(
        self,
        images: torch.Tensor,  # (B, 3, 256, 256) in [0, 1]
        intrinsics: Optional[torch.Tensor] = None,  # (B, 3, 3)
        keypoints_2d: Optional[torch.Tensor] = None  # (B, J, 2)
    ) -> Dict[str, torch.Tensor]:
        """
        Runs feed-forward coarse pose estimation.
        Returns:
            - theta: (B, 24, 3) joint axis-angle rotations
            - rotmats: (B, 24, 3, 3) rotation matrices
            - beta:  (B, 10) shape parameters
            - trans: (B, 3) camera translation
            - joints: (B, 24, 3) 3D joint locations
        """
        B = images.shape[0]

        # If actual full ViT backbone is initialized with weights
        if self.has_weights and self.model is not None:
            with torch.no_grad():
                out = self.model(images)
                pred_smpl = out["pred_smpl_params"]
                # Concatenate global orientation and body pose: (B, 1, 3, 3) + (B, 23, 3, 3) -> (B, 24, 3, 3)
                if "global_orient" in pred_smpl and "body_pose" in pred_smpl:
                    rotmats = torch.cat([pred_smpl["global_orient"], pred_smpl["body_pose"]], dim=1)
                elif "body_pose" in pred_smpl and pred_smpl["body_pose"].shape[1] == 24:
                    rotmats = pred_smpl["body_pose"]
                else:
                    raise ValueError(f"Unexpected HMR2 output keys: {pred_smpl.keys()}")

                theta = rotation_matrix_to_axis_angle(rotmats)  # (B, 24, 3)
                beta = pred_smpl["betas"][:, :10]
                trans = out.get("pred_cam_t", torch.tensor([[0.0, 0.2, 2.8]], device=self.device).repeat(B, 1))

                smpl_out = self.smpl(theta=rotmats, beta=beta, trans=trans)
                return {
                    "theta": theta,
                    "rotmats": rotmats,
                    "beta": beta,
                    "trans": trans,
                    "joints": smpl_out["joints"],
                    "is_coarse": True,
                    "source": "4D-Humans-HMR2.0"
                }

        # Kinematic unposed natural prior fallback (Zero ground-truth dependence)
        # Initializes canonical standing posture with realistic depth scale (~2.8m)
        theta = torch.zeros(B, 24, 3, device=self.device)
        theta[:, 1, 2] = 0.05   # Slight hip rest angle
        theta[:, 2, 2] = -0.05
        theta[:, 16, 2] = 0.15  # Slight shoulder abduction
        theta[:, 17, 2] = -0.15

        beta = torch.zeros(B, 10, device=self.device)
        trans = torch.tensor([[0.0, 0.2, 2.8]], device=self.device).repeat(B, 1)

        with torch.no_grad():
            smpl_out = self.smpl(theta=theta, beta=beta, trans=trans)
            joints = smpl_out["joints"]

        return {
            "theta": theta,
            "beta": beta,
            "trans": trans,
            "joints": joints,
            "is_coarse": True,
            "source": "Kinematic-Coarse-Prior"
        }
