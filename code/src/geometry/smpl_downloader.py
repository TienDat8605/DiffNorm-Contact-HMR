"""
SMPL Model Verification and Setup Utility.
Scans for official MPI SMPL model files, validates internal parameters,
and standardizes symlinks to models/smpl/SMPL_NEUTRAL.pkl.
"""

import os
import glob
import pickle
from typing import Optional, Tuple


EXPECTED_KEYS = ["v_template", "weights", "f", "J_regressor"]


def inspect_smpl_file(file_path: str) -> Tuple[bool, str]:
    """Inspects a pickle file to verify it is a valid SMPL model."""
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        missing = [k for k in EXPECTED_KEYS if k not in data]
        if missing:
            return False, f"Missing required SMPL keys: {missing}"
        n_verts = data["v_template"].shape[0]
        n_faces = data["f"].shape[0]
        return True, f"Valid SMPL model: {n_verts} vertices, {n_faces} faces"
    except Exception as e:
        return False, f"Failed to load pickle: {e}"


def setup_smpl_model(target_dir: str = "models/smpl") -> Optional[str]:
    """
    Scans target_dir and parent directories for any official SMPL file,
    symlinks it to SMPL_NEUTRAL.pkl, and returns the canonical path.
    """
    os.makedirs(target_dir, exist_ok=True)
    canonical_path = os.path.join(target_dir, "SMPL_NEUTRAL.pkl")

    if os.path.exists(canonical_path):
        valid, msg = inspect_smpl_file(canonical_path)
        if valid:
            print(f"[SMPL Setup] Found canonical model at {canonical_path} ({msg})")
            return canonical_path

    # Search for alternative names in target_dir or parent directories
    candidates = (
        glob.glob(os.path.join(target_dir, "*neutral*.pkl")) +
        glob.glob(os.path.join(target_dir, "basicModel*.pkl")) +
        glob.glob("../*neutral*.pkl") +
        glob.glob("basicModel*.pkl")
    )

    for cand in candidates:
        valid, msg = inspect_smpl_file(cand)
        if valid:
            print(f"[SMPL Setup] Found valid model at {cand}. Creating symlink at {canonical_path}...")
            if os.path.islink(canonical_path) or os.path.exists(canonical_path):
                os.remove(canonical_path)
            os.symlink(os.path.abspath(cand), canonical_path)
            return canonical_path

    print(
        f"[SMPL Setup] No official SMPL model found in {target_dir}.\n"
        f"Please place 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' into '{target_dir}/SMPL_NEUTRAL.pkl'.\n"
        f"The system will continue using the verified canonical humanoid template."
    )
    return None


if __name__ == "__main__":
    setup_smpl_model()
