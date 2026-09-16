"""
Utility script to extract, organize, and verify the 3DPW dataset downloaded via aria2c.
Extracts sequenceFiles.zip and imageFiles.zip into data/3dpw/.
"""

import os
import sys
import zipfile
import pickle
import glob
from pathlib import Path


def extract_zip(zip_path: str, target_dir: str, desc: str = ""):
    if not os.path.exists(zip_path):
        print(f"[-] {zip_path} not found yet.")
        return False
    
    # Check if download is still active (aria2 creates .aria2 control files while downloading)
    aria2_file = f"{zip_path}.aria2"
    if os.path.exists(aria2_file):
        print(f"[!] {zip_path} is still downloading (active .aria2 control file found).")
        return False
        
    print(f"[+] Extracting {desc or zip_path} to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print(f"[✔] Successfully extracted {zip_path}.")
    return True


def verify_3dpw(root_dir: str = "data/3dpw"):
    seq_dir = Path(root_dir) / "sequenceFiles"
    img_dir = Path(root_dir) / "imageFiles"
    
    print("\n" + "=" * 60)
    print("3DPW Dataset Verification & Inventory")
    print("=" * 60)
    
    splits = ["train", "validation", "test"]
    total_seqs = 0
    total_frames = 0
    
    for split in splits:
        split_dir = seq_dir / split
        if not split_dir.exists():
            # Sometimes 3DPW uses 'val' instead of 'validation'
            split_dir = seq_dir / "val" if split == "validation" else split_dir
            
        if not split_dir.exists():
            print(f"[-] Split '{split}' directory not found at {split_dir}")
            continue
            
        pkl_files = list(split_dir.glob("*.pkl"))
        print(f"[+] Split '{split}': Found {len(pkl_files)} sequence files.")
        total_seqs += len(pkl_files)
        
        split_frames = 0
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                num_frames = len(data.get("poses", [[]])[0]) if "poses" in data else 0
                split_frames += num_frames
            except Exception as e:
                print(f"    [!] Error reading {pkl_file.name}: {e}")
                
        print(f"    Total frames in '{split}': {split_frames}")
        total_frames += split_frames

    print("-" * 60)
    print(f"Total Sequences Verified: {total_seqs}")
    print(f"Total Video Frames: {total_frames}")
    
    if img_dir.exists():
        seq_dirs = [d for d in img_dir.iterdir() if d.is_dir()]
        total_images = len(list(img_dir.glob("*/*.jpg")))
        print(f"[+] Found {len(seq_dirs)} image sequence directories with {total_images} RGB images.")
    else:
        print(f"[-] imageFiles directory not found at {img_dir}")
        
    print("=" * 60 + "\n")


def main():
    root_dir = "data/3dpw"
    downloads_dir = "data/downloads"
    
    seq_zip = os.path.join(downloads_dir, "sequenceFiles.zip")
    img_zip = os.path.join(downloads_dir, "imageFiles.zip")
    demo_zip = os.path.join(downloads_dir, "readme_and_demo.zip")
    
    extract_zip(demo_zip, root_dir, "Readme & Demo")
    extract_zip(seq_zip, root_dir, "Sequence Files")
    extract_zip(img_zip, root_dir, "Image Files")
    
    verify_3dpw(root_dir)


if __name__ == "__main__":
    main()
