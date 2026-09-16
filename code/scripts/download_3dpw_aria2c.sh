#!/usr/bin/env bash
# High-speed parallel downloader for 3DPW Dataset using aria2c
set -e

DOWNLOAD_DIR="data/downloads"
mkdir -p "$DOWNLOAD_DIR"

BASE_URL="https://virtualhumans.mpi-inf.mpg.de/3DPW"

echo "============================================================"
echo "3DPW High-Speed Downloader (aria2c Multi-Connection)"
echo "Target Directory: $DOWNLOAD_DIR"
echo "============================================================"

# 1. readme_and_demo.zip
echo "[1/3] Downloading readme_and_demo.zip..."
aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
  -d "$DOWNLOAD_DIR" -o readme_and_demo.zip \
  "$BASE_URL/readme_and_demo.zip"

# 2. sequenceFiles.zip
echo "[2/3] Downloading sequenceFiles.zip (277 MB)..."
aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
  -d "$DOWNLOAD_DIR" -o sequenceFiles.zip \
  "$BASE_URL/sequenceFiles.zip"

# 3. imageFiles.zip
echo "[3/3] Downloading imageFiles.zip (4.6 GB)..."
aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
  -d "$DOWNLOAD_DIR" -o imageFiles.zip \
  "$BASE_URL/imageFiles.zip"

echo "============================================================"
echo "[✔] All downloads finished! Unpacking and preparing dataset..."
python3 code/scripts/prepare_3dpw.py
echo "============================================================"
