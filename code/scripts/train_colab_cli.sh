#!/usr/bin/env bash
# ==============================================================================
# Google Colab CLI Training & Orchestration Pipeline
# Script: code/scripts/train_colab_cli.sh
#
# Orchestrates training on Google Colab via the `colab` CLI:
# 1. Checks and reuses existing active sessions to prevent redundant uploads
# 2. Packages and synchronizes local code incrementally
# 3. Verifies remote dataset (reuses existing data on VM; uploads if missing)
# 4. Executes multi-epoch training on Tesla T4 with full step-by-step logging
# ==============================================================================

set -eo pipefail

SESSION_NAME="${SESSION_NAME:-diffnorm-hmr}"
GPU_TYPE="${GPU_TYPE:-T4}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-4800}"
DATASET="${DATASET:-3dpw}"
LOG_FILE="colab_training.log"
SCRATCH_DIR="/tmp/colab_cli_orchestration"

mkdir -p "$SCRATCH_DIR"
touch "$LOG_FILE"

log() {
    local msg="$1"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

log_step() {
    local step_num="$1"
    local step_title="$2"
    echo "" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP ${step_num}] ${step_title}" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
}

log "################################################################################"
log "# DiffNorm-Contact HMR: Google Colab CLI Training Orchestrator"
log "# Target Session: ${SESSION_NAME} | GPU: ${GPU_TYPE} | Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE}"
log "# Log File:       ${LOG_FILE}"
log "################################################################################"

# ------------------------------------------------------------------------------
# STEP 1: Session Management (Reuse vs. Provision)
# ------------------------------------------------------------------------------
log_step "1/6" "Colab Session Management (Reusing Active Session)"

SESSION_STATUS=$(colab status -s "$SESSION_NAME" 2>&1 || true)
if echo "$SESSION_STATUS" | grep -q "gpu-"; then
    log "[✔] Existing session '${SESSION_NAME}' found and ACTIVE!"
    log "    Details: ${SESSION_STATUS}"
    log "    [POLICY] Reusing active session to preserve uploaded dataset and dependencies."
else
    log "[+] No active session named '${SESSION_NAME}' detected. Provisioning fresh session..."
    log "    Command: colab new -s ${SESSION_NAME} --gpu ${GPU_TYPE}"
    colab new -s "$SESSION_NAME" --gpu "$GPU_TYPE" | tee -a "$LOG_FILE"
    log "[✔] Session '${SESSION_NAME}' successfully provisioned with GPU ${GPU_TYPE}."
fi

# ------------------------------------------------------------------------------
# STEP 2: Remote Hardware & Environment Diagnostics
# ------------------------------------------------------------------------------
log_step "2/6" "Remote Hardware & CUDA Verification"

cat << 'EOF' > "${SCRATCH_DIR}/check_hardware.py"
import torch
import os
import sys

print("[Remote VM] Environment Diagnostics:")
print(f"  Working Directory: {os.getcwd()}")
print(f"  PyTorch Version:   {torch.__version__}")
print(f"  CUDA Available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU Hardware:      {device_name} ({vram_gb:.2f} GB VRAM)")
    if vram_gb < 12.0:
        print("  [WARNING] Allocated VRAM is under 12 GB. High batch sizes may OOM.")
else:
    print("  [ERROR] No CUDA device detected! Please ensure --gpu T4 was specified.")
    sys.exit(1)
EOF

colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/check_hardware.py" --timeout 120 | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 3: Package and Synchronize Local Code to Remote VM
# ------------------------------------------------------------------------------
log_step "3/6" "Code Packaging & Incremental Synchronization"

CODE_TAR="${SCRATCH_DIR}/hmr_code.tar.gz"
log "[+] Packaging local codebase (code/src, code/scripts, code/tests)..."
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -czf "$CODE_TAR" -C code .
log "[✔] Packaged code archive: $(du -h "$CODE_TAR" | cut -f1)"

log "[+] Uploading code archive to /content/hmr_code.tar.gz..."
colab upload -s "$SESSION_NAME" "$CODE_TAR" /content/hmr_code.tar.gz | tee -a "$LOG_FILE"

cat << 'EOF' > "${SCRATCH_DIR}/unpack_code.py"
import os, tarfile

os.makedirs("/content/code", exist_ok=True)
os.makedirs("/content/checkpoints", exist_ok=True)
os.makedirs("/content/data/cache", exist_ok=True)

with tarfile.open("/content/hmr_code.tar.gz", "r:gz") as tar:
    tar.extractall("/content/code")

# Symlink src into /content so 'import src' works globally on the VM
for d in ["src", "scripts", "tests"]:
    src_link = f"/content/{d}"
    if not os.path.exists(src_link):
        os.symlink(f"/content/code/{d}", src_link)

print("[Remote VM] Code successfully extracted and symlinked at /content/code.")
EOF

colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/unpack_code.py" --timeout 120 | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 4: Remote Dataset Verification & Ingestion (Idempotent)
# ------------------------------------------------------------------------------
log_step "4/6" "Dataset Verification on Remote Session (Reuse / Ingestion)"

cat << 'EOF' > "${SCRATCH_DIR}/check_dataset.py"
import os, glob

seq_train = glob.glob("/content/data/3dpw/sequenceFiles/train/*.pkl")
img_dirs = [d for d in glob.glob("/content/data/3dpw/imageFiles/*") if os.path.isdir(d)]

print(f"DATASET_CHECK: {len(seq_train)} train sequences, {len(img_dirs)} image sequence dirs")
if len(seq_train) >= 20 and len(img_dirs) >= 20:
    print("STATUS: COMPLETE")
elif len(seq_train) >= 20:
    print("STATUS: SEQUENCES_ONLY")
else:
    print("STATUS: MISSING")
EOF

DATASET_STATUS=$(colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/check_dataset.py" --timeout 120 | grep "STATUS:" | cut -d' ' -f2 || true)
log "[+] Remote dataset status: ${DATASET_STATUS}"

if [ "$DATASET_STATUS" == "COMPLETE" ]; then
    log "[✔] 3DPW dataset is ALREADY COMPLETE on remote session. Reusing existing data (0 MB upload needed)!"
else
    log "[!] Dataset incomplete on remote session. Running rapid preparation pipeline..."
    
    cat << 'EOF' > "${SCRATCH_DIR}/fetch_remote_dataset.py"
import subprocess, os, zipfile, glob

downloads_dir = "/content/data/downloads"
data_dir = "/content/data/3dpw"
os.makedirs(downloads_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], check=True)
base_url = "https://virtualhumans.mpi-inf.mpg.de/3DPW"

# 1. sequenceFiles.zip
seq_zip = f"{downloads_dir}/sequenceFiles.zip"
if not os.path.exists(seq_zip) or os.path.getsize(seq_zip) < 200 * 1024 * 1024:
    print("[Colab] Fetching sequenceFiles.zip via aria2c...")
    cmd = f"aria2c -c -x 16 -s 16 -k 1M --file-allocation=none -d {downloads_dir} -o sequenceFiles.zip {base_url}/sequenceFiles.zip"
    subprocess.run(cmd, shell=True, check=True)

# 2. readme_and_demo.zip
demo_zip = f"{downloads_dir}/readme_and_demo.zip"
if not os.path.exists(demo_zip):
    print("[Colab] Fetching readme_and_demo.zip via aria2c...")
    cmd = f"aria2c -c -x 16 -s 16 -k 1M --file-allocation=none -d {downloads_dir} -o readme_and_demo.zip {base_url}/readme_and_demo.zip"
    subprocess.run(cmd, shell=True, check=True)

# Extract sequences
print("[Colab] Unpacking sequenceFiles...")
with zipfile.ZipFile(seq_zip, "r") as z:
    z.extractall(data_dir)
with zipfile.ZipFile(demo_zip, "r") as z:
    z.extractall(data_dir)

# 3. imageFiles.zip
img_zip = f"{downloads_dir}/imageFiles.zip"
if not os.path.exists(img_zip) or os.path.getsize(img_zip) < 4 * 1024 * 1024 * 1024:
    print("[Colab] Fetching imageFiles.zip via aria2c (Colab high-speed connection)...")
    cmd = f"aria2c -c -x 16 -s 16 -k 1M --file-allocation=none -d {downloads_dir} -o imageFiles.zip {base_url}/imageFiles.zip"
    subprocess.run(cmd, shell=True, check=True)

print("[Colab] Unpacking imageFiles...")
with zipfile.ZipFile(img_zip, "r") as z:
    z.extractall(data_dir)

print("[✔] Remote 3DPW preparation complete!")
EOF

    colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/fetch_remote_dataset.py" --timeout 900 | tee -a "$LOG_FILE"
fi

# ------------------------------------------------------------------------------
# STEP 5: Remote Environment Dependencies & Synthetic SMPL Check
# ------------------------------------------------------------------------------
log_step "5/6" "Remote Python Dependencies & Model Setup"

cat << 'EOF' > "${SCRATCH_DIR}/setup_deps.py"
import subprocess, sys

# Ensure required libraries are installed
pkgs = ["h5py", "plyfile", "tqdm", "scipy", "geffnet", "timm", "huggingface_hub"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

# Initialize and verify dataset loader & fallback SMPL model
sys.path.insert(0, "/content/code")
from src.pipeline.dataset_3dpw import Dataset3DPW
from src.geometry.smpl_wrapper import SMPLWrapper

smpl = SMPLWrapper()
print(f"[Remote VM] SMPL Wrapper initialized: V={smpl.v_template.shape[0]}, F={smpl.faces.shape[0]}, Device={smpl.faces.device}")

ds = Dataset3DPW(root_dir="/content/data/3dpw", split="train")
print(f"[Remote VM] Dataset3DPW verified: {len(ds)} valid frames loaded successfully!")
EOF

colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/setup_deps.py" --timeout 300 | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 5b: Precomputed HDF5 Feature Cache & DSINE Normal Ingestion
# ------------------------------------------------------------------------------
log_step "5b/6" "High-Speed Feature Caching Pipeline & DSINE Normals"

cat << EOF > "${SCRATCH_DIR}/check_and_build_cache.py"
import os, subprocess, sys
import h5py, numpy as np

cache_path = "/content/data/cache/3dpw_train_cache.h5"

# Check if cache exists and has required datasets
needs_rebuild = False
if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 1024 * 1024:
    needs_rebuild = True
else:
    try:
        with h5py.File(cache_path, "r") as f:
            if "theta" not in f or "trans" not in f:
                print("[Remote VM] Cache missing pose tensors ('theta'/'trans'); rebuilding...")
                needs_rebuild = True
    except Exception:
        needs_rebuild = True

if needs_rebuild:
    if os.path.exists(cache_path):
        os.remove(cache_path)
    print(f"[Remote VM] Generating fast HDF5 pre-cache ({cache_path})...")
    cmd = [
        sys.executable,
        "/content/code/scripts/build_3dpw_cache.py",
        "--data_dir", "/content/data/3dpw",
        "--output_h5", cache_path,
        "--max_samples", "${MAX_SAMPLES}",
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("[Remote VM] Cache generation warning; will fall back to direct loader.")
    else:
        print("[Remote VM] Feature cache base generated successfully!")

# Extract DSINE surface normals into cache if not present or pseudo
needs_normals = True
if os.path.exists(cache_path):
    try:
        with h5py.File(cache_path, "r") as f:
            if f.attrs.get("dsine_computed", False):
                needs_normals = False
                print(f"[Remote VM] Verified existing DSINE normals in cache ({f['normals'].shape})")
            elif "normals" in f and f["normals"].shape[0] > 0:
                sample_n = f["normals"][0]
                if float(np.var(sample_n[..., 0]) + np.var(sample_n[..., 1])) > 1e-4:
                    needs_normals = False
                    print(f"[Remote VM] Verified existing authentic DSINE normals in cache ({f['normals'].shape})")
    except Exception:
        pass

if needs_normals and os.path.exists(cache_path):
    print("[Remote VM] Precomputing real DSINE surface normals and silhouette masks...")
    cmd_norm = [
        sys.executable,
        "/content/code/scripts/extract_dsine_normals.py",
        "--cache_h5", cache_path,
        "--batch_size", "16",
        "--max_samples", "${MAX_SAMPLES}",
        "--device", "cuda",
    ]
    res_norm = subprocess.run(cmd_norm)
    if res_norm.returncode == 0:
        print("[Remote VM] Real DSINE surface normals successfully pre-cached!")
    else:
        print("[Remote VM] Warning: DSINE normal extraction returned non-zero code.")

print(f"[Remote VM] Feature cache ready: {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.1f} MB)")
EOF

colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/check_and_build_cache.py" --timeout 900 | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 6: Execute Batched Training Pipeline with Cross-Session Resumption
# ------------------------------------------------------------------------------
log_step "6/6" "Launching Training Job on Colab GPU (${GPU_TYPE})"

# Upload local checkpoint if present to resume training seamlessly across sessions
if [ -f "checkpoints/diffnorm_contact_hmr_checkpoint.pt" ]; then
    log "[+] Found local checkpoint. Synchronizing to Colab VM for stateful resume..."
    colab upload -s "$SESSION_NAME" "checkpoints/diffnorm_contact_hmr_checkpoint.pt" "/content/checkpoints/diffnorm_contact_hmr_checkpoint.pt" || true
fi

cat << EOF > "${SCRATCH_DIR}/run_training.py"
import subprocess, sys

cmd = [
    sys.executable,
    "/content/code/scripts/train_colab.py",
    "--dataset", "cache",
    "--epochs", "${EPOCHS}",
    "--batch_size", "${BATCH_SIZE}",
    "--max_samples", "${MAX_SAMPLES}",
    "--num_workers", "2",
    "--data_dir", "/content/data/3dpw",
    "--cache_path", "/content/data/cache/3dpw_train_cache.h5",
    "--checkpoint_dir", "/content/checkpoints",
    "--resume",
]

print(f"[Remote VM] Running training command: {' '.join(cmd)}")
res = subprocess.run(cmd)
if res.returncode != 0:
    print(f"[Remote VM] Training failed with returncode {res.returncode}")
    sys.exit(res.returncode)
else:
    print("[Remote VM] Training finished successfully!")
EOF

colab exec -s "$SESSION_NAME" -f "${SCRATCH_DIR}/run_training.py" --timeout 7200 | tee -a "$LOG_FILE"

log "[+] Downloading trained checkpoint from Colab VM to local workspace..."
mkdir -p checkpoints
colab download -s "$SESSION_NAME" "checkpoints/diffnorm_contact_hmr_checkpoint.pt" "checkpoints/diffnorm_contact_hmr_checkpoint.pt" || \
colab download -s "$SESSION_NAME" "content/checkpoints/diffnorm_contact_hmr_checkpoint.pt" "checkpoints/diffnorm_contact_hmr_checkpoint.pt" || true

log "[+] Stopping Colab session '${SESSION_NAME}' to prevent idle compute burn (Anti-Idle Policy)..."
colab stop -s "$SESSION_NAME" || true

log "================================================================================"
log "[$(date '+%Y-%m-%d %H:%M:%S')] Training Pipeline Complete! Logs saved to ${LOG_FILE}."
log "================================================================================"
