# Repository Reorganization & 3DPW Dataset Preparation

## 1. Directory Structure

Per the user's instructions, all documentation, research papers, proposals, notes, and codebase modules have been organized into the following clean hierarchy:

```
/home/dat/HMR/
├── documents/
│   ├── pdf/                          # Formatted for human reading & review
│   │   ├── papers/                   # Foundational & related research papers
│   │   │   ├── 2308.04079v1_compressed.pdf
│   │   │   ├── 2409.04196v2.pdf
│   │   │   ├── 2605.02784v2.pdf      # HumanSplatHMR
│   │   │   ├── 3603618.pdf           # HMR Survey & Benchmarks
│   │   │   └── frai-8-1709229.pdf
│   │   ├── notes/                    # Implementation plans, reports, walkthroughs
│   │   │   ├── diffnorm_contact_hmr_implementation_plan.pdf
│   │   │   ├── walkthrough.pdf
│   │   │   ├── papers_analysis_report.pdf
│   │   │   └── repository_organization_and_dataset_setup.pdf
│   │   └── proposals/                # Research proposals and comparative evaluations
│   │       ├── proposal.pdf          # DiffNorm-Contact HMR Full Proposal
│   │       ├── hmr_3dgs_critique_and_proposal.pdf
│   │       ├── comparison_proposal_vs_humansplathmr.pdf
│   │       ├── falsifiable_hmr_3dgs_proposal.pdf
│   │       └── deep_research_hmr_3dgs_proposals.pdf
│   └── md/                           # Formatted for AI agent parsing & analysis
│       ├── papers/
│       │   └── 3603618.md            # Markdown transcription of survey & tables
│       ├── notes/
│       │   ├── diffnorm_contact_hmr_implementation_plan.md
│       │   ├── walkthrough.md
│       │   ├── AGENTS.md             # Project guidelines & agent instructions
│       │   └── repository_organization_and_dataset_setup.md
│       └── proposals/
│           ├── proposal.md
│           ├── hmr_3dgs_critique_and_proposal.md
│           └── comparison_proposal_vs_humansplathmr.md
│
├── code/                             # Complete codebase
│   ├── src/                          # Core geometry, physics, rendering, & pipeline
│   │   ├── geometry/                 # SMPL wrapper, kinematic segments, mesh graph
│   │   ├── gaussian/                 # Tangential disks & spatial covariances
│   │   ├── physics/                  # Analytical Gaussian overlap K_ij & collision loss
│   │   ├── rendering/                # Screen-space normal rasterizer & loss functions
│   │   ├── optimization/             # Dual-frequency optimizer & gradient router
│   │   └── pipeline/                 # 3DPW loader, feature cache, single-image optimizer
│   ├── scripts/                      # Training, evaluation, preprocessing, & download
│   │   ├── download_3dpw_aria2c.sh   # High-speed parallel 3DPW aria2c downloader
│   │   ├── prepare_3dpw.py           # Extraction, verification, and sequence inventory
│   │   ├── train_colab.py            # Multi-epoch batched trainer for Google Colab T4
│   │   ├── preprocess_3dpw.py        # HDF5 foundation feature streamer
│   │   ├── eval_3dpw.py              # 3DPW benchmark evaluator (MPJPE, PA-MPJPE, PVE)
│   │   ├── eval_rich_contact.py      # RICH self-penetration evaluator (V_pen)
│   │   └── eval_cape_clothing.py     # CAPE loose garment evaluation runner
│   ├── tests/                        # 14 automated unit tests (100% passing)
│   └── notebooks/                    # Google Colab T4 interactive notebook
│
├── data/                             # Dataset storage & caching
│   ├── downloads/                    # Download archives (.zip) managed by aria2c
│   ├── 3dpw/                         # Extracted sequenceFiles & imageFiles
│   └── cache/                        # Precomputed DSINE normal maps & masks (HDF5)
├── models/                           # SMPL body templates (models/smpl/SMPL_NEUTRAL.pkl)
└── checkpoints/                      # Trained model weights & optimization checkpoints
```

---

## 2. 3DPW Dataset aria2c Status & Extraction

The 3DPW dataset is downloaded directly from the official Max Planck Institute server (`https://virtualhumans.mpi-inf.mpg.de/3DPW/`) using `aria2c` multi-connection segmented downloading (`-x 16 -s 16 -k 1M -c --file-allocation=none`):

1. **`readme_and_demo.zip` (4.5 KB):**
   - Status: **Complete & Extracted** (`data/3dpw/readme.txt`).
2. **`sequenceFiles.zip` (277 MB):**
   - Status: **100% Complete & Extracted** (`data/3dpw/sequenceFiles/`).
   - **Audit Results:**
     - **Train Split:** 25 sequences, 17,914 frames (yielding 22,745 valid actor frame instances).
     - **Validation Split:** 12 sequences, 8,509 frames.
     - **Test Split:** 24 sequences, 26,240 frames.
     - **Total:** 61 sequences, 52,663 fully annotated video frames.
     - Formats verified: 72-dim SMPL axis-angle poses, 10-dim betas, 3-dim root translations, $4 \times 4$ camera extrinsics, and $3 \times 3$ intrinsic matrices.
3. **`imageFiles.zip` (4.6 GB):**
   - Status: **100% Complete & Extracted** (`data/3dpw/imageFiles/`).
   - **Audit Results:** 62 image sequence directories containing **53,806 RGB images** (`.jpg`).
   - Download accelerated via `aria2c` segmented streaming, completing in ~35 minutes.

### Automation Scripts & Loading Verification

- **Parallel Downloader Script:** [`code/scripts/download_3dpw_aria2c.sh`](file:///home/dat/HMR/code/scripts/download_3dpw_aria2c.sh)
- **Automatic Unpacker & Inventory Auditor:** [`code/scripts/prepare_3dpw.py`](file:///home/dat/HMR/code/scripts/prepare_3dpw.py)
- **Verified Dataset Loader:** [`code/src/pipeline/dataset_3dpw.py`](file:///home/dat/HMR/code/src/pipeline/dataset_3dpw.py) successfully loads live RGB frames, camera matrices, and aligned ground-truth SMPL parameters across all 22,745 training instances.

---

## 3. Verification

All 14 unit tests have been verified from both the repository root and `code/`:
```bash
PYTHONPATH=. pytest code/tests/ -v
```
All tests pass cleanly.
