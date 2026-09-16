# Project Guidelines & Agent Instructions

## PDF Reading Tool Preference
- **Default PDF Reader**: Always use **PyMuPDF4LLM** (`pymupdf4llm` or `read-pdf`) as the default tool to read, parse, and extract content from PDF files.
- When asked to read, analyze, summarize, or extract content from a PDF file:
  ```bash
  pymupdf4llm <path_to_pdf> [-o output.md] [--pages <range>] [--write-images]
  ```
  Or read text directly:
  ```bash
  pymupdf4llm <path_to_pdf> --stdout
  ```
- PyMuPDF4LLM reconstructs multi-column layout, headers, and markdown tables with near-instant speed and zero model download overhead.

## Review Deliverables Format (MANDATORY: PDF Presentation)
- **Always Present PDFs for Review**: Whenever presenting reports, research proposals, analysis summaries, or any document for the user to review, **you MUST always provide it as a PDF file, NOT just a Markdown file**.
- **Markdown-to-PDF Conversion Tool**: Use `render-pdf` (or `md-to-pdf`):
  ```bash
  render-pdf <path_to_markdown.md> [output.pdf]
  ```
  *(Or `md-to-pdf <path_to_markdown.md> [output.pdf]`)*
- **Engine Architecture**: Powered by Pandoc (`--mathjax --wrap=none`), MathJax 3 SVG vector typesetting, and Playwright Chromium. It guarantees:
  - Flawless LaTeX math (both inline `$ ... $` and block `$$ ... $$` with AMS extensions)
  - Native Mermaid.js flowchart and diagram rendering
  - Syntax highlighted code fences and GFM tables
  - Dynamic page numbering ("Page X of Y") and orphan-heading protection
  - *Note*: Do not use raw marked/npm `md-to-pdf` without MathJax, as it strips LaTeX backslashes and corrupts subscripts.
- Always ensure the generated PDF is saved/copied to the workspace and provide a clickable markdown link using the `file://` scheme to the `.pdf` file.

## Google Colab CLI Training Execution Rule (MANDATORY)
- **Execution Orchestration**: Whenever executing training or running GPU workloads in Google Colab, you MUST execute a dedicated orchestration bash script (e.g. `code/scripts/train_colab_cli.sh`) that invokes the Google Colab CLI (`colab`).
- **Session Reuse & Idempotency**:
  - Always check for an active Colab session before creating a new VM.
  - **Never kill or re-provision an active session on failure unless the VM is unresponsive**. Reusing sessions ensures datasets (like 3DPW ~5GB) and environment dependencies do not need to be uploaded and reinstalled repeatedly.
- **Dataset & Code Synchronization**:
  - Check remote dataset presence first (`/content/data/3dpw`). If already extracted and valid on the remote Colab instance, skip data upload.
  - Upload local code archives (`code/`) to `/content/code` and synchronize changes incrementally.
- **Explicit Step Logging**:
  - The orchestration bash script must log every step explicitly with clear section banners, timestamps, and return codes (e.g. `[STEP X/N]`) for effortless debugging.
  - Pipe and capture all command output to a dedicated local log file (e.g. `colab_training.log`) as well as stdout.

## Periodic 20-Minute Monitoring & Fail-Fast Rule (MANDATORY)
- **Periodic 20-Minute Progress Monitoring**:
  - Whenever `train_colab_cli.sh` (or any long-running GPU training workload) is active, the agent MUST monitor and report progress every **20 minutes** using the `schedule` tool (e.g., recurring cron `*/20 * * * *` or a 1200-second timer).
  - Each periodic check-in must audit:
    1. Session state and kernel activity (`colab status -s <name>`).
    2. Batch/epoch throughput, loss metrics, and GPU memory utilization.
    3. Real-time stderr logs for unexpected warnings or runtime bottlenecks.
- **Fail-Fast & Anti-Idle Policy**:
  - **Stop Fast on Failure**: If the training script crashes, exits with a non-zero code, produces NaNs/divergence, or hangs indefinitely without progress, immediately terminate the execution (`colab restart-kernel -s <name>` or task termination).
## Scientific Report & Document Writing Style (MANDATORY: Professional Review Format)
- **Reference Standard**: All reports, monographs, technical audits, and research notes MUST follow the rigorous, objective, and mathematically precise structure demonstrated in [`documents/md/reviews/diffnorm_mathematical_review.md`](documents/md/reviews/diffnorm_mathematical_review.md).
- **Core Principles**:
  1. **Executive Audit Table**: Begin with an executive audit table contrasting areas, verdicts (e.g., *Correct*, *Incomplete*, *Flawed*, *Overclaim*), and concise root causes.
  2. **Mathematical Precision & Consistent Conventions**: State explicit mathematical definitions, index bounds, dimensions (e.g. $\boldsymbol{\theta} \in \mathbb{R}^{24 \times 3}$), and normalization factors. Never mix normalized probability densities with unnormalized exponential Gaussian kernels.
  3. **Rigorous Differentiation & Lie Group Formulations**: Avoid hand-wavy infinitesimal shortcuts (e.g. do not equate Lie algebra generators at $\boldsymbol{\theta} = \mathbf{0}$ with finite $3 \times 3$ rotation Jacobians). Include full projection and normalization derivatives for unit-vector objectives ($\hat{\mathbf{N}} \cdot \mathbf{N}^*$).
  4. **Uncompromising Scientific Honesty & Nuance**:
     - Do not equate continuous surrogate kernel overlaps with physical triangle mesh penetration volumes ($\text{cm}^3$). State explicitly when a quantity is a surrogate proxy.
     - Never label small-sample sanity checks (e.g. 10 frames) or synthetic noise fallbacks as full benchmark evaluations (e.g. official 3DPW, RICH, or CAPE).
     - Replace hyperbole (*"eliminates Bas-relief"*, *"infinitely differentiable"*, *"strict spectral decomposition"*) with precise physical and algorithmic realities (*"reduces out-of-plane rotation ambiguity"*, *"piecewise smooth within active culling thresholds"*, *"explicit autograd gradient detachment"*).
  5. **Structured Actionable Layout**: Organize sections systematically into:
     - `## 1. Executive Audit`
     - `## 2. Theoretical & Mathematical Formulations (Correct Parts vs. Required Corrections)`
     - `## 3. Implementation & Algorithmic Reality`
     - `## 4. Empirical Benchmarks & Limitations`
     - `## 5. High-Priority Actionable Repairs / To-Do List`


