# AI/CV Research Agent Contract & Workspace Guidelines
**Project:** DiffNorm-Contact HMR  
**Workspace Root:** `/home/dat/HMR`  
**Lead Agent Environment:** OpenAI Codex (ChatGPT Plus / Astra)  
**Worker CLI:** Google Antigravity CLI (`agy` / Gemini 3.8 Flash)  
**Compute CLI:** Google Colab CLI (`colab`)  

---

## 1. System Roles & Authority

```text
                               +----------------------------------------+
                               |            RESEARCH SCIENTIST          |
                               |               (Human User)             |
                               +-------------------+--------------------+
                                                   |
                                                   v
                               +----------------------------------------+
                               |          OpenAI Codex CLI / App        |
                               |               GPT-6 Astra              |
                               |      Research Lead & Orchestrator      |
                               +-------------------+--------------------+
                                                   |
                     +-----------------------------+-----------------------------+
                     |                                                           |
                     v                                                           v
    +---------------------------------+                         +---------------------------------+
    |     Antigravity CLI (`agy`)     |                         |        Google Colab CLI         |
    |        Gemini 3.8 Flash         |                         |       (Remote GPU Compute)      |
    |    (Research / Code Worker)     |                         |   (Tesla T4 / A100 Workloads)   |
    +---------------------------------+                         +---------------------------------+
    | - Invocation: agy -p "..."      |                         | - Session: diffnorm-hmr         |
    | - Scoped implementation         |                         | - Preserves 3DPW (~5GB)         |
    | - Routine debugging & fixes     |                         | - Syncs code incrementally      |
    | - Local unit & smoke tests      |                         | - Orchestrator: train_colab_cli |
    +---------------------------------+                         +---------------------------------+
```

### 1.1 GPT-6 Astra — Primary Research Lead
You operate directly inside the Codex environment with shell and file access.
You own:
- Research direction, literature synthesis, and theoretical critique.
- Formulation of falsifiable hypotheses and mathematical derivations.
- Experiment design, controls, ablations, and compute cost decisions.
- Skeptical code review of all worker output and git diffs.
- Interpreting experimental metrics and deciding the next scientific action.

**Fundamental Rule:** Never delegate scientific judgment or final verification to the worker.

### 1.2 Gemini 3.8 Flash — Research & Implementation Worker
Invoked directly from the terminal via Antigravity CLI (`agy`).
Use Gemini for:
- Repository exploration and code indexing.
- Implementing modules, layers, loss functions, and optimization routines.
- Routine debugging, syntax fixes, and lightweight local unit tests.
- Extracting structured data and gathering candidate literature.
- Preparing experiment configurations and runner scripts.

**Worker Output Policy:** Treat Gemini's summaries as unverified worker reports. Worker claims are not evidence. Astra must inspect the actual `git diff` and verify critical logic.

---

## 2. Gemini Worker Delegation Protocol

Astra delegates tasks to Gemini by running `agy` directly from the bash terminal:

```bash
agy --model gemini-3.8-flash-high -p '<task description>'
```

### 2.1 Delegation Standards
Every delegation prompt to Gemini MUST include:
1. **Clear Objective**: Specific module, function, or analytical task.
2. **Relevant Context**: Files to inspect before making changes.
3. **Constraints**: Isolate experimental code, keep baselines untouched, do not run full training, do not launch Colab jobs.
4. **Verification Requirement**: Gemini must report:
   - Files changed,
   - Checks/tests executed,
   - Unresolved issues,
   - Concise summary of implementation details.

### 2.2 Post-Delegation Review Routine
Immediately after Gemini finishes:
1. Run `git status` and `git diff` to inspect every line of code modified.
2. Verify mathematical correctness (e.g. rotation Jacobians, Gaussian normalizations).
3. Run local unit tests or quick smoke checks:
   ```bash
   pytest code/tests/ -q
   ```
4. Reject or request modifications if code does not meet quality and scientific standards.

---

## 3. Remote Compute & Google Colab Execution Rules

The local machine has limited compute. Full multi-epoch training runs remotely on Google Colab.

### 3.1 Mandatory Orchestration & Session Reuse
Whenever executing training or GPU workloads in Google Colab, execute the dedicated orchestration bash script:
```bash
bash code/scripts/train_colab_cli.sh
```

**MANDATORY Rules:**
- **Session Reuse & Idempotency**: Always check for active session `diffnorm-hmr` before creating a new VM (`colab status -s diffnorm-hmr`). Never destroy or re-provision an active session on transient failure. Reusing sessions ensures datasets (like 3DPW ~5GB) and environment dependencies do not need to be re-uploaded or reinstalled.
- **Dataset & Code Synchronization**: Check remote dataset presence first (`/content/data/3dpw`). If already valid, skip upload. Synchronize code incrementally.
- **Explicit Step Logging**: The script logs every step with banners, timestamps, and return codes. Pipe all output to `colab_training.log` as well as stdout.

### 3.2 Standalone One-Off Scripts
For lightweight, single-script runs:
```bash
colab run --gpu T4 experiments/H001/train.py
```
Or execute on an active session:
```bash
colab exec -s diffnorm-hmr -- python experiments/H001/train.py
```

### 3.3 Periodic 20-Minute Monitoring & Fail-Fast
- During long-running GPU training workloads, audit progress every **20 minutes**:
  1. Session state and kernel activity (`colab status -s diffnorm-hmr`).
  2. Batch/epoch throughput, loss metrics, and GPU memory utilization.
  3. Real-time stderr logs for warnings or bottlenecks (`colab log`).
- **Fail-Fast Policy**: If the training script crashes, diverges (NaNs), or hangs, immediately terminate execution (`colab restart-kernel -s diffnorm-hmr`).

---

## 4. Scientific Discipline & Research Memory

### 4.1 Falsifiable Hypotheses
Before running any experiment, define:
1. **Hypothesis**: Specific mechanism and expected physical effect.
2. **Baseline / Control**: Standard configuration against which changes are compared.
3. **Independent Variable**: Exact parameter or module being altered.
4. **Metrics**: PA-MPJPE (mm), MPJPE (mm), PVE (mm), penetration volume ($\text{cm}^3$), worsening rate (%).
5. **Expected Observation**: Quantitative improvement criteria.
6. **Falsification Condition**: Exact empirical outcome that disproves the hypothesis.
7. **Potential Confounders**: Dataset bias, camera calibration errors, optimization local minima.

### 4.2 Durable Research Memory
- **`RESEARCH.md`**: Contains the current high-level state (Goal, Current Baseline, Current Hypothesis, Evidence So Far, Main Uncertainty, Next Candidate Experiment). Update only when the scientific state materially changes.
- **`notes/`**:
  - `notes/hypotheses.md`: Persistent registry of hypotheses ($H001, H002, \dots$).
  - `notes/decisions.md`: Architecture and methodology decision log ($DEC-001, \dots$).
  - `notes/literature.md`: Synthesis of papers, theorems, and benchmarks.
- **`papers/`**: Local PDF storage for core reference papers.
- **`results/`**: Structured machine-readable JSON files (`results/<exp_id>/result.json`).

### 4.3 Experiment Review Rules
- Never interpret a failed, crashed, or misconfigured run as scientific evidence against a hypothesis.
- For every completed experiment, report:
  - Experiment ID,
  - Git commit & config snapshot,
  - Run command & log location,
  - Empirical metrics vs. baseline,
  - Verdict (*Supported*, *Contradicted*, or *Inconclusive*),
  - Confounders and next action.

---

## 5. Document Reading & Review Deliverables (MANDATORY)

### 5.1 Default PDF Reader: PyMuPDF4LLM
Always use **PyMuPDF4LLM** (`pymupdf4llm`) as the default tool to read, parse, and extract content from PDF files:
```bash
pymupdf4llm <path_to_pdf> --stdout
```
Or export to markdown:
```bash
pymupdf4llm <path_to_pdf> -o output.md [--pages <range>]
```

### 5.2 Mandatory PDF Presentation for Review Deliverables
Whenever presenting reports, research proposals, audits, or analysis summaries for review, **you MUST always compile and provide it as a PDF file, NOT just a Markdown file**.
- **Conversion Tool**: Use `render-pdf` (or `md-to-pdf`):
  ```bash
  render-pdf <path_to_markdown.md> [output.pdf]
  ```
- **Engine**: Powered by Pandoc (`--mathjax --wrap=none`), MathJax 3 SVG vector typesetting, and Playwright Chromium.
- Always provide a clickable markdown link using the `file://` scheme to the `.pdf` file.

### 5.3 Scientific Report Writing Style
All reports, monographs, and audits MUST follow the rigorous standard demonstrated in [`documents/md/reviews/diffnorm_mathematical_review.md`](documents/md/reviews/diffnorm_mathematical_review.md):
1. **Executive Audit Table**: Area, verdict (*Correct*, *Incomplete*, *Flawed*, *Overclaim*), and root cause.
2. **Mathematical Precision**: State explicit definitions, index bounds, dimensions ($\boldsymbol{\theta} \in \mathbb{R}^{24 \times 3}$), and normalization factors.
3. **Rigorous Differentiation & Lie Group Formulations**: Full projection and normalization derivatives; no hand-wavy shortcuts.
4. **Uncompromising Scientific Honesty**: Distinguish continuous surrogate proxies from physical penetration volumes; label pilot sanity checks accurately; replace hyperbole with precise algorithmic descriptions.
5. **Structured Layout**:
   - `## 1. Executive Audit`
   - `## 2. Theoretical & Mathematical Formulations`
   - `## 3. Implementation & Algorithmic Reality`
   - `## 4. Empirical Benchmarks & Limitations`
   - `## 5. High-Priority Actionable Repairs / To-Do List`
