# Subscription-Only Research Architecture Report: Astra + Gemini
**Project:** DiffNorm-Contact HMR  
**Date:** September 17, 2026  
**Auditor / Engineer:** Antigravity AI  

---

## 1. Executive Audit & Architecture Transition

| Area | Previous API/Coordinator Design | New Subscription-Only Codex Architecture | Verdict |
|---|---|---|---|
| **Orchestration Runtime** | Custom Python script (`research.py`) + `openai-agents` SDK | **OpenAI Codex CLI / App (ChatGPT Plus)** | **Dramatically simplified** — zero custom agent code to maintain or debug. |
| **OpenAI Billing** | Required `OPENAI_API_KEY` with per-token API billing | **ChatGPT Plus subscription allowance** | **Zero API billing** — uses personal plan quota in Codex. |
| **Gemini Delegation** | Python wrapper calling `subprocess.run(["agy", ...])` | **Direct shell execution** (`agy -p "..."`) | **Zero overhead** — uses signed-in Google account (Google AI Pro). |
| **Colab Execution** | Python tool wrapper | **Direct CLI / Orchestration script** (`train_colab_cli.sh`) | **Native & idempotent** — reuses `diffnorm-hmr` session and cached data. |
| **Research Memory** | Dispersed between script and files | **Pure Git + Markdown** (`RESEARCH.md`, `notes/`) | **Auditable, persistent, and clean**. |

---

## 2. System Architecture

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

### 2.1 Role Division & Authority
* **GPT-6 Astra (Research Lead)** runs natively inside Codex.
  * Owns research questions, literature critique, falsifiable hypotheses, experiment design, controls/ablations, skeptical code review, and final scientific decisions.
  * Inspects every code change made by Gemini using `git status` and `git diff`.
* **Gemini 3.8 Flash (Research Worker)** is invoked directly from bash by Astra via `agy`.
  * Handles repository exploration, modular feature implementations, bug fixes, routine unit tests, and script preparation.
  * Claims made by Gemini are treated as unverified worker notes until audited.
* **Google Colab CLI (`colab`)** handles expensive remote GPU compute.

---

## 3. Directory Structure & Eliminated Artifacts

### 3.1 Files Completely Removed
The following files were eliminated to remove all API key dependencies and custom Python runtime overhead:
- ❌ `research.py` (at root and in starter pack)
- ❌ `run_research.sh` (at root and in `code/scripts/`)
- ❌ `.env` and `.env.example`
- ❌ `requirements.txt` / `openai-agents` dependency

### 3.2 Final Clean Structure
```text
/home/dat/HMR/
├── AGENTS.md                  # Unified system contract & project rules
├── RESEARCH.md                # Live high-level state (baseline, hypotheses, uncertainty)
├── .gitignore
│
├── notes/                     # Durable scientific memory
│   ├── literature.md          # Synthesized literature notes (DSINE, HMR2.0, etc.)
│   ├── hypotheses.md          # Falsifiable hypotheses (H001-H003)
│   └── decisions.md           # Decisions log (DEC-001 - DEC-004)
│
├── code/                      # Research implementation codebase
│   ├── src/                   # DiffNorm, Gaussian collision, and HMR models
│   ├── scripts/               # Colab orchestration & evaluation scripts
│   └── tests/                 # Unit and smoke tests
│
├── documents/                 # Documentation, review reports, and PDFs
│   ├── md/reviews/            # Markdown reports
│   └── pdf/                   # Publication-quality rendered PDFs
│
├── results/                   # Machine-readable JSON metrics (pilot experiments)
└── (astra_gemini_research_starter removed: fully integrated into root)
```

---

## 4. Operational Protocols

### 4.1 Delegating Tasks to Gemini via `agy`
When Astra needs code exploration or implementation, Astra runs:
```bash
agy --model gemini-3.8-flash-high -p '
Task: Implement Condition-F stage-gated normal loss router.
Scope: Inspect code/src/optimization/capacity_scheduler.py and update gating threshold.
Constraints: Keep baselines untouched; do not run full training; add a unit test.
Report: Files changed, tests executed, and concise summary.
'
```

### 4.2 Reviewing Worker Changes
Immediately after Gemini finishes, Astra executes:
```bash
git status
git diff
pytest code/tests/ -q
```
Astra verifies that mathematical conventions, Lie group rotation formulations, and loss derivatives remain rigorous.

### 4.3 Launching Colab Experiments
Per repository rules in [`AGENTS.md`](file:///home/dat/HMR/AGENTS.md), Astra executes the dedicated orchestration script:
```bash
bash code/scripts/train_colab_cli.sh
```
This guarantees:
1. Reusing the active `diffnorm-hmr` session without re-provisioning.
2. Preserving the uploaded 3DPW dataset (~5GB) and dependencies on the VM.
3. Synchronizing code incrementally.
4. Logging all execution steps to `colab_training.log`.

### 4.4 Ingesting Papers & Presenting Review Deliverables
* **Reading PDFs**: PyMuPDF4LLM is the default:
  ```bash
  pymupdf4llm papers/example.pdf --stdout
  ```
* **Review Deliverables**: Mandatory PDF presentation using `render-pdf`:
  ```bash
  render-pdf documents/md/reviews/report.md documents/pdf/report.pdf
  ```

---

## 5. Summary & Verification

- [x] Removed all Python coordinator scripts (`research.py`, `run_research.sh`, `.env`).
- [x] Zero API key dependencies (uses ChatGPT Plus in Codex and Google AI Pro in `agy`).
- [x] Consolidated all starter assets into the repository root and eliminated redundant `astra_gemini_research_starter/`.
- [x] Initialized [`RESEARCH.md`](file:///home/dat/HMR/RESEARCH.md) and [`notes/`](file:///home/dat/HMR/notes) with live DiffNorm-Contact HMR research state.
