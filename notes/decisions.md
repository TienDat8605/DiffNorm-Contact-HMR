# Research Decisions Log: DiffNorm-Contact HMR

## DEC-001: Mandatory Google Colab Session Reuse
- **Date**: 2026-09-16
- **Context**: 3DPW dataset is ~5GB and re-uploading or re-extracting dataset and environment packages on every VM provision drains compute units and creates severe latency.
- **Decision**: Orchestration bash script `code/scripts/train_colab_cli.sh` must check for existing active session `diffnorm-hmr` before creating a new VM. Sessions must never be destroyed unless unresponsive.

## DEC-002: Deliverables Must Be Publication-Quality PDFs
- **Date**: 2026-09-16
- **Context**: Markdown documents often suffer from rendering discrepancies, especially with complex LaTeX mathematical formulas, AMS equations, and Mermaid charts.
- **Decision**: Use `render-pdf` (Chromium + MathJax 3 SVG engine) to compile all research deliverables, audit reports, and proposals into `.pdf` files.

## DEC-003: PyMuPDF4LLM as Preferred Document Parser
- **Date**: 2026-09-16
- **Context**: Standard PDF text extraction loses multi-column structure and corrupts LaTeX equations and tables.
- **Decision**: Use `pymupdf4llm` to extract rich Markdown representation from local research papers and reports.

## DEC-004: Adopting the Astra + Gemini Research Coordinator
- **Date**: 2026-09-17
- **Context**: Need a minimal, auditable pair-programming and research framework separating high-level scientific reasoning (GPT-6 Astra Lead) from high-volume implementation and local repository tasks (Gemini 3.8 Flash Worker).
- **Corrected 2026-09-17**: The contract assigns Codex scientific leadership and Gemini scoped assistance through `agy`. No `research.py` coordinator was found in the repository; deployment is not established.

## DEC-005: Narrow the contribution and require validity gates
- **Date**: 2026-09-17
- **Decision**: Primary direction is H004 nuisance-aware, fixed-shape, test-time pose refinement. Collision is deferred. E is an experimental heuristic, not a validated Schur solver; F is capacity gating, not keypoint-triggered normal gating.
- **Evidence**: Source and artifact audit in documents/md/current/progress_and_evidence_audit_2026_09_17.md.
- **Next action**: V001 evaluation validity, M001 mechanism checks, then P001 paired pilot; do not scale or train first.

## DEC-006: Preserve history, replace active claims
- **Date**: 2026-09-17
- **Decision**: Current reports live in documents/md/current and documents/pdf/current. Superseded documents are preserved in documents/archive with a relocation manifest. Root state and notes remain contract-required exceptions.
- **Publication scope**: User explicitly requested committing documentation plus existing research code/results. Snapshotting prototypes does not certify correctness.
