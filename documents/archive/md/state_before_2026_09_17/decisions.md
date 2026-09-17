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
- **Decision**: Deployed `research.py` coordinator with integrated Colab orchestration and safety sandboxing.
