# DiffNorm-Contact HMR

Research prototype investigating **nuisance-aware test-time human pose refinement with surface-bound 3D Gaussians**. No validated HMR benchmark improvement or collision-free guarantee is currently established.

## Start here

- [Current proposal and experiment roadmap (PDF)](documents/pdf/current/research_proposal_and_roadmap_2026_09_17.pdf)
- [Progress and evidence audit (PDF)](documents/pdf/current/progress_and_evidence_audit_2026_09_17.pdf)
- [Research state](RESEARCH.md) and [document index](documents/md/INDEX.md)
- [Hypotheses](notes/hypotheses.md), [decisions](notes/decisions.md), [literature](notes/literature.md)

## Current status

The six-condition runner, mesh/Gaussian renderers, gradient-filtering heuristic and capacity scheduler exist. The recorded 24-example pilot covers 20 sequence names, uses an unsafe evaluation path, and does not establish an authentic HMR2/SMPL benchmark. The next step is **evaluation validity**, not multi-epoch training. See the audit for exact blockers.

Local CPU verification on 17 September 2026: **33 passed, 2 skipped, 12 warnings**. Unit tests demonstrate limited software properties, not research validity.

## Layout

- `code/`: experimental implementations, scripts and tests.
- `documents/md/current/`: authoritative proposal and audit sources.
- `documents/pdf/current/`: matching review PDFs.
- `documents/archive/md/`, `documents/archive/pdf/`: historical, superseded material; do not reuse its benchmark claims.
- `documents/pdf/papers/`: existing external reference PDFs (retained in place).
- `results/`: preserved historical artifacts; new runs use `results/<exp_id>/result.json`.
- `RESEARCH.md`, `AGENTS.md`, `notes/`: deliberate operational Markdown exceptions.

## Safe local checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=code pytest code/tests/ -q
python code/scripts/audit_pilot_artifact.py --out results/DOC-AUDIT-20260917/result.json
```

Do not use the existing pilot or training script to claim official benchmark performance until the validity gates in the roadmap pass. GPU execution follows `AGENTS.md`; no training is authorized by this documentation update.

## Roles and attribution

Codex is the research lead and owns scientific judgment and final verification. Gemini Antigravity is a scoped assistant; its reports are unverified until checked. No published paper or arXiv identifier is claimed for this project. Historical citation placeholders and headline scores have been removed from this overview. No repository LICENSE file was found; the previous MIT badge did not establish a license grant. Body models and datasets have separate licensing requirements.
