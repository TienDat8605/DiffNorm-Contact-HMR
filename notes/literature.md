# Literature and Claim Boundaries

Corrected 17 September 2026; primary-source links checked. This is a working synthesis, not an exhaustive novelty search.

- [DSINE: Rethinking Inductive Biases for Surface Normal Estimation](https://github.com/baegwangbin/DSINE), Bae and Davison, CVPR 2024. Uses pixel-ray information and relative normal rotations. Uncertainty-capable checkpoint variants exist; do not assume every checkpoint outputs uncertainty. Predicted normals are observations with error, not ground truth.
- [Humans in 4D: Reconstructing and Tracking Humans with Transformers](https://shubham-goel.github.io/4dhumans/), ICCV 2023. Intended initialization source. Our wrapper requires verified official inference/preprocessing and camera conversion before baseline attribution.
- [GST](https://arxiv.org/abs/2409.04196): single-image body prediction with Gaussian splatting. 3DGS applied to human-body recovery is not itself new.
- [HumanSplatHMR](https://arxiv.org/abs/2605.02784): joint pose/avatar refinement precedent. Do not claim the first pose–Gaussian feedback loop.
- [UGS-Loc](https://arxiv.org/abs/2603.16538), CVPR 2026: uncertainty-aware Gaussian camera localization using pose sampling and geometric uncertainty. Narrow the proposed distinction to articulated pose versus adaptive avatar nuisance; full-method comparison remains required.
- [Estimating Nuisance Parameters in Inverse Problems](https://arxiv.org/abs/1206.6532): nuisance elimination is established mathematics, not our novel theorem.
- [Keep it SMPL](https://files.is.tue.mpg.de/black/papers/BogoECCV2016old.pdf), ECCV 2016: Gaussian overlap body-fitting collision precedent. Proxy overlap does not establish physical mesh intersection volume or continuous collision detection.

Candidate contribution: a computationally useful nuisance-aware articulated-pose diagnostic and step/capacity policy that beats matched conventional controls. Novelty and effectiveness both remain unverified. Existing paper PDFs remain under documents/pdf/papers/; do not duplicate licensed assets or treat archived project prose as a primary source.
