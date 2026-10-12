# Mild-roll validation check — 2026-10-13

Fresh software-rotated landmark validation, not rotated camera images or isolated causal augmentation ablation. Same current admitted membership; no training/test access.

| Landmark rotation | Adapted landmark accuracy | Change from upright |
|---|---:|---:|
| -12° | 95.10% | -0.18 pp |
| -6° | 95.32% | +0.05 pp |
| +0° | 95.27% | +0.00 pp |
| +6° | 95.05% | -0.23 pp |
| +12° | 94.87% | -0.40 pp |

Upright adapted3770/3957 reproduces95.27%;worst sampled angle+12°3754/3957=94.87%,
16fewer correct,0.4043percentage-point decline. Base model on same inputs:
−12°75.87%,−6°76.50%,0°76.70%,+6°76.80%,+12°76.60%.
AdaptedCitizen-only364/365/364/365/366 of378 acrossangles; base362/365/366/364/363.
No measured guarantee for every angle between grid points, physicalcamera pitch/yaw,
extractor behavior on rotated pixels, or finalmultimodalruntime. Differenttrainingdata
andobjectives prevent causal attribution of before/after difference solelytoaugmentation.

Recipe recordsfullrollprobability0,mild12°. Trainer augment_v17 includes mirror.5,
isotropicXYscale.88–1.12,XYtranslation±.04,GaussianXYZnoiseSD.008 (batchtrigger.6),
nearest-neighbourtemporalwarpingrate.82–1.18 (batchtrigger.5),drop2–6nodes (batchtrigger.25).
These augmentations occur on landmark tensors, not RGB images. Basemodelprovenance only
namesaugment_v17_random_per_batch; do not claim each augmentation was newly introduced.
