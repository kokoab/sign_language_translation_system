# Current-lineage component validation — 2026-10-13

Fresh PyTorch inference on identical admitted validation inputs for every component. No training, protected-test access or paper changes. Accuracy is total correct divided by total evaluated.

| Component | Citizen | SemLex | Local | Overall |
|---|---:|---:|---:|---:|
| landmark_only | 96.30% | 87.51% | 97.94% | 95.27% (3770/3957) |
| hand_only | 80.69% | 72.61% | 63.29% | 67.20% (2659/3957) |
| fusion | 96.83% | 88.04% | 97.87% | 95.40% (3775/3957) |
| phrase | 96.83% | 87.62% | 97.75% | 95.22% (3768/3957) |
| interval | 96.03% | 88.25% | 97.60% | 95.20% (3767/3957) |
| current_primary | 96.30% | 88.77% | 97.83% | 95.50% (3779/3957) |
| transition_adapted | 96.56% | 88.88% | 97.94% | 95.63% (3784/3957) |

Standalone landmark/hand states exactly match pinned fusion parents. Current primary and adapted totals independently reproduce the prior full validation verification. Results are component-level; the adapted output alone is not the conditional runtime accuracy. PyTorch only here, not new phone or FP16 evaluation.

Current lineage is rooted in the 96.83% part-wise/global base, not the historical full-roll checkpoint. Historical 95.50/80.69/96.30 numbers were Citizen-only and cannot be compared directly with the overall column.

This review uses 953 admitted SemLex examples rather than the older fusion record’s 978. All components were rescored on the same reduced membership.
