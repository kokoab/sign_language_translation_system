# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_local_tiera10_balanced/best_model.pth` (epoch 54)
- Samples: 378
- Top-1: 95.77%
- Top-5: 99.74%
- Macro F1: 95.31%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 96.06% | 99.70% |
| [0.90, 1.01] | 33 | 93.94% | 100.00% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | ANSWER | GO |
| 1 | WHY | BIG |
| 1 | WAIT | WANT |
| 1 | SAD | WAIT |
| 1 | NIGHT | USE |
| 1 | NIGHT | DOCTOR |
| 1 | LOVE | HOSPITAL |
| 1 | LIKE | MY |
| 1 | LIKE | HUNGRY |
| 1 | GO | ASK |
| 1 | FIND | DRINK |
| 1 | COME | HOME |
| 1 | BAD | GOOD |
