# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced_d384/best_model.pth` (epoch 51)
- Samples: 378
- Top-1: 95.24%
- Top-5: 99.74%
- Macro F1: 94.90%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 95.45% | 99.70% |
| [0.90, 1.01] | 33 | 93.94% | 100.00% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | LIKE | HUNGRY |
| 1 | WHY | BIG |
| 1 | TALK | WATER |
| 1 | SAD | WANT |
| 1 | NIGHT | DOCTOR |
| 1 | LOVE | HOSPITAL |
| 1 | HEAR | GOODBYE |
| 1 | GO | ASK |
| 1 | FIND | GOODBYE |
| 1 | FIND | DRINK |
| 1 | FEEL | MY |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
