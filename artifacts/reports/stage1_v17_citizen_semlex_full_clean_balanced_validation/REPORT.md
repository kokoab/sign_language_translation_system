# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth` (epoch 93)
- Samples: 378
- Top-1: 95.77%
- Top-5: 100.00%
- Macro F1: 95.51%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 100.00% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 96.06% | 100.00% |
| [0.90, 1.01] | 33 | 90.91% | 100.00% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 1 | YES | NEED |
| 1 | WHY | SICK |
| 1 | WAIT | WANT |
| 1 | SCHOOL | OUR |
| 1 | NIGHT | DOCTOR |
| 1 | LOVE | HOSPITAL |
| 1 | LIKE | MY |
| 1 | LIKE | HUNGRY |
| 1 | I | HUNGRY |
| 1 | FIND | DRINK |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
