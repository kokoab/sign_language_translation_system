# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_full_clean_labelsmooth005/best_model.pth` (epoch 85)
- Samples: 378
- Top-1: 95.50%
- Top-5: 100.00%
- Macro F1: 95.18%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 95.76% | 100.00% |
| [0.90, 1.01] | 33 | 93.94% | 100.00% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | GOOD | THANKYOU |
| 1 | YES | NEED |
| 1 | WHY | SICK |
| 1 | WAIT | WANT |
| 1 | SICK | FATHER |
| 1 | SCHOOL | TALK |
| 1 | SAD | WAIT |
| 1 | NIGHT | DOCTOR |
| 1 | LIKE | MY |
| 1 | LIKE | HUNGRY |
| 1 | FIND | DRINK |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
