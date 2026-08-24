# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_balanced/best_model.pth` (epoch 46)
- Samples: 378
- Top-1: 93.92%
- Top-5: 100.00%
- Macro F1: 93.46%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 94.55% | 100.00% |
| [0.90, 1.01] | 33 | 87.88% | 100.00% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 3 | WHO | YOU |
| 2 | WAIT | WANT |
| 2 | THANKYOU | GOOD |
| 2 | ANSWER | GO |
| 1 | YES | NEED |
| 1 | WHY | BIG |
| 1 | WHEN | SIGN |
| 1 | WATER | TALK |
| 1 | SICK | FATHER |
| 1 | SCHOOL | YOUR |
| 1 | LOVE | HOSPITAL |
| 1 | LIKE | HUNGRY |
| 1 | I | WE |
| 1 | I | SORRY |
| 1 | GOOD | THANKYOU |
| 1 | EXCITED | MAYBE |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
