# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_augmented/best_model.pth` (epoch 37)
- Samples: 378
- Top-1: 92.86%
- Top-5: 100.00%
- Macro F1: 92.25%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 66.67% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 93.94% | 100.00% |
| [0.90, 1.01] | 33 | 87.88% | 100.00% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | I | SORRY |
| 2 | ANSWER | GO |
| 1 | YES | NEED |
| 1 | WHY | BIG |
| 1 | WHO | YOU |
| 1 | WHO | NEED |
| 1 | WE | NIGHT |
| 1 | WATER | TALK |
| 1 | WAIT | WANT |
| 1 | STOP | MORNING |
| 1 | SICK | FATHER |
| 1 | SCHOOL | OUR |
| 1 | SAD | WANT |
| 1 | LOVE | HOSPITAL |
| 1 | LIKE | MY |
| 1 | LIKE | HUNGRY |
| 1 | GOOD | THANKYOU |
| 1 | FIND | GOODBYE |
| 1 | FIND | DRINK |
