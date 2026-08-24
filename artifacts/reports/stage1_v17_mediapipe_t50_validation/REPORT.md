# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_mediapipe_t50/best_model.pth` (epoch 110)
- Samples: 378
- Top-1: 89.95%
- Top-5: 97.35%
- Macro F1: 89.81%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 3 | 66.67% | 100.00% |
| [0.50, 0.75) | 12 | 91.67% | 100.00% |
| [0.75, 0.90) | 344 | 90.70% | 98.26% |
| [0.90, 1.01] | 19 | 78.95% | 78.95% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | GOOD | THANKYOU |
| 2 | ANSWER | GO |
| 1 | YES | NEED |
| 1 | YEAR | MAKE |
| 1 | WHY | WANT |
| 1 | WANT | MAYBE |
| 1 | WAIT | WANT |
| 1 | TOMORROW | HOW |
| 1 | TOMORROW | HELP |
| 1 | THANKYOU | GOOD |
| 1 | SLEEP | HOSPITAL |
| 1 | SICK | FATHER |
| 1 | SCHOOL | THANKYOU |
| 1 | SAD | WANT |
| 1 | SAD | WAIT |
| 1 | PLEASE | MY |
| 1 | MAYBE | STOP |
| 1 | MAYBE | GIVE |
| 1 | LOVE | THEY |
| 1 | LIKE | MY |
