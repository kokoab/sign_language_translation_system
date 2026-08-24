# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_baseline/best_model.pth` (epoch 44)
- Samples: 378
- Top-1: 93.12%
- Top-5: 99.47%
- Macro F1: 92.53%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 77.78% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 93.33% | 99.70% |
| [0.90, 1.01] | 33 | 93.94% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | I | WE |
| 2 | ANSWER | GO |
| 1 | WHY | BIG |
| 1 | WHO | YOU |
| 1 | WHO | THEY |
| 1 | WHO | ASK |
| 1 | WAIT | WANT |
| 1 | WAIT | SAD |
| 1 | THANKYOU | GOOD |
| 1 | SICK | FATHER |
| 1 | SCHOOL | WOMAN |
| 1 | SAD | WANT |
| 1 | MY | PLEASE |
| 1 | MOTHER | HEAR |
| 1 | LIKE | WOMAN |
| 1 | LIKE | HUNGRY |
| 1 | LEARN | SICK |
| 1 | HOT | THEY |
| 1 | HEAR | HELLO |
| 1 | GOOD | THANKYOU |
