# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_partwise_seed3407_kokoab_result_v2/stage1_v17_partwise_seed3407_v2/best_model.pth` (epoch 33)
- Samples: 378
- Top-1: 95.77%
- Top-5: 99.21%
- Macro F1: 95.38%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 96.36% | 99.39% |
| [0.90, 1.01] | 33 | 90.91% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 1 | YES | NEED |
| 1 | WHY | BIG |
| 1 | SAD | WAIT |
| 1 | MAKE | YEAR |
| 1 | LIKE | WOMAN |
| 1 | LIKE | HUNGRY |
| 1 | I | WHO |
| 1 | I | SORRY |
| 1 | GO | ANSWER |
| 1 | FIND | DRINK |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
