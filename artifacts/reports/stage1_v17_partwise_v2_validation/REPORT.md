# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth` (epoch 100)
- Samples: 378
- Top-1: 96.83%
- Top-5: 99.21%
- Macro F1: 96.61%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 97.58% | 99.39% |
| [0.90, 1.01] | 33 | 90.91% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 1 | WHY | BIG |
| 1 | WAIT | WANT |
| 1 | SAD | WANT |
| 1 | MAKE | YEAR |
| 1 | LIKE | WOMAN |
| 1 | HOSPITAL | SCHOOL |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
