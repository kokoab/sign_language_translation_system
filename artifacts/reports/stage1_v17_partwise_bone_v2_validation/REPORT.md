# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_partwise_bone_kokoab_pull_v1/stage1_v17_partwise_bone_v2/best_model.pth` (epoch 71)
- Samples: 378
- Top-1: 96.30%
- Top-5: 99.47%
- Macro F1: 96.01%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 96.67% | 99.70% |
| [0.90, 1.01] | 33 | 93.94% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | SAD | WAIT |
| 2 | GOOD | THANKYOU |
| 1 | WHY | BIG |
| 1 | LIKE | WOMAN |
| 1 | HOSPITAL | SCHOOL |
| 1 | FIND | GOODBYE |
| 1 | COME | KNOW |
| 1 | BAD | HOT |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
