# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_masked_pose_kokoab_result_v1/stage1_v17_masked_pose_finetune_v1/best_model.pth` (epoch 50)
- Samples: 378
- Top-1: 95.24%
- Top-5: 99.47%
- Macro F1: 94.79%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 77.78% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 96.06% | 99.70% |
| [0.90, 1.01] | 33 | 90.91% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | SAD | WAIT |
| 2 | FIND | DRINK |
| 1 | WHY | BIG |
| 1 | WAIT | WANT |
| 1 | SCHOOL | TALK |
| 1 | MAKE | YEAR |
| 1 | LIKE | HUNGRY |
| 1 | I | SORRY |
| 1 | FIND | GOODBYE |
| 1 | FEEL | PLEASE |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ASK | NEED |
| 1 | ANSWER | GO |
