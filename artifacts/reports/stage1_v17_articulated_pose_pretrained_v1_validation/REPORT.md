# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_articulated_pose_kokoab_result_v1/stage1_v17_articulated_pose_pretrained_v1/best_model.pth` (epoch 53)
- Samples: 378
- Top-1: 96.83%
- Top-5: 99.47%
- Macro F1: 96.37%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 97.27% | 99.70% |
| [0.90, 1.01] | 33 | 93.94% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 1 | WRITE | WEEK |
| 1 | WHY | BIG |
| 1 | SAD | WANT |
| 1 | LIKE | MY |
| 1 | LIKE | HUNGRY |
| 1 | I | HUNGRY |
| 1 | GO | ANSWER |
| 1 | COME | HOME |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
