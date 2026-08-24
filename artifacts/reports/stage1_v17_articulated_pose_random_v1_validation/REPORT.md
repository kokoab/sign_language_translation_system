# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_articulated_pose_kokoab_result_v1/stage1_v17_articulated_pose_random_v1/best_model.pth` (epoch 111)
- Samples: 378
- Top-1: 96.83%
- Top-5: 99.21%
- Macro F1: 96.56%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 97.27% | 99.39% |
| [0.90, 1.01] | 33 | 93.94% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 1 | WHY | BIG |
| 1 | TALK | WATER |
| 1 | STOP | MORNING |
| 1 | SCHOOL | TALK |
| 1 | SAD | WANT |
| 1 | SAD | WAIT |
| 1 | NIGHT | SCHOOL |
| 1 | FIND | GOODBYE |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
