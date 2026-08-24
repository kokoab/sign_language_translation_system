# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_static_hand_kokoab_result_v1/stage1_v17_static_hand_low_motion_v1/best_model.pth` (epoch 117)
- Samples: 378
- Top-1: 97.09%
- Top-5: 98.94%
- Macro F1: 96.86%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 97.58% | 99.09% |
| [0.90, 1.01] | 33 | 93.94% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 1 | WHY | BIG |
| 1 | THANKYOU | GOOD |
| 1 | SAD | WAIT |
| 1 | LIKE | HUNGRY |
| 1 | HOSPITAL | SCHOOL |
| 1 | HE | WHERE |
| 1 | FIND | GOODBYE |
| 1 | FIND | DRINK |
| 1 | COME | KNOW |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
