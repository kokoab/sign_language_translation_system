# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1/best_promotion_gate_model.pth` (epoch 21)
- Samples: 378
- Top-1: 95.50%
- Top-5: 98.68%
- Macro F1: 95.15%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 88.89% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 95.76% | 99.39% |
| [0.90, 1.01] | 33 | 93.94% | 93.94% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 2 | SAD | WAIT |
| 1 | WHY | WHAT |
| 1 | TIRED | HAVE |
| 1 | THINK | LISTEN |
| 1 | THEY | HE |
| 1 | LIKE | HUNGRY |
| 1 | HOT | TAKE |
| 1 | HOSPITAL | USE |
| 1 | HEAR | FATHER |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
| 1 | CHILD | GOODBYE |
| 1 | BAD | GOOD |
| 1 | ANSWER | GO |
