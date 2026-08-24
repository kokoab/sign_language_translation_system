# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/generated/kaggle_stage1_attention_score_mix_kokoab_result_v1/stage1_v17_attention_score_mix_v1/best_model.pth` (epoch 54)
- Samples: 378
- Top-1: 96.03%
- Top-5: 99.47%
- Macro F1: 95.62%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 9 | 88.89% | 100.00% |
| [0.50, 0.75) | 6 | 100.00% | 100.00% |
| [0.75, 0.90) | 330 | 96.36% | 99.70% |
| [0.90, 1.01] | 33 | 93.94% | 96.97% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 2 | THANKYOU | GOOD |
| 1 | WHY | BIG |
| 1 | WAIT | WANT |
| 1 | THEY | HE |
| 1 | SAD | WANT |
| 1 | MAYBE | WANT |
| 1 | LIKE | WOMAN |
| 1 | LIKE | HUNGRY |
| 1 | HOT | THEY |
| 1 | HOSPITAL | SCHOOL |
| 1 | DOCTOR | NIGHT |
| 1 | COME | KNOW |
| 1 | BAD | THANKYOU |
| 1 | ANSWER | GO |
