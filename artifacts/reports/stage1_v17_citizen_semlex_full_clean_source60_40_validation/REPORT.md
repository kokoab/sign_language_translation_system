# v17 Stage 1 val evaluation

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_full_clean_source60_40/best_model.pth` (epoch 56)
- Samples: 378
- Top-1: 95.24%
- Top-5: 99.47%
- Macro F1: 94.99%

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
| 1 | WHY | BIG |
| 1 | WATER | TALK |
| 1 | SICK | FATHER |
| 1 | NIGHT | WORK |
| 1 | NIGHT | DOCTOR |
| 1 | LOVE | HOSPITAL |
| 1 | LIKE | HUNGRY |
| 1 | HE | ASK |
| 1 | GOOD | THANKYOU |
| 1 | GO | ANSWER |
| 1 | FIND | DRINK |
| 1 | DOCTOR | NIGHT |
| 1 | COME | HOME |
| 1 | ANSWER | GO |
