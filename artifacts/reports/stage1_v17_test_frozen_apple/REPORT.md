# v17 Stage 1 test evaluation

- Checkpoint: `artifacts/models/stage1_v17_baseline/best_model.pth` (epoch 44)
- Samples: 1247
- Top-1: 87.57%
- Top-5: 98.64%
- Macro F1: 87.39%

## Accuracy by archived hand-active coverage

| Coverage | Clips | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| [0.00, 0.50) | 1 | 100.00% | 100.00% |
| [0.50, 0.75) | 42 | 92.86% | 92.86% |
| [0.75, 0.90) | 1038 | 87.67% | 99.04% |
| [0.90, 1.01] | 166 | 85.54% | 97.59% |

## Most frequent confusions

| Count | True | Predicted |
| ---: | --- | --- |
| 7 | GOOD | THANKYOU |
| 7 | ANSWER | GO |
| 6 | GO | ANSWER |
| 5 | MY | PLEASE |
| 4 | UNDERSTAND | THINK |
| 3 | THEY | HE |
| 3 | TALK | WATER |
| 3 | NEED | WHERE |
| 2 | YOU | WHO |
| 2 | YEAR | MAKE |
| 2 | WOMAN | TALK |
| 2 | WHO | YOU |
| 2 | WHERE | ASK |
| 2 | WAIT | WANT |
| 2 | TIRED | HAVE |
| 2 | PLEASE | MY |
| 2 | MOTHER | WATER |
| 2 | HOME | EAT |
| 2 | FIND | GOODBYE |
| 2 | FEEL | HAPPY |
