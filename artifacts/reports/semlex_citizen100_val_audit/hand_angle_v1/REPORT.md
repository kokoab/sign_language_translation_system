# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_hand_angle_kokoab_result_v1/stage1_v17_hand_angle_v1/best_model.pth`
- Clips/classes: 978/98
- Top-1: 85.69%
- Top-5: 95.40%
- Macro F1 over present classes: 82.96%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
