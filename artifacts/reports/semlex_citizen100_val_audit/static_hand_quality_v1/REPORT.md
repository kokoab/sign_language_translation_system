# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_static_hand_kokoab_result_v1/stage1_v17_static_hand_quality_v1/best_model.pth`
- Clips/classes: 978/98
- Top-1: 87.01%
- Top-5: 96.01%
- Macro F1 over present classes: 84.26%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
