# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_masked_pose_kokoab_result_v1/stage1_v17_masked_pose_finetune_v1/best_model.pth`
- Clips/classes: 978/98
- Top-1: 83.74%
- Top-5: 95.19%
- Macro F1 over present classes: 80.64%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
