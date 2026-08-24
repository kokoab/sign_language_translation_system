# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_articulated_pose_kokoab_result_v1/stage1_v17_articulated_pose_pretrained_v1/best_model.pth`
- Clips/classes: 978/98
- Top-1: 86.09%
- Top-5: 95.81%
- Macro F1 over present classes: 83.41%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
