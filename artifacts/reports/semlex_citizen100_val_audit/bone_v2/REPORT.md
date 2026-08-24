# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_bone_kokoab_pull_v1/stage1_v17_bone_v2/best_model.pth`
- Clips/classes: 978/98
- Top-1: 86.40%
- Top-5: 96.42%
- Macro F1 over present classes: 84.02%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
