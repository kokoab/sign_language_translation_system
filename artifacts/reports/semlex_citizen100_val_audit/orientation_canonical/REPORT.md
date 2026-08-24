# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_orientation_canonical_pull_v1/stage1_v17_orientation_canonical_v1/best_model.pth`
- Clips/classes: 978/98
- Top-1: 86.71%
- Top-5: 94.79%
- Macro F1 over present classes: 84.50%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
