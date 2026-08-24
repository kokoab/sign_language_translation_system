# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/models/stage1_v17_baseline/best_model.pth`
- Clips/classes: 978/98
- Top-1: 73.72%
- Top-5: 88.75%
- Macro F1 over present classes: 70.56%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
