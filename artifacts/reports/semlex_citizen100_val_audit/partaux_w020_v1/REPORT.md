# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_partaux_w020_kokoab_result_v2/stage1_v17_partaux_w020_v1/best_model.pth`
- Clips/classes: 978/98
- Top-1: 88.04%
- Top-5: 96.42%
- Macro F1 over present classes: 85.85%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
