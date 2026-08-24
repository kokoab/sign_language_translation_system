# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_partmix_kokoab_pull_v3/stage1_v17_partmix_p50_v2/best_model.pth`
- Clips/classes: 978/98
- Top-1: 85.58%
- Top-5: 95.30%
- Macro F1 over present classes: 83.45%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
