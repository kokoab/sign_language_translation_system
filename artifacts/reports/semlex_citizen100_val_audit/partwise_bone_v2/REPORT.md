# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/generated/kaggle_stage1_partwise_bone_kokoab_pull_v1/stage1_v17_partwise_bone_v2/best_model.pth`
- Clips/classes: 978/98
- Top-1: 87.83%
- Top-5: 97.03%
- Macro F1 over present classes: 85.18%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
