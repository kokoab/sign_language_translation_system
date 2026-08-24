# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth`
- Clips/classes: 978/98
- Top-1: 85.89%
- Top-5: 96.11%
- Macro F1 over present classes: 82.60%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
