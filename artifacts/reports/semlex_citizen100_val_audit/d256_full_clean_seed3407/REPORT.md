# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced_seed3407/best_model.pth`
- Clips/classes: 978/98
- Top-1: 85.07%
- Top-5: 95.50%
- Macro F1 over present classes: 82.47%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
