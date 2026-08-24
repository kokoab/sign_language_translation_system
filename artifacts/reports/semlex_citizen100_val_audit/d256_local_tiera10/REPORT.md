# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/models/stage1_v17_citizen_semlex_local_tiera10_balanced/best_model.pth`
- Clips/classes: 978/98
- Top-1: 86.50%
- Top-5: 96.22%
- Macro F1 over present classes: 84.25%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
