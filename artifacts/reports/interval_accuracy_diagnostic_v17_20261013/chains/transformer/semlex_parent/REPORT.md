# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `/Volumes/secret/SLT/SLT/artifacts/reports/interval_accuracy_diagnostic_v17_20261013/parents/transformer/best_model.pth`
- Clips/classes: 978/98
- Top-1: 85.58%
- Top-5: 95.60%
- Macro F1 over present classes: 82.55%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
