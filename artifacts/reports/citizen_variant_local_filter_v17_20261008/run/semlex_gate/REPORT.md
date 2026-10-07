# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `/Volumes/secret/SLT/SLT/artifacts/reports/citizen_variant_local_filter_v17_20261008/run/local_replay/best_promotion_gate_model.pth`
- Clips/classes: 978/98
- Top-1: 87.53%
- Top-5: 96.42%
- Macro F1 over present classes: 84.55%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
