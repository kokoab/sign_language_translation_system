# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `/Volumes/secret/SLT/SLT/artifacts/reports/canonical_recognition_comparison_v17_20261007/chain_9683_floor361_mildroll/local_replay/best_promotion_gate_model.pth`
- Clips/classes: 978/98
- Top-1: 87.32%
- Top-5: 96.63%
- Macro F1 over present classes: 84.12%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
