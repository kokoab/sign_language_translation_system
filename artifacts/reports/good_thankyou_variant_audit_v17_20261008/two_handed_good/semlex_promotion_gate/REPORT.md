# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `/Volumes/secret/SLT/SLT/artifacts/reports/good_thankyou_variant_audit_v17_20261008/two_handed_good/local_replay/best_promotion_gate_model.pth`
- Clips/classes: 978/98
- Top-1: 87.83%
- Top-5: 96.93%
- Macro F1 over present classes: 84.86%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
