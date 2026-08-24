# SemLex val secondary-domain evaluation

**Status:** evaluation only; this split is never training data.

- Checkpoint: `artifacts/models/stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1/best_promotion_gate_model.pth`
- Clips/classes: 978/98
- Top-1: 87.93%
- Top-5: 96.42%
- Macro F1 over present classes: 84.82%

SemLex validation mostly reuses SemLex train signer identities, so this is a
cross-domain clip diagnostic rather than an unseen-signer production test.
