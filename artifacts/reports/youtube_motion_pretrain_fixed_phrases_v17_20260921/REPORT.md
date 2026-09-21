# Matched connected-motion pretraining pilot

Two paired seeds; identical supervised CTC recipe and decoder. Apple Stage1 is frozen in all arms. Reused exact initial heads from `artifacts/models/youtube_motion_pretrain_v17_20260921`; motion pretraining was not rerun. Phrase root: `data/local/stage2_v17_grounded_phrases_fixed_20260921`.

| Seed | Arm | Local WER | ASLLRP WER | NCSLGR WER | Isolated CTC exact |
|---|---|---:|---:|---:|---:|
| 17321 | baseline | 48.52% | 54.17% | 92.00% | 83.92% |
| 17321 | pretrained | 43.15% | 66.67% | 86.00% | 84.07% |
| 17322 | baseline | 50.19% | 62.50% | 98.00% | 83.41% |
| 17322 | pretrained | 44.44% | 54.17% | 88.00% | 83.26% |

Paired WER/retention gates passed: 1/2. No automatic promotion.

Unlabeled signer overlap is unknown; source-video separation is not signer separation. Citizen test remained sealed. Frozen Stage1 isolated logits are unchanged; isolated CTC behavior is measured above.

## Behavior

| Seed/arm | Synthetic hold exact | Synthetic repeat exact | Real repeat recovery | Matched ASLLRP emission delay median |
|---|---:|---:|---:|---:|
| baseline_17321 | 10/20 | 7/20 | 0/0 | 0.0 s |
| pretrained_17321 | 6/20 | 6/20 | 0/0 | -0.03336666666666667 s |
| baseline_17322 | 6/20 | 6/20 | 0/0 | 0.0 s |
| pretrained_17322 | 6/20 | 5/20 | 0/0 | 0.03336666666666667 s |

Detailed substitution/deletion/insertion counts, emitted gloss IDs, and duplicate outputs are in `behavior_*.json`. Synthetic hold/repeat probes never select checkpoints. Real held/repeat coverage is not established. Delay is relative to annotated sign end and conditional on correct matches; it is not hardware latency.
