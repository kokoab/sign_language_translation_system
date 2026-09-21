# Matched connected-motion pretraining pilot

Two paired seeds; identical supervised CTC recipe and decoder. Apple Stage1 is frozen in all arms.

| Seed | Arm | Local WER | ASLLRP WER | NCSLGR WER | Isolated CTC exact |
|---|---|---:|---:|---:|---:|
| 17321 | baseline | 46.85% | 54.17% | 90.00% | 84.22% |
| 17321 | pretrained | 48.89% | 62.50% | 82.00% | 84.00% |
| 17322 | baseline | 49.07% | 62.50% | 100.00% | 82.96% |
| 17322 | pretrained | 44.44% | 45.83% | 96.00% | 84.00% |

**Decision: no consistent benefit established; do not promote or acquire more data on this evidence.** Paired WER/retention gates passed: 1/2.

Unlabeled signer overlap is unknown; source-video separation is not signer separation. Citizen test remained sealed. Frozen Stage1 isolated logits are unchanged; isolated CTC behavior is measured above.

## Behavior

| Seed/arm | Synthetic hold exact | Synthetic repeat exact | Real repeat recovery | Matched ASLLRP emission delay median |
|---|---:|---:|---:|---:|
| baseline_17321 | 6/20 | 4/20 | unavailable (0 examples) | 0.0 s |
| pretrained_17321 | 6/20 | 6/20 | unavailable (0 examples) | 0.0 s |
| baseline_17322 | 4/20 | 3/20 | unavailable (0 examples) | 0.0 s |
| pretrained_17322 | 7/20 | 2/20 | unavailable (0 examples) | 0.06673333333333334 s |

Detailed substitution/deletion/insertion counts, emitted gloss IDs, and duplicate outputs are in `behavior_*.json`. Synthetic hold/repeat probes never select checkpoints. Real held/repeat coverage is not established. Delay is relative to annotated sign end and conditional on correct matches; it is not hardware latency.

## Verification and interpretation

Both seeds preserved isolated CTC retention within the predefined tolerance, but connected improvements were inconsistent. Local adjacent output duplicates increased from58 to82 and86 to87; these counts alone do not establish held-sign errors. Synthetic repeat exact improved4→6/20 in one seed and worsened3→2/20 in the other.

The learned reconstruction also failed to beat the causal last-observation baseline on the same endpoint-observed subset:0.01795 vs0.01322 (seed17321), and0.01661 vs0.01322 (seed17322), lower is better. Noncausal interpolation scored0.00335/0.00326. This objective has not demonstrated better motion reconstruction than the simple causal control.

Post-run verification checked all four connected error totals against saved predictions and verified identical non-temporal initialization weights. A reporting-only repeat-count error was corrected: removing OTHER had made five nonadjacent equal-gloss pairs appear consecutive. Full references contain zero genuine adjacent-repeat pairs. Checkpoints, predictions, WER and selection are unchanged.

Next safe action: inspect the reconstruction objective and source-to-CTC transfer using the retained data before considering another run. No further acquisition or runtime promotion is justified by this pilot.

## Subsequent label-provenance caveat

The user reports potentially mixed phrase content within local folders. Code tracing confirms local targets came from folder names; existing audits checked media/features and signer grouping, not every clip’s signed transcript. Local training and validation labels therefore remain semantically unverified. The comparison is reproducible under those labels, but cannot settle pretraining usefulness on clean local phrase supervision. A clip-level visual transcript audit is required before further local-label-based conclusions. No mislabeled-clip count has yet been established.
