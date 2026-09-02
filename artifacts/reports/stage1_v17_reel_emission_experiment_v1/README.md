# Stage-1 Reel learned-emission experiment

Date: 2026-09-03

## Question

Can Stage 1 support faster, pause-free Reel interaction if it learns whether a rolling
window contains a complete sign, instead of relying only on hand-down/neutral boundaries
or treating every window as a phrase?

## Separate model

The accepted Reel model was not overwritten. The experiment freezes its 100-gloss
landmark classifier and adds a small temporal `__NO_EMIT__` head. The head sees four
ordered temporal quarters, the start-to-end feature change, and the unchanged Stage-1
logits. It is trained on complete signs, incomplete prefixes, and between-sign
transitions. The original 100 gloss logits are bit-for-bit unchanged (`max_abs = 0.0`).

Training used 8,671 development samples:

- Citizen and SemLex isolated train clips for vocabulary-wide complete/prefix replay.
- All permitted local phrase training caches for complete signs, prefixes, and
  transitions.
- Manually aligned ASLLRP continuous training phrases from CORY, RACHEL, and
  BENJAMIN_JAMES_BAHAN.

Validation used 3,452 samples. ASLLRP validation is signer-disjoint JONATHAN. No test or
`external_evaluation_reserved` material was accessed.

The first linear-head trial was rejected: at 95.5% complete-sign acceptance it caught
only 13.7% of incomplete windows. The selected temporal-head checkpoint at its
validation-selected 0.77 threshold accepts 97.68% of complete windows and catches
83.56% of incomplete/transition windows overall. Its held-out ASLLRP result is weaker:
20/24 complete windows accepted and 16/36 incomplete windows caught.

The live preset uses a more recall-oriented 0.95 threshold. On the same validation set
it accepts 99.02% of complete windows and catches 70.60% of incomplete windows overall;
for ASLLRP it accepts 21/24 complete windows but catches only 7/36 incomplete windows.

## Core ML

- Package: `artifacts/coreml/Stage1ReelEmissionV17FP16.mlpackage`
- Size: 13.48 MiB
- Mac latency: 11.93 ms median / 14.04 ms p90 after warm-up
- Citizen validation parity: 378 clips, zero top-1 mismatches and zero emission-decision
  mismatches versus PyTorch
- Maximum FP16 logit difference: 0.01216

These are Mac host measurements, not iPhone performance evidence.

## End-to-end local phrase replay

The selected live preset starts probing at 0.32 seconds, probes every 0.08 seconds, and
requires two agreeing proposals. Stage 2, speech, and the lip specialist were disabled.

| Expected | Output |
| --- | --- |
| GOOD MORNING | GOOD MORNING |
| HELLO HOW YOU | HELLO HOW YOU |
| MY NAME | MY NAME |
| PLEASE HELP I | I |
| THANKYOU FRIEND | GOOD |

- Exact: 3/5
- Token edits: 4/12
- Median committed candidate duration: 0.93 seconds
- Learned `NO_EMIT` rejections: 46/102 probes

This ties the accepted 0.50/0.12 Reel preset on exact phrases and token edits, but is
slower at commit time (0.93 versus 0.67 seconds median). The two remaining phrase errors
are classifier/verifier errors, not boundary errors, so the emission head cannot fix
them.

## End-to-end ASLLRP replay

The same 12 held-out-signer development clips used for the prior Reel check were replayed.

| Metric | Accepted Reel | Learned-emission Reel |
| --- | ---: | ---: |
| Exact sequences | 0/12 | 0/12 |
| Token edits / 24 | 20 | 17 |
| Emitted glosses | 5 | 8 |
| Median committed candidate | 0.67 s | 0.57 s |
| Median proposal latency | 20.30 ms | 23.18 ms |

The learned model emits three more glosses and reduces edits by three, but still never
recovers both signs from a fast two-sign clip. Most clips provide only two to six rolling
probes, and one wrong `WRITE` remains. This is an improvement in under-emission, not a
solution for natural continuous signing.

## Decision

Do not replace the accepted Reel path with this experiment. Keep it available as a
separate prototype:

```bash
venv/bin/python scripts/live_reel_emission_stage1_v17.py --camera 0
```

The result supports Stage-1 fine-tuning for signer/domain classification and a learned
boundary signal as an auxiliary component. It does **not** support the claim that an
isolated Stage-1 recognizer plus a gate can learn unrestricted conversational sequences.
A genuinely natural, fast, unmodified signing stream still needs a temporal sequence
model trained with continuous signing and explicit blank/boundary supervision; it need
not reuse the current slow whole-phrase Stage-2 live policy.

## Post-experiment live optimization

The first webcam run exposed a separate implementation bottleneck. Landmark proposals
took 37.11 ms median / 48.95 ms p90, but 26 hand-image verification calls took 745.07
ms median / 979.22 ms p90 and competed with live landmark extraction. The 103-second
session dropped 1,604 stale camera frames.

The learned-emission entry point now commits its two-hit stable landmark proposal
directly and does not load or run the hand-image verifier. The original Reel entry
point is unchanged. `--full-visual-verifier` restores the old experiment for
comparison. On an 18-second saved-video smoke, the fast path processed all 270 source
frames, made 60 proposals at 37.84 ms median, and made no additional visual-verifier
calls. This removes the measured 0.65--1.09 second per-commit operation; a new webcam
session is still required to measure sustained live FPS and the accuracy tradeoff.
