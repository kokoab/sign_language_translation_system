# Stage-2 landmark cascade and early-lock findings

The new preview is a separate experiment. It does not replace or modify the working
isolated classifier, motion-valley path, or accepted multimodal Stage-2 selector.

## What was trained

`train_stage_2_landmark_cascade_v17.py` distilled the accepted multimodal Stage-2
selector into a CTC preview that consumes only frozen Stage-1 landmark tokens and
landmark logits. It uses no hand RGB. The selected epoch-6 checkpoint has validation
edit counts `[18, 17, 47]` for ASLLRP contiguous phrases, local phrases, and held-out
ASLLRP segmented context respectively. By itself it is weaker than the multimodal
selector and must not replace it.

## Safe final-context cascade

On the existing 363 development-validation rows, a minimum greedy emission
probability of 0.978 routed 133 rows (36.6%) through the fast preview without worsening
any domain relative to the full selector:

| Domain | Preview coverage | Full edits | Cascade edits |
| --- | ---: | ---: | ---: |
| ASLLRP contiguous phrase | 3/12 (25.0%) | 9 | 9 |
| ASLLRP segmented context | 101/254 (39.8%) | 43 | 42 |
| Local phrases | 29/97 (29.9%) | 6 | 6 |

This threshold was selected on these same validation rows. It is a development gate,
not an independent accuracy claim.

The combined landmark encoder + CTC preview was exported as
`artifacts/coreml/Stage2LandmarkCascadePreviewV17FP32.mlpackage`. Core ML matched the
PyTorch decoded sequence on all 109 local/ASLLRP phrase validation rows. On this Mac,
the fixed graph measured 11.56 ms median and 18.13 ms p90 after warm-up. The package is
39.90 MiB.

## Early hard locking is not ready

Partial-window probes were made at 8, 16, 24, 28, and 32 model frames. Confidence
alone was unsafe: even a 0.999 threshold produced a false lock. Requiring a strict
hypothesis extension to persist across four consecutive probes removed observed false
locks, but locked only 11 tokens across 9/109 rows and never locked a complete phrase
before its final probe. That is too little coverage for the requested UX.

The evidence supports two display states:

- show the landmark result immediately as a **provisional** label;
- append a permanent gloss chip and speak it only after the full Stage-2 result or an
  independently validated lock gate confirms it.

This also matches the reference reel's visible behavior: its single-word label changes
during signing, while committed chips form separately and a FINISH sign triggers LLM
interpretation and voice. The reel does not reveal its model, thresholds, data, or
accuracy and is not evidence that it uses streaming CTC.

## Is more phrase data needed?

Not to repair the current live timebase or demonstrate a bounded prototype: the
elapsed-time Stage-2 replay remains exact on all five familiar test phrases at 15
extracted observations per second.

Yes for reliable early hard locking, novel combinations, and a research claim. The
current genuine multi-gloss training cache contains only 44 unique sequences, 50
unique directed adjacent-gloss pairs, and 46 of the 100 glosses appear in any
multi-gloss recording. Only six local phrase identities have substantial repeated
coverage; most ASLLRP continuous phrases have one example. The segmented 1,116-row
ASLLRP expansion improves isolated/contextual gloss evidence but does not provide
1,116 genuine inter-gloss transitions.

A targeted 20-30 native-signer phrase collection is therefore worthwhile as the next
pilot. It should maximize new adjacent-gloss pairs and include low-motion pronouns and
the observed confusion contexts (`YOU/I/HE/THEY/NEED/TELL/UNDERSTAND`) rather than add
more copies of only the six familiar phrases. The planned 10 train, 2 validation, and
1 sealed signer split with three genuine performances per phrase is enough for that
next experiment, but not enough to claim general 100-gloss continuous ASL coverage.
Simultaneous camera angles are correlated views of one performance, not independent
signer performances.

No sealed or test split was accessed in training, threshold selection, partial-window
probing, or Core ML parity.
