# Matched Reel comparison — completed 2026-09-22

Ran the current Reel landmark proposal and full visual verifier on the same observations
from 12 approved, previously-used ASLLRP validation videos. Raw-video hashes matched the
combined manifest. No training or protected test access. The comparison uses annotated
sign cores, those same cores with 100/250 ms added on each side, and separate annotated gaps.

## Corrected matched results

| Input window | Proposal top-1 correct | Verifier top-1 correct | Correct conditional commits | Wrong conditional commits |
| --- | ---: | ---: | ---: | ---: |
| Annotated core | 9/17 | 11/17 | 5/17 | 1/17 |
| Core +100 ms each side | 8/17 | 11/17 | 9/17 | 0/17 |
| Core +250 ms each side | 7/17 | 7/17 | 5/17 | 1/17 |

Every arm contains the exact same 17 identities. Existing wrist-motion trimming remains
active in every arm. These results do not support removing all context: modest context
helped commitment on this sample, while more context degraded recognition. A surrounding
crop can include a neighboring sign; a changed label is not automatically a stream insertion.

## Direct transition failure

Three non-overlapping gaps between known sign annotations had at least four observations,
after 20 ms edge guards; no OTHER/other annotation overlaps were allowed. One gap
(asllrp:842935.mp4:span00, 0.5205–0.8475 s) produced:

- Proposal: WORK, score 0.5943.
- Visual verifier: WORK, score 0.5173.
- Both accepted; the checked commit conditions passed.

The other two gaps failed those commit conditions. This is direct evidence that a checked
non-sign interval can fool both models. Scores are uncalibrated. Gap truth comes from
existing reviewed annotations, not a new human annotation of this replay.

## Correct evidence can also be blocked

Six core windows had the correct verifier top-1 but failed conditional commitment.
Examples: LIKE verifier 0.601 with a low-score GOODBYE proposal; NOW verifier 0.696 with
a low-score/low-margin NOW proposal; TIME verifier 0.639 with a low-score TIME proposal.
Conversely one FRIEND core had FRIEND proposal 0.786 but ANSWER verifier 0.535 and passed
checked commit conditions. Thus verifier disagreement is not automatically a correction.

Conditional commitment starts a fresh VerifiedCommitLock and checks proposal acceptance,
verifier acceptance, agreement and commit score. It bypasses candidate activation and
proposal temporal stability/history. It is not a replay of actual live emissions or WER.
All 17 core windows met a within-window wrist-start test; this sample cannot answer the
stationary-wrist question.

## Execution correction and limits

The initial diagnostic scheduled each observation at current_time+1/20. Reel instead uses
max(previous_deadline+1/20,current_time). This mattered on 30 fps video. Fixed the diagnostic,
added a 30-to-20 Hz scheduling self-check, and reran all clips. Initial results are preserved
under initial_sampling/ and are superseded. Corrected results include all 17 cores rather
than 11. A paired identity assertion passed across all three arms.

Five recent app recordings could not be decoded (missing MP4 moov atom); candidate sessions
had no video. These results therefore do not replay the user's exact HELLO→MY or I GO event.
They are a small existing-development diagnostic, not a new generalization benchmark.
No inference weights, confidence thresholds, or segmentation rules were changed.

## Next action supported by this comparison

Keep the current models as reference. Test transition rejection on reviewed hard negatives
like this WORK gap, together with true WORK/GOODBYE and low-motion positives. Include
window-duration and proposal/verifier disagreement cases; blanket threshold relaxation
would trade failures without addressing the demonstrated false-positive mechanism.
App recording cleanup was repaired separately (89 focused tests passed) so actual user mistakes can join this
same-input analysis rather than being inferred from output-only history.
