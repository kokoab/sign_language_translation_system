# Reel temporal decision probe — 2026-09-22

**Result: failed; no promotion or live changes.**

Frozen current Reel proposal and visual verifier were replayed on 56 approved ASLLRP
source videos at the actual 20 Hz scheduling policy. The original 44 training and 12
validation roles were preserved. Training parents were deterministically divided into
fit and calibration groups before fitting. Re-extracted validation predictions and
conditional commits match the previous matched replay (check.py passed).

The lightweight decision model is a class-balanced linear ridge fit (fixed regularization
100) on four ordered bins of normalized landmark features plus proposal/verifier scores.
It is a quick temporal-feature feasibility probe, not the proposed full learned visual
emission head. No RGB embeddings, candidate-identity conditioning, or learned temporal
encoder were added. Existing recognition weights stayed frozen. Features use the shared
live landmark preprocessing without wrist-motion trimming; classifier paths keep their
existing trimming. Oracle interval windows still prevent any full-stream latency claim.

Fit: 48 distinct sign cores, each with tight/+100ms/+250ms representations, and 14 gaps.
Calibration: 11 cores with those three representations and 2 gaps.
Validation: 17 cores with those three representations and 3 gaps. Representations of a
core are not independent signs. Thresholds preserve all calibration positive windows;
validation was not used for fitting, regularization choice or threshold selection.

| Validation measure | Current decisions | Confidence filter | Temporal decision filter |
|---|---:|---:|---:|
| Gap conditional commits (lower better) | 1/3 | 1/3 | 1/3 |
| Correct tight-core conditional commits | 5/17 | 5/17 | 5/17 |
| Correct +100ms conditional commits | 9/17 | 9/17 | 8/17 |
| Correct +250ms conditional commits | 5/17 | 5/17 | 4/17 |

The temporal gate rejected both calibration gaps but none of the three validation gaps.
The demonstrated false WORK survives. Its apparent calibration separation did not transfer.
Across context-window representations it suppresses two previously correct decisions.
This is a negative result for this small linear probe, not proof that transition detection
or a richer temporal/visual model is impossible. Tiny negative coverage, representation
limitations and source variation remain unresolved; no causal attribution among them.

## Limits and next action

These are conditional window decisions, not full scheduler emissions, stream WER or an
independent benchmark. Gap targets use guarded annotation gaps in admitted complete
sequences, not newly reviewed physical-transition labels. Positive-only O5S5 and unlabeled
YouTube were not treated as negative supervision. The user's HELLO-to-MY failure is still
not available in a readable paired recording. Low-wrist-motion activation and genuine
GOODBYE retention were not established by this small set. No runtime improvement claimed.

Do not stack this filter or tune it against the three validation gaps. Next useful work is
auditing hard negatives from existing training-source continuous video, paired with real
confusable signs, then testing richer visual temporal evidence on separately held-out
parents. Preserve this failed baseline and current live model. No new acquisition needed
for that audit; sufficient reviewed coverage has not yet been established.

## Reproducibility

recipe.json pins input/script hashes and the narrow recipe contract; the generic phrase
training gate remains false. run.py verifies approved membership and raw video hashes,
replays models, fits the linear probe and writes evidence/results. check.py tests schedule,
overlap exclusion and a synthetic linear separation, then compares validation predictions
with the preceding replay and checks veto counts. It passed, as did git diff --check.
No checkpoint or live threshold was changed. Protected test data were not accessed.
