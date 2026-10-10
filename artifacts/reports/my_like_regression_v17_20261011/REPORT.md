# Phone MY → LIKE regression — 2026-10-11

## Finding

A real recognizer regression is reproduced from the phone's saved inputs. The Oct8 replacement
of LocalA by the CitizenVariant model flips the first reviewed MY → NAME occurrence from MY
to LIKE. Every later deployed recognizer tested retains that error. The Oct10 HAVE decoder
change does not cause this substitution. No production source, model, training data or installed
app was changed during this investigation. No new correction has been validated.

## Phone evidence

Session `20261011_102647` contains 9 MY and 15 LIKE commits; these counts are predictions,
not an assertion that all 15 LIKEs are errors. Videos03/04 and the saved trace were pulled
from the paired iPhone13 app container into `data/local/phone_session_pull_v17_20261011/`.
The first reviewed failure is video03, approximately2.82–3.09s after recording begins:
64.068–64.335session seconds, live LIKE .86345. The video shows an open palm contacting the
chest before the hand transitions into NAME (`first_my_contact.png`). A second reviewed
MY-to-NAME contact at approximately24s also gets LIKE (`later_contact.png`).

The complete103-frame word-mode run before NAME mode permits identical-input native replay.
The current runtime matches HELLO LIKE NAME and the first LIKE start/end frame indices56–61,
with score .86188 versus .86345 live (small desktop/phone numerical difference).
The saved decoder from before the HAVE closure gives the same LIKE span and score; its old
ordering bug outputs HELLO NAME LIKE. This comparison isolates the decoder change only.

## Model swap on identical saved inputs and fixed current decoder

| Recognizer | Output | P(MY), frames56–61 | P(LIKE), frames56–61 |
|---|---|---:|---:|
| LocalA | HELLO MY NAME | 0.7908 | 0.0628 |
| CitizenVariant | HELLO LIKE NAME | 0.0501 | 0.8215 |
| CitizenVariantV2 | HELLO LIKE NAME | 0.0341 | 0.8500 |
| Fragment | HELLO LIKE NAME | 0.0222 | 0.8593 |
| FragmentReject | HELLO LIKE NAME | 0.0142 | 0.8619 |

Word scores are normalized over100 word classes. This holds decoder/labels/landmarks/hand
embeddings constant and replaces the recognizer. It establishes an actual model regression
on this occurrence, not a recording frame-rate discrepancy or Stage3 word substitution.
Historical letter thresholds are not replayed: this is a controlled recognizer comparison.

## Mechanism and localization

On the current model, an early-contact window (50–57, with normal context) is MY .934;
the eventual56–61 window is LIKE .862 and MY .014. The same eventual window remains MY .791
in LocalA. The new model is sensitive to the short contact followed by the hand changing
shape for NAME. The62-window diagnostic grid gives MY as winner in6 windows for the current
model versus35 for LocalA; overlapping windows are diagnostics, not independent accuracy cases.

Native-prepared inputs were exported for a read-only PyTorch stage comparison (`stage_probe.py`,
`stage_scores.json`). Before phrase adaptation, the new landmark branch already gives LIKE
.9048 / MY .0015 on the completed window, and fusion gives LIKE .9121. The old final landmark
branch is uncertain (LIKE .3469 / MY .2986), and old fusion recovers MY .7871. New phrase and
interval stages retain the error. The pre-filter96.83-derived chain also gives LIKE .9316
at fusion and .8606 at final recognition. Therefore the local Citizen-variant exclusion filter
is not sufficient to explain it; the replacement model lineage already contains the weakness.
This does not isolate one training sample, augmentation, or architecture choice as its sole cause.
MY and LIKE local examples were not removed by that filter.

Removing100ms context from the final56–61 window still gives LIKE .8998 in new fusion;
shortening further to56–59 restores MY as its top class. Simply deleting span padding is not
a demonstrated repair. These branch scores use FP32 PyTorch and are not phone calibration
claims; the final v1 classifier agrees with its native counterpart on the error.

## Why the old validation missed it

Saved Oct8 confusion matrices show both old-phone and CitizenVariant recognizers at MY30/31:
Citizen4/4, SemLex4/5, local22/22. Isolated whole-clip MY retention did not test this live short
MY-to-NAME transition. Those validation results cannot establish unchanged streaming behavior.

## Wider replay and rejected shortcuts

A fully captured800-frame run from76.735–116.702seconds reproduces current
LIKE MY LIKE MY LIKE NAME. LocalA gives LIKE NAME MY PLEASE MY LIKE NAME: rollback does not
resolve all later misses. The172-frame opening of take04 gives MY MY MY NAME with both models.
Reviewed later palm contact remains LIKE with LocalA as well. Do not claim a wholesale rollback
is a safe fix, or relabel LIKE globally.

The saved trace omits automatic NAME/fingerspelling-mode frames and reset markers. We used
only contiguous recorded word-mode runs for these comparisons; it cannot support exact replay
of the entire204-second session. Video replay is a separate, weaker comparison elsewhere.

## Required correction and regression gate

Repair the recognizer's tolerance for the short MY contact and following transition. Keep the
reviewed phone events as explicit development regressions; use genuine LIKE motion, MY,
PLEASE, FEEL, I, NAME and HUNGRY controls, plus current HAVE/BIG and full-vocabulary retention.
Do not teach English context to silently turn LIKE NAME into MY NAME. Any adaptation must
use a reviewed data/recipe contract and preserve the protected-test gates. New phone data
used to select a fix must not then be described as independent validation.

A correction is not yet built or installed. The evidence supports the cause and rejects a
blind rollback/context deletion; it does not establish a safe universal decision rule.

## Reproduction

`models_main.swift` runs each bundled model against `first_miss_trace.json`; the compiled
`model_probe` accepts resource directory, trace path, output path and the MODEL environment
variable. `dump_harness/main.swift` exports native span inputs. `stage_probe.py` scores preserved
training-stage checkpoints without training. Per-model JSON, window-grid JSON, stage scores,
live words, logs and contact sheets are retained in this directory. The long and opening trace
runs are separately named. `summary.json` contains compact results and modification status.
