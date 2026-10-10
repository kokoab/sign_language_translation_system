# Full app-change and rollback audit — 2026-10-11

This extends and qualifies the initial model-only diagnosis. The user explicitly confirms
that some LIKEs were intentional, without a remembered timestamp. Predicted LIKE counts
must not be treated as error counts. Only takes03/04 survive in the refreshed phone video
inventory; deliberate LIKE is not confidently located in those files. Palm-contact sheets
and12complete prepared spans are diagnostic visual review, not user-confirmed training labels.
No production edit, model promotion, new training, protected-test access or installation.

## Proven additional cause: early commitment changes NAME-mode timing

The Oct10 HAVE omission patch also introduced a **global** ordering guard in
`LiveReelDecoder.swift` (current line801) and Python `segmental_runtime_v17.py` (line888).
It blocks early identity locking whenever an earlier segment remains unresolved. Its reach
is all word classes/decoder configurations, not just HAVE. NAME is allowed to lock early;
MY is in `earlyUnsafe`. Delaying NAME identity changes which interval is eventually locked
and when the engine enters/exits its four-second automatic NAME/fingerspelling mode.
Word mode and NAME mode use different decoder configurations and reset/replay state on switching.

A full Vision/crop/decoder/automatic-mode replay of take03, using source presentation times
and the **same current recognizer**, establishes this interaction:

| Variant | NAME mode entry/exit after first NAME | Contact around9.2s |
|---|---|---|
| Current production sources |4.767 /8.767s|LIKE, start9.200s|
| Saved pre-HAVE/ANGRY decoder + pre-trace engine |5.267 /9.267s|MY, start9.167s|
| Current sources, only new ordering guard disabled |5.267 /9.267s|MY, start9.167s|
| Current sources, guard restricted to pending HAVE |5.267 /9.267s|MY, start9.167s|

Disabling only the guard reproduces both full-video hypotheses AND mode-event times from
the pre-HAVE/ANGRY source comparison. This isolates the additional code contribution.
Deleting only the guard's unresolved-rest clause does not restore the old behavior.
Take04 gives the same hypotheses under the old/current source comparison with a fixed model.
These are matched-video counterfactuals, **not exact phone-frame replay**: the recording's
sampling phase/compression changes some outcomes versus live. The first exact phone-frame
MY substitution still occurs before automatic NAME mode and survives decoder rollbacks.
Thus neither “only the model changed” nor “the HAVE fixes explain every miss” is supported.

## What changed, verified against saved sources and dated logs

| Component | Change / evidence | Finding |
|---|---|---|
| Recognition model and letter head |Oct8 CitizenVariant, Oct9 V2 and Fragment, Oct10 FragmentReject|Model contribution demonstrated separately on identical phone frames.|
| Hidden letter competition |Oct8 word gate .95; earlier ungated|Both gated/ungated and words-only still give first LIKE.|
| Word segmentation |Oct9 rest allowance3→8, activity floor.5, lag1→6; minimum remains3|Reverting individually does not fix first LIKE. Removing floor makes initial take04MY HUNGRY; lag1 addsLIKE.|
| Word threshold |Oct10 .5→.4 with FragmentReject|Restoring .5 does not fix examined first substitution.|
| HAVE/ANGRY hand/crop cues |Oct9 pair check; Oct10 one-way rescue|Turning HAVE-specific changes off leaves all three contiguous word-mode hypotheses unchanged.|
| TIRED and HAVE closure |Oct10 downward-roll check, accepted-contact retention|Not sufficient explanation for recorded MY substitutions.|
| Global early ordering |Oct10 HAVE omission patch|Causal full-engine NAME-mode interaction above.|
| Finish gesture |Oct8 two palms→right palm/left fist above shoulders|Only difference in core observation source versus Oct8 backup; zero held frames in both reviewed full videos.|
| Vision/features/hand cropping |Core and model-wrapper source comparisons|Vision, tracking, normalization, crop/embedding preparation unchanged in compared snapshots; model wrapper differs only default package name.|
| Camera/UI capture |Oct2→pre-trace view-controller source comparison; camera backup|Capture processing/sampling code unchanged in comparison; UI change was separate Practice letters preference. Older ReelCamera diff only Stage3 warmup.|
| NAME mode |Existing4s automatic mode; no direct timer/transition implementation edit in pre-trace→current diff|Indirect behavior changed because NAME commitment changed.|
| Trace logging |Oct10 engine/app/view-controller additions|Per-frame word-mode snapshots and synchronous whole-trace JSON save every≥20s on recognition queue. Potential load issue; not proven cause of these MY substitutions.|
| Boundary network / encoder choice |Saved engine constructors / reports|Same word/letter boundary and image-encoder names; boundary retrain was explicitly rejected. Historical immutable hashes are not available for every asset, so this is not a blanket binary-identity claim.|

Source patches and hashes: `source_audit/manifest.json` plus `.patch` files. iOS sibling has
no Git repository; history is reconstructed from preserved snapshots and deployment logs,
not an assumed clean Git baseline. Current production source hashes remain unchanged.

Trace timing near11of12 fully saved reviewed spans has max33/67ms camera-sampling intervals;
one has100ms after the span. No held control-pose frame there. Larger4s gaps are omitted
NAME-mode observations, not evidence of a4s camera stall. Trace records only word mode and
omits resets, so it cannot establish full-session real-time performance or missing camera inputs.

## Rollback experiments

- Four bundled recognizers ×3saved word-mode traces through actual pre-pair decoder sources.
- Nine decoder settings ×3saved word-mode traces, current recognizer fixed.
- Full two-video app-engine replay for current, historical decoder/engine, ordering-off,
  words-only-ordering guard, HAVE-only-ordering guard, and historical Fragment version.
- Seven preserved model checkpoints on the same3959isolated validation clips,100-class
  confusion matrices and saved logits. Counts are development retention, not new test accuracy.

| Recognizer checkpoint | Correct /3959 isolated clips |
|---|---:|
|old_phone|3743|
|old_base_fusion|3764|
|old_base_clean_phrase|3757|
|old_base_clean_recognizer|3745|
|v2|3772|
|fragment|3770|
|current|3784|

Both current and original LocalA score MY30/31 andLIKE45/48 in this matched isolated set.
The older clean-split recognizer exists already; re-evaluating it avoids unnecessary retraining.
It does not eliminate the contact-window confusion. Twelve prepared phone spans are fixed
current-selected windows, not full-decoder accuracy, and their intent labels are not confirmed.

The Fragment predecessor restores MY at111.5session seconds in the long exact-input run,
where FragmentReject returnsLIKE. Other changes trade different errors. A wholesale rollback
cannot be called a complete repair. Old LocalA with current HAVE logic still has10HAVEs in
old take11, but the broad isolated retention is41clips worse overall; this alone is not a
deployment selection rule, especially given known local-variant and historical split issues.

## Narrow rollback candidate and decision

Restricting the new early-order guard to pending HAVE reproduces the old NAME-mode behavior
and retains exactly the current hypotheses on take11 (10HAVE) and09:55 (9HAVE). However it
also restores duplicate NAME and NAME-before-LIKE output. It is a diagnostic candidate,
**not a clean fix and not promoted**. Broad training or reinstalling a base model cannot
repair that application-state interaction by itself.

Recommended repair order: preserve early-recognized identity independently from publishing
words in temporal order; test full engine including automatic NAME mode, then reassess
remaining MY/LIKE model errors on intent-reviewed controls. Keep intentional LIKE examples
separate from MY attempts. The recognition evidence supports a mixed model/runtime issue;
there is no validated universal rollback or retraining recipe from this audit.

`verify_audit.py` verifies the central saved comparisons. `settings_harness`, `historical_harness`,
`video_historical`, `video_order`, `order_trace`, and `compare_bases.py` reproduce the experiments.
Production Python/Swift and the installed app are unchanged.
