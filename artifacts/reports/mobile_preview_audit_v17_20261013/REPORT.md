# Mobile gloss preview audit — 2026-10-13 PHT

The mobile preview catalog had real training-variant discrepancies. BIG, SIGN, COME,
and CHILD used local forms excluded from the current Citizen-variant recipe. The
preview quality shortlist was not a training admission manifest. Model weights,
runtime thresholds, datasets and training gates were preserved.

## Locations

- Flutter/mobile: `ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples/`.
- Native iOS: `ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/gloss_examples/`.
- Desktop reference assets: `artifacts/app_assets/gloss_examples/`.
- Citizen original training clips: `data/local/citizen100_v17/raw/train/<GLOSS>/`.
- Local original recordings: `data/raw_videos/ASL VIDEOS/<GLOSS>/`; admitted training
  copies: `data/local/local_deep_clean_v17/raw/train/<GLOSS>/`.
- Current local training membership: `data/local/web_variant_supplement_v17_20261008/local_train_citizen_variant_v2.json`.
- Additional isolated supervision: `data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json`.
- Reviewed phrase input contracts: `ATLAS/python/recipes/phrase_segment_recipe_manifest_20261007.json`.

Every catalog row records `source_video`, its SHA256, the exact pinned Citizen raw
gloss/ASL-LEX code, class index, and rendered filenames. For example, the `I` class
is pinned to Citizen's `ME` sign.

## Provenance findings

The two original catalogs and all 200 media files matched byte-for-byte. There were
no duplicate video files, and all 100 class indices matched the recognizer label order.
Original selection: 77 local recordings and 23 Citizen training clips.

Of the 77 local previews, 50 matched current eligible training inputs, seven matched
local validation inputs, and 20 were absent from current train/validation. Matched
paths and content hashes agreed; none of the 57 matched records had a different
canonical label. The old train-only metadata therefore overstated its local scope.
Validation previews: WORK, SLEEP, BAD, YESTERDAY, MORNING, FATHER, NAME.

BIG and SIGN were explicitly removed as differing variants. Local COME and CHILD
classes were also excluded for variant differences. The remaining absent previews
are not automatically mislabeled: their original exclusion reasons are unproven.
ASK was restored in the v2 recipe and remains admitted. Detailed membership and
source paths: [provenance.md](provenance.md), [provenance.json](provenance.json).

## Changes and verification procedure

`build_gloss_examples_v17.py` now requires a local quality candidate's label and
source content hash to match an eligible train record in the current v2 recipe.
Local hashes, train split and admission-manifest path are included in new catalogs.
A regression check rejects validation, ineligible, wrong-label and wrong-hash rows.
Citizen fallback remains strictly train-only and honors the rejection list.

Initial native diagnostic checks used the deployed FP16 image encoder and the
harness's default cpuAndGPU recognizer: original 78/100 exact outputs; membership
repair 85/100. All subsequent checks explicitly used `.all` for the recognizer and
image encoder, matching LiveReelApp, with ConfidenceDual and current decoder sources.
Thus the initial percentages are diagnostics, not a controlled backend comparison.

132 training-only alternatives were rendered and replayed. Later complete checks
revealed schedule-sensitive misses/extra words, so 144 paired original/preview
replays, 24 additional LEARN checks and 44 further candidate checks were used to
select references that passed both forms twice. Apple Vision's eight-frame auxiliary
cadence persists across resets; different input durations/order can change it.
These repetitions are correlated development checks, not independent examples.

The final original-video set passed 100/100 exact hypotheses. One 480px ANGRY preview
produced an extra COLD; keeping that recording's 640px source width passed four
replays. Its frame count and source identity stayed unchanged. The complete delivered
preview check and final delivery counts are recorded in `delivery.json` and
`delivered_replay.json`; the original-video results are `latest_source_replay.json`.

The delivered candidates contain 63 local training sources and 37 Citizen training
sources; 46 source choices changed. Final native checks passed 100/100 exact previews
and 100/100 exact original recordings. Fifteen focused tests passed after delivery. Source label/hash admissions are checked before delivery; actual poster and
loop hashes are stored per catalog row. Existing changed files are backed up under
`before/`. The runnable delivery gate asserts all 100 previews and originals give
exactly their reference word, verifies pinned source/model hashes, and synchronizes
Flutter, native iOS and desktop assets. It refuses delivery if any check fails.

## Scope

This checks selected known training references through native Swift/Core ML on the
Mac, using real decoded pixels, Apple Vision, hand crops, boundaries and decoding.
End-of-file flushing is included. Automatic finish-gesture detection is disabled.
It is not independent signer accuracy, every-training-video accuracy, continuous
translation accuracy, or a physical-phone camera/latency check. No official Citizen
test clip was accessed, and no model training, checkpoint selection, threshold
changes, acquisition or phone installation was performed. The seven old local
validation previews were already displayed and are explicitly identified above;
they are not being presented as new independent validation.

Rebuild/reinstall the app to put the synchronized assets on the phone, then check
fresh signing on the device. Regeneration or future model changes require renewed
native preview/source checks; quality ranking alone cannot establish recognition.

## Evidence and reruns

- `audit_provenance.py` reproduces original 50/7/20 membership counts from the saved catalog.
- `iteration_summary.json` preserves earlier failures; intermediate results were not overwritten.
- `source_model_hashes.json` pins the tested native source and deployed model payloads.
- `tests_final.log` records 15 passing focused tests; `full_diff_check_final.log` is clean.
  `artifacts/LARGE_FILES.md` was regenerated after delivery.
- `deliver.py` reruns admissions/hashes, exact-output delivery gates and bundle synchronization.
