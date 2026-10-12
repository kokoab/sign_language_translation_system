# Android alignment before installation — 2026-10-13

User chose **align Android first, then install**. Samsung SM-A236E (Android 14)
is connected and ADB-authorized. No older APK was installed.

## Verified differences

| Component | Android now | Current iOS | Required work |
| --- | --- | --- | --- |
| Landmarks | Separate MediaPipe family, fingerprint d17b7cd2ecc5614f | Apple Vision | Keep distinct input/weight contracts |
| Word recognizer | October 4 local_a, hand-image and landmark-only modes | Fragment/rejection adaptation plus ConfidenceDual | Android-specific adaptation and exports |
| Recognizer outputs | One 100-word output | Primary/adapted words plus 27 letter classes | Multi-output Kotlin/TFLite contract |
| Decoder | maxInnerO3, lag1, theta0.5, no inner floor | Word-mode maxInnerO8, lag6, theta0.4, floor0.5 plus scoped repairs | Port behavior and validate with corresponding models |
| Letters | No live model/letter decoder; formatting supports fs strings only | Letter head/boundary and mode switching | MediaPipe letter inputs, training and runtime |
| English | Tiny-T5 multisentence checkpoint | Same checkpoint | Preserve and verify; no LM replacement needed |

Both language token tables declare `stage3_multisentence_tiny_v17_20260929`,
have identical word IDs/pieces, and use counted sentences. Five bundled Android
TFLite model hashes exactly match approved October 4 exports. These checks identify
the installed-source lineage; they are not new model-accuracy measurements.

## Work contract for review

`ATLAS/python/recipes/android_alignment_recipe_20261013.json` pins current source,
parent models, extractor schema and admission manifests. It is **disabled**. It
proposes an eight-epoch, seed27927, frozen-encoder word-head adaptation with
fragment probability0.3, retained fragment lengths25–85%, rejection probability0.15
for10–25% pieces, rejection weight0.5. Both Android modes need matching contracts.
Letter/transition jobs require separately pinned source/event/label contracts and
budgets before they may run; the draft does not enable those jobs.

The canonical 494-archive phrase verifier passes; canonical training_ready remains
false. The October7 exception admits frozen-encoder phrase/head adaptation only.
The current iOS interval command does not supply `--freeze-encoders`, while the
trainer otherwise updates encoders. Therefore blindly replaying that command on
Android would exceed this exception. The Android contract explicitly freezes
encoders; any wider encoder update needs separate review.

## Execution and acceptance

1. Reconcile current admitted train/validation membership against the existing
   MediaPipe caches. Pin exact derived phrase/letter/transition inputs and labels;
   preserve signer roles and the current identity exclusions. Use only admitted
   source roots. No acquisition, Apple landmark inputs or protected inference.
2. Finalize reviewed Android-specific training contracts, with measured baseline
   per-domain/class retention floors. Write new checkpoints; preserve originals.
3. Port model-independent state bookkeeping and configured span/decoder behavior;
   add current multi-output/letter/transition paths with their matching exports.
   Do not activate lower thresholds against the older unadapted model.
4. Verify export/PyTorch and Kotlin/reference agreement on identical MediaPipe
   development inputs; check fragment/rejection/repeat/rest/ordering controls.
   Keep protected sets sealed. The historical local tuning pool is development,
   not independent signer-generalization evidence.
5. Build and measure signed-input latency/memory on this Samsung before selecting
   its hand-image/landmark-only default. Install the aligned build only after its
   gates pass, preserving app data, then verify launch and camera/model startup.

User review of the scoped contract is required before enabling new phrase/model
training. Scope changes can be recorded in that contract. This plan does not
authorize unfreezing older mixed-data trainers, flipping the canonical flag,
copying Apple weights into MediaPipe inputs, or changing the installed iOS app.

Receipts: `audit.json` (source/parent hashes, model hashes and token checks) and
`recipe_draft.json`. No model/runtime changes or training were performed in this audit.
