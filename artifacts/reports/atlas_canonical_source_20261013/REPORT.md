# ATLAS canonical source relocation — 2026-10-13

User authorized ATLAS as the canonical source, including the mobile application,
with compatibility preserved. Relocation is complete. No training, acquisition,
protected-test inference, model-weight changes, phone installation or cleanup of
unrelated work was performed.

## Result

- 81 Python sources under `ATLAS/python/`, grouped by purpose. Includes selected
  final model definitions/trainers/exports, transitive helper modules, the report-
  scoped transition repair and tiny-T5 ancestor/composition training sources.
- 16 recipe/configuration/environment files under `ATLAS/python/recipes/`,
  byte-identical to their original files. Includes the selected recognition-chain
  recipe, fragment/rejection command record and confidence-head training recipe.
- Full Flutter project moved from the sibling directory to
  `ATLAS/mobile_app/slt_mobile_app/`, including Swift/Kotlin sources, bundled
  models, assets and existing local caches. Older bundled model resources remain.
- All 97 former Python/resource paths are relative compatibility symlinks.
  The former sibling mobile-project path is also a relative symlink. Public
  `active.v17` and `scripts` imports remain stable; edit the canonical ATLAS files.
- Root anchors account for the new folder depth. Relative cross-purpose imports
  use the stable namespace. Experiment source retains explicit historical report
  directories for frozen inputs/outputs. The renamed temporal-repair helper also
  imports correctly directly from its canonical package.

`ATLAS/source_manifest.json` and `migration.json` record individual file mappings
and before/after hashes. Original source snapshots are under `before_sources/`.
Datasets, historical checkpoints/results, vendor files and the existing Python
environment remain at their established locations. This is workspace organization,
not a standalone dataset/training package.

## Validation

| Check | Result |
| --- | --- |
| Canonical layout and compatibility links | 97/97 files plus mobile link pass |
| Python source parsing | 81/81 pass |
| Canonical/legacy imports and root anchors | 15 pass |
| Extractor, family adapter, segmental runtime tests | 44 pass |
| English composition tests | 8 pass |
| Actual current primary checkpoint restore/inference | Bit-identical logits before/after on a deterministic two-input smoke fixture |
| Canonical approved phrase verifier | 494 archives, unchanged manifest hash; training_ready=false |
| Inventoried mobile source/assets | 582/582 unchanged hashes, none missing |
| iOS Release build from ATLAS | Pass, unsigned; Runner.app produced |
| Compiled iOS model weights | 11/11 match source package weight bytes |
| Android Debug build from ATLAS | Pass, offline; 3m46s, APK produced |
| Android APK | All ZIP CRCs pass; boundary, both recognizers and T5 model resources present |
| Changed Python source whitespace | 68 changed-source diffs, zero findings |
| Repository diff whitespace | `git diff --check` passes |

Inference smoke inputs are synthetic and do not produce an accuracy or
generalization claim. No official Citizen test or other protected pool was used.

## Build failures resolved

The first Android attempt had no configured Java runtime. Used the existing
`artifacts/generated/android_tools/jdk17/Contents/Home`, with the previously used
bounded JVM/two-worker settings. The next attempt failed while invalidating a stale
Flutter debug-output directory after relocation; that directory had cleared by
inspection, and the retry passed. No Gradle/JDK/Flutter upgrades were performed.

The first iOS build regenerated CocoaPods configuration and failed linking Flutter
symbols. Added `OTHER_LDFLAGS = $(inherited) -framework Flutter` to
`ios/Flutter/Debug.xcconfig` and `Release.xcconfig`. The subsequent Release build
passed in 46.5s. These two build configurations are the only intentional mobile
source/configuration edits. Generated Flutter application/package paths now point
to ATLAS. Signing settings and application inference source remain unchanged.

Logs: `ios_build_attempt1.log`, `ios_build.log`, `android_build.log`,
`python_tests.log`, `english_tests.log`. Machine-readable receipts:
`verification.json`, `mobile_post_build_hashes.json`, `source_diff_check.json`.

## Future work constraints

Keep compatibility links until every caller and frozen reference is deliberately
migrated. Historical source-hash contracts were not silently rewritten. Changed
source snapshots are preserved, but future training using changed source pins
requires a reviewed new recipe; do not bypass hash checks or enable the canonical
phrase manifest. Existing training/test restrictions still apply.

Build checks do not install either application or establish new phone recognition,
latency, thermal or signer-generalization results. The previously installed app
remains untouched. Next safe action: work from ATLAS, open the canonical Xcode
workspace for a separately requested signed build, and retain the migration map.
