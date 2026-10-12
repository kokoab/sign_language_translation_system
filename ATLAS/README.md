# ATLAS canonical source

ATLAS is the canonical source location for the current mobile AI/ML pipeline,
effective 2026-10-13. Edit the files here. The former Python filenames and
`/Volumes/secret/SLT/mobile_app/slt_mobile_app` are compatibility symlinks to these
files, not independent copies. Keep those links while historical commands,
imports, input contracts and development tools still use them.

## Layout

| Directory | Purpose |
| --- | --- |
| `python/extraction/` | Apple Vision/MediaPipe extraction, geometry, hand crops and feature schemas |
| `python/recognition/` | Landmark Squeezeformer, hand-image branch and unified fusion architectures |
| `python/boundary/` | Boundary features, Apple Vision student and offline teacher adapter |
| `python/letters/` | Letter head attached to the span recognizer |
| `python/english/` | Gloss encoding, language contracts and Core ML English inference |
| `python/training/` | Recognition, phrase/span, boundary, letter and English trainers; selected experiment runners |
| `python/export/` | Core ML and Android TFLite exports, including the deployed dual-head export |
| `python/recipes/` | Frozen manifests, training gate, vocabulary and environment requirements |
| `python/data/` | Training corpus/cache preparation |
| `python/evaluation/` | Existing component evaluators and offline segmental lab |
| `python/runtime/` | Shared recognition/runtime helpers |
| `python/tests/` | 24 focused test files for canonical ATLAS components |
| `mobile_app/slt_mobile_app/` | Complete Flutter project, native Swift/Kotlin pipelines and bundled models |

`source_manifest.json` lists every relocated file and its former path. Supporting
research architectures are included where imported by the selected chain; their
presence is not a deployment endorsement. Optional desktop/UI helpers remain at
their existing paths. This layout is part of the SLT workspace, not a standalone
training distribution: datasets, checkpoints, caches, reports, vendor code and
the Python environment remain under the existing repository locations.

The focused Python suite lives in `python/tests/` (245 defined test methods).
Former `test/test_*.py` paths for these 23 files are compatibility symlinks;
the remaining 125 test files stay in the root `test/` directory. Existing
`test.<module>` invocations remain supported. Keep the test compatibility links.

## Current mobile pipeline

iOS source: `mobile_app/slt_mobile_app/ios/Runner/LiveReel/`.

- `LiveReelCore.swift`: Vision observations, signer selection and feature contracts.
- `LiveReelModels.swift`: Core ML loading, hand-image encoding and span/letter inference.
- `LiveReelDecoder.swift`: segmental decoding and scoped transition repair.
- `LiveReelEngine.swift`: word/letter runtimes and pipeline orchestration.
- `LiveReelStage3.swift`: tiny T5 English generation and incremental sentence locking.
- `ReelCamera.swift`: camera frame delivery.
- `Models/`: bundled Core ML packages and inference metadata.

The current recognizer is `SpanRecognizerV17ConfidenceDualB8FP16`, retaining the
primary output from `run_frag25_reject` alongside the transition-adapted output.
Its lineage is the 96.83% part-wise/global base → run_v2 local replay → fusion →
phrase adaptation → fragment/rejection span adaptation → confidence dual export.
`python/training/train_confidence.py` and
`python/export/export_dual_confidence.py` are the final repair/export sources.

Android source: `mobile_app/slt_mobile_app/android/app/src/main/kotlin/com/example/slt_mobile_app/live/`.
Bundled MediaPipe/TFLite files: `mobile_app/slt_mobile_app/android/app/src/main/assets/`.
Android has its own extractor/model family; the iOS repair is not an Android model update.

Verification: 52 focused Python tests pass; the unsigned iOS Release and offline
Android Debug builds pass from this location. All 582 inventoried mobile source/
asset hashes and 11 compiled Core ML weight payloads were preserved. Build setup
now explicitly links Flutter in the Debug/Release xcconfigs. Full receipts:
`../artifacts/reports/atlas_canonical_source_20261013/REPORT.md`.

## Paths, imports and frozen evidence

The public Python import namespaces remain `active.v17` and `scripts` for
checkpoint and caller compatibility. They resolve to the canonical files through
the old-path links. Cross-folder imports use those stable namespaces. Repository
root calculations now account for `ATLAS/python/<purpose>/`. The moved experiment
scripts explicitly retain their original report directories for frozen inputs and
outputs; moving source does not move experiment evidence or silently start a run.

Recipe/configuration file bytes and mobile source/model bytes were preserved.
Python source edits only adjust import/path plumbing. Historical source-hash pins
are not rewritten: original source snapshots are retained in
`artifacts/reports/atlas_canonical_source_20261013/before_sources/`. Any future
training run requiring changed source hashes needs a reviewed new recipe. Do not
bypass a pin mismatch or enable the canonical phrase manifest. The official
Citizen test stays sealed.

Generated mobile build/dependency state is retained locally and remains ignored.
Older bundled model packages are retained because existing project resources may
reference them; no model package was removed or replaced.

## Commands

Run Python from the SLT root with `venv/bin/python`, for example:

```sh
venv/bin/python ATLAS/python/extraction/extract_v17.py --help
venv/bin/python ATLAS/python/training/train_span_recognizer_v17.py --help
venv/bin/python ATLAS/verify_layout.py
```

Run the focused ATLAS Python suite from the SLT root:

```sh
venv/bin/python -m unittest discover -s ATLAS/python/tests -t . -p 'test_*.py'
```

The original root discovery command still finds these files through their
compatibility links, together with the other repository tests.

Canonical discovery silences library output during imports and test execution,
including native Vision/MediaPipe logs. Unittest progress, failures and the final
summary remain visible; add `-v` for individual test names. Output descriptors
are restored afterward. Direct old-path module invocations do not use this hook.

The user requested removal of `test_stage3_mobile_naturalizer_v17.py` and its
compatibility link on 2026-10-13. Its eight methods were blocked by a class-setup
recognizer/Stage-2 hash mismatch. This removes that coverage; the runtime hash
check and model contracts are unchanged.

Build the mobile app from the new project directory:

```sh
cd ATLAS/mobile_app/slt_mobile_app
flutter pub get --offline
flutter build ios --release --no-codesign --no-pub
```

Open `mobile_app/slt_mobile_app/ios/Runner.xcworkspace` in Xcode for signed iOS
builds. Signing settings and the installed phone application were not changed by
the source relocation. See the migration report for the exact verification results.
