# v17 independent portrait-iPhone confirmation protocol

## Purpose and freeze

This set confirms the already frozen v17 Stage-1 candidates on the intended capture
domain. It is not training data and must not select checkpoints, fusion weights,
thresholds, augmentations, or architectures.

- Vocabulary: the exact 100 canonical labels and ASL-LEX variants in
  `active/v17/citizen100_manifest.json`.
- Minimum collection: five people × two repetitions × 100 labels = 1,000 target
  clips.
- Recommended UNKNOWN set: each person also records 20 natural non-target signs or
  non-sign gestures = 100 OOV clips.
- Signers must not occur in Citizen, SemLex, or the local training supplement.
- The current landmark checkpoint, hand checkpoint, mouth/lower checkpoints, and
  fixed fusion weights must be hashed before the first evaluation.
- Evaluate the complete frozen set once. Do not delete difficult clips or tune from
  its errors afterward.

## Exact-variant gate and executable setup

Do not begin capture from the English labels alone. The frozen Citizen manifest still
has status `metadata_frozen_pending_asl_review`; a matching ASL-LEX code identifies the
intended lexical variant, but an ASL-fluent reviewer must confirm every reference
before it becomes a capture prompt.

The 100-row review sheet is
`active/v17/portrait_iphone_variant_review_v17.csv`. For each row, the reviewer opens
the linked ASL-LEX visualization, verifies that the pinned Citizen raw gloss/code and
reference are the intended variant, and fills `review_status`, `reviewer_id`,
`reviewed_utc`, and optional notes. `reviewed_utc` must include a timezone. A rejected
or pending row blocks the entire capture pack; never substitute a normalized label or
another numeric variant.

All 100 currently pinned rows received explicit project-owner approval on
2026-08-13. The approved sheet is hash-locked into the capture pack; any later edit
invalidates its audit and requires a new declared pack rather than silent replacement.

ASL-LEX permits personal searches of its reference videos but prohibits saving,
displaying, or reusing those videos without permission. The review sheet therefore
stores links only; do not download, embed, screen-record, or redistribute a reference
video. See the official [ASL-LEX data and license page](https://asl-lex.org/download.html).

After all 100 rows are approved, create the immutable capture pack with five new
pseudonymous signer IDs:

```bash
venv/bin/python scripts/build_portrait_iphone_eval_v17.py build-pack \
  --review active/v17/portrait_iphone_variant_review_v17.csv \
  --output-dir data/local/portrait_iphone_eval_v17 \
  --signer-id S01 --signer-id S02 --signer-id S03 --signer-id S04 --signer-id S05
```

This produces two independently randomized 100-label schedules per signer, a separate
20-slot OOV schedule per signer, a 1,100-row capture ledger, and a hash manifest. It
does not run inference. The current pack is already built at
`data/local/portrait_iphone_eval_v17/`. Immediately verify the untouched setup:

```bash
venv/bin/python scripts/build_portrait_iphone_eval_v17.py audit-pack \
  --pack-dir data/local/portrait_iphone_eval_v17 \
  --review active/v17/portrait_iphone_variant_review_v17.csv \
  --candidates active/v17/portrait_iphone_candidates_v17.json \
  --phase setup \
  --report artifacts/reports/portrait_iphone_eval_v17_setup_audit.json
```

The audit supports recapture without erasing a failure: preserve the rejected attempt,
append a row with the same `planned_id`, increment `attempt`, and use the matching
`capture_id` suffix (for example `-a02`). The later `pre-inference` audit requires
exactly one objectively accepted attempt per planned target/OOV slot, no pending rows,
unique content hashes and paths, complete device metadata, the prompt-hidden
confirmation, and exact target-variant confirmation. No model may be run until that
audit reports `ready_for_first_inference: true`.

## Frozen candidates and pre-inference media gate

`active/v17/portrait_iphone_candidates_v17.json` pins the exact six checkpoints,
runtime sources, evidence reports, MobileCLIP2 asset hash, and three fixed fusion
definitions. The primary research teacher remains the flat-landmark/mouth/lower-face/
hand composition with weights 0.30/0.15/0.35/0.20 and development evidence of
370/378 Citizen and 882/978 SemLex. The compact standalone candidate remains the
part-wise+global landmark model at 366/378 and 853/978. The manifest prohibits
recalibration; changing any checkpoint, runtime source, evidence artifact, member, or
weight invalidates the pack audit.

After every plan has exactly one accepted capture and the ledger is complete, run:

```bash
venv/bin/python scripts/build_portrait_iphone_eval_v17.py audit-pack \
  --pack-dir data/local/portrait_iphone_eval_v17 \
  --review active/v17/portrait_iphone_variant_review_v17.csv \
  --candidates active/v17/portrait_iphone_candidates_v17.json \
  --phase pre-inference --decode-workers 4 \
  --report artifacts/reports/portrait_iphone_eval_v17_pre_inference_audit.json
```

This phase verifies every accepted file hash and uses `ffprobe` plus a full
video-stream-only `ffmpeg -xerror` decode. It checks rotation-aware portrait dimensions
and frame rate against the ledger and never selects or decodes audio. Inference remains
forbidden unless all 1,100 files decode and the report says
`ready_for_first_inference: true`.

## Capture procedure

Use the real application capture path on an iPhone in portrait orientation. Record
upright video without aspect-ratio stretching; retain the original file and rotation
metadata.

1. Show the label prompt before recording, then hide it before the signing interval.
2. Frame the signer from approximately waist/chest to above the head, with both hands
   able to remain in frame.
3. Ask for natural signing and natural facial behavior. Do not require speaking or
   mouthing the English word.
4. Record two repetitions in separately randomized class orders. Prefer different
   sessions, backgrounds, clothing, or lighting for the two repetitions.
5. Use the front camera if that matches production. Preserve whether the saved video
   is mirrored; do not silently flip files during collection.
6. Keep audio in the raw file if the app normally records it, but all v17 evaluation
   extractors must continue to use visual input only.

Recommended minimum video properties are 1080×1920 portrait, 30 fps, and enough lead
and tail context to contain the complete sign. Lower-resolution devices are allowed
as an explicit device stratum; never upscale them to satisfy the ledger.

## Ledger and quality control

Start from `active/v17/portrait_iphone_capture_template.csv`. `signer_id` must be a
study-local pseudonym. Keep consent records separately from the repository and never
put names or contact information in the ledger. `canonical_label` is one frozen label
or `UNKNOWN`; UNKNOWN rows also require `performed_gloss`. For target rows,
`performed_gloss` confirms the exact pinned `expected_raw_gloss`, not merely the
normalized English label.

Quality control may reject only objective capture failures before any model is run:

- file cannot decode completely;
- the sign is truncated by recording start/end;
- the signer or required hands leave the frame for most of the sign;
- the ledger label and performed prompt are known to disagree;
- the file is an exact duplicate.

Preserve every rejection and reason. Blur, unfamiliar background, difficult signer,
low model confidence, or model disagreement are not valid post-evaluation deletion
reasons.

## Frozen evaluation

Report these results separately and together:

- compact part-wise+global landmark Squeezeformer (the standalone landmark winner);
- the frozen flat landmark member used by the existing teacher;
- multisource compact hand RGB;
- mouth, lower-face, and learned mouth+lower diagnostics;
- the already declared fixed 75/25 landmark+hand fusion;
- the existing fixed 0.30/0.15/0.35/0.20 flat-landmark/mouth/lower/hand teacher;
- optionally, the already measured part-wise substitution with those same fixed
  weights. Do not recalibrate either composition on this set.

For the 1,000 target clips report top-1, top-5, macro F1, per-signer accuracy,
per-class accuracy, and 95% signer-clustered bootstrap confidence intervals. Report
paired corrections/regressions and exact McNemar tests against landmarks. For the OOV
clips report maximum-softmax and energy score distributions, but do not choose an
UNKNOWN threshold on this set unless a separate portrait development set has first
been collected.

This confirmation set becomes consumed after evaluation. Its errors may guide a
future study, but that study requires a new independent confirmation set.
