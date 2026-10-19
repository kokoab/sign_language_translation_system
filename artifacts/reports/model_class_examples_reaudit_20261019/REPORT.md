# Actual copied-video primary-recognizer re-audit — 2026-10-19 PHT

**All100actual files in `data/local/model_class_examples/` return their expected class
from the deployed primary output and from the pinned PyTorch primary checkpoint.**
No videos were replaced or renamed during this re-audit. This is direct recognition of
selected training examples, not a new validation/generalization measurement.

## Model identity and procedure

Current native `LiveSpanRecognizer` defaults to the bundled
`SpanRecognizerV17ConfidenceDualB8FP16.mlpackage`. It exposes adapted `word_logits`
and retained primary `original_word_logits`. The canonical loader normally uses the
retained primary; only an eligible scoped MY transition repair can substitute adapted
scores. This audit explicitly passes `repairEligible: [false]` and reads the primary
word logits. Expected class names never enter inference, span selection or trimming.

Each actual copied MP4 was decoded again with AVFoundation, sampled at the app's20fps,
and processed using canonical Apple Vision extraction, native hand crops and the
bundled FP16 image encoder. Vision starts fresh per independent file. Canonical
`LiveSpanRecognizer.inputs` builds the full-clip input, with its native hand-activity
trim. One whole-clip primary prediction is recorded; there is no boundary search,
best-of-many candidate selection, dictionary replacement or English postprocessing.
Models use `.all`, matching deployment. The audit harness only adds input/output
recording; application source and model files remain unchanged.

The exact fresh native tensors were then fed to
`artifacts/reports/go_split_root_cause_v17_20261009/run_frag25_reject/span_recognizer/best_model.pth`,
the documented current primary (hash starts `99a18801`). Its full hash is verified
against the prior current-lineage model record and saved in `primary_verified.json`.
This checks the retained primary directly, rather than interpreting the adapted output
or an old v16/Citizen-only checkpoint as the primary model.

| Check | Result |
|---|---|
| Deployed CoreML primary, actual copied pixels |100/100expected top1|
| Current PyTorch primary, identical native tensors |100/100expected top1|
| CoreML/PyTorch top1 agreement |100/100|
| Official Citizen train membership, pinned class/raw gloss/ASL-LEX code |100/100|
| Copy/source/provenance SHA256 equality |100/100|
| Unique copied training video hashes |100|
| Earlier canonical source/model/configuration/provenance pins |34/34unchanged|

`per_video.csv` lists all100files, both predictions, scores and raw-video SHA256.
Scores are softmax model outputs, not calibrated probabilities of linguistic correctness.
ANSWER is the weakest top1 at47.98%; it is correctly predicted but should not be
presented as a particularly strong example. No failure was hidden or replaced.

## ASK versus YOU

| File | Pinned raw class/code | CoreML primary top1 | Score |
|---|---|---|---|
|ASK.mp4|ASK / B_02_023|ASK|98.03%|
|YOU.mp4|YOU / D_02_065|YOU|99.57%|

Both files are distinct recordings from Citizen signerP40, and both match their
official training provenance. The index-pointing handshape is visually similar.
The pinned local ASL-LEX metadata distinguishes ASK by **finger flexion change**
(`FlexionChange.2.0=1`, movement categoryNone), while YOU has **straight movement
without flexion change** (`FlexionChange.2.0=0`). Sampled-frame comparison with the
exact pinned citation videos shows the index curling in ASK and a straighter pointing
finger in YOU. No evidence of an ASK/YOU filename swap was found.

See `ASK_YOU_reference_comparison.jpg` and `ask_you_review.json`. The reference videos
are already-local citation material; no new media was acquired or admitted to training.

## Visual review and limits

All100copied clips were visually screened through sampled-frame contact sheets
`visual_00.jpg` through `visual_09.jpg`;97also have side-by-side exact-code ASL-LEX
references. THEY, DIFFERENT and CHILD lack a locally available citation video in that
reference cache; their official training identity and both primary predictions pass.
These excerpts did not reveal an obvious label swap. This is not an ASL-fluent,
frame-by-frame linguistic certification, and model agreement alone cannot prove that.

The earlier selected native streaming checks remain separate evidence. The known
shared-stream auxiliary-cadence/order sensitivity is not fixed by this direct isolated
primary audit. Full-clip correctness does not guarantee every partial sign, camera
replay, playback order or live boundary decision. No phone-camera accuracy is claimed.

## Reproduction and records

- `copied_videos.json`: all100actual destination paths and expected references.
- `main.swift` / `audit_body.swift`: native direct-primary audit; canonical reader and
  geometry/crop/model code reused. Binary:`/private/tmp/slt-primary-examples-audit`.
- `primary_native.json`: complete native per-video top5 outputs and frame counts.
- `inputs/`: fresh per-video float tensors, used identically by both backends.
- `check_primary.py`: verifies primary checkpoint identity, source/copy hashes, class
  order and PyTorch parity; writes `primary_verified.json` and `per_video.csv`.
- `training_membership.json`: repeated official-train identity/hash checks for all100.
- `source_model_hashes.json`: canonical source/model pins plus audit sources.

Native compilation succeeded with the existing unused-value warnings. PyTorch
checkpoint inference completed on CPU despite pre-existing optional-library/version
warnings; no conversion, dependency upgrade, training, model change or protected-test
access occurred. Next safe action: use the per-video evidence to inspect these known
examples; obtain ASL-fluent review if linguistic certification is required.
