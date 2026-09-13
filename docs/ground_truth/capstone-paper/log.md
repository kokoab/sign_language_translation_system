# capstone-paper — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

4 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-03 20:30 PST — Capstone extractor/architecture claims corrected and latency benchmarked

The Capstone revision package was re-audited after the user challenged the extractor
and part-wise descriptions. The frozen local evidence confirms Apple Vision, not
MediaPipe, won the engineering decision: median active output coverage tied at 87.50%,
while Apple had higher pre-trim source detection (42.65% versus 38.54%), slightly higher
post-trim detection (85.24% versus 84.73%), 0.678 versus 1.230 seconds/clip extraction,
and 93.12% versus 89.95% matched validation top-1. MediaPipe's 82.24% versus 79.52%
mean output coverage came after trimming/interpolation and must not be presented as
superior genuine active-hand detection. The report and extractor chart now foreground
the tied median output and Apple's source-detection, speed, and classifier wins while
retaining MediaPipe's steadier bone/denser-output proxies as secondary diagnostics.
Official Apple/Google sources describe both live APIs but provide no controlled
cross-framework comparison, so the local frozen bakeoff remains decisive.

Direct inspection of the exact current Reel checkpoint
`artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth`
confirms its landmark configuration is still `temporal_encoder=partwise_global`,
`part_depth=1`, `dim=256`, `depth=4`. The correct name for the whole Stage-1 classifier
is unified multimodal Squeezeformer: part-wise+global is its landmark submodel, joined
to the RGB hand-crop temporal Squeezeformer and learned fusion head.

A new matched batch-one PyTorch CPU single-thread benchmark used one real
`[1,32,61,5]` validation tensor, 20 warmups, three rotated rounds, and 300 timed
predictions per architecture. Median/p90 model-only latency was 11.42/11.86 ms graph
replacement, 7.35/7.80 ms wider flat d384, 4.90/5.09 ms flat d256, and 6.50/6.96 ms
part-wise+global. The raw result is
`artifacts/reports/capstone1_v17_revision_checklist_v1/architecture_latency_benchmark.json`.
The architecture chart now shows both controlled accuracy and this matched latency.
A new Stage-1 training figure reads the actual 30-epoch history from the final
phrase/activity adaptation result and plots recorded loss, validation/context-crop
top-1, learning rate, and selected epoch 26; it does not synthesize curves. The report
also separates model-only latency from live behavior: 16.31 ms Core ML landmark
proposal, 8.64 ms unified Core ML with precomputed embeddings, 359.00 ms baseline full
visual verification, 304.15 ms cached experimental verification, and 0.67 seconds
median committed candidate duration. Chart regeneration, link validation, terminology
scan, compilation, and `git diff --check` pass. No runtime model/checkpoint or protected
test data was changed or accessed.

## 2026-09-03 19:59 PST — Capstone 1 paper revision package grounded in current v17 evidence

The 69-page Capstone 1 PDF at
`/Users/frnzlo/Downloads/For Checking ATLAS (September) (1).pdf` was rendered and
audited against the current v17 implementation and existing experiment reports. The
recommended concise title is “ATLAS: A Squeezeformer-Based System for Sign Language
Recognition and English Translation.” The manuscript should describe cooperating
extraction, recognition, streaming-control, completion, and English-output modules
rather than preserve the obsolete required three/four-stage pipeline. The current Reel
default has the CTC arbiter and targeted MediaPipe mouth verifier disabled; mouth
markers are supplemental, and the Finish button plus held ten-finger gesture are the
two completion controls. The report also corrects the raw v17 tensor to
`[B, 32, 61, 5]`, identifies RGB as cropped-hand evidence, and distinguishes persistent
display landmarks from model observations.

The new private authoring package is
`artifacts/reports/capstone1_v17_revision_checklist_v1/README.md`. It contains a
paper-wide checklist, copy-ready purpose/objectives/scope text, a Mermaid architecture
flow, and matched-data result tables for recognition, hand-crop RGB versus skeletal
landmarks, Apple Vision versus MediaPipe, controlled Squeezeformer variants, contextual
adaptation, and English rephrasing. Five PNG/SVG charts were regenerated from recorded
metrics by `make_charts.py`. Dataset brands are deliberately omitted from this
user-facing package and replaced with “100-gloss corpus”; the authors must insert final
provenance/citations before academic submission. No protected test was rerun and no
new accuracy experiment was needed because matched comparisons already existed.

`STAGE3_HUMAN_EVALUATION.md` provides a 30-item blinded rating sheet: the complete 26
controlled held-out long examples plus four short examples, with semantic adequacy,
grammar, faithfulness, and overall-acceptability rubrics. It intentionally contains no
fabricated human ratings. The current automatic English results remain 94.00% exact /
99.02 chrF++ within the locked 100-gloss scope and 100% exact / 99.57 chrF++ on the 26
controlled long examples; these are prepared-reference metrics, not general
translation evidence. Chart regeneration, local image-link validation, forbidden
dataset-name scan, and `git diff --check` pass. No runtime code, model, data, or
checkpoint was changed for this documentation task.

## 2026-08-24 21:21 PST — v17 source and evidence published to GitHub

The full eligible reorganization, v17 source, iOS projects, tests, documentation,
manifests, and compact reports were committed as
`090d3149db4f0387bdcf55b6d5e1924b8413c0f2` (`Add v17 mobile SLT pipeline and
evidence`) and pushed successfully to
`https://github.com/kokoab/sign_language_translation_system.git`, branch `master`.
The push was a fast-forward from `2a6b743` and local `master` was configured to track
`origin/master`. The separate repository-default `main` branch was not merged,
rewritten, or otherwise modified.

The commit author is the user's configured
`kokoab <francis.batiancela@intechsive.com>` and contains no co-author trailer. The
GitHub publication excludes the large/local paths documented in `.gitignore`; those
assets remain present on this Mac for iPhone builds and experiments.

## 2026-08-24 21:19 PST — physical-iPhone deployment guide and GitHub hygiene

The physical installation procedure is now documented at
`mobile_benchmark/OrientationBenchmarkV17/DEPLOY_TO_IPHONE.md`. It covers exact local
model prerequisites, Apple ID/Personal Team setup, Developer Mode, automatic signing,
unique bundle identifiers, physical-device selection, first installation, file-video
inference, JSON export, physical benchmark discipline, and common signing/device/model
failures. The project remains iOS 17.0+, file-picker based, and not a live-camera app.

Before GitHub publication, the worktree was audited rather than staged blindly. Three
local files exceed GitHub's 100 MB per-file limit, and the repository contains roughly
20 GB of datasets, model assets, Core ML packages, checkpoints, generated build trees,
tool environments, archives, and mobile build products. `.gitignore` now excludes
those reproducible/local products plus generated media and mobile build output while
retaining source, documentation, manifests, and compact evidence reports. Two
reproducible Stage-2 plan JSON files above 10 MB are also excluded. The eligible
untracked publication set is approximately 69.2 MiB across 1,112 files, with an 8.6
MiB largest file. The configured Git author remains the user's
`kokoab <francis.batiancela@intechsive.com>`; no co-author trailer will be added.
