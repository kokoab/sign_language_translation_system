# SLT Project Ground Truth

**Last updated:** 2026-09-13 PHT (+0800, Asia/Manila)

This file is the current state of the project, and it is the only file you must read
before changing the pipeline. It states what is true and what binds you now.

Everything that has ever happened is in `docs/ground_truth/` — 354 dated entries split
by topic. **Do not read that archive start-to-finish.** `rg <term> docs/ground_truth/`
when you need history; the map is at the bottom of this file.

After any material decision, experiment, dataset action, or validation result: append an
entry to the matching `docs/ground_truth/<topic>/log.md`, and update the section here
that it changes. If it changes nothing here, it does not belong here.

---

## Product goal

Build a fully offline, iOS-first ASL translator that retains the highest practical
accuracy while remaining viable on low-end and medium-spec iPhones. Accuracy and
generalization come first; distillation and aggressive compression are deferred until
the best accurate baseline exists.

The isolated-sign vocabulary is **100 signs**, locked. Splits must be signer-disjoint.
Target minimum is 20 training and 5 test signers, ideally 5 clips per person per sign.
The old seven-person dataset is not acceptable evidence of generalization.

## Pipeline state

| Stage | Selected artifact | Status | Best measured result |
|---|---|---|---|
| **Stage 0** extractor | Apple Vision, `active/v17/extract_v17.py` | **Frozen** | Beat MediaPipe 93.12% vs 89.95% val top-1 |
| **Stage 1** isolated | v17 Squeezeformer | **Frozen — test consumed** | 87.57% top-1 / 98.64% top-5 / 87.39% macro F1 on 1,247 Citizen test clips |
| **Stage 2** continuous | general primary/specialist CTC selector | Accepted runtime | ASLLRP contiguous 9/24 edits; local phrases 6/259; ASLLRP contextual 43/254 |
| **Stage 2** repair | `stage2_v17_transition_repair_v3/seed_1702.pth` | Passes dev gates | local/exact/contextual 6/9/43, Citizen 331/378, STEM 16/21 |
| **Stage 3** translation | `stage3_v17_t5_efficient_tiny_locked100_v1` | Bounded research module | 93.36% normalized exact on one-time synthetic test |
| **Live / streaming** | — | **Nothing promoted** | See below |

### Live and continuous work — current frontier

Earlier streaming experiments completed and **failed promotion**. A bounded Stage-1
window experiment has completed its bounded training; nothing new is promoted:

- Unified streaming CTC (rolling 32-frame, stride 4): 76.15% exact local, but 0/12 exact
  on signer-disjoint ASLLRP and it drops Citizen isolated from 95.77% to 73.28%.
- Causal continuous v2: 53% local exact, 0/12 ASLLRP. Not promoted.
- Live-matched v1 temporal adaptation: no epoch passed both retention and matched gates.
- Earlier immutable-prefix experiment: only 29.90% local confirmed exact; stable
  agreement does not guarantee correctness.
- Revisable transcription is implemented behind `--revisable-transcript`: recent words
  may change, and Finish re-decodes all retained visual features in overlapping chunks.
  The user selected this experience; holding each sign until a Stage-1 lock is not the target.
- Transition-aware gap/prefix and positive-core training each completed 12 epochs; neither
  passed promotion. Best new connected WER was 73.94%, but 153/284 reference signs were
  deleted (53.87%), and familiar/isolated retention failed. Annotated-gap emissions did
  not improve over the preceding adaptation. Report: `artifacts/reports/stage2_v17_revisable_v1/README.md`.
  Verification: 72 focused tests, 45 raw replays, exact-subset raw/cache agreement 12/12
  per model. Short-phrase first-output latency remains unresolved.

The approved Stage-1 window experiment completed all 12 seed 17111 epochs and failed
promotion. Best connected WER was 146.13% (epoch 4) versus 168.31% CTC, but deletions
increased 11→33; familiar WER 117.37%, Citizen 93.39%, and SemLex 83.64% fail retention.
Epoch 1 alone passes isolated retention but fails connected/familiar/transition gates.
No confirmation seed or Core ML export ran. The opt-in `--transcript-backend stage1-window`
backend is available for research; existing defaults remain unchanged.

The frozen comparison covers 334 recordings and 4,514 raw/replay inputs. Actual epoch 1
replay matched cached transcripts 15/15. On nine verified-boundary clips, 7/15 signs were
recognized: median delay 0.304s, p95 0.681s, misses 53.33%; full-pool latency remains
unverified. CPU/MPS parity covered 7,212 window labels/334 transcripts/16 transition
labels. Sixty-one focused tests pass. Missing independent repetitions, long holds,
background and phone coverage prevents reliability claims. Recent-tail replacement is
tested, but no useful spontaneous online corrections were demonstrated. Matched HUNGRY
diagnostics found strong window sensitivity and no OTHER-suppression explanation.
Report: `artifacts/reports/stage1_window_v17/README.md`.

Default runtime and checkpoints are unchanged. `--sequence-preview`,
`--stage2-live-checkpoint` and `--revisable-transcript` are explicit experimental flags.

## Locked decisions

1. **Never rerun or tune against the official Citizen test.** It was consumed once for
   the frozen Apple Vision + v17 Squeezeformer selection. Do not select checkpoints from
   its errors. Do not describe the 93.12% validation score as test accuracy.
2. **ASL Citizen is the sole primary training dataset**, using its official
   signer-disjoint split. Never random-split videos or mix signer identities across
   splits. Never use aspect-ratio distortion.
3. **Per-class floor is 10 train / 3 validation / 5 test signers.** Pin one exact raw
   gloss plus one ASL-LEX code per class. Do not merge numeric variants by normalized
   label. Citizen has 35/6/11 signers overall — do not claim every class has 20.
4. **Apple Vision is the locked v17 extractor.** MediaPipe stays a separately
   fingerprinted challenger; never mix its archives with Apple archives, and never
   replace Apple without new signer-disjoint evidence.
5. **v17 is a clean schema boundary.** The 96% v16 checkpoint is not compatible with v17
   features. Do not silently connect them or report v17 accuracy from a v16 model. v16
   scored 40.28% top-1 on an external Citizen audit and had reversed Apple chirality,
   fake zero-valued gap fills, and unconditional presence masks — the specific defects
   are in `docs/ground_truth/stage1-architecture/high.md`. Its aspect-distortion
   augmentation must never be carried into v17 training.
6. **PopSign is one-handed smartphone signing.** A PopSign-only model is a one-handed
   isolated-sign recognizer, never a general two-handed ASL translator. It is not the
   primary v17 dataset. Do not resume the paused audit download unless portrait
   one-handed auditing is specifically useful.
7. **Portrait is the canonical capture mode**, but the extractor must accept portrait,
   landscape, square, rotation-tagged, explicitly rotated, and mirrored input without
   geometric stretching.
8. **Do not distill or compress yet.** Mobile readiness requires measured Core ML size,
   memory, cold start, sustained latency, thermals, and accuracy on real iPhones.
   Desktop parameter count or MPS timing is not evidence.
9. **Do not use standard YOLO Pose** as the hand extractor — its keypoints lack the 21
   finger joints ASL needs.
10. **CTC blank is index 0.** `OTHER` is a Stage-2-only class (101 nonblank, `OTHER` at
    index 101) removed after collapse. Blank is a no-emission symbol, not a transition
    class. Never label unverified surrounding signing as blank.
11. **Synthetic and generated phrases are training-only.** They must never enter
    validation, sealed testing, or model-selection truth. Rejected synthetic motion is
    barred from every dataset and renderer.
12. **Rylo is a reference, not a corpus.** Its hosted `.pose` outputs may be qualitative
    comparators only — never training, validation, or test labels. Do not scrape its
    media. SignBank+ is CC BY-NC 4.0 notation/text, not continuous signer video.
13. **ASL Citizen is research/noncommercial data.** It cannot be assumed to license a
    commercial shipping model.
14. Mirror/TTA must swap hand indices `0-20 <-> 21-41` with X-axis sign flips on
    coordinate and motion channels.

## v17 extractor state

**Location:** `active/v17/` — `schema_v17.py`, `geometry_v17.py`, `extract_v17.py`,
`audit_v17.py`, plus `src_v17/` wrappers and `test/test_v17_extractor.py`.

Feature tensor: **`[32 frames, 61 nodes, 5 channels]`**, float16.

- Nodes: 21 left hand, 21 right hand, 15 face samples, 4 upper body.
- Channels: body-relative X, body-relative Y, relative log-scale depth proxy, binary
  presence, confidence.
- Missing spatial/depth/confidence values are exactly zero.
- Every archive embeds a schema fingerprint; a config mismatch is rejected on load.
- `window_stride` is part of the Stage-2 feature config. Stride-32 keeps the original
  fingerprint; stride-8 archives get `44e9f97c67a003c4`.

Default extraction: 32 output frames from at most 96 uniformly sampled source frames;
long side capped at 1280px without upscaling or distortion; body/face every 8 sampled
frames, hands every frame; minimum joint confidence 0.15; hand gaps up to 3 frames and
auxiliary gaps up to 16 interpolated only when bounded by real observations;
leading/trailing inactivity trimmed with 2 frames context; at least 2 hand frames
required.

Orientation: OpenCV honors rotation metadata; `--rotation 0|90|180|270` overrides it;
`--input-mirrored` flips stored mirrored pixels exactly once before Vision. Vision always
receives upright, unmirrored pixels. Coordinates use isotropic geometry from the longest
image side.

## Data and storage state

- Broader data-sufficiency goal is blocked pending corpus access, exact-variant
  review and independent phone recordings. See
  `artifacts/reports/continuous_asl_acquisition_20260912/DATA_NEEDED.md`.
  Three corpus access inquiries are prepared in the same report folder as
  `ACCESS_REQUESTS.md`; none sent. New MoLo interview and Daily Moth EAF checks
  do not establish full gloss targets;
  do not equate English translation tiers with sign annotations.
- Continuous ASL acquisition completed 2026-09-13 under
  `data/local/continuous_asl_acquisition_20260912/`; inventory and verification:
  `artifacts/reports/continuous_asl_acquisition_20260912/README.md`.
  Épée provides 1,200 timed sequences from six source signer IDs, with 68/100 exact
  raw-label matches, but is MediaPipe-only with no raw video; it is incompatible
  with the Apple Vision input contract. MoLo adds 17.29 minutes of original raw
  video, two annotated signers (1,517 hand annotations), and verified signer crops.
  RIT adds a 6.5-second public fluent-coded sample. These remain outside training
  pending annotation completeness, alignment and exact visual-variant review;
  unannotated/OOV spans are not background. No phone-generalization claim follows.
- Raw/local datasets go only under `data/local/`. Generated reports under
  `artifacts/reports/`, disposable outputs under `artifacts/generated/`.
- Never commit or delete datasets, checkpoints, reports, or metrics without an explicit
  request.
- PopSign is ~1.1 TB in full — never download wholesale. One sign/split archive at a
  time, check free space before transfer, preserve license provenance.
- Free space was ~13 GiB at last measurement (2026-08-09).

## Environment

- Host: macOS on Apple Silicon. Project Python: **`venv/bin/python`** (3.9).
- System Python lacks the PyObjC Vision/Quartz bridge — real extraction and tests must
  run through `venv/bin/python`.
- Isolated research envs: `artifacts/generated/mobileclip2_env`,
  `artifacts/generated/movinet_env` (CPU-only; TF Metal unsupported for its Conv3D graph).
- MPS is correct for training but wrong for token-by-token live decoding: live Stage 3
  measured 137.96 ms on CPU versus 1,678.02 ms on MPS. Live Stage 3 defaults to CPU.

## Immediate next actions

1. Review the completed revisable-transcription report. The user has chosen continuous
   transcription with provisional revisions. Further research must address temporal
   alignment, missed signs, and latency; do not repeat the completed gap/prefix/core
   experiments or substitute hold-until-lock behavior. Use independently checked
   boundary/hold/repetition evidence before claiming transition robustness.
2. Collect a new portrait-iPhone signer-disjoint evaluation set. This is the only valid
   dataset for measuring future model changes without contaminating the consumed Citizen
   test.
3. Measure Core ML package size, memory, cold start, sustained latency, and thermals on
   real low/medium-spec iPhones.
4. Design UNKNOWN / out-of-vocabulary rejection and evaluate it on independently held-out
   nonsign clips before presenting the classifier as an app feature.
5. Materialize the 111 frozen ASL STEM Wiki spans (participant-disjoint split) and run a
   bounded Stage-2 adaptation with existing replay and retention gates.
6. Obtain ASL-fluent review of the raw-gloss/ASL-LEX mappings and frequent confusions.

## Known boundary

A v17 classifier is trained with a one-time official signer-disjoint test result of
87.57% top-1. **Not established:** independent portrait-iPhone accuracy, UNKNOWN
rejection behavior, ASL variant review, real-device performance, and any continuous or
conversational translation claim. Stage 3 is a bounded synthetic/reviewed renderer, not
a general ASL translator. Do not claim production mobile readiness, continuous sign
recognition, or end-to-end conversational translation.

---

## Archive map

Full history: **`docs/ground_truth/`** — 354 entries, 805 KB, nothing deleted.
Search it with `rg`, do not read it. Detail and conventions: `docs/ground_truth/MAP.md`.

| Topic | high | log | Covers |
|---|---:|---:|---|
| `stage1-architecture/` | 19 | 94 | Encoder, extractor bakeoffs, component ablation ladder |
| `text-to-sign/` | 6 | 67 | SignWriting, avatar rendering, motion generation |
| `live-streaming/` | 10 | 45 | Reel path, streaming CTC, commitment and latency |
| `data-sources/` | 8 | 33 | Acquisition, licensing, admission audits, split policy |
| `stage2-ctc/` | 9 | 17 | Continuous recognition, CTC contracts, selectors |
| `signing-voice/` | 2 | 17 | Style transfer, coarticulation, transition synthesis |
| `mobile-deployment/` | 7 | 11 | Core ML export, orientation contract, iPhone/Flutter |
| `stage3-translation/` | 2 | 3 | Gloss-to-English models and their gates |
| `capstone-paper/` | 0 | 4 | Paper revisions, benchmarks, repository hygiene |

`high.md` holds the dated evidence behind the constraints stated above. `log.md` holds
measured results and rejected approaches — `rg` it before re-running an experiment, so
you do not repeat one that already failed.
