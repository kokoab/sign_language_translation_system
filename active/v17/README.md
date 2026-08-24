# SLT v17 extractor

v17 is the orientation-safe Apple Vision feature pipeline for the new iOS-first,
100-sign ASL Citizen baseline. It replaces v16 extraction, not the trained v16 model.
The current v16 checkpoint cannot consume v17 features because the channel semantics
and missing-data rules deliberately changed.

Apple Vision is the selected hand extractor, not an untested default. A frozen
300-clip quality bakeoff and equal full-corpus Stage 1 run compared it with the official
MediaPipe Hand Landmarker at 0.30 and 0.50 thresholds. Apple reached 93.12% top-1,
99.47% top-5, and 92.53% macro F1 on the signer-disjoint validation split; the strongest
MediaPipe run reached 89.95%, 97.35%, and 89.81%. Apple was also faster on the measured
M4 host and recovered overlapping two-hand signs more reliably. The official test split
was not used for this selection. After model selection was frozen, the Apple model was
evaluated once on all 1,247 clips from the 11-signer Citizen test partition: 87.57%
top-1, 98.64% top-5, and 87.39% macro F1. This test result must not be used for further
tuning. Full evidence lives in `PROJECT_GROUND_TRUTH.md` and
`artifacts/reports/EXTRACTOR_BAKEOFF_V17.md`.

## Feature contract

Every isolated clip becomes a float16 tensor with shape `[32, 61, 5]`:

| Axis | Meaning |
| --- | --- |
| 61 nodes | 21 left-hand + 21 right-hand + 15 face + 4 upper-body |
| Channel 0 | body-relative X |
| Channel 1 | body-relative Y |
| Channel 2 | relative log-scale depth proxy |
| Channel 3 | binary presence |
| Channel 4 | Apple Vision confidence |

Each `.v17.npz` includes the tensor, extractor diagnostics, metadata, full schema,
and a schema fingerprint. Loading fails on a mismatched extractor configuration.
Missing nodes always have zero XY, depth, and confidence.

## Orientation contract

The extractor converts every source into upright, unmirrored pixels before calling
Apple Vision. OpenCV video rotation metadata is honored by default. For incorrectly
tagged or continuously rolled files, pass any finite clockwise correction with
`--rotation`, such as `--rotation 37.5`. Non-right-angle transforms expand the canvas
so no source pixels are cropped or stretched. Pass
`--input-mirrored` only when the stored pixels themselves are mirrored. Portrait,
landscape, and square videos retain their aspect ratio; a 1280-pixel long-side cap and
a deterministic 96-frame reservoir fallback keep high-resolution or incorrectly tagged
PopSign clips memory-safe.

Do not add aspect-ratio distortion as training augmentation. Geometric augmentation
must use valid rotations, crops, scale, translation, or mirroring with anatomical
left/right hand swapping.

## Setup and commands

The checked local environment uses Python 3.9 and PyObjC Vision/Quartz 11.1:

```bash
source venv/bin/activate
python active/v17/extract_v17.py path/to/video.mp4 \
  --output data/local/ASL_landmarks_v17/example.v17.npz
```

Build and acquire the exact Citizen100 corpus without downloading the full archive:

```bash
venv/bin/python scripts/build_citizen100_v17.py
venv/bin/python scripts/download_citizen100_v17.py --dry-run
venv/bin/python scripts/download_citizen100_v17.py --workers 4
venv/bin/python active/v17/audit_citizen100_raw.py
```

For a class-directory dataset:

```bash
venv/bin/python active/v17/extract_v17.py data/local/asl_citizen_v17/train \
  --output data/local/asl_citizen_landmarks_v17/train
```

The batch layout is `split/GLOSS/video.v17.npz`. Extraction is sequential by design:
Apple Vision request objects are reused, all due requests share one handler, and the
implementation avoids fragile Python multiprocessing around Objective-C objects.

## Validation

```bash
venv/bin/python -m unittest test.test_v17_extractor -v
venv/bin/python active/v17/audit_v17.py \
  data/local/ios100_audit/landmarks_v17
```

The tests cover exact rotate/unrotate and mirror/unmirror equivalence using real
Apple Vision detections, chirality assignment, isotropic portrait/landscape geometry,
bounded interpolation, binary masks, zero-valued missing nodes, resolution limiting,
and schema-enforced save/load.

MediaPipe remains a separately fingerprinted research challenger in
`extract_mediapipe_v17.py`; its archives cannot be mixed with Apple archives. It is not
the production v17 default. The bakeoff can be reproduced with
`extractor_bakeoff_v17.py`, and the visual disagreement audit with
`render_extractor_bakeoff_v17.py`.

The frozen class contract is `citizen100_manifest.json`. Each canonical class maps to
exactly one Citizen raw gloss and ASL-LEX code. Dotted fingerspelling is rejected when
an eligible lexical sign exists, and numeric variants are never merged automatically.

## Stage 1 training

`model_v17.py` accepts only the archived five-channel tensor. It derives masked XYZ
velocity/acceleration and valid hand-shape distances inside the model, so missing
landmarks never create motion spikes and mobile preprocessing cannot drift from Python.
`train_stage_1_v17.py` reads the official `train`, `val`, and `test` directories without
creating a random split and applies `rejections.csv` explicitly.

Run the focused checks and an optimizer/checkpoint smoke test:

```bash
venv/bin/python -m unittest test.test_v17_stage1 -v
venv/bin/python active/v17/train_stage_1_v17.py \
  --smoke --output artifacts/generated/v17_stage1_smoke
```

Start the accuracy-first baseline without reading the held-out test split:

```bash
venv/bin/python active/v17/train_stage_1_v17.py \
  --output artifacts/models/stage1_v17
```

The first accuracy baseline is `dim=256`, `depth=4` (about 6.5 million parameters),
because v16's controlled capacity check favored 256 over 384 and this corpus has only
1,475 usable training clips. Capacity remains configurable and should be increased only
if signer-disjoint validation improves. The official test gate has already been used
once for the frozen Apple model and must not be reopened for tuning. Distillation,
quantization, and Core ML optimization remain later evidence gates.

## MobileCLIP2 RGB challenger

The controlled frozen-RGB challenger is implemented separately in
`extract_mobileclip2_v17.py`, `model_mobileclip2_v17.py`, and
`train_stage_1_mobileclip2_v17.py`. It uses the official MobileCLIP2-S0 `dfndr2b`
image tower on 16 upright, aspect-preserving letterboxed frames from the same Apple
hand-activity interval. The isolated Python 3.10 requirements are pinned in
`mobileclip2_requirements.txt`; do not install them into the Apple Vision venv.

The original full-frame, globally pooled run reached only 39.68% top-1, 72.49% top-5,
and 36.46% macro F1. That result rejects only that particular frozen-global design; it
does not reject RGB or MobileCLIP2 as a whole. A corrected frozen hand-aware branch uses
left-hand, right-hand, and union crops selected by Apple Vision, explicit missing-view
masks, crop trajectories, view attention, and a sign-specific temporal head. It reached
70.37% top-1, 91.27% top-5, and 69.13% macro F1, confirming that hand-scale pixels were
materially lost by the first experiment.

The completed spatial experiment cached MobileCLIP2 FastViT maps before global pooling,
applied temporal shift independently to each valid view, and fine-tuned the late visual
projection with the sign-specific temporal head. It reached 70.63% top-1. A
zero-initialized gated feature residual reached 93.92% for canonical seed 1701, but the
net three-clip gain was not significant and averaging five diagnostic seeds returned to
Apple's 93.12%. The added 48-view image path is therefore not a selected mobile runtime
dependency. The one-time Citizen test evaluation is complete and must not guide future
tuning.

## Sign-specialized MoViNet challenger

`train_stage_1_movinet_v17.py` uses the official pretrained MoViNet-A0 backbone as a
video model, not as a per-frame image encoder. Shared weights process left-hand,
right-hand, and union/context crop sequences. The model also consumes explicit
missing-view masks, crop trajectories, and anatomical view identity. A visual-only
auxiliary loss teaches sign discrimination while a zero-initialized residual jointly
fuses the video representation with frozen Apple pooled features/logits. Consequently,
fusion begins bit-exactly at the Apple baseline and RGB must learn useful corrections.

The isolated environment and official checkpoint are under
`artifacts/generated/movinet_env` and `artifacts/model_assets/movinet`; dependencies are
pinned in `movinet_requirements.txt`. TensorFlow Metal 1.2 cannot execute Model Garden's
XLA grouped Conv3D graph, so the reproducible local training path is CPU-only. CUDA on
Linux is the practical route for a full fine-tune. A focused smoke command is:

```bash
artifacts/generated/movinet_env/bin/python \
  active/v17/train_stage_1_movinet_v17.py \
  --smoke --max-train-batches 2 --batch-size 4 \
  --output artifacts/generated/stage1_movinet_v17_smoke
```

The Citizen test split is structurally rejected. Any future comparison uses only the
existing train/validation evidence and then a new independent portrait-iPhone set; the
already-consumed official test cannot be reopened for selection.

## Migration boundary

The next model must be trained from scratch on v17 archives with ASL Citizen's official
signer-disjoint splits. Until then:

- Use v16 extraction + v16 checkpoint for the old model.
- Use v17 extraction for the frozen ASL Citizen 100-class corpus and future v17 model.
- Never rename a v17 archive to look like a v16 `.npy` feature file.
