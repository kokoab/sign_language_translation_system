# Implementation Plan: Replace MMPoseInferencer with Direct model.test_step Batch Inference

## Goal

Replace the sequential `MMPoseInferencer` generator (which runs RTMDet + RTMW = 96 forward passes per 48-frame video) with direct `model.test_step()` batch inference (~2-4 forward passes total per video). This is the **only remaining path** to faster extraction — GPU utilization is already at 99%.

## Why This Matters

- **Current**: `list(inferencer(batch_frames))` looks like batching but is actually a Python generator that processes frames one-by-one (1 RTMDet + 1 RTMW per frame = 96 forward passes for 48 frames)
- **Target**: True GPU batching via `model.test_step()` — process all 48 frames in 1-2 forward passes each for RTMDet and RTMW
- **Expected speedup**: 5-10x on the pose estimation bottleneck (the dominant GPU cost)

## Environment

- **mmpose==1.3.2, mmengine==0.10.7, mmcv==2.2.0, mmdet==3.3.0**
- **PyTorch 2.2.0, CUDA 12.1, Python 3.10**
- **Docker image**: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`
- **Hardware**: 8x RTX 4090 (24GB each), 64 workers (8 per GPU)
- **RTMW-l checkpoint**: `models/rtmw_l_wholebody.pth`
- **RTMW-l config**: `rtmw-l_8xb320-270e_cocktail14-384x288` (from mmpose metafile registry)

---

## Step 0: Deep Research (MANDATORY — Do This First)

Before writing ANY code, you MUST read and understand the mmpose source code on the running instance. Do NOT guess APIs — verify them.

### 0.1 Understand MMPoseInferencer internals

```bash
# Find the inferencer source
python3 -c "from mmpose.apis import MMPoseInferencer; import inspect; print(inspect.getfile(MMPoseInferencer))"

# Read the __call__ method — this is the generator that processes frames
# Look for: how it calls the detector, how it passes bboxes to the pose model,
# the data format it constructs for each frame
```

**What to look for:**
- How `MMPoseInferencer.__call__()` iterates over inputs
- How it calls `self.detector` (RTMDet) — what method? what input format?
- How it calls `self.pose_estimator` (RTMW-l) — what method? what input format?
- The data structure passed to pose estimation (likely `dict` with 'inputs', 'data_samples')
- How bounding boxes from RTMDet are formatted and passed to RTMW

### 0.2 Understand model.test_step

```bash
# Find the pose model's test_step
python3 -c "
from mmpose.apis import init_model
# Use the EXACT config and checkpoint we use
model = init_model(
    'configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py',
    'models/rtmw_l_wholebody.pth',
    device='cuda:0'
)
import inspect
print(type(model))
print(inspect.getfile(type(model)))
# Look at test_step signature
help(model.test_step)
"
```

**What to verify:**
- `model.test_step(data_batch)` — what is `data_batch`? A dict? A list of dicts?
- Does it expect preprocessed tensors or raw images?
- What preprocessing does MMPoseInferencer do that we'd need to replicate?
- What does `test_step` return? How to extract keypoints + scores?

### 0.3 Understand RTMDet batch inference

```bash
# Find how the detector works
python3 -c "
from mmpose.apis import MMPoseInferencer
inf = MMPoseInferencer(pose2d='rtmw-l_8xb320-270e_cocktail14-384x288', device='cuda:0')
print(type(inf.detector))
import inspect
print(inspect.getfile(type(inf.detector)))
help(inf.detector.test_step)
"
```

**What to verify:**
- RTMDet's `test_step` input format
- How to batch multiple frames for detection
- Output format: how bboxes are structured

### 0.4 Understand inference_topdown (fallback approach)

```bash
# This is mmpose's built-in batch-aware function
python3 -c "
from mmpose.apis import inference_topdown
import inspect
print(inspect.getsource(inference_topdown))
"
```

**What to verify:**
- Does `inference_topdown` accept a batch of images + bboxes?
- Or does it process one image at a time with multiple bboxes?
- What preprocessing does it handle internally?

### 0.5 Verify config loading

```bash
# Verify we can load the config from mmpose's metafile registry
python3 -c "
from mmpose.apis import init_model
# Try metafile name (how MMPoseInferencer loads it)
model = init_model('rtmw-l_8xb320-270e_cocktail14-384x288', 'models/rtmw_l_wholebody.pth', device='cpu')
print('Model loaded OK')
print('Input size:', model.cfg.codec.get('input_size', 'unknown'))
print('Type:', model.cfg.model.type)
"
```

---

## Step 1: Implementation

### Option A: Direct model.test_step (preferred — true batch inference)

Create a new function `_detect_pass_rtmw_batched()` in `src/extract.py`. This replaces the body of `_detect_pass_rtmw()`.

**Architecture:**

```
48 frames
    |
    v
[RTMDet batch inference] -- 1-2 forward passes (batch_size=24 or 48)
    |
    v
48 bounding boxes (one per frame, best person detection)
    |
    v
[Crop + resize + normalize each frame to 384x288 using bbox]
    |
    v
[RTMW-l batch inference via test_step] -- 1-2 forward passes (batch_size=24 or 48)
    |
    v
48 sets of 133 keypoints with scores
    |
    v
[Extract hand + face keypoints, same logic as current code]
```

**Critical implementation details:**

1. **RTMDet batching**: Build a proper `data_batch` dict/list that RTMDet's `test_step` expects. This likely involves:
   - Preprocessing frames (resize, normalize) according to RTMDet's pipeline
   - Wrapping in `DetDataSample` objects
   - Collating into a batch

2. **Bbox selection**: From RTMDet output, select the highest-confidence person bbox per frame. If no person detected, use full-frame bbox `[0, 0, w, h]`.

3. **RTMW-l batching**: For each frame + bbox pair:
   - Crop the person region from the frame using the bbox
   - Resize to 384x288 (RTMW-l's expected input size)
   - Normalize according to RTMW-l's data pipeline
   - Build `PoseDataSample` with bbox info
   - Collate all 48 into a batch
   - Run `model.test_step(batch)`

4. **Keypoint extraction**: From RTMW-l output, extract the 133 keypoints + scores per frame. The keypoint indices are:
   - Body: 0-16
   - Feet: 17-22
   - Face: 23-90
   - Left hand: 91-111 (21 keypoints)
   - Right hand: 112-132 (21 keypoints)

**Pseudo-code (verify all APIs against actual source before implementing):**

```python
def _detect_pass_rtmw_batched(frames, cfg, w, h):
    """
    True batch inference: RTMDet + RTMW-l with GPU batching.

    Args:
        frames: list of numpy arrays (BGR, uint8), already resized to <=512px
        cfg: PipelineConfig
        w, h: original video dimensions (for coordinate normalization)

    Returns:
        Same as _detect_pass_rtmw: (l_seq, r_seq, l_valid, r_valid, face_seq, face_valid, diag)
    """
    inferencer = _get_rtmw_inferencer()
    if inferencer is None:
        return None

    # Access the underlying models
    detector = inferencer.detector        # RTMDet
    pose_model = inferencer.pose_estimator  # RTMW-l

    # ── Phase 1: Batch person detection ──
    # BUILD data_batch for RTMDet — VERIFY THIS FORMAT against actual source
    # ... (format depends on what test_step expects — Step 0 research)

    det_results = detector.test_step(det_data_batch)

    # Extract best person bbox per frame
    bboxes = []
    for det_result in det_results:
        pred = det_result.pred_instances
        if len(pred.bboxes) > 0:
            # Filter by confidence, take highest score
            best_idx = pred.scores.argmax()
            bboxes.append(pred.bboxes[best_idx].cpu().numpy())
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))

    # ── Phase 2: Batch pose estimation ──
    # BUILD data_batch for RTMW-l — VERIFY THIS FORMAT against actual source
    # Each item needs: cropped+resized image tensor, bbox metadata
    # ... (format depends on what test_step expects — Step 0 research)

    pose_results = pose_model.test_step(pose_data_batch)

    # ── Phase 3: Extract keypoints (same logic as current code) ──
    # ... extract hand/face keypoints from pose_results
    # ... build l_seq, r_seq, face_seq etc.
    # ... normalize coordinates by (w, h)
```

### Option B: inference_topdown (fallback — simpler but possibly less batching)

If `model.test_step` requires complex data pipeline replication, use `inference_topdown` as a middle ground:

```python
from mmpose.apis import inference_topdown

# For each frame, pass the frame + bbox
# inference_topdown may handle preprocessing internally
results = inference_topdown(pose_model, frame, bboxes_for_frame)
```

**Tradeoff**: `inference_topdown` may process one frame at a time (with multiple bboxes per frame), so it won't batch across frames. But it handles all preprocessing, so it's safer to implement.

### Option C: Hybrid — batch RTMDet only, use inference_topdown for pose

This gives the biggest win (RTMDet is called 48 times currently) with lower risk:

1. Batch all 48 frames through RTMDet in 1-2 passes
2. Use `inference_topdown` for pose estimation (one frame at a time, but no RTMDet overhead)

This halves the forward passes (48 RTMDet eliminated) with minimal code change.

---

## Step 2: Integration with Config Flag

Add a config flag for easy A/B testing and rollback:

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    use_batched_rtmw: bool = True  # True = new batch inference, False = old MMPoseInferencer
```

In `_detect_pass_rtmw()`:

```python
def _detect_pass_rtmw(frames, cfg, w, h):
    if cfg.use_batched_rtmw:
        return _detect_pass_rtmw_batched(frames, cfg, w, h)
    # ... existing sequential code ...
```

---

## Step 3: Verification Test

Before deploying, run a comparison test on ~20 videos:

```python
# test_batch_vs_sequential.py
"""
Compare old (MMPoseInferencer) vs new (batch test_step) extraction.
Verifies that keypoint outputs are numerically close (within floating point tolerance).
"""

import numpy as np
from extract import _detect_pass_rtmw, _detect_pass_rtmw_batched, CFG

def test_equivalence(video_path, threshold=0.01):
    # Load and sample frames from video
    frames = load_and_sample(video_path)
    w, h = get_dimensions(video_path)

    # Run old path
    CFG.use_batched_rtmw = False
    old_result = _detect_pass_rtmw(frames, CFG, w, h)

    # Run new path
    CFG.use_batched_rtmw = True
    new_result = _detect_pass_rtmw_batched(frames, CFG, w, h)

    # Compare keypoint arrays
    if old_result is None or new_result is None:
        print(f"One path returned None: old={old_result is None}, new={new_result is None}")
        return

    old_l, old_r, _, _, old_face, _, _ = old_result
    new_l, new_r, _, _, new_face, _, _ = new_result

    # Keypoints should be very close (small differences from preprocessing rounding)
    for name, old_arr, new_arr in [('left', old_l, new_l), ('right', old_r, new_r), ('face', old_face, new_face)]:
        if len(old_arr) != len(new_arr):
            print(f"WARNING: {name} sequence length mismatch: {len(old_arr)} vs {len(new_arr)}")
            continue
        if len(old_arr) == 0:
            continue
        old_np = np.array(old_arr)
        new_np = np.array(new_arr)
        max_diff = np.abs(old_np - new_np).max()
        mean_diff = np.abs(old_np - new_np).mean()
        print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < threshold, f"{name} keypoints diverged: max_diff={max_diff}"

    print("PASS: outputs are equivalent within tolerance")
```

**Run on the instance:**
```bash
python3 test_batch_vs_sequential.py
```

---

## Step 4: Deployment Checklist

1. SSH into instance: `ssh -p 10650 root@217.138.104.104`
2. Backup current extract.py: `cp src/extract.py src/extract.py.bak`
3. Upload new extract.py: `scp -P 10650 src/extract.py root@217.138.104.104:/workspace/src/extract.py`
4. Run verification test (Step 3) on ~20 videos
5. If PASS: run full extraction with `python3 src/extract_do.py --workers 64`
6. Monitor first 500 videos: check save rate (should be >= 50%) and speed (should be >> 360/min)
7. If save rate drops below 30%: STOP and rollback

---

## Step 5: Rollback Plan

If anything goes wrong:

```bash
# Immediate rollback — restore backup
cp src/extract.py.bak src/extract.py

# Or use config flag (if integrated)
# Edit extract.py: CFG.use_batched_rtmw = False
```

---

## Common Pitfalls (From This Project's History)

These are mistakes that were already made in this project. Do NOT repeat them:

| Pitfall | What Happened | How to Avoid |
|---------|---------------|--------------|
| **Resize too small** | 384px resize killed RTMDet person detection (save rate 71% -> 5%) | Keep frame resize at max 512px. RTMW internally resizes crops to 384x288, but RTMDet needs higher-res input |
| **RTMDet-once bbox caching** | Caching first-frame bbox for all frames dropped save rate from 27% to 14% | Run RTMDet on every frame (but BATCH them) |
| **OOM permanently disabling RTMW** | A global `_rtmw_available = False` flag meant one OOM killed all future videos for that worker | Never permanently disable — return None for that video, retry on next |
| **Assuming MMPoseInferencer batches** | `list(inferencer(batch_frames))` looks like batching but is a sequential generator | This is exactly what we're fixing |
| **Untested bbox format** | Passed bboxes to MMPoseInferencer in wrong format, silently produced bad results | Always verify data formats against source code, test on small batch first |
| **Unnecessary restarts** | Each restart re-processes large failing files first (sorted by size desc) | Never restart a running extraction unless there's a code bug |

---

## Implementation Order

1. **Do Step 0 COMPLETELY** — read all source code, understand all APIs
2. **Decide Option A/B/C** based on what Step 0 reveals about API complexity
3. **Implement in extract.py** with config flag
4. **Run verification test** (Step 3)
5. **Deploy** (Step 4)

**Time budget for implementation session**: The implementer should spend at least 30 minutes on Step 0 research before writing any code. Rushing past research is how the previous mistakes (resize, bbox caching, format errors) happened.

---

## Key Files to Read on the Instance

```bash
# These are the critical source files to understand before implementing:

# 1. MMPoseInferencer — how it orchestrates detection + pose
find /opt/conda/lib/python3.10/site-packages/mmpose/apis/ -name "*.py" | head -20

# 2. RTMDet model — test_step and data format
find /opt/conda/lib/python3.10/site-packages/mmdet/models/ -name "*.py" | grep -i rtm

# 3. RTMW/RTMPose model — test_step and data format
find /opt/conda/lib/python3.10/site-packages/mmpose/models/ -name "*.py" | grep -i rtm

# 4. Data transforms — preprocessing pipeline
find /opt/conda/lib/python3.10/site-packages/mmpose/datasets/transforms/ -name "*.py"

# 5. inference_topdown source
python3 -c "from mmpose.apis import inference_topdown; import inspect; print(inspect.getsource(inference_topdown))"
```
