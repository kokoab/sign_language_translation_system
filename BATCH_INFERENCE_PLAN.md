# Batch Inference + Bbox Stabilization Implementation Plan

Replace the sequential `MMPoseInferencer` generator with true GPU-batched `test_step()` calls, and fix the root cause of the 46.5% jitter rate by stabilizing bounding boxes before pose estimation.

**Status**: PENDING APPROVAL — no code has been changed.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Research Findings](#2-research-findings)
3. [Architecture](#3-architecture)
4. [Implementation Details](#4-implementation-details)
5. [Enhancement: Bbox Temporal Stabilization](#5-enhancement-bbox-temporal-stabilization)
6. [Enhancement: Audit Threshold Recalibration](#6-enhancement-audit-threshold-recalibration)
7. [Speed & Accuracy Analysis](#7-speed--accuracy-analysis)
8. [Instance Verification Steps (Step 0)](#8-instance-verification-steps-step-0)
9. [Risks & Mitigations](#9-risks--mitigations)
10. [Deployment & Rollback](#10-deployment--rollback)
11. [Appendix: Source Code Evidence](#11-appendix-source-code-evidence)

---

## 1. Problem Statement

### What happens today

In `src/extract.py`, `_detect_pass_rtmw()` (line 656) calls:

```python
batch_results = list(inferencer(batch_frames, return_vis=False))  # line 674
```

Despite passing a **list** of 48 frames, `MMPoseInferencer.__call__()` is a Python **generator** that processes frames **one at a time**. For each frame:

1. One RTMDet forward pass (person detection)
2. One RTMW-l forward pass (133-keypoint pose estimation)

**48 frames = 96 GPU forward passes per video.** Each forward pass carries:
- CUDA kernel launch + synchronization overhead (~2-5ms)
- CPU-to-GPU memory transfer for one frame
- Python generator yield/resume overhead (~1-2ms)
- Tiny batch size (1) underutilizes GPU tensor cores

The GPU shows 99% utilization — but it's **busy being inefficient**, running 96 tiny sequential operations instead of 2-4 large batched ones.

### What we want

True GPU batching: process all 48 frames in 2-4 forward passes total, eliminating ~92 synchronization cycles and enabling full tensor core utilization.

---

## 2. Research Findings

### Finding 1: MMPoseInferencer is confirmed sequential

**Source**: [mmpose/apis/inferencers/pose2d_inferencer.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/apis/inferencers/pose2d_inferencer.py)

The `preprocess()` method is a generator that yields one image at a time:

```python
def preprocess(self, inputs, batch_size=1, ...):
    for i, input in enumerate(inputs):
        data_infos = self.preprocess_single(input, index=i, ...)
        yield self.collate_fn(data_infos), [input]  # one at a time
```

And `preprocess_single()` calls the detector per-image:

```python
det_results = self.detector(input, return_datasamples=True)['predictions']
```

The `batch_size` parameter exists but is effectively ignored for top-down models. Each image goes through detection, then each detected bbox goes through pose estimation, all sequentially.

**Conclusion**: `list(inferencer(frames))` is NOT batching. It's 48 sequential iterations through a generator.

### Finding 2: `inference_topdown()` already demonstrates the batching pattern

**Source**: [mmpose/apis/inference.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/apis/inference.py)

```python
def inference_topdown(model, img, bboxes=None, bbox_format='xyxy'):
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    data_list = []
    for bbox in bboxes:
        data_info = dict(img=img)
        data_info['bbox'] = bbox[None]              # (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    batch = pseudo_collate(data_list)               # collate ALL items
    results = model.test_step(batch)                 # SINGLE forward pass
    return results
```

This function already collates multiple items and calls `test_step()` once. It just happens to use one image with N bboxes. **Our approach: N images with 1 bbox each.** The pipeline output format is identical either way — each item is a cropped, resized 288x384 image with metadata. `test_step()` doesn't care where the crops came from.

### Finding 3: `inference_detector()` supports list-of-images batching

**Source**: [mmdet/apis/inference.py](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/inference.py)

```python
results = inference_detector(model, [frame1, frame2, ..., frame48])
# Applies test_pipeline per frame, collates all, single forward pass
# Returns list of 48 DetDataSample objects
```

For numpy array inputs, the pipeline automatically substitutes `LoadImageFromNDArray` for `LoadImageFromFile`. Handles Resize(640x640), Pad, Normalize internally.

**Detection output structure:**
```python
det_result.pred_instances.bboxes   # (N, 4) xyxy format
det_result.pred_instances.scores   # (N,) confidence
det_result.pred_instances.labels   # (N,) class IDs (0 = person in COCO)
```

### Finding 4: `test_step()` auto-calls data_preprocessor

**Source**: [mmengine/model/base_model/base_model.py](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py)

```python
def test_step(self, data):
    data = self.data_preprocessor(data, training=False)  # auto
    return self._run_forward(data, mode='predict')
```

The data_preprocessor handles:
- BGR-to-RGB conversion
- Normalization (ImageNet mean/std)
- Padding to common size within batch
- Stacking into NCHW tensor
- Moving to GPU

**We never call data_preprocessor manually.** Build data_list, collate, call test_step.

### Finding 5: RTMW-l configuration specifics

**Source**: [rtmw-l_8xb320-270e_cocktail14-384x288.py](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py)

```python
input_size = (288, 384)  # (W, H) — 288 wide x 384 tall

data_preprocessor = dict(
    type='PoseDataPreprocessor',
    mean=[123.675, 116.28, 103.53],   # ImageNet BGR
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True
)

# val/test pipeline (identical):
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='PackPoseInputs')
]
test_dataloader = val_dataloader  # test uses val pipeline
```

After `TopdownAffine`, every crop is exactly **288x384** regardless of original frame size or bbox dimensions. This means **zero padding waste** when batching — all 48 crops stack perfectly into a `(48, 3, 384, 288)` tensor.

### Finding 6: Keypoint coordinates are in original image space

The model's head postprocessor automatically reverses the `TopdownAffine` transform. Output keypoints are in the **original image coordinate system**, not crop coordinates. This is handled internally by the model and requires no manual coordinate conversion.

```python
result.pred_instances.keypoints        # (1, 133, 2) in original image coords
result.pred_instances.keypoint_scores  # (1, 133)
```

---

## 3. Architecture

### Current flow (per video)

```
48 RGB frames
    |
    v
[MMPoseInferencer generator — 48 iterations]
    For each frame (sequential):
        RTMDet forward pass (1 frame)     ~5-10ms + overhead
        RTMW-l forward pass (1 frame)     ~15-25ms + overhead
    = 96 forward passes total
    |
    v
48 sets of 133 keypoints
```

### Proposed flow (per video)

```
48 RGB frames
    |
    v
[Phase 1: Batch Person Detection]
    inference_detector(det_model, all_48_frames)
    - RTMDet test_pipeline: Resize(640) -> Pad -> Normalize
    - 1-2 forward passes (batch of 24-48)
    - Returns 48 DetDataSample
    |
    v
48 person bounding boxes (xyxy)
    |
    v
[Phase 2: Batch Pose Estimation]
    For each (frame, bbox):
        Apply pose pipeline: LoadImage -> GetBBoxCenterScale
            -> TopdownAffine(288x384) -> PackPoseInputs
    pseudo_collate(all_48_items)
    pose_model.test_step(batch)
    - 1-2 forward passes (all 48 crops = same size, zero padding)
    |
    v
48 PoseDataSample (133 keypoints each, original image coords)
    |
    v
[Phase 3: Keypoint Extraction]
    IDENTICAL to existing code (lines 690-729 of extract.py)
    Same indices, thresholds, normalization
    Returns (l_seq, r_seq, face_seq, l_valid, r_valid, face_valid)
```

**Total forward passes: 2-4** (down from 96).

### The jitter problem (new — addresses quality audit)

The current pipeline has a second, independent problem beyond speed: **positional jitter**.

```
Current jitter chain:

Frame 1: RTMDet bbox = [102, 50, 380, 490]  ──> RTMW normalizes within this crop
Frame 2: RTMDet bbox = [98, 53, 375, 485]   ──> RTMW normalizes within THIS crop (shifted)
Frame 3: RTMDet bbox = [105, 48, 383, 492]  ──> RTMW normalizes within THIS crop (shifted again)

Even though the signer didn't move, the keypoints jump because each
frame's bbox is slightly different → the crop region shifts → RTMW
produces keypoints at different relative positions.

The 1-Euro filter at line 1020 can only partially smooth this AFTER the fact.
```

Quality audit evidence (`quality_audit.json`):
- **high_jitter**: 14,302 samples (24.8%)
- **extreme_jitter**: 12,511 samples (21.7%)
- **Total jitter-flagged: 46.5%** of all 57,561 samples
- `mean(max_jump) = 1.298` — the **average** sample exceeds the high_jitter threshold of 1.0

The proposed architecture adds a **Phase 1.5: Bbox Stabilization** between detection and pose estimation to fix this at the source.

```
Proposed flow with stabilization:

48 RGB frames
    |
    v
[Phase 1: Batch Detection]        ──> 48 raw bboxes (jittery)
    |
    v
[Phase 1.5: Bbox Stabilization]   ──> 48 smoothed bboxes (stable)
    - EMA smoothing (alpha=0.7)
    - Interpolate lost detections from neighbors
    |
    v
[Phase 2: Batch Pose Estimation]  ──> stable crops → stable keypoints
    |
    v
[Phase 3: Keypoint Extraction]
```

---

## 4. Implementation Details

### 4.1 Files changed

| File | Change | Scope |
|------|--------|-------|
| `src/extract.py` | Add config flags to `PipelineConfig` | 2 lines, ~line 112 |
| `src/extract.py` | Add `_stabilize_bboxes()` function | ~35 new lines |
| `src/extract.py` | Add `_detect_pass_rtmw_batched()` function | ~95 new lines, after line 731 |
| `src/extract.py` | Add dispatch at top of `_detect_pass_rtmw()` | 3 lines, ~line 657 |

**No other files are modified.** Both new functions are self-contained. Config flags enable instant A/B switching and independent toggling of batching vs stabilization.

Post-deployment follow-up (separate task, not part of this implementation):
| `src/verify_extraction_quality.py` | Recalibrate jitter thresholds based on new distribution | Lines 102-105 |

### 4.2 Config flags

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    use_batched_inference: bool = True   # True = batch test_step, False = old sequential
    bbox_smooth_alpha: float = 0.7       # 0.0 = no smoothing, 1.0 = no smoothing (current frame only)
```

Both flags are independent:
- `use_batched_inference=True, bbox_smooth_alpha=0.7` — full new pipeline (recommended)
- `use_batched_inference=True, bbox_smooth_alpha=1.0` — batching only, no stabilization
- `use_batched_inference=False` — old sequential path (rollback)

### 4.3 Dispatch in existing function

```python
def _detect_pass_rtmw(frames_rgb, frame_indices, cfg, device=None):
    if getattr(cfg, 'use_batched_inference', False):
        return _detect_pass_rtmw_batched(frames_rgb, frame_indices, cfg, device)
    # ... existing sequential code unchanged below ...
```

### 4.4 New function: `_detect_pass_rtmw_batched()`

```python
_RTMW_DET_BATCH_SIZE = 24   # Sub-batch for RTMDet (if 48 OOMs)
_RTMW_POSE_BATCH_SIZE = 48  # Sub-batch for RTMW-l (all crops same size)

def _detect_pass_rtmw_batched(frames_rgb, frame_indices, cfg, device=None):
    """True batch inference: RTMDet + RTMW-l via test_step().

    Same inputs/outputs as _detect_pass_rtmw() — drop-in replacement.
    Processes all frames in 2-4 GPU forward passes instead of 96.
    """
    inferencer = _get_rtmw_inferencer(device=device)
    if inferencer is None:
        return [], [], [], [], [], []

    import torch
    from mmdet.apis import inference_detector
    from mmcv.transforms import Compose
    from mmengine.dataset import pseudo_collate

    # Access underlying models from cached MMPoseInferencer
    # VERIFY ATTRIBUTE NAMES ON INSTANCE (Step 0, items #1 and #2)
    det_model = inferencer.detector.model    # raw RTMDet nn.Module
    pose_model = inferencer.model            # raw RTMW-l TopdownPoseEstimator

    # Build pose preprocessing pipeline from model config
    pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)

    # ── Phase 1: Batch person detection ──────────────────────────
    all_det_results = []
    for i in range(0, len(frames_rgb), _RTMW_DET_BATCH_SIZE):
        sub_batch = frames_rgb[i:i + _RTMW_DET_BATCH_SIZE]
        det_results = inference_detector(det_model, sub_batch)
        all_det_results.extend(det_results)

    # Extract best person bbox per frame
    bboxes = []
    bbox_scores = []
    for det_result, frame in zip(all_det_results, frames_rgb):
        h, w = frame.shape[:2]
        pred = det_result.pred_instances.cpu().numpy()
        # Person class = label 0 in COCO (VERIFY ON INSTANCE, Step 0 #4)
        person_mask = (pred.labels == 0) & (pred.scores >= 0.3)
        if person_mask.any():
            person_bboxes = pred.bboxes[person_mask]
            person_scores = pred.scores[person_mask]
            best_idx = person_scores.argmax()
            bboxes.append(person_bboxes[best_idx])
            bbox_scores.append(float(person_scores[best_idx]))
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))
            bbox_scores.append(0.0)  # Mark as "no detection"

    # ── Phase 1.5: Bbox temporal stabilization ───────────────────
    if cfg.bbox_smooth_alpha < 1.0:
        bboxes = _stabilize_bboxes(bboxes, bbox_scores, frames_rgb,
                                    alpha=cfg.bbox_smooth_alpha)

    # ── Phase 2: Batch pose estimation ───────────────────────────
    l_seq, r_seq, face_seq = [], [], []
    l_valid, r_valid, face_valid = [], [], []
    hand_conf_threshold = 0.25  # Same as current code

    for pose_start in range(0, len(frames_rgb), _RTMW_POSE_BATCH_SIZE):
        pose_end = min(pose_start + _RTMW_POSE_BATCH_SIZE, len(frames_rgb))
        sub_frames = frames_rgb[pose_start:pose_end]
        sub_bboxes = bboxes[pose_start:pose_end]
        sub_indices = frame_indices[pose_start:pose_end]

        # Build pipeline inputs for each (frame, bbox) pair
        data_list = []
        for frame, bbox in zip(sub_frames, sub_bboxes):
            data_info = dict(img=frame)
            data_info['bbox'] = bbox[None]  # shape (1, 4), xyxy
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)
            data_info.update(pose_model.dataset_meta)
            data_list.append(pipeline(data_info))

        batch = pseudo_collate(data_list)
        with torch.no_grad():
            pose_results = pose_model.test_step(batch)

        # ── Phase 3: Extract keypoints (same logic as lines 690-729) ──
        for result, frame, fidx in zip(pose_results, sub_frames, sub_indices):
            h, w = frame.shape[:2]

            kps_raw = result.pred_instances.keypoints[0]        # (133, 2)
            scs_raw = result.pred_instances.keypoint_scores[0]  # (133,)

            kps = np.array(kps_raw, dtype=np.float32) if not isinstance(kps_raw, np.ndarray) else kps_raw.astype(np.float32)
            scs = np.array(scs_raw, dtype=np.float32) if not isinstance(scs_raw, np.ndarray) else scs_raw.astype(np.float32)

            if len(kps) < 133:
                continue

            inv_wh = np.array([1.0 / w, 1.0 / h], dtype=np.float32)

            # Left hand (91-111)
            l_scs = scs[_RTMPOSE_LHAND_START:_RTMPOSE_LHAND_START + 21]
            if l_scs.mean() >= hand_conf_threshold:
                l_kps = kps[_RTMPOSE_LHAND_START:_RTMPOSE_LHAND_START + 21] * inv_wh
                coords = np.zeros((21, 3), dtype=np.float32)
                coords[:, :2] = l_kps
                if not l_valid or l_valid[-1] != fidx:
                    l_seq.append(coords)
                    l_valid.append(fidx)

            # Right hand (112-132)
            r_scs = scs[_RTMPOSE_RHAND_START:_RTMPOSE_RHAND_START + 21]
            if r_scs.mean() >= hand_conf_threshold:
                r_kps = kps[_RTMPOSE_RHAND_START:_RTMPOSE_RHAND_START + 21] * inv_wh
                coords = np.zeros((21, 3), dtype=np.float32)
                coords[:, :2] = r_kps
                if not r_valid or r_valid[-1] != fidx:
                    r_seq.append(coords)
                    r_valid.append(fidx)

            # Face (5 landmarks)
            face_idx_arr = np.array(_RTMPOSE_FACE_INDICES)
            if np.all(scs[face_idx_arr] >= 0.25):
                f_kps = kps[face_idx_arr] * inv_wh
                fcoords = np.zeros((5, 3), dtype=np.float32)
                fcoords[:, :2] = f_kps
                if not face_valid or face_valid[-1] != fidx:
                    face_seq.append(fcoords)
                    face_valid.append(fidx)

    return l_seq, r_seq, face_seq, l_valid, r_valid, face_valid
```

### 4.5 Behavioral difference: person selection

**Current path** (via MMPoseInferencer):
- RTMDet detects all people per frame
- RTMW-l runs pose estimation on ALL detected people
- `_pick_primary_person()` selects the largest person from pose results

**New path** (batched):
- RTMDet detects all people per frame
- We pick the highest-confidence person bbox BEFORE pose estimation
- RTMW-l runs on only that one person per frame

**Why this is acceptable:**
- Saves compute (no wasted pose estimation on background people)
- The highest-confidence detection from RTMDet is almost always the same person that `_pick_primary_person()` would select (largest bbox = highest confidence for prominent signer)
- Verification test on 20 videos will confirm equivalence

**If this causes regressions** (unlikely): we can run pose on top-K bboxes per frame and keep `_pick_primary_person()`. This only adds K-1 items to the batch.

---

## 5. Enhancement: Bbox Temporal Stabilization

### 5.1 The problem in detail

RTMDet runs independently on every frame. Even when the signer is stationary, the detected bounding box shifts by a few pixels between frames due to:
- Non-maximum suppression randomness
- Feature map quantization at different input positions
- Minor preprocessing differences (resize rounding)

Since RTMW normalizes the person crop to 288x384 before pose estimation, a 5-pixel bbox shift on a 400-pixel-wide bbox translates to a ~1.25% coordinate shift in the crop — which propagates directly into the final keypoints.

This is the **root cause** of the 46.5% jitter rate in the quality audit. The existing 1-Euro filter (extract.py line 1020) runs AFTER normalization and can only partially compensate.

### 5.2 Research backing

Bbox temporal smoothing is a well-established technique:
- [RTMPose paper (ECCV 2023)](https://ar5iv.labs.arxiv.org/html/2303.07399) uses OKS-based pose NMS + OneEuro filter in post-processing for smooth video prediction
- [SmoothNet (ECCV 2022)](https://arxiv.org/abs/2112.13715) is a dedicated temporal refinement network for pose jitter
- Standard practice in production pose pipelines: apply EMA or Kalman filter to detection bboxes before pose estimation

Our case is simpler than the general case because:
- Single person (primary signer) — no multi-person tracking needed
- Short clips (~48 frames) — no long-term drift concern
- Offline processing — can use bidirectional smoothing

### 5.3 Implementation: `_stabilize_bboxes()`

This function runs between Phase 1 (detection) and Phase 2 (pose estimation). ~25 lines.

```python
def _stabilize_bboxes(bboxes, scores, frames_rgb):
    """Temporal stabilization of detected bounding boxes.

    Two operations:
    1. Interpolate lost detections: if RTMDet missed a frame (score=0),
       linearly interpolate the bbox from nearest valid neighbors instead
       of falling back to full-frame [0,0,w,h] which causes spike jitter.
    2. EMA smoothing: apply exponential moving average across frames to
       dampen per-frame bbox jitter from RTMDet.

    Args:
        bboxes: list of (4,) numpy arrays, xyxy format. Full-frame fallbacks
                have score=0.
        scores: list of float, detection confidence per frame. 0 = no detection.
        frames_rgb: list of frames (for dimensions on fallback).

    Returns:
        smoothed_bboxes: list of (4,) numpy arrays, stabilized.
    """
    n = len(bboxes)
    if n == 0:
        return bboxes

    bboxes_arr = np.array(bboxes, dtype=np.float32)  # (N, 4)
    scores_arr = np.array(scores, dtype=np.float32)   # (N,)

    # ── Step 1: Interpolate lost detections ──
    # Find frames where RTMDet actually detected a person
    valid_mask = scores_arr > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # No detections at all — nothing to stabilize
        return bboxes

    if len(valid_indices) < n:
        # Some frames have no detection — interpolate from neighbors
        for i in range(n):
            if not valid_mask[i]:
                # Find nearest valid neighbors
                left = valid_indices[valid_indices < i]
                right = valid_indices[valid_indices > i]

                if len(left) > 0 and len(right) > 0:
                    l_idx, r_idx = left[-1], right[0]
                    # Linear interpolation
                    alpha = (i - l_idx) / (r_idx - l_idx)
                    bboxes_arr[i] = (1 - alpha) * bboxes_arr[l_idx] + alpha * bboxes_arr[r_idx]
                elif len(left) > 0:
                    bboxes_arr[i] = bboxes_arr[left[-1]]  # Extend last known
                elif len(right) > 0:
                    bboxes_arr[i] = bboxes_arr[right[0]]  # Extend next known

    # ── Step 2: EMA smoothing ──
    # alpha=0.7: 70% current frame, 30% previous smoothed value.
    # This dampens frame-to-frame jitter while following real motion.
    # Chosen because signing motion is fast — higher alpha preserves responsiveness.
    alpha = 0.7
    smoothed = np.copy(bboxes_arr)
    for i in range(1, n):
        smoothed[i] = alpha * bboxes_arr[i] + (1 - alpha) * smoothed[i - 1]

    return [smoothed[i] for i in range(n)]
```

### 5.4 Integration point

In `_detect_pass_rtmw_batched()`, between Phase 1 and Phase 2:

```python
    # ── Phase 1: Batch person detection ──
    # ... (existing RTMDet batched detection code) ...

    # Extract best person bbox per frame
    bboxes = []
    bbox_scores = []
    for det_result, frame in zip(all_det_results, frames_rgb):
        h, w = frame.shape[:2]
        pred = det_result.pred_instances.cpu().numpy()
        person_mask = (pred.labels == 0) & (pred.scores >= 0.3)
        if person_mask.any():
            person_bboxes = pred.bboxes[person_mask]
            person_scores = pred.scores[person_mask]
            best_idx = person_scores.argmax()
            bboxes.append(person_bboxes[best_idx])
            bbox_scores.append(float(person_scores[best_idx]))
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))
            bbox_scores.append(0.0)  # Mark as "no detection"

    # ── Phase 1.5: Bbox temporal stabilization ──
    bboxes = _stabilize_bboxes(bboxes, bbox_scores, frames_rgb)

    # ── Phase 2: Batch pose estimation ──
    # ... (existing batched pose code, now receives stabilized bboxes) ...
```

### 5.5 Why EMA alpha=0.7

| alpha | Behavior | Risk |
|-------|----------|------|
| 0.5 | Heavy smoothing — cuts jitter by ~50% per frame | May lag behind fast hand motion in signing |
| 0.7 | Moderate — cuts jitter by ~30% per frame, responsive to motion | Good balance for ASL signing speed |
| 0.9 | Light — cuts only ~10% per frame | Barely helps; jitter mostly persists |
| 1.0 | No smoothing (current behavior) | 46.5% jitter rate |

We start with alpha=0.7 and tune based on the verification test. If fast signs get blurred, increase toward 0.8. If jitter persists, decrease toward 0.6.

This is configurable — can be added to `PipelineConfig` as `bbox_smooth_alpha: float = 0.7`.

### 5.6 Why interpolation instead of full-frame fallback

Current behavior when RTMDet misses a person:
```
Frame 10: bbox = [100, 50, 380, 490]  (good detection)
Frame 11: bbox = [0, 0, 640, 480]     (no detection → full frame fallback)
Frame 12: bbox = [102, 48, 382, 488]  (good detection)
```

RTMW runs on a full-frame crop for frame 11 → keypoints are wildly different because the crop scale changed from ~280x440 to 640x480. This creates a **spike** in the keypoint trajectory that even the 1-Euro filter can't fully remove.

With interpolation:
```
Frame 10: bbox = [100, 50, 380, 490]  (good detection)
Frame 11: bbox = [101, 49, 381, 489]  (interpolated from neighbors)
Frame 12: bbox = [102, 48, 382, 488]  (good detection)
```

The crop stays consistent → no spike → clean keypoints.

### 5.7 Expected impact on quality audit metrics

| Metric | Current | After bbox stabilization | Reasoning |
|--------|---------|------------------------|-----------|
| high_jitter (>1.0) | 14,302 (24.8%) | est. ~5-8% | Bbox jitter is the dominant source |
| extreme_jitter (>2.0) | 12,511 (21.7%) | est. ~3-5% | Spike jitter from lost detections eliminated |
| Total jitter-flagged | 46.5% | est. ~8-13% | Combined effect of EMA + interpolation |
| bone_cv | 0.002 mean | ~same | Bbox smoothing doesn't affect bone ratios |

These are estimates. The real numbers come from re-running the quality audit after extraction.

---

## 6. Enhancement: Audit Threshold Recalibration

### 6.1 The problem

The jitter thresholds in `src/verify_extraction_quality.py` (lines 102-105) are:

```python
if max_jump > 2.0:
    issues.append("extreme_jitter")
elif max_jump > 1.0:
    issues.append("high_jitter")
```

But the actual distribution from `quality_audit.json`:
```
max_jump: mean=1.298, std=1.443
```

**The average sample exceeds the `high_jitter` threshold.** This means the thresholds are calibrated for a different extraction method (likely MediaPipe, which outputs image-space coordinates with a different noise profile than RTMW's bbox-relative-then-renormalized coordinates).

### 6.2 Recommendation

**Do NOT recalibrate thresholds before bbox stabilization.** The current 46.5% flag rate is inflated by two overlapping causes:
1. Real jitter from bbox instability (the dominant cause — being fixed)
2. Thresholds miscalibrated for RTMW output characteristics

The correct sequence:
1. Implement bbox stabilization (Section 5)
2. Re-extract or process a representative sample (~1000 videos)
3. Profile the new `max_jump` distribution
4. Set thresholds at the 95th and 99th percentiles of the NEW distribution

### 6.3 Implementation (post-extraction)

After the new extraction completes:

```python
# Sample the new distribution
jumps = [audit_sample(f)['metrics']['max_jump'] for f in sample_files]
p95 = np.percentile(jumps, 95)
p99 = np.percentile(jumps, 99)
print(f"Recommended thresholds: high_jitter={p95:.2f}, extreme_jitter={p99:.2f}")
```

Then update `verify_extraction_quality.py` lines 102-105 with the data-driven thresholds.

### 6.4 Why this is a separate step, not part of the main implementation

- Threshold recalibration requires data from the NEW extraction to be meaningful
- It only affects the audit tool, not the extraction pipeline itself
- Getting it wrong doesn't break anything — it just means the audit flags wrong samples
- We want to measure the actual improvement from bbox stabilization before re-tuning

**This goes on the post-deployment task list, not the implementation checklist.**

---

## 7. Speed & Accuracy Analysis

### Speed

| Metric | Current (sequential) | Proposed (batched + stabilized) |
|--------|---------------------|-------------------------------|
| RTMDet forward passes | 48 | 1-2 |
| RTMW-l forward passes | 48 | 1-2 |
| Total GPU forward passes | 96 | 2-4 |
| CUDA sync overhead | ~96 x 3ms = ~290ms | ~4 x 3ms = ~12ms |
| Python generator overhead | ~48 x 1.5ms = ~72ms | 0ms |
| Bbox stabilization (CPU) | N/A | ~0.1ms (trivial) |
| Detection+pose time per video | ~1.5-2.0s | ~0.25-0.4s |
| Video I/O + postprocess (unchanged) | ~0.3-0.5s | ~0.3-0.5s |
| **Total per video** | **~1.8-2.5s** | **~0.55-0.9s** |
| **Speedup on bottleneck** | 1x | **~5-7x** |
| **Overall pipeline speedup** | 1x | **~3-4x** |

Note: Bbox stabilization adds negligible CPU time (~0.1ms for 48 bboxes). The speedup comes entirely from batching — eliminating synchronization overhead and achieving better tensor core utilization.

### Memory budget (RTX 4090, 24GB)

| Component | Memory |
|-----------|--------|
| RTMW-l model weights | ~600MB |
| RTMDet model weights | ~300MB |
| RTMDet batch input: 24 x 3 x 640 x 640 (float32) | ~118MB |
| RTMW-l batch input: 48 x 3 x 384 x 288 (float32) | ~64MB |
| Intermediate activations (peak) | ~2-3GB |
| **Total peak per worker** | **~4GB** |
| 2 workers per GPU | ~8GB / 24GB available |

Comfortable headroom. Sub-batching (24 for RTMDet, 48 for RTMW-l) provides a safety valve if memory is tighter than expected.

### Accuracy: batching component

**Mathematically identical within float32 precision** (batching alone, without bbox stabilization).

| Component | Sequential path | Batched path | Same? |
|-----------|----------------|-------------|-------|
| RTMDet weights | Same checkpoint | Same checkpoint | Yes |
| RTMDet preprocessing | Resize(640) + Pad + Normalize via test_pipeline | Same via `inference_detector` | Yes |
| RTMW-l weights | Same checkpoint | Same checkpoint | Yes |
| RTMW-l preprocessing | LoadImage + GetBBoxCenterScale + TopdownAffine(288x384) + PackPoseInputs | Same pipeline from `model.cfg.test_dataloader.dataset.pipeline` | Yes |
| BatchNorm behavior | eval mode, running stats (batch-independent) | eval mode, same running stats | Yes |
| Keypoint postprocessing | Model reverses affine -> original coords | Same | Yes |
| Hand indices | 91-111, 112-132 | Same constants | Yes |
| Face indices | Same `_RTMPOSE_FACE_INDICES` | Same | Yes |
| Confidence thresholds | 0.25 hand, 0.25 face | Same | Yes |
| Coordinate normalization | `kps * [1/w, 1/h]` | Same | Yes |

The only possible difference is float32 accumulation ordering (~1e-7 level), which is imperceptible. Verification test target: `max_diff < 0.001`.

### Accuracy: bbox stabilization component

Bbox stabilization intentionally **changes** the keypoint outputs (that's the point — smoother, less jittery). The change is strictly beneficial:

| Quality metric | Current | After stabilization | Why |
|---------------|---------|--------------------|----|
| Jitter-flagged rate | 46.5% | est. ~8-13% | Bbox EMA removes per-frame bbox noise |
| Spike jitter | Common (lost detection → full-frame fallback) | Eliminated | Interpolation from neighbors instead |
| Temporal coherence | Degraded by bbox shifts | Improved | Stable crop → stable keypoints |
| Bone length consistency | bone_cv mean=0.002 | Same or better | Bbox smoothing reduces within-hand jitter |
| Downstream model accuracy | Baseline | Expected improvement | Cleaner input features for Stage 1/2 |

The verification test for this component measures jitter reduction, not numerical identity:
```
assert new_max_jump < old_max_jump  # per-sample
assert new_jitter_rate < old_jitter_rate * 0.5  # overall: at least 50% reduction
```

---

## 8. Instance Verification Steps (Step 0)

These are the **only unknowns** that cannot be resolved from documentation alone. They must be verified on the running Vast.ai instance before writing any code.

### Verification 1: Access raw detector model

```python
from mmpose.apis import MMPoseInferencer
inf = MMPoseInferencer(
    pose2d='rtmw-l_8xb320-270e_cocktail14-384x288',
    pose2d_weights='models/rtmw_l_wholebody.pth',
    device='cuda:0'
)
print(type(inf.detector))           # Expected: DetInferencer
print(type(inf.detector.model))     # Expected: RTMDet nn.Module
print(dir(inf.detector))            # Look for .model attribute
```

**If `inf.detector.model` doesn't exist**: check `inf.detector.cfg`, try `inf.detector.visualizer`, etc. Worst case: load RTMDet separately via `init_detector()`.

### Verification 2: Access raw pose model

```python
print(type(inf.model))  # Expected: TopdownPoseEstimator
print(inf.model.cfg.test_dataloader.dataset.pipeline)  # Expected: pipeline list
```

### Verification 3: inference_detector with numpy arrays

```python
from mmdet.apis import inference_detector
import numpy as np
test_frame = np.zeros((256, 256, 3), dtype=np.uint8)
results = inference_detector(inf.detector.model, [test_frame])
print(type(results))       # Expected: list
print(type(results[0]))    # Expected: DetDataSample
print(results[0].pred_instances.bboxes.shape)
print(results[0].pred_instances.labels)
```

### Verification 4: Person class ID

```python
print(inf.detector.model.dataset_meta)  # Look for 'classes' containing 'person'
# Expected: person = label 0
```

### Verification 5: Pose pipeline builds correctly

```python
from mmcv.transforms import Compose
pipeline = Compose(inf.model.cfg.test_dataloader.dataset.pipeline)
print(pipeline)  # Should show: LoadImage, GetBBoxCenterScale, TopdownAffine, PackPoseInputs
```

### Verification 6: Batched pose test_step works

```python
from mmengine.dataset import pseudo_collate
import torch

test_frame = np.random.randint(0, 255, (384, 288, 3), dtype=np.uint8)
bbox = np.array([0, 0, 288, 384], dtype=np.float32)

data_list = []
for _ in range(4):  # Small batch first
    data_info = dict(img=test_frame.copy())
    data_info['bbox'] = bbox[None]
    data_info['bbox_score'] = np.ones(1, dtype=np.float32)
    data_info.update(inf.model.dataset_meta)
    data_list.append(pipeline(data_info))

batch = pseudo_collate(data_list)
with torch.no_grad():
    results = inf.model.test_step(batch)

print(len(results))  # Expected: 4
print(results[0].pred_instances.keypoints.shape)       # Expected: (1, 133, 2)
print(results[0].pred_instances.keypoint_scores.shape)  # Expected: (1, 133)
```

### Verification 7: Full batch of 48 doesn't OOM

```python
# Same as above but with 48 items
data_list_48 = []
for _ in range(48):
    data_info = dict(img=test_frame.copy())
    data_info['bbox'] = bbox[None]
    data_info['bbox_score'] = np.ones(1, dtype=np.float32)
    data_info.update(inf.model.dataset_meta)
    data_list_48.append(pipeline(data_info))

batch_48 = pseudo_collate(data_list_48)
with torch.no_grad():
    results_48 = inf.model.test_step(batch_48)
print(f"Batch of 48: OK, {len(results_48)} results")
# If OOM: reduce _RTMW_POSE_BATCH_SIZE to 24
```

**Estimated total time for all verifications: 15-20 minutes on instance.**

---

## 9. Risks & Mitigations

### Risk 1: Attribute names differ from expected

| Aspect | Expected | If different |
|--------|----------|-------------|
| `inferencer.detector.model` | RTMDet nn.Module | Inspect `dir(inferencer.detector)`; fallback: `init_detector()` to load separately |
| `inferencer.model` | TopdownPoseEstimator | Inspect `dir(inferencer)`; should be standard BaseInferencer attribute |
| `model.cfg.test_dataloader.dataset.pipeline` | Pipeline config list | Try `model.cfg.val_dataloader.dataset.pipeline` |

**Likelihood**: Low (standard OpenMMLab patterns). **Impact**: Low (easy to fix on instance).

### Risk 2: OOM with batch=48

**RTMDet**: 48 frames at 640x640 = ~235MB input. With activations: ~1.5GB peak. Already sub-batched to 24 as safety measure.

**RTMW-l**: 48 crops at 288x384 = ~64MB input. With activations: ~1GB peak.

**If OOM occurs**: Reduce `_RTMW_DET_BATCH_SIZE` to 16 or `_RTMW_POSE_BATCH_SIZE` to 24. Still 3-6 forward passes total — still a massive improvement over 96.

**Likelihood**: Low on RTX 4090 (24GB). **Impact**: Low (just reduce batch size).

### Risk 3: `inference_detector` doesn't handle numpy arrays in mmdet 3.3.0

Documentation and source confirm it does (substitutes `LoadImageFromNDArray`). But mmdet version pinning on the instance may differ.

**Fallback**: Build DetDataSample manually:
```python
from mmdet.structures import DetDataSample
# ... apply test_pipeline transforms manually
```

**Likelihood**: Very low. **Impact**: Medium (more code to write, but well-documented pattern).

### Risk 4: Person selection differs between paths

**Current**: MMPoseInferencer runs pose on ALL detected people, then `_pick_primary_person()` picks the largest by body keypoint area.

**New**: We pick the highest-confidence person bbox BEFORE pose estimation.

These usually select the same person (the prominent signer in frame). But edge cases exist:
- Two people with similar confidence but different body sizes
- Person partially occluded but with high detection confidence

**Mitigation**: Verification test on 20 videos compares keypoint outputs between old and new. If regressions appear, run pose on top-3 bboxes and reintroduce `_pick_primary_person()`.

**Likelihood**: Low (primary signer is usually the highest-confidence detection). **Impact**: Medium if it happens, but caught by verification.

### Risk 5: Pipeline transform ordering or behavior differs

The pose pipeline (`LoadImage -> GetBBoxCenterScale -> TopdownAffine -> PackPoseInputs`) is loaded from the **same model config** used by MMPoseInferencer internally. The transforms are identical objects.

**One subtlety**: `LoadImage` behavior with numpy arrays vs file paths. When `data_info` contains `'img'` key (numpy), `LoadImage` should pass it through without file I/O. This is standard mmcv behavior but worth verifying (Step 0 #6).

**Likelihood**: Very low. **Impact**: High if wrong (bad keypoints). Caught by verification test.

### Risk 6: Numerical drift from different batch sizes

BatchNorm in eval mode uses frozen running statistics — batch-size independent. Convolutional layers are inherently batch-independent. The only source of drift is float32 accumulation ordering in operations like softmax or layer normalization, which causes ~1e-7 level differences.

**Likelihood**: Guaranteed to have tiny differences. **Impact**: Zero (imperceptible, well within 0.001 tolerance).

### Risk 7: Bbox EMA over-smoothes fast signing motion

If `alpha=0.7` is too aggressive, the smoothed bbox will lag behind rapid hand movements, causing the crop to clip the hands at the edge.

**Mitigation**:
- alpha=0.7 means only 30% influence from previous frame — lag is at most 1-2 frames
- RTMW's `GetBBoxCenterScale` applies a 1.25x padding factor to the bbox before cropping, providing a generous margin
- The parameter is configurable (`bbox_smooth_alpha` in PipelineConfig) — can tune after seeing results
- Verification test on fast-signing videos (e.g., fingerspelling) will reveal if this is an issue

**Likelihood**: Low (signing motion moves the whole arm, not just the bbox edge). **Impact**: Medium if it happens, but easily tunable.

### Risk 8: Bbox interpolation creates phantom detections

If RTMDet misses 5+ consecutive frames (e.g., person leaves frame entirely), interpolation will create bboxes where no person exists, producing garbage keypoints.

**Mitigation**:
- This is already better than current behavior (full-frame fallback also produces garbage)
- Could add a max interpolation gap: if more than N consecutive frames have no detection, don't interpolate — mark as truly missing
- In practice, RTMDet rarely misses more than 1-2 frames for a visible signer. Multi-frame gaps indicate the person actually left the frame, which the downstream pipeline handles via coverage thresholds.

**Likelihood**: Very low for signing videos. **Impact**: Low (same or better than current full-frame fallback).

---

## 10. Deployment & Rollback

### Pre-deployment checklist

- [ ] All 7 instance verifications pass (Section 8)
- [ ] `extract.py` backed up: `cp src/extract.py src/extract.py.bak`
- [ ] New function + bbox stabilization added with config flags
- [ ] Batching verification: test passes on 20 videos (`max_diff < 0.001` with stabilization OFF)
- [ ] Stabilization verification: jitter reduced on 20 videos (with stabilization ON)
- [ ] Fast-signing spot check: no clipped hands on fingerspelling videos

### Deployment

```bash
# On the Vast.ai instance:
python3 src/extract_do.py --workers 64
```

### Monitoring (first 500 videos)

Watch for:
- **Save rate**: Should remain >= 50%. If drops below 30%, STOP.
- **Speed**: Should see 3-4x improvement over baseline.
- **Errors**: Any new exceptions in logs (OOM, shape mismatch, etc.)

### Rollback options

**Option 1 — Config flag (instant, no restart):**
Edit `extract.py` line ~112: `use_batched_inference: bool = False`

**Option 2 — Full restore (if code is broken):**
```bash
cp src/extract.py.bak src/extract.py
```

**Option 3 — On next restart:**
Workers pick up changes on restart since ProcessPoolExecutor uses `spawn`.

---

## 11. Appendix: Source Code Evidence

### A. MMPoseInferencer.__call__ is a generator

From `mmpose/apis/inferencers/base_mmpose_inferencer.py`:
```python
def __call__(self, inputs, ...):
    # ...
    inputs = self.preprocess(inputs, batch_size=batch_size, ...)
    for proc_inputs, ori_inputs in inputs:      # <-- iterates generator
        preds = self.forward(proc_inputs, ...)
        # ...
        yield results                            # <-- yields per-frame
```

### B. Detector called per-image

From `mmpose/apis/inferencers/pose2d_inferencer.py`:
```python
def preprocess_single(self, input, ...):
    # ...
    det_results = self.detector(
        input, return_datasamples=True)['predictions']  # single image
    pred_instance = det_results[0].pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
```

### C. inference_topdown batching pattern

From `mmpose/apis/inference.py`:
```python
data_list = []
for bbox in bboxes:
    data_info = dict(img=img)
    data_info['bbox'] = bbox[None]
    data_info['bbox_score'] = np.ones(1, dtype=np.float32)
    data_info.update(model.dataset_meta)
    data_list.append(pipeline(data_info))

batch = pseudo_collate(data_list)
with torch.no_grad():
    results = model.test_step(batch)  # single call for ALL items
```

### D. test_step auto-preprocesses

From `mmengine/model/base_model/base_model.py`:
```python
def test_step(self, data):
    data = self.data_preprocessor(data, training=False)
    return self._run_forward(data, mode='predict')
```

### E. RTMW-l val/test pipeline

From config:
```python
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),  # (288, 384)
    dict(type='PackPoseInputs')
]
```
