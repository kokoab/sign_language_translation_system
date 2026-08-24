# Extract v7.0 — Aggressive Detection Changes

**Date:** 2026-03-21
**Goal:** Maximize extraction yield without quality tradeoffs on landmark accuracy.

---

## Changes Summary

### 1. Resolution Bump: 384px → 512px
**File:** `src/extract.py` — `process_single_video()`
**What:** Max input dimension for MediaPipe raised from 384px to 512px.
**Why:** MediaPipe's palm detection model needs enough pixels to find hand bounding boxes. 512px gives ~1.8x more pixels, directly improving detection for small/distant hands. Landmarks are normalized coordinates — resolution doesn't affect output quality.
**Before:** `if max(h, w) > 384: scale = 384 / max(h, w)`
**After:** `if max(h, w) > 512: scale = 512 / max(h, w)`

### 2. CLAHE Preprocessing
**File:** `src/extract.py` — `process_single_video()`
**What:** Contrast-Limited Adaptive Histogram Equalization applied to every frame before MediaPipe detection.
**Why:** Many webcam videos have uneven lighting where hands blend into backgrounds. CLAHE enhances local contrast in the L channel (LAB color space) without color shifts. MediaPipe gets better-separated hand edges. Non-destructive — only affects MediaPipe input, not output coordinates.
**Settings:** `clipLimit=2.0`, `tileGridSize=(8, 8)` (standard conservative values).
**Code:**
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

### 3. Adaptive Frame Skip
**File:** `src/extract.py` — `process_single_video()`
**What:** Short videos (< 80 frames) process every frame (`skip=1`). Longer videos use 2x oversampling (`total_est // 64`) instead of 1.5x (`total_est // 48`).
**Why:** Fast signs like DONT/WANT last ~1 second. With the old formula, a 120-frame video skipped every other frame — the only frames with visible hands could be the skipped ones. More processed frames = more detection opportunities.
**Before:** `skip = max(1, total_est // 48)` (1.5x oversampling, always)
**After:**
```python
if total_est < 80:
    skip = 1
else:
    skip = max(1, total_est // 64)
```

### 4. Two-Pass Detection (Video + Static Mode)
**File:** `src/extract.py` — `process_single_video()`, new `_detect_pass()`
**What:** Every video is processed twice — once with MediaPipe in video mode (temporal tracking) and once in static mode (independent per-frame detection). For each component (left hand, right hand, face), the pass with more valid detections is selected.
**Why:** Video mode's temporal tracker can "lose" a hand mid-sequence and never re-acquire. Static mode re-detects every frame independently but lacks smoothness. By running both and picking the best per-component, we get the best of both worlds.
**Merge strategy:** Per-component majority wins. Example: if video mode detected left hand in 25/30 frames but static mode only 18/30, video mode's left hand is used. But if static mode detected right hand in 22/30 vs video mode's 15/30, static mode's right hand is used.
**Cost:** ~2x extraction time per video (acceptable for a one-time batch extraction).

---

## Quality Thresholds (Tightened Back from v6.1)

v6.1 relaxed these to compensate for poor detection. v7.0's detection improvements (512px, CLAHE, two-pass) make the relaxation unnecessary — tightening them back up filters out genuinely poor clips.

| Setting | v6.1 (relaxed) | v7.0 (restored) | Rationale |
|---------|---------------|-----------------|-----------|
| `min_raw_frames` | 5 | **8** | With better detection, <8 frames means genuinely too short |
| `max_missing_ratio` | 0.55 | **0.40** | Two-pass fills detection gaps; >40% missing = genuinely problematic |
| `min_detection_conf` | 0.80 | 0.80 | Inviolable constraint (train/inference alignment) |
| `min_tracking_conf` | 0.80 | 0.80 | Same |
| `model_complexity` | 1 | 1 | Best landmark accuracy |

---

## Files Changed

| File | Change |
|------|--------|
| `src/extract.py` | v7.0 rewrite: 512px, CLAHE, adaptive skip, two-pass detection, `_detect_pass()` helper, `extraction_stats.json` output |
| `src/camera_inference.py` | `MIN_RAW_FRAMES` restored to 8 (aligned with extract.py v7.0) |
| `src/test_video_pipeline.py` | `MIN_RAW_FRAMES` restored to 8 (aligned with extract.py v7.0) |
| `src/train_stage_1.py` | Per-class accuracy logged on final test set, saved to `history.json` |
| `src/train_stage_2.py` | Example decoded sequences (ref vs hyp) logged on final test set, saved to `stage2_history.json` |

---

## Re-Extraction Instructions

Since all .npy files should be consistent under the new settings:

```bash
# 1. Delete all existing extractions (use find to avoid ARG_MAX on large dirs)
find ASL_landmarks_float16 -name "*.npy" -delete

# 2. Run fresh extraction with v7.0
python src/extract.py

# 3. Validate the new manifest
python src/validate_manifest.py
```

**Expected improvements:**
- Higher extraction yield (fewer failed clips, especially for fast/brief signs)
- Better landmark quality for clips that previously barely passed (more detection frames → better interpolation)
- Consistent processing across all clips (no mix of old/new settings)

---

## What Was NOT Changed (And Why)

| Suggestion | Decision | Reason |
|------------|----------|--------|
| MediaPipe Holistic | Skipped | Deprecated by Google, less accurate hand landmarks than standalone Hands |
| `model_complexity=0` | Skipped | Strictly worse landmark accuracy than complexity 1 |
| Multi-scale detection | Skipped | 512px resolution bump achieves the same goal more simply |
| Post-processing smoothing | Skipped | Doesn't recover failed clips (the core problem); training augmentation already regularizes jitter |
