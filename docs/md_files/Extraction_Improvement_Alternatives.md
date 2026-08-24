# Extraction Improvement Alternatives

Post-video alternatives to improve extraction/detection without retaking videos.

---

## 1. Re-extract with a Better Pipeline (Same Videos)

| Change | Effect |
|--------|--------|
| **Higher input resolution** | Resize to 512px or 640px instead of 384px before MediaPipe. Detection often improves on small/far hands. |
| **Less aggressive frame skip** | For short videos, use `skip=1` (decode every frame) so fewer frames are discarded before detection. |
| **MediaPipe Holistic** | Use `Holistic` instead of separate Hands + FaceMesh for more consistent hand/face co-detection. |
| **`model_complexity=0`** | Sometimes better on webcam-quality or low-res; worth testing on your worst classes. |
| **Two-pass: static + video** | Run once in static mode, once in video mode, merge or pick the pass with more detections. |
| **Multi-scale per frame** | Run detection at 1.0× and 1.25× scale, keep the pass with higher confidence. |

These require re-running `extract.py` with modified logic but on the **same** video files.

---

## 2. Post-process Existing `.npy` Files (No Re-extraction)

For clips that already produced `.npy` output, you can:

| Operation | What it does |
|-----------|--------------|
| **Temporal smoothing** | Reduce jitter by smoothing XYZ over time (e.g., Gaussian, Savitzky–Golay). |
| **Recompute kinematics** | After smoothing XYZ, recompute velocity and acceleration. |
| **Outlier handling** | Clip or replace extreme velocity/acceleration spikes. |
| **Mask refinement** | (If per-frame confidence were stored) downweight low-confidence points—but current format doesn't store that. |

**What you cannot fix in `.npy` alone:**  
If MediaPipe never detected hands in a given frame, there are no hand landmarks for that frame. The pipeline already interpolates across valid detections; you can't invent new detections from the stored `.npy`.

---

## 3. For Videos That Never Produced `.npy`

Those have no `.npy` to edit. The only options are:

1. Re-extract with **relaxed thresholds** (e.g., `min_raw_frames` 8→5, `max_missing_ratio` 0.40→0.55).
2. Re-extract with **alternative models** (Holistic, different MediaPipe params, etc.).

---

## 4. Concrete Recommendations (No Retakes)

1. **Re-extract failed videos** with:
   - Higher resolution (e.g., 512px)
   - `skip=1` when `total_frames < 20`
   - Possibly `model_complexity=0` or Holistic

2. **Add a post-processing script** that:
   - Loads each `.npy`
   - Applies temporal smoothing to XYZ
   - Recomputes velocity and acceleration
   - Optionally clips outliers
   - Saves back (or to a new dir like `ASL_landmarks_float16_smoothed`)

3. **Do not try to "fix" data where hands were never detected**; those frames stay interpolated from neighbors.

---

## Implementation Options

- **`.npy` post-processing script** (smoothing + kinematic recomputation) to run on `ASL_landmarks_float16`.
- **Changes to `extract.py`** for higher-res, adaptive skip, and optional Holistic / multi-scale detection (for re-extraction of existing videos only).
