# Extract v7.0 — Review, Suggestions & Timing

**Date:** 2026-03-21

---

## Review of Extract_v7_Changes.md

### What v7.0 Gets Right

| Change | Assessment |
|--------|------------|
| **512px resolution** | Solid choice. More pixels for palm detection without going overboard (640px+ would slow things significantly). |
| **CLAHE preprocessing** | Good for webcam videos with uneven lighting. Conservative `clipLimit=2.0` avoids over-enhancement. |
| **Adaptive frame skip** | Smart. Short videos (<80 frames) getting `skip=1` directly addresses fast signs (DONT, WANT, etc.) that were failing. |
| **Two-pass detection** | Strong idea. Video mode loses hands; static mode re-detects. Merging per-component gives best of both. |
| **Tightening min_raw_frames (8) and max_missing_ratio (0.40)** | Correct. With better detection, you don't need to relax quality thresholds. |

### Doc Inconsistency

The Extract_v7 "Files Changed" table says:
> `camera_inference.py` | `MIN_RAW_FRAMES` 8→5
> `test_video_pipeline.py` | `MIN_RAW_FRAMES` 8→5

That implies going *to* 5, which contradicts "Quality Thresholds (Tightened Back)" where v7.0 restores 8. The intent is likely **5→8** (align inference/test *with* extract's 8). Verify `camera_inference.py` and `test_video_pipeline.py` use `MIN_RAW_FRAMES=8` to match extract.

### rm -rf Fix

The doc suggests `rm -rf ASL_landmarks_float16/*.npy` — with tens of thousands of files this hits "argument list too long". Use:
```bash
find ASL_landmarks_float16 -name "*.npy" -delete
```

---

## model_complexity: Is There a 2?

**No.** MediaPipe **Hands** only supports `model_complexity` 0 or 1:

| Value | Model | Accuracy | Speed |
|-------|-------|----------|-------|
| 0 | LITE | Lower landmark accuracy | Faster |
| 1 | FULL | Higher accuracy | Slower |

The `pipeline_review.md` mention of `model_complexity=2` refers to **Holistic** (different API), which you've correctly skipped (deprecated, worse hand accuracy than standalone Hands). **Sticking with Hands `model_complexity=1` is the right call** — it's the highest accuracy option for this solution.

---

## Additional Suggestions for Detection/Extraction

### 1. Denoising / Unsharp Mask (Optional)

Before CLAHE, a mild unsharp mask can sharpen hand edges:
```python
gaussian = cv2.GaussianBlur(frame, (0, 0), 1.0)
frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
```
Test on a subset first; can over-sharpen noisy webcam footage.

### 2. Slight CLAHE Tuning

If some classes still fail, try `clipLimit=2.5` or `tileGridSize=(4, 4)` for stronger local contrast. Start with current values; only adjust if needed.

### 3. Fallback: Lower min_detection_conf for Marginal Cases

Per your inviolable rules, 0.80 is required for train/inference alignment. **Do not lower it globally.** If you ever wanted a *diagnostic* pass to see how many extra clips would pass at 0.70, run a separate script — but don't change the main pipeline.

### 4. Process Order for Retries

Videos that fail v7.0 could be logged to a list. A second pass could try:
- `model_complexity=0` (different model, sometimes catches different cases)
- 640px resolution
- `clipLimit=3.0` CLAHE

as a last-resort recovery. Not implemented in v7; just an idea for future.

### 5. Per-Frame Confidence in Output (Future)

If you ever extend the .npy format to store per-frame detection confidence, training could downweight low-confidence frames. Not in scope for v7.

---

## Extraction Time Estimate

**Videos:** 62,576 (all pending; ASL_landmarks_float16 was cleared)

**v7.0 pipeline per video:**
- 2× MediaPipe passes (video + static)
- 512px resize (vs 384px)
- CLAHE per frame
- Adaptive skip: short videos process *every* frame (more work)

**Rough timing:**
- Old pipeline: ~1.1–1.2 s/video single-threaded, ~2.5–3.5 h with 8 workers for ~62k
- v7.0: ~2–2.5× per video → **~5–8 hours with 8 workers** for 62,576 videos

Actual runtime depends on CPU, video lengths, and I/O. Plan for **~6 hours** as a middle estimate.
