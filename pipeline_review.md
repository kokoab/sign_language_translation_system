# Pipeline Review: `test_video_pipeline.py`

## All Issues Found and Fixed

### 1. VFR Frame-Drop Bug (FIXED)
**Problem:** OpenCV `cap.read()` silently fails on Variable Frame Rate (VFR) videos from phone cameras. Reported 130 frames but only read 30.

**Fix:** Added `reencode_to_cfr()` — ffmpeg re-encodes the video to 30fps CFR before OpenCV touches it. Falls back to direct read if ffmpeg is unavailable. Also switched from `cap.read()` to `cap.grab()` + `cap.retrieve()` (matching `extract.py`).

---

### 2. Wrong MediaPipe Detector (FIXED)
**Problem:** Pipeline used `mp.solutions.holistic.Holistic` (full body). Training used `mp.solutions.hands.Hands` (hands only). Different models produce different coordinate distributions.

**Fix:** Switched to `mp.solutions.hands.Hands` with identical config: `model_complexity=1`, `max_num_hands=2`, `min_detection_confidence=0.65`, handedness-aware left/right separation.

---

### 3. Missing Wrist-Centering + Bone-Length Scaling (FIXED)
**Problem:** Raw MediaPipe coordinates (0-1 range) were fed directly. Training data was normalized: wrist-centered (median of non-zero wrist positions subtracted), then divided by median wrist-to-middle-MCP bone length.

**Fix:** Ported `normalize_sequence()` verbatim from `extract.py`.

---

### 4. Wrong Velocity/Acceleration Formula (FIXED)
**Problem:** Pipeline used forward difference: `vel[t] = xyz[t] - xyz[t-1]`. Training used central difference: `vel[t] = (xyz[t+1] - xyz[t-1]) / 2.0`, with boundary copies `vel[0] = vel[1]`, `vel[-1] = vel[-2]`.

**Fix:** Ported `compute_kinematics()` with central-difference formula from `extract.py`.

---

### 5. Missing Interpolation + Temporal Resampling (FIXED)
**Problem:** Pipeline kept raw frame count and used a sliding window. Training interpolated missing hand detections across gaps, then resampled every video to exactly 32 frames.

**Fix:** Ported `interpolate_hand()` and `temporal_resample()` from `extract.py`. Output is always `[32, 42, 10]` — one clip per video, matching training.

---

### 6. Aspect Ratio Squashing (FIXED)
**Problem:** Pipeline squashed frames to `384x384`. Training used aspect-preserving resize (scale longest edge to 384).

**Fix:** Replaced `cv2.resize(frame, (384, 384))` with aspect-preserving resize logic from `extract.py`.

---

### 7. Stage 2 Model Return Unpacking (FIXED — previous session)
`SLTStage2CTC.forward()` returns `(logits, out_lens)`. Was indexed as a raw tensor. Now properly unpacked.

### 8. Stage 2 Input Shape Mismatch (FIXED — previous session)
5D `[1, W, 32, 42, 10]` reshaped to 4D `[1, W*32, 42, 10]` before model call.

### 9. T5 Prompt Prefix Mismatch (FIXED — previous session)
Changed `"translate ASL to English:"` to `"translate ASL gloss to English:"` matching training.
