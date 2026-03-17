# Fix: Stage 2 Continuous Sequence (No Temporal Resampling)

## Problem
`extract_features_from_video()` squeezed the entire video to `[32, 42, 10]` via `temporal_resample()`. Stage 2 expects a **continuous** sequence where each 32-frame chunk = one sign. A 130-frame video should yield `[1, 160, 42, 10]` (5 clips), not `[1, 32, 42, 10]` (1 clip with crushed derivatives).

## What Changed

### `extract_features_from_video()`
- **Removed** `temporal_resample()` call — all frames are kept
- **Removed** frame skipping (`skip`, `use_static`) — every frame is processed
- **Set** `static_image_mode=False` — enables MediaPipe tracking mode for consecutive frames (faster + smoother)
- **Added** pad-to-32 at the end — `np.pad(..., mode='edge')` so total is divisible by 32 (Stage 2 does `.view(num_clips, 32, 42, 10)`)
- **Returns** `[1, Total_Frames, 42, 10]` (4D) instead of `[1, 1, 32, 42, 10]` (5D)

### `run_stage2_recognition()`
- Removed the 5D-to-4D reshape since input is now already 4D

## Data Flow
```
Video (130 frames @ 30fps)
  -> ffmpeg CFR re-encode
  -> MediaPipe Hands on ALL frames (no skip)
  -> interpolate missing hand detections
  -> normalize (wrist-center + bone-length scale)
  -> central-difference velocity + acceleration
  -> pad to 160 frames (nearest multiple of 32)
  -> [1, 160, 42, 10]
  -> Stage 2 internally splits into 5 x [32, 42, 10] clips
```
