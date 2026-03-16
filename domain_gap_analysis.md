# Stage 2 Domain Gap Analysis

## Model Stats
- **Best Val WER:** 1.15% (epoch 11 of 50)
- **Vocab:** 321 glosses (including `<BLANK>`)
- **HOW** = index 132, **YOU** = index 313 — both in vocabulary

## Why 1.15% WER Doesn't Guarantee Real-Video Accuracy

### Training Data (Synthetic)
Stage 2 was trained on `SyntheticCTCDataset` which:
1. Picks 2-6 random glosses
2. Loads each as a pre-processed `.npy` file (exactly `[32, 42, 10]`)
3. Concatenates them: e.g. `[HOW, YOU]` → `[64, 42, 10]`

Each 32-frame clip was produced by `extract.py` which temporally resampled
the full sign video to exactly 32 frames. So one sign = exactly one
32-frame chunk, every time.

### Inference Data (Real Continuous Video)
A real video at 30fps with 2 signs doesn't split neatly at frame 32.
- Sign boundaries don't align to 32-frame chunk edges
- There are transition movements (hand repositioning) between signs
- Each sign may occupy more or fewer than 32 raw frames

### The Gap
| Aspect | Training | Inference |
|--------|----------|-----------|
| Frames per sign | Always exactly 32 | Variable (20-60+) |
| Sign boundaries | Perfectly aligned at 32n | Arbitrary |
| Transition frames | None | Present between signs |
| Temporal density | Normalized (speed-invariant) | Raw frame-rate |

The model learned a world where every sign is exactly 32 frames with
clean boundaries. Real video violates both assumptions.

### What This Means
The code pipeline is now correct. If predictions are still wrong after
fixing protobuf + ffmpeg, the issue is training data, not code. Possible
improvements would be:
- Train on real continuous video with annotated gloss boundaries
- Data augmentation: jitter sign boundaries, insert transition frames
- Use a sliding-window approach with overlapping chunks instead of hard 32-frame splits
