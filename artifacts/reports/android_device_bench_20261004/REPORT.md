# On-device benchmark: Huawei nova Y70 (MGA-LX9) — 4 GB Android target (2026-10-04)

Device (read over adb): Android 10 / API 29, Kirin 710 (4× Cortex-A73 2.0 GHz + 4× Cortex-A53 1.7 GHz),
Mali-G51 (OpenGL ES 3.2, OpenCL present), 3.6 GB usable RAM (~1.7 GB free at test time).
Tool: LiteRT `benchmark_model` (android_aarch64 nightly), 30 runs after 3 warm-ups, random inputs,
cores pinned with `taskset` (f0 = big, 0f = little). GPU = LiteRT GPU delegate (OpenCL). Models are the
FP32 exports in `artifacts/tflite/mediapipe_v17_20261004/` plus MediaPipe's own task sub-models.
Raw logs: `logs/`; table: `summary.json`; sustained run: `sustained_concurrent.json`.

## Per-call latency (ms, average)

| Model | 1 big core | 4 big cores | 4 little cores | GPU (nodes on GPU) | Best | Peak MB |
|---|---:|---:|---:|---:|---|---:|
| MediaPipe hand landmarks | 255.0 | 36.0 | 47.9 | **19.6** (165/165) | GPU | 30–69 |
| MediaPipe palm detector (only when tracking is lost) | 95.7 | 32.0 | 75.0 | **26.2** (272/272) | GPU | 33–58 |
| MediaPipe pose detector (every 8th frame) | 82.1 | 27.7 | 96.1 | 27.0 | GPU/CPU | 33–78 |
| MediaPipe pose landmarks (every 8th) | 67.5 | 22.8 | 49.4 | 19.6 | GPU | 42–65 |
| MediaPipe face detector (every 8th) | 9.3 | **3.0** | 7.0 | 5.7 | CPU | 11 |
| MediaPipe face landmarks (every 8th) | 38.3 | 16.7 | 24.8 | **11.2** | GPU | 26–58 |
| Boundary student | 33.2 | **9.2** | 17.6 | 36.9 (23/259) | CPU | 27 |
| Span recognizer, batch 8 | 530.9 | **150.6** | 315.2 | 531.0 (758/2477) | CPU | 121 |
| Span recognizer int8, batch 8 | 328.7 | **113.8** | 288.8 | 341.2 | CPU | 53 |
| **MobileCLIP2-S0, one hand crop** | 655.9 | **192.9** | 363.1 | 540.0 (198/498) | CPU | 112 |
| T5 encoder | 41.4 | **12.9** | 31.5 | 47.0 (19/199) | CPU | 34 |
| T5 decoder step (per token) | 160.1 | **42.6** | 71.2 | fails (43/321) | CPU | 113 |
| T5 encoder / decoder int8 | 24.0 / 86.6 | **8.1 / 23.5** | 22.3 / 66.0 | — | CPU | 16 / 44 |

Only MediaPipe's models are fully GPU-compatible. Our transformer/attention exports fall back to the CPU
for most operations under the Mali GPU delegate and are slower there than on the 4 big cores.

## Sustained concurrent load (hand landmarks on GPU + span recognizer on big cores, 6 × 30 s)

Hand landmarks 30.1 → 22.6 ms (stable); span recognizer 173.7 → 208.8 ms (+20%); battery 35 → 38 °C.
Mild throttling, no collapse.

## Per-frame budget at 20 Hz (50 ms), from measured costs × measured call rates

| Item | ms per frame |
|---|---:|
| Hand landmarks (GPU), palm detector occasionally | ~20–26 |
| Pose + face, amortised over 8 frames | ~8 |
| Boundary student (CPU) | ~9 |
| Span recognizer: ~0.25 batch calls/frame × 150–210 ms (bursty) | ~40–50 |
| **MobileCLIP hand crops: ~1.6 crops/frame × 193 ms** | **~310** |
| **Full chain (as shipped on iPhone)** | **~390 → ~2.5 fps** |
| Without the hand-crop encoder | ~80 → ~12 fps |

Stage 3 on Finish: FP32 ≈ 13 ms + ~43 ms/token → ~0.9 s for 20 tokens (int8 ≈ 0.5 s; output 192/200
identical to FP32); a KV-cache export would cut the per-token cost further.

## Conclusion

The full crop-based chain is ~8× over budget on this phone; the MobileCLIP hand-crop encoder alone is
~6× the whole frame budget, and the Mali-G51 GPU delegate does not accelerate it. A landmark-only
recognizer (no hand crops) brings the frame to ~80 ms with the current span recognizer, and a
landmark-only span model would be lighter still (target ≤ 50 ms with fewer span calls or batch 4).
Memory is not the constraint: all models together peak well under the ~1.7 GB available.
Accuracy, conversion parity and the unseen-signer caveat are in
`artifacts/tflite/mediapipe_v17_20261004/REPORT.md` and `artifacts/reports/mediapipe_rebuild_v17_20261004/REPORT.md`.
