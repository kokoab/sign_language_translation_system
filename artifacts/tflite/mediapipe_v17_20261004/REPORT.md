# Android (MediaPipe) chain → LiteRT/TFLite: accuracy, speed, size (2026-10-04)

All five neural models of the words-only Android chain were converted from PyTorch with
`litert-torch` (isolated env `artifacts/generated/litert_env`, torch 2.13) by
`scripts/export_tflite_mediapipe_v17.py`. FP32 is the shipping candidate; dynamic-range int8 was
tried for each and is reported but not adopted (none met the exact-parity bar). MediaPipe's own
hand/pose/face `.task` files are already TFLite. **No phone was available: every timing below is an
Apple M4 Mac, not Android.** Details and hashes: `manifest.json`.

## Accuracy (conversion parity)

| Model | Check (MediaPipe validation/tuning data) | FP32 | int8 (dynamic range) |
|---|---|---|---|
| Boundary student | per-frame O/B/I argmax, 5,161 tuning frames | **0 changes** (max Δlogit 2.6e-6) | 13 changes |
| Span recognizer (batch 8) | top-1 on 4,232 isolated val clips | **0 changes**; 94.44 / 88.09 / 97.74 % = PyTorch | 6 changes (SemLex −0.10, local −0.03 pp) |
| MobileCLIP2-S0 hand encoder | cosine vs PyTorch, 600 val crops | **≥ 0.9999998** | broken (min cosine 0.007) |
| Stage 3 T5 (encoder + decoder step) | greedy output, 200 corpus validation rows | **200/200 identical** | 192/200 |

End to end, the full live replay with the TFLite models (MediaPipe detector, TFLite boundary,
recognizer and hand encoder) gives **identical words to the PyTorch pipeline on every video**:
tuning pool 89/89 (13.27% WER) and held-out 72/72 (**8.60% WER**; local60 1.85%, unseen ASLLRP12
54.17%). Conversion therefore costs no accuracy; the accuracy picture is the rebuild report's
(`artifacts/reports/mediapipe_rebuild_v17_20261004/REPORT.md`), including the unseen-signer gap.

## Speed (Mac M4 CPU, LiteRT 2.2 interpreter, XNNPACK)

| Model | per call, 1 thread | per call, 4 threads | calls per frame |
|---|---:|---:|---:|
| MediaPipe hands + pose/face (Mac GPU) | — | 5.8 ms (21 ms on CPU) | 1 (aux every 8th) |
| Boundary student | 0.63 ms | 1.17 ms | 1 |
| Span recognizer, batch of 8 spans | 16.8 ms | 16.3 ms | ~0.25 |
| MobileCLIP2-S0, one hand crop | 77.8 ms | 39.7 ms | ~1.6 |
| Stage 3 T5, one sentence (on Finish) | — | 112 ms (5.3 ms/token) | per sentence |

Per processed frame (measured call counts × LiteRT costs, plus measured MediaPipe and overhead):
**≈ 85–90 ms on the M4 CPU**, against a 50 ms budget at 20 Hz, of which **≈ 64 ms is the
MobileCLIP hand-crop encoder**. The same chain with the PyTorch recognizer and the Core ML hand encoder
runs 52 ms/frame on this Mac (Apple chain 50 ms). (TensorFlow 2.16's bundled interpreter, used for the
parity replays, is 5–8× slower than LiteRT 2.2 and measured ~300 ms/frame; that is not the Android
runtime.) For reference, the iPhone 13 encodes the three crops in ~31 ms on its Neural Engine.

**Implication for a 4 GB Android phone.** A low-end phone CPU is several times slower than an M4, so
the full chain cannot hold 20 Hz on CPU. The hand-crop encoder must run on the phone GPU (LiteRT GPU
delegate; MobileCLIP2-S0 is a mobile CNN/attention hybrid designed for this) or the Android build must
use the landmark-only fallback, which removes the 64 ms and leaves ≈ 20–25 ms/frame on the M4 CPU
(this needs a span recognizer trained without hand-crop inputs; the current one consumes them).
This is the decision deferred to on-device measurement.

## Size and memory

| Model | FP32 file | int8 file | RAM added (FP32, CPU) |
|---|---:|---:|---:|
| Boundary student | 9.3 MB | 2.6 MB | 22 MB |
| Span recognizer | 49.6 MB | 13.9 MB | 111 MB |
| MobileCLIP2-S0 | 46.1 MB | (12.6 MB, unusable) | 113 MB |
| T5 encoder + decoder step | 128.6 MB | 33.9 MB | 132 MB |
| MediaPipe hand + pose lite + face | 17.4 MB | — | — |
| **Total** | **≈ 251 MB** | — | **≈ 378 MB** (+ MediaPipe) |

RAM is ~2× file size because XNNPACK repacks weights; the GPU delegate holds its own copies. This fits
a 4 GB phone but is heavy. If memory or download size matters, int8 span recognizer (−36 MB file, 6/4,232
predictions changed) and int8 T5 (−95 MB, 8/200 renderings changed) are the candidates to test on-device;
the T5 decoder and encoder also duplicate the vocabulary embedding (a shared-weight export would cut ~40 MB).

## Files

`boundary_mediapipe_fp32.tflite`, `span_recognizer_mediapipe_b8_fp32.tflite`,
`mobileclip2_s0_image_fp32.tflite`, `stage3_t5_encoder_fp32.tflite`, `stage3_t5_decoder_step_fp32.tflite`
(+ `*_int8.tflite` candidates). Input/output contracts, SHA-256, parity and timings: `manifest.json`.
Runtime: `active/v17/segmental_runtime_v17.py --backend tflite` with
`artifacts/reports/mediapipe_rebuild_v17_20261004/stream_config_mediapipe_v2_tflite.json`.
Replays: `live_replay_{tune,test}_mediapipe_tflite.json`. Profiles: `artifacts/generated/mp_profile_frame_*`.
