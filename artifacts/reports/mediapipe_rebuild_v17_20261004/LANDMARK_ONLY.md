# Android landmark-only mode (no hand images) — 2026-10-04

Why: on the 4 GB target phone (Huawei nova Y70) the MobileCLIP hand-crop encoder costs ~193 ms per crop,
~310 ms per frame; hand images are kept as an optional mode for capable phones (user decision: automatic
first-launch default + manual override, both modes bundled).

Shortcut rejected first: the hand-image recognizer with hand views masked kept isolated accuracy but the
tuning-pool live replay fell from 13.3% to 24.8% WER (deletions 15 -> 38).

Model: `artifacts/models/mp_span_recognizer_v17_local_a_landmark_only_20261004/best_model.pth`
(v17 landmark format). Same span recipe as the hand-image recognizer (6,254 non-prefix local spans, 8
epochs, seed 27927, isolated replay with base KD, tuning-pool decode WER selection at alpha 1.0, floors =
base - 1, SemLex kept at base 86.04), started from the MediaPipe landmark branch (reel_v2 adapts a fusion
head and has no landmark-only counterpart, so it is skipped). Selected epoch 4 (tuning decode 26.1%).

| | Apple span | Android hand images | **Android landmark-only** |
|---|---:|---:|---:|
| Citizen / SemLex / local isolated (missing clips = errors) | 95.24 / 89.67 / 96.31 | 94.44 / 87.73 / 97.20 | **93.92 / 85.69 / 95.79** |
| Tuning pool live WER (89 videos) | 10.18% | 13.27% | **17.70%** |
| Held-out 72/186 live WER (replayed once) | 11.83% | 8.60% | **9.14%** (gate <= 16.83%) |
| — local60 / ASLLRP12 | 8.64% / 33.3% | 1.85% / 54.2% | 4.32% / **41.7%** |
| Precision / recall (held-out) | 92.6 / 94.6 | 97.2 / 93.0 | 94.6 / 94.1 |
| Mac compute per frame (median) | 50 ms | 52 ms | **24 ms** |

TFLite FP32 (`span_recognizer_landmark_only_b8_fp32.tflite`, 27.8 MB): 0/4,232 top-1 changes vs PyTorch
(int8: 16 changes, rejected). Phone: 108.8 ms per batch of 8 on the 4 big cores (GPU 401 ms), 75 MB peak
(hand-image recognizer 150.6 ms, 121 MB). Per frame on the phone: hand tracking ~28-34 ms (mostly GPU) +
boundary ~9 ms + spans ~27 ms average (bursty) ≈ 70 ms sequential, or ~36 ms CPU overlapped with ~26 ms GPU
— within 20 Hz when camera/GPU work is pipelined with CPU scoring. Early-display guard recomputed (14 glosses).
Configs: `stream_config_mediapipe_v2_landmark_only.json` (torch), `..._landmark_only_tflite.json` (app).
