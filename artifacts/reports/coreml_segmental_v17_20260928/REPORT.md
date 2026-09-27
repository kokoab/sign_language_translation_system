# Core ML segmental Reel — export, desktop benchmark, hypothetical iPhone 13 (2026-09-28)

## Packages (all parity-checked on non-test data)

| Package | Precision | Size | Parity |
|---|---|---:|---|
| `artifacts/coreml/AVBoundaryStudentV17L6FP16.mlpackage` | FP16, iOS15 | 4.5 MB | 0/1,944 tuning frames top-1 mismatch, max dp .001 |
| `artifacts/coreml/SpanRecognizerV17LocalAFP16.mlpackage` | FP16, batch 1, iOS15 | 23.6 MB | 0/378 Citizen val mismatch |
| `artifacts/coreml/SpanRecognizerV17LocalABatchedFP16.mlpackage` | FP16, batch 1/2/4/8/16, iOS18 | 23.5 MB | 0/378 mismatch; runtime always uses batch 8 |
| `artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage` (existing) | FP32 | 44 MB | reference |
| `artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP16.mlpackage` | FP16 | ~22 MB | per-crop gate FAILED (min cosine .91); end-to-end tune 84/89 identical, 203 vs 205 correct |

Held-out replay through the all-Core-ML runtime (FP32 encoder, batch-8 recognizer): 72/72 identical
hypotheses to the PyTorch path, 11.83% WER, P 92.6, R 94.6.

## Findings that shape the phone build
- Alternating enumerated batch shapes re-plans the model: 283 ms/call vs 15.5 ms fixed batch 8.
  Use one fixed shape.
- On this Mac the recognizer is fastest on GPU (8.9 ms batch 8) and slow on the Neural Engine
  (64 ms); the FP16 hand encoder is fast on the Neural Engine.
- Boolean `~mask` traces to bitwise_not, which Core ML rejects at batch > 1; export traces it as
  logical_not (same result).

## Per-frame cost (tuning videos, 346 frames, real Apple Vision)

| Config | Desktop median / p90 / p99 ms | iPhone 13 estimate median ms (fps) | p90 ms |
|---|---|---|---|
| FP32 encoder (ALL), recognizer GPU | 24.2 / 38.0 / 49.7 | 66-81 (12-15 fps) | 103-125 |
| FP16 encoder (Neural Engine), recognizer GPU | 14.0 / 25.3 / 33.9 | 30-40 (25-33 fps) | 61-77 |

Stages (FP32 config, desktop median/p90): Vision 5.7/12.9, hand crops 13.6/22.5, boundary 1.5/1.9,
span inputs 0/2.2, recognizer 0/8.5 (only on frames with new spans; 0.66 spans/frame), decode <0.3.

## iPhone 13 figures are estimates, not measurements
Each desktop stage is scaled by the hardware it mostly uses (A15 vs M4, slower-is-larger):
CPU 1.6-1.8x, GPU 3.0-3.5x, Neural Engine 2.2-2.6x, Apple Vision 2.0-3.0x. Thermals, memory
pressure, camera capture and UI drawing are not included. Mobile readiness still requires measured
size, memory, cold start, sustained latency and thermals on a real iPhone (locked decision 8).
Benchmark: `scripts/benchmark_segmental_coreml_v17.py`; JSON beside this report.

## v3 dual decoder (letters) — added 2026-09-28 evening
Packages: `AVBoundaryStudentV17L6LettersFP16.mlpackage` (4.5 MB, 0/1,944 mismatch) and
`SpanRecognizerV17LocalALettersBatchedFP16.mlpackage` (24.0 MB; outputs word_logits + letter logits;
0/378 word and 0/312 letter top-1 mismatches). Core ML dual replay = PyTorch dual (89/89 tune clips).

| Config | Desktop median / p90 / p99 ms | iPhone 13 estimate median ms (fps) | p90 ms |
|---|---|---|---|
| Dual, FP32 encoder | 22.8 / 34.0 / 50.8 | 59-76 (13-17 fps) | 92-113 |
| Dual, FP16 encoder (Neural Engine) | 14.7 / 29.6 / 42.3 | 35-44 (22-29 fps) | 75-91 |
