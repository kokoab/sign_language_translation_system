# Stage 3 candidate latency on iPhone 13 — 2026-09-29

User requirement: a Finish covering **multiple sentences** must translate in **under one second**.
Step 2 of the Stage 3 plan: before any training, time candidate architectures on the physical
iPhone 13 (iOS 26.7) to learn which can meet the budget. No training; no renderer or app change.

## Method

- `scripts/bench_stage3_latency_export_v17.py` builds each candidate architecture with **random
  weights** (latency depends on shapes, precision and decoding scheme, not trained values) and exports
  two Core ML graphs, matching how the phone generates:
  - **T5**: encoder → per-layer cross-attention K/V, then a decoder step per output token, either
    `nocache` (reruns all 64 decoder positions every token, the deployed scheme) or `kv`
    (self-attention K/V held in stateful Core ML state, one position per call).
  - **Decoder-only LM**: 64-token prompt prefill, then a stateful one-token step (128 positions).
- Correctness checks on the Mac: the T5 `kv` step reproduces the `nocache` step token for token over 12
  steps; the LM stateful step reproduces full prefill recomputation over 10 steps. Exported LM parameter
  counts match the published models (SmolLM2 134.5M/361.8M, Gemma-3 268.1M).
- The real deployed v2 packages were timed alongside as a reference. The rebuilt tiny `nocache` model
  times the same (8.4 vs 8.5 ms per token), so the benchmark modules are faithful.
- Hosted XCTest `RunnerTests.testStage3CandidateLatency` loads packages from the app's Documents,
  compiles on device, times the first graph and every step (2 warm-up generations, 8×16-step and
  3×48-step runs) under `.cpuAndGPU` and `.all` (Neural Engine allowed). Idle phone, no camera.
- Output lengths from the multi-sentence evaluation set (T5 tokenizer; Qwen and SmolLM2 tokenizers give
  the same counts): one sentence p90 **11** tokens, whole session median **19**, p90 **28**, max 44.
  Estimate = first-graph median + N × per-token median. `measured_48_ms` is a direct measurement.

## Results (median ms, idle iPhone 13, best compute-unit setting per model)

| Model | Params | Setting | Per token | 1 sentence (p90) | Session median | Session p90 |
|---|---:|---|---:|---:|---:|---:|
| tiny T5, KV cache, FP16 | 15.6M | all | 6.1 | 70 | 119 | **173** |
| **deployed v2** (tiny, no cache, FP32) | 15.6M | cpuAndGPU | 8.5 | 97 | 165 | **241** |
| t5-small, KV, FP16 | 60.5M | cpuAndGPU | 8.2 | 106 | 172 | **246** |
| flan-t5-small, KV, FP16 | 77M | all | 8.7 | 101 | 170 | **249** |
| flan-t5-base, KV, FP16 | 248M | all | 13.6 | 159 | 268 | **391** |
| SmolLM2-135M, KV, FP16 | 135M | cpuAndGPU | 14.6 | 193 | 310 | 441 |
| Qwen2.5-0.5B, KV, int4 weights | 494M | cpuAndGPU | 15.9 | 232 | 359 | **502** |
| flan-t5-small, no cache, FP32 | 77M | all | 23.7 | 279 | 469 | 682 |
| flan-t5-base, KV, int8 weights | 248M | all | 26.8 | 362 | 577 | 818 |
| SmolLM2-360M, KV, FP16 | 362M | cpuAndGPU | 31.2 | 394 | 644 | 925 |
| Gemma-3-270M, KV, int4 | 268M | all | 39.9 | 531 | 850 | 1210 |
| Gemma-3-270M, KV, FP16 | 268M | cpuAndGPU | 42.4 | 583 | 923 | 1305 |
| Qwen2.5-0.5B, KV, FP16 | 494M | — | — | — | — | **app killed while loading** |

All 29 timed rows (both settings) are in `summary.json`; failures in `failures.json`.

Failures: Qwen2.5-0.5B FP16 killed the app process on load twice (the package holds its ~1 GB of FP16
weights in both graphs, beyond the iPhone 13 per-app memory limit). SmolLM2-135M and -360M fail to
compile for the Neural Engine (`.all`) and run on the GPU only.

## Findings

1. **The budget is already tight for multiple sentences.** The deployed tiny model needs about 241 ms
   for a p90 session on an idle phone. Live Finish logs show 318–335 ms for v2's first Finish of a
   session on a short two-sentence output, versus about 110 ms predicted idle. The live number includes
   the camera and recognizer running concurrently, and the app does not warm Stage 3 after loading. So
   live latency runs roughly 1–3× the idle figures; budget with that headroom.
2. **A KV cache matters for anything larger than tiny.** flan-t5-small: 682 → 249 ms per p90 session.
   For the tiny model the gain is small (241 → 173 ms), because per-call overhead dominates.
3. **Weight quantization is slower on this phone** for T5 (flan-t5-base int8 818 vs FP16 391 ms). Int4 is
   what makes Qwen2.5-0.5B loadable at all.
4. **Fits whole-buffer translation with ~3× headroom (≤ ~330 ms idle p90):** tiny, t5-small,
   flan-t5-small. **Fits with ~2–2.5× headroom:** flan-t5-base (391), Qwen2.5-0.5B int4 (502).
   **Out:** Gemma-3-270M (1.2–1.3 s; its 262K-token output layer dominates), SmolLM2-360M (925),
   Qwen FP16 (does not load).
5. **Translating incrementally makes the larger models safe.** If finished sentences are translated
   while the user is still signing, Finish only renders the last one: flan-t5-base 159 ms, Qwen int4
   232 ms at p90, which is 4× or more headroom.

## Limits

Random weights: numerical behaviour of trained weights (in particular FP16 overflow in T5 v1.1/FLAN
activations) is untested and must be checked with real weights before choosing FP16. Idle-phone timing
only; not measured under camera + recognition load, sustained use or thermal throttling. Package sizes
here double-count weights shared by the two graphs. iPhone 13 only.

## Phone state

Benchmark packages were copied to the app's `Documents/stage3_bench`, then removed by the test itself
(a few KB of JSON remain). The signed Release app was rebuilt without testability, codesign verified,
and reinstalled; all 10 saved Live sessions are intact. The test added to
`ios/RunnerTests/RunnerTests.swift` skips when no benchmark manifest is present; the pre-edit file is
`RunnerTests.swift.before`.

Random-weight packages: `artifacts/coreml/stage3_latency_bench_v17/` (7.6 GB, regenerable, never usable
as renderers).
