# Transformer / Squeezeformer phone feasibility

Measured on physical iPhone 13 with a separate Release benchmark application. Existing matched base-classifier checkpoints; same canonical validation inputs, batch one, 32-frame landmark clips, Core ML compute units ALL. Four counterbalanced passes, 30 warm-up predictions before each measurement. No retraining, protected-test access, production-model replacement or paper changes.

| Model | Precision | Top-1 | Top-5 | Median inference (ms/clip) | Run medians range | P90 (ms) | Package (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | FP32 | 95.50% | 98.94% | 7.28 | 7.25–7.85 | 7.34 | 25.84 |
| Transformer | FP16 | 95.50% | 98.94% | 0.95 | 0.90–1.02 | 1.22 | 13.04 |
| Squeezeformer | FP32 | 96.30% | 98.94% | 8.98 | 8.93–8.99 | 9.05 | 26.23 |
| Squeezeformer | FP16 | 96.30% | 98.94% | 0.99 | 0.90–1.00 | 1.38 | 13.29 |

The Transformer FP32 base classifier was about 19% faster than Squeezeformer FP32,
with 0.79 percentage points lower validation Top-1. Both FP16 exports preserved every
original Top-1 prediction on the phone and Mac; no FP16 accuracy loss was observed here.
The approximately 0.04 ms difference between FP16 medians is small relative to pass
variation, so it does not establish a meaningful speed advantage. Transformer FP32
was still substantially slower than Squeezeformer FP16 in this isolated measurement.
All recorded thermal states were nominal; low-power mode was off.

These results support evaluating Transformer further, but do not establish that the
original architecture choice was wrong. The accuracy-first choice had a small measured
validation advantage in one seed. Nor do these results establish that FP16 was needed
to make the base classifier fast: both FP32 classifiers were under 9 ms at the median.
The deployed recognizer includes additional inputs/components and adaptation, so its
previously measured FP16 accuracy change is a separate result on a different model.

Next safe step: design a matched multimodal/interval-adapted comparison with the same
training inputs, selection protocol and complete phone pipeline before considering a
replacement. Do not start training or consume the protected test from this report.

Timing is synchronous model prediction only, with prepared inputs. It excludes camera capture, landmark extraction, hand-image encoding, boundary detection, decoding, English generation and UI. Median and P90 columns are medians of the four respective per-pass statistics. Package size is export storage, not resident memory.

The original PyTorch and exported-wrapper predictions agreed across validation; empty-input output parity was also checked. Mac FP32 and FP16 conversions preserved all original predictions for both families. Phone parity and thermal states appear in `phone_results.json`; repeated phone predictions and top-five rankings were stable. Low-power mode was off and no pass began or ended in serious/critical thermal state.

This compares the existing flat Transformer and part-wise/global Squeezeformer designs. It is not a pure attention-versus-convolution ablation, a multi-seed accuracy study, or a comparison of complete multimodal streaming systems. FP32/FP16 speed differences include runtime execution placement; they cannot be attributed to precision alone. Do not equate these timings with the manuscript’s historical 28 ms system result.

Toolchain: Xcode 27.0 (27A266a), macOS 27.0, PyTorch 2.8.0, coremltools 9.0, NumPy 1.26.4. Evidence: `PLAN.md`, `prepare.py`, `conversion_summary.json` (checkpoint/package hashes), `phone/Resources/manifest.json` (input hash/reference predictions), `phone/BenchTests.swift`, `phone_test.log`, and `phone_results.json`. The initial unsupported boolean-OR conversion failure is retained in `prepare_failed_mask.log`; an equivalent torch.where mask passed reference-parity checks before successful export.
