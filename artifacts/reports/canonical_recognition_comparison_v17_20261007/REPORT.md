# Pinned recognition comparison — 2026-10-07

The historical part-wise checkpoint still reproduces 96.83% validation Top-1. Its FP32 and FP16 phone exports preserve all original predictions. These are selected existing checkpoints evaluated on identical inputs, not a new matched from-scratch training experiment. No protected test was accessed.

| Model / stage | Validation Top-1 | Top-5 | Mac CPU inference (ms) |
| --- | ---: | ---: | ---: |
| flat transformer | 95.50% | 98.94% | 2.58 |
| flat squeezeformer | 95.77% | 100.00% | 4.51 |
| partwise global squeezeformer | 96.83% | 99.21% | 5.97 |
| adapted landmark branch | 95.50% | 98.68% | 6.01 |
| combined recognizer | 96.30% | 99.21% | 8.87 |
| phrase activity adapted | 96.03% | 99.21% | 8.89 |
| interval recognizer | 95.24% | 98.94% | 8.88 |

Mac timings use FP32 PyTorch, one CPU thread, batch one, identical prepared validation inputs, 30 warm-ups and three ordered passes. Values are medians of run medians. The initial timings overlapped export preparation and are excluded; `verified_results.json` contains the isolated rerun and exact checkpoint/input hashes. Timing excludes extraction, camera, image encoding, decoder, English and UI.

## Physical iPhone 13

| Classifier | Top-1, both precisions | FP32 median (ms) | FP16 median (ms) |
| --- | ---: | ---: | ---: |
| Transformer | 95.50% | 7.20 | 0.95 |
| FlatSqueezeformer | 95.77% | 7.16 | 0.89 |
| PartwiseSqueezeformer | 96.83% | 8.88 | 0.99 |

Six rotating orders place every configuration in every position. Core ML ALL, prepared inputs, 30 warm-ups per measurement, synchronous prediction only. All recorded thermal states nominal, low-power off, all predictions unchanged from PyTorch and Mac conversion checks. Small FP16 timing differences should be interpreted with per-pass variability, available in `phone_results.json`. These are not full-system 28 ms timings.

## Correct training-stage interpretation

Flat Squeezeformer exceeds flat Transformer by 0.26 percentage points in these selected checkpoints. Part-wise/global Squeezeformer improves on the flat Squeezeformer by 1.06 points and is the most accurate base classifier here. Different historical training runs and selection histories mean this is not proof of a universal architecture advantage.

The deployed branch initializes from the orientation-robust checkpoint, which is a separate part-wise model, rather than directly from the historical 96.83% weights. The local-adapted landmark branch is embedded in the unified multimodal recognizer; phrase/activity adaptation starts from that unified model, and interval adaptation starts from phrase/activity adaptation. Source checkpoint hashes in these artifacts were checked where provided. Preserve this intermediate orientation/local adaptation in any lineage explanation; do not draw an unsupported direct weight-inheritance arrow.

Later stages optimize additional domains or candidate intervals. Their isolated-sign Top-1 scores are not monotonically increasing; do not transfer 96.83% to those stages or claim a stream-level gain from this isolated evaluation.

## Next authorized experiment

See `TRAINING_PLAN.md`: current trainer, strict warm-start from the preserved96.83% checkpoint, mild-roll control versus full-circle rotation treatment, equal20epoch budgets. Check `finetune/status.json` for completion before assuming results. The original remains the fallback. Full downstream retraining and manuscript updates remain pending; this report does not claim those tasks are complete.

Phone installation first hit the free-profile app limit. Reusing the previously created benchmark bundle and a fresh build directory resolved it; production app/models stayed unchanged. Failed logs are preserved. No manuscript edits were made during this work.
