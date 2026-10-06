# Matched iPhone precision comparison — 2026-10-07

## Outcome

The physical iPhone 13 benchmark passed. The selected configuration averaged 24.05 ms across the two run medians, versus 57.43 ms with only the recognizer replaced by FP32, and 112.64 ms with all three visual/recognition components in FP32. This does not support assuming that an FP32 replacement retains the historical 28 ms processing time.

Restricting the same FP16 recognizer to CPU/GPU raised the measured time to 57.47 ms. The deployment advantage therefore involves both numerical precision and the execution resources Core ML can use. These timings do not isolate arithmetic precision from hardware placement, and the factors should not be treated as additive speedups.

## Device results

Same physical iPhone 13; same 226 recorded-input frames per configuration per run, rendered at 1280×720 from the bundled HELLO, MY, NAME, HOW, YOU, PLEASE, HELP, THANKYOU examples. Two runs in forward/reverse configuration order, 20 warm-up frames and model warm-up before each measured run. No protected evaluation recordings were used. All runs reported nominal thermal state and low-power mode disabled.

| Configuration | Run medians (ms/frame) | Median of run medians (ms/frame) |
| --- | ---: | ---: |
| selected_fp16_all | 24.40 / 23.70 | 24.05 |
| recognizer_fp32_only | 57.95 / 56.91 | 57.43 |
| encoder_fp32_only | 87.17 / 87.40 | 87.29 |
| all_visual_fp32 | 110.34 / 114.94 | 112.64 |
| selected_recognizer_cpu_gpu | 57.67 / 57.27 | 57.47 |

Configurations change the hand-image encoder, fixed-batch recognizer and word-boundary export as named; all use the same model checkpoints. `selected_fp16_all` uses FP16 for those components and permits all compute resources. `recognizer_fp32_only` and `encoder_fp32_only` change only the named export. `all_visual_fp32` changes all three. `selected_recognizer_cpu_gpu` keeps the FP16 exports but restricts recognizer execution. The unused letter boundary remains unchanged; the test uses words-only mode. English generation is not timed or changed.

Timing covers the engine's frame preparation and recognition call, including landmark extraction and hand-image processing. It excludes camera capture, video decoding/rendering performed before timing, model loading, UI display work and English/speech output. This short recorded-input workload does not establish sustained live-camera performance or end-to-end response time. Model outputs may change with precision, affecting decoder work; timings describe the resulting integrated configurations, not identical operator traces.

The historical 28 ms result remains a separately recorded measurement. This run used the current source and explicit warm-up/reversed ordering without the historical separate stage probes; it measured approximately 24 ms for the selected configuration. Do not silently overwrite the historical result or describe the difference as a new optimization.

## Recognition conversion check

The new FP32 fixed-batch recognizer uses the original recognizer and auxiliary head checkpoints. On all378validation examples, with identical cached landmark/hand inputs and the final partial batch padded, FP32 matched the original PyTorch top-1 predictions exactly:360/378correct (95.2381%); Top5 374/378 (98.9418%). This is a Mac conversion check, not on-phone accuracy or complete live-pipeline accuracy. No training or protected test access.

The saved September30FP16 check scored359/378correct (94.9735%) on those inputs. FP32 retains one more correct prediction in this check; these results alone do not establish statistically reliable superiority in live communication. The new FP32 boundary export agreed on all35checked tuning windows; this is export agreement, not annotated-boundary accuracy. The exporter's recognizer parity loop excludes the final partial batch, so the separate378-example check supplies full coverage.

## Interpretation

FP16's value here extends beyond the approximately28.68MB storage saving for recognizer and boundary packages. A recognizer-only FP32 substitution increased the combined median frame-processing cost by about2.39times; all-visual FP32 increased it about4.68times. Retaining the current deployment is supported by this short timing comparison, while the small validation difference remains documented. No production model-selection change was made.

## Reproducibility and state

Initial device launch failed because the developer certificate was untrusted. Following the user's continuation, the compiled XCTest ran successfully in193.502seconds. One initial export command referenced a nonexistent letter-head filename; corrected before successful export. Original engine, XCTest source and Xcode project were restored exactly against source_hashes_before.json. New model exports are additive. Production-source Release rebuild and phone reinstallation both succeeded, recorded in restore_build.log and restore_install.log. The default model selection remains unchanged.

Artifacts: device_retry.log, device_samples.json (per-frame timing), device_summary.json, export_recognizer.log, export_boundary.log, recognizer_summary.json, recognizer_predictions.json, fp32_validation.log, prepare_benchmark.py, RunnerTests_benchmark.swift, source_hashes_before.json, source_backup/, benchmark source snapshots. FP32 model packages are under artifacts/coreml/.

No manuscript, presentation, datasets, training, thresholds or default model choices were changed. Detailed experiment records stay in the repository; they are not automatically added to the capstone paper.
