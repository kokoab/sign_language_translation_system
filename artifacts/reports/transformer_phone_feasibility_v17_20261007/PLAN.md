# Base-classifier phone feasibility — 2026-10-07

User authorized proceeding with the staged Transformer comparison. First gate only:
existing matched family checkpoints, Transformer and part-wise/global Squeezeformer,
each FP32 and FP16. No training, protected-test access, threshold tuning or production
replacement. This compares whole classifier designs (including regional processing),
not a controlled ablation of attention versus convolution alone.

Export batch-one inputs [1,32,61,5] and identical FP32 input/output interfaces. Verify
export-wrapper predictions against original models on the entire canonical validation
split and an empty-input edge case. Save checkpoint, input and package hashes.
Evaluate conversion accuracy on Mac, then load the same input bytes on iPhone13 in a
separate benchmark app. Measure synchronous prediction only; exclude loading, input
allocation, extraction, hand encoder, decoding, English and UI. Four counterbalanced
orders, warm-up before each configuration, capture all timings, top1/top5 predictions,
thermal and low-power state. Report median and p90, per-pass variability, parity and
package storage. FP32 versus FP16 includes runtime placement effects, not precision alone.

A favorable base-classifier result permits planning the next matched fusion/interval
experiment; it does not justify replacing the deployed multimodal recognizer. Keep
production source and app untouched. No manuscript edits until evidence is reviewed.
