# Pretrained boundary regression: controlled BIO readout experiment

No training, deployment or distillation. Existing weights, splits, reports and frozen Reel preserved.
Same72 development videos/186 signs; familiar-signer reused development, not independent generalization.

## Results

| Backbone / readout | WER | Correct /186 | Retained frozen /127 | Insertions | Exact /72 |
|---|---:|---:|---:|---:|---:|
| Existing frozen_pretrained_bio | 39.78% | 127 | 127 | 15 | 17 |
| Existing finetune_epoch7 | 54.30% | 96 | 77 | 11 | 5 |
| Existing augmented_epoch8 | 53.76% | 100 | 76 | 14 | 2 |
| finetune_epoch7_original_bio | 40.32% | 130 | 124 | 19 | 18 |
| augmented_epoch8_original_bio | 42.47% | 124 | 120 | 17 | 17 |

## Interpretation and minimal change

All72 frozen hypotheses AND interval boundaries reproduce the original saved evaluation exactly. Each adapted checkpoint has an unchanged temporal CNN, input normalization and original BIO head (exact tensor equality); only attention and the edge head differ. Restoring the original BIO readout/decoder isolates the major regression to the replacement readout/decoding path. This does not distinguish edge-head quality from decoder policy, nor prove absence of smaller representation changes.

The original BIO decoder groups B/I activity, accepts three-frame spans and closes at EOF. The edge decoder requires rising START/END pairs, uses 0.15–4.0s duration limits, replaces unmatched starts and drops unfinished EOF intervals. Training optimizes masked, weighted frame BCE, not these discrete interval/commit decisions. EOF alone is not established as the sole cause.

Small code change: scripts/evaluate_boundary_expanded_v17.py run_arm now accepts readout="bio" independently of whether a checkpoint is loaded. Existing defaults remain unchanged. Regression test compares actual adapted-backbone BIO logits against the direct original-head path; invalid readout is rejected. The diagnostic script retains a separate controlled path and full frozen-output reproduction checks.

## Supervision, preprocessing and timing audit

Reconstructed every cached target/index/split/parent: {'train': 30014, 'calibration': 5007, 'validation': 7593}. Unknown labels have exactly zero gradient. Sampled clean projection parity max absolute error 0.00005722. Pose/source hashes pass. No negative supervision was found outside approved intervals; positives can occupy the documented ±50ms edge band. No unknown gap was relabeled background.

Training/evaluation share20Hz past-observation sampling,64-frame windows,target53,10 future frames (500ms), bounded-window normalization and explicit zero-confidence padding. CNN caches include temporal context. Raw/cache paths agree. Augmentation changes past observation holds without moving target timestamps. Existing clock/future/padding tests pass. All arms in the new comparison share identical original decoding,100ms classifier context and commit rules.

## Training-held calibration and stopping decision

Only2 existing calibration videos (4 signs) match complete approved locked-vocabulary sequence targets.129 other calibration records provide boundary supervision but do not establish complete locked-vocabulary whole-video transcripts. Splits were not changed and no OOV region was silently scored as blank.

| Backbone | Readout | Calibration WER | Correct /4 | Insertions |
|---|---|---:|---:|---:|
| frozen_pretrained_bio | bio | 100.00% | 0 | 0 |
| finetune_epoch7 | bio | 100.00% | 0 | 0 |
| finetune_epoch7 | edges | 75.00% | 1 | 0 |
| augmented_epoch8 | bio | 100.00% | 0 | 0 |
| augmented_epoch8 | edges | 75.00% | 1 | 0 |

The predeclared calibration rule chooses finetune_epoch7 / edges, based on one correct sign; it does not select a BIO-adapted checkpoint. No candidate is promoted. Expanded results are descriptive only, not used to override calibration or select an epoch/threshold. This exposes an inadequate recognition-selection signal, so another fine-tune is not justified yet. The bounded experiment used fixed existing weights and zero gradient steps.

## Runtime

Controlled72-video three-arm replay: 489.26s. Calibration five-arm replay: 15.88s. These wall times include some overlapping audit/test work and are not isolated throughput benchmarks.

| Backbone | Readout | Batch1 median ms | p95 ms |
|---|---|---:|---:|
| frozen_pretrained_bio | bio | 30.30 | 42.75 |
| finetune_epoch7 | bio | 30.19 | 38.97 |
| finetune_epoch7 | edges | 31.00 | 48.87 |
| augmented_epoch8 | bio | 31.18 | 44.57 |
| augmented_epoch8 | edges | 31.39 | 40.13 |

Synchronized MPS benchmark,30 measured windows after5 warmups per arm,real cached calibration pose,seeded shuffled interleaving. Initial sequential benchmark (preserved as benchmark.json) drifted from14ms to31ms across equal-size models; the interleaved repeat avoids attributing that order/load drift to model or readout speed. Includes normalization,CNN,attention,readout and CPU output. Excludes MediaPipe frontend,Reel and camera scheduling. Intrinsic future delay remains500ms. This is not a live/iPhone readiness measurement.

Shared72-video Apple Vision frontend: 73.80s; shared bounded normalization/CNN projection: 71.58s.
- frozen_pretrained_bio_original_bio: attention/BIO 8.41s; Reel classification 108.67s.
- finetune_epoch7_original_bio: attention/BIO 8.17s; Reel classification 103.22s.
- augmented_epoch8_original_bio: attention/BIO 7.98s; Reel classification 106.54s.

## Validation and next safe action

23 focused tests pass, including the new original-BIO override test (first failed because the option was absent). All saved edit counts, summaries, video/checkpoint hashes and historical comparison source hashes rechecked. No original result files or weights overwritten. Small audit setup error (some combined records lack source_item_id) was corrected by filtering that field before matching; no model/data changes resulted.

Keep untouched pretrained BIO as the reference. Preserve the original head/decoder in future adaptation comparisons. Resolve complete recognition scoring within the existing calibration split before new checkpoint-selection training; do not select on expanded72 or simply train longer. The fixed-weight readout recovery is measured; a deployable adaptation selected by reliable held-out recognition is not established.
