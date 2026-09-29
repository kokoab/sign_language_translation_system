# Saved CTC comparison — 2026-09-30

The user requested validation of the locally saved CTC model for the capstone comparison.
No retraining, threshold changes, model selection, or production changes were made.

## Protocol

The evaluator loaded the general selector and Core ML packages selected by the defaults
of `scripts/live_stage2_ctc_v17.py`. It verified all 72 video hashes and matched IDs and
reference transcripts exactly to `score_test_C.json`. It replayed input at the CTC path's
native 30 Hz, using its elapsed 32/30-second windows, frozen encoder, primary/specialist
selection, greedy collapse, prefix rollover, and final partial-window processing.
The current pipeline's saved result retains its native input rate. Comparison is of
complete pipelines on common recordings, not an isolated change of decoder. All
execution occurred on Mac. Wall time of this run is not a phone or live response metric.

`protocol.json` records model/package hashes, vocabulary, selector configuration, video
membership and hashes. `result.json` and `predictions.jsonl` preserve every prediction.
No protected isolated-sign test videos were read. The existing phrase verifier is used
by the shared row loader. Evaluation took approximately 172 seconds after model loading.

## Training-membership audit

The saved CTC specialist references `active/v17/stage2_training_manifest_v17.json`.
Its SHA256 matches the hash in the checkpoint. Exact video-hash joins identify 51 of
the local60 videos as train and 9 as validation; the remaining 12 unseen-signer videos
are validation. Therefore the original larger-set CTC result is not independent
validation. `training_membership_audit.json` records the join for each video.
Other ancestral training and development sources have not been exhaustively audited.

After the overlap was discovered, all three pipelines were scored on the same 21
historical-validation recordings, containing 47 reference signs. Selection uses the
saved membership roles, never prediction success. Earlier and current predictions
were rescored from saved records without running inference again. References agree
exactly across pipelines. This common subset is a development comparison, not a new
untouched generalization test.

| Common historical-validation subset | Substitutions | Deletions | Insertions | WER |
| --- | ---: | ---: | ---: | ---: |
| Saved CTC selector | 4 | 6 | 1 | 23.40% |
| Earlier boundary-based recognition | 1 | 21 | 1 | 48.94% |
| Current Swift system | 2 | 6 | 2 | 21.28% |

`common_validation.json` contains the subset IDs, rescored predictions and metrics.
These are the values used in the revised paper-review chart.

## Larger-set results retained for audit

On all 72 recordings / 186 reference signs, CTC returned 4 substitutions, 6 deletions
and 2 insertions: 6.45% WER. Current Swift WER was 9.68%; earlier boundary WER39.78%.
CTC familiar-signer WER1.23%, unseen-signer WER41.67%; current6.17% and33.33%.
The larger-set CTC total includes known training recordings and must not be presented
as independent validation or substituted into the common-subset table.

## Interpretation of historical CTC selection

The older general selector is distinct from later repaired causal and Finish-time CTC
checkpoints. September2 live records show roughly1.07seconds of input before its first
full-window inference, revisable early outputs and a stable-prefix speech policy.
The initial camera timing mismatch was later repaired; it is a historical development
finding, not a claim that the present replay still has that defect. September16 records
also document a held-sign duplicate before rollover. Later causal checkpoints had high
connected-validation errors. These findings support discussion of live interaction and
variant-specific accuracy, not blanket inferiority of CTC as a method.

Source: `docs/ground_truth/live-streaming/log.md`, September2 and September16 entries;
`docs/ground_truth/stage2-ctc/log.md`; original reports linked in those entries.

## Reproduction and checks

Run `venv/bin/python scripts/evaluate_capstone_ctc_comparison_v17.py --output <new-directory>`.
The evaluator refuses to overwrite an existing output directory. The common-subset
audit can also be rerun separately without inference. Five CTC prefix tests and13live
CTC contract tests pass. Source/input hashes and reference membership were checked;
WER is recomputed from edit counts. No model or application files were changed.
