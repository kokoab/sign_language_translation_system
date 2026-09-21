# v16 local phrase history review

The exact approximately15% local WER was not located in available targeted v16 logs.
The strongest located historical summary is docs/md_files/SESSION_BRIEFING.md:72:
117held-out files,20.2% overall mean per-clip WER and56.4% exact;75multi-clip files
(T>=64)9.1% WER/73.3% exact;42single-clip files approximately45% WER/30% exact.
These are archived documented results, not newly rerun or independently reproduced.

Evaluation provenance: scripts/stage2_eval_suite.py:87 uses seed42 random15% file
holdout, explicitly described at line169 as same signer as train. It sums individual
clip WER and divides by clip count; current v17 reports aggregate edits/reference tokens.
active/v16/train_stage_2_v16_fixed.py:235 also splits random files, not signer groups.
The historical labels came from PHRASE_GLOSSES/filenames, before user PHRASES FIXED
review. The current local development set is200clips from held-out signer02, not117
random files. The earlier reported score therefore is not a matched v16/v17 comparison.

Architecture differs too: active/v16/model_v16.py:494 describes32-frame clip encoding,
4tokens/clip and4sequence layers; forward at587 passes full sequence through blocks
without causal mask. The current experiment has frozen Apple Stage1 and3causal128dim
blocks over8-frame rolling observations. v16 permits encoder unfreezing; current motion
experiment deliberately freezes the ancestor to isolate transferred motion weights.
These differences may matter; no evidence here attributes the full gap to signer split.

The only located artifacts/model_assets/models/output_stage2_v16/history.json has3
entries (216.46%,141.77%,100% WER), clearly not supporting the good historical result.
The evaluator's checkpoint points to /Users/frnzlo/Downloads/results (1)/models/
output_stage2_v16_phrases/stage2_best_model.pth, which no longer exists at that path.
The archived src_v16.zip holds Stage1history/evaluation, not the missing phrase log.
No exact15% claim can be verified from the located artifacts. Do not relabel synthetic
validation WER as local real-phrase WER or conflate15%holdout fraction with measuredWER.

Next useful comparison: recover the exact v16phrase checkpoint and predictions, establish
training exposure, then evaluate compatible v16extraction on user-reviewed phrase data.
If its training included signer02, that is a seen-signer diagnostic, not an independent
held-out-signer benchmark. No checkpoint was run, no training started, no tests accessed.
