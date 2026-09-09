# v2 transition regression: bounded diagnostic evidence

Two existing local familiar-domain validation clips were inspected as 20-frame
timestamped contact sheets. This is visual inspection of sampled frames, not
full-motion playback or an expert ASL semantic admission. Both clips were then
replayed on CPU through the pinned accepted selector, initialized candidate,
and both with-STEM best checkpoints, using existing frozen features.

| Clip | Accepted and initialized | Both with-STEM candidates |
| --- | --- | --- |
| local:PLEASE_HELP_ME:3e65d621 | PLEASE HELP I | PLEASE HELP I I |
| local:HELLO_HOW_YOU:HELLO_HOW_YOU_20 | HELLO HOW YOU | HELLO HOW |

`model_probe.json` records zero-based CTC output-step indices, not video-frame
timestamps. The first candidate repeats I at steps 28 and 36 with blanks between;
the second loses the baseline YOU emission at step 21. Both seeds agree. These
results reproduce the saved report and establish learned emission changes in
these examples; they do not establish why training produced those changes.
The initialized candidate is correct on both, so the aggregate starting-model
mismatch alone does not explain these particular regressions.

The PLEASE HELP I sampled frames show a prolonged final chest-pointing posture;
the HELLO HOW YOU sampled frames show the final forward-pointing gesture still
present. This supports checking held poses and window position in model diagnostics,
without changing the supplied transcript or claiming expert variant verification.

Research leads, reviewed 2026-09-09:
- CTC sums over alignments and distinguishes adjacent duplicate labels separated
  by blank: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- CTC teacher/student spike alignment can limit framewise distillation (speech
  evidence, not a proven diagnosis for this ASL model): https://arxiv.org/abs/1904.08311
- Sequence-level distillation addresses a different preservation objective:
  https://www.isca-archive.org/interspeech_2018/huang18d_interspeech.html
- Sign-recognition visual/alignment inconsistency is an established diagnostic
  concern; VAC changes visual training and is not a drop-in fix for frozen features:
  https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html

Next diagnostics should separate visual evidence loss, blank/known/OTHER competition,
window/hold sensitivity, teacher agreement and missing replay coverage. No more data
collection is justified by this probe alone. No training, code/model/runtime edits,
protected tests, decoding-threshold selection or automatic label admission occurred.
