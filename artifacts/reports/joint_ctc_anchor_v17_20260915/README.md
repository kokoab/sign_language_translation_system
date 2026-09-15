# Verified-anchor CTC repair: completed training result

Sparse verified event anchors improved training beyond positive-only CTC, but did
not adequately fit complete sequences. All12fresh seed17111 epochs completed with
full data coverage and an unchanged architecture, optimizer recipe and decoder.

| Final full-training metric | Positive CTC only | Positive CTC + anchors |
| --- | ---: | ---: |
| Known continuous matches | 183 / 1,291 | 330 / 1,291 |
| Known continuous deletions | 974 | 817 |
| Known continuous insertions | 26 | 65 |
| Known continuous WER | 87.84% | 79.47% |
| Isolated single-clip CTC correct | 1,780 / 1,901 | 1,858 / 1,901 (97.74%) |

`anchor_audit.json` records7,379anchors for7,380events, including all1,291known
signs. Overlaps and unsampled intervals never receive invented supervision. Known
and OTHER anchor weights use fixed per-source epoch populations; partition-invariance
and label/gradient tests pass. An initial incorrect batch-weighted attempt was stopped
before its first checkpoint and retained under `interrupted_batch_weighting/`.

`training_anchor_diagnosis.json` measures64training sequences at epoch10: blank wins
87/106known anchors, while62/106would classify correctly if blank were excluded.
This is a diagnosis, not a decoder change. `loss_scale_diagnosis.json` reproduces a
remaining local minimum: on a stationary32frame example, target-normalized CTC plus
one positive CE converges to p(target)=0.0627 and empty greedy output. Per-frame CTC
normalization reaches0.9954 with the same optimizer and initialization.

The continuation in `../joint_ctc_balanced_v17_20260915/` retains this final model and
optimizer, changing only CTC likelihood normalization. No held-out outputs from this
run were used to select that correction. No Citizen test access, data acquisition,
runtime promotion or deployment occurred. `training_summary.json` preserves all epochs.
