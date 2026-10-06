# Reproductions of the August downstream stages (2026-10-07)

Run before rebuilding the downstream chain, to pin the exact commands. Checkpoints were not kept.

- `phrase_reel_v2_trial_a`: `train_unified_phrase_adapt_v17 --selection-key reel_v2` (current defaults).
  Baseline identical; selected epoch 27 (361/869/2812/170/718) ≠ August epoch 26. Pair loss options postdate reel_v2.
- `phrase_reel_v2_trial_b`: same plus `--pair-loss-weight 0 --pair-sampling-multiplier 1`.
  **Reproduces reel_v2 exactly**: epoch 26, 363/872/2810, phrase 172/259, activity 723/1036.
- `span_local_a`: `scripts/train_span_recognizer_v17.py --epochs 8 --seed 27927 --sets local_train
  --exclude-prefix-spans --floors {95.03,89.16,96.03} --cfg {lookahead 8, both, alpha .7, w_r 2, w_a .25, w_b .5, c -1, log_theta -1.6094}`.
  **Reproduces span_recognizer_v17_local_a exactly**: all 9 epochs' domains and tune WER identical; selected epoch 4,
  95.24/89.67/96.31, tune WER 27.43%.

Both reproductions use the August (leaking) data splits by design; the rebuilt chain uses
active/v17/phrase_segment_recipe_manifest_20261007.json instead.
