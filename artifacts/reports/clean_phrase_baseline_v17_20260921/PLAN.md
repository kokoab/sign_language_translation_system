# Approved phrase-only baseline — 2026-09-21

User approved implementation and two-seed training. Training runs detached with a macOS
completion/failure notification; do not poll it in-session. No acquisition or test access.

## Fixed experiment

- Data: approved phrase manifest v2, exactly 283 training / 211 validation clips.
- Sources: local phrases, ASLLRP contiguous, six established OTHER-containing sequences,
  and one verified WHEN/OTHER span. No isolated replay, NCSLGR, Flores or blank/rest clips.
- Stage 1: frozen `artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth`.
  This is the same frozen evidence encoder used by the preceding Flores comparison;
  its file hash is pinned. No Stage 1 optimizer, fine-tuning, or test evaluation.
- Input: cached Apple features, trailing8-frame windows at stride4, frozen pooled
  Stage1 embedding plus logits. Encode once; pin resulting cache hash.
- Head: existing causal unified CTC model, hidden128, three blocks, dropout0.1.
  Random independent initializations, seeds17321/17322. Train all head parameters;
  the gloss correction head does not modify the frozen Stage1 weights.
- MPS encoder/head (35% process memory fraction), two CPU threads for the unsupported
  CTC-loss operation; differentiable logits transfer preserves head backpropagation.
  18epochs, batch16; AdamW lr0.002 / weight decay0.0001;
  mean per-sequence CTC loss, blank0, OTHER101, zero_infinity=false, gradient clip5.
  Each training clip appears once per shuffled epoch. No task weighting or auxiliary loss.
- Selection: arithmetic mean of local and ASLLRP-contiguous validation known WER.
  Earliest epoch wins ties. Both seeds use identical fixed settings; no automatic promotion.
- Report per-source and overall known WER, exact sequence accuracy including OTHER,
  substitutions/deletions/insertions, blank-only output and OTHER emissions. Known WER
  removes OTHER, so it must always be read beside exact/OTHER-emission metrics.
  Save per-clip final predictions, initial-head diagnostics, all epoch metrics and hashes.

## Scope of permission

`active/v17/clean_phrase_baseline_manifest_20260921.json` authorizes only
`scripts/train_clean_phrase_baseline_v17.py`. It pins the data manifest, same approved
files, frozen base, encoded evidence, recipe and relevant code hashes. Other trainer
entry points explicitly reject this recipe-scoped manifest. General v2 training readiness
stays false: the older full recipes still require unaudited inputs and absent metrics.

This baseline does not require adding blank/rest truth or certifying unseen-OOV examples,
because neither is used or evaluated. Its limited purpose is performance on the retained
phrase development split. Validation was reused and is not a fresh test. Only seven
training clips contain OTHER; there is no OTHER validation set, so no unknown-recognition,
live-stream, full100phrase-coverage, iPhone or general ASL claim is permitted.

## Preparation and outputs

Preparation verifies exact manifest membership/hashes; checks loaded identities and
283/211 counts; freezes Stage1; caches evidence; runs a finite CTC forward check without
an optimizer step. Focused tests cover edit metrics and exclusion/training gates.

- `preflight.json`: measured checks and immutable inputs.
- `training.log`: detached stdout/stderr.
- `launch.json`: PID and launch command.
- `status.json`: running/completed/failed state.
- `results.json`, `RESULTS.md`: both results after completion.
- `notification.json`: notification delivery result.
- Models: `artifacts/models/clean_phrase_baseline_v17_20260921/seed_<seed>.pth`.

Next session: read status first, then selected summary fields from results, and review
both seeds before deciding on further work. Never infer completion from a launch receipt.
