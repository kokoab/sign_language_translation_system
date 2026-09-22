# Authorized augmented boundary fine-tune

2026-09-22. One seed 17621 from original pretrained weights, not the overfit checkpoint;
five head epochs then all four attention blocks. Preserve splits, targets, float32,
Reel, 20Hz source clock, 64-frame context and 500ms lookahead. Maximum80; patience8,
no40-epoch floor; LR plateau patience3 gives reductions time before stopping.

Reuse hash-verified original cache as clean variant and unchanged held-out data.
Build two training-only raw-pose observation variants before shared normalization:
past-only sensor sampling15–20Hz and observation dropout0–15%, holding the last
available observation. This is acquisition-rate augmentation, NOT tempo warping;
labels/timestamps never move and no future observations are added. Randomly choose
one of three variants per example per epoch; do not triple epoch size or claim
new independent examples. About5.9GB additional cache; check free space first.

Tests: unchanged clean features, augmentation determinism, missing/padded frames,
future-bound, sample/label alignment, untouched held-out partitions, cache/direct
forward parity, finite MPS gradients, stop/scheduler behavior, recipe/code hashes.
Benchmark actual loss/backward/clipping/sync path plus calibration before detached
launch. Notify on completion/failure; paired whole-video evaluation mandatory,
no auto-promotion or seed2 restart. Preserve old reports, weights and cache.

Ruling: reuse current feature worktree because input artifacts/uncommitted pipeline
are already here; do not relocate/revert them. Reuse trainer/evaluator with explicit
new-run paths rather than copy training logic. New contract records changed hashes.
Ruling: sensor-rate augmentation preserves fixed deployment clock/future contract;
upstream tempo/fps transforms cannot be copied blindly into timestamped targets.
Ruling: unknown gaps remain masked; false-gap supervision cannot be fabricated.

Progress: contract design recorded; implementation and validation pending.

Review finding resolved: shared evaluation_rows now accepts an explicit report output;
pretrained evaluator passes its run path. COMBINED/CURATED and baseline reference
hashes are pinned. First unlaunched augmented contract is retained for audit but
superseded by _v2.json; no training/cache used it. Its benchmark established parity,
but benchmark/preflight will repeat under the final v2 contract.

Completed implementation/prelaunch checks: 22 focused tests pass (new tests observed
failing before implementation), git diff --check passes, independent review resolved
its evaluation-report destination/input pin finding. MPS finite-gradient preflight:
4,727,042 trainable params, zero actual training optimizer steps before launch.
Clean cache parity across all three splits maxabs5.722e-6; augmented direct/cached
predictions match. Benchmark32steps + full calibration: median307.19ms/step,
calibration3.72s, projected75.91s/epoch before extra paging/I/O/thermal overhead.
Additional two-cache disk requirement checked with2GiB reserve; float32 retained.

Final recipe active/v17/pretrained_boundary_augmented_recipe_20260922_v2.json,
SHA256 a80d7c7683ce469b1042201dbfafe77e15c53c184d88d9de01e262d80508a6e9.
Detached caffeinatePID17054 launched2026-09-22T08:24:42Z. Do not poll training.
Estimate40–75min for roughly15–25epochs including preparation/evaluation; up to
about3h if improvement continues toward80epochs. No accuracy/time guarantee.
Actual completion/epochcounts/results await notification; launch is not completion.
Next safe action: read completion/history/evaluation on user update, compare retained
signs and false outputs before any deployment or additional seed. Seed2 remains stopped.
