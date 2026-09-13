# SLT working guide

## Token-efficient workflow

Read this file and `PROJECT_GROUND_TRUTH.md`, then inspect only task-relevant files and
the current git diff. Do not rescan the repository or reread large historical documents
unless the task specifically needs them.

`PROJECT_GROUND_TRUTH.md` is current state only (~11 KB). Project history lives in
`docs/ground_truth/`, split by topic — 354 dated entries. Never read that archive
start-to-finish; `rg <term> docs/ground_truth/` when you need history, and check
`<topic>/log.md` before re-running an experiment to see whether it already failed.
`docs/ground_truth/MAP.md` is the file map.

Keep simple work in the main thread. Use a subagent only for an independent, materially
useful task; give it exact paths, a stopping condition, minimal steps, and a concise
evidence-only return format.

Prefer `rg`/`rg --files` and targeted reads. Do not paste large files or raw command
output into handoffs.

**Never read a result file under `artifacts/` whole.** The largest are several MB — one
read is ~1M tokens and ends the session. `artifacts/LARGE_FILES.md` indexes every file
over 1 MB with its top-level keys and metrics; consult it, then extract only what you
need (`jq '.field' <file>`, `jq -r 'keys[]' <file>`, `head -5 <file.csv>`). Regenerate
the index with `venv/bin/python scripts/index_large_artifacts_v17.py` after producing new
reports. The same applies to reading a report directory: `ls` it first, never `cat -r`.

This is enforced, not just advised: `tools/slt-guard/` is a local plugin whose PreToolUse
hook refuses any Bash command that reads a >1 MB file end-to-end. One script serves both
harnesses — Claude Code loads it via `.claude/settings.json`, Codex via the
local marketplace at `tools/slt-guard/`. Run its `--selftest` after editing it.

Record changed files, decisions, measured results, failures, and the next safe action after every material decision, code/data action, experiment, or
validation result. Append the dated entry to the matching `docs/ground_truth/<topic>/log.md`
(newest first), and update `PROJECT_GROUND_TRUTH.md` only for what changes current state.
Promote an entry to `<topic>/high.md` when it establishes something that binds future
work. `PROJECT_GROUND_TRUTH.md` is canonical: on any conflict, it wins.

Preserve unrelated user changes. This worktree contains a large pre-existing,
uncommitted reorganization; never revert or clean it as part of v17 work.

Run the smallest affected tests first. Expand only after a focused failure, a
cross-cutting change, or before handoff. Run `git diff --check` before a code handoff.

## Repository map and focused validation

| Area | Canonical path | Focused validation |
| --- | --- | --- |
| v17 extraction/schema | `active/v17/` | `venv/bin/python -m unittest test.test_v17_extractor -v` |
| v17 compatibility | `src_v17/` | invoke `src_v17/extract_v17.py --help` |
| v17 extracted archives | `data/local/**/landmarks_v17/` | `venv/bin/python active/v17/audit_v17.py <root>` |
| v16 trained pipeline | `active/v16/` | task-specific script/test only |
| v16 compatibility | `src_v16/` | keep old imports and mobile export working |
| dataset utilities | `scripts/` | run the affected script against the smallest sample |
| generated reports | `artifacts/reports/` | inspect the generated Markdown/CSV/JSON |
| project history | `docs/ground_truth/` | `rg` only; never read start-to-finish |
| oversized artifacts | `artifacts/LARGE_FILES.md` | index of >1 MB files; `jq` them, never read whole |
| local datasets | `data/local/` | never commit or delete without explicit request |
| legacy code | `src/`, `legacy/` | do not touch unless explicitly in scope |

The required Apple Vision environment is `venv/bin/python`; system Python lacks the
Vision/Quartz bridge.

## Current scope and hard gates

The current work is v17 extraction plus a Citizen-only 100-sign baseline for a future
iOS-first model. v17 extractor outputs are deliberately incompatible with the trained
v16 checkpoint. Do not silently connect them or report v17 classifier accuracy until a
new model is trained from scratch.

Use ASL Citizen's official signer-disjoint train/validation/test splits. Never
random-split videos, mix signer identities across splits, or use aspect-ratio
distortion. Each class must pin one exact raw gloss and ASL-LEX code; the frozen floor
is 10/3/5 train/validation/test signers. Do not merge numeric variants by normalized
label.

The official Citizen test gate has been consumed once for the frozen Apple Vision +
v17 Squeezeformer selection: 87.57% top-1 on 1,247 clips. Do not rerun it during model
development, select checkpoints from its errors, or describe the 93.12% validation
score as test accuracy. Use a new independent portrait-iPhone set for future selection.

PopSign specifically captures one-handed smartphone signing. It is not the primary v17
dataset. A PopSign-only model is a one-handed isolated-sign recognizer; never describe
it as general two-handed ASL coverage or a conversational translator. Do not resume the
paused PopSign audit download unless that one-handed portrait audit is useful.

Accuracy and generalization are the first gates. Distillation, quantization, ONNX pose
replacement, and aggressive model shrinking remain deferred until a strong uncompressed
baseline exists. Mobile readiness requires measured Core ML size, memory, sustained
latency, thermals, and accuracy on real iPhones; scaffolding or desktop timing alone is
not sufficient evidence.

Do not claim continuous sign recognition or translation from PopSign isolated-sign
training. Stage 2/translation work begins only after Stage 1 signer-disjoint accuracy is
credible.

## Existing v16 reference

**Current default:** v16 Squeezeformer pipeline: Apple Vision-style landmarks -> Stage 1 isolated classifier -> Stage 2 CTC gloss decoder -> optional Flan-T5 translation.

### Quick reference

| Area | Current file | Purpose |
| --- | --- | --- |
| Stage 0 | `active/v16/extract_v16.py` | v16 landmark extraction/features |
| Stage 1 | `active/v16/train_stage_1_v16.py` | Isolated sign classification |
| Stage 2 | `active/v16/train_stage_2_v16_fixed.py` | Continuous recognition with CTC |
| Deploy | `active/v16/inference_v16.py` | Video inference, trimming, sliding windows, TTA, beam search |
| Model | `active/v16/model_v16.py` | Squeezeformer Stage 1/2 definitions |
| Compatibility | `src_v16/*.py` | Wrappers for existing scripts and mobile export |
| Legacy | `src/` | Older DS-GCN/Transformer pipeline |

### Context files

- **`CURRENT_PIPELINE.md`** - current v16 commands and repo organization.
- **`docs/md_files/DOCUMENTATION_FILE_MAP.md`** - map of v16 docs, legacy docs, metrics, and reports.
- **`docs/md_files/SESSION_BRIEFING.md`** - recent v16 inference status and evaluation notes.
- **`docs/md_files/context.md`** - useful architecture summary, but verify against v16 docs when naming matters.

### Critical rules

- Prefer `active/v16/` for new v16 code changes.
- Keep `src_v16/` wrappers working for old imports and scripts.
- Keep root compatibility symlinks working for datasets, `models`, and `weights`.
- Put new local datasets under `data/local/` and new generated outputs under `artifacts/`.
- Do not delete datasets, checkpoints, reports, or metrics during cleanup unless explicitly requested.
- CTC blank index is `0`.
- Mirror/TTA must swap hand indices `0-20 <-> 21-41` with X-axis sign flips for coordinate/motion channels.

### Tests

- `test/test_batched_forward.py` - legacy Stage 2 forward smoke.
- `test/SLT_test.py` - legacy Stage 2 batch WER.
- `test/test_offline_pipeline.py` - legacy Stage 2 -> Stage 3 E2E.
- v16 scripts under `scripts/` provide evaluation smoke tests when local datasets/checkpoints are available.
