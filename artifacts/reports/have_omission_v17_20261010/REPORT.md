# Missing HAVE in I HAVE BIG — 2026-10-10 PHT

## Decision

No candidate is promoted. Sixteen decoder configurations, including the baseline, were
examined on take 11. No tested configuration both resolves the omissions and preserves
the other signs. No production Python/Swift source, checkpoint, phone installation or
training manifest was changed. No protected test was evaluated.

## Matched inputs and scope

The experiment copies the current Swift phone sources into `swift/`; original hashes
are in `source_hashes.json`. The recognizer is
`SpanRecognizerV17FragmentRejectLettersB8FP16`, with the existing one-way HAVE/ANGRY
check and TIRED motion rescue. Encoder is explicitly FP16; recognizer and encoder
compute units are `.all`, matching `LiveReelApp.swift`.

The old harness uses frame-index/fps except when supplied live timestamps. The
experimental harness uses actual video PTS for ordinary video replay. Take 11 is
captured once; all variants receive identical observations and hand embeddings.
Native replay and the traced baseline produce exactly the same word sequence.
This is a recording replay, not the original live-camera frame schedule for take 11.
The 10:06 control uses its saved exact live trace.

The new take-11 baseline has 5 HAVE and 1 ANGRY. Older narratives and harness outputs
have different counts and are not substituted as the experimental baseline. The
encoder, timestamps and model inputs are pinned here; no claim that PTS alone explains
all differences between older outputs.

## What the trace establishes

Around 4.1 seconds, HAVE is selected for four updates (probability approximately
.416–.462). As the interval extends into BIG, the decoder replaces it with one longer
BIG segment before the HAVE can commit. Disabling early BIG does not prevent that
merge. Around 7 seconds, the selected contact candidate peaks below the .4 word gate.
The contact sheet shows chest contact around 7.1 seconds, release around 7.3, and BIG
around 7.5. This is both a span-selection and recognition-evidence problem.

An exhaustive diagnostic classified short windows around six contact regions, even
when the boundary model did not propose those windows. Best HAVE probabilities were
.527, .667, .424, .736, .936 and .675. Several useful windows begin earlier than the
boundary-selected onset. These maxima use known contact regions and are **oracle
diagnostics**, not an automatic decoder or a measured accuracy improvement. In
particular, the .424 case remains weak evidence.

## Decoder experiments

All rows use the same take-11 trace. The decision gate remains .4 throughout.

| Experiment | Finding |
|---|---|
| Disable early BIG | HAVE 5→6; recovers the ~11.57s word, other misses remain |
| Disable all early emission | Same sequence as disabling early BIG |
| Remove future context | Misses remain; adds WHEN |
| Remove both context sides | Misses remain; adds WHEN |
| Reduce future context to .05s | Misses remain; first ANGRY becomes silent |
| Allow HAVE early at .4 | Adds a HAVE but loses a later BIG |
| Halve boundary-start weight | Adds HAVE but also ANGRY/WHEN |
| Boost already-top HAVE only when both existing cues support a higher score | No gain over disabling early BIG |
| HAVE segment cost −2 or −2.5 | HAVE rises to 8 but misses remain; −2 adds false HAVE on validation |
| HAVE boost plus cost −2 | Same take-11 sequence as cost −2 alone |
| All-sign segment cost −2 | Adds another ANGRY; misses remain |
| Extend preceding context to .2/.3/.4s | Adds ANGRY/HAPPY/WHEN; omissions remain |

Exact 10:06 live trace:

- Baseline: `HAVE HAVE HAVE I BIG HAVE`
- No early BIG: `HAVE HAVE HAVE I ANGRY BIG HAVE`

Thus even the one-word improvement from disabling early BIG is not safe to promote.

## Development regression checks

Historical Python cached inputs, matched between the three variants below. These are
separate from the Swift/PTS replay and do not establish phone accuracy. Tuning is the
89-phrase development pool; no held-out accuracy claim is made.

| Measure | Baseline | No early BIG | No early BIG + HAVE cost −2 |
|---|---:|---:|---:|
| 89 phrases / 226 words: WER | 15.04% | 15.04% | 15.04% |
| Substitutions / deletions / insertions | 6 / 17 / 11 | 6 / 17 / 11 | 6 / 17 / 11 |
| Citizen validation: correct target present / 378 | 332 | 332 | 332 |
| Citizen validation: exact single-word output | 301 | 301 | 300 |
| Citizen validation: extra words | 39 | 38 | 39 |
| Citizen validation: false HAVE clips | 1 | 1 | 2 |
| Pair validation: correct target present / 103 | 82 | 82 | 82 |
| Pair validation: exact single-word output | 58 | 58 | 57 |
| Pair validation: false HAVE clips | 0 | 0 | 1 |

The new false HAVE in pair validation is `local__ANGRY__ANGRY_ANGRY_20`:
`ANGRY` becomes `ANGRY HAVE`. The unchanged aggregate phrase WER conceals this failure.

This evaluator uses a one-second held-last-frame tail for isolated clips. The historical
helper increments each new padded timestamp from the previous padded timestamp by
.05×(k+1), producing growing gaps and decoder resets. Its older isolated exact/extra
counts therefore are not directly comparable to this corrected-tail experiment.
The held tail is a stress condition, not newly annotated signing.

## Files and reproduction

- `take11_pts.json`: native replay words; `take11_pts_trace.json`: captured inputs.
- `ablation.json`, `round2.json`, `round3.json`, `round4.json`: hypotheses, candidate
  scores and per-update selected segments. Extract fields with jq; do not print whole.
- `variants_*.json`: recorded experimental settings.
- `oracle.json`: short-window classification diagnostic.
- `exact_live_controls.json`: exact 10:06 live-frame ablation.
- `regression_pair_val_tune_citizen_val.json`: 570 development clips, three variants.
- `evaluate_cached.py`, `runtime_experiment.py`: isolated Python regression harness.
- `swift/`: isolated Swift harness and source snapshot; production files unchanged.
- `verify.py`: checks source hashes, native/trace parity and measured result invariants.

Verification passed. Task-scoped tracked-file whitespace check passes; repository-wide
`git diff --check` flags unrelated whitespace at `scripts/app_shell_v17.py:632`, preserved.

Build the final experimental harness from repository root:

```sh
swiftc -O -swift-version 5 -target arm64-apple-macos15.0 artifacts/reports/have_omission_v17_20261010/swift/*.swift -o /tmp/have_omission_harness
venv/bin/python artifacts/reports/have_omission_v17_20261010/verify.py
```

`tracewords` takes resources directory, trace path and a new output path. Set
`LIVE_REEL_ENCODER=MobileCLIP2S0ImageEncoderV17FP16` and `VARIANTS` to a recorded variant
file; `PROBE=1` runs the oracle diagnostic instead. Preserve the saved results on rerun.

## Next safe experiment

Prepare manually reviewed contact/transition intervals from the existing recordings,
including genuine BIG, ANGRY and TIRED negatives, and reserve a complete recording
session from fitting/selection. Use these to test a boundary/input-contract change that
separates the chest-contact sign from the following sign. A classifier-only retrain or
more confidence tuning is not justified by this experiment. Earlier user-HAVE adaptation
already failed to provide a clean general fix; do not repeat it unchanged.

Any training must first satisfy the repository's approved-root and recipe/input-contract
requirements with a reviewed new recipe where needed. This experiment authorizes no
bypass of those gates. No new recordings or dataset acquisition are required to begin
the annotation/contract work. A clean solution and fresh live-signing validation remain
outstanding; the omission is not fixed.
