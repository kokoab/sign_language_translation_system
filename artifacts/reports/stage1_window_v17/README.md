# Stage-1 continuous-window experiment — seed 17111

**Outcome: not eligible; existing defaults retained.** The only epoch within both
isolated-retention limits, epoch 1, fails connected WER, insertions, familiar WER,
and transition emissions. Epochs 2–12 fail isolated retention. No confirmation seed,
new Core ML export, protected-test evaluation, or deployment was performed.

## Frozen comparison

The [approved plan](PLAN.md), [development freeze](development_freeze.json),
[evaluation manifest](evaluation_manifest.json), and [checkpoint provenance](checkpoint_provenance.json)
pin the source identities, hashes, recipes, and decoding. There are 334 development
recordings: 225 connected ASLLRP, 12 exact-phrase ASLLRP, and 97 familiar local phrases.
The freeze hashes 4,514 raw/replay inputs. Its `candidate_training_launched: false`
records the immutable prelaunch state; training subsequently completed.

Current revisable CTC uses `stage2_v17_transition_repair_v3/seed_1702.pth`.
Its existing matched baseline and raw-parity evidence are pinned in the freeze;
older Reel's 97 familiar recordings were replayed afresh with the same source hashes.
Known-sign WER removes explicitly marked OOV reference events; insertions during
those regions still count. This is not full-vocabulary ASL accuracy.

| Development comparison | Reference words | WER | Substitutions | Deletions | Insertions |
|---|---:|---:|---:|---:|---:|
| Connected, current revisable CTC | 284 | 168.31% | 144 | 11 | 323 |
| Connected, Stage-1 window epoch 1 | 284 | 316.20% | 125 | 3 | 770 |
| Familiar, older Reel | 259 | 56.37% | 18 | 127 | 1 |
| Familiar, current revisable CTC | 259 | 7.34% | 4 | 1 | 14 |
| Familiar, Stage-1 window epoch 1 | 259 | 129.34% | 48 | 68 | 219 |
| Exact ASLLRP subset, current CTC | 24 | 45.83% | 3 | 8 | 0 |
| Exact ASLLRP subset, Stage-1 window epoch 1 | 24 | 54.17% | 3 | 9 | 1 |

WER may exceed 100% because insertions are unbounded. Eligibility requires connected
WER ≤151.48%, insertions ≤323, deletions ≤11, familiar WER ≤56.37%, and transition
false emissions ≤15/16. Own-start isolated validation is Citizen 95.2381% and SemLex
85.2761%; the corresponding floors are 94.2381% and 84.2761%. Epoch 1 reaches
94.4444% and 85.1738%. Later epochs cross at least one retention floor.

The transition comparison uses the same 16 timestamped, annotation-checked 0.53-second
background supports with fresh classifier/CTC context. It measures whether a window
emits any known sign. CTC emits on 15/16; epoch 1 emits on 16/16. This small gap set
does not establish independent idle-motion or OOV rejection performance.

## All-epoch selection results

All 12 epochs were evaluated on every frozen recording (7,212 scheduled windows per epoch, including Finish tails). Selection is fail closed: **no eligible checkpoint**. Epoch 4 has the lowest connected WER, but is a diagnostic result, not a selected model. Epoch 1 was used for actual runtime checks because it alone met both isolated-retention floors.

| Epoch | Connected WER | I | D | Familiar WER | Citizen | SemLex | Gap emissions /16 | Measured failed gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 316.20% | 770 | 3 | 129.34% | 94.44% | 85.17% | 16 | connected WER, insertions, familiar WER, transitions |
| 2 | 185.92% | 392 | 16 | 116.99% | 94.18% | 84.25% | 9 | connected WER, insertions, deletions, familiar WER, Citizen, SemLex |
| 3 | 147.18% | 259 | 34 | 112.36% | 93.12% | 83.74% | 7 | deletions, familiar WER, Citizen, SemLex |
| 4 | 146.13% | 258 | 33 | 117.37% | 93.39% | 83.64% | 5 | deletions, familiar WER, Citizen, SemLex |
| 5 | 153.52% | 289 | 27 | 119.31% | 93.65% | 84.46% | 4 | connected WER, deletions, familiar WER, Citizen |
| 6 | 165.14% | 320 | 29 | 117.37% | 94.18% | 84.05% | 4 | connected WER, deletions, familiar WER, Citizen, SemLex |
| 7 | 155.28% | 288 | 34 | 115.44% | 93.39% | 84.05% | 5 | connected WER, deletions, familiar WER, Citizen, SemLex |
| 8 | 162.68% | 312 | 32 | 118.15% | 93.65% | 84.05% | 5 | connected WER, deletions, familiar WER, Citizen, SemLex |
| 9 | 163.38% | 323 | 28 | 114.29% | 92.86% | 83.95% | 5 | connected WER, deletions, familiar WER, Citizen, SemLex |
| 10 | 166.90% | 333 | 26 | 116.60% | 93.12% | 83.84% | 6 | connected WER, insertions, deletions, familiar WER, Citizen, SemLex |
| 11 | 168.31% | 337 | 23 | 114.67% | 93.12% | 84.15% | 6 | connected WER, insertions, deletions, familiar WER, Citizen, SemLex |
| 12 | 168.31% | 338 | 24 | 114.67% | 93.12% | 84.15% | 6 | connected WER, insertions, deletions, familiar WER, Citizen, SemLex |

Full-pool runtime latency remains unverified for every epoch and is an additional eligibility failure. The 9-clip runtime timing subset is reported separately, never substituted for that gate.

Epochs 1–4 used CPU evaluation; 5–12 used MPS after full epoch-1 CPU/MPS label/transcript/transition parity. The device adapter in `evaluate_mps.py` leaves the frozen evaluator, model, data, schedule and decoder unchanged. Batched GPU/CPU timings are not used as runtime latency.

Machine-readable details: [selection.json](selection.json), [all evaluations](evaluations/), [CPU/MPS parity output](mps_parity_epoch01.json).

## Matched HUNGRY diagnosis

[Diagnostic evidence](hungry/diagnostics.json), [summary](hungry/summary.json), and
[timestamped examples](hungry/timestamped_examples.csv) compare unchanged models and
frames at origins 0, ¼, ½, and ¾ of the 32/30-second window period. The recovered
640×360 recording has reduced resolution. Original source timestamps are retained,
and the damaged final video frame is excluded. The supplied sidecar has 1,393 entries;
the retained decoded stream has 1,392 observations. This recording has no verified
whole-session transcript and is diagnostic only.

Across 83 windows, OTHER suppression changed **zero** decoded outputs. Proposal
HUNGRY top-1 counts by phase were 3/3/2/2; image-verifier counts were 3/3/4/1.
At phase ½, 105.08–106.05 seconds, proposal/verifier HUNGRY scores were 0.763/0.788;
at phase ¾, 105.35–106.32 seconds, they were 0.809/0.847 while CTC produced EAT.
These model scores are not calibrated correctness probabilities. Window placement
matters, but no reproducible extraction/timestamp/suppression repair was established
that met the gates and could replace the planned training experiment.

Epoch 1's full [paced HUNGRY replay](hungry_candidate_diagnostic.json) produced HUNGRY
among many spurious/repeated outputs. All 453 windows with no detected hand frames
still predicted a known sign. Detection absence is not verified idle ground truth.
The 112-second source took 153.92 seconds of capture processing, with a further
46.25-second Finish pass under concurrent CPU evaluation. This is not real-time
readiness evidence. No HUNGRY WER or recognition accuracy is claimed.

## Candidate and supervision

The phrase-adapted Stage-1 Squeezeformer and existing emission wrapper were reused.
The encoder, 100-class classifier, and appended NO_EMIT head were trainable; checkpoint
tensor comparisons confirm encoder and classifier changes. The expensive hand-image
verifier remains a diagnostic comparator, outside the candidate loop.

The [preparation audit](prepared_data_audit.json) contains 5,814 contextual positives
and 494 checked background windows for training; validation has 1,331/16. Training
contains 53 distinct signs, 41 distinct sign pairs, and three signers; validation
contains 36 signs, eight distinct pairs, and one disjoint signer. Annotation coverage
is approximately 54.7%/56.6%. OOV intervals are explicitly retained in the manifest,
not relabeled as background. Equal-duration phrase partitions are not supervision.

Durations are 0.27/0.53/1.07 seconds, plus the exact trailing 0.53/0.13-second runtime
schedule. Windows retain real neighboring motion and resample to 32 frames using
timestamps, binary presence masks, and window-local normalization. Foreground-only
loss uses checked foreground masks with observed hand evidence. Background requires
checked intervals and a 0.10-second guard; unchecked regions are excluded.

Seed 17111 completed 12 epochs, 3,000 draws each, batch 64, AdamW 2e-5, weight decay
1e-4, cosine decay, clipping 1.0. Every epoch draws exactly 1,500 isolated replay,
900 contextual positives, and 600 verified backgrounds, balanced within sources and
classes. Loss weights are 1.0 window, 0.5 foreground, 1.0 frozen-teacher consistency;
consistency applies only to known-sign replay. All 12 checkpoints are preserved.
Training took 111.22 seconds on MPS; no additional training recipe was tried.

## Runtime verification and limitations

The opt-in backend evaluates actual timestamped trailing windows every 0.13 seconds,
requires two agreeing predictions, and merges a same-label run. An intervening
NO_EMIT or different prediction permits the same sign again. Retained predictions
can replace a recent two-second tail; Finish reclassifies all retained raw observations
with the same schedule and a final partial window before grammar.

The baseline's fixed historical predictions produced **zero online word replacements**
in the measured HUNGRY replay. The replacement API is tested, but this experiment does
not demonstrate useful spontaneous correction from later motion. Identical signs
without a predicted separator merge; a spurious separator can duplicate a hold.
These are measured/structural limitations, not solved recognition cases.

[Actual runtime results](runtime_epoch01/results.json) cover 16 recordings: all 12
exact-development clips, three connected clips chosen by sorted identity, and HUNGRY.
All processes exited successfully. Final transcripts match cached evaluation on
**15/15** comparable development recordings. HUNGRY has no transcript comparison.

The [runtime analysis](runtime_summary.json) uses only nine recordings with verified
complete boundary/reference alignment: **15 signs, seven recognized, eight missed
(53.33%)**. Among recognized signs, startup-inclusive median first-correct delay is
**0.304 seconds**, p95 **0.681 seconds**; Finish-only recognition is timed at completion.
Per-window normalization plus inference has median 58.14 ms and p95 106.47 ms across
all replay windows. Runs used fresh processes and concurrent CPU evaluation; these
are neither warmed-camera nor iPhone measurements. Latency eligibility remains
unverified for the complete frozen annotated pool. Other measured gates already fail.

[Coverage](coverage.json) identifies 173 signs shorter than 0.27 seconds and 14 rapid
adjacent pairs, but **no identical known-sign repetitions and no holds longer than
1.07 seconds** in continuous validation. There is one held-out continuous signer.
Independent portrait-iPhone, repetition, long-hold, idle-motion, and broader signer
coverage are missing. OOV contexts and hand-detection loss are diagnostic coverage,
not independently scored recognition/rejection benchmarks.

Focused tests cover timestamp validation/gaps, absent hands, no future normalization
leakage, exact sampling proportions, foreground/replay loss masks, seed/freeze gates,
two-agreement merging, repetitions with/without separators, long holds, Reset state,
recent-tail replacement, Finish gaps/partial tails, unchanged CLI defaults, and
fail-closed selection. Actual replay exercises full capture, extraction, classifier,
transcript, Finish and literal grammar. PyTorch/Core ML agreement is intentionally
not run: no candidate qualifies for the conditional Core ML export.

## Reproduce the experimental Mac path

From the repository root, using the required Apple Vision virtual environment:

```sh
venv/bin/python scripts/live_reel_continuous_v17.py \
  --transcript-backend stage1-window \
  --stage1-window-checkpoint artifacts/models/stage1_window_v17_seed17111/epoch_01.pth \
  --no-speech --naturalizer literal
```

This launches a rejected research checkpoint for inspection, not a recommended model.
Existing default commands and accepted checkpoint files are unchanged. The exact
paced replay commands and source hashes are saved in `runtime_epoch01/results.json`.
`analyze_runtime.py` reproduces the runtime summary. [Timestamped candidate examples](timestamped_candidate_examples.csv)
and every epoch's complete window outputs support error inspection without reopening
protected tests. A fresh independently annotated evaluation is required before broader
reliability claims; this experiment does not authorize another speculative training run.

## Final verification

[Verification record](verification.json): **61 focused tests passed**, affected Python
files compile, and `git diff --check` passes. Every frozen input/artifact hash was
rechecked after evaluation; all 12 checkpoint hashes and training-freeze references
match. Independent selector review reproduced and closed a runtime-pool binding bug;
latency evidence now requires the frozen manifest hash and exact annotated identities.
There are no task training/evaluation/replay processes left running.
