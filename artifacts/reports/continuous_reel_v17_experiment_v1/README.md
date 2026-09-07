# Continuous Reel experiment — measured result

The separate runtime is implemented and tested. It is not promoted as an accuracy improvement: the final paired run has 1/5 exact Stage-1 transcripts and 5/12 token edits in both lanes. Stage-2 suggestions are 5/5 exact on these familiar short recordings, but remain review-only.

Run from the repository root:

```bash
venv/bin/python scripts/live_reel_continuous_v17.py
```

Keep signing through amber `GLOSS?` previews. Press F or the Finish button once per utterance. Recent tentative output may change. Speech is emitted for the finished Stage-1 sentence only. The amber sequence suggestion is separate; this prototype has no accept-suggestion control and never silently substitutes it. R resets; Q exits. The exact 100-gloss vocabulary and acceptance thresholds are unchanged. The original `live_reel_stage1_v17.py` remains the comparison command. `--no-stage2-arbiter` disables the optional Finish review.

## What changed

- Preserve observations newer than the committed clip, including frames captured during verification.
- Show the tentative accepted landmark gloss centrally and in the rail before commitment.
- Store exact Stage-2 observations in a private temporary-file FIFO; perform sequence decoding after Finish and a finite Stage-1 tail drain. RAM does not grow with queued RGB windows, but temporary disk usage and Finish work grow with utterance duration.
- Process new eligible tail evidence once, without manufacturing repeat stability hits on the same frames. Short or uncertain tails can still remain uncommitted.
- Accept the next utterance's Finish while previous naturalization completes; keep their timing contexts separate. Reset excludes in-flight results from the prior epoch.

## Final five paired development replays

| Phrase | Baseline Stage 1 | Continuous Stage 1 | Finish suggestion |
|---|---|---|---|
| GOOD MORNING | GOOD STOP | GOOD MORNING | GOOD MORNING |
| HELLO HOW YOU | HELLO HOW YOU | HELLO HOW | HELLO HOW YOU |
| MY NAME | LIKE NAME | LIKE NAME | MY NAME |
| THANKYOU FRIEND | THANKYOU | THANKYOU NAME DOCTOR | THANKYOU FRIEND |
| TOMORROW SCHOOL GO | TOMORROW STOP | TOMORROW GO | TOMORROW SCHOOL GO |

Tentative display: 0.904s median from capture-loop start, range 0.539–6.311s. This is first guess, not first correct gloss. Examples: GOOD MORNING initially previews MORNING; HELLO HOW YOU initially previews FATHER.

Finish decoding (before naturalization): 1091ms median, range 794–1267ms. Retained 55 pending observations across commits. All sequence updates occurred after Finish.

These are paced, sequential saved-video replays, with no display, no speech, no gesture filtering, and a literal naturalizer. Camera drops, human repeat attempts, real webcam FPS, and independent accuracy were not measured. There was a startup/capture outlier in MY NAME; preserve it in the timing distribution. The paired run was interrupted between videos and resumed, so hardware conditions were not identical across all pairs.

The initial pre-review pass was 2/5 exact and 4/12 edits in both lanes. Async timing changes which candidate frames are classified, even in the baseline. Individual phrases can regress; do not interpret one pass as a statistically supported accuracy change. Recognition and boundaries remain the main bottleneck.

## Validation

- 41 focused runtime/HUD/cache/CTC tests pass, including delayed verification, Finish-tail drain, two successive Finishes, reset during CTC, quit without Finish, disk FIFO, and tentative rendering.
- Real Core ML runs completed in both lanes; default tiny-naturalizer smoke also completed (see below).
- Python syntax checks and `git diff --check` pass.
- Visual checks: [preview](preview.png), [finished suggestion](finished.png).
- Machine-readable final measurements: [comparison.json](final/comparison.json). The runnable `run_replays.py` reuses completed histories; it does not rerun them unless given a new output folder.
- No weights, thresholds, training data, protected splits, or default model selection changed.

Default tiny-naturalizer smoke: input `HELLO HOW`; sentence `How is you?`; full Finish time 1321ms. The sequence suggestion remains `HELLO HOW YOU`, separately logged and displayed. This smoke confirms integration, not transcript correctness.
