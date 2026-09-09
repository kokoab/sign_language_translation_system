# v17 Stage-2 retention repair — v3

**The v2 retention regression is repaired against all original frozen development
gates.** Both seeds preserve every accepted local, exact, contextual, Citizen and
STEM prediction. Seed 1702 is enabled in the continuous Reel command.

This is a conservative repair: it keeps more real target signs than v2, but leaves
more false insertions. It does **not** retain v2's 217/223 total connected errors,
and it does not establish general continuous-ASL or independent iPhone accuracy.

| Model | Connected target edits / 284 | Target S / D / I | Local / exact / contextual edits | Citizen exact / 378 | STEM exact / 21 |
| --- | ---: | --- | --- | ---: | ---: |
| Accepted baseline | 603 | 134 / 2 / 467 | 6 / 9 / 43 | 331 | 16 |
| v2 with-STEM 1701 | 217 | 39 / 160 / 18 | 15 / 16 / 57 | 333 | 16 |
| v2 with-STEM 1702 | 223 | 37 / 167 / 19 | 14 / 14 / 51 | 331 | 15 |
| Repair 1701 | 485 | 139 / 13 / 333 | **6 / 9 / 43** | **331** | **16** |
| Repair 1702 — selected | 470 | 137 / 12 / 321 | **6 / 9 / 43** | **331** | **16** |

The selected repair reduces target edits by 22.1% from the accepted baseline.
Target-only exact sequences are 10/225 versus baseline 4/225; performance remains
limited. WER is 470/284 = 1.6549, **not an accuracy percentage**. Original gates
remain target<=542, local<=6, exact<=9, contextual<=43, Citizen>=328. Both seeds
also retain STEM16/21, exceeding the supplemental minimum15.

## What was determined

- The v2 candidate already started behind the accepted composite selector. A plain
  temporal head omitted its context adapter, specialist and emission calibration.
- In a formerly correct HELLO HOW YOU example, YOU remained the highest-scoring
  known label, but blank rose to 96–98% probability. OTHER was negligible.
- In a formerly correct PLEASE HELP I example, both v2 seeds added a second I
  spike during a held final posture. Removing a final window was not safe: it
  could instead repeat HELP.
- Replay/distillation did not guarantee preservation of the decoded sequence.
  Controlled restoration of accepted emission probabilities recovered local
  retention, locating the immediate mechanism in changed emission behavior.
- An existing 1,116-example contextual training pool was omitted from v2. Its
  three training signers are distinct from contextual validation signer JONATHAN.
  These examples were reused; no new videos or manual reviews were needed.

Detailed posterior, window, video-frame and rejected-probe evidence is in
[the diagnosis report](../stage2_v17_transition_diagnosis_v1/README.md). The evidence
establishes the emission regression and a working preservation intervention; it
does not isolate every upstream optimization cause.

## Implementation and selection disclosure

The accepted two-head selector remains frozen. A separately fitted linear OTHER
head uses frozen temporal representations from each existing v2 seed. Only this
small head was retrained; Stage1 and both temporal backbones were not retrained.
Training uses original CTC labels with 10x cost for false rejection on known-only
replay, including the missing contextual training pool.

Inference retains the accepted known/blank conditional probabilities. It can veto
a complete contiguous nonblank emission run when OTHER evidence exceeds a shared
margin. It cannot split that run, invent a new known sign, or modify a blank run.
There is no phrase, label, source, signer or duration lookup at inference. OTHER
is removed after CTC collapse in the live transcript, preserving genuine repeated
known signs and their temporal positions.

The margin 1.2725113 is the maximum negative training run score across both seeds
(0.5793641) plus log(2). The run policy and safety factor were chosen **after
inspecting development failures**. This is repeated development selection, not a
pre-registered experiment or an independent test. Earlier routing/linear-head
probes that failed retention remain saved. The larger research composite adds a
CPU evidence model; it is not a compressed mobile graph.

## Verification and runtime limits

- Both self-contained checkpoints were reloaded and evaluated on all 987 unique
  frozen development examples. Every original gate passes in both seeds.
- Actual Core ML selector plus CPU rejection inference matches checkpoint decoding
  on 987/987 examples. The selector configuration, selector checkpoint, primary and
  specialist packages, frozen encoder and hand-image encoder are pinned.
- 82 focused model, extraction, training, selection, decoding and Reel lifecycle
  tests pass. Regression tests were observed failing before the relevant fixes.
  Compilation and git diff --check pass.
- Two paired raw-video smoke tests have identical old/new Stage2 suggestions and
  selected transcripts, with zero stale camera frames. Their **existing** live
  errors persist: the live suggestions are KNOW HOW YOU and PLEASE HELP I I. The
  frozen-cache corrections must not be presented as fixes to those separate live
  preprocessing/domain differences. Stage2 remains review-only in this command.
- On this Mac, cached-feature Stage2 p95 is 13.38 ms including rejection versus
  7.44ms for the accepted selector; added evidence processing p95 is 6.06 ms. This
  excludes extraction and does not establish full-Finish or iPhone performance.
  The two raw-video Finish observations are recorded individually, not as p95.

Authoritative evidence: [validation.json](validation.json),
[verification.json](verification.json), [replay_comparison.json](replay_comparison.json),
[tests.log](tests.log). No protected Citizen, SemLex or local test was accessed.

## Run or roll back

```sh
venv/bin/python scripts/live_reel_continuous_v17.py --camera 0
# Explicitly restore the original Stage2 selector behavior:
venv/bin/python scripts/live_reel_continuous_v17.py --camera 0 --no-stage2-other-preservation
```

The legacy isolated/Reel entry points keep their defaults. The selected package is
`artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth`.

## Reproduce

The executed cache and two fitting scripts are archived in the diagnosis directory,
and their train/validation tensor snapshots remain under
`artifacts/models/stage2_v17_other_preservation_v1/`. The second fitting run is
`train_other_preservation_v2.py`; its head checkpoints and full 20-epoch histories
are under `artifacts/models/stage2_v17_other_preservation_v2/`. These files preserve
the exact executed experiment; rerunning the archived scripts writes those output
locations, so copy them or change the output root before a new experiment.

```sh
venv/bin/python scripts/build_stage2_other_preservation_v17.py
venv/bin/python scripts/evaluate_stage2_other_preservation_v17.py --runtime
```

The builder recomputes the shared training margin and pins all runtime packages.
The evaluator recomputes the accepted baseline, both repaired checkpoints, original
gates, per-example predictions and optional actual runtime parity.
