# Frozen versus joint CTC — completed negative result

Completed 2026-09-14. **Neither arm passes promotion.** All 12 epochs of each arm
completed and all 24 checkpoints were evaluated on the same 334 development recordings
(8,016 recording evaluations). No checkpoint, runtime default, dataset split or Citizen
test policy changed. No confirmation seed, export or iPhone experiment ran.

## Measured result

Final epoch 12, with the same examples, initialization, seed and 1,320 optimizer updates
per arm:

| Metric | Frozen encoder | Joint encoder |
|---|---:|---:|
| Connected ASLLRP WER | 100.00% | 100.00% |
| Connected deletions | 284 / 284 | 284 / 284 |
| Familiar phrase WER | 100.00% | 100.00% |
| Familiar deletions | 259 / 259 | 259 / 259 |
| ASLLRP contiguous deletions | 24 / 24 | 24 / 24 |
| O5S5 raw-core training top-1 | 74 / 199 = 37.19% | 186 / 199 = 93.47% |
| LG raw-core top-1 | 12 / 57 = 21.05% | 12 / 57 = 21.05% |
| Citizen development isolated-head top-1 | 95.24% | 95.50% |
| SemLex development isolated-head top-1 | 85.28% | 85.99% |
| Verified-gap known-sign emissions | 0 / 16 | 0 / 16 |

**Every epoch in both arms emits no known signs on all 334 development recordings.**
The 100% WER is entirely deletion, not an improvement worth promoting over the accepted
168.31% connected WER with 11 deletions and 323 insertions. The familiar baseline is
56.37% WER. Both candidates fail deletion and familiar-recognition gates. Silence on
verified gaps is not useful rejection behavior when the model is also silent on signs.

LG results above have **one result per original annotated event**. They are not the
previous 318-window, contextual-core numbers. For the original model, this experiment's
matched contextual-core probe scores 52/199 train and 16/57 LG; physically cropped-core
inputs score 74/199 and 12/57. Four train events contain one observed pose, repeated as
static input with no invented trajectory. All 199/57 events are retained.

## What failed

The experiment exposes two separate problems:

1. **Continuous CTC fitting failed even on training data.** At epoch12 the frozen arm
   emits zero known signs across 923 ASLLRP training sequences. The joint arm emits just
   two correct known signs out of 1,291 references. Its 236,988 training token decisions
   include 236,974 blanks, 10 OTHER decisions and only four known-gloss decisions that
   collapse to those two signs. Both final models choose blank on every token of the
   237 ASLLRP validation sequences. This is an observed blank-dominated decoding failure,
   not evidence that lack of unseen signers alone explains the continuous result.
2. **Positive-core fitting did not transfer to LG.** Joint training increases raw-core
   train accuracy by 56.28 percentage points while LG top-1 remains unchanged. LG macro
   accuracy changes from 17.47% to 21.95%; train macro accuracy reaches 91.95%. The
   scalar top-1 tie does not mean that individual LG predictions are identical.

The isolated classification head retains its ability, but this must not be confused
with CTC recognition of isolated clips. As a separate diagnostic, single-clip CTC exact
accuracy is 4/378 Citizen and 18/978 SemLex for the frozen arm, and **0/378 and 0/978**
for the joint arm. Retaining pooled classification alone did not teach the CTC head
when to emit.

The preflight did fit four short, distinct train-only transcripts exactly after70
updates (loss0.04293), using its declared disposable higher learning rate. That proved
the implementation could fit a tiny subset; it did not prove the fixed full-data
recipe would escape blank collapse. Full-data losses decrease, but decoded recognition
does not follow. No learning rate, blank weight, decoder threshold or data policy was
changed after inspecting the development results.

## What this establishes and what comes next

Reusing the temporal head with a differentiable encoder was technically viable and
improved positive-core training fit. **This bounded recipe did not produce a usable
continuous recognizer.** It does not establish that all joint CTC architectures fail.

The immediate model task is to resolve full-data CTC blank collapse on training
sequences before another generalization comparison: inspect alignment/blank pressure,
token-to-target density and whether isolated/core supervision must directly constrain
the CTC output. These are hypotheses for a separately bounded training-only diagnostic,
not measured fixes. Do not extend this run, tune blank suppression on LG, or repeat
the recipe merely because its aggregate WER looks lower.

Additional signer/variant coverage and independent portrait-iPhone development/test
groups remain necessary for generalization. They cannot, by themselves, explain away
failure to emit known signs on this experiment's existing training sequences. No new
download is required to establish that first training gate.

## Coverage, fairness and verification

- Each epoch visits all 44 contiguous and 879 OTHER-aware ASLLRP train sequences,
  all199 O5S5 cores,1,901 isolated replay clips and494 verified background windows.
  Per-epoch schedules and coverage hashes match between arms; no replacement repeats.
- Complete ASLLRP transcripts include OTHER. O5S5 has no full-narrative CTC or fabricated
  blank targets. Isolated CE/teacher KL and verified-gap blank CE follow the fixed plan.
- Frame tokens retain timestamp ownership with32-frame chunks; measured max chunk
  span0.5005s, max sequence528tokens. Existing causal dilated CTC blocks are reused.
  Both encoders run in eval mode, with gradients enabled only in the joint arm.
- All4,520 frozen input hashes, all24 checkpoint hashes, initial encoder/head identity,
  unchanged frozen encoder and changed joint encoder were reverified. All per-recording
  edit counts and gates were recomputed from predictions; all24 focused tests pass.
- Initialized CPU/MPS token predictions agree256/256. Final models each agree628/628
  tokens across eight checked recordings, maximum logit difference4.77e-6.
- Successful arm training took724.62 seconds total, excluding preparation, evaluation,
  preflight, verification and an interrupted run. An initial frozen attempt stopped
  after epoch8 with an Apple Metal internal command-buffer error. Its artifacts remain
  in `frozen_interrupted`; the complete same-recipe retry started from the original
  initialization, since those checkpoints did not contain optimizer state.
- Cached CPU forward for eight recordings took about0.52/0.53s (frozen/joint), excluding
  extraction and scheduling. All296 temporally scored signs were missed. No meaningful
  emission latency or iPhone readiness can be claimed. Stored source-clock timestamps
  use the last observation of each chunk, not measured timer/extraction completion.
  The runtime-inclusive latency gate remains explicitly unverified and failed closed.

## Reproducibility

- [Frozen plan](PLAN.md), [recipe/dependency hashes](training_freeze.json)
- [Preparation audit](data_audit.json), [preflight](preflight.json)
- [All epoch comparisons](epoch_comparison.csv)
- [Verification and final train/validation diagnostics](verification.json)
- Training/evaluation: [run_experiment.py](run_experiment.py)
- Independent reductions/provenance checks: [verify_results.py](verify_results.py)
- Checkpoints: `artifacts/models/joint_ctc_v17_20260914/{frozen,joint}/epoch_01..12.pth`

Verification can be rerun without training:

```sh
venv/bin/python artifacts/reports/joint_ctc_v17_20260914/verify_results.py
venv/bin/python -m unittest test.test_joint_ctc_v17 test.test_unified_streaming_ctc_v17 test.test_stage1_window_training_v17 test.test_stage1_window_evaluation_v17 test.test_prepare_o5s5_citizen100_v17 -q
```

Training commands refuse existing output directories. Preserve completed results;
do not reuse their paths for a different experiment.
