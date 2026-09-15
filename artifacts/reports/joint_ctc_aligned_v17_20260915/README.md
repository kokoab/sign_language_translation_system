# CTC collapse repair: verified results

**Training blank collapse is repaired. Held-out continuous recognition still fails
promotion.** No new data or architecture was used. The final fixed checkpoint is
`artifacts/models/joint_ctc_aligned_v17_20260915/epoch_42.pth`.

| Measure | Failed original joint model | Repaired final model |
| --- | ---: | ---: |
| Continuous TRAIN known matches | 2 / 1,291 | 1,088 / 1,291 (84.28%) |
| Continuous TRAIN WER | almost all deletions | 32.77% |
| Isolated TRAIN CTC exact | not measured on train | 1,901 / 1,901 (100%) |
| Nonempty development recordings | 0 / 334 | 263 / 334 |
| Citizen validation single-clip CTC | 0 / 378 | 339 / 378 (89.68%) |
| SemLex validation single-clip CTC | 0 / 978 | 762 / 978 (77.91%) |
| Connected ASLLRP WER | 100% | 98.24% |
| Connected ASLLRP deletions | 284 / 284 | 90 / 284 |
| Contiguous ASLLRP WER | 100% | 50.00% |
| Familiar local-phrase WER | 100% | 107.72% |
| LG raw-core pooled accuracy | 12 / 57 (21.05%) | 15 / 57 (26.32%) |

These are training and development/validation results, never Citizen test accuracy.
The official Citizen test was not accessed. The prior all-blank model has zero
insertions because it emits nothing; its100%WER is not a meaningful recognition
baseline by itself. Compared with the earlier operational development baseline,
connected WER improves168.31%→98.24%, but deletions worsen11→90 and familiar WER
worsens56.37%→107.72%. **The repaired model is not promoted.**

## What changed and why

The pooled isolated/core classifier loss had no gradient route into the CTC head.
Direct single-token positive CTC restores that route. Sparse verified event anchors
then teach emission timing. Training-only diagnostics showed that target-normalized
CTC still has a blank local minimum beside positive CE. Temporarily normalizing its
likelihood by valid output frames lets the head learn nonblank evidence, but initially
produces many insertions. Direct known/OTHER/guarded-gap CE in the actual continuous
sequence teaches the missing context. A final fixed six-epoch alignment pass restores
original target-normalized CTC after that supervised warm start.

The final42-epoch model comprises12anchor epochs,12frame-normalized epochs,12verified-
frame epochs and6alignment epochs,4,620optimizer updates. A separate rejected12epoch
positive-only repair is preserved: total54completed repair epochs/5,940updates. This
is a diagnosed curriculum, not evidence that a single short loss change is sufficient.
All stages retain the original Squeezeformer/causal head,100known+OTHER+blank outputs,
data, learning rates and greedy decoder. Continuations retain optimizer state. Every
epoch covers923complete sequences,199positive cores,1,901replay clips and494verified
background windows. Losses never label incomplete O5S5 narratives as blank.

The frame audit admits150,876verified tokens:23,454known,103,251OTHER,24,171blank
inside0.10s guarded interior gaps.86,112positions remain ignored by frame CE. Sparse
anchors cover7,379/7,380events and all1,291known signs. Overlaps, unguarded gaps, clip
edges and padding receive no invented CE labels. Full source populations make added
loss weights independent of batch partitioning.

## Remaining failures

Final full-training errors are169deletions,34substitutions and220insertions. Training
fitting is now substantial, but not perfect. Connected validation has65substitutions,
90deletions and124insertions across284reference signs. Familiar phrases have112
substitutions,74deletions and93insertions across259signs. Contiguous ASLLRP has3
substitutions,6deletions and3insertions across24signs.

Pooled Citizen/SemLex validation accuracy retains95.77%/85.28%; that must not be
confused with actual CTC recognition of89.68%/77.91%. The domain gap is particularly
clear on LG: pooled cores15/57, but **actual CTC exact only4/57 (7.02%), with42/57empty**.
Training cores are199/199correct through both outputs. No full O5S5 narrative WER is
claimed because those annotations are incomplete. More diverse signer/domain coverage
remains a separate need; this repair does not establish reliable conversational ASL.

One of16verified background windows emits a known sign. The four unchanged failed
gates are connected deletions, familiar WER, median runtime delay and runtime-inclusive
latency. Source-clock timing recognizes138/296timed signs, misses158/296 (53.38%),
with0.0167smedian/0.3337sp95among recognized signs. Those timestamps exclude extraction,
normalization, scheduling and inference runtime; they are **not live latency**. The
Stage1encoder remains noncausal within bounded chunks. No iPhone/runtime/export or
confirmation-seed result is claimed.

## Verification and artifacts

`verification.json` verifies all4,520input hashes,54checkpoint hashes and code/recipe
freezes, exact epoch schedules/full coverage, saved optimizer states, recomputed final
training outputs,334development identities/references/edit counts and promotion gates.
Actual isolated/core CTC metrics are independently computed from final checkpoint
inference. CPU/MPS agrees on628/628checked tokens across8recordings; max logit difference
is1.38e-5. A63token prefix preserves the full sequence's earlier decisions. These are
bounded numerical checks, not full-device certification. All33focused tests pass;
compilation and `git diff --check` pass.

Corrections and checkpoint choices were determined from training data only. The fixed
finalepoch42was evaluated once on the development suite; no further tuning was performed
from those errors. Independent review cleared positive, anchor and loss-scaling work;
the reviewer hit a usage limit for the frame refinement, which the root agent reviewed
directly. One initial batch-weighted anchor attempt was interrupted before a checkpoint
and preserved; its weighting defect was corrected before the completed run.

- `PLAN.md`, `run_aligned.py`: final fixed alignment recipe and runnable entry point.
- `training_summary.json`: six final alignment epochs and checkpoint hashes.
- `evaluation.json`: all334development outputs, pooled/core metrics and failed gates.
- `verification.json`, `verify_results.py`: independent provenance/output verification.
- `focused_tests.log`:33focused test results.
- Sibling `joint_ctc_repair`, `joint_ctc_anchor`, `joint_ctc_balanced`, `joint_ctc_frames`
  reports preserve every prior recipe, diagnosis and training result.

Default runtime/checkpoints remain unchanged. The repair resolves the training-collapse
failure; the next product gate remains signer/domain generalization and sequence
precision, followed by real runtime measurement. Do not deploy this research checkpoint.
