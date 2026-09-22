# Continuous recovery: first execution pass

The user approved beginning the five-step recovery on2026-09-22. [PLAN.md](PLAN.md)
is the persistent checklist. This report records completed diagnostics, not completion
of the recovery program or a promoted model. No training, protected test access, source
dataset edits, or live behavior changes occurred.

## Fresh input verification

Loaded every combined record with the existing representation-aware loader; rehashed
features and6158distinct raw-video paths; verified seven pinned source manifests.
All6421records /7367windows passed in9.766seconds:4547train /1874validation.
The canonical494manifest integrity verifier also passed; its historical generic recipe
gate remains false. That does not prohibit preparing a new recipe-scoped contract.
Details: [input_verification.json](input_verification.json).

| Supervision | Train records | Train known labels | Validation records | Validation known labels |
|---|---:|---:|---:|---:|
| Approved phrase/subspan |283|44|211|26|
| Verified single sign |4264|100|1663|100|

The combined view is useful and not shown defective. It contains5927single-sign records
and494phrase/subspan records, not6421phrases. Label coverage is not signer diversity or
coverage of every transition. Seven approved training sequences include OTHER; no new
unknown/background target was inferred. Details: [supervision_coverage.json](supervision_coverage.json).

## Representation decision

| Representation | Records | Reuse and limit |
|---|---:|---|
| isolated_v17 |4194|Normalized32-frame single-sign input; suitable for identity replay. It cannot become timestamp-owned continuous supervision by concatenation. |
| windowed_landmarks_v17 |1975|Source-frame ranges retained. Only494are approved phrases/subspans;1481are single-sign records. Preserve the distinction and source clock. |
| timestamp_normalized_positive_core |252|O5S5core inputs; original timestamped raw-observation provenance permits rematerialization if needed without necessarily rerunning Vision on RGB. |

139ASLLRP records share baseline parent utterances within their roles; sampling must not
treat them as independent evidence. SemLex951validation rows share source-local signer
IDs with training, allowed by the current policy but not held-out-signer evidence.

Decision: no blanket re-extraction. The earlier native-rate1280px cache already exists;
its matched no-blank ablation failed. A new run must change a named supervision/input
condition, not repeat that run under a new filename. The current actual gap is the absent
combined training recipe, not a failed manifest or missing all100single-sign coverage.

## Same-checkpoint replay probe

Selected a short approved TRAIN source with two complete curated human-annotated cores:
asllrp:3378758.mp4:span00, NOW READ. Source SHA matches the approved manifest. Extracted
cores at the annotated source-frame boundaries using FFV1; every decoded pixel matched
the corresponding original frame. Original source was never modified.

Ran the unchanged live Stage2 script on the full recording and each core with identical
model provenance. Expected labels are used only for the summary, not model conditioning.
Disabled display, speech and English rewriting. Saved emission logits, feature windows,
source timestamps, commands, logs, history and model provenance under this directory.

| Input | Reference | Final output |
|---|---|---|
| Full recording |NOW READ|NOW READ|
| Supplied NOW interval |NOW|NOW|
| Supplied READ interval |READ|READ|

All three exited0and accepted their single window. This is a positive diagnostic control:
the active checkpoint can recognize this example both connected and cropped. It does not
locate the user's failure or establish useful streaming quality. The original clip is
0.934s, shorter than the1.067s model window; all results use EOF-tail processing. Startup
wall time is not recognition latency. Core replays reset frontend context and resampling,
so changes between a core and full stream would require further attribution.

Inputs/results: [paired_probe_inputs.json](paired_probe_inputs.json),
[paired_probe_results.json](paired_probe_results.json),
[paired_probe_status.json](paired_probe_status.json), [verification.json](verification.json).

### Additional controls and validation probe

Selected the first two lexicographic approved train contiguous recordings with all cores
eligible, at least two signs and1.1–4s duration: FAMILY IMPORTANT and FAMILY SIGN. Both
full recordings and all four supplied cores decoded exactly under the unchanged Stage2.
Then selected the first eligible validation recording longer than1.1s, before looking
at its predictions: FRIEND MAYBE (1.802s). Full stream emitted FRIEND, then FRIEND STOP;
oracle cores emitted FAMILY and STOP respectively. No checkpoint/threshold was changed.

All12replays exited0and had identical model provenance. The nine train-role probes are
positive controls, not an accuracy claim. The three validation-role probes are one
reused development source, not an independent test or enough evidence for a generalization
rate. Current manifest roles do not by themselves prove historical checkpoint exposure.
The longer validation source traverses two windows; the short train controls largely
exercise one-window/EOF behavior. Core extraction resets frontend context/resampling.

The validation failure survives supplied boundaries for MAYBE; boundaries alone do not
repair this example. FRIEND is correct in context and incorrect cropped, so an oracle
core is not automatically a superior input. This does not yet separate encoder error,
CTC-head error, resampling effects or visual variant mismatch.

See additional_probe_{inputs,status,results}.json and
validation_probe_{inputs,status,results}.json for exact selections, hashes and updates.

### Current Reel proposal/verifier on the same supplied cores

Reused ReelCascadeClassifier and the existing observation helper with the current default
20fps/640px contract; bypassed runtime activation/minimum candidate duration and commit
logic to isolate classifier decisions. Replayed eight lossless cores; no checkpoint edits.

| Core | Role | Landmark proposal / accepted | Visual verifier candidate / accepted |
|---|---|---|---|
| NOW |train|NOW / yes|NOW / yes|
| READ |train|READ / yes|READ / yes|
| FAMILY (7028838) |train|FAMILY / yes|FAMILY / yes|
| IMPORTANT |train|GO / yes|SMALL / no|
| FAMILY (49985301) |train|SCHOOL / no|SCHOOL / yes|
| SIGN |train|BIG / no|SIGN / no|
| FRIEND |validation|SAME / yes|COLD / no|
| MAYBE |validation|MAYBE / yes|MAYBE / no|

MAYBE's correct verifier candidate is rejected for low score and margin. This establishes
a correct top choice versus acceptance distinction on the supplied core, not proof that
thresholds should be lowered. FAMILY(49985301) shows why reporting verifier top1 alone
also misleads: the incorrect verifier candidate is accepted, but the proposal is rejected.
These are forced oracle-stage probes, not claims that runtime actually emitted those words.

Reel and legacy CTC use different weights/frontends. MAYBE versus STOP is not a controlled
head-only comparison. Both true identity errors and evidence/acceptance losses remain;
the working hypothesis is not simply 'all Stage1 signs are correct, only boundaries fail'.
Raw decisions/provenance: [reel_oracle_results.json](reel_oracle_results.json).

Final provenance check detected concurrent user app-shell changes to the Reel script
after the CTC checks. Preserved those edits and repeated only the8Reel core probes.
All proposal/verifier labels and acceptance decisions reproduced8/8; before/after
code hashes matched during that recheck. The pinned recheck supersedes the original
Reel run for reproducibility: [reel_oracle_recheck.json](reel_oracle_recheck.json).
All12CTC and16Reel-core executions completed successfully. `git diff --check` passed;
the large-artifact index was regenerated. No product code was edited by this recovery.

## Existing evidence and remaining diagnosis

The older clean_boundary_subset20260920 audit is genuinely paired within its research
checkpoint: held-out ASLLRP strict core accuracy192/261(73.56%) versus detected known
events152/261(58.24%). These are different measures, not a15.32point causal accuracy
improvement. That checkpoint is not the current live CTC selector; do not pool its
metrics with the new live replays.

The user's latest webcam session lacks timestamped human reference and saved raw logits.
Its hypothesis revisions establish UI instability, not an exact WER or a per-sign failure
cause. Use existing fully annotated examples for further paired diagnosis; if attributing
individual webcam mistakes, obtain human timing references rather than labeling guesses.

Next: use the same current Stage1 weights and matched temporal representation for identity
and sequence comparisons; separate raw top1, confidence rejection and committed output.
Pin a combined-data recipe whose objectives distinguish identity replay from sequence CTC.
Steps3–5remain explicitly open in PLAN.md. Do not infer that the successful positive
control solves generalization or that a new training run is already ready to launch.

## Combined-data comparison launched — 2026-09-22

First preflight exited134 on Apple MPS matmul dtype assertion. Verified isolated/windowed
caches are float16 and positive-core cachesfloat32. New runner had cast only CTC chunks;
fixed make_records once to widen all cached inputs tofloat32, covering preflight/identity/
evaluation paths without geometric normalization. Failed attempt JSON/log preserved.
Second preflight passed all6421records,7367full-model windows per arm; largest32single
records use41windows and4phrases21windows. Frozen base gradients absent; adapted base
and both heads finite/nonzero. Zero optimizer steps. Four focused tests including dtype,
ID collision, repeated CTC/OTHER and normalized weighting passed; independent final
review clear, git diff --check clean. These preparation defects do not establish causes
of earlier trained-model failures.

Pinned recipe SHA b077e6939c6dc71c581cea2ee669348e2b4cc1d1cedef50eb7f8a9fd85c4ee64;
runner3c1ea6c7cfb4e7698eace0839d9a70de897a329335819f4400b65b4fd7b7a9fb;
preflight3ce024fca29bbb43ce3765270efeea7a11b520d432d0d055fc433850cae76999.
Launched dedicated --train detached under caffeinate, PID98647,2026-09-21T17:45:51.436154UTC
(2026-09-22PHT). Two seeds, two arms,12epochs each. Completion/failure notifications wired.
No training polling or completion claim. Reports/models use combined_frozen_joint_v17_20260922.
Current ground truth and all-five checklist updated; source datasets/live defaults and
old generic gate unchanged. Next safe action: read status/results next session, compare
selected-best training versus validation and single-sign retention before any deployment.

