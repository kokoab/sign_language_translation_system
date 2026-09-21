# live-streaming — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

45 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-21 — matched no-blank CTC ablation failed promotion

Precheck found that the existing aligned grounded CTC run already used source-rate
1280px ASLLRP, native30fps signer-disjoint local phrases, an8-frame causal window,
stride4, the exact-core-adapted Stage1 initialization, locked100+OTHER outputs and timed
alignment. The audit's proposed native-rate CTC was therefore not repeated.

The only missing controlled change removed standalone transition clips labeled blank.
It completed18 MPS epochs in106s and selected epoch8. Local held-out-signer performance
regressed from25.5% exact/37.04% WER to16.0%/45.19%; ASLLRP contiguous remained33.33%/
41.67%; NCSLGR regressed5.41%/78% to0%/84%. Isolated exact improved slightly from
82.37% to83.19%. The checkpoint fails promotion and runtime remains unchanged. These
results reject blank clips as the dominant cause and close another decoder/supervision
patch. Citizen test and external reserved evaluation stayed sealed. Report:
`artifacts/reports/native_ctc_no_blank_v17_20260921/`.

## 2026-09-21 — data-path audit identifies the next minimal Stage-2 experiment

The source videos and annotations remain usable; the continuous preprocessing contract
does not preserve them faithfully. The continuous observer caps detection at640px and
20fps even though all audited ASLLRP videos are1280px on the long side at29.97fps. The
current confident filter then rejects6,020/11,936 events under its four/six-observation
floors. This compounds the already measured context-window defect where all6,641 known
windows ended before the sign end. Exact-core held-out ASLLRP accuracy of72.95%/82.35%
on other/contiguous data confirms that sign identity survives better than localization.

The next experiment is one small causal frame-sequence CTC head with an exact-core
classification auxiliary, trained from a native-rate1280px ASLLRP cache plus the
existing signer-disjoint local phrase split. Short signs stay in through masks rather
than arbitrary sample floors. O5S5 remains positive-core supervision only. Do not add
another boundary state machine, slow/interpolate cached sequences, or acquire replacement
data before evaluating this corrected input contract. Audit:
`artifacts/reports/stage2_data_path_audit_20260921/`.

## 2026-09-21 — boundary tolerance was strict, but it was not the detector's only failure

The source/Luna review shows that±100ms exact-edge scoring should not be used to declare
ASLLRP annotations bad. The official source interval and the single-reviewer Luna
interval differ partly in semantics, especially final holds. However, the coherent
decoder still reached only40.46% F1 at±200ms and0/20 intentional-repeat probes, so the
failed promotion is unchanged. Review:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/comparison.html`.

## 2026-09-21 — frozen coherent decoder improved insertions but still failed

Three user-requested low-effort Luna agents independently audited the failed segment
head, existing activity code and timing. A Luna implementation then froze the checkpoint,
averaged overlapping-window state evidence on absolute source timestamps, applied the
existing v16-style hand-presence outer range and required coherent
OUTSIDE→START→SIGNING→END paths with END→START repeats. Edge bias was selected only on
888 training sources; the unchanged0.56 known gate and232 validation sources were used
once. Predictions overlapping excluded annotations were ignored consistently.

The evaluation improves±100ms boundary F1 from14.44% to18.79% and±200ms F1 from24.41%
to40.46%. Visible WER improves204.69%→103.65%, mainly because insertions fall261→33,
but deletions rise66→121; substitutions are45 over192 references. Precision/recall is
20.96%/17.04% at±100ms and45.12%/36.67% at±200ms. Synthetic probes worsen to2/20 held
once and0/20 repeated twice. Thus incoherent peak pairing caused many insertions, but the
learned boundary evidence itself remains inadequate. Do not promote or patch further.
Verification confirms D+I+S arithmetic, train-only selection, sealed Citizen test and
unchanged frozen checkpoint. Report:
`artifacts/reports/segment_first_coherent_decode_v17_20260921/REPORT.md`.

Next authorized action is a small Luna-reviewed offline boundary-annotation pilot. Use
dense train-only frame strips, require independent agreement and admit only consensus
edges before deciding whether this can scale. Luna output is provisional supervision,
not ground truth, until agreement is measured.

The user then required exactly one Luna per clip. The completed24-clip pilot produced19
medium/high, non-censored provisional intervals, but only6/19 match both existing edges
within100ms and14/19 within200ms; median absolute end disagreement is128ms. With one
reviewer per clip there is no independent consensus, and the disagreement is not a
consistent timing offset. Do not train or scale these pseudo-labels as truth yet. Pilot:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/`.

## 2026-09-21 — segment-first confident-only experiment completed and rejected

The corrected detached run completed eight full-coverage MPS epochs in5m05s. An
initial18-second startup attempt failed before epoch1 because the runner required every
SemLex replay class even though SemLex train lacks CHILD/TAKE/THEY; the failed evidence
is preserved under `failed_attempt_01/`. The loader now follows the established rule:
all100 Citizen classes are mandatory and available SemLex classes are optional. The
expanded precheck exercises the exact replay loaders and counts500 Citizen train,480
SemLex train,378 Citizen validation and978 SemLex validation examples.

The model fails promotion. Across232 held-out complete ASLLRP sources, absolute boundary
F1 is14.44% at±100ms and24.41% at±200ms. At±100ms precision is9.44% and recall30.69%;
at±200ms precision is15.96% and recall51.89%. Visible locked100 WER is204.69% with66
deletions,261 insertions and66 substitutions over192 reference glosses. Exact-core
known/unknown balanced accuracy is63.34%; exact-core gloss is169/226=74.78%, including
ASLLRP contiguous81.82%, ASLLRP other81.22% and O5S538.24%. Matched predicted-segment
gloss is76/106=71.70%. Isolated validation is Citizen92.33% and SemLex83.23%. Synthetic
time-warped probes pass6/20 held-once and2/20 intentional-repeat cases.

Verification passes3/3 focused tests, full coverage and64/64 CPU/MPS gloss and gate
decisions. Citizen test stayed sealed. Do not promote the checkpoint or change runtime.
The result confirms that clean whole-sign identity is substantially more learnable than
automatic boundary localization; this four-state frame head does not solve segmentation.
Report: `artifacts/reports/segment_first_v17_20260921/REPORT.md`.

## 2026-09-21 — segment-first confident-only experiment launched

User authorized the minimal segment-first experiment with one exit notification and no
polling. Precheck passed the exact builder over all5,331 confident events:7,788 training
boundary windows,4,303 continuous training segment cores,1,797 validation boundary
windows and1,028 validation segment cores. It verified the locked100 vocabulary,
signer-disjoint roles, questionable-annotation masking, zero O5S5 background targets,
real MPS execution and a tiny fit from1.613 to0.198. Focused tests pass3/3.

Detached PID42246 is training for eight full-coverage epochs from the plain Stage-1
checkpoint. The model has one OUTSIDE/START/SIGNING/END linear head, one internal
KNOWN/UNKNOWN head and the existing100-gloss classifier. The final report will measure
absolute boundary F1, rejection, predicted-segment gloss accuracy, visible WER,
isolated retention and held/repeat probes. No CTC, alphabet expansion, gap-derived
transition target, runtime promotion or Citizen-test access is part of this run.
Report directory: `artifacts/reports/segment_first_v17_20260921/`.

## 2026-09-20 — strict whole-sign audit completed: ASLLRP identity learnable, localization fails

Detached PID55540 completed in85.7s. It evaluated8,367 strict whole-sign cores and
42,111 complete-ASLLRP endpoint windows without training. Verification passes;12
failure/control videos and a contact sheet were generated and inspected. Citizen test
remained sealed.

The adapted encoder recognizes strict ASLLRP-other cores at901/1,040=86.63% train and
178/244=72.95% held-out, versus the starting encoder31.15%/51.23%. Limited neighbouring
context lowers adapted held-out accuracy to67.21%, so missing context is not the current
identity bottleneck. Contiguous strict-core adapted accuracy is88.14%/82.35% train/
held-out. This establishes meaningful learnability for clean ASLLRP sign evidence.

O5S5 remains a source/signer failure: adapted strict-core accuracy is101/133=75.94%
train versus10/43=23.26% LG held-out; cleaning incomplete/overlapping events does not
close it. Boundary localization remains inadequate: train-selected phase threshold
has43.14% held-out known recall and79.10% non-known recall (61.12% balanced), original
three-way phase accuracy49.79%, and correct gloss+phase detects152/261=58.24% eligible
held-out ASLLRP events. Gloss confidence alone is59.57% balanced. Gaps remain annotation
gaps, not certified physical transitions.

Decision: keep locked100; reject checkpoint. Use strict whole-sign manifest for identity
work and separately reviewed boundary evidence for localization. Do not infer a physical
TRANSITION class from annotation gaps. Reports and playable examples:
`artifacts/reports/clean_boundary_subset_20260920/`.

The follow-up confident-only manifest further restricts supervision to at least six raw
target observations,80% hand visibility, maximum1.20s duration and maximum80ms target
timestamp gaps. It retains5,331 events, including1,017 known and4,314 unknown, across
61 train classes. Use this manifest for the next controlled experiment; do not silently
fall back to the combined supervision manifest. Report:
`artifacts/reports/confident_supervision_v17_20260920/REPORT.md`.

## 2026-09-20 — strict whole-sign recognition/localization audit authorized and prechecked

User authorized the corrective report-only experiment with exit notification and no
polling. It will evaluate the unchanged phrase-adapted encoder and failed boundary-phase
checkpoint on one strict derived subset. ASLLRP rows with lost crop-completeness are
excluded; every retained event is non-overlapping, inside the raw clock, has at least
four raw target observations, and covers both annotated edges within60ms. O5S5 supplies
exact sign cores only, never gap/boundary evidence. Annotation gaps remain named gaps,
not asserted physical transitions. Train-only thresholds will be applied unchanged to
held-out signers.

Full dry run retained8,367/11,936 annotated cores:1,536 known and6,831 unknown, plus
42,111 fixed0.27s endpoint windows from complete ASLLRP crops. Exclusions include3,122
events with fewer than four target observations,60 outside the raw clock,141 within
incomplete source crops,357 overlapping annotations and65 without both-edge coverage;
reasons may overlap. Real MPS precheck and100-label/output contracts pass. The worker
will report whole-core and whole-core-plus-context recognition, known/unknown/gap
rejection, event localization, and selected playable failures. No training, protected
test access or automatic promotion. Runner/precheck:
`artifacts/reports/clean_boundary_subset_20260920/`.
Detached MPS worker PID55540 launched with one completion/failure notification; do not
poll it.

## 2026-09-20 — direct boundary-data audit corrects earlier mismatch claim

User requested concrete diagnosis of cropped signs, learnability, and alphabet expansion.
Compared all 1,166 manifest rows with original ASLLRP CSV/crop provenance, inspected
sampled frames from six examples, and created ten original-speed source/input excerpts.
No training ran. Re-evaluated final checkpoint on all 38,043 train and 9,380 validation
context windows; validation known counts reproduce 708/1,275. ASLLRP-other known gloss
train/held-out accuracy is 88.14%/59.90%, phase 69.33%/51.04%; O5S5 known 68.39%/32.02%.

21 ASLLRP source crops were previously marked incomplete (22 clipped occurrences, all
OTHER); materialization did not preserve that flag. All 5,366 known train windows end
before sign completion, 2,349 have less than half target observations, 31 known events
produce no window, 46 O5S5 window endpoints overlap another annotation, and two FEEL
windows contain zero raw target samples. These are not physically cropped exact cores.
11,428/11,731 TRANSITION windows overlap preceding signs: valid in principle for an
endpoint task, but annotation gaps were not visually certified as physical transitions.
Training uses selected event/gap moments; replay samples continuously and fuses durations.
Thus previous claims of no remaining mismatch and architecture failure were too strong;
corrected original report and current state, preserving model and metrics.

ASLLRP train OTHER has 5,070 lexical occurrences (83.26%) and 377 fingerspelled (6.19%)
out of 6,089. Adding alphabet classes would not resolve ordinary unsupported vocabulary.
73 known classes occur in continuous train annotations; 16 have one signer. Data has
learnable signal, but no proof of reliable unseen-signer continuous recognition. Next
safe action: curate complete and unambiguous evidence with reviewed boundaries, then
separate recognition from localization on that fixed subset before further training.
Report, measured fit, source audit, videos, and verification:
`artifacts/reports/boundary_data_audit_20260920/`. Citizen test remains sealed.

## 2026-09-20 — clean boundary-phase experiment completed and rejected

Detached MPS worker PID95985 completed all12 full-coverage epochs in24m52s. The fixed
0.27s/0.53s trailing-window model used39,944samples and625updates per epoch. Its locked
100-gloss plus separate KNOWN/UNKNOWN/TRANSITION contract and saved CPU/MPS decisions
passed64/64 checks; Citizen test remained sealed.

The result fails runtime use: validation phase accuracy50.23%, known-core gloss55.53%
(708/1,275), and complete-ASLLRP online WER106.82% with86 deletions,110 substitutions,
133 insertions and55/237 empty transcripts. Isolated Citizen/SemLex fell from the
starting checkpoint's95.24%/85.28% to93.39%/84.76%. Therefore the earlier failures
cannot be attributed only to replacement sampling,1.07s dilution, combined NO_EMIT
competition, or train/live window mismatch. Short independently classified windows do
not yet generalize reliable phase boundaries and gloss identity. Reject checkpoint;
default runtime unchanged. Report:
`artifacts/reports/boundary_phase_v17_20260920/REPORT.md`.

## 2026-09-20 — boundary-phase fixed-window experiment launched

User authorized the researched non-CTC experiment with one exit notification and no
polling. It keeps the phrase-adapted 100-gloss Stage-1 model and adds one three-way
endpoint phase head: KNOWN, UNKNOWN or TRANSITION. Both training and online replay use
the shared 32-frame normalization at0.27s/0.53s; phase is defined at the trailing-window
endpoint. Held identical glosses stay one run, and the same gloss may repeat only after
a predicted transition. UNKNOWN is internal and cannot enter the visible transcript.

Strict routing is executable: all1,160 complete ASLLRP rows can supply known cores,
explicit OTHER intervals as UNKNOWN, and guarded interior gaps as TRANSITION; the six
incomplete O5S5 rows can supply known cores only. Full normalization dry run produced
38,043train and9,380validation context windows, with no O5S5 negative target. Every
admitted training sample will appear exactly once per epoch; no replacement sampler,
1.07s target or CTC loss is used. All1,166 raw archives,100 label indices, source and
signer gates, model output shapes, and real-MPS tiny-fit passed. Citizen test remains
sealed; no automatic runtime promotion. Runner/precheck:
`artifacts/reports/boundary_phase_v17_20260920/`.
Detached MPS worker PID95985 is running with one completion/failure notification;
do not poll it.

## 2026-09-20 — Finish-time CTC completed: lower WER masks deletion collapse

Detached MPS run completed20 epochs in10m16s with full coverage each epoch. The frozen
Stage-1 + one400,174-parameter bidirectional GRU CTC head reaches83.80% connected and
88.03% familiar WER versus repaired causal98.24%/107.72%, but deletes215/284 and
191/259 reference glosses. It emits only74/284 connected and68/259 familiar tokens;
160/225 connected outputs are empty and familiar exact is0/97. Contiguous WER83.33%,
LG core2/57, isolated Citizen344/378 and SemLex812/978, verified blanks16/16 empty.
Six adjacent duplicate outputs remain versus three references; this is not an expert
held-once/twice evaluation. Locked100 visible-label contract verified; blank/UNKNOWN
internal, protected test untouched, no promotion.

Root-cause review found the initial reporter summed alignment `match` operations into
WER. Added a failing regression, centralized the three edit operations, regenerated
evaluation/report and independently recomputed every source. Model/checkpoint/predictions
were unchanged. Ten focused tests and compilation pass. Report/correction/verification:
`artifacts/reports/finish_ctc_v17_20260920/`. Decision: keep current runtime; the simple
frozen-head recipe underfits continuous training (best71.34%, final74.05% WER), so the
next work must improve continuous evidence/coverage rather than add decoder heuristics.

## 2026-09-20 — Finish-time bounded CTC experiment authorized and ready to launch

User chose the researched bounded continuous direction and authorized the experiment
with exit notification and no polling. The fixed spike freezes the Stage-1 encoder and
trains one bidirectional GRU CTC head on the existing complete phrases, isolated clips,
verified positive cores and verified blank clips using one objective. Internal outputs
are blank + locked100 + UNKNOWN; only the100 glosses can reach the visible transcript.
Seed17201,20 full-coverage epochs, fixed final evaluation against the repaired causal
CTC on the same334 development recordings; Citizen test remains sealed and no runtime
promotion is automatic. Nine focused tests, compilation and real MPS packed-GRU
forward/backward pass. Runner/precheck:
`artifacts/reports/finish_ctc_v17_20260920/`. Final action is a detached launch with one
completion/failure notification; do not poll.

## 2026-09-20 — direct-translation root cause established: caption memorization

Completed a read-only audit of frozen data, features, losses, outputs and pairing
controls. BART reproduces a 994-row training caption exactly on 11/12 held-out clips;
mT5 has median nearest-training-caption similarity0.803 and six of12 at least0.80.
BART zero-visual emits one exact training caption for all12 rows. Correct pairing does
affect chrF, so the visuals weakly route caption choice, but neither hybrid composes a
translation. The994 pairs cover1.63h/four signers; two supply80.4%. Extraction is not
globally broken:6/4,813invalid windows,98.7%median hand presence,72/994downsampled.
Eight rows exceed six English words/s and are viewer flags, not proven bad labels.
Root cause is the isolated-window encoder/linear bridge trained end-to-end with an
oversized decoder and insufficient aligned continuous data. Do not rerun another LM
swap on this recipe. Keep gloss-free work separate; current Stage2 needs one temporal
blank+100-gloss head and signer-disjoint hold/repeat/rest/transition supervision.
Report/viewer: `artifacts/reports/direct_translation_failure_diagnosis_20260920/`.
No training or protected test access. Fresh verification passes report self-check,
Python compilation,44video-panel HTML audit, JSON parsing, artifact indexing and
`git diff --check`.

## 2026-09-20 — English comparison complete; independently verified negative translation result

User requested update. No active workers. All20 BART epochs completed,10552.19s.
Recomputed BART/trim/mismatched BLEU+chrF from saved outputs: exact agreement.
Trimmed mT5 keeps21/21 predictions identical,303.52M vs589.39M parameters (-48.5%).
BART146.41M: BLEU1.061/chrF17.464 vs mT5 1.076/18.989; fails1-point chrF margin.
Mismatched BART BLEU1.287 exceeds paired1.061, weakening grounding claim. Inspected
predictions contain unrelated content. Isolated Citizen358/378, SemLex841/978.
No deployment promotion; no new training. Appended interpretation to REPORT and
updated canonical state. Next safe action: discuss alignment/generalization diagnosis,
not another unexamined model swap. Twelve correlated sentences insufficient for
accuracy equivalence; official Citizen test untouched.

## 2026-09-20 — English comparison exit

English model comparison completed; see english_comparison_20260917/REPORT.md. No model promotion; review predictions and controls.

## 2026-09-20 — approved English comparison resumed after storage available

User asked to continue. Baseline already completed epoch20 with poor translations;
no active worker. Internal free58.5GiB now exceeds startup guard. Verified4,295
frozen input/code/model/tokenizer hashes and completed baseline checkpoint SHA.
No missing assets or recipe changes. Prior failed queue records archived under
english_comparison_20260917/failed_attempt_05; precheck JSON records verification.

Proceed with approved trim evaluation and one BART pilot, internal storage only;
existing completed mT5 is a weak comparison baseline, not a deployment candidate.
No model/code modifications or unnecessary training rerun. Final action: detach
comparison worker with terminal notification, no assistant polling. Need final
predictions, visual controls and isolated retention before interpreting results.

## 2026-09-18 — final hybrid results verified; English comparison storage-blocked

User-requested update: baseline completed epoch20, no active processes. Independently
recomputed saved initialized/final/zero-visual BLEU and chrF, exact matches. Final
1.076/18.989 vs unchanged Uni-Sign8.206/38.059; training loss0.16688. Citizen isolated
360/378 unchanged; SemLex832/978 vs834 initially. Inspected translations contain
unrelated details. Automated weak descriptive-check pass is not translation success.
Appended review to REPORT.md and replaced stale epoch13 current-state text.

English comparison failed in startup checkpoint-space guard before trim evaluation
or BART training. Internal free7.7GiB (<8GiB); SSD not mounted. Do not lower safety
threshold or restart blindly. Next: adequate storage and assessment of poor baseline
before progressing the approved comparison. No checkpoint changes/test-gate access.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation completed. See stage1_direct_translation_20260917/REPORT.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — user-requested update: baseline saved epoch14; comparison waiting

One requested status check confirms live MPS baseline worker, completed/saved epoch14
of20, all994 translation/1901 isolated samples covered that epoch; translation loss
0.34381 (epoch13 0.41154). Current provenance says recovery fromepoch13, checkpoint
root internal artifacts/models, consistent with current ground truth. No new failure
record. Comparison supervisor45994 waits on baseline45878; BART has not started.
Corrected stale comparison status.json from failed to queued/waiting; canonical
state was already queued. No running code, model or training changes; no polling.

## 2026-09-18 — epoch13 saved; external EIO at epoch14; verified internal recovery

User requested update. Saved epochs8–13 after file-handle repair; SSD returned EIO
at epoch14 save/flush. No workers active and no BART training. Epoch13 training
loss0.411538/isolated0.004517, not validation scores. Copied SSD latest to internal
epoch_13_before_internal_resume.pth; source/copy hashes, all ZIP CRCs, and epoch13
mmap metadata verified. All valid source checkpoints retained; no SSD writes in retry.

Both runs now target internal artifacts/models. Internal free14GiB before2.37GB
recovery copy. Split storage gates:8GiB startup for model loading versus4GiB running
checkpoint free space (known2.37GB maximum), plus1GiB internal report reserve.
Added regression for phase-specific limits, archived failed attempt/source manifests,
recorded internal_recovery.json and EPOCH13_RECOVERY_REPORT.md, refreshed code hashes
without data/architecture/optimizer/recipe changes. Next: focused checks, epoch14
resume and comparison event queue; no polling. Final evaluation still pending.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — reproduced SSD native-writer failure; full-size file-handle alternative passed

User requested update. No live workers. Baseline reached epoch8 save then rename
failed ENOENT; temporary file now exists at exactly1,879,048,192 bytes but invalid
ZIP. Epoch7 remains valid; BART never started. Disk free14GiB internal/280GiB SSD.
Reproduced direct-filename torch.save failure using synthetic2.4GB CPU storage test.
Same-size Python binary file-handle save + flush/fsync/rename passed full ZIP CRC
verification (2,400,001,577 bytes,27.80s). Evidence JSONs and SSD_WRITE_REPORT.md saved.
Root PyTorch/ExFAT defect not identified; observed API-path difference verified.

Shared baseline/BART checkpoint writer now uses tested file-handle path and distinct
latest.filehandle.tmp.pth (old corrupt output retained). All valid checkpoints kept.
Archived attempt04 baseline and03 comparison; refreshed code/baseline hashes without
data/recipe changes. Compilation/diff checks pass. Next: resume fromepoch7 and queue
comparison, with notifications and no polling.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — SSD launch failed on internal ENOSPC before training; reserve raised

User requested update. No live worker; last verified epoch remains7. SSD recovery
loaded model then failed writing provenance and failure/status due internal ENOSPC.
Checkpoints on SSD alone did not prevent internal exhaustion during model loading;
swap/memory pressure is a likely contributor, not separately instrumented. Comparison
never trained. Current internal free space observed16GiB, SSD282GiB. Archived attempt
03 baseline and attempt02 comparison, keeping all checkpoints.

Raised internal-space gate from512MiB to8GiB; external8GiB/mount guards unchanged.
This is conservative headroom, not a guarantee against concurrent disk consumption.
Refreshed frozen code/baseline-manifest hashes, data unchanged. Next: verify focused
recovery checks and relaunch from epoch7, requeue comparison without polling.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — approved corrupt-temp cleanup and verified SSD recovery

User explicitly permitted deleting corrupt latest.tmp.pth and using attached SSD.
Deleted only that invalid ZIP (2,063,536,128 bytes). All valid local checkpoints
retained. Verified writable ExFAT SSD /Volumes/secret (UUID recorded), ~283GiB free.
Copied epoch7 to SSD; SHA matches original. Actual mmap load and small atomic
checkpoint replacement verified on SSD.

Changed baseline runner for external checkpoint root, verified recovery record,
epoch7 optimizer restoration, mount/path guards, >=8GiB checkpoint free-space and
>=512MiB internal report-space checks before work/epochs/saves. Comparison runner
uses external root and recovered baseline launch record. Added test_ssd_recovery_v17;
red tests exposed original epoch1-only restore and absent space guards, then five
focused recovery/queue/label tests pass. Archived previous scripts/manifests/failed
queue terminal files. Updated frozen code hashes in both manifests while asserting
all input hashes unchanged. Recovery report/record/verification preserve provenance.
Next: detached epoch8 resume and event-based requeue; no polling. Keep SSD attached.

## 2026-09-17 — disk exhaustion identified; verified epoch-7 recovery checkpoint

User reported failed comparison. Baseline actually failed saving epoch 8, then
errno28 prevented status rename; stale status/GT said training/running. No active
baseline/comparison process. Comparison stopped on failed prerequisite and never
started BART or trim evaluation. Assistant missed the prerequisite failure when
queuing. Verified latest.pth ZIP CRCs and mmap metadata: epoch7, history1–7,
543 optimizer states. latest.tmp.pth is an invalid ZIP (~1.9GiB); free disk ~3GiB.
No files deleted. Archived failed status/logs under failed_attempt_02 and corrected
canonical/status files. Added comparison/DISK_FAILURE_REPORT.md and diagnosis JSON.
Next: obtain explicit permission for corrupt temporary checkpoint deletion per
AGENTS.md, prepare epoch7 recovery with disk-space checks, and requeue comparison.
Keep all valid checkpoints. Current resume helper is hard-coded to epoch1 and
must not be blindly rerun. No final translation-quality result yet.

## 2026-09-17 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-17 — English comparison prepared and reviewed; queue launch next

Prepared pinned BART-base weights/tokenizer, 64,000-entry mT5 tokenizer and explicit
row mapping from 500 Brown documents plus TRAIN-only requirements. All protected
training/prefix/basic-character token sequences and decoded strings preserved.
BART target mean 21.56, maximum 63, no truncation. Frozen assets/code/data manifest
and corpus SHA recorded under english_comparison_20260917. No new GPU work yet.

Added vocab_trim.py, run_comparison.py and focused tests; reused existing fixed
baseline loading/training/evaluation without modifying its source. CPU check covers
retained mT5 logits exactly; queue test uses a real short child process exit event.
Review corrected supervisor exit-code propagation and canonical-status updates;
second tokenizer/integration review found no further high/medium issues. BART repeat
suppression explicitly disabled. PRECHECK_REPORT.md explains approval, recipe and
limits. Next action after final checks: detach queue on existing baseline supervisor
exit; no polling. Stop/report if baseline fails or fails the zero-visual grounding
screen. Otherwise trim-evaluate, preflight/reset, train one BART and report results.

## 2026-09-17 — user approved English comparison; preparing event-based queue

User approved the documented BART-base pilot and no-retraining mT5 vocabulary
comparison, requested results only and no polling. Current baseline still training,
last checked epoch 5; do not overlap GPU jobs. Preparing new report-local runner
`artifacts/reports/english_comparison_20260917/run_comparison.py`; reuse existing
frozen-input loader/train/eval helpers without changing their source or live defaults.
Queue waits for baseline supervisor process exit with kqueue NOTE_EXIT, then verifies
successful completion and baseline zero-visual grounding before full new training.
Focused queue/bucket tests failed first for absent runner, then uncovered Python 3.9
kqueue lacks a context manager; fixed with contextlib.closing, now 2/2 pass.
Pinned BART download started; no new GPU computation/optimizer steps yet. BART's
released no_repeat_ngram_size=3 must be overridden to 0 to honor allowed repetition.
Independent trim helper/test and bounded runner review underway. Next action: verify
CPU preparation and freeze hashes, then detach comparison queue; no polling.

## 2026-09-17 — English text-model alternatives researched; approval pending

User requested a review before any further training. Added report-local
`english_text_model_review_20260917/{REPORT.md,inspect_candidates.py,verification.json}`,
official pinned configs, inspection log and `bart_api_check.json`. Parameter counts
use meta allocations only; train-target statistics use the 994 frozen TRAIN
references. No evaluation references used for vocabulary design and no vocabulary
actually trimmed. No new pretrained weights, GPU benchmark, optimizer step or new
training run. Existing run and its files unchanged.

Verified current text parameters 582.40M; embeddings/output 384.17M (65.2% of the
589.39M hybrid). BART-base hybrid 146.41M; T5 v1.1 and FLAN base 254.57M; FLAN small
83.88M. Hypothetical 64K / 32K mT5 trims 303.52M / 254.57M. Target mean 24.66,
median 23, p95 45, maximum 70, padding 64.8%. Tiny random CPU BART compatibility
forward/generation passes through existing DirectTranslation; this is not accuracy
evidence. Official model cards and vocabulary-trimming research cited in REPORT.

Recommendation for approval: one BART-base English challenger plus no-retraining
trimmed-checkpoint comparison after completed mT5 results. Full recipe and proposed
retention margins documented; 12 existing sentences cannot establish equivalence.
FLAN is not strictly English-only. No speedup or mobile-readiness claim. Next action
is user review/approval, not launch. Preserve all existing training and data gates.

## 2026-09-17 — user-requested resumed-run throughput check

Saved provenance confirms MPS, float32, 589,390,373 parameters. Resumed worker
exists and no completion/failure file is present. Epochs 2/3/4 took approximately
1,003/1,132/1,123 seconds (497 updates each, 2.02–2.28 seconds/update), versus
142 seconds for projection-only epoch 1. Full encoder/text training, batch two,
fixed masked padding and twice-per-step MPS synchronization/cache release explain
why this is substantially heavier than warmup; individual overhead contributions
have not been profiled. At recent throughput, 16 remaining epochs take roughly
five hours plus final evaluation, not a guaranteed completion time. No training
changes or restart; continue detached and inspect results after exit notification.

## 2026-09-17 — epoch-2 MPS failure diagnosed; checkpoint resume verified

Initial run completed epoch 1 (994 paired / 1,901 isolated examples, 497 steps,
translation loss 2.902774, isolated loss 0.065100, 142.043 seconds), then failed in
Adafactor with MPS memory exhaustion. Train-only tiny-fit and gradient checks passed;
no final translation-quality result. Preserved original run artifacts under
`stage1_direct_translation_20260917/failed_attempt_01/` and epoch-1 checkpoint as
`epoch_01_before_resume.pth`.

Changed `active/v17/direct_translation_v17.py`, its focused test and the report-local
runner: fixed masked source/target padding, unused MPS cache release around optimizer
updates, checkpoint alias preservation, verified epoch-1 resume and stale completion
archiving. Two real-network tests pass (including padded/unpadded loss equivalence).
Four maximum-length real joint train-only updates passed; updates discarded.
Observed driver allocations at measurement points <=10.33 GB before / 8.78 GB after
cleanup; not continuous peak measurements. Inputs, architecture, objective and
20-epoch recipe unchanged; all non-code manifest fields match original. Epoch 2
restarts with seed + epoch, not the interrupted partial epoch RNG state. Added
RESUME_REPORT.md and hashed resume_provenance.json. No test-gate access or promotion.
Next action: detached resume, then inspect exit report after notification; no polling.

## 2026-09-17 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-17 — approved Stage-1 direct-translation pilot prepared

User authorized the hybrid experiment, accepted repetition for this stage, and
requested results with detached execution/no polling. Kept the existing experiment
branch and all unrelated changes. Added `active/v17/direct_translation_v17.py`,
`test/test_direct_translation_v17.py`, the report-local `run_experiment.py` and
`docs/superpowers/plans/2026-09-17-stage1-direct-translation.md`.

Architecture preserves unpooled, ordered Stage-1 frame tokens, projects them to
the released ASL mT5 input, and directly generates English. The existing isolated
head is trained with auxiliary CE. No CTC, gloss decoding or repeat heuristics.
Full text-component state is required to load strictly from the previously hashed
Uni-Sign How2Sign checkpoint. This is a hybrid pilot, not native Uni-Sign replication.

Prepared 994 paired TRAIN utterances (How2Sign signers 3/5/8/11), excluding 32 clips
from adaptation-held signers 1/2 and one previously failed feature extraction.
Reused existing Apple continuous features. Extracted 21 evaluation videos under
the same schema; corrected reused extractor bookkeeping to mark validation access.
Retained verified 1,901 isolated TRAIN / 1,356 validation entries. All 4,272 input
hashes check; max source tokens 256, max English target tokens 70, no target
truncation. No official Citizen test access. External mT5 pretraining may include
evaluation signers; whole-system signer disjointness is not claimed.

Two real-network checks failed first for the missing new module, then pass after
implementation: variable sequence masks, encoder/projection/text gradient flow,
isolated-head gradient, target-free generation, and deterministic full coverage.
Initial data preparation rejected SemLex's different train/val root convention;
corrected source-specific path checks and reran successfully before training.
Added an external subprocess wait to report fatal worker crashes without polling.

Fixed 20 epochs (one projection warmup, 19 joint), seed17111, Adafactor, translation
CE + 0.5 isolated CE, all admitted samples used each epoch. Full-model train-only
gradient/tiny-fit preflight runs inside the worker and resets before main training.
Compare final epoch with initialization, unchanged Uni-Sign and zero-visual input;
report isolated retention, losses, coverage and all translations. Repetition is
not a failure gate. Final action is detached launch; next action after notification
is to inspect REPORT/FAILURE. No production replacement or model-quality claim.

## 2026-09-16 — clarify Uni-Sign adaptation potential during discussion

User observed that some translations are close and asked about fine-tuning,
architecture lessons and isolated-sign failures. Rechecked official model/training
code and paper plus saved predictions. Some full-sentence outputs preserve useful
content; exact-match count must not be interpreted as semantic accuracy. Rejection
applies to unchanged deployment, not fine-tuning feasibility. The architecture has
separate downstream isolated-recognition fine-tuning; the evaluated checkpoint is
the How2Sign sentence translator. Short-clip failure is confounded with domain,
signer and capture differences. Added this interpretation to REPORT.md. No training
or implementation started. A proposed adaptation comparison needs correctly paired
target-domain data and unseen-signer evaluation; keep the native architecture first.

## 2026-09-16 — reviewed Uni-Sign results; reject drop-in replacement

On the user's status request, verified completed status and all 21 predictions;
independently recomputed BLEU 8.206457 and chrF 38.059096 over the 12 paired
development utterances. Strict checkpoint loading succeeded. Nine diagnostics
remain unscored for English accuracy because they lack expert English references.
Outputs include invented details (“I really like that” → “And, I like that from a
spoon”) and a brush sentence on the HELLO HOW YOU annotated clip. Webcam EAT gives
“I'm sorry, I'm sorry.” These results do not support replacement of Stage 2/Reel
or a claim of corrected repetition. They do not identify the dominant error cause
or disprove gloss-free translation generally. Updated REPORT.md and current state.
No production changes or training. Next safe action remains verified target-domain
hold/repeat examples before another model change. No new background job launched.

## 2026-09-16 — Uni-Sign worker exit

Detached Uni-Sign baseline completed; see `artifacts/reports/unisign_asl_baseline_20260916/REPORT.md`. Measured 12 paired development utterances and nine diagnostics: BLEU 8.21, chrF 38.06.
No training or production promotion. Next action: inspect outputs and limitations before deciding whether this challenger helps.

## 2026-09-16 — authorized pretrained Uni-Sign comparison prepared

User approved one inference-only gloss-free ASL comparison and asked to return with
results, retaining the earlier detached-work/no-polling instruction. Added the
report-local `unisign_asl_baseline_20260916/run_baseline.py`. Reuses existing download
and hashing helpers, official model/pose code at eed438bcb49e30405cd6ccdfcccca330c134e830,
and the released How2Sign checkpoint at eab251b7fe7e8521afc0e67be98add670ea40a0d.
Full strict state loading is required; mT5 is initialized from its configuration
before loading the complete checkpoint, avoiding an unused base-weight download.

The available local How2Sign subset is training data. Selected instead the pinned
validation mirror at 1231830fc1e8d77555a245ca22353b144e168111: first four eligible
source names in raw validation shard001, at most two sources per filename signer,
first three sorted sentence IDs of duration 2–12s each. Metadata/remote ZIP-directory
precheck confirms 12 utterances, four sources, signer IDs 1 and 2. This is a small
development slice, not signer-disjoint generalization. Raw clips will be cut at
manually realigned CSV boundaries; original sentence clips have different timing.
Nine prior difficult recordings have no expert English targets and stay qualitative.

Installed ONNX Runtime 1.19.2 and its missing logging dependencies in a separate
`data/local/tools/unisign_native_deps` target; main environment unchanged. Native
pose-only AST adapter excludes unused CUDA training imports without changing the
selected definitions. Self-check passes shape/direct-output equivalence, confidence
masking and empty inference targets; compilation and git diff --check pass.
No model-quality result yet. Final action is detached launch, with REPORT/results/
summary/provenance or FAILURE, completion record, history update and macOS exit
notification. No assistant polling, training, runtime replacement, or Citizen test
access. Next safe action after notification: inspect measured outputs and limitations.

## 2026-09-16 — replay result and decision; repeat errors survive beam decoding

Confirmed completion of all 40 runs over 10 recordings. Rollover correction changes
zero final outputs; corrected live/offline differ in three long-context runs.
Window-origin sensitivity affects both count and identity; original I I did not
reproduce, and saved-video re-extraction cannot recover original camera features.

Following the user's Stage-3 cleanup suggestion, ran one fixed existing beam8/topk12
control on the new saved emission matrices (no tuning/training). It changes 2/40
outputs, leaves all EAT/WHEN repeat examples unchanged, and increases labelled
diagnostic edit errors 32→34 across 32 reference tokens. These 16 labelled runs
are four correlated phases of four selected clips. Exact phase-zero WHEN sequence
log probabilities: single -6.460, double -0.00621; the repeated preference is in
the model emissions. Greedy outputs were checked against all saved report rows.

Added report-local `check_beam.py`, `beam_control.json`, `DECISION.md`; updated
current ground truth. No production, model, annotation or training changes.
Stage 3 may improve wording, but equal strings alone cannot establish intended
event count. Do not implement blanket repeat removal or promote beam decoding.
Next comparison chosen: one released pose-only Uni-Sign ASL checkpoint at utterance
level with native preprocessing and fixed paired development references, before
fine-tuning. That comparison is not launched; no new background job exists.

## 2026-09-16 — boundary correction and detached diagnostic replay prepared

User authorized the proposed diagnosis/correction sequence and requested ending
the turn after launching long work, with completion/failure notification and no
polling. Created branch `codex/stage2-held-sign-diagnostics` in the existing checkout,
preserving all pre-existing changes. No worktree/data/checkpoint duplication.

Added a regression demonstrating that a continuing raw CTC run is emitted again
at a rolled context start. It failed before implementation. `collapse_ctc_path`
now accepts the preceding raw token; both live callers retain that token from the
exiting window and clear boundary state on Finish/reset. 51 focused CTC/Reel tests
pass, including the actual Reel event-loop fixture spanning multiple rollovers
and preservation of blank/OTHER-separated repeats. This is a narrow runtime
correction, not proof of held-sign accuracy or a fix for early short-context errors.

Changed `scripts/live_stage2_ctc_v17.py`, `scripts/live_reel_stage1_v17.py`,
`test/test_live_stage2_ctc_v17.py`, `test/test_live_reel_continuous_v17.py`.
Opt-in `--ctc-trace-dir` records compressed emissions, frozen window features and
source observation times, with exclusive file creation and trace latency included.
Reel events include raw path, boundary token, context offset and emission file path.

Prepared report-local `run_replay.py`, `manifest.json`, timestamp sidecars and a
long webcam excerpt in `stage2_held_sign_diagnostics_20260916`. Frozen 10 video
hashes/readable streams/timestamp arrays verified; 40 runs compare legacy/corrected
collapse on identical logits and final single-context/stitched output. Four corpus
references are scored; partial O5S5/webcam remain diagnostics. This re-extracts
saved low-resolution video and cannot recreate original features or true live latency.

The worker records REPORT/results/summary/provenance plus completion JSON, or a
failure report, and sends one macOS notification on exit. It does not poll or
automatically reopen the chat. Launch is the final action of the turn; inspect
its exit notification/completion report in the next interaction. Training is not
launched: independent human hold/repetition labels and a justified objective are
still missing. Plan: `docs/superpowers/plans/2026-09-16-stage2-held-sign-diagnostics.md`.

## 2026-09-16 — held-sign diagnosis, video evidence, and gloss-free research

Completed the user's research request without another runtime patch or training run.
Read the saved `live_reel_stage1_v17/20260915_220852_978390` session and verified the
general selector SHA. It predates/does not load the repaired joint epoch-42 model.
Sampled-frame review finds one chest-point episode decoded as I I at two accepted
windows (token positions 6,10), before Finish or eight-window rollover. Two separate
salutation episodes produce HELLO HELLO and are retained as a repetition control.
Gesture descriptions are provisional, not expert ASL labels. Recorded Stage 3 also
changes I to “Is that?” and I I to “Is it?”.

Production CTC helper assertions verify held A A A collapses once and A blank A
twice. An independent synthetic counterexample proves prefix rollover can count a
single ongoing A run twice. This is not the cause of the early two-window duplicate.
76/130 session sequence updates are rejected; only accepted windows enter context.
No raw logits were saved, so the reason the model fragmented the early gesture
cannot be established retrospectively. Current continuous wrapper defaults differ;
the report describes both ordinary Finish review and revisable sequence mode.

Created `artifacts/reports/stage2_research_review_20260915/`: standalone nine-video
viewer, exact annotation/event rows and provenance JSON, report-local builder/checks,
contact sheets, seven full primary papers and selected official Uni-Sign source.
The viewer separates recorded live predictions, research continuous CTC predictions,
and pooled exact-core predictions. LG HELLO (0.46s) and WHEN (0.88s) are misclassified
as LESS/SIGN even with exact core boundaries. Existing partial/unverified annotations
are labelled honestly. Webcam frame indices map through saved source timestamps;
two trailing unmapped frames are excluded. No protected test clips were accessed.

Reviewed GFSLT-VLP, Sign2GPT, FLa-LLM, Uni-Sign, SHuBERT, simultaneous SLT and the
2026 sentence-finalization preprint. Screened SignLlama's abstract and the standardized
comparison's abstract/official code notes; comparison PDF requests timed out. Released
Uni-Sign pose-only ASL checkpoints are the closest structural challenger, but their
133-keypoint preprocessing/mT5 stack is not compatible with Apple v17 features.
Gloss-free requires video/English pairs and would replace Stages 2+3; it does not
automatically resolve online event counting or demonstrate iPhone readiness.

Decision: first freeze a signer-disjoint hold/true-repeat/rest/OTHER evaluation and
capture framewise logits/absolute time; isolate model emission errors from rollover
ownership. Retain one encoder/one temporal head as the simple recognition candidate;
evaluate a released gloss-free ASL model separately before considering replacement.
No promise that speed changes, bulk data, or a new architecture alone will solve it.
Verification: report builder passes its production-helper counterexamples, selected
source/model hashes, nine valid media exports and webcam frame-count checks. Gallery
links, embedded annotations, JavaScript syntax, Python syntax and research-checkpoint
hash pass. Contact sheets inspected; browser interaction and expert linguistic review
remain unperformed. Large-artifact index regenerated; git diff --check passes.
Verification details are saved in the report's `verification.json`.
Next safe action: deliver the annotated viewer and report for discussion.

## 2026-09-15 — repair handoff checks complete

Fresh final checks pass:33focused tests,23Python files compiled, report consistency
assertions, code whitespace andgitdiffcheck. Large-artifact index regenerated.
Completion audit maps the authorized training-collapse repair to measured full-training,
held-out, provenance and boundary evidence. Results remain a failed-promotion research
checkpoint; general continuous recognition and LG remain unsolved. All requested repair
experiments/results are complete; no additional training or deployment action remains.
Artifacts:joint_ctc_aligned_v17_20260915/{README.md,verification.json,handoff_checks.json,
completion_audit.json}. Next action:return the verified result candidly.

## 2026-09-15 — CTC repair verified; training collapse resolved, generalization still fails

Finalepoch42evaluated once on all334frozen development recordings after training-only
corrections. TRAIN1,088/1,291known matches,32.77%WER,1,901/1,901isolatedCTC;263/334
development recordings emit known signs. Connected WER98.24% (65S/90D/124I), familiar
107.72% (112S/74D/93I), contiguous50% (3S/6D/3I). ActualCitizen/SemLex validationCTC
339/378 (89.68%)/762/978 (77.91%); pooled95.77%/85.28% retained. LG pooled15/57 but
actualCTC4/57with42empty; trainingcores199/199bothoutputs. Generalization remains weak.

Verified4,520input hashes,54checkpoints, all frozen recipes/full schedules/optimizer
states, recomputed finaltraining and developmentcounts/gates. CPU/MPS628/628tokens
agree,63tokenprefix consistent.33focused tests pass. Source-clock latency misses
158/296signs; runtime excluded. Failedpromotion gates:connecteddeletions,familiarWER,
medianruntime delay,runtimeinclusion. No Citizen test, confirmation, export or promotion.
Finalreport:joint_ctc_aligned_v17_20260915/README.md; canonicalcurrentstate updated.
Decision: training-collapse repair complete, no more tuning from these held-outerrors.
Next product work must address signer/domain generalization and sequenceprecision;
this checkpoint is not a deployment candidate. Final handoff: compile/whitespace,
report consistency and large-artifact index, then return verified results.

## 2026-09-15 — final alignment completed; held-out evaluation started

All six fixed epochs37–42completed. Final TRAIN known matches1,088/1,291 (84.28%),
169deletions,34substitutions,220insertions,32.77%WER; isolated CTC1,901/1,901 (100%).
Compared with the original failed joint2/1,291known matches, the CTC training collapse
is substantially repaired; training sequence accuracy is still imperfect. Final model
has4,620updates across42epochs; the rejected first12epochpositive-only run is separate.
Launched the predefined single finalepoch42evaluation on the frozen development suite.
No held-out outputs informed loss corrections or checkpoint selection. Next: inspect
all334recordings/isolated/LG results, independently verify provenance and metrics,
record limitations and report the completed repair result without promoting failures.

## 2026-09-15 — verified-frame phase completed; final alignment launched

All12epochs25–36completed. Final TRAIN:1,263/1,291known matches (97.83%),3deletions,
25substitutions,1,370insertions,108.29%WER; isolatedCTC1,897/1,901 (99.79%). The
sign evidence is learnable, but extra outputs still dominate. Launched the predefined
six-epoch37–42alignment continuation with original target-normalized CTC restored,
retaining all verified supervision and optimizer state. Loss routing verified directly;
no held-out evaluation has informed any correction. Next: complete final alignment,
evaluate fixedepoch42once, verify provenance/metrics and report measured limitations.

## 2026-09-15 — final alignment curriculum prepared from training-only evidence

Verified-frame epoch28 recognizes1,186/1,291known training signs, but2,387insertions
remain. A controlled original-CTC+CE optimization stays blank from a low-logit start
(p=0.062706) but preserves a strong nonblank solution from the warm start(p=0.999520).
This supports temporary frame normalization to escape collapse, followed by restoring
original sequence alignment strength once sign evidence has been learned.

Prepared a fixed six-epoch37–42continuation after currentepoch36, preserving all
verified-frame/anchor/pooled/background objectives, model/optimizer/learning rates/data.
Only original target-normalized complete/positive CTC functions are restored. No
held-out outputs or decoder changes inform the choice. Minimal runner/plan:
`joint_ctc_aligned_v17_20260915/`; it reuses the verified-frame training loop. Next:
complete current run, execute the final alignment pass, then evaluate fixedepoch42
once and verify all results. No further tuning from held-out errors is planned.

## 2026-09-15 — balanced continuation completed; verified frame refinement audited

All12continuation epochs13–24 completed. Final TRAIN:1,117/1,291matches (86.52%),
25deletions,149substitutions,2,336insertions and194.42%WER. Isolated CTC1,891/1,901
(99.47%). Blank suppression is removed, but sequence precision still fails.
No held-out outputs were inspected. The next dense-frame audit admits150,876training
tokens:23,454known,103,251OTHER,24,171guarded blank;86,112tokens remain ignored.
The exact0.10s interior-gap guards are reused; clip edges/overlap/unguarded contexts
remain unsupervised. Independent reviewer hit a usage limit before this review;
root directly checked helper/loop routing, global frame population scaling, resume,
freeze/evaluation and copied imports. No blocking issue found. Next: focused checks,
then fixed25–36refinement followed by final held-out evaluation and full verification.

## 2026-09-15 — emission locations expose missing continuous-context supervision

Epoch16 TRAIN location probe samples32recordings per source. ASLLRP OTHER produces
129known emissions for39references:35inside known intervals,52inside OTHER intervals,
42in gaps. These are timestamp-location diagnostics, not event-alignment accuracy.
The current recipe teaches one sequence point per sign and blank on standalone
background windows; it leaves most OTHER motion and short guarded gaps without direct
sequence-frame supervision. Existing complete annotations can supply those labels.

Prepared `joint_ctc_frame_supervision_v17.py` and two regression tests: known/OTHER
intervals,0.10s guarded interior gaps, incomplete-annotation refusal, overlap/padding
masking, gradients and partition-invariant population normalization. Both pass.
The new recipe adds only this sequence CE after the current fixed epoch24 completes,
retaining prior objectives/optimizer for25–36. Runner/plan:
`joint_ctc_frames_v17_20260915/`. No new data, unverified blanks or held-out tuning.
Next: audit frame coverage/review; complete current run, then execute fixed refinement.

## 2026-09-15 — first frame-normalized epoch confirms blank suppression mechanism

After one unchanged-data continuation epoch, known TRAIN matches rise330→879/1,291,
deletions fall817→65, and blank steps fall235,498→50,515/236,988. This directly
supports the loss-scale diagnosis on real training data, beyond the toy counterexample.
However insertions rise65→2,608 and TRAIN WER is233.93%; emission recovery alone is
not accurate recognition. Isolated CTC remains1,830/1,901. The fixed epoch13–24 run
continues; no held-out outputs or decoder tuning informed this result. Next: judge
final precision/deletions jointly, then evaluate the fixed candidate and verify.

## 2026-09-15 — per-frame CTC correction reviewed and launched

Reviewer cleared nonrecursive loss replacement, complete/positive normalization,
epoch12 optimizer continuation and provenance/evaluation routing. All31focused tests
pass. Launched fixed epochs13–24 from the completed anchor checkpoint, changing only
CTC NLL normalization from targets to valid output frames. All prior objectives,
learning rates and full coverage remain. Next: verify full-training recognition,
then evaluate the fixed final candidate and report measured limits.

## 2026-09-15 — anchor run completed; loss-scale counterexample identifies remaining blank basin

All12anchor epochs completed. Final TRAIN:330/1,291known matches,817deletions,
65insertions,79.47%WER,1,858/1,901isolated CTC correct (97.74%). No development
results were inspected. A64-training-sequence epoch10 probe finds blank at87/106
known anchors; removing blank would identify62/106. Both competition and label
confusion remain, so blank suppression would not solve recognition.

A controlled32frame single-target optimization reproduces a concrete scale failure:
target-normalized CTC plus positive CE settles at p(target)=0.062706 with blank greedy
output; frame-normalized CTC plus the same CE reaches0.995391/nonblank. New shared
`joint_ctc_loss_balance_v17.py` reuses existing CTC validation/autograd and normalizes
by valid input length. Two red/green tests verify the opposing gradients and padding/
length scaling;17focused CTC tests pass. A bounded continuation preserves anchor12
model/optimizer and changes only CTC normalization for epochs13–24. Plan/runner:
`joint_ctc_balanced_v17_20260915/`. Next: review, train, evaluate and verify full results.

## 2026-09-15 — verified anchors produce earlier nonblank fitting

Corrected anchor epoch4 recovers53/1,291known continuous training signs and correctly
decodes1,259/1,901isolated training clips, versus0/1,291 and9/1,901 at epoch4 of the
preceding positive-CTC-only repair. This is an intermediate training-only result;
it is not yet adequate full-data fitting or evidence of held-signer generalization.
The original fixed12-epoch run continues without recipe/decoder changes. Next:
finish the run and evaluate the final candidate before claiming a repair.

## 2026-09-15 — corrected anchor objective reviewed and launched

Reviewer cleared fixed population weighting and its epoch/source scaling; all29
focused tests pass. The corrected fresh seed17111 run is launched for12epochs with
full-training recognition measured each epoch. All1,291known training targets have
anchors. No held-out evaluation or decoder adjustment has occurred. Next: finish
training, evaluate the final candidate, then independently verify checkpoints,
full-data coverage, greedy CTC retention and reported development metrics.

## 2026-09-15 — review corrected epoch-level known/OTHER anchor weights

Review found minibatch means would overweight OTHER-only batches. Stopped the new
run before any checkpoint; preserved its recipe, freeze and log under
`joint_ctc_anchor_v17_20260915/interrupted_batch_weighting/`. The repaired objective
uses fixed full-training per-source known/OTHER anchor populations and sum reductions,
so batch partitioning cannot change total weighting. New partition-invariance test
passes alongside both anchor safety/gradient checks. Anchor audit covers 7,379/7,380
training events, including all 1,291 known targets; one ambiguous OTHER is skipped.
Manifest timing SHA is verified against the original preparation audit. Next: final
bounded review and run corrected frozen recipe; no held-out outputs inspected.

## 2026-09-15 — first positive repair completed; verified-anchor correction prepared

All 12 positive-CTC epochs completed with complete coverage. Final isolated TRAIN CTC
is 1,780/1,901 (93.63%), but continuous TRAIN known recall is 183/1,291 (14.18%),
974 deletions and 87.84% WER. No held-out evaluation was used to choose the correction.
The missing positive route is repaired; sequence alignment still fits poorly.
New `joint_ctc_anchors_v17.py` uses one unambiguous token within each verified event,
and event-balanced CE directly teaches the actual CTC head. Repeated targets remain
separate; overlapping/unsampled intervals receive no invented target. Two new red/green
regression tests pass. The next bounded recipe adds these anchors and positive midpoint
CE while retaining prior objectives, data, architecture, original initialization and
12-epoch schedule. Report/plan: `joint_ctc_anchor_v17_20260915/`. Next: inspect anchor
coverage and review, run the frozen training recipe, then evaluate and verify results.

## 2026-09-14 — train-only probability diagnosis separates CTC marginal and greedy collapse

First direct-positive repair is still training; epoch4 recovers0/1,291known continuous
training signs and9/1,901isolated train clips. A16-training-clip epoch3 probe measures
mean blank0.9660/target0.02745; correct transcript marginal beats empty8/16, but greedy
is empty16/16. A controlled stationary32-frame CTC calculation has a blank-dominated
local basin near target probability0.03 (loss0.95464); uniform increase to0.10 raises
loss1.98916. These are training-only diagnostics, not held-out selection or a decoder
change. Evidence in joint_ctc_repair_v17_20260914/DIAGNOSIS.md and corresponding JSONs.
Next action: complete fixed first repair and judge full training recognition, not loss.

## 2026-09-14 — positive-CTC repair reviewed and training launched

Independent review clears training loss routing, pooled-forward equivalence and
optimizer recovery. Added the requested pre-evaluation recipe/checkpoint identity
validation before freezing the run. Corrected joint training is now running for
12epochs from the original seed17111 initialization; data, architecture, learning
rates and full-coverage schedules remain unchanged. New direct replay/core CTC losses
are the sole objective correction. Full-training known-sign recovery and all1,901
isolated CTC training outputs are evaluated every epoch; held-out results remain
uninspected until this bounded training run completes. Next action: inspect training
fitting, then evaluate or continue training-only diagnosis if collapse persists.

## 2026-09-14 — failed-checkpoint probe confirms missing positive CTC gradient route

On the final failed joint checkpoint and16training replay clips, pooled CE has no
CTC-head gradient. Direct one-token CTC gives blank-bias gradient+0.054630 (gradient
descent lowers blank), while verified-background CE gives-0.017992 (raises blank).
Training transcripts contain1,291known and6,089OTHER targets across236,988tokens.
This establishes a missing direct positive route, not that it is the sole cause.

Added `active/v17/joint_ctc_supervision_v17.py`, two red/green regression tests and
report-local `joint_ctc_repair_v17_20260914/run_repair.py`. Twelve focused CTC tests
pass. Repair retains original losses and adds single-token CTC on replay and exact
cores, sharing encoding with pooled CE. Checkpoints now retain optimizer state and
recover an interrupted summary write; previous frozen files are untouched. Next:
independent bounded review, then fixed12epochs with full training recognition measured.

## 2026-09-14 — user authorized repairing CTC blank collapse

Code trace identifies a candidate missing route: pooled replay/core losses update
Stage1 only, whereas verified-gap CE directly teaches the CTC head blank. Measure its
gradient consequences before changing the objective. New bounded plan:
`artifacts/reports/joint_ctc_repair_v17_20260914/PLAN.md`; reuse frozen data, architecture,
seed/schedules/learning rates, add direct single-token CTC on isolated/core positives,
and require full-training recognition diagnostics. No held-out target changes,
decoder bias tuning or previous artifact mutation. Next action: gradient probe/tests.

## 2026-09-14 — both joint-CTC arms complete; blank collapse fails every epoch

Completed12frozen+12joint seed17111 epochs,1,320updates each,8,016 evaluations across
the same334 recordings. Every epoch yields no known development signs: connected100%
WER/284deletions, familiar100%WER/259deletions, contiguous24/24deletions. Zero verified
gap emissions is silence, not useful rejection. Final joint pooled isolated accuracy
95.50%Citizen/85.99%SemLex retains the baseline95.24%/85.28%. O5S5 raw-core train
accuracy rises74/199→186/199 (37.19%→93.47%); LG stays12/57 (21.05%).

Post-run final-checkpoint inference on every complete ASLLRP training sequence finds
frozen0/1,291 and joint2/1,291known signs recovered. Joint236,988training token decisions
are236,974blank,10OTHER,4known (two collapsed signs). Both final models choose blank
on every ASLLRP validation token. Separate single-clip CTC diagnostic scores frozen
4/378Citizen,18/978SemLex; joint0/378,0/978. Pooled retention does not imply CTC retention.

Verified all4,520input hashes,24checkpoint hashes, unchanged frozen/changed joint
encoders, identical starting head/encoder and per-epoch schedules/coverage. Each epoch
covers923complete sequences,199cores,1,901replay,494background. Recomputed recording edit
counts/gates;24focused tests pass. Final CPU/MPS628/628tokens agree per arm; max logit
error4.77e-6. Successful training724.62s excludes preparation/evaluation and the
preserved8-epoch Metal-aborted attempt. Complete retry used unchanged original recipe.

No confirmation, runtime/device gate, export or promotion ran. All296temporally scored
signs missed; cached CPU timings exclude extraction/scheduling and are not live latency.
Decision: next model work must diagnose training CTC blank collapse before another
generalization comparison; more data alone is not a measured fix. Signer/variant/phone
coverage remains necessary separately. Do not scale the same failed recipe or tune
blank suppression from LG. Result/report scripts and verification are in
`artifacts/reports/joint_ctc_v17_20260914/`; PROJECT_GROUND_TRUTH.md updated. Next safe
action: return the completed negative result. Final handoff checks passed:24tests,
compilation, tracked/new-file whitespace, report assertions and large-artifact index.
Binding evidence promoted to live-streaming/high.md; no required experiment work remains.

## 2026-09-14 — Metal runtime aborted frozen arm after epoch8; recipe unchanged

Authoritative process inspection shows no surviving training process. The frozen log
ends with Apple M4 Metal command-buffer Internal Error(00000001), after eight complete
epoch checkpoints/evaluations; joint has not launched. All eight froze isolated accuracy
and emitted no known connected signs (100%WER,284deletions), failing gates. This is not
a completed experiment. Preserve partial files as frozen_interrupted and restart the
same arm from original initialization because checkpoints lack optimizer state. No
hyperparameter or data change is permitted from the observed development results.
Next action: MPS smoke check, caffeinate-protected same-recipe retry, then joint arm.

## 2026-09-14 — final preflight passed; matched CTC training launched

Final preparation retains all923train/237validation complete ASLLRP sequences,
199train/57LG cores (four static one-observation cores), 1,901/1,356 isolated replay,
494/16 verified backgrounds and all334 development recordings. All4,520 inputs match
their frozen hashes. Final9,239 chunks have max duration0.5005s, max sequence528tokens.
Train-only preflight again reaches4/4 exact at70updates/loss0.04293; encoder gradient
L1>0, frozen gradients absent, initialized CPU/MPS token predictions agree256/256.
Independent reviewer reports no remaining blocking issue;24focused tests and diff
check pass. Recipe and dependency hashes are sealed in training_freeze.json.

Frozen then joint12-epoch arms launched sequentially; no inference-mode encoder cache
used for sequence training. Current experiments remain research-only and no checkpoint
is eligible until all gates pass. Raw-core original-base probe gives74/199train and
12/57LG versus contextual-core52/199 and16/57 on one result per original event.
These differ from previous overlapping-window metrics and are not interchangeable.
Next action: collect all24 epoch evaluations and compare matched coverage, retention,
deletions and event-level LG; do not alter recipe after observing development results.

## 2026-09-14 — reviewed chunk-duration correction before full training

Independent review found wall-grid chunk boundaries could exceed0.53s when observation
times undershoot successive boundaries. Reproduced with a failing30-frame10Hz test,
then anchored each next endpoint to the preceding observed endpoint+0.53s. Ten focused
CTC tests now pass, including full-sequence/available-prefix equality and timing bound.
Prepared caches before this correction are retained as data_initial195.pt and
data_initial_grid.pt; final cache is being regenerated and will report maximum duration.
The corrected199-core preflight had already repeated4/4 tiny-fit success at step70;
it will rerun after timing correction. CPU/MPS initialized-token predictions agree256/256,
max logit difference2.86e-6. All16 validation background feature arrays exactly match
the frozen baseline gate. No full arm has started; next action final freeze/preflight.

## 2026-09-14 — joint CTC preflight fits train-only sequences; one-frame cores retained

Added `active/v17/joint_ctc_v17.py`, four focused contract tests, and report-local
`joint_ctc_v17_20260914/run_experiment.py`. Eight new/existing CTC tests pass after
observed red tests for the new module and single-observation core helper. First cache
contains 923/237 train/validation complete sequences, 1,901/1,356 replay clips,
494/16 verified backgrounds and all 334 development recordings. Sequence encoding
has 8,875 chunks, max 528 owned tokens; no sequence rejection occurred.

Train-only preflight: four distinct transcripts fit exactly by update70, CTC loss
0.04293; encoder gradient L1 162621.20, frozen encoder gradients absent. Initial
preparation excluded four one-observation O5S5 cores; corrected to repeat their observed
static pose, with no invented trajectory, so all199 events can participate. Zero-frame
cores still reject. Initial195 cache/reports retained under renamed paths; corrected
preparation and preflight will rerun before full training. Both arms explicitly use
eval-mode encoders so gradient enablement, not encoder dropout, differs. Independent
review underway. Next action: validate corrected preparation, then matched12-epoch arms.

## 2026-09-14 — frozen-versus-joint experiment authorized and recipe fixed

User explicitly requested execution through measured results. Plan is frozen before
training at `artifacts/reports/joint_ctc_v17_20260914/PLAN.md`: existing causal CTC
blocks, differentiable frame encoding, timestamped <=0.53s chunks, complete ASLLRP
sequence CTC, positive-only O5S5 cores, isolated retention and verified-gap blank loss.
Compare frozen and joint arms for 12 seed-17111 epochs with matched coverage/updates,
after gradient/alignment/tiny-fit preflight. Reuse existing development gates and
frozen replay identities; no official Citizen test, default changes or acquisition.
Free disk at start: approximately 12 GiB. Next action: focused tests and implementation.

## 2026-09-14 — independent next-step review qualifies the CTC recommendation

Fresh 14-test run passes. Streamed saved CSV reductions reproduce 8,978 windows,
positive-window exposure deficits, 29.0891% long-window O5S5 foreground, 199/57
O5S5 train/LG events, and 23/49 one-signer classes; combined manifest hash matches.
Stored core metrics were checked, not rerun inference. Initial verification assumptions
about globally unique IDs and one model_summaries row per model failed: the latter is
a grouped metric list; isolated has 15 model entries. Background IDs collide in 52
groups/57 extra rows because per-gap ordinal resets; features differ, targets agree,
and positive context IDs are unique. Include gap/timestamp in future coverage identities.

Code inspection qualifies the prior recommendation: core pooling follows full-window
noncausal encoding, so it is not a raw-core oracle. The encoder has unmasked attention,
symmetric convolution and a 32-position table. Causal head alone does not establish
causal streaming. Existing UnifiedStreamingCTCHeadV17 already provides shallow dilated
blocks; its trainer caches inference-mode features. Existing window training also
already has foreground CE and replay KL. CTC needs 102 outputs, including OTHER.

Decision: next model experiment should first pass gradient/alignment/timing and tiny-fit
preflights, then compare reuse of that head with frozen versus joint encoder training
under matched corrected supervision/coverage. Defer multi-level/shared-head additions.
Fresh portrait-iPhone signer groups and targeted per-class coverage remain necessary.
Primary-source research supports shallow temporal/visual auxiliary learning, but the
2025 pose paper tests unseen sentences, not this project's ASL signer/iPhone setting.
CVF direct fetches failed 403; official indexed paper text supplied reviewed details.

Changed report: `artifacts/reports/stage2_data_learnability_audit_v17/NEXT_STEP_REVIEW.md`;
current-state frontier/actions clarified in PROJECT_GROUND_TRUTH.md. No pipeline code,
checkpoint, runtime, dataset or external message changed; Citizen test stayed sealed.
Next safe action is the bounded preflight/comparison described in the report, not a
new architecture sweep. Validation: large-artifact index regenerated (23,525 bytes),
review/current-state/history reference assertions pass, and git diff --check passes.

## 2026-09-14 — exhaustive supervision audit separates window dilution from LG generalization

Audited every 8,978 continuous window and all 3,257 isolated replay clips against the
original model, all 12 O5S5 checkpoints and prior no-O5S5 comparators. Data extraction
is globally healthy (98.68% O5S5-train and 100% LG any-hand frame coverage), exact
feature conflicts are absent, and O5S5 is not uniquely fast: median target duration is
0.259s versus 0.267s for ASLLRP OTHER; LG is slower at 0.340s but remains weak.

The training contract is the measured bottleneck. Across 12 epochs the source-balanced
sampler never selected 3,516/5,515 ASLLRP OTHER windows or 154/1,005 O5S5 windows.
O5S5 targets occupy only 29.1% of a 1.07s window on average. Exact-core pooling raises
epoch-12 O5S5-train accuracy from 54.03% known-window to 61.09%, but LG remains
27.67%, so segmentation is only part of the failure. O5S5 per-class signer support is
thin: 23/49 train classes have one signer, and three LG classes are absent from O5S5
train. The underlying isolated classifier remains 87.68% on its full validation replay
at epoch 12, confirming that continuous recognition is the main failure while Citizen
retention still regresses.

Decision: do not repeat pooled 1.07s target classification or call unverified O5S5
surrounding signing blank. Next candidate is a shared frame encoder with isolated
auxiliary CE and a shallow temporal CTC head, deterministic full-pool coverage and
exact-core O5S5 supervision. No model, runtime or default changed; Citizen test was not
accessed. Reports: `artifacts/reports/stage2_data_learnability_audit_v17/`.

## 2026-09-14 — O5S5 final verification passed; negative result sealed

`o5s5_augmented_v17_20260914/verification.json` records 14 passing focused tests,
clean `git diff --check`, identical recomputed selection, unchanged original manifest,
4,520 verified frozen inputs and 12 checkpoint hashes. All 4,008 recording evaluations
(86,544 scheduled predictions) and held-out LG checks completed. The first-run gates
failed; confirmation was not launched. Report and CSV comparisons are complete.
Next safe action remains retention/deletion analysis under a separately bounded scope;
no further run is queued by this experiment.

## 2026-09-14 — O5S5 augmentation completed; no eligible candidate

Frozen manifest SHA256 `6b506552f7d35014e539e5df9f5e9d8d40a8e227544974ddb407f8d52cae4178`.
Fresh original-base seed17111, paired with the prior data-only comparison, completed
all12epochs/564updates in108.43s. Added1,005 O5S5 context windows while retaining
1,901 isolated training replay clips and494 verified ASLLRP background windows.
All12checkpoints evaluated on334 development recordings/7,212windows each, plus318
LG positive windows. No Citizen test access, acquisition or MediaPipe conversion.

Best connected checkpoint epoch3:123.5915% WER,113S/42D/196I over284 signs, versus
frozen CTC168.3099%,144S/11D/323I. WER improves26.57% relative, but deletions fail.
Familiar WER111.9691% versus56.3707%; Citizen93.3862% versus95.2381% own start;
SemLex84.2536% versus85.2761%. Both isolated drops exceed1point. Verified gap
emissions improve15→6/16. Prior no-O5S5 best was146.1268% WER/33D/258I.
Only epoch1 preserves both isolated accuracies, but fails connected311.6197% WER,
768 insertions, familiar128.5714%, and16/16gap emissions. All epochs fail measured
gates. Runtime-inclusive latency remains unverified; raw/phone replay was not run
after accuracy rejection. No confirmation17112, export, promotion or default changes.

LG positive-window accuracy: original base76/318=23.90%, prior epoch1 73/318=22.96%,
new epoch1 82/318=25.79%, new best-connected epoch3 67/318=21.07%. No full-narrative
LG WER because annotations are incomplete; windows overlap and are not independent.
CPU/MPS labels agree636/636 across LG epochs1/3. Fourteen focused tests passed.
All4,520 frozen raw/replay hashes,12checkpoint hashes and exact isolated replay
identities reverified; encoder/classifier updates confirmed. Initial sandbox MPS
launch failed before optimizer updates; authorized MPS run succeeded.

Artifacts: `artifacts/reports/o5s5_augmented_v17_20260914/` contains README, frozen
manifest, development freeze, all-epoch evaluations, LG predictions, selection,
CSV comparisons/error examples, parity and provenance. Checkpoints:
`artifacts/models/stage1_window_o5s5_v17_seed17111/`. Changed only report-local
evaluation orchestration, reports/index and current/history documentation.
Next safe action: retain rejected artifacts and accepted defaults. Do not run
confirmation merely on aggregate WER improvement; the retention/deletion gate failed.

## 2026-09-14 — all12 O5S5 epochs completed; streaming evaluation underway

Fresh original-base seed17111 completed all12epochs in108.43s, saving every checkpoint
under `artifacts/models/stage1_window_o5s5_v17_seed17111/`. Training audit confirms
6,819 contextual positives and494 verified ASLLRP background windows, with no O5S5
background. Epoch1 reaches Citizen94.9735%/SemLex85.4806%; epochs2–12 fail at least
Citizen's94.2381% retention floor. First LG comparison: original base76/318 (23.90%),
prior no-O5S5 epoch1 73/318 (22.96%), new epoch1 82/318 (25.79%). These are correlated
positive-window diagnostics, not full-narrative WER. Full334-recording evaluation is
running for all12epochs. Next: freeze complete error/gate results; do not confirm
merely on LG window gains. Citizen test remains untouched.

## 2026-09-14 — O5S5 training launched on MPS after sandbox failure

Initial sandbox attempt failed at `model.to(mps)` before optimizer updates or checkpoint
creation. Original failure log retained. Authorized escalated command started normally;
first five epochs show isolated-retention regression after epoch1. Added report-local
`evaluate_experiment.py` reusing the frozen full-stream evaluator and MPS adapter, with
assertions for all334 recordings/284 connected tokens and318 LG positives. No production
code changed. A compile check hit the system Python cache sandbox restriction and was
rerun with the existing approved compile permission. Next: finish all12epochs and
evaluate every checkpoint; no gate/threshold changes.

## 2026-09-14 — bounded O5S5 augmentation frozen before training

User authorized one fresh development run and conditional confirmation, without further
acquisition or Citizen test access. Frozen byte-identical combined supervision under
`artifacts/reports/o5s5_augmented_v17_20260914/`; original manifest unchanged.
Pinned all 4,514 prior raw/replay inputs plus six O5S5 archives. Prior baseline artifacts
are unchanged; trainer differs only by previously implemented positive-only loading.
Use fresh original-base initialization and paired seed17111, unchanged 12-epoch
50/30/20 recipe. O5S5 adds 1,005 train positives; LG's 318 positives are validation-only.
Incomplete LG annotation prohibits full-narrative WER; report positive-window accuracy.
Fourteen focused training/selection/O5S5 tests passed. Next: train all12 epochs, evaluate
334 existing recordings plus LG, then apply unchanged gates before any confirmation.

## 2026-09-12 — Stage-1 window final handoff verification passed

Final verification.json records 61 passing focused tests, successful compilation and
clean git diff --check; all frozen inputs/artifacts and 12 checkpoint hashes reverified.
All 12 complete evaluations and 16 actual replays are finished; no task processes remain.
The final comparison report includes all epochs, exact failure gates, timestamped
examples, provenance, timing/misses, missing coverage and a runnable experimental Mac
command. Defaults and accepted artifacts are unchanged. Outcome: completed bounded
experiment, no eligible model. No additional training or export follows this handoff.

## 2026-09-12 — bounded Stage-1 window experiment completed without promotion

All12 seed17111 checkpoints have complete334-recording evaluations. Lowest connected
WER is146.1268% at epoch4 (258 insertions,33 deletions), versus168.3099% CTC (323/11).
Epoch4 fails deletions, familiar WER117.3745%, Citizen93.3862%, SemLex83.6401%.
Epoch1 alone passes isolated retention but fails connected WER/insertions, familiar
WER and transition emissions. Selection.json records no eligible candidate; full-pool
latency remains unverified, independently of these measured failures. No seed17112,
Core ML export, protected-test access or default promotion occurred.

Report README.md, all-epoch evaluation JSON, timestamped_candidate_examples.csv,
checkpoint_provenance.json and runtime summaries are under
artifacts/reports/stage1_window_v17. Runtime measured15/15 raw/cache transcript agreement
and7/15 recognized signs on the verified timing subset; missing coverage and zero
observed spontaneous online corrections are explicit. All16 replay processes completed,
including partial tails and the112-second HUNGRY source. Its Finish pass took46.25s
under contended CPU execution; this is not mobile/realtime readiness.

Final independent gate review found that runtime evidence was not bound to the frozen
annotation pool. Added a regression that reproduced false eligibility, then fixed
selector checks to require matching manifest SHA and exact annotated identities.
Re-review closed the issue. New selector/test/report-side analyzer changes do not alter
any training-frozen code or checkpoint. All12 checkpoint hashes bind the unchanged
freeze, and encoder/classifier changes were verified. The first review agent could not
run because its model was unavailable; a supported routine reviewer completed the check.
Next safe action: inspect the rejected research artifacts or obtain fresh independent
annotations under a new explicit scope; do not silently restart training or promote.

## 2026-09-12 — paced runtime verified; CPU/MPS parity enables remaining evaluation

Epoch1 actual runtime completed 16 recordings (12 exact ASLLRP, three connected,
diagnostic HUNGRY). All 15 comparable final transcripts exactly match cached results.
Verified alignment is available for nine clips/15 signs: seven recognized, eight
missed (53.33%); median first-correct delay 0.304s, p95 0.681s including runtime and
Finish. This small subset does not satisfy full-pool latency verification. Per-window
median/p95 is 58.14/106.47ms during concurrent CPU evaluation. HUNGRY took153.92s
capture processing plus46.25s Finish;453/453 no-hand-detected windows emitted known
labels. No whole-session WER is claimed. Report artifacts include runtime_summary,
hungry_candidate_diagnostic and timestamped_candidate_examples.csv.

CPU evaluation epochs1–4 completed. To avoid repeated slow CPU evaluation, the remaining
epochs use the same frozen evaluator with only model/input device transfer to MPS.
Epoch1 full parity verified all7,212 window labels,334 final transcripts and16 transition
labels before this change in execution device. CPU/MPS probabilities need not be bitwise
equal. No checkpoint, feature, decoder or training change was made. Scripts in the report
folder reproduce this evaluation and timing summary. Next: finish all12, apply selector,
record final gates and verify code/provenance.

## 2026-09-12 — first complete Stage-1 window evaluation fails streaming gates

Epoch 1: connected 316.20% WER (125 substitutions, 3 deletions, 770 insertions),
familiar 129.34%, exact subset 54.17%; transition false emissions 16/16. CTC connected
baseline is 168.31% WER (144/11/323), older-Reel familiar 56.37%, transition 15/16.
Epoch 1 alone retains both isolated accuracies within 1 point; epochs 2–12 already
fail isolated retention. No epoch can therefore qualify, even before full latency
measurement. All 12 complete-stream evaluations remain required for the report.

The serial evaluator was stopped during epoch 2 after preserving completed epoch 1;
remaining epochs use three independent CPU processes, unchanged code/data/checkpoints.
This is evaluation scheduling only, not retraining. Paced runtime epoch 1 covers all
12 exact-development recordings, three connected recordings selected by sorted identity,
and diagnostic-only HUNGRY. Startup-inclusive delay and CPU contention will be disclosed.
A final test invocation initially named a nonexistent test module; corrected invocation
passed all 60 tests. Added selector check covers missing epochs/runtime evidence and ties.
Next: finish evaluations/replay, freeze failure report and retain defaults.

## 2026-09-12 — Stage-1 window baselines frozen and bounded seed completed

Older-Reel paced familiar replay completed all 97 recordings: 56.370656% WER
(127 deletions, 18 substitutions, 1 insertion / 259 reference words). The matched
CTC transition check emitted on 15/16 checked background windows. Development freeze
pins 334 stream identities, 4,514 raw/replay inputs, model hashes, code and decoding.
Seed 17111 completed all 12 authorized epochs in 111.22 seconds; all checkpoints are
saved separately under artifacts/models/stage1_window_v17_seed17111. No protected
test was accessed. Sixty focused tests passed before launch. Complete-stream evaluation
of every epoch is running; no candidate is selected and no confirmation seed or export
is authorized yet. Next: apply every frozen gate, measure actual runtime replay,
and report failure if none qualifies. The immutable freeze records prelaunch state.

## 2026-09-11 — training gates and complete-stream evaluator added

Implemented scripts/evaluate_stage1_window_v17.py for all frozen raw development
windows, Finish tails, source-time latency with explicit misses, and fail-closed
promotion gates. Cached/batched latency is explicitly ineligible as runtime latency.
Added preparation of a common16-window guarded-background comparator; this evaluates
both classifiers on identical support rather than comparing incompatible output rates.
Training now refuses unfinished/changed baseline freezes, uses exact1,500replay/
900context/600background samples per epoch, and limits frozen-teacher consistency to
known replay. All12epochs remain separately saved and unselected.

Corrected Finish decoding across timestamp gaps so identical predictions on separate
capture runs do not collapse together. Recent prediction corrections are restricted
to the last2s, and frame-mapping/exclusion validation rejects malformed input.
60focused tests and git diff --check passed before the last metadata/comment-only
edits. Existing defaults unchanged. Matched older-Reel familiar replay continues with
all completed processes successful; no candidate training launched yet.
Next: finish97baseline replays, freeze raw validation/transition inputs and baseline
results, train seed17111 once, then evaluate every epoch and report actual failed gates.

## 2026-09-11 — raw preparation passed; experimental runtime smoke verified

Materialized1,160 checked all-sign videos through the original live observation
contract; raw feature archives preserve actual sampled timestamps and per-source
hash/observer provenance. Preparation:train923 clips,5,814 contextual windows at
.27/.53/1.07s (2,528 on exact.53/.13 schedule),494 guarded background windows;
validation237 clips,1,331context/16background. Source signers remain disjoint3/1.
Coverage report finds173short known events,14rapid known pairs,1,560OOVevents;
zero identical-sign repetitions and zero known signs longer than1.07s in validation.
No independent portrait-phone/hold/repetition claims are supported.

Shared window conversion now normalizes only within the requested time slice,
resamples continuous channels by timestamps, uses binary nearest-neighbor presence,
and zeroes missing nodes. Runtime receives raw frames, uses only landmark classifier,
retains compact arrays for Finish, and handles partial tails. An untrained emission
head smoke produced2provisional windows and3Finish windows including the final partial;
this proves integration only, not accuracy. Unit checks cover explicit CLI/defaults,
timestamp invalidity/gaps, repeated outputs, correction of a recent prediction tail,
missing-node zeroing, and no future normalization leakage. Training sampler now uses
exact1,500/900/600 per3,000 draws with source/class balancing.

Frozen development identities334 (225connected,97familiar,12exact); older Reel paced
familiar baseline is running. Existing revisable CTC cached baseline and raw parity
reports are pinned for reuse. No candidate trained yet; defaults/checkpoints unchanged.
Next: complete matched baseline freeze, run seed17111 once for12epochs, evaluate all
complete streams and retention gates, then conditionally confirm seed17112.

## 2026-09-11 — matched HUNGRY diagnostic and checked-window audit

Four fixed period-origin diagnostic comparisons completed; evidence/hashes/settings:
artifacts/reports/stage1_window_v17/hungry/{diagnostic_freeze,diagnostics,summary}.json.
Shared frames:1,392 decoded,285 retained at source90–111s,640x360; damaged final
frame excluded. Timestamp sidecar actually has1,393 entries (earlier recovery prose
said1,394); no verified transcript or sign-boundary truth, therefore no WER/latency
accuracy claims. Fixed origins0,.25,.5,.75 of32/30s are engineering diagnostics, not
an exact reconstruction of session resets or original1280px pixels.
At phase.5,105.08–106.05s HUNGRY ranks1 with proposal.763/verifier.788; at phase.75,
105.35–106.32s ranks1 with.809/.847 yet CTC emits EAT in its recent hypothesis.
Window placement substantially changes both paths; compared CTC evidence before/after
OTHER suppression does not explain these HUNGRY failures. No reproducible extraction
or suppression repair yet meets gates; proceed with bounded candidate preparation.

Full all-sign annotation audit:train923 items/3signers/53known signs/41known pairs,
validation237/1/36/8; estimated guarded0.53s background windows494train/16validation.
These are annotation-time estimates pending raw-observation materialization, not
training-ready samples. Existing resampled Stage2 caches cannot establish the new
raw-clock window contract. Added shared raw Vision conversion and per-window
normalization so future frames cannot influence window scale. Existing defaults remain.
Next: complete raw preparation/runtime integration and freeze matched baseline results
before seed17111. No training or protected tests accessed.

## 2026-09-11 — Stage-1 contextual-window implementation started

User approved the bounded plan saved at artifacts/reports/stage1_window_v17/PLAN.md.
Existing dirty reorganization and runtime defaults are preserved. Work is split into
verified-data/trainer, opt-in timestamped runtime, and matched diagnosis/evaluation.
Recovered HUNGRY preview has 1,393 frames for 1,394 timestamps; exclude its damaged
final frame and disclose reduced resolution. Original timestamp mapping includes an
initial ~12.5s gap; constant-rate preview playback cannot reproduce source timing.
No causal regression finding yet. Required supervision will fail closed rather than
convert equal-duration local phrase partitions or unreviewed gaps into ground truth.
Next: freeze comparator hashes/identities/settings, diagnose shifted window origins,
and audit checked foreground/background before authorizing seed17111 training.

## 2026-09-11 08:13 PHT — architecture research for Stage-1-first continuous recognition

User requested research and discussion before further fixes: whether Squeezeformer
makes CTC/Stage 2 unnecessary and whether Stage-1 fine-tuning is preferable. No model,
runtime, data, or test changes were made. Reviewed current model/trainer code and prior
unified-streaming v2 report rather than repeating the experiment.

Verified distinctions: Stage-1 forward pools temporal features into one clip label;
encode retains frame tokens. Current Stage-2 retains unpooled frozen Stage-1-derived
features, then applies its own temporal Transformer and CTC output. Latest live
adaptation explicitly records stage1_frozen=True. Earlier unified streaming trainers
also cache Stage-1 evidence under inference_mode before optimizing a separate causal
head, so those failures do not establish failure of joint encoder sequence training.
Separate prior Stage-1 core adaptation improved local core accuracy63.32->89.58% and
ASLLRP cores66.67->83.33% (24 samples), Citizen95.24->94.18% on that experiment's
validation baseline. Short-window downstream CTC improved ASLLRP exact0/12->4/12 and
WER62.50->41.67%, but failed promotion. These are historical development results,
not the current multimodal baseline or the protected official Citizen test.

Primary research:
- Squeezeformer uses CTC in its speech implementation; encoder architecture and
  sequence alignment objective are complementary:
  https://arxiv.org/abs/2206.00888 and https://github.com/kssteven418/Squeezeformer
- CTC maps unsegmented input to variable-length labels with a no-emission symbol;
  it does not require this repository's separate Stage-2 Transformer:
  https://www.cs.toronto.edu/~graves/icml_2006.pdf
- EMNLP2024 online CSLR trains isolated recognition on continuous-derived contextual
  crops, includes background, foreground saliency supervision, and online duplicate
  removal. Its dictionary uses a CTC segmentor; it is not evidence that isolated
  dictionary training alone solves streaming. Different benchmarks, not our ASL gate:
  https://aclanthology.org/2024.emnlp-main.619.pdf
- VAC supports supervising visual representations as well as alignment, not only the
  downstream decoder: https://arxiv.org/abs/2104.02330
- Streaming Conformer constrains context during training and caches activations;
  supports both CTC and RNNT, so replacing CTC is not a prerequisite for streaming:
  https://arxiv.org/abs/2312.17279

Recommendation for discussion, not an approved new training specification: prioritize
existing Stage-1 recognition adapted to real contextual windows and independently
checked foreground/background evidence; preserve isolated replay and revisable UX.
Use matched HUNGRY Stage-1/Stage-2 evidence to localize the immediate regression before
choosing an implementation. CTC remains a valid baseline; direct Stage-1 online
classification is a credible alternative but requires emission/repetition handling.
No evidence yet justifies a new backbone, ensemble, or RNNT migration. Do not repeat
prior short-window/head-only runs under a new name or treat incomplete signs/OOV
signing indiscriminately as background. Existing data coverage/alignment should be
measured before requesting more videos. Next action: discuss this recommendation.

## 2026-09-11 08:06 PHT — user HUNGRY regression inspected; recording recovered

User reports revisable mode worse than earlier Reel and identifies HUNGRY near the
end of session 20260911_075501_769734. Recovered 1,393 frames from unfinalized mp4v
mdat using a locally generated 640x360/15fps codec header; original recording remains
untouched, final frame damaged. Inspected timestamped ending contact sheets. Visible
repeated chest movements cross sequence-window boundaries near 102.6 and 105.8s;
context hypotheses add GOOD, then MY/COME, with no HUNGRY in 41 accepted updates.
This does not establish lexical-variant correctness, per-sign accuracy, or causality.
Session uses repaired seed1702, not rejected adapters; Stage-1 predictions are empty.
Code confirms revisable mode bypasses older Reel's phrase-adapted Stage-1 classifier.
Median accepted total processing 378.51ms, hand image encoding 339.43ms, plus collection
of roughly one-second windows. Logs lack evidence to distinguish weak visual scores
from CTC/OTHER suppression. Diagnostic files and recovery limitations:
artifacts/generated/revisable_hungry_diagnosis/README.md. Next safe experiment is a
matched Stage-1/Stage-2 comparison and window-origin sensitivity check on these motions;
640px recording cannot reproduce original 1280px retained input exactly. No runtime
changes, training, or protected tests. User requested discussion before further changes.

## 2026-09-11 07:53 PHT — completion revalidated and condensed handoff corrected

User requested continuation before final report delivery. Verified the completed revisable
experiment against current files: all recorded source hashes and report SHA match; both
12-epoch runs have zero eligible epochs, 72 focused tests are recorded passing, and all45
raw replays completed with12/12 exact-subset raw/cache agreement per model. No experiment
rerun or runtime change. Corrected PROJECT_GROUND_TRUTH.md to distinguish the earlier
immutable-prefix experiment from implemented revisable transcription, preserve the user
choice, and record completed transition/core failures. Full evidence remains in
artifacts/reports/stage2_v17_revisable_v1/{README.md,verification.json}. Next safe action:
user review of the report; further alignment/recall work requires its own evidence-based
experiment rather than repeating failed losses.

## 2026-09-10 07:02 PHT — retention repair verified and enabled in continuous Reel

Completed the v2 retention-regression repair using existing data. Both final packages
pass all original frozen development gates plus STEM retention. Seed1701 target
485/284 (S/D/I139/13/333); selected seed1702 target470/284 (137/12/321), down22.1%
from accepted603. Both retain local/exact/contextual6/9/43, Citizen331/378 and
STEM16/21, with zero per-example prediction changes across those retention pools.
This preserves accepted behavior while reducing connected false insertions; it does
not retain v2's217/223 total edits or establish general continuous recognition.
The development-tuned run policy/margin and rejected experiments are disclosed.

Final self-contained package hashes supersede the earlier packaging hashes above:
- seed1701: 5656d21941462fc1b284d6e048a4e37ec0f4bb2ab1dea03e45d53a5ac8dda76b
- seed1702: d7bdc71d68f3093311fe85feef85e9b0b90c838f6d4a11a903afb596248d5685
Review identified unpinned selector configuration and frozen encoder provenance;
both were fixed with observed failing/passing regression checks. Runtime pins now
cover configuration, selector checkpoint, primary/specialist Core ML packages,
frozen encoder and upstream hand-image encoder. Both final packages reload.

Final validation:82 focused tests pass; compilation and git diff --check pass.
Actual Core ML selector plus CPU evidence matches checkpoint decoding on987/987
frozen development samples, zero mismatches. Mac cached-feature p95 is13.38ms
versus7.44ms baseline (added evidence p956.06ms); this excludes extraction and is
not sustained iPhone/full-Finish evidence. Two paired raw-video smoke tests preserve
old/new suggestions and selected transcripts with zero stale frames. Existing live
suggestions KNOW HOW YOU and PLEASE HELP I I remain erroneous; cached-development
retention does not fix those pre-existing live errors.

scripts/live_reel_continuous_v17.py now defaults to selected seed1702. Roll back
with --no-stage2-other-preservation. Legacy entrypoint defaults remain unchanged;
Stage2 remains review-only and does not overwrite the trusted transcript. No camera
session was automatically launched, no new data acquired, and no protected Citizen,
SemLex or local test was accessed. No git commit was made.

Final evidence: artifacts/reports/stage2_v17_transition_repair_v3/README.md,
validation.json, verification.json, tests.log and replay_comparison.json. Builder:
scripts/build_stage2_other_preservation_v17.py; evaluator:
scripts/evaluate_stage2_other_preservation_v17.py --runtime. The completed scope is
the measured v2 retention regression. Remaining live recognition errors and mobile
readiness are separately unresolved; do not claim them fixed or rerun protected
tests for development. User can run the continuous Reel command with this repair.

## 2026-09-10 08:48 PHT — live-lock diagnosis in progress: session and paired core evidence

User requested a completed diagnosis report of latest webcam locking difficulty,
temporal learning and live extraction. Examined session20260910_071844_645159:
66 Stage1 probes,40 accepted,14 stable proposals,10 commits;4 keyboard Finishes.
Stage2 runs at Finish and is review-only; it cannot rescue the live locks. First
Finish temporarily decodes HELLO HOW YOU, then repeats YOU; last temporarily gives
I READ TOMORROW before repetition/prefix changes. Intended user transcripts remain
unverified; do not assign a webcam WER. Read-only independent code/history analysis
confirms growing multi-sign windows and verifier disagreements, not simply high
commit thresholds. Raw history and lowres video inspected; fixed15fps recording
omits original timestamp/gap timing, so it cannot exactly recreate capture timing.

New diagnostic artifacts only under artifacts/reports/stage2_v17_live_lock_diagnosis_v1.
All12 exact-variant dev phrases replayed through current runtime (no cherry-picking).
Matched24 annotated cores via ASLLRP occurrence/frame provenance;20 have existing
valid archives,4 are2–3-frame cores excluded by the existing4-frame minimum.
Isolated frozen Stage1 gets16/20, repaired CTC18/20. On8 complete pairs/16 tokens:
full cached phrase5 edits, stitched independent CTC cores2 edits, oracle core-window
concatenation2 edits. Cropping/window resampling changes as well as boundaries;
this supports temporal/window sensitivity without proving every upstream cause.
A first probe stopped on missing short-core archives, then explicitly recorded those
exclusions rather than padding or silently shrinking the denominator.

Controlled landmark-resolution/sampling intervention now running, holding cached
hand evidence and window edges fixed. Runtime/checkpoints/data splits untouched;
no protected evaluation or new training. Final report pending this experiment and
reconciliation of evidence. No more model ensemble proposed.

## 2026-09-10 08:53 PHT — completed continuous-lock diagnosis report

Completed requested latest-webcam history, temporal and live-pipeline diagnosis.
Canonical report: artifacts/reports/stage2_v17_live_lock_diagnosis_v1/README.md;
verification.json hashes all quantitative evidence. No runtime/model/training-data
change. Latest webcam remains20260910_071844_645159:66 probes,40 accepted,14 stable
proposals,10 commits; hand detection coverage mean99.54%. Commit-only threshold
reduction to0.25 would add DAY for a COME proposal, not recover either SORRY proposal;
these are fixed-proposal counterfactuals, not full changed-policy accuracy tests.
Stage1 isolated growing crops drive locks; Stage2 runs at Finish and is review-only.
User intended transcripts unverified; no webcam WER or expert lexical claim made.

All12 exact-variant JONATHAN development phrases (24 tokens) replayed successfully:
cached Stage2 WER37.50%,4/12 exact; raw Stage2 WER45.83%,3/12 exact; committed
transcript WER79.17%,0/12 exact. Raw/cached match7/12. Six committed outputs empty.
No cherry-picking, no protected tests. Manual core probe evaluates20/24 cores:
frozen Stage1 correct16/20; CTC18/20. Four2–3-frame cores explicitly unavailable.
Eight complete pairs: cached WER31.25%, isolated-core CTC/concatenated-core-window
CTC both12.50%, with different failure identities. This supports temporal/context
sensitivity; manual boundaries also alter duration resampling/extraction context.

Controlled frontend test retains cached hand evidence and source window edges:
cached inputs through CoreML and fresh1280px source-rate landmarks reproduce all12
cached predictions (max encoder/cache difference0.001947). Resolution640px alone
changes9→11 edits. Reducing samples to20Hz makes one tail3frames/unusable. Shared
11-clip/22-token cohort: cached/fresh1280-source8 edits;640-source10;1280-20Hz9;
640-20Hz10. Failed first probe and unavailable tail explicitly retained. Actual
live hand crops, detector state, auxiliary phase and timing remain jointly measured,
not individually causally isolated. Do not present640px as the only live cause.

42 focused existing tests pass; diagnostic scripts contain cohort/provenance/result
assertions; report links/hashes and all12 replay exits/history outputs reconciled.
No additional ensemble or blanket threshold change recommended. Next priorities:
continuous provisional/commit design, matching training/live extraction contracts,
then temporal adaptation on existing connected sequences including short signs and
endings/repetitions. New targeted timestamped/transcript-confirmed capture only if
needed for remaining domain gaps. Recorder currently writes fixed15fps without
preserving source timestamps/gaps, so lowres webcam replay is not exact timing.
This completes diagnosis/report scope; live recognition is not claimed fixed.

## 2026-09-10 18:54 PHT — user confirms revisable sequential transcription; research discussion

User prefers continuous transcription and explicitly accepts revisions to recent displayed words. Earlier proposed hold-until-Stage1-lock experience is not the product target. User requests discussion/research first, not implementation. Transition/coarticulation false emissions are a priority; small residual mistakes may be corrected at phrase completion, but final recognition should retain visual evidence rather than rely on grammar alone.

Primary-source research: AWS documents revisable partial transcripts and optional last-few-word stabilization, with accuracy/latency tradeoff (https://docs.aws.amazon.com/transcribe/latest/dg/streaming-partial-results.html). Google distinguishes interim stability from finality (https://docs.cloud.google.com/speech-to-text/docs/reference/rest/v2/StreamingRecognitionResult); streaming transducers condition on accumulated signal and output history (https://www.research.google/blog/an-all-neural-on-device-speech-recognizer/). Two-pass deliberation uses both original signal and first-pass hypotheses (https://research.google/pubs/deliberation-model-based-two-pass-end-to-end-speech-recognition/). EMNLP2024 Towards Online Continuous Sign Language Recognition and Translation uses sliding-window isolated recognition trained with background/coarticulation, surrounding-clip augmentation and saliency supervision (https://aclanthology.org/2024.emnlp-main.619.pdf); evidence is on its own sign-language benchmarks, not validation on our ASL runtime.

Decision: separate transcription UX from recognizer architecture. Keep existing visual/temporal learning as a starting point; investigate transition-aware training and revisable live sequence output before choosing any transducer replacement. No training or runtime changes made in this discussion.

## 2026-09-10 19:32 PHT — revisable runtime verified; transition experiments running

Implemented --revisable-transcript in continuous Reel: no prefix locks, changeable HUD, per-revision events, disk-retained visual windows and Finish decode over overlapping8-window chunks with absolute-time logit stitching/one CTC collapse. Baseline raw replay completed all12 exact-variant clips,2 local clips and user saved recording. User recording retained22 windows;4 Finish chunks covered all of them;12 live revisions. No verified intended transcript/source timing, therefore no user WER. 39 focused tests passed after updating HUD mock signature; screenshot hud_preview.png rendered and inspected. Worker usage limit interrupted HUD follow-up, main completed remaining integration and checks.

Annotation supervision payload:1609 guarded gap bins (1505train/104development),2407 eligible nonempty prefix targets,1128 full alignment matches. Prefix OTHER collapse and feasibility checked; per-span frame rate used. Second independent supervision artifact adds1595 guarded known cores (1299train/296development) to counter silence; original running-training artifact unchanged. Paired12-epoch training jobs live, gap+prefix seed17101 and same plus.25core CE. No epoch eligible so far. Previous adaptation error decomposition is material: OTHER-domain insertions323->32 but deletions11->138. Final beam8/topk12 control reuses existing CTC decoder; it leaves exact errors unchanged, changes connected478->477 and worsens local19->20, so no default beam change justified. Finite beam input floors run-veto negative infinity to-10000; no language model rescoring.

## 2026-09-10 19:46 PHT — revisable transcription experiment complete; no promotion

Completed authorized runtime, transition-aware training experiments and final report artifacts/reports/stage2_v17_revisable_v1/README.md. Evidence verification.json, training_verification.json, raw_verification.json and tests.log. --revisable-transcript is opt-in, changeable live hypotheses, recent8windowcontext and whole retained utterance overlapping visual Finish re-decode. Existing defaults/checkpointunchanged. No transducer/ensemble replacement or grammar-only lexical repair.

Both12epoch runs completed: gap+prefix1738s selectedepoch12; plusknowncore1821s selectedepoch7. Zero eligible epochs across24. Matched core variant connected210/284 (73.94%WER), local17/259 (6.56%), exact12/24 (50%). Connected S/D/I41/153/16, versusbaseline144/11/323 and precedingadaptation56/138/32. Thus56.07% fewer totalconnectederrors thanbaseline still deletes53.87% knownreference signs. Corevariant isolatedCitizen316/378 (83.60%) versus331/378 (87.57%); originalphrase retention local16vs6,exact13vs9,context56vs43 allfail. STEM16/21 unchanged. Gap knownemissions baseline13/104,precedingadaptation3/104,newgap3/104,newcore4/104: no new demonstrated gap-suppression gain. No promotion.

45 successful rawpaced replays completed (baseline,gap,core each12exact+2local+userdiagnostic), excluding retained initialfailedadaptedflagattempt. Eachmodel exactraw/cacheagreement12/12. RawexactWER45.83%baseline versus50%bothadapted; nonemptyupdatesbeforeFinish1/12baseline and0/12adapted, so latency unresolved. All3userreplays retain22windows/fourFinishchunks and finalvisualoutput may reviseearlierprefix. No verifiedusertranscript/timing, no userWER. LongerlocalHELLO correctedbynewmodels; PLEASE HELP I stillduplicatesHELP. Wholemetricsgovern decision.

72 focusedtests pass includingprovenanceflagregression;4best/lastcheckpoints reloadfinite,supervisionhashesmatch. Fixedinputcontract anddefaultbaselineSHAunchanged. HUDscreenshotrendered/inspected. gitdiff--checkpasses. Protectedtestneverused,unrelatedworktreepreserved. Limitations explicitlydocumented: annotationbins approximate,sparsegapcoverage,noexplicitnewholdaugmentation,noiPhonemeasurements ortranslationaccuracytest. Goalexperimentcomplete; robustnaturalcontinuousASLnotclaimedfixed. Nextsafeaction:userreviewreport anddecide whethernewindependentlyverified boundary/hold datajustifyfurtheralignment/recallwork.

## 2026-09-09 20:02 PHT — existing videos reproduce learned v2 emission regressions

User reports no additional observed live failure, challenges requests for more data,
and wants discussion of what is actually needed. Decision: do not request new
recordings or launch another acquisition; current evidence supports diagnosis with
existing videos, labels, frozen features and checkpoints. Recommend locked100
connected recognition with unfamiliar combinations and tolerance of unsupported
signing as the next bounded target; user has not yet adopted a new product scope.

Inspected 20 timestamped sampled frames each from two existing local familiar-domain
validation videos, then ran bounded CPU inference with accepted selector, initialized
candidate and both with-STEM best checkpoints. All four model results reproduce
the saved predictions. Both accepted/initialized models correctly emit PLEASE HELP I
and HELLO HOW YOU. Both adapted seeds emit PLEASE HELP I I and HELLO HOW. The extra
I is a new spike at CTC step 36 after I at step 28, with blanks between; YOU at step
21 disappears in both adapted models. These are output steps, not video timestamps.
Sampled frames show a prolonged final chest-pointing posture in the first clip and
a final forward-pointing gesture in the second. This is not expert ASL semantic
validation or full-motion playback. The two initially correct examples establish
an adaptation effect separate from the aggregate initialization/selector mismatch.

Artifacts: artifacts/reports/stage2_v17_transition_diagnosis_v1/README.md,
model_probe.json and two contact-sheet JPGs. No code/dataset/checkpoint/runtime
changes, training, protected evaluation or additional test-suite run occurred.

Web research checked primary CTC (Graves 2006), CTC spike-guidance (Kurata/Audhkhasi
2019), sequence-level KD (Huang 2018), VAC CSLR (Min 2021), and official Citizen use
guidance. These support testing temporal alignment/preservation objectives; they do
not prove a specific loss change will fix this model. Links and limits are recorded
in the diagnostic README. Next safe action: inspect timewise known/blank/OTHER
scores and source/replay coverage on existing development inputs, distinguishing
model discrimination, emission calibration, held poses/window sensitivity and
teacher disagreement before proposing a bounded training change.

## 2026-09-09 20:40 PHT — posterior diagnostics support conditional-emission preservation

Cached accepted/plain-initial/with-STEM seed outputs on 987 unique development
examples and accepted/with-STEM train outputs on 3,994 examples (including 1,116
previously omitted contextual training cores; all embedded encoder hashes match).
Reports under stage2_v17_transition_diagnosis_v1 include posterior_diagnostics.json,
window_diagnostics.json, counterfactual_diagnostics.json, training_detection.json.

YOU remains the top known sign at its old emission step, but blank probability rises
from 0.385 accepted to 0.962/0.982 adapted. Its probability falls 0.614 to 0.037/0.018;
OTHER is negligible. Held final I gets an extra step-36 spike: accepted I probability
0.0067, adapted 0.522/0.758. Restoring the old blank odds recovers seed1701's local
6 edits; restoring the full old conditional distribution recovers local 6, contextual
43/44 and STEM 16/21 but still fails exact retention (12). These controlled output
interventions locate the local mechanism; they do not isolate every training cause.
Removing the final cached window removes the extra I but seed1702 then repeats HELP,
so blind trimming is not a safe fix. A fixed majority-OTHER router failed exact and
contextual gates (10;46/45), and was rejected, not integrated.

Selected repair experiment: keep accepted known/blank conditional logits frozen and
fit only the OTHER output on frozen v2 temporal representations using original CTC
labels plus previously omitted context negatives. Reuse all accepted-model behavior;
no identity, phrase, source or duration routing at inference. This is a larger research
composite, not model compression or an iPhone-ready claim. A behavioral regression
for preserve_known_ctc_logits was observed failing, then passed after adding the
shape-checked concatenation helper to model_stage2_v17.py. Training/validation caches
are under artifacts/models/stage2_v17_other_preservation_v1; bounded 20-epoch, two-
seed linear-head fitting now runs on CPU (no Stage-1 or temporal-backbone training).
No protected test accessed; existing runtime/artifacts unchanged.

## 2026-09-07 20:20 PHT — continuous Reel experiment delivered; metadata candidates acquired

Completed the authorized separate experiment. Run:
`venv/bin/python scripts/live_reel_continuous_v17.py`.
Amber GLOSS? is tentative; F/button finishes one utterance. Incoming frames newer
than a committed clip survive verification. Finish drains eligible new tail evidence
once, then computes Stage 2 over private disk-spooled exact observations. Its candidate
is displayed separately for review and cannot auto-replace or speak over the Stage-1
transcript. No accept-suggestion control is implemented. Speech is finished-sentence
only. Reset, quit without Finish, reset during CTC, and a second Finish during prior
naturalization have regression coverage. Original Reel defaults and frozen100 retained.

Final paired paced development run: both lanes 1/5 exact with 5/12 token edits;
there is no aggregate Stage-1 accuracy improvement. Continuous retained55observations,
first tentative output median0.904s (range0.539–6.311s; early guesses can be wrong),
Finish decode median1091ms (range794–1267ms). Sequence review candidates5/5exact on
the same familiar short recordings. Baseline itself varied from the initial2/5pass;
async scheduling changes candidate windows. A capture-start outlier and interruption
between pairs prevent claims of controlled hardware timing. Webcam FPS, human repeat
attempts, and unseen-signer accuracy were not measured. Default tiny-naturalizer smoke
also completed: Stage1 HELLO HOW -> "How is you?", candidate HELLO HOW YOU,
Finish total1321ms. This confirms integration while exposing remaining recognition
and language quality limits. Do not promote this prototype or its5/5candidate result.

Reports: `artifacts/reports/continuous_reel_v17_experiment_v1/README.md`,
`final/comparison.json`, runnable `run_replays.py`, and visually checked preview/finished
PNGs. Runtime changes: `scripts/live_reel_continuous_v17.py`, opt-in hooks in
`scripts/live_reel_stage1_v17.py`, provisional rendering in `scripts/reel_hud_v17.py`,
and `test/test_live_reel_continuous_v17.py`. All41focused Reel/CTC/HUD/cache tests pass;
syntax checks and git diff --check pass. Core ML/syntax validation required approved
unsandboxed execution due to GPU/model compilation and Python bytecode-cache paths.

Completed bounded public metadata audit after delegated audit failed with usage limit.
`artifacts/reports/continuous_reel_v17_web_audit_v1/` contains reports, hashes, official
NCSLGR index and MoLo listing/sample EAF. NCSLGR index564370bytes,1887uniqueutterances,
38collections; excluding existing166leaves1721. Conservative case-sensitive exact
Citizen raw-label matching found77textual contiguous-span candidates in76parents,
42distinctsequences/39rawclasses. These lack per-sign timing and participant IDs and
are NOT variant-approved training data. Current DAI requires login for XML downloads.
MoLo public API successfully lists32files including16EAFs. One inspected timed EAF
has299right/124leftannotations;34right/13left exact raw-label occurrences across13
classes, with two-hand duplicates not resolved. No video acquisition or training.
Sources: https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html ,
https://dai.cs.rutgers.edu/dai/s/daioriginal , https://ida.gallaudet.edu/molo/ ,
https://api.osf.io/v2/nodes/9uevh/files/osfstorage/ .

Next safe work: user webcam comparison of the separate prototype and exact-variant,
two-hand/timing/signer audit of these candidate annotations before selective video
acquisition. No model retraining, threshold tuning, protected split access, baseline
replacement, or new continuous-accuracy claim. All work from this request is left
reviewable in the existing worktree; no commit or cleanup of unrelated changes.

## 2026-09-07 18:02 PHT — initial matched results and Finish review corrections

Five initial paired paced replays finished: both commands 2/5 exact with 4/12 token
edits, so buffer preservation alone did not improve aggregate Stage-1 accuracy.
Continuous lane retained 59 pending observations; first tentative display median
1.006s versus baseline first commit 1.726s (different semantics, some guesses wrong).
Review-only Stage-2 candidates were 5/5 exact; median Finish decode 805ms. These are
familiar short recordings, not independent/generalization evidence. THANKYOU FRIEND
regressed in Stage 1 while TOMORROW SCHOOL GO improved; do not hide individual changes.

Independent review identified two valid Finish edge cases: buffered new evidence
could be discarded on Finish, and a second utterance's Finish could be rejected during
prior naturalization. Added failing real-loop regressions and fixed both behind the
experiment behavior: Finish may probe an eligible newer tail once (never duplicate
stability hits on identical evidence); Stage 2 waits for that finite drain. A second
Finish is sealed while the previous language job completes; timestamps belong to each
utterance context. Also cleared stale uncommitted previews at candidate reset.
Review suggestion to disable baseline ten-finger controls was rejected: that control
was pre-existing user work, not introduced in this experiment. Original defaults stay.
Reviewer delivered findings before a usage-limit error; no further agent review run.

Final five-pair replay rerun is underway under
`artifacts/reports/continuous_reel_v17_experiment_v1/final/` because Finish behavior
changed. Reset-during-CTC test additionally checks stale result exclusion. No training,
threshold edits, model promotion, or protected split access.

## 2026-09-07 17:56 PHT — continuous Reel smoke passed; paced comparisons in progress

First real Core ML smoke completed after sandbox compilation/GPU access failed and
the user approved an escalated saved-video run. Unpaced HELLO HOW YOU replay returned
Stage 1 HELLO HOW and review candidate HELLO HOW YOU; candidate did not overwrite the
transcript. Added optional --realtime-video to pace saved sources for comparison;
new parser regression failed first and all 32 relevant tests then passed. Rendered
and visually inspected preview.png and finished.png in
`artifacts/reports/continuous_reel_v17_experiment_v1/`: amber HELLO? appears centrally
before commit, and final sequence suggestion is explicitly marked for review.

`run_replays.py` in that report folder is running five already-used familiar sources
against original/continuous commands, with identical thresholds, 20fps observations,
literal naturalizer, no display/speech/control-gesture filtering, and timestamp pacing.
It logs capture elapsed time, first tentative/committed output, verifier timing, retained
frames, final edits and Finish decode latency; no protected data is read. Early results
include an ~8s initial capture stall in one continuous HELLO replay despite ~10ms
proposal calls, so do not promise uniformly fast webcam response. Early tentative
glosses can be wrong (GOOD MORNING first previews MORNING); first-preview speed is not
first-correct-gloss latency. Matched results/report and independent async-state review
remain pending. No model weights or thresholds changed.

## 2026-09-06 21:16 PST — actual-video causal v2 improves to5/6 with context, SCHOOL still omitted

Six-video replay completed under session10885; no background process remains from
this turn. Result: continuous_rebuild_v17_v1/video_causal_v2/result.json. Raw exact
4/6,13.33%WER; existing guarded beam context5/6,6.67%WER. Prior same-video v1raw2/6,
46.67%WER and context3/6,33.33%WER. HELLO HOW YOU and MY NAME now exact raw;
context recovers FRIEND after THANKYOU. TOMORROW SCHOOL GO still omitsSCHOOL in
raw and corrected. These are six development templates, not unseen-combination or
live-camera accuracy. Per-frame p95 about104–344ms on this run, so do not claim
sustained30Hz real-time performance from these desktop/video measurements.
Initial aggregation import used wrongmodule; corrected to existing streaming edit
helper, no duplicate distance implementation. Saved targets, predictions, sourcepaths
and timing. No official test accessed; no checkpoint promoted by this smoke.

Added a guard rejecting degenerate palm-plane finger direction in the symbolic
handshape helper; focused18 tests pass and both v3rendered trajectories were verified
bit-identical after guard, reporthash updated. Source-match test passed separately
in19-test aggregate before guard. git diff --check passes. Human review of v3YOU/NEED
is pending. Full SignWriting motion generation, continuous joins, all100combinations,
real transfer from synthetictraining and robust continuous translation remain unmet.
Next useful actions: implement/test explicit symbol orientation/movement for this
bounded pilot; use pending fluent handshape feedback; evaluate motion-proposal fallback
against the omittedSCHOOL and OOV controls before optionalruntime integration.

## 2026-09-06 16:26 PST — pointing review identifies detector curl; lexical extension audit and causal model finish

User's v6 feedback: pointing YOU finger is not good. Root inspected raw source hand
closeups and estimated3D chains across source frames13–20: index bends fluctuate
35–75deg to120–150deg, confirming detector geometry is unstable. A tight Apple-box
MediaPipe crop probe also yields unstable curls, so crop-only replacement is rejected.
Existing pinned ASL-LEX annotations explicitly identify YOU as handshape1, selectedi,
FullyOpen; NEED isBent. Added opt-in lexical extension constraints for FullyOpen
selected fingers only, with a regression verifying bent entries/nonselected fingers
are untouched. Ten rig tests pass. Source comparisonv8 is rendering with lexiconhash
and explicit annotation provenance; these are generated constraints, not observations.

Causal local extraction completed287train/200validation. Matched-camera continuousv2
finished25epochs with duration1.5x/2x training replays. The selected metrics are in
artifacts/models/continuous_evidence_v17_causal_v2/result.json; finalepoch localWER
26.11%, not yet the selected metric or runtime score. Training a matching Stage3
context model now; original context hash cannot be silently attached to a newrecognizer.

Stage3's earlier cyclic-shift control was flawed: adjacent sorted rows often have the
same label. Corrected to different-label same-source near-duration controls; five
context tests pass. Proposal-only v1 exact on local183/200 vs0/200 with different-label
motion, Citizen357/378 vs0/378, SemLex758/978 vs0/978. This demonstrates motion use on
these development sets, not general compositional translation. Reranking still relies
heavily on language patterns (mismatched local106/200 vs matched112/200). Existing
reports are preserved; corrected controls are context_mismatch_control.json. No
runtime promotion or claim that all new recordings can be eliminated.

## 2026-09-06 16:15 PST — automatic utterance-end integration verified; causal train loader validated

Created a diagnostic fixture from validation PLEASE_HELP_ME/09d3e2c0 plus60 black
frames (not training data). Current live script finalized PLEASE HELP I at5.433s,
rendered Please help me once, then processed another.933s with an empty next utterance.
Thus automatic no-hand-gap finalization happens before EOF and does not lock input.
Artifact: continuous_rebuild_v17_v1/utterance_end_smoke;fixture is marked by its path.
This tests integration only, not natural pause detection reliability.

Actual causal training loader check passed:4509total training examples,861local
(287original plus1.5x/2x duration replays), unchanged target/signer identities and
expected frame counts, unique sample identities. Saved causal_loader_check.json.
Validation remains unaugmented by construction; its full load still awaits extraction.
The extraction job is live at320/487completed examples. All44 focused tests passed
before the subsequent isotropic-body fix; nine rig tests passed after that fix.
V7 HOW source-side preview was inspected and uses correct anatomical side and metric
body scaling. Pending fluent review still concerns source-comparisonv6, not blanket
approval of all later variants.

## 2026-09-06 10:38 PST — causal runtime equivalence checked; camera path and guarded context added

Eight development clips match full-sequence vs frame-by-frame final CTC tokens;
maximum raw-logit difference <9e-6. Cache FP16 rounding contributes <.003 logits.
Report: `continuous_rebuild_v17_v1/runtime_equivalence.json`. This isolates the
remaining recognition errors from incremental convolution/CTC implementation.

Added `continuous_vision_v17.py` (past-only 32-observation normalization; missing
hands remain absent; no activity trim) and `scripts/live_continuous_v17.py` (camera
or video, continuous partial/stable display, Enter finalizes an utterance, asynchronous
English rendering, optional completed-utterance speech). Two causal camera tests
passed after initial import failure; CLI help works. Actual video replay is running
on validation PLEASE_HELP_ME/41b5e8c9. It is a diagnostic: camera normalization differs
from cached whole-window normalization, and elapsed 30-Hz duplicate ticks do not
represent additional camera observations. No measured live/iPhone claim.

Optional context checkpoint now pins recognizer hash/vocabulary and ranks final
utterance candidates only. Blank/OTHER-leading hypotheses bypass correction because
the observed open-vocabulary regression makes rewriting them unjustified. The new
regression passed after failing before implementation; all three context tests pass.
Partial recognition remains independent; context is opt-in, not promoted as default.

Avatar v5 also rotates hand surfaces with the palm plane instead of using bone axes
alone; eight avatar tests pass. The v5 contact sheet was inspected: solid human mesh,
visible hands, but planar handshape/retrieved-rest naturalness still needs fluent
assessment. A concrete MP4 review question is pending with the user; this is quality
assessment, not implementation permission. Synthetic training eligibility remains false.
`git diff --check` passed before the latest camera additions; rerun before handoff.

## 2026-09-04 12:19 PST — corrected short-window streaming experiment shows real gain

The recommended bounded follow-up was completed without modifying the accepted Reel
path or accessing protected/reserved tests. A new Stage-1 checkpoint was adapted on 92
manually aligned ASLLRP training sign cores with Citizen, SemLex, and local phrase-core
replay. At selected epoch 9 it changed Citizen isolated top-1 from 95.24% to 94.18%,
SemLex from 85.28% to 85.79%, local phrase cores from 63.32% to 89.58%, and the 24
ASLLRP validation cores from 66.67% to 83.33%. This supports the manual annotations and
continuous-domain adaptation while passing the configured isolated-retention gates.

The corrected CTC experiment removed incomplete isolated prefixes from hard blank
supervision, added 260 full local phrase trajectories containing known-plus-`OTHER`
targets, and balanced exact ASLLRP/local phrases plus Citizen/SemLex replay. With the
old 32-frame trailing window, isolated retention improved substantially and transition
false emissions fell, but ASLLRP remained 0/12 exact. A direct observation audit found
that 32-frame pooled Stage-1 windows exposed both expected glosses in 0/12 validation
clips; 8-frame trailing windows exposed both in 8/12. These clips are only 21--54
source frames, so long windows mix the first sign, transition, and short second sign.

The selected 8-frame-window / four-frame-stride causal run reached 4/12 ASLLRP exact
and 41.67% WER, versus 0/12 and 62.50% for the original/corrected 32-frame runs. It also
reached 80.41% exact on 97 local phrase clips, 92.59% on 378 Citizen isolated clips,
82.21% on 978 SemLex clips, 96.15% exact on 52 added local `OTHER` phrases, and 2.30%
aggregate transition false emission. A matched two-frame-stride run tied ASLLRP at
4/12 / 41.67% but reduced local exact to 55.67%, Citizen to 89.15%, and took 120.75
seconds instead of 86.91; it is rejected.

The short-window result is the first genuine signer-disjoint continuous gain, but it
is not promoted: ASLLRP WER remains above 35%, Citizen retention is below the desired
93--94%, ASLLRP transition false emission is 2/12, and ASLLRP `OTHER` known-gloss WER
is 89.08%. Keep the stable Reel prototype. Proceed with the planned 30 phrases x 13
signers x 3 connected performances, using two natural-speed and one slower connected
take, exact gloss targets, manual boundaries for all validation/sealed clips and a
representative training subset, plus ordinary non-sign activity. Full methods and
matched results are in
`artifacts/reports/unified_streaming_ctc_v17_experiment_v2/README.md`.

## 2026-09-03 11:33 PST — exact-input cached Reel experiment passes offline regression gate

The current user session was measurably slower than the earlier stable Reel session:
effective landmark processing fell from 19.2 to 16.5 FPS, camera-frame dropping rose
from 10.1% to 38.5%, landmark proposals rose from 12.2 to 26.9 ms median, and full
verification rose from 224.7 to 362.9 ms median. Stage 2 was disabled. The hand image
encoder consumed 268.8 ms median and remained the dominant modeled bottleneck. A
standalone component benchmark measured 10.46 ms hands-only Apple Vision, 6.01 ms live
lips, 3.72 ms HUD, and negligible Finish geometry, implicating full-verifier/system
contention rather than the ten-finger control. The machine also had 23/24 GiB memory in
use, 10 GiB compressed, and 12.75 GiB swap occupied, but no thermal warning.

A separate `scripts/live_reel_cached_stage1_v17.py` path now reuses MobileCLIP
embeddings only for byte-identical RGB crops through a bounded 512-entry LRU. It keeps
all 16 time points, three views, landmarks, boxes, and validity masks. Twelve/eight
time-point and reduced-view alternatives were rejected because each lost at least one
Citizen validation prediction. On a real two-window overlap benchmark, cached and
baseline embeddings were bit-exact and median encoding improved 784.07 -> 563.86 ms
(1.39x). Across matched five-video local replays, both lanes produced the same five
final sequences and the same 2/5 exact count. Fourteen verifier calls per lane measured
359.00 -> 304.15 ms median total and 272.51 -> 215.15 ms median hand encoding, with an
observed 15.52% cache hit rate. This passes an offline no-regression gate but is not yet
a webcam FPS or independent accuracy result. Evidence is in
`artifacts/reports/live_reel_cached_stage1_v17_experiment_v1/README.md`. The original
Reel entry point retains the original verifier; only small dependency-injection hooks
were added so the separate command can reuse its loop. No test or sealed split was
accessed.

## 2026-09-03 08:07 PST — tiny causal sequence head is fast but fails the accuracy gate

A new non-destructive streaming experiment compared (1) a raw 61-node causal TCN and
(2) a 26,861-parameter causal depthwise TCN/CTC head over rolling logits from the
accepted phrase-adapted Stage 1. The raw model collapsed to CTC blank in its fail-fast
screen and was rejected before a full training budget. The Stage-1 evidence head is a
119 KiB checkpoint and retains per-block causal state. On this Mac/MPS, Stage 1 measured
29.33 ms median and the head added 3.06 ms median, excluding landmark extraction.

The selected development-only head reached 44.33% exact / 40.93% WER on 97 local
phrase clips and 0/12 exact / 62.50% WER on signer-held-out ASLLRP phrases. It retained
94.71% Citizen and 83.95% SemLex isolated exact accuracy. This misses the existing
Stage-2 validation reference of 92.78% exact / 2.70% WER local and 16.67% exact /
45.83% WER ASLLRP, so the new head is **not promoted or connected to live inference**.
Its speed proves the head is not the bottleneck; learned continuous boundaries and
domain coverage are.

A controlled modality ablation confirms that all existing v17 landmarks are needed:
all-landmark versus hands-only accuracy was 95.24% vs 65.08% Citizen, 85.28% vs 59.51%
SemLex, and 63.32% vs 44.02% on local phrase segments. New live-only MediaPipe mouth
nodes were not added because stored phrase archives lack them and would create schema
mismatch. A separate Stage-1 contextual adaptation added 92 genuine ASLLRP training
segments with Citizen/SemLex/local replay. Its gated epoch improved local segment
accuracy 59.85% -> 66.41% and ASLLRP held-out segment accuracy 66.67% -> 75.00%, with
Citizen 95.24% -> 94.97% and SemLex 85.28% -> 85.17%, without changing architecture or
latency. But the downstream sequence head became worse (43.24% local and 66.67%
ASLLRP WER), so this checkpoint also remains experimental.

The user's new fast learned-emission histories confirm the previously documented
accuracy tradeoff: sessions `20260903_073822_141602` and `20260903_074154_998790` did
not load the RGB hand verifier and contained many low-confidence rejects; the later
`20260903_074617_667378` comparison did load it via `--full-visual-verifier`. No default
was silently changed again during this experiment. Full model/data/latency evidence is
in `artifacts/reports/streaming_stage1_head_v17_experiment_v1/README.md`. No protected
test or external-reserved sample was accessed.
Eleven focused streaming/emission tests, compilation of all new entry points, and
`git diff --check` pass.

## 2026-09-03 07:29 PST — emission live bottleneck removed; lightweight streaming direction researched

The user's first learned-emission webcam session at
`artifacts/reports/live_reel_emission_stage1_v17/20260903_071450_141909/`
confirmed an implementation bottleneck distinct from model latency. Its 130 learned
landmark proposals took 37.11 ms median / 48.95 ms p90, while 26 subsequent hand-image
verification calls took 745.07 ms median / 979.22 ms p90 (maximum 1,088.00 ms). The
103-second session produced 1,273 landmark observations but dropped 1,604 stale camera
frames. Repeated Core ML hand-crop encoding was the dominant source of the reported
lag, not the 101-logit emission model.

`scripts/live_reel_emission_stage1_v17.py` now defaults to a landmark-only fast commit:
after two stable learned-emission proposals, it reuses that accepted result and does
not load or call the expensive hand-image verifier. `--full-visual-verifier` restores
the prior comparison path. The accepted `scripts/live_reel_stage1_v17.py` behavior is
unchanged because the new switch is enabled only by the separate wrapper. An 18-second
saved-video smoke processed all 270 frames, produced 60 proposals at 37.84 ms median,
and made no additional verifier inference. It establishes removal of the 0.65--1.09
second operation, not webcam FPS or accuracy; those require a new live session. The
speed/accuracy tradeoff is deliberate because the image verifier may reject or correct
some landmark proposals.

A primary-source review found no compatible downloadable checkpoint for the fixed
100-ASL v17 vocabulary. The direct EMNLP 2024 online CSLR model is conceptually similar
to Reel but uses two S3D streams and reports 5.1 GB V100 memory. Skeleton CSLR evidence
from CoSign and ICCVW 2025 supports grouped hand/body/face/mouth keypoints, short 1D
temporal convolution, and CTC supervision. The recommended next experiment is therefore
a sub-million-parameter causal landmark TCN with cached per-frame state and a 101-way
CTC output (blank plus 100 glosses), trained on genuine phrase sequences plus isolated
augmentation. It is a new streaming sequence recognizer, not the existing whole-phrase
Stage-2 live policy. Full paper/model assessment is in
`artifacts/reports/lightweight_streaming_stage2_research_v1/README.md`. Twenty-seven
focused tests, compilation, and `git diff --check` pass.

## 2026-09-03 00:48 PST — learned Stage-1 emission helps but does not replace sequence recognition

A complete separate Stage-1/Reel experiment now exists in
`active/v17/train_stage_1_reel_emission_v17.py`,
`active/v17/model_reel_emission_v17.py`, and
`scripts/live_reel_emission_stage1_v17.py`. Neither accepted Reel checkpoint nor its
default entry point was replaced. The selected model freezes the phrase-adapted
100-gloss classifier and learns a tiny ordered-temporal `__NO_EMIT__` head from
complete signs, incomplete prefixes, and between-sign transitions. The original 100
gloss logits are preserved bit-for-bit. Training uses permitted Citizen/SemLex replay,
all local phrase caches, and manually aligned ASLLRP training phrases; JONATHAN remains
signer-disjoint ASLLRP validation. No test or reserved RIT data was accessed.

At its validation-selected 0.77 threshold, the temporal head accepts 97.68% of 1,639
complete validation windows and catches 83.56% of 1,813 incomplete windows. The held-out
ASLLRP slice is materially weaker: 20/24 complete accepted and 16/36 incomplete caught.
The safer live threshold 0.95 accepts 99.02% complete and catches 70.60% incomplete
overall; its ASLLRP figures are 21/24 and 7/36. A first linear-head attempt is retained
as a failed baseline: it caught only 13.7% incomplete at a 95.5% complete-accept gate.

The FP16 Core ML export is 13.48 MiB and measured 11.93 ms median / 14.04 ms p90 on this
Mac. Across 378 Citizen validation clips it had zero top-1 and zero emission-decision
mismatches against PyTorch. The separate live preset probes at 0.32 seconds, every 0.08
seconds, and requires two agreeing proposals. It tied the accepted Reel timing on five
local phrases at 3/5 exact and four token edits, but committed more slowly (0.93 versus
0.67 seconds median). On the same 12 fast ASLLRP clips, it remained 0/12 exact but
improved from 20 to 17 token edits, emitted 8 rather than 5 glosses, and reduced median
committed duration from 0.67 to 0.57 seconds.

Therefore this model remains an experimental auxiliary gate, not the new default. It
demonstrates that Stage-1 temporal fine-tuning can reduce partial-window emissions, but
it cannot recover unrestricted natural sequences: the fast ASLLRP clips still lack
enough stable probes and retain classification/domain errors. A future natural stream
needs a temporal sequence model with explicit blank/boundary supervision, but does not
need to reuse the current slow whole-phrase Stage-2 live policy. Full evidence and the
run command are in
`artifacts/reports/stage1_v17_reel_emission_experiment_v1/README.md`. Twenty-five focused
tests, compilation, checkpoint/Core ML load smoke, and `git diff --check` pass.

## 2026-09-02 22:23 PST — Reel local-phrase gains do not transfer to fast ASLLRP clips

The user's newest completed webcam history at
`artifacts/reports/live_reel_stage1_v17/20260902_213908_704489/history.json` is genuinely
better under deliberately clearer articulation. Stage 2 was disabled. It produced five
nonempty FINISH sequences, including `I GO DOCTOR TOMORROW MORNING I FEEL SICK`, and the
five provisional `LESS` appearances never committed. This supports the Reel interaction
for the current signer, but expected labels were not recorded for every attempt and some
suspicious `CHILD` commits remain, so the session is not an accuracy benchmark.

A new development-only external check replayed all 12 vocabulary-covered ASLLRP
contiguous validation-cache clips that were excluded from the local phrase adapter. The
underlying rows are still `train_candidate` material, not sealed test data. With default
Reel timing and Stage 2/lips/speech disabled, exact sequence accuracy was 0/12 and total
WER was 20/24 = 83.33%. Only five glosses committed: four aligned correctly and one was a
wrong `WRITE`; deletion/under-emission dominated because these short, fast videos yielded
only two to four probes.

A model-only equal-segment comparison separated classification from live emission. On 24
ASLLRP gloss crops, original versus adapted top-1 was 10/24 versus 8/24 for the landmark
proposal and 13/24 versus 12/24 for the full verifier; full-verifier top-5 tied at 18/24.
Thus the local phrase/activity adaptation did not transfer to this small ASLLRP domain.
The failure combines signer/domain mismatch with a Reel policy too conservative for fast
clips. Full details are in
`artifacts/reports/live_reel_asllrp_external_adapted_v1/README.md`. Sixteen focused Reel
tests pass, and no protected test split was accessed.

## 2026-09-02 21:21 PST — Reel delay reduced and unsafe lip override disabled

Read-only diagnosis of the user's completed Reel session at
`artifacts/reports/live_reel_stage1_v17/20260902_205147_630595/history.json` confirmed
that the reported delay was real. Across 422.75 seconds, extraction completed 6,674
landmark observations (15.79 FPS), down from the preceding optimized session's 18.96
FPS, and discarded 4,638 of 12,180 total captured frames as stale. The 150 activity
candidates split evenly into 75 commits and 75 failures; 48 timed out. Successful
candidates required 1.00 seconds median / 1.88 seconds p90 from detected activity,
while failures occupied 2.37 seconds median. The cheap proposal itself remained fast
at 17.81 ms median, but full verification rose to 337.32 ms median and the proposal
and verifier disagreed 61/159 times.

The every-frame Apple face/body experiment caused avoidable contention. The Reel path
now detects Apple face/body on the training-matched every-eighth-frame schedule and
holds the last valid points continuously for display. Hands and MediaPipe lip markers
remain live. The optional `--dense-model-auxiliary` experiment remains available;
the isolated path is unchanged.

The 27-example GOOD/THANKYOU lip specialist was unsafe on this signer. Of 19 targeted
evaluations, it disagreed with the landmark proposal 11 times and with the original
full-verifier top class 15 times, commonly at essentially 1.0 confidence. The previous
0.999 threshold therefore did not calibrate it. Lip-based classification is now off by
default; `--lip-marker-verifier` explicitly enables it. Even then, lips may only break
a disagreement where both the landmark proposal and accepted full verifier already
select GOOD/THANKYOU. It cannot override agreement, a rejected verifier, or an
unrelated class. The raw model candidates are preserved in diagnostics.

A development-only timing sweep used five existing local phrase recordings with
unchanged confidence/margin gates and no Stage 2 or lip verifier. The old 0.62/0.14
second candidate/probe defaults scored 1/5 exact with five token edits and 1.07-second
median committed candidate duration. The selected 0.50/0.12 preset scored 3/5 exact
with four edits and 0.67 seconds median. A more aggressive 0.45/0.10 preset also scored
3/5 and four edits but added a false STOP, so it was rejected. This is fitted local
development evidence, not independent accuracy. The compact report is
`artifacts/reports/live_reel_stability_sweep_v1/README.md`; generated histories/videos
remain local. Thirty-nine focused tests, compilation, and `git diff --check` pass.

## 2026-09-02 20:09 PST — reel proposal and verifier decoupled from the display loop

The user's no-Stage-2 session at
`artifacts/reports/live_reel_stage1_v17/20260902_195337_949818/` confirmed a second,
more direct latency bug. Of 112 proposals in about 107 seconds, 83 fell through the
supposed landmark cascade into a full MobileCLIP classification; 65 stable proposals
then ran the same full classifier again synchronously on the UI thread. Proposals used
24.60 seconds and the second verifier passes used 18.21 seconds. The final candidate
contained only 21 observations across 1.91 seconds (10.49 observed FPS), and its
synchronous verifier froze the UI for 777 ms. Disabling Stage 2 therefore could not
fix the remaining lag.

`scripts/live_reel_stage1_v17.py` now keeps every provisional probe landmark-only and
schedules the full visual/hand verifier asynchronously only after two consistent cheap
proposals. The expensive verifier never runs inside the capture/display callback. One
verified hit now commits, replacing two repeated visual passes after the new two-hit
landmark stability gate. The default extraction rate is 20 FPS at a 640-pixel detector
input so the newest-frame display can refresh independently; full Stage 2 is now
opt-in with `--stage2-arbiter`. Lip landmarks remain enabled. Full nested JSON printing
is opt-in with `--verbose-predictions`, while complete structured results are still
written to the session history. Adjacent identical commits are suppressed to prevent
a transient proposal from turning one held sign into duplicate spoken glosses.

In a 16-second live-camera smoke with no person deliberately signing, the 13 landmark
proposals measured 14.3 ms median and 30.0 ms maximum with zero hand-image encoding,
versus 235.9 ms median and 501.4 ms maximum for the prior no-Stage-2 session. The smoke
was terminated externally and therefore has no valid display-FPS footer; the user must
confirm perceived camera smoothness in the actual window. Normal-speed saved-video
replays retained exact GOOD MORNING. HELLO HOW YOU produced the intended four commits
`HELLO HOW HOW YOU`; the new adjacent-duplicate guard collapses the duplicate HOW.
Twenty-two focused reel/lip/Stage-2 tests pass, compilation and `git diff --check`
pass. No original isolated source/model, sealed split, or Citizen test was touched.

## 2026-09-02 19:46 PST — first real reel-path session exposes duplicated visual inference

Read-only diagnosis of the user's first webcam run at
`artifacts/reports/live_reel_stage1_v17/20260902_194002_860837/` confirms genuine
throughput loss. During 96.96 seconds, the camera supplied approximately 28.09 FPS,
but the main loop completed only 20.86 landmark observations per second and discarded
700 stale frames. The reel path therefore lost about one quarter of captured frames;
this is not merely a display impression.

MediaPipe lips are not the primary bottleneck. A 240-frame replay of the recorded
640x360 session measured 3.11/3.49/4.84 ms median/p90/max for the 40-point lip tracker.
The architectural cost is duplicated visual inference. The 38 Stage-1 proposals used
4.40 seconds total; 27 subsequent full Stage-1 verifications used another 9.33 seconds
(277.0 ms median), including 6.89 seconds of MobileCLIP hand encoding. In parallel,
30 accepted Stage-2 windows used 9.15 seconds (221.1 ms median, 732.8 ms p90, 974.4 ms
maximum). Stage 2 attempted 78 windows in total because the current reel script feeds
it every elapsed-time window, including idle/background periods, rather than only an
active utterance. The Stage-1 verifier and Stage-2 arbiter also load separate instances
of the same hand-image encoder and can contend while the detector/display loop runs.

Prediction latency is additionally increased by policy: every stable Stage-1 proposal
runs a full verifier, and weak proposals require two verified hits. This protects
against transition fragments but can add two 0.2-0.7 second verifier passes after the
initial 0.62-second candidate. The isolated prototype feels smoother because it stops
landmark extraction while its one classification future is running and continues to
read/display camera frames; it does not run full Stage 2 every 1.067 seconds.

The live session also invalidates promotion based only on the five local replays. It
committed HELLO, YES, HOW, HELLO, LESS, YOU, HOW, CHILD, HELLO, HEAR, HOW, I, I, while
Stage 2 produced unstable sequences such as WHY COME and HELLO MORNING HOW. The next
fix should not tune the lip tracker. The minimal architectural correction is to keep
the fast Stage-1 landmark proposal and lip overlay live, remove continuous full CTC
from the display-critical path, and run full Stage-2 arbitration only for buffered
active signing/FINISH (or use the 11 ms landmark preview live). Full Stage-1 hand-image
verification should be reserved for genuinely ambiguous proposals rather than every
commit attempt. No source code, model, or dataset was changed, and no sealed/test split
was accessed.

## 2026-09-02 19:34 PST — separate reel path now uses visible lip markers and Stage-2 final arbitration

The accepted pause-delimited prototype `scripts/live_isolated_v17.py` and its models
remain unchanged. A separate `scripts/live_reel_stage1_v17.py` experiment now performs
frequent Stage-1 landmark proposals, invokes the full unified classifier as a verifier,
uses a two-hit commit lock for weak transition fragments, and feeds the already-trained
Stage-2 CTC model asynchronously as a final multi-sign sequence arbiter. Stage 2 never
replaces a one-gloss result unless a stable multi-gloss CTC sequence contains the
Stage-1 evidence, so isolated signs are not automatically expanded into phrases. For
utterances beyond the Stage-2 model's eight-window context, emissions leaving the
window are frozen into a prefix instead of silently losing the sentence beginning.

The new lip path is no longer display-only. `active/v17/lip_marker_v17.py` extracts 40
MediaPipe FaceMesh points from the outer and inner lip contours on every processed
frame, immediately discards RGB, and normalizes face position, scale, and roll before
computing shape and temporal-change features. The exact same 40 points are now drawn
on screen as white markers with black outer and inner contours. The tiny closed-pair
model at `artifacts/models/lip_marker_good_thankyou_phrase_crops_v17/model.npz` is only
allowed to choose GOOD versus THANKYOU after Stage 1 proposes one of that pair; it
cannot introduce an unrelated class. On permitted validation it scored 22/27 overall,
including 16/20 phrase crops and 6/7 Citizen clips. This helps when mouthing differs,
but deliberately does not claim that neutral-mouth GOOD and THANKYOU are separable by
lips alone.

The phrase/activity-adapted Stage-1 model is separate at
`artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth`.
Its selected validation results were 96.03% Citizen, 89.16% SemLex, 97.03% local
isolated, 66.41% equal phrase segments, and 69.79% activity crops. Its FP16 Core ML
export is 23.58 MiB, measured 8.64 ms median after warm-up, and had zero top-1
mismatches across 378 permitted validation archives. A more aggressive pair-targeted
checkpoint improved phrase/activity accuracy to 83.01/82.43% but regressed the SemLex
GOOD/THANKYOU pair to 38.1%; it is preserved as an experiment and is not the default.

Five normal-speed held-out local phrase replays now finish exactly through the hybrid
router: HELLO HOW YOU, GOOD MORNING, MY NAME, PLEASE HELP I, and THANKYOU FRIEND. In
the last THANKYOU FRIEND replay, the raw Stage-1 proposal was GOOD and the lip marker
specialist corrected it to THANKYOU at confidence 0.99999999; a later transition
fragment was still READ, while Stage 2 supplied `THANKYOU FRIEND FRIEND`, adjacent
duplicate collapse produced `THANKYOU FRIEND`, and FINISH selected the exact sequence.
This is useful replay evidence, not an independent signer or production-accuracy claim.
Fourteen focused reel/lip unit tests pass, Python compilation passes, and
`git diff --check` passes. No Citizen test or sealed split was accessed.

## 2026-09-02 14:34 PST — newest live Stage-2 session confirms utterance-stream over-emission

Read-only inspection of the user's newest webcam session,
`artifacts/reports/live_stage2_ctc_v17/20260902_142833_531729/history.json`, confirms
that the reported extra words are model hypotheses rather than a HUD-only artifact.
Examples include `HELLO -> HELLO LIKE -> HELLO LIKE WHY -> HELLO LIKE WHY WHO ->
HELLO LIKE WHY WHO MAYBE`, and a later stream revision from `HELLO` to
`NEED HELLO GOODBYE` after its third accepted window. The first attempted stream also
resolved as `HELLO HOW ASK`, rather than the expected familiar `HELLO HOW YOU` pattern.
Across the session, 39 windows were accepted and 11 rejected; accepted post-landmark
inference latency was 282.17 ms median, with hand-image embedding still dominant.

The behavior is partly the intended Stage-2 contract and partly a live-design/data
failure. The script treats every accepted 1.067-second window between RESET/FINISH as
another part of one open continuous utterance, gives each window eight CTC time slots,
and has no explicit one-sign endpoint or idle/no-sign class. Therefore a hold,
transition, partial sign, or incidental hand motion can extend or revise the entire
phrase hypothesis and can legitimately decode two or three nonblank tokens. Greedy CTC
collapse itself is operating as implemented; the unsuitable assumption is using this
continuous utterance decoder as the default reel-like single-sign interaction.

The selected UX direction is consequently a separate Stage-1 isolated-lock loop for
the reel behavior: frequent rolling complete-sign candidates, immediate provisional
labels, stability/debounce plus duplicate suppression before committing a chip, and
RESET/FINISH defining the sentence buffer. Stage 2 remains optional for explicit
continuous-utterance mode or later confirmation, not required for the fast default.
The current Stage-1 model was trained on completed clips, so raw frame-by-frame labels
must not be committed directly; an endpoint/stability gate is still required, and the
previous wrist-motion valley alone is not reliable enough. No source code, model, or
dataset was changed during this diagnosis, and no test or sealed split was accessed.

## 2026-09-02 14:18 PST — Stage-2 cascade is fast and safe for provisional/final routing, not hard early lock

The separate landmark Stage-2 preview retraining completed in 320.66 seconds on MPS.
The selected seed 9811 epoch 6 checkpoint is
`artifacts/models/stage2_v17_landmark_cascade_preview_v1/best_model.pth` (SHA-256
`b4cce35376858d774652f39665e985fde12f408e9db54cf383c8fcce4ed6d484`). It has
[ASLLRP contiguous phrase, local phrase, held-out ASLLRP segmented] validation edit
counts [18, 17, 47], so it is weaker than the full selector and is not a replacement.

A new validation-only cascade sweep selected minimum greedy nonblank emission
probability 0.978. It used the landmark preview for 3/12 ASLLRP phrase rows, 101/254
held-out ASLLRP contextual rows, and 29/97 local phrase rows. Relative to the accepted
full selector, per-domain edits changed 9->9, 43->42, and 6->6. Thus it provides 133/363
(36.6%) fast coverage without an observed domain regression on the selection data.
This same-validation selection is optimistic and must be confirmed on new independent
portrait recordings before promotion.

The combined landmark encoder and CTC preview was exported to the new
`artifacts/coreml/Stage2LandmarkCascadePreviewV17FP32.mlpackage`. Across 109 permitted
phrase validation archives Core ML had zero decode mismatches versus PyTorch, maximum
absolute logit difference 1.62e-5, and measured 11.56 ms median / 18.13 ms p90 after
warm-up on this Mac. The FP32 package is 39.90 MiB. This proves that a reel-like
provisional label can be computed quickly after landmark extraction.

Hard early locking remains unproven. Partial-window probes at 8/16/24/28/32 model
frames showed a false lock even at 0.999 confidence. Requiring the same strict
hypothesis extension across four consecutive probes removed observed false locks but
locked only 11 tokens in 9/109 rows and zero complete phrases before their final probe.
Accordingly, the supported architecture is immediate **provisional** landmark output
followed by permanent-chip/full-Stage-2 confirmation. It is not safe to speak or
permanently append every high-confidence partial prediction.

The genuine multi-gloss training cache contains only 44 unique sequences, 50 unique
directed bigrams, and 46/100 glosses in any multi-gloss sequence. Only six local phrase
identities have repeated coverage; most of the 38 ASLLRP multi-gloss sequences have a
single example. The 1,116 ASLLRP segmented train spans add contextual isolated evidence,
not 1,116 genuine transitions. Additional phrases are not needed to fix the current
timebase or demonstrate familiar phrases, but they are needed for useful hard-lock
coverage, unseen combinations, and research generalization. The existing 20-30 phrase,
10 train + 2 validation + 1 sealed native signer plan with three genuine performances
per phrase is a meaningful next pilot if it maximizes new bigrams and low-motion
pronoun/confusion transitions; it is not enough for a general 100-gloss continuous-ASL
claim. Simultaneous cameras are correlated views, not independent performances.

New source files are `train_stage_2_landmark_cascade_v17.py`,
`evaluate_stage_2_landmark_cascade_v17.py`,
`export_stage2_landmark_cascade_coreml_v17.py`, and
`evaluate_stage_2_early_lock_v17.py`. Existing isolated and motion-valley source files
were not edited for this experiment. Seven focused live Stage-2 tests pass and
`git diff --check` passes. Full findings are in
`artifacts/reports/stage2_v17_landmark_cascade_preview_v1/README.md`. No sealed or test
split was accessed.

The code, tests, ground-truth handoff, and compact reports were committed as
`e027f72` (`Add time-normalized Stage 2 cascade experiments`). Local webcam histories,
session videos, trained checkpoints, and Core ML packages remain uncommitted/ignored;
no personal recording was added to git.

## 2026-09-02 14:02 PST — motion-valley YOU errors are boundary-dependent; landmark CTC retraining started

The user's latest motion-valley session is
`artifacts/reports/live_motion_valley_v17/20260902_133345_224087/history.json`.
Read-only inspection confirms the classifier can recognize YOU, but the wrist-motion
boundary produces inconsistent pieces of repeated attempts. In the concentrated
318-336 second interval, clips labeled NEED, YOU, HE, NEED, NEED, THEY, NEED, NEED,
NEED, YES, YOU, YES ranged from 7 to 17 motion-trimmed observations. Elsewhere YOU
was correctly accepted many times, including high-confidence clips. Across all
post-250-second candidates in the confusion family there were 29 YOU, 12 NEED, 9
TELL, 8 THEY, and 2 UNDERSTAND clips; median motion-trimmed lengths varied from 13
frames for YOU to 19 for UNDERSTAND. This disproves a simple missing-YOU-class
explanation and supports the user's cutoff diagnosis. Low-motion lexical signs and
low-motion inter-sign transitions are structurally ambiguous under this trigger, so
further global threshold tuning is not the selected path.

A new, separate `active/v17/train_stage_2_landmark_cascade_v17.py` experiment was
added without modifying the isolated or motion-valley scripts. It trains a Stage-2
CTC preview from the landmark-token slice of the existing approved caches, reconstructs
the 612-D contract as 256 landmark tokens + 256 zero hand features + 100 frozen
landmark logits, and distills the accepted full multimodal selector while applying
supervised CTC. A 24-sample smoke completed on MPS and improved validation in one
epoch. The first full run reached a best [ASLLRP phrase, local phrase, ASLLRP
contextual] edit vector of [17, 17, 48] at epoch 5, then encountered a non-finite MPS
CTC batch before packaging. The new script now uses the project's existing bounded
non-finite-batch discard policy and the safer 5e-6 learning rate; this failed partial
run is not a promoted artifact. No test or sealed split was accessed.

## 2026-09-02 12:27 PST — elapsed-time Stage 2 stays 5/5 exact at live-like 15 FPS

The separate `scripts/live_stage2_ctc_v17.py` experiment now partitions model input
by 1.067 seconds of source time rather than by 32 successful detector calls. Webcam
capture is drained on a background thread and stale frames are discarded, so slow
feature work cannot build a seconds-long camera backlog. The fixed eight-window stop
was replaced by a rolling eight-window CTC context: emissions leaving the context are
locked into a prefix while the recent context remains revisable. Apple Vision face
features are now computed only at the trained every-eighth-observation interval;
MediaPipe FaceMesh supplies display-only moving lip landmarks on each processed frame.
It never enters model features. Seven focused unit tests pass, including elapsed-time
partitioning, CTC emission positions, rolling-prefix behavior, CTC score parity, and
stable-prefix speech.

Five genuine familiar-domain phrases were replayed at only 15 extracted observations
per second, close to the webcam's measured 14.47 FPS. All remained exact: GOOD MORNING,
HELLO HOW YOU, MY NAME, THANKYOU FRIEND, and TOMORROW SCHOOL GO. Most full model
windows contained 16 observed frames spanning one second and were resampled to 32,
instead of incorrectly spanning about two seconds. This confirms the timebase repair
for familiar recordings; it is not independent signer/generalization evidence.

The 16 accepted updates still had variable post-landmark cost: 488.70 ms median,
949.41 ms p90, and a 2045.51 ms maximum in this sequential replay. MobileCLIP hand
embedding remained dominant. Therefore time normalization addresses inability to sign
at a natural pace, while a separately trained landmark-first Stage-2 preview/gate is
still needed to target reel-like immediate feedback. Evidence is under
`artifacts/reports/live_stage2_ctc_v17_timebase15_eval_v1/`. No sealed or test split
was accessed.

## 2026-09-02 12:18 PST — reference reel is an isolated-lock UX, not evidence of streaming CTC

The user supplied the local 720x1280, 30 FPS, 17.62-second copy of the previously
linked Instagram reel at `/Users/frnzlo/Downloads/What if AI could bridge the gap
between sign language and spoken EnglishThis prototype uses Medi.mp4`. Frame-level
inspection shows a MediaPipe-style hand overlay, a transient single-word label, and a
separate row of committed word chips. The visible sequence is TECHNOLOGY, USE,
IMPROVE, LIFE, WAR, NOT; transient labels change during motion, but a word is appended
only after it persists. A deliberately separate FINISH sign then advances the UI from
Hand Tracking/Emotion Detection to LLM Interpretation and Voice Output and produces
“Let's use technology to improve lives, not war.” The reel itself does not expose its
source, weights, timing thresholds, evaluation set, or accuracy and therefore is not
evidence of a continuous CTC architecture.

Its useful target behavior is a fast isolated-sign lock/debounce loop with rolling
gloss chips and an explicit utterance terminator—not frame-by-frame translation and
not necessarily Stage 2. This matches the project's cascade/isolated direction more
closely than the first Stage-2 live prototype. The current Stage-2 experiment can
still offer coarticulation-aware correction, but must first fix its observed timebase
and eight-window ceiling. No dataset or sealed evaluation set was accessed.

## 2026-09-02 11:51 PST — first webcam Stage-2 session exposes a live-timebase mismatch

The user's first real webcam run of `live_stage2_ctc_v17.py` is preserved at
`artifacts/reports/live_stage2_ctc_v17/20260902_114604_186832/`. No code was changed
in response; this entry records the read-only diagnosis. The Stage-2 experiment does
not use the isolated path's landmark-first cascade. Every accepted Stage-2 window
encodes all valid crops among 16 temporal samples x left/right/union views, then runs
the frozen multimodal encoder and both CTC heads. In this session MobileCLIP hand
embedding measured 151.43 ms median, 378.61 ms p90, and 465.93 ms maximum. The full
post-landmark update measured 179.04/411.07/497.58 ms median/p90/max. The isolated
cascade can skip hand embedding when its landmark evidence is strong; the current
Stage-2 graph has no equivalent pre-hand decision point.

More importantly, the requested 30 processed FPS was not achieved. Across 44 full
windows, the median 32-frame wall-time span was 2.143 seconds (p90 2.583), equivalent
to only 14.47 processed FPS. The selected Stage-2 training/replay contract uses
32 source frames, and the exact local HELLO HOW YOU reference is 30 FPS / 3.567
seconds. Therefore the successful offline replay fed about 1.067 seconds per full
window, while the webcam fed about twice as much real motion into the same 32-frame
tensor. Signing more slowly compounds this mismatch rather than helping: a lexical
sign can occupy multiple CTC windows and be emitted twice.

After the final reset, the webcam hypothesis evolved HELLO -> HELLO HOW -> HELLO HOW
HOW -> HELLO HOW HOW YOU. FINISH received `HELLO HOW HOW YOU`; tiny Stage 3 rendered
`Hello, how are you?`, but the recognition buffer itself was not exact. Stable-prefix
speech adds another full update before speaking a new prefix, so at the measured rate
it adds roughly 2.1-2.6 seconds. YOU was delayed further by an intervening rejected
zero-hand window. Nine accepted windows in the full session had at most ten detected
hand frames, and eight had hand presence below 0.2; these sparse windows can consume
context or preserve unstable hypotheses. The user reset eleven times.

The fixed eight-window checkpoint limit is also a hard UX ceiling: once eight accepted
windows are stored, the prototype clears incoming frame buffers and stops extracting
until RESET or FINISH. This is unsuitable for an open-ended conversational live loop.
The priority is now timebase correction and streaming-state design, not threshold
tuning: decouple capture from expensive extraction, form model windows by elapsed time
at the training-equivalent rate, avoid making slower signing the workaround, and then
measure whether a display-only lightweight lip tracker can retain continuous mouth
motion while Apple Vision face features remain sampled at their trained interval. A
Stage-2 cascade cannot be enabled by toggling the existing isolated cascade; it would
require a separately validated landmark-only preview/gate or a conditional Stage-2
encoder. No sealed test or held-out dataset was accessed.

## 2026-09-02 11:42 PST — separate live v17 Stage-2 CTC experiment passes familiar phrases

The accepted v17 phrase-agnostic general CTC selector was found in the existing logs
and connected to a new, separate `scripts/live_stage2_ctc_v17.py` path. The working
`live_isolated_v17.py`, failed fixed-overlap experiment, and motion-valley experiment
remain unchanged. The new path uses the parity-validated Core ML frozen multimodal
encoder and primary/specialist CTC heads, then mirrors the saved general selector's
90/10 blend, +0.30 blank calibration, and exact specialist CTC path-score rule in
NumPy. It processes non-overlapping 32-source-frame windows, matching Stage-2 training,
and supports at most the checkpoint's eight-window context. It does not use beam search
or the stride-8 overlap previously shown to degrade accuracy.

The live camera continues extracting and drawing face landmarks every processed frame,
while only every eighth face sample enters model features to match the v17 training
contract. The UI updates the current greedy CTC hypothesis after each completed window.
Only the prefix surviving a following update is eligible for immediate gloss speech;
FINISH closes a 4-31-frame tail, naturalizes the latest full hypothesis, clears queued
gloss speech, and speaks the finished sentence. RESET clears only current visible
state while keeping JSON/video evidence. The frozen encoder and both CTC heads run in
Core ML; no large PyTorch research graph is loaded in the live path.

Five genuine, vocabulary-covered local development/reference recordings were replayed
through the full path. All five final hypotheses were exact: GOOD MORNING, HELLO HOW
YOU, MY NAME, THANKYOU FRIEND, and TOMORROW SCHOOL GO. This is familiar-domain
execution/development evidence, not independent accuracy: those phrases are inside the
model's established local development domain. Across 16 accepted updates, median/p90/
maximum post-landmark latency was 235.77/291.89/351.34 ms. The CTC selector itself took
roughly 3-5 ms; MobileCLIP hand-image embedding dominated. At 30 processed FPS the
first full-window update still requires about 1.07 seconds of signing context before
that inference cost. GOOD MORNING first displayed an unstable THANKYOU, then revised
to GOOD and finally GOOD MORNING; stable-prefix speech correctly withheld the unstable
first guess.

A separate EOF-FINISH smoke on MY NAME logged one completed utterance and rendered
`My name.` through the deterministic literal path. Four focused tests cover exact
NumPy/PyTorch CTC-score parity, repeat collapse, stable-prefix speech, and isolated
defaults. Full evidence is under
`artifacts/reports/live_stage2_ctc_v17_local_phrase_eval_v1/` and the FINISH smoke is
under `artifacts/reports/live_stage2_ctc_v17_finish_smoke_v1/`. No sealed test,
Citizen test, SemLex test, local test, 2M-Flores devtest, or consumed RIT row was
accessed.

## 2026-09-02 11:08 PST — motion-valley live trigger made more responsive

After the user's first real live motion-valley session, the experimental defaults were
changed from hybrid, motion ≤0.008 for 0.16 seconds to cascade, motion ≤0.010 for 0.12
seconds. The latest session had six clips lasting 1.38–3.12 seconds and 238–489 ms
classification latency; it accepted HELLO twice and HOW once. The higher motion cutoff
is less vulnerable to small wrist jitter, the shorter hold saves one processed frame,
and cascade can skip expensive hand-image encoding when landmark evidence is strong.
Uncertain signs still use the unified fallback. This explicitly trades some boundary
precision for faster response and may split internal holds; the neutral/pause path is
unchanged. The user must validate the new live behavior before any accuracy claim.

## 2026-09-02 10:46 PST — no-pause path fails the nine-local-phrase accuracy gate

The unchanged 1.2-second/0.2-second-stride streaming experiment was evaluated on all
nine project-owned representative phrase recordings from the genuine local-motion
reference report. These are development/train-source diagnostics, not a new
signer-disjoint test. Citizen validation/test, SemLex test, and the local test partition
were not accessed.

All nine runs completed and wrote independent history/video evidence. Across 97 windows,
69 passed the existing Stage-1 gates. Median classification latency was 548.88 ms and
p90 was 1,334.46 ms. No stale-window backlog accumulated. Nevertheless, exact sequence
accuracy was 0/9 overall and 0/5 among the phrases entirely covered by the locked 100.
Outputs were: GOOD_MORNING -> `EAT MORNING`; HELLO_HOW_YOU -> `HELLO HOW`;
I_WANT_FOOD -> `UNDERSTAND WANT`; MY_NAME -> `NAME`; PLEASE_HELP_ME ->
`STOP HELP I`; SORRY_I_LATE -> `SORRY`; THANKYOU_FRIEND -> `BAD NAME DOCTOR`;
TOMORROW_SCHOOL_GO -> `TOMORROW LESS`; YESTERDAY_TEACHER_MEET -> `ANSWER`.

Four prompts are structurally impossible to recover fully with the current vocabulary:
FOOD, ME, LATE, TEACHER, and MEET are absent class labels. More importantly, even the
five fully covered prompts fail because fixed windows include partial signs/transitions
and short signs may not survive the two-window agreement gate. The mechanics pass but
the accuracy gate fails. Do not replace the working pause-delimited prototype or simply
accept every window; the next serious no-pause experiment needs a learned boundary or
framewise/Stage-2 sequence model. Full evidence is under
`artifacts/reports/live_streaming_v17_local_phrase_eval_v1/`.

## 2026-09-02 10:36 PST — separate no-pause overlapping-window live experiment added

The working pause-delimited path in `scripts/live_isolated_v17.py` remains intact. A
separate `scripts/live_streaming_v17.py` experiment now continuously extracts Apple
Vision landmarks, classifies the newest 1.2-second window at most every 0.2 seconds,
and never queues stale windows. It defaults to the fast cascade, keeps landmark/lip
sampling aligned with Stage-1 training, preserves the white-point/black-bone display,
records low-resolution video and full JSON evidence, and retains RESET, FINISH, local
tiny Stage 3, and native speech. It does not require a neutral pose or explicit pause.

Two consecutive accepted windows must agree before a gloss enters the visible buffer.
The same prediction run emits once; a different stable label or two rejected windows
rearms it. This suppresses overlap duplicates but deliberately cannot distinguish two
adjacent repetitions of the same sign. The approach is overlapping-window isolated
Stage 1, not CTC Stage 2, so windows can still contain transition motion or pieces of
neighboring signs and no continuous-accuracy claim is justified yet.

Four focused stabilizer/default tests and direct CLI/compile checks pass. An end-to-end
smoke used only the quarantined Citizen training clip
`6226330398612929-W.H.A.T.mp4`; Citizen validation/test were not accessed. It processed
six windows, emitted one stabilized gloss, wrote history/video, and measured roughly
341–406 ms per fallback classification without building a backlog. It stabilized as
TELL rather than the quarantined folder label WHAT. This is successful execution and
latency evidence, not accuracy evidence; the next meaningful check is a labeled live
session of naturally connected locked-100 signs.

## 2026-09-02 09:46 PST — explicit RESET/FINISH utterance UX and local Ollama rephrasing implemented

The live laptop prototype now implements the preferred endpoint-delimited path:
isolated Stage-1 predictions accumulate in a visible bottom gloss buffer; `RESET`/`R`
clears all visible utterance state without deleting predictions, video, or events; and
`FINISH`/`F` waits for an in-flight classifier, closes a genuinely active sign if one
exists, consumes the accepted non-UNKNOWN gloss buffer, and sends that one utterance to
the installed `llama3.2:1b`. A reset epoch prevents an old classifier or Ollama result
from reappearing or speaking after RESET, although the result remains in the audit log.
This is still pause/endpoint-delimited Stage 1 plus text rendering, not continuous
recognition and not a replacement accuracy claim for Stage 2.

Ollama uses the local `/api/generate` endpoint on a separate single-worker queue and is
warmed asynchronously with a 30-minute keep-alive. The prompt requests one short
meaning-preserving sentence and JSON containing the exact input gloss audit. The result
is accepted only when `used_glosses` exactly matches the input sequence and the sentence
is nonempty/bounded; API, JSON, or gloss-audit failure uses the existing reviewed
Stage-3 template when available and otherwise literal gloss-preserving English. This
guard detects dropped/reordered/replaced glosses but cannot prove that arbitrary natural
language is semantically faithful, so all prompt/response/fallback evidence is retained
and LLM output must not be treated as linguistic ground truth.

Native macOS speech is now queued rather than stop/restarted: each accepted gloss is
spoken after its HUD update and the finished sentence is queued behind those glosses.
The finished gloss sequence remains visible while its final sentence is displayed or
spoken. `history.json` format version 2 records predictions across resets, buffer/display
epochs, reset/finish events, raw utterance glosses, prompt, raw model response, fallback
decision, model latency, final sentence, and speech queue/start times. The low-resolution
session video behavior is unchanged.

A real local `llama3.2:1b` integration probe returned `I feel sick.` for
`[I, FEEL, SICK]` with an exact gloss audit. Cold asynchronous warm-up was 5.11 seconds
and the subsequent generation was 2.47 seconds on this Mac; that generation is outside
the camera/extractor thread. Eleven focused live-script tests pass, including control
hit-testing and accepted/rejected Ollama audits. A no-display/no-speech/no-Ollama replay
of a quarantined Citizen training clip completed end to end and wrote the version-2
history/video; its rejected WHAT prediction is smoke evidence only, not accuracy
evidence. No Citizen validation or sealed test data was accessed in this change.

## 2026-09-02 09:20 PST — live SICK/FATHER confusion is a one-hand variant and local coverage gap

The latest live session `20260902_090932_943567` contains 47 predictions, including a
long user-described SICK/FATHER/MOTHER/FEEL comparison sequence. Exact intended labels
were not stored per event, so the session must not be silently relabeled for training.
The observed outputs nevertheless form a coherent learned confusion: SICK, FATHER,
WHY, KNOW, and THINK repeatedly occupy the top candidates. One early attempt was
accepted as SICK; later attempts variously put SICK first but reject it, or rank FATHER
or WHY above it. Hand/face coverage is generally present and the cascade calls the
unified fallback on the ambiguous clips, so this is not merely a missing-hand or
boundary failure.

The training-cache audit explains the domain failure. Current Citizen train/validation
support for SICK, FATHER, MOTHER, and FEEL is respectively 14/4, 14/3, 14/3, and 15/4,
and the unified head gets all fourteen focused Citizen validation clips correct.
SemLex train/validation support is 23/18, 10/6, 14/6, and 16/12, with only one SICK
validation error. In contrast, the local cache has **zero SICK** train or validation
examples but 85/18 FATHER, 135/26 MOTHER, and 153/27 FEEL. Moreover, 92.9% of Citizen
and 92.0% of SemLex SICK landmark archives show both hands, whereas the user's SICK is
the one-handed reduction and the FATHER/MOTHER/FEEL training examples are overwhelmingly
one-handed. This is a domain/lexical-reduction coverage gap, not a class-index mismatch.

Three already-downloaded PopSign SICK audit clips are genuinely one-handed. The
landmark primary predicts two as SICK with scores 0.829 and 0.889 and misses one as
SLEEP at 0.268. They are useful reviewed supplement candidates, but three clips alone
are not enough for durable adaptation and do not change PopSign's non-primary status.
The current Citizen-only mouth and lower-face teachers are also not safe fixes for this
set: mouth validation is SICK 1/4, FATHER 2/3, MOTHER 3/3, FEEL 1/4; lower-face is
SICK 0/4, FATHER 2/3, MOTHER 1/3, FEEL 1/4. Do not activate them on this confusion
without retraining and a focused gate.

The live cascade itself demonstrates the intended UX advantage but also a transfer
gap. Across 44 cascade predictions, 28 (63.6%) invoked the unified fallback versus
11.4% on Citizen validation. Primary-only median post-clip work was 21.75 ms, fallback
median was 276.65 ms, and actual within-clip processing cadence was 14.69 FPS median.
The preferred prototype architecture is therefore pause/endpoint-delimited Stage 1 ->
token accumulation -> explicit FINISH -> Stage 3/speech, with Stage 2 retained as an
optional research/continuous-sign path rather than the default UX. This remains a
controlled-signing interface: without separable endpoints it cannot decode fully
coarticulated continuous signing.

Retraining is warranted, beginning with reviewed one-handed SICK plus balanced hard
negatives FATHER, MOTHER, FEEL, WHY, KNOW, and THINK, while retaining two-handed SICK.
Handshape and hand-to-face location are the primary signal. Detailed Apple Vision lip/
face motion should be a separate supplementary expert, promoted only if it improves the
focused confusion gate without reducing all-class signer-disjoint validation. The
current landmark augmentation mirrors, rotates, scales, warps time, and drops a few
random nodes, but it never models the complete one-hand SICK reduction. A targeted
label-preserving augmentation can use the existing two-handed SICK clips: retain the
hand closest to the face and mask the lower/stomach hand on only a subset of repeats,
while keeping the unmodified two-hand form in training. This provides substantially
more one-hand evidence than the three real PopSign candidates without fabricating
handshape or motion. Before
using new live captures for supervision, the collector must save intended gloss, exact
v17 tensors, hand crops, detailed lip landmarks, and event boundaries; the current
low-resolution session MP4 is reference evidence, not safe training input. Full numeric
evidence is recorded in
`artifacts/reports/live_isolated_v17_sick_confusion_audit_v1/report.json`. No sealed
test split was accessed.

## 2026-09-02 09:05 PST — default/cascade live A/B mode added as clickable controls

The laptop isolated-sign UI now exposes two clickable controls without replacing the
default. `DEFAULT` retains the existing unified-first hybrid. `CASCADE` runs the
13.29-MiB `Stage1OrientationV17` landmark model first and invokes the unified
landmark+hand model only when the primary softmax score is below the validation-selected
0.70 threshold. Existing targeted GOOD/THANKYOU real-pixel visual reranking remains
available, as does the YOU/NEED landmark reranker after a unified fallback. The selected
mode is snapshotted at classification submission, so clicking during an in-flight
classification safely changes the next sign. `--mode cascade` starts directly in the
experimental mode, while the ordinary command still starts in `hybrid`/`DEFAULT`.

The cascade avoids hand-crop extraction and MobileCLIP2 encoding on confident primary
clips rather than merely running both models and choosing afterward. A real saved
Citizen validation THANKYOU replay produced the correct accepted gloss with primary
score 0.9039, no unified fallback, zero hand-image encoding, and the expected targeted
GOOD/THANKYOU visual reranker. Forcing the threshold to 1.0 on the same clip exercised
the fallback and remained correct, with `cascade_fallback_used=true` and 219.34 ms of
hand-image encoding. The ordinary-threshold and forced-fallback histories are under
`artifacts/reports/live_isolated_v17_cascade_smoke/20260902_090447_087648/` and
`artifacts/reports/live_isolated_v17_cascade_smoke/20260902_090521_494952/`.
A second ordinary-threshold Citizen validation replay recognized HELLO correctly with
primary score 0.9129, no hand or visual fallback, 16.23 ms classifier work, and 25.38 ms
total post-clip processing. Its history is under
`artifacts/reports/live_isolated_v17_cascade_smoke/20260902_090730_513829/`. These
single-clip timings are implementation smoke evidence, not an end-to-end live latency
distribution or accuracy estimate.

MediaPipe Face Mesh was not added. Apple Vision already exposes fuller outer/inner-lip
regions, but the locked v17 schema intentionally samples four mouth anchors among its
15 face nodes. Adding more Apple or MediaPipe points to the classifier would change the
input schema and require training a new face-motion branch; untrained extra points do
not improve accuracy. For the present prototype, every-frame Apple lip anchors provide
continuous non-RGB tracking/display while the already-trained genuine mouth/lower-face
pixel teachers provide targeted supplementary evidence. The previously mentioned
"fresh signer-disjoint" set means only a small untouched portrait laptop-camera
transfer gate, not another large training collection; the existing signer-disjoint
corpora remain the basis for model fitting and offline validation.
Eight live-prototype tests plus all seventeen extractor tests pass (25 total), including
button hit-testing. Both cascade primary-only and forced-fallback saved-video paths pass;
Python compilation, report JSON parsing, CLI help, and `git diff --check` pass.

## 2026-09-02 08:56 PST — cascade measured; continuous lip landmarks enabled without changing Stage 1 inputs

A Citizen validation-only cascade study measured the lightweight landmark/orientation
model at 362/378 (95.77%) and the unified landmark+hand model at 364/378 (96.30%). A
landmark-primary score threshold of 0.70 invoked the unified fallback for 43/378 clips
(11.38%) and matched the unified model's 364/378 result; the existing targeted
GOOD/THANKYOU visual reranker raised the simulated result to 366/378 (96.83%). The
two-model oracle ceiling was 367/378 (97.09%). The threshold was selected on the same
validation split and some primary errors are high-confidence, so the cascade was not
made the default. It first needs a fresh portrait, signer-disjoint development gate.
No Citizen test or other sealed split was accessed.

An extraction benchmark on 90 frames of one Citizen validation video, with a
same-aspect maximum side of 720 pixels and body every eight frames, measured hands plus
sparse face at 8.06 ms median / 22.60 ms p90 and hands plus face every frame at 18.44 ms
median / 26.70 ms p90. The latter stays below the 33.33 ms per-frame 30-FPS budget on
this Mac; it is not an iPhone or sustained-thermal claim.

The live prototype now requests genuine face landmarks on every processed frame, so
lip movement remains continuously visible without requiring RGB display. A new
`face_for_features` observation flag keeps only every eighth face sample in Stage 1's
landmark tensor, preserving the sparse training-time input contract while allowing the
UI and quality tracking to use all intervening face/lip observations. Real RGB remains
available only to the targeted visual tie-break when invoked. The measurement and
decision record is
`artifacts/reports/live_isolated_v17_fast_cascade_validation_v1/report.json`.
Seven live-prototype tests plus all seventeen v17 extractor tests pass (24 total),
including the new regression test that every detected face can reach display while
only scheduled faces reach Stage 1. Python compilation and `git diff --check` pass.

## 2026-09-02 08:47 PST — reel architecture reviewed; fast cascade recommended, not implemented

The referenced Instagram reel `DXJ0G8ADPc1` publicly describes MediaPipe tracking 21
points per hand and 468 face points, sign recognition/token accumulation, a local LLM
for emotional/contextual English rewriting, and ElevenLabs speech. It publishes no
vocabulary size, signer-disjoint split, held-out recognition accuracy, WER, end-to-end
latency, or sustained-device measurements. Its apparent demo accuracy/speed must not be
treated as benchmark evidence; it may be a small scripted vocabulary and signer-specific
demonstration. The LLM and speech layers do not improve the upstream gloss recognizer.

The comparable prototype does not require replacing v17. The recommended next accuracy/
speed experiment is a cascade: `Stage1OrientationV17` landmark Core ML as the immediate
primary, followed only for low-score/low-margin or predeclared hard-confusion cases by
the unified hand-RGB model and targeted face/landmark specialists. The existing primary
package is 13.29 MiB, has 95.77% Citizen validation top-1, exact 378/378 Core ML/PyTorch
top-1 parity, and a recorded 16.31 ms median / 21.67 ms p90 classifier-only latency on
this Mac. The unified student has 96.30% Citizen validation top-1 but adds the 43 MiB
MobileCLIP image encoder and per-crop work. A new local probe observed 8.68 ms median /
12.94 ms p90 for the orientation package, but the canonical recorded benchmark remains
the reported figure. Neither figure includes camera extraction, boundary time, or
iPhone thermals. No sealed test split was accessed.

A separate explicit FINISH gesture is viable later as a control channel: it commits the
accumulated gloss buffer to Stage 3/LLM and speech, and should not be part of the 100
lexical outputs. It only marks utterance end; it does not segment adjacent signs inside
a continuous phrase. Without pauses or per-sign commit gestures, Stage 2/online temporal
segmentation is still required. Per the user's priority, no finish gesture, LLM, cloud
voice, or cascade code was added in this discussion turn; recognition accuracy and
camera-to-gloss latency remain the next gate.

## 2026-09-02 08:39 PST — YOU/NEED specialist, live scheduling, synchronized speech

The newest live history `20260902_082833_471330` contains seven accepted events. Two
intended YOU attempts were strongly classified NEED with YOU second (NEED 0.960 versus
YOU 0.007, then NEED 0.935 versus YOU 0.022), while another performance was strongly
YOU (0.953 versus NEED 0.007). The recorded frames show the difficult domain case: the
straight index points nearly into the camera, so 2-D foreshortening resembles NEED's
bent-index silhouette. This is not a label-map error.

That session also revealed the nominal 30-FPS loop was only observing about 12-13 FPS.
When the loop was even slightly late, its deadline advanced a full interval from the
current time and systematically skipped the next camera frame. The deadline now catches
up without adding that extra interval. Apple Vision is also paused while the post-clip
Core ML classifier owns the hardware; previously concurrent Vision submissions caused
hand-encoder latency to spike from the roughly 0.2-0.4-second offline range to 1.2-2.3
seconds, with total live classification reaching 2.65 seconds. New histories record
the actually observed processing FPS. A fresh interactive session is still required to
measure sustained live FPS and confirm the contention fix on the camera.

Default hybrid inference now invokes the existing selected landmark-only Stage-1 model
only when unified Stage 1's top two are exactly YOU and NEED, using it to rerank only
that pair. This specialist is appropriate for the user's straight-index distinction and
keeps all other classes unchanged. The landmark checkpoint and the final targeted rule
both retain 4/4 YOU and 3/3 NEED on Citizen validation. This seven-clip check is a small
development gate, not a new accuracy estimate; no sealed test data was accessed. A raw
YOU saved-video smoke remains correct and accepted at 29.289 observed FPS with 351.9 ms
post-clip classification.

Speech no longer launches a new `say` process before the HUD is painted. One native
`NSSpeechSynthesizer` is retained in memory; the accepted word is drawn, `waitKey`
commits that frame, and the same result then starts speech. Any older utterance is
stopped rather than queued. The main overlay now draws black hand/body bones and white
hand/body/face landmark points. Six live-prototype tests, including an exact pixel-color
rendering check, plus all seventeen extractor tests pass (23 total); Python compilation,
the native speech API probe, and `git diff --check` also pass.

## 2026-09-02 08:26 PST — live/Stage-1 mismatch fixed; fast-first hybrid gated

The user's first live sessions exposed systematic `THANKYOU` predictions for intended
GOOD/HUNGRY and a head scratch accepted as HOME. The principal implementation mismatch
was concrete: body/face requests were scheduled on the global camera frame counter and
then filtered a second time by each clip's unrelated local frame index. Seven of eight
recorded live clips consequently had zero shoulder coverage, while only 19.0% of 378
Citizen validation archives have zero shoulder coverage. The second filter is removed;
any globally scheduled auxiliary detection is retained. A regression test now proves a
body frame at a nonzero clip offset survives and selects shoulder-width normalization.

Live defaults now restore the training-side temporal/image contract where practical:
30 processed FPS, fixed body/face interval 8, 1280-pixel real RGB crop frames, and a
same-aspect 720-pixel Vision detection copy. Face detection is no longer run on every
frame. Clips are trimmed to their moving interval before the 32-frame resample, and the
neutral close delay is 0.20 seconds instead of 0.55 seconds. The 720-pixel detector was
retained because replaying the 640-pixel reference recordings lost the hand detections;
speed is obtained from sparse auxiliary requests and fast-first classification, not by
sacrificing the primary hand signal.

Default mode is now `hybrid`: selected unified landmark+hand Core ML runs first. Frozen
real-pixel mouth/lower-face teachers run only when the fast model's top two are exactly
GOOD and THANKYOU, and only rerank that pair. HUNGRY and all other classes remain
hand/body-led. On the existing Citizen validation logits, the unified model got 9/11
GOOD/THANKYOU/HUNGRY clips correct (GOOD 4/4, THANKYOU 1/3, HUNGRY 4/4); the targeted
tie-break corrected both THANKYOU errors while retaining GOOD 4/4 and HUNGRY 4/4.
This 11-clip development check is not a new accuracy estimate. No sealed test data was
accessed.

Protective prototype rejection is active by default: score 0.25, margin 0.08, maximum
2.5-second motion interval, and an explicit insufficient-hand-evidence rejection. A
rejected result is `UNKNOWN` and is never spoken. Pair-reranked clips use combined fast
GOOD+THANKYOU evidence against the third class for this gate, rather than applying a
threshold across incompatible logit scales. This remains a heuristic closed-set gate;
reliable open-set rejection still requires independently held-out nonsign/background
data and likely an explicit OTHER/no-sign training class.

Fresh saved-video smoke results at the live 720/1280 settings are correct and accepted:
GOOD in 660.4 ms, THANKYOU in 752.2 ms, and HUNGRY in 271.2 ms after clip close. With
the 0.20-second neutral close, these correspond to approximately 0.86, 0.95, and 0.47
seconds before speech launch on this Mac, excluding ordinary camera scheduling jitter.
The user's saved head-scratch reference is now `UNKNOWN` for insufficient hand evidence
instead of HOME. The low-resolution 640-pixel session recordings are references rather
than exact extractor replays; the user's three earlier ambiguous sign segments also lost
hand evidence when re-extracted from those recordings, so a fresh live signer retest is
still required before claiming the reported mistakes are solved.

`test/test_live_isolated_v17.py` now also covers motion trimming and global/local
auxiliary scheduling. Its five tests plus all seventeen focused v17 extractor tests pass
(22 total); Python compilation, CLI help, and `git diff --check` pass. The usage guide
documents hybrid behavior, rejection, timing, and the extraction contract.

## 2026-09-02 07:56 PST — laptop live isolated 100-gloss prototype implemented and saved-video gated

`scripts/live_isolated_v17.py` is now the runnable laptop prototype. It uses the
locked checkpoint label map, built-in/OpenCV camera input, Apple Vision hands+face on
each processed frame and body at a time-adjusted auxiliary interval, one shared
detection pass for segmentation/landmarks/real-pixel crops, a neutral-return plus
low-motion automatic boundary, top-three model scores and margin, hand/face/motion
quality, boundary progress, accepted history, nonblocking macOS `say`, and a
single-worker classification queue. It saves an atomic `history.json` after every
prediction and a 640-pixel-wide 15-FPS MP4 without aspect distortion. Live frame
history is bounded to five seconds so long camera sessions do not retain unbounded
full-resolution frames. Saved `--video` development files are intentionally treated
as one isolated clip and classified once at EOF; only the webcam path uses automatic
segmentation.

The default `--mode lip-aware` reproduces the frozen per-sample-zscore four-stream
teacher weights: 0.30 landmark, 0.15 mouth, 0.35 lower face, and 0.20 hand. Both
visual views use their actual face-aligned pixels. Their identical frozen Auto-AVSR
frontend is shared in memory and receives mouth+lower-face as one two-view batch;
this retained the exact `HELLO` output scores while reducing the post-clip latency
from 1,140.2 ms to 736.8 ms on the same run setup. `--mode fast` uses the selected
unified landmark+hand Core ML model and therefore exposes only the four lip
landmarks, not full visual lip reading. Core ML/MPS lazy compilation is paid before
the camera opens. The initial un-warmed 9.07-second measurement is not representative
and is retained only as an artifact; all reported runtime measurements below are
after explicit warm-up.

Two Citizen signer-disjoint **validation** videos passed the saved-video gate in both
architectures. Final representative results are: lip-aware `HELLO -> HELLO`, score
0.5801, margin 0.5141, mouth/lower validity 100%, 736.8 ms after clip close; lip-aware
`THANKYOU -> THANKYOU`, score 0.5460, margin 0.5164, both visual views 100% valid,
755.6 ms before the exact batching optimization; fast `HELLO -> HELLO`, score 0.8934,
margin 0.8893, 375.4 ms; and fast `THANKYOU -> THANKYOU`, score 0.8329, margin
0.8180, 177.4 ms. These are pipeline smoke cases, not a new accuracy estimate. The
low-resolution reference output was probed as 640x480, 15 FPS, 25 frames / 1.667 s
for the final `HELLO` run. Final lip-aware evidence is under
`artifacts/reports/live_isolated_v17_lip_validation/20260902_075549_892643/`; the
latest fast evidence is under
`artifacts/reports/live_isolated_v17_fast_validation/20260902_075436_955044/` for
`HELLO` and `20260902_075342_952405/` for `THANKYOU`.

`test/test_live_isolated_v17.py` covers motion-to-rest emission, EOF fallback, and
the critical rule that a static hold away from the learned neutral pose must not end
a sign. The three new tests plus all seventeen focused v17 extractor tests pass (20
total); Python compilation and targeted `git diff --check` pass. Usage and explicit
scope/score caveats are in `artifacts/reports/live_isolated_v17/README.md`. The
MacBook camera index 0 opens successfully and returned one 1920x1080 frame. The live
UI has not been signer-tested in this noninteractive run, so the next gate is a user
session with neutral -> one normal-speed sign -> neutral.
No Citizen test, other sealed split, Stage 2, or naturalization model was accessed.

Research basis: Apple's live Vision examples support per-frame pixel-buffer requests
with explicit prediction backpressure; Apple also provides sequence request handling
and live face tracking. The 2024 EMNLP online CSLR work supports isolated-dictionary
recognizers as an online baseline but identifies the offline-CTC/short-window mismatch
that prevents calling this continuous recognition. Manual/nonmanual fusion evidence
supports retaining mouth/face input. Raw neural softmax is not generally calibrated,
so the HUD and JSON deliberately call these values `model_score`, not confidence.
Primary references: Apple Vision hand pose, live object recognition, gesture sample,
and face tracking documentation; Sincan et al., EMNLP 2024, *Towards Online
Continuous Sign Language Recognition and Translation*; Gueuwou et al., LREC 2020,
*Sign Language Recognition Using Neural Network*; and Guo et al., ICML 2017,
*On Calibration of Modern Neural Networks*.

## 2026-08-26 23:07 PST — live-camera v17 inference integrated into the Flutter app

The original Flutter design at
`/Users/frnzlo/Documents/machine_learning/mobile_app/slt_mobile_app` is now a
functional iOS-first record/stop/translate app rather than a UI mock-up. It uses the
official Flutter `camera 0.12.0+2` plugin, defaults to the front camera, records a
complete clip without audio, and sends the saved file through the proven Apple Vision
v17 orientation/aspect-ratio-safe extractor. The locked Stage-2 Core ML CTC model
naturally returns one gloss for a single recognized sign or multiple glosses for a
recognized phrase. The bounded Stage-3 renderer exposes the gloss sequence, reviewed
English when an exact template exists, and the literal gloss-preserving fallback
otherwise. The UI supports portrait and landscape layouts and includes clear status,
failure, retry, and conversation actions. Android native inference remains untouched
and explicitly unavailable.

A persistent Settings switch enables device benchmarking; it is off by default and
normal users see no benchmark panel. An enabled run performs one extraction, five
warmups, and 20 timed Stage-2 inferences, then shows extraction time, median/p90 model
time, before/after resident memory, and thermal state. It writes and shares an atomic
JSON report pinned to the Stage-2 candidate/checkpoint/package/vocabulary hashes and
Stage-3 manifest hash. Simulator reports set
`hardwarePerformanceClaim=false` and `thermalsInterpretable=false`; physical-device
runs identify themselves separately. Normal inference skips the five benchmark
warmups. Camera captures are marked end-to-end camera-to-gloss evidence, while no
physical-iPhone performance or accuracy result is claimed until the app is run on the
phone.

The app bundles exact copies of the three selected Core ML packages and exact
manifests. Manifest SHA-256 values are unchanged: vocabulary
`3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885eab7cf`,
Stage-2 mobile contract
`342101de35b0c3065d730b44172e112b348281234b6e870f421ffd069c32adfa`,
and Stage-3 naturalizer
`68c7ce67632f66ee70fa3b3d36eb8df33ad72dc674edbf3b720e93c1240f84a6`.
The copied Stage-2 Swift runtime differs from the proven benchmark source only by a
configurable warmup count.

Validation passed: `flutter analyze` has zero issues, both focused Flutter tests
pass, both plist files lint cleanly, and the unsigned iOS Release build succeeds for
arm64 with iOS 17.0 minimum and bundle ID `com.kokoab.sltMobileApp`. The resulting
`Runner.app` is 126.2 MB and contains exactly the three compiled model bundles plus
the three pinned manifests. Large local model packages are ignored by the mobile
app's `.gitignore`. No Citizen, SemLex, local, ASLLRP, or 2M-Flores test split was
accessed.

## 2026-08-10 09:25 PST — Streaming MoViNet baseline completed; Kaggle failure reconciled

The eight private train/validation multipart datasets and the private offline Model
Garden wheel dataset are all complete and `ready` on Kaggle. The assembled archive is
696 MiB with SHA-256
`699834265f70ae6226b4692a0058b7c1ef2ea325d935941bbdf608af2b9c8bab`; the test split
is absent. Kaggle kernel versions 1–5 fixed, in order, recursive input discovery,
offline package installation, the archived v17 package initializer, and explicit GPU
selection. The final server metadata records `enable_gpu: true` and an exact
`NvidiaTeslaT4` or `NvidiaTeslaP100` machine shape. The account API reports six unused
GPU hours, but every batch worker still had no `/dev/nvidia*`, no `nvidia-smi`, a
CPU-only PyTorch build, and no TensorFlow GPU. Separate script and notebook probe
kernels reproduced the same scheduler failure even with embedded Kaggle accelerator
metadata. No Kaggle run produced model metrics.

The local host is an M4 MacBook Air with 24 GiB memory. A clean Python 3.11 TensorFlow
2.16.1 / Model Garden 2.16 / Metal environment confirmed that the official TensorFlow
MoViNet graph still fails at its XLA-compiled Conv3D stem. Its CPU path passed but took
45.5 seconds for one training batch plus one validation batch, so it remains unsuitable
for the complete schedule.

A second, faithful execution path is now implemented in
`active/v17/train_stage_1_movinet_torch_v17.py`. It uses the MIT-licensed
Atze00/MoViNet-pytorch streaming A0 architecture with 2+1D convolutions and converted
official Kinetics-600 weights. Source is pinned to commit
`c2d1edf48fc6c5259707f9d833f22171b4f63493`; the A0 stream weight SHA-256 is
`447c0554daa6bebdcf6fc69b2651b25b29cc69e003da4e6ff56f9a2488f403cf`. PyTorch Metal
runs its convolutions and falls back to CPU only for the small unsupported AvgPool3D
operation. A real end-to-end smoke passed source/weight verification, bit-exact Apple
initialization, forward/backward, checkpoint save/reload, and test isolation. The model
has 2,489,393 parameters, including a 1,533,543-parameter MoViNet backbone. A ten-batch
unfrozen benchmark passed; batch 4 was selected because batch 8 doubled work without a
throughput gain. Benchmark subset scores are not accuracy evidence.

The first detached launch was terminated by the execution shell before Python started;
PID `28083` and the empty log are not evidence of a run. The complete signer-disjoint
experiment was immediately relaunched locally on Apple Metal; its output directory is
`artifacts/models/stage1_v17_sign_movinet_stream_fusion/`. The fixed protocol is five
frozen-backbone warm-up epochs plus at most 35 end-to-end epochs, batch size 4, patience
8. The untouched Apple validation checkpoint is saved at epoch 0 before training, so a
degraded fusion cannot become the reported best model. The official Citizen test stays
sealed. The full epoch-0 audit reproduced 93.12% top-1 on all 378 validation clips.
Completed full-validation rows are:

| Epoch/phase | Loss | Fused top-1 | Visual top-1 | Mean gate | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / protected Apple baseline | n/a | 93.12% | 1.06% | 0.124 | n/a |
| 1 / warm-up | 2.1673 | 93.12% | 1.32% | 0.007 | 381.7 |
| 2 / warm-up | 2.0973 | 93.12% | 4.23% | 0.012 | 384.9 |
| 3 / warm-up | 1.9617 | 93.12% | 7.41% | 0.007 | 393.5 |
| 4 / warm-up | 1.8844 | 93.12% | 10.58% | 0.003 | 405.8 |
| 5 / warm-up | 1.8277 | 93.12% | 13.76% | 0.009 | 414.6 |
| 6 / joint fine-tune | 1.8794 | 93.12% | 8.20% | 0.010 | 812.7 |
| 7 / joint fine-tune | 1.8626 | 93.12% | 9.52% | 0.010 | 814.6 |
| 8 / joint fine-tune | 1.8511 | 93.12% | 6.88% | 0.009 | 806.4 |
| 9 / joint fine-tune | 1.8692 | 93.12% | 7.14% | 0.010 | 814.2 |
| 10 / joint fine-tune | 1.8654 | 93.12% | 7.41% | 0.011 | 825.4 |
| 11 / joint fine-tune | 1.8687 | 93.12% | 7.41% | 0.011 | 852.6 |
| 12 / joint fine-tune | 1.8566 | 93.12% | 7.67% | 0.011 | 837.0 |
| 13 / joint fine-tune | 1.8422 | 93.12% | 7.41% | 0.012 | 792.8 |

Epoch 13 was the eighth consecutive non-improving joint epoch, so patience 8 stopped
the controlled run after 13 total epochs. The protected epoch-0 Apple checkpoint
remains best: 93.12% fused top-1, 99.47% top-5, and 92.53% macro F1 on all 378 official
validation clips. An independent strict checkpoint load reproduced those three fused
metrics exactly. The auxiliary visual path at epoch 0 is randomly initialized and its
standalone scores/gate vary under nondeterministic MPS kernels; because the saved
residual classifier is exactly zero, this cannot change the protected fused logits.
The full history has 13 sequential rows, every row covers 378 validation samples,
`last.pt` records `joint_stale: 8`, and both result and checkpoint metadata record
`test_evaluated: false`. No fused epoch improved on Apple-only, while the best RGB-only
validation top-1 was 13.76% during warm-up and ended at 7.41%. This completes and
rejects this exact three-stream MoViNet-A0 fusion challenger; it does not displace the
frozen Apple Vision plus v17 Squeezeformer selection. The completed local Python
process remained idle after all final artifacts were closed and was terminated normally
to release host resources; no training or result file was open when it was stopped.

The authenticated Kaggle CLI was rechecked after completion to resolve where the epoch
logs originated. `francisbatiancela/slt-v17-movinet-end-to-end` is `ERROR`, and its
server log ends after 57 seconds: no `nvidia-smi`, TensorFlow reported zero GPUs, CUDA
initialization failed with error 303, and the fail-closed launcher stopped before any
epoch. The separate `slt-v17-kaggle-gpu-probe` is also `ERROR`; it saw no NVIDIA device,
CPU-only PyTorch 2.10, and zero TensorFlow GPUs. Therefore none of epochs 1–13 ran on
Kaggle; they came from the local MPS process. No Kaggle model metrics exist. The latest
Kaggle source/metadata and retained partial working output were downloaded for audit to
`artifacts/generated/kaggle_movinet_v17/kernel_pulled/` and `kernel_output/` (about
32 MiB); these are transport/debug artifacts, not model evidence.
