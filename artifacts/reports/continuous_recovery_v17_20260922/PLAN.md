# Five-step continuous recognition recovery

User approved starting this plan on 2026-09-22, including useful parallel work and
evidence-based re-extraction/retraining. This file is the persistent checklist; keep all
five steps visible until completed or explicitly replaced by the user.

## Intended behavior and constraints

Recognize any locked100 sign individually or naturally connected, without requiring a
pause/reset or a familiar phrase. Stable visible words; roughly0.5–1s delay is acceptable,
faster preferred. Translation follows recognition and may add grammatical words without
inventing meaning. The goal remains offline/iOS-first; desktop latency is not iPhone proof.

Reuse existing Stage1 weights, extraction, diagnostics and trainers where appropriate.
Preserve original data/checkpoints and the user's uncommitted changes. No acquisition,
protected Citizen test access, automatic model promotion or silently relaxed data gates.
Shared/missing signer IDs are allowed under the latest supplement admission: preserve
roles and distinguish familiar-signer evaluation from held-out-signer evidence.

The user authorized beginning the work, not skipping input validation. The old generic
phrase manifest remains training_ready=false. A new recipe must explicitly pin the
combined manifest, input transforms, allowed supervision, checkpoint and evaluation.
Training, once ready, runs detached on MPS with completion/failure notification; do not poll.

## 1. Establish where the current system fails — IN PROGRESS

- [x] Inspect current Reel activation/trim, Stage2 temporal inputs and visible output.
- [x] Inspect latest saved Stage2 session without inventing timestamped human labels.
- [x] Identify existing annotated cores and run current-checkpoint full/core paired probes.
- [x] Run12fixed-setting CTC replays (three train sources, one validation source; full/core).
- [x] Run current Reel proposal/verifier on eight supplied cores; keep branch provenance separate.
- [ ] Compare complete-sign recognition, uninterrupted stream output and visible commits.
- [ ] Report separate identity errors, missed/extra events, holds/repeats and delay.

Reuse scripts/live_reel_stage1_v17.py, scripts/live_stage2_ctc_v17.py and existing
stage2_research_review_20260915 / clean_boundary_subset_20260920 diagnostics. Latest
webcam session is qualitative development evidence until temporal references are supplied;
do not turn its hypotheses into labels or reuse it as independent evaluation.

Exit: locate the failure stage on valid paired evidence, or name exactly which annotation
is missing. Historical oracle results and a different model's stream WER are not a pair.

## 2. Prepare the combined data for that task — BOUNDED RECIPE READY

- [x] Run current approved494 manifest integrity verifier.
- [x] Revalidate combined manifest pins and representation-aware feature loading.
- [x] Inventory sequence/interval/isolated label coverage, timings and correlated parents.
- [x] Audit cached temporal coverage: contiguous ranges; no demonstrated need for blanket re-extraction.
- [x] Produce TRAINING_CONTRACT.md preserving original roles and supervision semantics.

Canonical input: data/local/combined_dataset_v17_20260922/manifest.json.
Reuse scripts/build_combined_dataset_v17.py:load_features without rerunning its mutating
builder. Never flatten the6421records into6421phrases; never infer blank truth from gaps.
Do not repeat native-rate1280px extraction already established by the20260921 audit.

Exit: pinned usable inputs with source/timing semantics, explicit exclusions and a
source/role sampling contract. Structurally valid arrays alone do not establish readiness.

## 3. Adapt Stage1 while preserving existing strengths — EXPERIMENT LAUNCHED

- [x] Select current Reel proposal initialization; measure retention again on the actual inputs.
- [x] Specify the cleaned combined-data contract as the difference from prior joint failures.
- [x] Pin two-seed frozen/adapted comparison; reviewed code and MPS preflight passed.
- [x] Verify gradients reach the intended encoder parameters and all input types load.
- [ ] Run the bounded experiment and report source-specific recognition and retention.

Candidate code to reuse: active/v17/model_v17.py, active/v17/joint_ctc_v17.py and the
existing exact-core/grounded supervision helpers. Do not commit to a new trainer until
the audits determine whether a representation change, sampling change or both are needed.
No promise that unfreezing alone repairs generalization. Any isolated-accuracy trade-off
must be explicit before selection, not adjusted after seeing the result.

## 4. Train one continuous recognition path — BOUNDED COMPARISON LAUNCHED

- [ ] Preserve temporal features and pin streaming window/stride/look-ahead semantics.
- [x] Reuse a small CTC sequence head as the baseline; keep blank0 and existing OTHER rules.
- [ ] Include singles, connected signs and varied sequence lengths under valid supervision.
- [ ] Evaluate identity/order/count with per-source S/D/I, holds and true repeats.
- [ ] Compare to the pinned baseline; retain failures rather than modifying gates afterward.

A shared encoder may be optimized with both identity and sequence objectives in one
controlled experiment if step3 establishes that contract; steps3–4 need not become two
unrelated trainers. No gloss-free LM swap, four-state redesign or architecture sweep.
No full100 continuous claim when phrase coverage or hold/repeat evidence is incomplete.

## 5. Stable display, then speed — DESIGN REVIEW PARALLEL; IMPLEMENTATION DEPENDENT

- [ ] Separate internal hypotheses from immutable visible commits.
- [ ] Remove wrist displacement as a required prerequisite only with a replacement validated
      on low-wrist-motion signs, rest and transitions; do not merely lower the threshold.
- [ ] Verify one held sign emits once and two intentional signs can emit twice.
- [ ] Measure source-event-to-visible-commit delay, misses and runtime on identical replays.
- [ ] Integrate only the successful candidate; keep current model/runtime rollback paths.

Use the existing scripts/live_reel_stage1_v17.py and scripts/reel_hud_v17.py interfaces;
preserve concurrent user UI edits. No mandatory pause/end gesture for individual words.
Window accumulation, look-ahead, compute and commitment all count toward latency.

## Parallel execution and verification

Two read-only workers independently review paired diagnostic reuse and data contracts.
Main thread preserves this plan, verifies pinned inputs and reviews failed recipes.
Steps3–4 cannot meaningfully train before the data/diagnostic decisions; step5's interface
can be reviewed now but model-dependent behavior is not guessed in advance.

Use venv/bin/python. Run smallest affected checks; keep generated evidence under this
report directory. Log material findings in docs/ground_truth/live-streaming/log.md and
update PROJECT_GROUND_TRUTH.md only for current state. Record commands/results and
limitations in REPORT.md. Do not mark the five-step program complete after an audit.

## Current execution — 2026-09-22

MPS preflight passed all6421records and7367windows/arm, worst-case gradients, zero optimizer
steps. Four focused tests pass; runner independently reviewed. First preflight datatype
abort retained and fixed by one shared float32 conversion; no data re-extraction.
Two-seed frozen/adapted12epoch comparison launched detached with notification, PID98647.
See ../combined_frozen_joint_v17_20260922/{preflight.json,launch.json}; do not poll training.
Read selected-checkpoint train/validation/source retention results next session. Steps1
and5still require event/hold/repeat and latency evidence; no claim of finished live recovery.

## Completion review — 2026-09-22

All four runs complete; checkpoint hashes, selection and exposure verified. Adapted local
validation WER50.84/53.26% versus frozen60.89/66.85%, despite adapted training0.48/1.74%.
Bounded experiment finished, but continuous recovery not achieved; no live promotion.
See ../combined_frozen_joint_v17_20260922/REVIEW.md. Next per-record identity/alignment
analysis; all five steps remain tracked, live event/hold/repeat/latency work still pending.

## Explicit temporal supervision — launched 2026-09-22

Matched56clip diagnostic completed with currentcheckpoints/cache; both identity and CTC
emission errors remain. See ../combined_transition_diagnostic_v17_20260922/REPORT.md.
Step4nowtests central knownsign cores plus boundedinternalgap supervision inside the same
connectedsequence forward pass. SelectedStage1frozen, cachedencoderfeatures,3epochs/head,
twoseedswithCTConlycontrol. Anchor/recipe reviewandMPSpreflightpassed, detachedPID40294.
See ../transition_anchors_v17_20260922/{CONTRACT.md,preflight.json,launch.json}.
No livepromotion, no trainingpolling; nextsession readstatus/results. Steps1and5stillneed
full identity/event/hold/repeat/latency evidence; no claim thefive-step goal iscomplete.

## Timed-anchor result — 2026-09-22

Completed and verified allfourruns. No improvement over matchedCTConlycontrol: phraseWER
50.98/52.41%botharms; anchoredexact29/22vscontrol29/24of211. No promotion or extraepochs.
See ../transition_anchors_v17_20260922/REVIEW.md. Five-step recovery remains unfinished.

## 2026-09-22 — matched older evaluation and local-signer exception

Older pipeline current211phraseWER39.04%, local38.73%; no promotion. User authorizes
local signer sharing. New bounded control/familiar comparison prepared under
local_familiar_signer_v17_20260922,139clips reassigned/60heldout; other roles unchanged.
Steps3–4 continue; all five original steps retained. Research and label audit in
coarticulation_research_v17_20260922/REPORT.md; augmentation not yet implemented.

Launch verified after final equality guard and repinned zero-step preflight:
6421records/15430windows, cache/direct maxdelta4.7684e-6, finite/nonzeroheadgradients,
basegradientsabsent,2focusedtestspass. Detached caffeinatePID78465 launched
2026-09-21T23:31:43.656638UTC (Sep22PHT), recipeSHA
 a91c756a3f8b04eda63f7d7185f373288efbfcc079243cb20908503f1e9437c9.
Completion not yet observed; no training polling. Read status/results next session, compare
both arms on common60local clips and source retention before deciding. Index regenerated;
git diff --check passed. Allfive recovery steps retained; liveUI/model unchanged.

## 2026-09-22 — familiar-signer trial completed; language-prior research

Reviewed all4completedruns: checkpoint hashes/earliest-best selection/paired baselines/
coverage verified. Same60localclips: control35.80%WER(epoch0both), familiar19.14%(epoch6)
and22.22%(epoch5); exact20→33/27of60. LocaltrainWER4.87/6.26%familiar. AllS/D/Ifell.
ASLLRP12knownWER33.33%both; O5S5still77–79%. Familiar-signer/six-template reused development
results, not arbitrary live/unseen-signer proof. FrozenStage1 means no evidence of new
encoder coarticulation learning. Added139clips also changes data amount/composition.
Previous standalone211evaluator countsOTHER; this trainer removesOTHER for knownWER;
localmetricsunaffected, do not compare cross-report pooledscores without normalization.

User asks next action and next-word research. Read2025 Sign Spotting Disambiguation using
Large Language Models: visual candidates+gloss conditional probabilities via beam search,
top1 latefusionWER47.24→44.38/44.73%, top5notdeployabletop1. Recommended candidate streaming
parity/live stress evaluation first; add separate offline greedy/beam/weaksmoothedglossprior
comparison with training-only transcripts, no forced phrase completion, unseen combinations
and latency/insertions checks. Then actual-context jitter as separate visual intervention.
No live promotion/newtraining/LM implementation. Full report:
artifacts/reports/local_familiar_signer_v17_20260922/RESULTS_REVIEW.md.


## 2026-09-22 — familiar candidate available through app shell

User requested live command and concurrent decoder experiment, then explicitly selected
scripts/app_shell_v17.py. Added opt-in --familiar-ctc-checkpoint branch to existing shell
warm/start; default Reel path preserved. New active/v17/familiar_live_v17.py verifies
reviewed candidate hash and original base/head provenance, restores familiar state, uses
8sourceframes/stride4 and bounded15evidence-step causal history with greedy CTC collapse.
No wrist-motion emission gate, no language prior, no English completion. Existing
scripts/live_continuous_v17.py supports prebuilt model/shell presentation; navigation
pauses extraction and resets model state, literal single-line transcript, Enter clears,
Q quits. Camera normalization is past-only and differs from cached clip normalization;
batch parity does not establish live accuracy. Candidate events saved under continuous_live_v17;
not yet integrated into app HISTORY's SessionRecorder format.

Command: venv/bin/python scripts/app_shell_v17.py --familiar-ctc-checkpoint
artifacts/models/local_familiar_ctc_v17_20260922/seed_17521_familiar.pth

Validation: real checkpoint CPU streaming logits match batch including residual endpoint
and boundedhistory (>15steps), CTC tokens match, reset/nonfinite rejection checked. MPS
live forward finite. 95 focused tests pass across new runtime, app pages/shell and causal
vision; CLI help pass; diffcheck pass. No camera opened by agent, no protectedtest access,
no live default promotion. Offline decoder comparison running separately on CPU; initial
weak bigram gains require uniform-prior control to distinguish a length penalty effect.


Offline decoder comparison COMPLETE, fixed1735validationrecords/two familiar heads,
beam8 and priorweight0.1,51distinct trainingphrase transcripts only. Local60WER:
17521 greedy19.14/beam17.90/uniform16.67/bigram16.05%;17522 22.22/22.84/21.60/20.99%.
Uniform prior isolates constant extension length penalty: learned pair preference adds
only one fewer edit (0.62percentagepoint) and one more exact clip per seed beyond uniform.
Most improvement is not demonstrated linguistic-context gain. All60local targetsequences
seen in training;9/12ASLLRPsequences novel and noWERchange(33.33%). No liveLM promotion.
Report/results/scripts/tests under familiar_decoder_v17_20260922 and
scripts/evaluate_familiar_decoder_v17.py. Decoderfocusedtests pass; no hypothesis weights
tuned on validation. CPUbigramdecode2.90/2.92msperexample averages, not camera latency.
Next safe action is user live app-shell evaluation of visually decoded candidate; use
novel combinations to assess reliability before promoting any language prior.

## 2026-09-22 — user live familiar failure on I GO; Reel history confirms emissions

User reports candidate recognizes familiar greeting but misses I GO; requested Reel history
comparison. Reviewed candidate sessions074639/074748 and Reelapp074918Sep22. Reel committed
I27.17/41.44s andGO29.54/42.95s (verifiersI.915/.962,GO.771/.768). Candidate logs emitted neither;
mostlyHELLO/HOW/YOU/OTHER/TOMORROW. These are distinct runs, not pairedgroundtruth/WER.
Training coverage: I28isolated and115phraseclips allPLEASE HELP I;GO23isolated and33phraseclips
allTOMORROW SCHOOL GO;zeroI GO pairs. Selectedcandidatecachedisolatedval I15/17headexact,
GO9/16; labelsnotmissing. CandidateusesolderasllrpcoreStage1, not currentReelproposal/verifier.
No liveLMenabled. Six-template19.14%validation did not establish user's arbitrary signing.

Livecandidate13.34/16.90observedfps with30Hzduplicatedfeatureticks and causalnormalization
remainpossibleinput-domain causes, not demonstrated attribution. Candidate saved onlydecoded
events, no video/features/logits: instrumentation insufficient to diagnose missedrealattempts.
Tried bounded same-recording replay via AppleVision, but latestReelMP4 lacksmoovatom and cannot
open;no replayheadprobabilities. No running app process atcheck; nonfinalizationcauseunknown.
New report/replaydiagnostic: artifacts/reports/live_familiar_diagnosis_v17_20260922/.
No thresholds/model/training changed. Keepcandidate experimental; next safeaction capture
replayable commoninput and per-window base/head evidence before trainingorLMpromotion.

## 2026-09-22 — executed matched Reel core/context/gap comparison

User authorized doing comparison. CurrentCoreMLproposal+fullverifier re-extracted12approved
ASLLRPdevelopmentvideos; hashesverified. Same17cores: verifier11/17tight,11/17+100ms,
7/17+250ms; correctconditionalcommits5/17,9/17,5/17; wrong1/0/1. Modestcontext helpssome
commits; largercontext harmsidentity. Gap842935(0.5205–0.8475s): proposalWORK.5943/verifier
WORK.5173, bothaccepted+conditionalcommit;1/3checkedgaps pass. Sixcorrectcoreverifiertop1
blocked; oneFRIENDproposal overridden toANSWER passescommit. No single threshold diagnosis.
Conditionalcommit bypasses activation/proposalstability/history; not actualstreamWER.
All17cores passwithin-windowwriststart; staticwrist behavior notcovered.

First replay incorrectly advanceddeadlinefromcurrenttime (effectively15Hzon30fpsinput).
Preservedinitialoutputs, fixed to actualReelmax(previousdeadline+1/20,currenttime), added
selfcheck, reranall12clips and assertedpaired17identities. Only correctedresults count.
Recent5appMP4unreadable, so no exactuserHELLO→MY replay. Report/script/results:
artifacts/reports/reel_matched_windows_v17_20260922/.

Separateboundedrecordingfix: SessionRecorder.finish_video() releasesMP4 beforemodelcleanup;
appnativewindowclose signalsquit. Historicalmissingmoov/unfinishedhistory shows nofinalization,
notproof ofexactexitroute. Worker and parent independently ran89focusedtests, allpass.
No modelweights,thresholds,training or dataset roleschanged. Next hard-negative+truepositive
comparison and replayableliveinput; preserveReeldefault.



## 2026-09-22 — fast Reel decision probe

Completed frozen-Reel temporal linear rejection comparison; failed, no promotion. False WORK remains; correct conditional context commits 9->8 and 5->4. See ../reel_decision_probe_v17_20260922/REPORT.md. Five-step recovery remains active; this is evidence for steps 1/4, not completion of continuous recognition. Next training-source hard-negative/true-positive audit before richer emission fitting.


## 2026-09-22 — pretrained Renz boundary reference completed

Official weights run on12approvedASLLRPvalidation clips. Edgepadding+publishedmidpoint extension yields62.50% offline conditional-pipeline WER,2/12exact; no live promotion. Source frontend adaptation and noncausality explicit. See ../renz_reel_v17_20260922/REPORT.md. Steps1/4 informed, not complete; no new training.


## 2026-09-22 — user priority: compare pretrained candidates before combined training

The user explicitly wants to see SHuBERT and the other pretrained candidates before choosing
an architecture or combining datasets. This supersedes immediately proceeding with another
Renz-specific fine-tuning or hard-negative training experiment. Preserve the original five
recovery steps; the following comparison informs their implementation.

| Candidate | Next comparison / status |
| --- | --- |
| Renz I3D + MS-TCN | Offline reference completed; buffered app mode available for user trial. |
| SHuBERT | Next pretrained ASL representation candidate. Verify its exact frontend and weights, then compare on the same approved videos. Not evaluated yet. |
| Zuo Online / TwoStream-SLR | Compare pretrained initialization and online method next; German/Chinese output heads require adaptation for ASL. Not evaluated locally. |
| Zhao MHB | Boundary/handshape reference; official downloadable weights were not verified. Do not count a paper review as a model experiment. |
| Commercial recognition | Optional external comparison only; no service calls, uploads, or purchase authorized by this note. |

Use common videos and label conventions where applicable; distinguish boundary quality,
sign identity, insertions/deletions, end-to-end WER, compute time and future-context delay.
Do not rank representation-only models against completed recognizers without equivalent
adaptation. No claim that every possible model can be exhaustively tested.

### Dataset retained for later: Renz sign segmentation release

- Official source: https://github.com/RenzKa/sign-segmentation
- Dataset schema: https://github.com/RenzKa/sign-segmentation/blob/master/data/README.md
- Published archive: data.zip, approximately 5.5 GB per the upstream README.
- Download script: artifacts/vendor/renz_sign_segmentation/download/download_data.sh;
  upstream Drive archive ID: 1rckvP0EvsJ3gA_NUycOp6mh_E9gYq3VF.
- Contents documented upstream: 1024-dimensional I3D features and metadata including signer,
  source video/time spans, train/eval/test assignments, frame-level sign/boundary annotations
  (0=sign, 1=boundary), and gloss labels. Source datasets include BSL Corpus and PHOENIX14.
- Intended future role: boundary/segmentation pretraining or adaptation, followed by ASL
  evaluation. Not direct additional locked100 ASL classification labels or Apple landmarks.
- Keep source splits and model/frontend compatibility explicit. Same translated gloss does
  not imply the same sign across languages. Source dataset terms govern data/model use;
  raw BSL videos have separate open/restricted access conditions.
- Status: noted for later, NOT downloaded or admitted into training. Renz code and weights
  already downloaded are distinct from this dataset archive. Defer combining/training until
  the pretrained-model comparisons are reviewed with the user.

### Current live trial

`scripts/app_shell_v17.py --renz-buffered --device mps` uses the unchanged Reel classifier
with overlapping four-second Renz buffers. MPS convolutions, explicit CPU fallback for
unsupported 3D max/average pooling. Real-video smoke passed; four-second repeated-image
throughput smoke took approximately29seconds, not real-time. The mode records skipped
backlog buffers and measured delay. It is opt-in; default Reel is unchanged.

## 2026-09-22 — SHuBERT setup and real-video probe completed

Official encoder and face/hand weights load strictly on MPS. Real approved validation clip:
41x768finitefeatures; CPU/MPS encoder check maxdelta1.824e-5. Published signer crop is necessary
on the checked clip: face detections0/41without,41/41with. Full details, limitations, timings,
reproduction command and data register: `../shubert_probe_v17_20260922/REPORT.md` and
`../shubert_probe_v17_20260922/DATASETS.md`. This is not a trained decision model or WER result.
Next: bounded frozen-feature readout comparison with same reviewed intervals and explicit
recipe; keep Zuo/TwoStream and Zhao review queued before broader combined training. No new
dataset videos acquired, no default live changes, no protected test consumed.

## 2026-09-22 — SHuBERT fixed readout comparison completed

See ../shubert_decision_probe_v17_20260922/REPORT.md. Same247intervals,45videos, fixedridge and
train-onlythreshold. SHuBERT blocksfalseWORK(conditionalgapcommits1->0), but correcttight-core
commits5->2; loses3FRIEND successes. Twoofthreegapsrejected, no low-motion-positivecoverage,
notWER. Allsource-clock/controlchecks passed aftercorrectingcroppedFPSroundingbeforefit.
No livepromotion or broadertraining. Nextcandidate: Zuo/TwoStream, perusercomparisonpriority.

## 2026-09-22 — Zuo online transfer completed

Actualauthorcheckpoint recovered fromreplacementDrive; originalSharePoint404. HRNetMPSfront end
working,938recognizer tensors strictlyloaded. Directblankveto on17core+3gapcentres: gaps3/3
rejected, falsecommits1->0, correctcommits5->0. DifferentprotocolfromSHuBERT; noencoder-ranking
claim or promotion. See ../zuo_twostream_probe_v17_20260922/REPORT.md andDATASETS.md.
Next: Zhao/MHB releaseavailability. No broadertraining or signingdatasetacquisition.

## 2026-09-22 — Zhao component availability review complete

Author code/weights found; strict segment MPS synthetic check passed. Complete MHB
raw-video inference contract not recovered: missing preseg producer, unused handshape
branch in active code, unresolved handshape mapping. No accuracy result or live promotion.
See ../zhao_mhb_availability_v17_20260922/REPORT.md for data register and evidence.
Next prepare ASL temporal-boundary coverage/evaluation recipe; broader training remains
unlaunched. All earlier five recovery steps remain tracked above.
