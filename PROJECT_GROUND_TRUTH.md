# SLT Project Ground Truth

**Last updated:** 2026-09-02 19:34 PST (+0800, Asia/Manila)

This is the single canonical handoff for the project. Every future session must read
this file before changing the pipeline and update it after every material decision,
implementation, dataset action, experiment, or validation result. Other documents
may provide detail, but conflicts are resolved in favor of this file and the current
code/tests.

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

## 2026-09-02 11:00 PST — low-motion/non-neutral boundaries are cleaner but still fail accuracy

A second separate path, `scripts/live_motion_valley_v17.py`, now closes a sign after a
short low-motion hold at the current hand position rather than requiring return to a
neutral pose. The original neutral/pause path and the overlapping-window experiment
remain available unchanged by default. The motion path keeps the same extractor,
Stage-1 classifier/gates, display, RESET/FINISH, tiny Stage 3, speech, history, and
low-resolution recording. Saved-video segmentation waits for each classification, so
offline frames are not dropped while Core ML is busy.

The initial motion ≤0.006 for 0.12 seconds configuration produced 15 clips and nine
accepted glosses across all nine local development/reference phrases. It scored 1/9
exact overall and 1/5 among fully vocabulary-covered phrases; only MY_NAME was exact.
Median/p90 classification latency was 448.01/529.23 ms. Compared with fixed overlap,
it reduced spurious insertions and recovered more meaningful components, but often
merged neighboring signs.

The nine phrase recordings were then used as intended for weak boundary calibration.
A motion-only sweep found that motion ≤0.008 for 0.16 seconds matches the expected sign
count in all five fully vocabulary-covered phrases. Reclassification of those same five
fitted development videos still scored only 1/5 exact: GOOD_MORNING -> `EAT MORNING`;
HELLO_HOW_YOU -> `HOW NEED`; MY_NAME -> `MY NAME`; THANKYOU_FRIEND -> `NAME`;
TOMORROW_SCHOOL_GO -> `TOMORROW GO`. Twelve clips were formed, nine accepted, with
360.94 ms median and 490.77 ms p90 latency. Better endpoint counts did not solve the
continuous/local classifier-domain errors. This fitted diagnostic is not independent
accuracy evidence.

Therefore phrase data is not needed merely to run a webcam loop, but remains essential
for threshold fitting, transition/domain learning, and honest continuous evaluation.
The existing 780 local recordings should be retained. Their phrase prompts can support
weak sequence training, but missing signer identity, five OOV prompt glosses, and absent
frame boundaries limit claims. Do not delete the phrase corpus, promote either new live
path, or lower gates to make the numbers look better. Full evidence is under
`artifacts/reports/live_motion_valley_v17_local_phrase_eval_v1/` and
`artifacts/reports/live_motion_valley_v17_local_phrase_tuned_v1/`.

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

## 2026-09-02 10:25 PST — tiny Stage 3 fine-tuned, long-buffer gate passed, and promoted to live FINISH

The 15.58M-parameter `visheratin/t5-efficient-tiny-grammar-correction` checkpoint at
revision `a98f126664317cf2e68d33234c7072fdd6b289f3` was fine-tuned for one MPS epoch and
saved at `artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1/` (59 MiB
`model.safetensors`, SHA-256
`c0875815d47d9242bae9d19bd1040f85647550a5b2f24c482bdb9b2f4525a5b0`). Six epochs
were originally scheduled, but epoch 1 already reached 100% normalized exact accuracy
on the controlled long validation slice; the later epoch was interrupted rather than
wasting compute. Checkpoint selection used validation only. The fixed synthetic test
split was then accessed exactly once by `evaluate_stage3_tiny_v17.py`; Citizen test and
2M-Flores devtest were not accessed.

The earlier Stage-3 CSVs were explicitly audited. `slt_stage3_dataset_final.csv`
(SHA-256 `90009c48075e1871664a6fe6d4ce9e59bea96ca66390fc3708d63583cac4156f`)
contains 15,843 unique synthetic pairs and 1,537 sequences of at least five glosses,
but only 742 rows are entirely within the locked 100 and only 11 of those are long.
It was used as broad rule-generated grammar supervision, never described as genuine
ASL. `slt_dialogue_dataset.csv` (SHA-256
`87f5287615b47f59af4157137316c91457ba2fb7424e4d32fe77438372b74f0b`)
contains 11,245 rows but only 35 unique pairs, 11,210 duplicates, and no five-gloss
examples; it was audited and excluded. Reviewed templates override conflicting CSV
targets. An additional 143 deterministic, locked-100 compositions of 5–12 glosses were
split by exact sequence into 82 train, 35 validation, and 26 test rows. The final
16,003-sequence manifest has zero exact-gloss overlap across splits. Weighted training
contains 14,599 rows; the held validation/test sets contain 1,578/1,596 sequences.

The promoted epoch-1 checkpoint scored 93.98% normalized exact overall on validation,
95.76% on validation sequences of 5+ glosses, and 100% on the 35 controlled long rows.
On the one-time synthetic test it scored 93.36% overall, 94.0% on the locked-100 slice,
91.90% across all 210 sequences of 5+ glosses, and 100% across the 26 unseen controlled
locked-100 long combinations. The two non-exact locked-100 long outputs differed only
by natural article choice (`to school` versus the CSV's `to the school`). Other
out-of-vocabulary synthetic failures include genuine semantic/fingerspelling errors,
so this remains a bounded synthetic/reviewed renderer, not a general ASL translator or
linguistic ground truth. Full manifests, predictions, metrics, hashes, and the passed
predeclared promotion gate are in
`artifacts/reports/stage3_v17_t5_efficient_tiny_locked100_v1/`.

The live prototype now defaults FINISH to this local tiny checkpoint, keeps exact
reviewed templates as the fastest safe route, and uses literal rendering if model
loading/generation fails. `--naturalizer ollama` retains the previous 1B comparison;
`--naturalizer literal` disables learned generation. The tiny model loads
asynchronously. A real single nine-gloss benchmark reproduced its reference exactly
and measured 137.96 ms median warm CPU generation versus 1,678.02 ms on MPS. MPS is
therefore correct for training but wrong for token-by-token live decoding on this Mac;
live Stage 3 defaults to CPU and leaves MPS/Core ML capacity to Stage 1. Sixteen focused
live/data tests, both Python compile checks, CLI help, and `git diff --check` passed.

## 2026-09-02 09:54 PST — 15.6M T5 is the lightweight naturalizer candidate, but needs in-domain tuning

Three Hugging Face encoder-decoder candidates were pinned, cached outside the repo, and
measured on the Mac using only existing reviewed Stage-3 templates and handcrafted
in-vocabulary combinations. No Citizen validation/test data or new video data was
accessed. `visheratin/t5-efficient-tiny-grammar-correction` at revision
`a98f126664317cf2e68d33234c7072fdd6b289f3` is the clear deployment candidate: 15.58M
parameters, 62.3 MB FP32 weights, and 32.16 ms median greedy inference over the first
14-phrase probe. On all 36 reviewed Stage-3 templates it reproduced 23/36 references
after case/punctuation normalization at 43.73 ms median. It handled `I NEED WATER`,
`I FEEL SICK`, `HELLO HOW YOU`, and `WHERE HOSPITAL`, but also hallucinated content
(`HELLO` -> `Hello everyone!`), changed intent (`GOODBYE` -> `Good night!`), and missed
important ASL order/negation/question cases. It is therefore not safe to replace the
current naturalizer unchanged.

`HamdanXI/t5_small_aslg_pc12` revision
`b35e3323732c7244236189674bdc0728f37b31e8` is directly trained gloss-to-English and
has 60.5M parameters/242 MB weights, but measured 115.78 ms median and failed badly on
unseen combinations (`FATHER SICK` and `MOTHER FEEL GOOD`). Its reported high in-domain
BLEU is not persuasive for this project because ASLG-PC12 was created by rule-transforming
Project Gutenberg English rather than from genuine ASL, and published work explicitly
warns that it is unreliable for SLT. The model card says Apache-2.0 while ASLG-PC12 is
reported as CC BY-NC 4.0, so downstream licensing also needs care.

`jbochi/coedit-small` revision `6ce9822b4ff6e4af86b70f979c890e9e41f04366`
has 77M parameters/approximately 296 MB cached and measured 63.48 ms median after its
first warm call. It fixes ordinary English but frequently leaves gloss order unnatural
or duplicates meaning, so it is a weaker starting point than T5-efficient-tiny for this
bounded task. The three research downloads currently occupy approximately 789 MB in the
user Hugging Face cache; they were not added to git or the repository and have not been
deleted.

Recommended next experiment: retain exact reviewed templates as the instantaneous
fail-closed path, then fine-tune the 15.58M T5-efficient-tiny checkpoint on genuinely
reviewed, locked-100 gloss-to-English pairs and evaluate on held-out phrase combinations.
Do not train on its own generated sentences or claim that generic grammar correction is
ASL translation. A successful checkpoint can be converted to Core ML or quantized ONNX;
the base architecture is about 31 MB FP16, 15 MB INT8, or 7.5 MB INT4 before packaging.
This is substantially smaller and faster than the 1B Ollama model, but accuracy—not raw
latency—is the gate for replacing it.

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

## 2026-09-02 07:42 PST — live 100-gloss prototype architecture and extractor timing frozen

The requested laptop prototype will be an isolated-sign Stage-1 diagnostic, not a
Stage-2 or continuous-translation claim. It will use the locked 100-label order and
Apple Vision, automatic motion/rest boundaries, top-three uncalibrated model scores,
hand/face input quality, boundary progress, accepted history, optional macOS speech,
and timestamped low-resolution session video plus JSON. The user accepts an explicit
neutral/rest pause for this prototype only; the later no-pause path still requires a
learned or sliding-window online boundary/decoder design.

The implementation decision is one Apple Vision pass per processed frame shared by
segmentation, landmark construction, and real-pixel hand/mouth crops. This avoids
rerunning hand and face detection once per modality at clip close. On 24 development
frames from a Citizen **validation** `HELLO` clip downscaled without distortion to a
720-pixel maximum side, the reusable Vision detector measured: hands-only 7.11 ms
median / 10.58 ms p90, hands+face 17.38 ms median / 38.66 ms p90, and hands+face with
body every fourth frame 19.84 ms median / 26.31 ms p90. The small sample is an
extractor feasibility measurement on this Mac, not sustained latency or camera
accuracy evidence. No Citizen test or other sealed split was accessed.

Two modes are planned for an honest latency/accuracy comparison. The default
`lip-aware` mode will reproduce the frozen four-stream 0.30 landmark / 0.15 mouth /
0.35 lower-face / 0.20 hand teacher, so mouth pixels genuinely affect the decision.
The optional `fast` mode will use the selected unified landmark+hand Core ML model;
its four lip landmarks remain present but it must not be described as full lip
reading. Both modes reuse the existing Core ML MobileCLIP2 hand encoder. Raw softmax
values will be labeled **model scores**, not calibrated confidence.

## 2026-09-02 07:24 PST — historical alphabet live loop located; reuse the interaction pattern, not its model contract

The requested pre-Stage-2 live alphabet implementation is in commit `4fd5f44`
(`007 - cursor 68%`, 2026-02-22) at `src/main_inference_CUDA.py`; commit `46ece22`
retains the same inference file while changing only the trainer. Neither commit
contains `src/train_stage2.py`. The live path is a 26-class, one-hand MediaPipe
Stage-1 classifier: it buffers at least ten webcam frames, resamples to 32 frames,
normalizes around the wrist, derives XYZ/velocity/acceleration features, applies a
0.85 softmax threshold plus four agreeing predictions, and displays the top label,
confidence bar, and accumulated held-sign sequence. A later revision in commit
`a4839c5` adds motion gating, cooldown, lock progress, top-three scores, FPS, and
editing controls; that same commit also introduces a Stage-2 trainer, but its live
camera entry point still calls Stage 1 only.

The matching local alphabet asset still exists at
`artifacts/model_assets/weights/SLT_Stage1_Results/best_model.pth`: SHA-256
`829c639c8cf3e4bde9b8bf522eaad021faccd15be03d58b5cdf5670004580401`, 26 labels
`A` through `Z`, 9 input channels, `d_model=256`, four transformer layers, epoch 35,
and a recorded 100% validation score. That accuracy is not independent evidence: the
historical trainer performs a stratified random sample split rather than a
signer-disjoint split, and the newer alphabet audit already warns that signer/session
independence is unknown. The model is also incompatible with v17 inputs: historical
features are one MediaPipe hand shaped `[32,21,9]`, whereas v17 uses Apple Vision,
two-hand/face/body landmarks shaped `[32,61,5]` plus three hand-image embeddings.
The surviving checkpoint was strictly loaded against the exact `46ece22` architecture
without missing or unexpected keys; a zero-input smoke produced finite `[1,26]`
logits from 3,633,926 parameters. This proves artifact/code compatibility only, not
recognition accuracy or live-camera generalization.

The old design is therefore useful as a live diagnostic UX and control-loop
reference, not as a replacement for Stage 2 and not as a checkpoint to connect to
the current mobile pipeline. Its steady-hand gate, buffer reset, and cooldown work
like a deliberate fingerspelling keyboard; using those as conversational sign
boundaries would suppress moving signs and discard natural coarticulation. The
current v17 mobile recognizer instead processes completed recorded files in
nonoverlapping 32-frame windows, emits 101-way CTC logits, and exposes no confidence
in the frozen Stage-2-to-Stage-3 contract. A true live path still needs a streaming
window adapter and provisional CTC state.

The separate Flutter app directory is not itself a Git repository. Its current camera
UI calls `startVideoRecording`, then `stopVideoRecording`, and only afterward invokes
the native v17 pipeline on the completed file. It does not call the camera plugin's
image-stream API. Therefore adding old-style confidence feedback is not merely a UI
toggle: frames, orientation, Apple Vision tracking, crops, embeddings, CTC state, and
backpressure must be connected as a new streaming input path.

Recommended consultation boundary: first resurrect the old HUD behavior as a
separate current-v17 Stage-1 live diagnostic (top-k scores, hand/input quality,
rolling-window state) to isolate camera/extractor/domain problems. Then add a distinct
Stage-2 streaming mode that first preserves the trained motion-anchored,
nonoverlapping 32-frame window contract, retains CTC blank/prefix state across
successive runs, shows provisional token posterior/margin outside the frozen
downstream contract, and commits a gloss only after blank-delimited hysteresis.
Overlapping/short-stride windows should be a later measured experiment because the
current head was selected on nonoverlapping windows. Do not let Stage-1 confidence
choose conversational sign boundaries.
No historical file was restored, no runtime/model/app code changed, and no data split
or test set was accessed in this inspection.

## 2026-09-02 07:17 PST — standalone SignWriting pilot runs locally but fails the first semantic gate

The public English-to-ASL SignWriting checkpoint was evaluated by itself before any
pose/video work. `scripts/evaluate_signwriting_symbolic_pilot_v17.py` runs the official
`signwriting-translation` code in an isolated Python 3.11 environment, renders Formal
SignWriting locally, and compares three symbolic strategies: natural-English input,
project-gloss-order input, and concatenation of separately generated isolated forms.
It does not generate landmarks, pose, or video.

The local non-split cache is
`data/local/signwriting_symbolic_pilot_v1/hf_cache/` (about 463 MB). The forward model
is `sign/sockeye-text-to-factored-signwriting` revision
`3a45c3bedc0c6ee08fcb3e87b1aaee602dfa06d9`, licensed CC BY-NC 4.0. The separate
diagnostic reverse model is `sign/sockeye-signwriting-to-text` revision
`8f314871ee42ad68f9e01ab23e818d8aaa695665`, whose model card says MIT. The existing
project venv was not modified; the official package needs Python 3.10+ whereas the
project venv is Python 3.9.6.

The review artifact is `artifacts/reports/signwriting_symbolic_pilot_v1/index.html`,
with raw FSW, local renders, and machine-readable results in `report.json`. All five
natural phrase outputs and all twelve isolated lexical outputs are structurally
parseable FSW. Syntax alone is insufficient: none of the five natural phrase outputs
strictly round-trips to its source text through the separate reverse checkpoint. The
diagnostics are: `Good morning -> daughter`, `Hello, how are you? -> you`, `My name ->
synagogues`, `Thank you friend -> face the punctuation`, and `I will go to school
tomorrow -> day before yesterday`. Only 7/12 isolated probes strictly round-trip.
Gloss-order prompting and concatenated isolated outputs remain parseable but do not
repair phrase semantics under the same diagnostic. The reverse model is not ground
truth and these results still require a fluent ASL signer who reads SignWriting, but
the severe omissions and unrelated returns are enough to fail an unattended
automation gate.

Decision: keep SignWriting as a potentially useful representation, but hold this
public checkpoint before connecting either Rylo or a local pose renderer. The next
safe comparison is native review of the report followed by a dictionary-grounded or
locally fine-tuned symbolic planner using only verified entries for the 100-sign
vocabulary. No pilot output was added to Stage-2 training/validation/test, no split or
sealed data was accessed, and no pose generation was resumed.

## 2026-09-02 06:59 PST — evaluate SignWriting as the reverse symbolic layer, not Stage-2 motion data

The reverse-generation branch is paused for architecture discussion. Rylo's public
design should be evaluated for reuse instead of rebuilding its language-production
stack: spoken text -> normalized text/SignWriting -> pose -> skeleton/avatar. Public
references are `https://github.com/sign/translate`,
`https://github.com/sign-language-processing/signwriting-translation`, and
`https://github.com/sign-language-processing/spoken-to-signed-translation`.

SignWriting and SignBank+ are valid candidate datasets for a future reverse symbolic
planner. They can supervise English/text -> ASL notation, expand lexical and
phonological coverage, and provide explicit handshape, orientation, location,
movement, and nonmanual constraints for a SignWriting-conditioned pose generator.
They are not continuous-landmark datasets: notation rows do not contain measured
frame timing, signer identity/style, natural inter-sign coarticulation, camera/view
variation, detector missingness, or genuine phrase video. Therefore they must not be
counted as Stage-2 CTC training examples or used by themselves as proof that generated
motion is human-correct.

A motion generator would still require paired SignWriting + genuine pose/video (and
preferably temporal alignment) or independently reviewed motion. Rylo's public
gloss-to-pose baseline primarily performs lexicon lookup, cropping, concatenation,
and smoothing; its output can be an external renderer/baseline but not recognition
ground truth. Keep three data roles separate: genuine continuous phrase landmarks for
Stage 2, SignWriting/text pairs for reverse language planning, and generated poses for
native review only unless separately validated.

No SignWriting data or model was downloaded or admitted to a split in this decision.
The newly staged transition-safe coverage audit remains unrun, and no further reverse
generation was started while this architecture is being discussed.

## 2026-09-01 22:41 PST — genuine-entry fallback removes sampled jerk; full reverse demo passes

All four v6 high-jerk failures originated at the join into the next genuine clip, not
inside the learned transition. `synthesize_join` now tries the original boundary and,
only if its local speed/acceleration/jerk gate fails, may skip at most the first one or
two complete genuine entry frames. It never fabricates a frame, never changes hand
participation, and does not use a longer crop merely to make a numerical gate pass.
The composed right-hand lexical segment is reclassified after this preparation and
must retain its exact Stage-1 gloss.

The balanced 48-pair fast report is now
`artifacts/reports/stage2_v17_grounded_transition_scalability_fast_v7/report.json`.
All 48 boundaries pass motion, presence, hand-side, and bone gates. Overall 46/48
(95.83%) pass after adding the exact post-preparation lexical-label gate; `HOT LESS`
and `LEARN DAY` are rejected because P37's only complete-edge source is no longer
classified as the intended right gloss. The prior 44/48 result therefore improved
without hiding lexical damage.

The reserved unbiased confirmation audit is
`artifacts/reports/stage2_v17_grounded_transition_scalability_200_v2/report.json`:
178/200 (89.0%) pass all required gates, versus 159/200 (79.5%) in the previous
200-pair report. Motion alone passes 189/200 (94.5%); thirteen rows fail the new
post-preparation Stage-1 label safeguard, with two overlapping motion failures. Only
five of 200 accepted/rejected samples needed a one- or two-frame entry fallback. The
one-to-two-hand stratum remains weakest at 46/56 (82.14%). This is evidence of a
remaining source-clip/data-quality gap, not permission to trim 3–4 lexical frames.

The grounded text renderer now retains every Citizen candidate clip per signer/gloss
instead of silently overwriting all but the last. For each phrase it evaluates the
primary exact candidate plus at most two one-clip substitutions per gloss, and
`--allow-different-signers` now actually selects a same signer independently for each
phrase rather than requiring every signer to cover the union of all requested words.
Both source clips and post-join gloss segments must classify exactly. Local boundary
motion, anatomy, bone, complete-hand, and hand-participation checks remain hard gates.
The corpus-wide normalized-motion ratio is now a ranking diagnostic, matching the
scalability audit, because isolated Citizen clips upsampled into a phrase have much
lower high-order motion than the nine directly extracted 128-frame local phrases; it
must not veto an otherwise exact and locally plausible boundary.

The complete current reverse-path run is
`artifacts/reports/stage2_v17_grounded_text_input_demo_v2/`. All five supported English
inputs render successfully with exact source and composed gloss predictions and all
hard machine gates: `GOOD MORNING` (P40), `HELLO HOW YOU` (P51), `MY NAME` (P52),
`THANKYOU FRIEND` (P52), and `TOMORROW SCHOOL GO` (P31). Three pass the global motion
diagnostic; `MY NAME` and `THANKYOU FRIEND` do not, which is explicitly recorded rather
than hidden. The contact sheet was visually inspected and shows complete hands plus
stable face/body rigs with correct one-/two-hand changes. Repaired outlier boundary
frames are separately shown in
`artifacts/reports/stage2_v17_grounded_transition_repaired_outlier_review_v1/contact_sheet.png`.

`venv/bin/python -m unittest test.test_signing_voice_v17 -v` passes all 18 tests and
`git diff --check` passes. Generated artifacts remain native-review-only and are not
training, validation, or test data. No Citizen test or other sealed/test split was
accessed.

## 2026-09-01 22:19 PST — grounded joins now use complete real hand edges; fast gate reaches 44/48

The shared grounded phrase path now separates Stage-1 recognition trimming from
transition-only edge trimming. Stage 1 still receives the original observed isolated
span. Immediately before joining, `trim_transition_span` removes partial-hand edge
frames and retains the actual one-/two-hand participation. Transition stabilization
then reconstructs a complete 21-node hand only for a side genuinely present in at
least one neighboring sign; a side absent from both signs remains absent. This fixes
the floating-point/partial-hand rendering defect without forcing every sign to use two
hands. The face/body anatomy rig remains complete on every rendered frame.

The timing path retains its learned duration except for a narrow kinematic guard: if
the generated transition speed is below 0.25x the genuine neighboring-gloss p95, it
tries shorter allowed spans and accepts a replacement only when speed, acceleration,
and jerk all fall inside the existing 0.25x–4x gate. This changed `IMPORTANT WATER`
from 12 to 11 transition frames without regressing an accepted pair.

The same seeded 48-pair/four-stratum audit now passes 44/48 (91.67%), up from 40/48
(83.33%):
`artifacts/reports/stage2_v17_grounded_transition_scalability_fast_v6/report.json`.
All presence, no-invented-hand, handless-frame, transition-only-node, and hand-bone
gates pass on all 48. The four rejected high-jerk boundaries are `LISTEN EAT`,
`SICK MY`, `SICK COLD`, and `TRY COLD`; testing every learned allowable duration
(4–12 frames) found no duration that passed, so these must remain rejected rather
than admitted as generated data. The earlier direct hand-completion trial failed the
transition-only-node gate (6/48), and spherical/interior-time smoothing trials
regressed the motion screen; neither trial is retained.

Two native-review-only renders visually confirm the repaired structural behavior:
`artifacts/reports/stage2_v17_grounded_text_input_visual_fix_v1/` renders
two-handed `IMPORTANT` into one-handed `WATER`, while
`artifacts/reports/stage2_v17_grounded_text_input_visual_one_hand_v1/` keeps
`FIND THEY` one-handed throughout. Both reports pass boundary gates, preserve source
and rig hand participation exactly, keep every active rig hand at 21 nodes, and keep
face/body nodes complete. Contact sheets were visually inspected; native ASL motion
review is still required. These artifacts remain ineligible for recognition training,
validation, or testing.

`venv/bin/python -m unittest test.test_signing_voice_v17 -v` passes all 17 tests, and
`git diff --check` passes. No Citizen test split or other sealed/test split was
accessed.

## 2026-09-01 22:03 PST — 48-pair fast transition gate replaces blind batch rendering

The scalability audit now uses seeded random sampling within each hand-participation
stratum instead of lexicographically spaced rows. Its focused regression passes. A
fast gate of 12 samples from each of the four observed one/two-hand transition
patterns completed in approximately 12 seconds and passed 40/48 pairs (83.33%). The
report is
`artifacts/reports/stage2_v17_grounded_transition_scalability_fast_v1/report.json`.

All eight failures were `transition_motion_orders_in_gloss_range`; two also failed the
non-blocking global motion diagnostic. There were no missing-body, missing-hand,
forced-second-hand, presence, or hand-bone failures. Pattern pass rates were 75.0%
one-to-one hand, 83.33% one-to-two, 83.33% two-to-one, and 91.67% two-to-two. The
failed ordered pairs were `FIND THEY`, `LIKE MY`, `LISTEN EAT`, `FATHER BAD`,
`HE HUNGRY`, `ANGRY HOME`, `WHAT WATER`, and `MAKE READ`.

Use this 48-pair numerical screen before rendering phrase batches. Render only
accepted phrases; retain the unbiased 200-pair audit as a later confirmation gate.
This shortens iteration but does not relax native-signer review or make generated
artifacts eligible for training, validation, or testing. No sealed/test split was
accessed.

## 2026-09-01 21:57 PST — Rylo is useful architecture reference, not an authorized Stage-2 corpus

The public Rylo Translate frontend and legacy gloss-to-pose pipeline are available at
`https://github.com/sign/translate` and
`https://github.com/sign-language-processing/spoken-to-signed-translation`. The live
frontend is an Angular/Ionic PWA. For spoken-to-signed output it calls a hosted cloud
function that returns a `.pose` sequence, while a separate hosted model translates
text to SignWriting. A single normal service check of `How are you?` returned a
549,536-byte `application/pose` artifact and a token/lemma-style `x-glosses` header;
no Rylo artifact was copied into this repository or admitted to any dataset split.

This does not provide natural continuous ASL ground truth. Rylo's own technology map
marks fluent-pose synthesis as currently skipped/low quality, and the open baseline
crops, concatenates, and filters isolated dictionary poses. Its current source also
hides lowered-hand landmarks to avoid frozen floating hands. These are precisely the
render/presence and synthetic-transition assumptions that must not teach the Stage-2
recognizer. Hosted `.pose` outputs may be used only as qualitative external
comparators, never as training, validation, or test labels.

Two public data resources are separately downloadable: SignBank+ provides roughly
2.03 million text/SignWriting rows under CC BY-NC 4.0, and its text-to-SignWriting
Sockeye model is also CC BY-NC 4.0. They contain notation/text rather than continuous
signer video, signer identities, or natural inter-sign motion, so they may help a
future language/notation layer but cannot close the present Stage-2 transition gap.
The open gloss-to-pose repository additionally bundles ASL fingerspelling pose clips,
not a conversational phrase corpus.

Rylo Dictionary is more interesting but not currently ingestible. Rylo states that
Deaf experts contribute human-signed lexical videos, AI avatars anonymize/normalize
them, a second annotator reviews each retarget, and the video/concept links are
non-commercial-only. The site exposes no authorized bulk dataset export, and Rylo's
general Terms prohibit automated scraping/downloading and using the site to build a
similar or competing service. Do not scrape its signed media or enumerate its private
API. Request a written research data/API license from `research@rylo.com`, including
permission for model training, retention, redistribution of derived landmarks, and
publication, before any acquisition. Even with permission, these are isolated lexical
signs and must not be treated as natural phrase-transition evidence.

## 2026-09-01 21:25 PST — natural-text input reaches the gated grounded renderer

`active/v17/text_to_sign_phrase_catalog_v17.json` is the first explicit reverse-path
language contract. It maps five curated conversational English inputs and bounded
aliases to the already gated gloss sequences. The generator now accepts repeated
`--text` inputs, records the original text and catalog hash, renders both the text and
resolved glosses in each video, and preserves the full lexical/transition/anatomy gate
chain. Unsupported English fails before model or artifact work and lists the supported
normalized inputs; this prototype does not pretend that a five-entry catalog is a
general English-to-ASL translator.

The complete natural-text run is
`artifacts/reports/stage2_v17_grounded_text_input_demo_v1/`: `Good morning`,
`Hello, how are you?`, `My name`, `Thank you friend`, and
`I will go to school tomorrow` all resolve to the intended gloss sequences and pass
the same exact-content, one-hand, motion, bone, transition, face/body, and artifact
gates. `index.html` is the canonical native-review entry point. The source catalog
and all generated artifacts remain training/validation/test-ineligible, and native ASL
review is still incomplete. No sealed or test split was accessed.

## 2026-09-01 21:20 PST — five-phrase grounded scalability gate passes after hand-kinematic repair

The grounded path now passes every locally benchmarkable conversational phrase whose
literal glosses exist in the Citizen 100: `GOOD MORNING`, `HELLO HOW YOU`, `MY NAME`,
`THANKYOU FRIEND`, and `TOMORROW SCHOOL GO`. Batch generation may select a different
train-only signer for each utterance while keeping every utterance internally
same-signer. The selected signers are P50, P11, P37, P50, and P27 respectively. All
12 exact isolated components are correctly recognized by frozen Stage 1 (confidence
range 0.5635--0.9550), and source versus final-rig hand participation matches for
every gloss. Five independent selected-signer `SORRY` audits remain exactly one-handed
and classify correctly.

The first five-phrase artifact (`...scalability_v1`) is rejected despite its older
gates: contact-sheet inspection exposed brief hand-skeleton shrinkage inside several
transitions. The cause was linear interpolation of absolute finger coordinates across
different endpoint poses. `stabilize_transition_hands` now retains the learned wrist
trajectory but reconstructs all finger chains with interpolated endpoint-derived bone
lengths. It preserves the original node/side presence mask, so it cannot add a hand or
create nodes present only inside a transition. A regression test begins with a fully
collapsed transition, restores every right-hand bone above its endpoint floor, and
keeps the source-absent left hand exactly zero.

The replacement output is
`artifacts/reports/stage2_v17_grounded_text_to_sign_scalability_v2/`. Across the five
phrases, p95 speed is 0.927--1.140x, acceleration 0.856--1.150x, and jerk
0.680--1.080x aggregate genuine local train. Transition/gloss p95 speed is
0.594--1.836x; median transition/gloss hand-bone length is 0.707--1.933x. Every
transition has zero handless frames, zero nodes present only within the generated
span, complete render hands when active, and complete face/body on every render
frame. All five machine gates pass. `index.html` provides a native-review page and
`contact_sheet.png` provides the visual audit overview. These remain synthetic
native-review-only artifacts and are explicitly ineligible for training, validation,
or testing.

The train-only Citizen inventory contains 32 signers. P33, P37, and P52 each cover
all 100 glosses; consequently every unordered pair (4,950/4,950) and triple
(161,700/161,700) has at least one same-signer source route. This proves source
coverage for a 50-phrase expansion, not linguistic validity or transition quality.
New phrases must still be curated by native signers and pass the same per-phrase
recognition, motion, bone, presence, render, and native-review gates. No sealed or test
split was accessed.

## 2026-09-01 21:11 PST — grounded text-to-sign pilot passes machine gates; native review required

The first reviewable unseen-combination path now grounds lexical motion in exact
train-only ASL Citizen isolated clips from one real signer and generates only the
short boundary between adjacent signs. `scripts/generate_grounded_text_to_sign_v17.py`
discovers train signers covering every requested gloss, verifies each isolated clip
with the frozen landmark Stage-1 checkpoint, preserves each source gloss's own
left/right-hand participation, and uses the all-real transition inpainter plus the
multi-corpus timing model for the boundary. Signer selection uses exact component
recognition, source anatomy coverage, and distance to the aggregate genuine local
train motion distribution. It does not inspect either held-out phrase trajectory.

The selected signer is P40. Stage 1 correctly recognizes all five requested source
clips: `GOOD` 0.7847, `MORNING` 0.9150, `TOMORROW` 0.9049, `SCHOOL` 0.9715, and `GO`
0.9116. The independent `SORRY` audit is exactly one-handed (`[true,false]`) and is
recognized as `SORRY`, proving the path does not add a second hand. Per-gloss source,
generated-observation, and render-rig participation match exactly: `GOOD` and
`TOMORROW` remain one-handed; `MORNING`, `SCHOOL`, and `GO` use both hands.

Both unseen compositions pass the aggregate genuine-motion gate. `GOOD MORNING` p95
speed/acceleration/jerk are 1.446x/1.051x/1.092x genuine local train;
`TOMORROW SCHOOL GO` is 1.183x/1.083x/0.957x. Every transition contains a hand, has
zero presence changes at its joins, and introduces zero nodes absent at both boundary
endpoints. Transition/gloss p95 speed ratios are 0.524x and 0.731x. The v3 animation
contract makes face and body complete on every rendered frame and every active hand a
complete 21-node hand, while leaving an unused hand absent. Contact-sheet inspection
confirmed readable bodies and hands without between-sign disappearance.

The review artifacts and videos are under
`artifacts/reports/stage2_v17_grounded_text_to_sign_pilot_v1/`; the machine-readable
record is `report.json`. Every artifact explicitly sets training, validation, and test
eligibility to false. Passing these gates permits native-signer review only; it does
not establish fluent coarticulation and must not be used as recognition ground truth.
No Citizen test, local test, SemLex test, How2Sign validation/test, or other sealed
split was accessed.

## 2026-09-01 20:58 PST — four whole-utterance code priors rejected before rendering

Four free-running temporal-code priors were evaluated with `GOOD MORNING` and
`TOMORROW SCHOOL GO` excluded from training. The plain prior ranked the intended
phrases 7th/3rd; pooled-text/history-dropout v2 ranked them 3rd/8th; monotonic-gloss
timeline v3 ranked them 4th/6th and incorrectly made `TOMORROW SCHOOL GO` one-handed.
The v4 text-only code/side anchor corrected that hand-participation error and retained
the exactly one-handed `SORRY I LATE` audit, but semantics still ranked 5th/5th and
unseen motion fell to 0.34--0.49x genuine on at least one derivative order. All four
remain rejected and unrendered. The v4 checkpoint is
`artifacts/models/temporal_code_prior_v17_text_anchor_v4/model.pth`, SHA-256
`4f26aaad3bd42beba187bbcac8e26f0d0efb9ad1d202679d806452da1f475a3c`;
its rejection report is
`artifacts/reports/stage2_v17_temporal_text_to_sign_v4/evaluation.json`.

This falsifies whole-utterance autoregression with only weak/equal gloss alignment as
the immediate reverse-generation route. The safer architecture is exact per-gloss
content grounding plus a separately learned, short coarticulation model. The temporal
tokenizer remains accepted as a representation, but none of its rejected priors may
produce training data.

## 2026-09-01 18:12 PST — 64-step temporal tokenizer passes every unseen reconstruction gate

The accepted tokenizer doubles temporal resolution to 64 discrete code steps for 128
frames, increases coordinate weight only inside the reconstruction objective, and
retains EMA codebook updates plus class-balanced absence supervision. Its checkpoint
is `artifacts/models/temporal_motion_tokenizer_v17_v2/model.pth`, SHA-256
`efc6e138a397bb48352a4686b814de2fec902913379880329964d67b4857fc09`, selected at
epoch 38. Neither `GOOD MORNING` nor `TOMORROW SCHOOL GO` appeared in tokenizer
training.

All held-out reconstruction gates pass: coordinate loss 0.0320 (<0.04), presence F1
0.9827, exact phrase-level hand-participation accuracy 1.000, 51 active codes with
23.53 aggregate perplexity, and reconstructed p95 speed/acceleration/jerk at
1.030x/0.957x/1.075x genuine motion. Ordinary validation has coordinate 0.0275,
presence F1 0.9900, hand participation 0.9973, and 81 active codes. This proves the
discrete temporal representation can reproduce unseen genuine phrase trajectories
without pose averaging or adding an unused hand. It authorizes training a
gloss-conditioned temporal code prior, but does not yet authorize an unseen generated
phrase render.

## 2026-09-01 18:00 PST — temporal tokenizer smoke restores real motion range; full gate pending

`model_temporal_motion_tokenizer_v17.py` replaces one utterance-wide latent with 32
time-varying discrete codes across each 128-frame trajectory. Its convolutional
encoder/decoder reconstructs observation XYZ, detector presence, and confidence; it
does not use the animation rig. `train_temporal_motion_tokenizer_v17.py` excludes both
compositional holdouts from self-supervised fitting, balances genuine source families,
and blocks temporal-prior training unless reconstruction anatomy, coordinate, motion,
and code-utilization gates all pass.

The first three-epoch smoke was rejected because its learned-gradient codebook
collapsed to three active codes and its reconstruction marked every hand/body node
present. EMA codebook initialization and class-balanced absence supervision fixed the
root mechanisms. After five smoke epochs, reconstructed holdout p95 motion reached
0.675x genuine speed, 0.786x acceleration, and 0.801x jerk, passing all three motion
range gates and substantially exceeding the rejected global-latent generators. It no
longer forces all nodes present. The smoke still fails overall: holdout presence F1 is
0.847, hand participation is 0.917, coordinate loss is 0.066, and only 10/128 codes
are active. Therefore it remains non-renderable and cannot yet authorize prior
training. The next action is a full tokenizer run, not a gloss generator.

## 2026-09-01 17:55 PST — multi-source global-latent generator rejected; temporal latent needed

The source-balanced full-trajectory CVAE trained on 1,606 non-holdout train rows and
369 validation rows, reserving all 24 validation performances of `GOOD MORNING` and
`TOMORROW SCHOOL GO`. With KL weight 0.01 and prior-loss checkpoint selection, early
stopping retained epoch 3 of 10. The 1,544,946-parameter checkpoint is
`artifacts/models/full_trajectory_generator_v17_multisource_holdout_v1/model.pth`,
SHA-256 `a6a5ee3e1931e0a342cfdea3a0bca47de3dc225075f319bb2ee7de22823ca941`.
Ordinary prior validation presence F1 is 0.919 at its calibrated 0.30 threshold and
hand-participation accuracy is 0.966, but coordinates lose to the source-family mean.
On the compositional holdout, prior presence F1 is 0.802 and hand participation is
0.917; coordinate loss also loses to the family mean.

The strict 32-prior-sample motion gate rejected both unseen combinations. The
`GOOD MORNING` output produced only 0.028x genuine p95 speed, 0.035x acceleration,
and 0.038x jerk;
`TOMORROW SCHOOL GO` produced 0.026x, 0.034x, and 0.036x. This is substantially worse
than the local-only CVAE's already-rejected motion. No review render or training sample
was written. The evaluator result is
`artifacts/reports/stage2_v17_full_trajectory_generation_v1/evaluation.json`.

The diagnosed architecture bottleneck is the single mean-pooled 32-D motion latent,
which is broadcast identically to all 128 frame queries. It can encode a target for
posterior reconstruction but cannot supply time-varying motion to the text-only prior;
coordinate regression therefore averages performances into nearly static poses.
Mixing more genuine sources does not fix that bottleneck. The next justified model is
a temporal/discrete motion representation (motion tokenizer plus conditional temporal
prior, or a temporal-latent diffusion model), trained on all genuine trajectories as
an unlabeled motion prior while keeping gloss-conditioned local content and hand-side
participation explicit. This is a meaningful architecture replacement and awaits user
consultation. Rejected synthetic motion remains barred from every dataset and renderer.
All 18 focused generation/signing-voice tests pass, Python compilation succeeds, and
`git diff --check` is clean.

## 2026-09-01 17:50 PST — 2,095 genuine whole-utterance trajectories complete and audited

The whole-utterance Apple Vision extraction completed in 3,353 seconds: 2,094 new
archives plus one previously verified archive, zero failures, and exact coverage of
all 2,095 manifest rows. A full cold-read integrity audit found no missing archives,
no non-finite values, and exact `[128,61,5]` observation shapes and manifest metadata.
Counts are 780 local phrases, 1,104 ASLLRP `OTHER`, 155 2M-Flores, and 56 exact
ASLLRP spans. Split counts remain 1,702 train and 393 validation with the ASLLRP signer
boundary intact. Observed source participation is 60 right-only, 81 left-only, and
1,954 two-hand trajectories; these are detector observations, not rig-forced hands.
Mean observation presence is 84.27% (range 26.95--100%), mean duration is 4.53 s
(range 0.3--42.13 s), and the corpus occupies 89 MB. No generated motion or sealed
test source is present. The canonical extraction report is
`artifacts/reports/stage2_v17_full_trajectory_generation_v1/extraction.json`.

## 2026-09-01 17:24 PST — signer-profile transfer is presence-invariant; jerk is a render gate

`apply_voice_profile_to_trajectory` now transfers a learned isolated-signer spatial
profile onto an arbitrary-length continuous trajectory without modifying detector
presence or confidence. Coordinates are transformed only where the source trajectory
is present; absent nodes remain exactly zero. Therefore this style/accent operation
cannot invent a second hand for a one-handed sign. Its resampled temporal curve is
disabled by default until continuous-motion evaluation justifies using it. A focused
128-frame one-handed regression test proves the unused hand remains absent and all
auxiliary channels are bit-identical; all 17 signing-voice and full-trajectory tests
pass at that step. The unseen-generation evaluator now selects candidate latents using speed,
acceleration, and jerk together and independently requires every generated holdout to
remain within 0.25--4x the genuine held-out p95 for each measure. This prevents a
right-speed but snapping trajectory from passing. All 18 focused tests pass after the
new three-order motion gate. This only establishes the representation safety invariant, not that generated
signer identity or motion is perceptually genuine. The whole-utterance extraction is
still live at 1,375/2,095 rows with one existing skipped archive and zero failures.
Re-evaluating the strongest local-only CVAE smoke under the stricter gate rejected it:
`GOOD MORNING` reached 0.239x genuine speed, 0.305x acceleration, and 0.279x jerk;
`TOMORROW SCHOOL GO` reached only 0.112x, 0.104x, and 0.107x respectively. It also
lost to the source-family mean on validation/holdout coordinates and holdout velocity.
The machine-readable rejection is
`artifacts/models/full_trajectory_generator_v17_local_cvae_kl_smoke/evaluation_motion_gate.json`;
rendering remains disabled.

Checkpoint selection for future full-trajectory runs now uses
`validation_prior.loss`, not target-conditioned posterior reconstruction loss. This
aligns early stopping with the actual text-to-sign inference path; posterior metrics
remain diagnostic only. The focused full-trajectory tests pass. Extraction is at
1,425/2,095 with zero failures.

## 2026-09-01 17:19 PST — whole-utterance conditional generation corpus/model underway; unsafe smokes rejected

The reverse path no longer uses independently normalized 32-frame pieces.
`scripts/prepare_full_trajectory_manifest_v17.py` combines every available labeled
real clip from all nine local phrase families, ASLLRP exact spans, ASLLRP `OTHER`
spans, and 2M-Flores into
`active/v17/full_trajectory_generation_manifest_v17.json`: 2,095 rows, 454 literal
gloss/special tokens, 1,702 train and 393 validation. Source roles remain unchanged;
ASLLRP validation is signer-held-out, local validation is familiar-source, and 2M dev
remains train-role only. Local `FOOD`, `LATE`, `TEACHER`, and `MEET` stay literal and
are not mapped into Stage 1. The extractor writes one 128-step trajectory per complete
video under one coordinate normalization. Its active run has reached 1,125/2,095 with
zero failures; all 780 local clips are complete.

`model_full_trajectory_v17.py` and `train_full_trajectory_v17.py` implement a
gloss-conditioned frame decoder with coordinate, detector-presence/confidence,
velocity, acceleration, jerk/scale, hand-participation, and duration supervision.
`GOOD MORNING` and `TOMORROW SCHOOL GO` are excluded entirely as compositional
holdouts because all five component glosses occur independently outside those exact
sequences. `HELLO HOW YOU` was not used as the holdout because `HELLO` has only one
training occurrence outside that phrase, which would confound composition with a
missing-token problem.

The first deterministic local-only smoke was rejected: although ordinary validation
coordinates beat a source-family mean, unseen coordinates did not, participation was
91.7%, and every hand node appeared in every generated frame. Stronger negative
presence and temporal-change loss removed the all-present collapse, but generated
speed remained only 10--11% of genuine holdouts. A conditional motion-latent encoder
was then added because point regression averages distinct performances into slow
motion. With posterior motion evidence, the local latent smoke reached 90.9% held-out
presence F1 and 97.9% participation, proving representation capacity; honest prior
samples reached 25.2% of genuine speed for `GOOD MORNING` but only 11.2% for
`TOMORROW SCHOOL GO`, and coordinate gates still failed. All these smokes remain
non-renderable. `evaluate_full_trajectory_generator_v17.py` now rejects all-present,
static-presence, slow-motion, participation, coordinate, and velocity failures before
any unseen-combination video can be written. No synthetic result entered any dataset
or review video.

## 2026-09-01 16:46 PST — v3 preserves one-handed signs; safe exact text-to-sign path added

The user correctly rejected the v2 rig assumption that every displayed sign should
contain two hands. `complete_landmark_anatomy` now preserves linguistic hand-side
participation: a hand never observed in the source remains exactly absent, a
participating hand bridges at most three missing detector frames, and longer inactive
spans remain absent. Face/body completion remains render-only. Artifact contract v3
stores explicit `animation_rig_presence` and `animation_rig_confidence` alongside XYZ;
loaders reject v2/legacy artifacts, require rig and observation hand participation to
match, and retain the hard training/validation/test-ineligible flags.

The rebuilt train-only anatomy package is
`artifacts/models/signing_landmark_anatomy_v17_v3/anatomy.npz`, SHA-256
`fdd286f73e49a69d686b7b014b9d736192b29b818f22d575cc1e8efca29e0f06`.
It preserves exact participation for all 100 gloss prototypes: 48 selected sources are
one-handed and remain one-handed; 52 are two-handed and remain two-handed. Both sparse
observation and v3 rig content accuracy are 100% on this train-only content gate. V2 is
explicitly marked obsolete and must not be used.

The all-nine genuine reference was regenerated under v3 after auditing every train
recording and choosing, within each phrase's top detection-quality quartile, the take
closest to median duration. `SORRY_I_LATE` is one-handed in this selected performance
and remains `[left=false, right=true]` in the rig; every other phrase's observed/rig
side participation also matches exactly. `scripts/render_text_to_sign_retrieval_v17.py`
adds an honest reverse baseline: plain text such as `hello how you` resolves to an
exact known local phrase and renders its genuine full trajectory as a landmark-only
skeleton. Unknown combinations fail closed instead of invoking the rejected isolated
stitcher. The example is
`artifacts/reports/stage2_v17_text_to_sign_retrieval_v1/phrase.mp4`. Twenty-five
focused tests pass and `git diff --check` is clean. This is retrieval, not yet unseen
phrase generation.

## 2026-09-01 14:47 PST — genuine all-nine phrase render establishes the visual baseline

`scripts/render_genuine_local_phrase_reference_v17.py` now selects the longest
train-role recording in each of the nine local phrase families and extracts each whole
performance under one coordinate normalization. It does not stitch independently
normalized windows, concatenate isolated signs, or generate transitions. The 53.9 s
side-by-side reference video is
`artifacts/reports/stage2_v17_genuine_local_phrase_reference_v1/genuine_local_phrase_reference.mp4`
(SHA-256 `22bf2ae1c159eabb4afcc4ae2dfc199ece6ed077f98f9283ec95f71da9093040`).
The left panel shows sparse detector observations; the right shows a render-only
completed rig driven by the exact same genuine motion. The report folder also contains
a nine-phrase contact sheet, machine-readable report, and one separated raw artifact
per phrase.

Across these full phrase references, real observation presence ranges from 60.3% to
79.5%. Every artifact preserves observed XYZ exactly after float16 serialization
(maximum measured error 0), retains observation masks separately, contains no
ambiguous `landmarks` tensor, and is explicitly reference-only/ineligible for dataset
splits. Twenty-two focused tests pass and `git diff --check` is clean. This proves the
render path can keep the avatar visible without altering recognizer evidence. It does
not prove novel-combination generation, learned full-utterance coarticulation, or
native linguistic naturalness; rejected synthetic generation remains disabled pending
review of this genuine-motion baseline.

## 2026-09-01 14:44 PST — render rigs can no longer masquerade as recognizer observations

The landmark artifact architecture now separates two representations. Detector-style
`recognition_prototypes` retain real sparse presence/confidence, while
`animation_rig_prototypes` are completed only for drawing. The anatomy builder writes
the v2 contract without the old ambiguous `content_prototypes` key. The phrase renderer
requires v2 and saves `animation_rig_xyz`, `observation_xyz`,
`observation_presence`, and `observation_confidence` separately; it never writes a
recognition-shaped `landmarks` tensor. Every synthetic file is explicitly ineligible
for training, validation, and testing. The review builder fails closed on legacy
ambiguous files and treats presence changes/missing nodes as detector evidence rather
than structural failures. This implements an architectural contamination barrier; it
does not make the current isolated-medoid phrase composer human-natural.

The train-only v2 anatomy package is
`artifacts/models/signing_landmark_anatomy_v17_v2/anatomy.npz`, SHA-256
`0098c0e2b2fd8d66edf73344b3e5bc612c4e6d33b9f112f988b6a44a1feb79ee`.
All 100 selected prototypes retain their correct class in both observation and rig
form. Genuine observation presence is 70.44%; only the render rig is 100% complete.
No held-out/test source was accessed. Eighteen focused anatomy, artifact-contract,
transition, and signing-voice tests pass. Novel phrase generation remains disabled;
the next gate is rendering a genuine local full-phrase trajectory through the
separated observation/rig path.

## 2026-09-01 14:31 PST — source-balanced all-real transition adaptation improves local motion without guard regression

The remaining downloaded domain reference is now compatible too. The three retained
OpenASL train-split clips were converted by
`scripts/prepare_openasl_transition_manifest_v17.py` and the standard extractor with
zero failures. One channel-proxy clip is train-role and two are validation-role; all
three preserve their original acquisition split, and OpenASL validation/test video was
not accessed beyond those already retained train-split derivatives.

`TransitionWindowDataset` now optionally reads either continuous-transition archives
or the existing Stage-2 multimodal v17 archives, filters roles and sources before
preloading, and accepts sources without signer metadata. Existing callers retain their
old narrow glob and behavior. `active/v17/train_transition_all_real_v17.py` reuses the
frozen 2.02M-parameter transition inpainter and adapts it with equal probability over
six genuine corpus families: all-nine local phrases (1,781 train windows), ASLLRP old
+ `OTHER` (3,761), 2M-Flores (2,686), How2Sign (4,938), one NCSLGR training signer
(424), and YouTube/OpenASL web motion (810). No synthetic trajectory participates.

Model selection uses 445 familiar-source local validation windows spanning all nine
phrases, 801 signer-held-out ASLLRP `OTHER` windows, and 137 windows from the held-out
NCSLGR signer. The full 4,938 How2Sign and 802 train-side YouTube windows are replay
guards; a candidate is rejected if any selection/guard domain regresses more than
0.005 relative reconstruction improvement from the frozen initialization. The 15
OpenASL validation windows are reference-only because that sample is too small for
selection.

The bounded run stopped after eight epochs by patience and selected epoch 4. Relative
masked-reconstruction improvement over endpoint interpolation changed from 16.07% to
20.74% on local (+4.67 points), 23.08% to 23.50% on ASLLRP, 20.41% to 21.27% on
held-out NCSLGR, 25.87% to 25.41% on the How2Sign guard (within the predeclared
0.5-point tolerance), 10.15% to 10.74% on the web guard, and 0.43% to 1.79% on the
tiny OpenASL reference. The selected checkpoint is
`artifacts/models/transition_all_real_v17_v1/model.pth`, SHA-256
`0faaa4f8289257fed02ba9f17c1fc1b5fd8035f921ff500292644efd2eb44123`;
training took 71.0 seconds. This is materially better genuine local-motion
reconstruction and demonstrates that using all sources with explicit balance works.
It is still a masked 4--12-frame motion model, not gloss-conditioned full-phrase
generation or a human-naturalness pass. Four focused audit/dataset tests pass. No
project or public test split was accessed.

## 2026-09-01 14:25 PST — all nine local phrase families now have v17 motion-only trajectories

The user authorized using all genuine downloaded sources for experimentation while
phrase generation remains stopped. `scripts/prepare_local_phrase_motion_manifest_v17.py`
now pins all 780 local videos as motion-only continuous rows. It preserves the existing
20-recording capture batches and every-fifth-repetition familiar-source validation
contract, yielding 624 train and 156 validation recordings across all nine phrase
families. The folder phrase is provenance only; target sequences are empty, so
`FOOD`, `LATE`, `TEACHER`, and `MEET` are not added to Stage 1 and no `FOOD -> EAT`
mapping is implied.

The standard v17 continuous extractor completed all 780 videos in 506.9 seconds with
zero failures and schema fingerprint `b872fa3dcc16aab5`. The new tree is
`data/local/local_phrase_motion_landmarks_v17`; its manifest is
`active/v17/local_phrase_motion_manifest_v17.json` (SHA-256
`a2b5eab75f61cb0dffae7ce7ac89bb56c6ad7150cf85ab4b84184d0a58c9388f`) and its
extraction report is
`artifacts/reports/stage2_v17_real_motion_reference_audit/local_phrase_motion_extraction.json`.
There are 2,226 valid 32-frame trajectories out of 2,349 attempted windows: 1,781
train-side and 445 validation. Every phrase has valid motion on both roles, including
the previously uncached I_WANT_FOOD (144/35 train/validation windows), SORRY_I_LATE
(298/75), and YESTERDAY_TEACHER_MEET (169/39). The validation rows are familiar-source
repetitions because signer identity is unavailable; they are useful engineering gates
but not signer-generalization evidence. Three focused manifest/audit tests pass. No
sealed or public test split was accessed.

## 2026-09-01 14:13 PST — generation stopped; all genuine sources expose anatomy/motion design failure

At the user's direction, phrase generation was stopped before any further training or
rendering. No generation or Stage-2 training process remains active. Both generated
review pilots are now explicitly rejected and quarantined by `REJECTED.md` files. V1
loses or fragments detected bodies/hands; v2's attempted anatomy completion forces all
61 detector-presence values to one while its motion remains visibly nonhuman. Neither
pilot may enter recognition training, validation, or testing.

The new reproducible audit is
`artifacts/reports/stage2_v17_real_motion_reference_audit/README.md`, backed by
`audit.json`, `reconstruction_transfer.json`, and
`scripts/audit_real_motion_reference_v17.py`. All nine local phrase folders were
visually sampled in `local_phrases_contact_sheet.png`. Their 780 raw clips comprise
GOOD_MORNING 60, HELLO_HOW_YOU 140, I_WANT_FOOD 60, MY_NAME 60, PLEASE_HELP_ME 140,
SORRY_I_LATE 140, THANKYOU_FRIEND 60, TOMORROW_SCHOOL_GO 60, and
YESTERDAY_TEACHER_MEET 60. The existing v16-era archives cover all 780 for
reference-only presence analysis; only the six vocabulary-compatible phrases have
current v17 Stage-2 archives (390 train clips / 1,144 valid windows). The v16 arrays
remain prohibited from v17 training.

The compatible real-data experiment used every currently extracted train-side source,
without accessing held-out or sealed splits: 390 local clips, 44 older exact-target
ASLLRP clips, 879 ASLLRP `OTHER`-CTC spans, 155 2M-Flores `dev` clips, 1,026 How2Sign
train clips, 166 NCSLGR clips, and 103 train-side YouTube-ASL channel proxies. The
three retained OpenASL clips were included in the inventory as raw visual/domain
references but could not be scored because no compatible v17 extraction exists.
How2Sign/YouTube English metadata were not promoted to ordered gloss truth.

Detector-presence distributions prove that v2 would poison recognition training. All
61 nodes are present in 100% of v2 frames, versus 54.6% across pooled genuine v17
train-side frames and 0% across local v17 phrase frames. V2 also suppresses genuine
dynamics: its hand speed/acceleration/jerk p95 values are 0.1843/0.2291/0.3523, while
the genuine-source p95 ranges are 0.3269--0.5497 / 0.4652--0.8063 /
0.8080--1.3894. Low jerk here is under-articulation, not evidence of human smoothness.
The local videos visibly contain preparation, overlapping articulation, continuous arm
travel, retraction, and rest that isolated medoids plus a short inpainted gap cannot
represent.

`scripts/evaluate_transition_real_sources_v17.py` then froze the existing
How2Sign+YouTube transition checkpoint and masked a deterministic 4--12 frame interval
in every compatible genuine window. Relative reconstruction improvements over endpoint
interpolation were positive on all sources: local 14.0%, older ASLLRP 29.0%, ASLLRP
`OTHER` 21.7%, 2M-Flores 21.0%, How2Sign 26.0%, NCSLGR 27.7%, and YouTube train 7.7%.
However only 50.1% of local windows improved, the weakest practical transfer result;
the current inpainter is therefore not ready to drive local phrase generation. These
are train-side masked-reconstruction diagnostics, not independent naturalness or
text-to-sign evidence.

The architectural correction is now locked for consultation before more generation:
use all genuine corpora with source-balanced sampling, learn/retrieve full continuous
phrase trajectories rather than joining isolated medoids, keep semantic/gloss
supervision separate from motion-only self-supervision, and split the product into a
persistent kinematic animation rig plus a separate learned detector-observation mask.
An always-present internal rig is appropriate for rendering but must never be written
into recognition features. Two focused audit tests pass. No Citizen, SemLex, local
sealed, RIT, How2Sign validation/test, YouTube internal validation, or 2M-Flores
`devtest` split was accessed.

## 2026-09-01 13:43 PST — native visual review rejects pilot anatomy continuity despite prior structural gate

The user reviewed the ten generated landmark-avatar videos and reported disappearing
bodies, disappearing/fragmentary hands, and a frequent single-hand appearance. The
pilot is therefore rejected for training and remains quarantined as synthetic review
evidence. Its previous machine pass was necessary but insufficient: it checked that at
least one hand node existed on every transition frame and that presence did not change
at the exact splice, but did not require complete 21-node hands, a persistent torso,
or stable handedness/anatomy throughout each gloss and transition.

Targeted tracing localizes the root cause upstream of the avatar renderer. The final
profile package stores one observed train-only medoid clip as the content prototype for
each gloss. `build_prototypes` minimizes XYZ error only over nodes present in a
candidate and has no coverage/completeness penalty, so a sparsely detected clip can be
selected as the medoid. `apply_voice_profile` modifies only XYZ and deliberately
preserves the prototype's presence/visibility channels. `trim_observed_span` trims on
any observed hand node, not on a complete hand/body criterion. The transition
composer's endpoint-anchored presence prevents new pop-in at the splice but necessarily
propagates missing source anatomy. Finally, the abstract renderer draws the torso only
when both shoulder nodes are present and draws each hand bone only when both endpoint
nodes are present; sparse masks therefore become visibly missing bodies and fragmented
wireframe hands.

The pinned prototypes confirm this directly. `HELLO` has no complete left-hand frame,
only 26/32 complete right-hand frames, and 0/32 complete-body frames; `I` has no
complete right-hand or complete-body frame; `NEED` has no complete left-hand or
complete-body frame; `READY` has a complete four-node body in only 1/32 frames. Some
active prototype frames contain only 5--10 of the 42 possible hand nodes. Across the
rendered pilot, the fraction of frames with a complete body is only 11--45% for nine
of ten phrases (the ASLLRP-heavy `GOOD NIGHT FAMILY` is 87%). `I NEED WATER` contains
both hands concurrently in 0% of its rendered frames because all three chosen medoids
carry only one observed/active hand. One-handed lexical signs may legitimately use one
active hand, but a human avatar should still retain the inactive hand and body rather
than treating nondetection as anatomical absence.

No generator fix or rerender was made during this diagnosis. The correct next fix is
to rebuild content prototypes with coverage-aware source selection and a persistent
avatar anatomy contract, distinguishing an unobserved/inactive landmark from an absent
body part. The revised native-review gate must measure full-hand and body continuity
over every frame, not merely any-hand presence at transition boundaries. No test split
was accessed.

## 2026-09-01 13:28 PST — first 10 generated phrase videos ready for native review

The first ten rows of `stage2_generated_phrase_review_plan_v17.json` were rendered as
30 complete synthetic landmark trajectories: ten phrase videos, each showing the same
sequence in the three novel profile voices Aster, Cobalt, and Juniper. The review
directory is `artifacts/reports/stage2_v17_generated_phrase_review_pilot_v1/`; its
`index.html` is a playable local gallery, `README.md` links every MP4, `review.csv`
contains one blank native-review row per phrase/voice, `contact_sheet.png` provides a
visual overview, and `manifest.json` pins all generation reports, raw landmarks,
videos, previews, source-voice mixtures, transition spans, hashes, and diagnostics.
Every output is labeled `synthetic_native_review_only`, validation/test-ineligible,
and human-review-required.

The first render audit found a genuine composition defect: isolated prototypes retained
their extractor padding, so several generated transitions began and ended on empty
frames, and the learned auxiliary presence output could make hands or body landmarks
appear only inside a generated interval. Those renders were overwritten and were not
admitted to the review bundle. `signing_voice_phrase_v17.py` now trims every isolated
prototype to its observed hand-motion span before duration resampling. Generated XYZ
coarticulation still comes from the frozen How2Sign/web transition inpainter, while
presence and visibility are anchored by interpolation between the two observed
endpoints. This prevents a landmark from flashing into or out of existence solely
inside the synthesized span without replacing learned spatial motion with a direct
join.

The corrected 30 voice/phrase trajectories all pass the bounded machine gates: every
isolated content prediction matches its requested gloss; every learned transition has
at least one observed hand on every frame; no landmark presence changes at either join;
and no node appears only inside a transition when absent at both endpoints. The ten
H.264 MP4 files independently decode at 1920x900 and 30 fps. Thirteen signing-voice
tests and the new review-auditor test pass. The profile and transition packages also
cold-reload and retain their pinned hashes. These structural results do not prove ASL
naturalness, grammatical acceptability, or absence of perceptual jerk; native review
is now the required gate before any generated item may be copied into training. No
Citizen, SemLex, local sealed, RIT, 2M-Flores `devtest`, How2Sign validation, or
How2Sign test split was accessed.

## 2026-09-01 13:08 PST — matched public-data Stage-2 ablation favors conservative 2M + ASLLRP transfer

All currently usable recommended data were already acquired, so no duplicate corpus
download was performed: 2M-Flores has 155 selected `dev` videos and frozen features;
ASLLRP has 1,104 `OTHER`-CTC spans (879 train / 225 signer-held-out validation) and
frozen features; NCSLGR has 166 utterances from two native signers; and the How2Sign
train-only subset has 1,027 acquired clips, 1,026 usable, across six source signer IDs.
NCSLGR and How2Sign remain transition/self-supervision sources because the current
artifacts do not provide compatible exact-variant ordered CTC targets. DSP video is
not downloadable under the published BU terms, Apple's annotation bundle was not
located, and ASL Homework remains access-gated. No large unlabeled corpus was added.

`train_stage_2_other_ctc_v17.py` now accepts the existing
`slt_stage2_temporal_pretrain_v17` checkpoint as a safe initialization. It verifies the
original teacher hash and matching configuration, proves that temporal pretraining did
not change the CTC head, optionally interpolates only the non-head state through
`--temporal-mix`, and keeps the original 100-class checkpoint as the distillation
teacher. The checkpoint/report record complete temporal provenance. Four focused tests
pass, including a 25% interpolation and original-teacher separation test; Python
compilation and `git diff --check` pass.

Full-strength temporal initialization followed by the original high-rate adaptation
was rejected: every trained epoch violated an old-data guard, and its selected epoch 8
had 12/259 local, 13/24 older-ASLLRP, and 365/682 new-ASLLRP full-sequence edits.
Initialization screens at 25%, 50%, and 75% showed that 50% is the strongest mix that
preserves both legacy gates before training. The original training recipe at 50% was
also rejected because epoch 0 remained selected and all trained epochs forgot old
phrases.

A predeclared conservative matched A/B then froze the backbone for all ten epochs,
used a `1e-4` head learning rate, raised replay distillation to 1.0, reduced new ASLLRP
sampling mass to 0.25, and used identical data, seed 1701, and selection gates in both
runs. ASLLRP-only selected epoch 5 with 617/682 new-ASLLRP full edits (90.47% WER),
450/284 target-only edits, 0/225 full exact sequences, 11/24 older-ASLLRP edits, and
7/259 local edits. The 50%-2M + ASLLRP combination selected epoch 6 with 542/682 full
edits (79.47% WER), 365/284 target-only edits, 4/225 full exact sequences, the same
11/24 older-ASLLRP edits, and an improved 6/259 local result. This is 75 fewer full
edits (12.2% relative) and 85 fewer target-only edits (18.9% relative) than the matched
ASLLRP-only run. Both saved artifacts reproduced their metrics exactly after cold CPU
reload.

The combined result is retained only as a research candidate, not promoted for app
deployment or autonomous generation: 79.47% full WER and 1.78% full exact-sequence
accuracy remain weak. It demonstrates useful cross-corpus temporal transfer and shows
that optimization/forgetting is part of the Stage-2 failure, but genuine signer and
transition supervision remain inadequate. How2Sign/NCSLGR-generated combinations must
be exported only as provenance-tracked `synthetic` review candidates; native approval
is required before training use, and generated samples remain prohibited from
validation/test. The detailed comparison is in
`artifacts/reports/stage2_v17_multidata_ablation_v1/EXPERIMENT.md`. No Citizen, SemLex,
local sealed, RIT, 2M-Flores `devtest`, How2Sign validation, or How2Sign test split was
accessed.

## 2026-09-01 12:48 PST — online overlap survey completed before further generation work

At the user's request, phrase generation and additional Stage-2 training are paused
while public datasets overlapping the locked Citizen-100 vocabulary are ranked. The
full evidence table is
`artifacts/reports/stage2_v17_online_overlap_survey/ONLINE_DATASET_OVERLAP.md`.

The best immediately usable ordered-gloss source remains the already acquired
2M-Flores-ASL `dev` material: 95/100 normalized lexical labels across 811/999
target-bearing sentences, with only `GOODBYE`, `PLEASE`, `SAD`, `SORRY`, and
`TOMORROW` absent. Its signer field is not adequate for a new signer-disjoint claim
(997 rows use local ID 0 and two use ID 1), and lexical strings are not proof of the
pinned ASL-LEX variants. The best exact-variant source remains ASLLRP continuous:
53/100 pinned variants and 1,483 target tokens in the 1,104 spans already acquired.
The already acquired NCSLGR static subset contributes only 17/100 lexical labels and
198 target tokens from two native signers.

DSP Sentences is the strongest newly quantified lead if permission can be secured. Its
official metadata contains 3,172 continuous sign tokens from 15 signers. Exact joining
through the pinned ASL-LEX `SignBankAnnotationID` finds 50/100 variants, 281 target
tokens, 218 target-bearing utterance files, and at least one target token from every
one of the 15 signer codes. Current BU documentation explicitly excludes
DawnSignPress data from downloadable video, so this is a permission/contact candidate,
not an immediately usable video corpus. RIT sentence metadata shows 30/100 exact
variants and 236 target tokens, but RIT remains permanently excluded from development
because its external evaluation was already consumed. This survey read only the
already retained RIT CSV to report corpus-level overlap; it did not access RIT video,
features, predictions, or metrics.

Apple and Gallaudet's April 2026 paper is the highest-upside future source. It reports
nearly 500 manually glossed ASL STEM Wiki videos, 8,655 sign annotations, 16 signers,
411 unique glosses in the ASL Citizen dictionary, and over 300 hours of
pseudo-annotations. The promised annotation data were not found on the Apple page, in
the arXiv source bundle, or in the current Microsoft ASL STEM Wiki repository, so
locked-100 overlap cannot yet be computed. The paper's own review distinguishes
native/native-like signers from multiple L2 signers; future ingestion must preserve
that split rather than pool all 16. How2Sign still exposes video/keypoints plus English
translations rather than downloadable ordered gloss targets and remains suitable only
for transition/self-supervised objectives under the current contract.

For isolated data, online lexical coverage is high but does not solve Stage 2: WLASL
has 99/100 lexical strings (`HE` absent), MS-ASL has 95/100 exact lexical strings or
96/100 with `BYE` treated only as a candidate alias, Sem-Lex official train metadata
maps 98/100 ASL-LEX-linked classes, and the bounded ASLLVD selection already covers
52/100 exact variants. These sources can reinforce Stage 1 or signer style, but they
contain no genuine phrase transitions. No project test split or test video was
accessed, and no new corpus video was downloaded.

## 2026-09-01 12:32 PST — Stage 2 data, collection, and generation plan locked

The user approved the following execution order: finish and audit the newly prepared
1,104-span ASLLRP `OTHER`-CTC expansion; train it together with the existing real
phrase replay and train-only synthetic replay on MPS; then use the already acquired
How2Sign train-only material to learn transition motion and timing; then compose new
locked-vocabulary phrase combinations; finally retrain Stage 2 and evaluate only on
genuine signer-disjoint recordings. Long MPS runs are authorized. The generated
review artifacts must be returned as a directory for native-signer inspection.

The vocabulary remains the exact pinned Citizen/ASL-LEX 100-class inventory. A single
explicit `OTHER` class is added only to the Stage-2 CTC head (101 nonblank classes,
blank index 0, `OTHER` CTC index 101), and `OTHER` is removed after CTC collapse. No
new lexical class or numeric/lexical variant is silently merged. The 1,104 natural
ASLLRP spans are combined with the old real local/ASLLRP phrase replay and the selected
train-only multivoice pool. Synthetic phrases are training-only: they must never enter
validation, sealed testing, or model-selection truth.

The bounded genuine collection remains 30 conversational gloss sequences, 13 native
ASL signers, three independent normal-speed performances per signer and phrase, and
three synchronized phone views per performance. Thus the planned capture is 1,170
independent performances and 3,510 videos; the three camera recordings of one
performance are correlated views, not three independent takes. Capture should include
the intended varied front/oblique angles at comparable quality. The preferred
signer-disjoint split is 9 training, 2 validation, and 2 sealed test signers. Every
signer identity and synchronized performance group must stay in exactly one split.
All 13 may record all 30 prompts, but the six composition-test prompts below remain
quarantined from training even when recorded by a training signer.

The approved 30-prompt collection inventory is:

| Role | ID | Exact gloss sequence |
| --- | ---: | --- |
| real base/train | 01 | `HELLO HOW YOU` |
| real base/train | 02 | `GOOD MORNING` |
| real base/train | 03 | `GOOD NIGHT` |
| real base/train | 04 | `THANKYOU FRIEND` |
| real base/train | 05 | `PLEASE HELP I` |
| real base/train | 06 | `I NEED HELP NOW` |
| real base/train | 07 | `YOU NEED HELP PLEASE` |
| real base/train | 08 | `WHAT YOUR NAME` |
| real base/train | 09 | `MY NAME` |
| real base/train | 10 | `I UNDERSTAND YOU` |
| real base/train | 11 | `I NO UNDERSTAND` |
| real base/train | 12 | `I KNOW MY FAMILY` |
| real base/train | 13 | `I NO KNOW WHERE HOME` |
| real base/train | 14 | `WHERE HOSPITAL` |
| real base/train | 15 | `WHERE DOCTOR` |
| real base/train | 16 | `I SICK` |
| real base/train | 17 | `I HUNGRY WANT EAT` |
| real base/train | 18 | `I WANT WATER PLEASE` |
| real base/train | 19 | `I WANT EAT NOW` |
| real base/train | 20 | `YOU READY GO` |
| real base/train | 21 | `WAIT PLEASE` |
| real base/train | 22 | `STOP PLEASE` |
| real base/train | 23 | `I GO HOME NOW` |
| real base/train | 24 | `SEE YOU TOMORROW` |
| held-out recombination | 25 | `HELLO HOW YOU READY` |
| held-out recombination | 26 | `SEE YOU NEED HELP` |
| held-out recombination | 27 | `HELP I WANT WATER` |
| held-out recombination | 28 | `I UNDERSTAND YOU NEED HELP` |
| hard unseen transition | 29 | `I NEED DOCTOR` |
| hard unseen transition | 30 | `I GO HOSPITAL` |

These are exact model gloss prompts, not claims about English word order. Native
signers must use the pinned lexical variants and may flag a prompt as linguistically
unnatural before capture; any approved replacement must remain inside the locked 100
and be recorded in this handoff before data collection. Prompts 01--24 provide the
real transition base. Prompts 25--28 test whole-sequence recombination from learned
parts, while 29--30 deliberately test transitions absent from the base inventory.

How2Sign is not a replacement for this phone collection. The local bounded subset has
1,027 train-only rows from six How2Sign signers and deliberately empty gloss targets,
so it is suitable for self-supervised transition inpainting, duration, rhythm, and
nonmanual/body-context learning but cannot directly supervise the locked-100 CTC
sequence. The already completed train-all transition package combines 4,938 How2Sign
windows with 994 train-only web windows. Its held-out How2Sign reconstruction gain is
20.2012%, its generated-vs-real discriminator remains above chance (61.1289% balanced
accuracy, AUC 0.648197), and its timing model reaches 92.3887% exact duration accuracy
with 0.1721-frame MAE. These are useful machine gates, not proof of human naturalness.

Phrase generation must preserve complete recognizable gloss cores and synthesize only
the missing coarticulation interval, conditioned on genuine left/right context and the
predicted 4--12-frame span. Direct concatenation, linear interpolation, unexplained
hand appearance/disappearance, and out-of-distribution position, velocity,
acceleration, jerk, bone geometry, or presence changes are rejection conditions.
Generated samples must retain the requested Stage-1/Stage-2 content before they may be
used for training. The first expansion target is 20 generated combinations, producing
a 50-phrase training inventory in conjunction with the 30 real prompts; this is a
training inventory only, and support for a generated combination may be claimed only
after it is recognized in a genuine held-out recording. Native review follows the
first training/generation run rather than blocking it.

The ASLLRP expansion's remaining preprocessing step is now complete. The frozen
selected Stage-1 encoder cached all 1,104/1,104 archives on MPS in 52.78 seconds with
zero failures, a 12% process cap, and 121,847,808 peak MPS driver bytes. The independent
frozen-input audit reports 1,104 expected/actual archives, all 4,505 windows valid,
879 train plus 225 signer-held-out validation rows, no unexpected archives, and a
valid 3,978-item/100-class train-only multivoice pool with 18,000 planned sequences.
Reports are `artifacts/reports/stage2_v17_asllrp_other_ctc/frozen_cache.json` and
`artifacts/reports/stage2_v17_asllrp_other_ctc/frozen_audit.json`. No test split was
accessed. The combined Stage-2 run is now unblocked.

The first full combined `OTHER`-CTC adaptation then ran on MPS and is rejected for
promotion. It completed ten epochs plus the epoch-zero baseline in 58.95 seconds under
the 12% cap; the selected result is deliberately epoch zero because no trained epoch
preserved both legacy guards (local at most 7 edits and sparse held-out ASLLRP at most
11 edits). The trained trajectory proves the new spans are learnable: natural-ASLLRP
full WER fell from 107.1848% at epoch zero to 53.3724% at epoch 10 and target-only WER
fell from 223.2394% to 79.5775%. However, sparse held-out-ASLLRP target edits worsened
from 11 to 12--14 through epoch 5 and 13--14 afterward; local edits ranged 7--11. The
hash-pinned epoch-zero safeguard is
`artifacts/models/stage2_v17_asllrp_other_ctc_v1/best_model.pth` (SHA-256
`8113f5b96d8b2e2f17bcc3e491dda66511b678bb50cf54070b7763efe6924a0b`), and the full
history is `artifacts/models/stage2_v17_asllrp_other_ctc_v1/result.json`. This is a
negative single-head result, not evidence against the acquired data; it shows that the
current objective trades legacy phrase competence for natural `OTHER` competence.
No test split or consumed external evaluation was accessed.

## 2026-09-01 06:53 PST — unsafe phone selector withdrawn; full ASLLRP OTHER-CTC rebuild underway

The 02:30 two-head general selector promotion was invalidated by the subsequently
available physical-phone diagnostic recordings. On the five saved `I NEED HELP`
attempts it introduced empty/`WHERE`-biased behavior despite its validation gains.
It has been removed from the mobile target. The app is restored to
`stage2_v17_full_activity_routed_hybrid_v2`: multi-sign input uses the pinned bare
`Stage2PhraseV17FP32` head, while a single-sign output uses the repaired full-motion
Stage-1 plus isolated-correction route. The restored manifest pins checkpoint
`b15b06ff...` and Core ML package tree `0b83fc...`. Flutter tests, the native
activity-alignment test, signed Release build, installation, and launch on the
connected iPhone 13 pass. This supersedes the mobile-promotion claim below; it is not
a new physical-phone accuracy claim.

Two further validation-only shortcuts were implemented and rejected. Reclassifying
CTC-emission segments through Stage 1 collapses to 24/24 ASLLRP and 264/259 local
edits because CTC emission locations are not sign boundaries. Frame-level Stage-1
identity-logit fusion selects zero identity weight on training and leaves ASLLRP at
11/24 while worsening local validation to 8/259. Neither is a deployment candidate.

The highest-leverage Stage-2 data defect is now fixed at the preparation boundary.
The latest ASLLRP sentence metadata contains 17,519 valid annotations from 2,130
utterances, but the old preparation retained only 44 training clips because every
out-of-vocabulary annotation broke the continuous span. The new exact-variant,
signer-disjoint plan keeps full target-bearing natural spans and maps intervening
annotations to an explicit CTC `OTHER` token. It contains 1,104 bounded spans (879
train from BEN/CORY/RACHEL and 225 validation from JONATHAN), 1,483 locked-vocabulary
target tokens, 2,103 collapsed OTHER tokens, and 1,090 unique parent utterances. Crops
are split only between manual annotations and never exceed 256 frames/eight mobile
windows. The plan is at
`artifacts/reports/stage2_v17_asllrp_other_ctc_plan/plan.json`; targeted acquisition
is active under `data/local/asllrp_other_ctc_v17/` rather than downloading an entire
unrelated corpus.

The 101-class Stage-2 contract and warm-start trainer are implemented. A critical
index-boundary test now guarantees acquisition CTC IDs 1..101 are converted exactly
once into archive class IDs 0..100 before the existing loader reapplies the blank
offset. The selected 100-gloss head is extended with a neutral OTHER row while all
old logits remain bit-identical at epoch zero; replay distillation and source-balanced
sampling protect the existing local and sparse-ASLLRP performance during adaptation.
The natural ASLLRP target subset has no exact locked `I`, `NEED`, `HELP`, or `HELLO`
occurrences but has 51 training `WHERE` occurrences, so natural ASLLRP is capped at
40% rather than allowed to dominate adaptation. The other 60% replays local real
phrases, sparse target-only ASLLRP, and the selected train-only multivoice synthetic
pool. That pool covers all 100 classes and includes 194 `NEED` compositions. The
explicit phone-development gate replays the five saved `I NEED HELP` tensor captures;
the current bare head gets 0/5 exact and is the pinned pre-training baseline.
Nineteen focused model/data tests and Python compilation pass. No Citizen, SemLex,
local, RIT, or other test split was accessed.

Integrity clarification: a subsequent over-broad text search for candidate gloss
spellings across `data/local/dataset_metadata/asllrp_signbank/*.csv` printed rows from
the already local RIT external metadata file as well as the intended ASLLRP file. No
RIT video, feature tensor, model prediction, metric, or checkpoint selection was run,
and no printed RIT row is used by the OTHER-CTC preparation or training. The consumed
RIT external evaluation was not rerun. Future variant inspection is pinned explicitly
to `asllrp_sentence_signs_2025_06_28.csv`; the preceding sentence should therefore be
read as “no test evaluation or test media access,” not as “no metadata filename was
ever printed.”

The targeted ASLLRP acquisition and Apple Vision/RGB extraction are now complete.
Acquisition verified 1,090/1,090 parent utterances and 1,104/1,104 bounded spans with
zero failures. The finalized manifest hash is
`35bef0af546fb0ed7a600614a0828f33209219191a2be789813bfd018584b881` and converts
acquisition CTC IDs to archive class IDs exactly once. The fail-closed extraction audit
reports 1,104 expected/actual/audited archives, zero missing/unexpected/failing
archives, 4,505 total windows, zero invalid landmark windows, and 96.9138% mean valid
hand-view coverage. The audit is
`artifacts/reports/stage2_v17_asllrp_other_ctc/extraction_audit.json`.

Hand embedding remains in progress under the prior crash-safe resource contract. Test
batches 64 and 48 were rejected by the 8% MPS watermark at approximately 1.3 GiB
allocated; the worker exited before system pressure or corruption, and the watermark
was not disabled or raised. Batch 32 is stable with short subprocess lifetimes. The
fixed 256px MobileCLIP2 transform was also simplified from per-image
NumPy→PIL→no-op resize→tensor into vectorized NumPy→tensor. A direct parity check
against the pinned transform measured max absolute input difference 0.0, and a focused
unit test passes. Existing completed embeddings are hash/schema-audited and reused.

## 2026-09-01 02:30 PST — validated two-head Stage-2 selector promoted to the iPhone app

The ASLLRP sentence metadata was re-audited against the downloaded parent utterances
and the 1,719 exact-variant segmented signs. It preserves manual start/end frames for
every target sign, and all 56 active train/validation phrase crops reconstruct the
declared target sequence exactly. The clips are therefore genuine, correctly ordered
continuous signing—not corrupt or arbitrary labels. Their limitation is statistical:
after the Jonathan holdout, only 44 sparse phrase crops remain, most are 23--40 source
frames long and contain a nearly unique two-sign transition. This is insufficient as a
main sequence corpus even though the same signer/style variation is useful supplemental
evidence and Stage 1 can learn broader isolated-sign invariance from far more examples.

A manual-boundary CTC fine-tune was implemented and screened to distinguish an
alignment problem from a coverage problem. It adds exact ASLLRP frame-interval
cross-entropy, real-phrase CTC, and warm-head distillation under a 12% MPS memory cap.
All 56 alignments pass their CTC-collapse integrity gate, and the 16-epoch run completed
in 22.97 seconds without memory pressure. It nevertheless worsened ASLLRP validation
from 11 edits to 14--16 and local validation from 7 edits to 8--15. The output
`stage2_v17_aligned_ctc_v1` therefore retains epoch zero and is explicitly rejected.
This negative result isolates the problem to phrase/signer coverage rather than a
missing alignment loss.

The existing phrase-agnostic general CTC selector was then promoted instead. It
combines the exact context-primary and transition-specialist heads with the frozen
0.10 blend, +0.30 blank bias, equal-length multi-token eligibility rule, and exact
specialist CTC likelihood comparison. It does not inspect gloss identity, phrase
identity, signer identity, or validation labels at inference. On the complete non-test
development gates it improves every compared domain relative to the prior primary:
ASLLRP contiguous phrases improve from 11/24 to **9/24 edits** (37.50% WER; 4/12
exact), local phrases improve from 7/259 to **6/259 edits** (2.3166% WER; 92/97
exact), and ASLLRP held-out-signer contextual signs improve from 44/254 to **43/254
edits** (16.9291% WER; 213/254 exact). These are development-validation results, not
an independent phone-signer accuracy claim.

Both exact heads were exported as FP32 Core ML packages. The primary package tree hash
is `92a4a2e49c9cfcfeb51189c4468f269030f10f6c714e911da02ae8a5620f88e7` and the
specialist hash is `db14bc692cdc76e466c902ed64aa0b7dbd5c5c505af8849dff236c7ed55d8082`.
Each individual export has zero decode mismatches on all 109 phrase-validation rows.
The combined Core ML selector was then checked on 363 phrase/context rows and has zero
PyTorch decode mismatches and zero selector-decision mismatches while reproducing the
9/24, 6/259, and 43/254 results exactly. Host timing is not physical-iPhone evidence.

The mobile candidate is now
`stage2_v17_general_selector_activity_routed_hybrid_v3`. Multi-sign recordings run the
validated two-head selector; one-sign CTC outputs remain routed through the repaired
full-activity Stage-1/isolated-correction hybrid. Benchmark JSON pins both package and
checkpoint hashes and reports which route won. The simulator build passes, both native
iPhone-13-simulator tests pass, Flutter tests pass, and a signed Release build was
installed and launched on the connected physical iPhone 13. Physical camera accuracy
still requires fresh owner-operated recordings; installation alone is not presented as
that evidence. Twenty focused Python tests, Python compilation, JSON/plist validation,
and repository `git diff --check` pass. No Citizen, SemLex, local, ASLLRP, RIT, or
other test split was accessed.

## 2026-09-01 02:06 PST — unsafe Stage-2 shortcuts rejected; single-sign mobile route repaired

The completed `stage2_v17_accuracy_repair_v1` experiment was recovered and audited.
Despite 30 epochs and 554.54 seconds of training, it reduced equal-weight isolated
validation accuracy from the selected Stage-1 model's 94.1517% to 89.998%, produced
15/24 ASLLRP validation edits, and also worsened the local phrase result. Its checkpoint
is rejected and must not replace either the Stage-1 classifier or the deployed phrase
head. This is further evidence that asking the sparse phrase CTC objective to relearn
the 100 isolated class identities is destructive rather than an accuracy repair.

A latent schema-integrity defect was fixed before testing denser temporal windows.
Stage-2 archives extracted with a non-default window stride previously received the
same schema fingerprint as non-overlapping stride-32 archives. `window_stride` is now
part of the Stage-2 feature configuration and temporal contract; the established
stride-32 fingerprint remains unchanged for compatibility, while stride-8 archives
receive the distinct `44e9f97c67a003c4` fingerprint. The extractor and MobileCLIP2
encoder now propagate and verify that value, and all eight focused extraction tests
pass.

All 12 existing ASLLRP held-out validation clips were re-extracted at a 32-frame window
with stride 8, hand-encoded, and frozen through the selected encoder without touching a
test split. The denser inputs made every evaluated phrase head worse: the deployed bare
head increased from 11/24 to 15/24 edits, while the general selector increased from
9/24 to 14/24. CTC prefix beam search was also evaluated on the original non-overlap
features across several blank biases and worsened both ASLLRP and local validation; the
existing greedy collapse remains better. Overlap and beam search are therefore rejected
for the current checkpoint rather than promoted as speculative fixes.

The iOS single-sign route was repaired independently of the phrase head. It now detects
the complete activity span, retains bounded motion context, resamples that entire span
for both Stage-1 landmarks and MobileCLIP2 hand features, and reruns the frozen encoder
on the aligned span before applying the validation-proven isolated hybrid. The previous
arbitrary most-active 32-frame window route was removed. The candidate is identified as
`stage2_v17_full_activity_routed_hybrid_v2`; Flutter tests pass, the new native iPhone-13
simulator activity-span test passes, and a signed Release build was installed and
launched on the connected physical iPhone 13. Physical accuracy is not yet claimed:
new user recordings are required to validate that route on-device. No Citizen, SemLex,
local, ASLLRP, or other test split was accessed.

## 2026-09-01 01:19 PST — Stage-2 phrase evidence is coverage-limited, not proof of corrupt ASLLRP video

The real continuous-training manifest was audited at the exact-sequence, token, and
signer levels after the physical-phone Stage-2 failures. ASLLRP contributes only 44
training clips, 38 distinct two-gloss sequences, 92 target tokens, and three training
signers. Thirty-four of the 38 training sequences occur exactly once; only `HOME WHEN`
and `WHEN FRIEND` occur three times and `HAVE TIME` and `SCHOOL TOMORROW` occur twice.
The 12-clip ASLLRP validation set contains 24 tokens from the held-out signer JONATHAN,
but five of its eight exact sequences were never seen in training. `BAD` is not present
as a training target token at all. The bare phrase CTC result of 11/24 edits (45.8333%
WER) and 2/12 exact sequences is therefore evidence that the current Stage-2 corpus is
too sparse for reliable held-out-signer sequence learning; it is not, by itself,
evidence that the underlying videos or annotations are corrupt.

The local phrase corpus has the opposite limitation. It contributes 390 training and
97 validation clips but only six exact phrases. Every validation phrase and every
validation token occurs in training, with heavy repetition: `HELLO HOW YOU` and
`PLEASE HELP I` each have 107 training examples, while the other four phrases have
32--48. The validation split uses the same local signer pool, as previously permitted
for data expansion. Its 7/259 edits (2.7027% WER) and 91/97 exact sequences (93.8144%)
mainly measure interpolation over repeated phrase templates and recording conditions;
they are not evidence of unseen-signer or novel-phrase generalization. The ASLLRP and
local scores therefore must not be compared as though they were equally difficult
accuracy tests.

Stage 2 is a CTC gloss-sequence recognizer rather than a phrase-ID classifier, so one
example of every exact sentence is not inherently required. It does, however, require
repeated gloss evidence across varied temporal contexts, transitions, durations, and
signers. The current ASLLRP subset is also deficient on those axes, while the local set
has repetitions but almost no phrase/context or signer diversity. ASLLRP remains useful
as a small hard supplemental/held-out diagnostic and should not be discarded, but it
cannot serve as the main training corpus or the sole model-selection gate.

The high isolated-sign validation results across Citizen, SemLex, and local data show
that Stage 1 learned substantial signer/style/camera invariance for the 100 pinned
classes; they do not prove that arbitrary lexical variants are interchangeable.
Project mappings still pin exact raw gloss/ASL-LEX variants and do not merge numeric
variants. A true variant mismatch can remain mislabeled even when a model handles
ordinary signer accent/style variation. The next high-leverage action is genuine
continuous data collection for a bounded functional phrase inventory, with repeated
transitions from multiple signers and a signer-held-out validation partition. No test
split was accessed during this audit.

## 2026-08-31 20:19 PST — physical-phone errors localized to Stage 2 and its WHERE adapter

The 22 successful post-fix captures in the connected iPhone 13's Files-visible
`Documents/Diagnostics` directory were copied read-only to a temporary host directory
and audited. All seven saved arrays in every capture have the expected finite values,
window/source masks agree with the reports, Apple Vision observed usable hands, and
the orientation path uses AVFoundation's preferred track transform. Replaying the
exact saved landmark, hand-embedding, validity, box, and window-mask tensors through
the pinned PyTorch graphs reproduces all 22 phone/Core ML gloss sequences exactly.
This rules out an iOS label-order error, corrupt NumPy export, stochastic Core ML
behavior, and a global class-index shift.

The deployed checkpoint is a context-adapted Stage-2 head, not the bare temporal CTC
head. Its development-selected residual has weight `1.5` and is allowed to change
only zero-based classes 9 and 86, `WHERE` and `HOME`. On capture `115101`, the bare
head decodes the exact phone tensors as `NEED NEED`; adding the deployed residual
changes the same tensors to `WHERE WHERE`. The residual changes five of the 22 phone
decodes and explains a material part of the observed WHERE collapse. It was selected
on ASLLRP development validation where NEED had no coverage, so its behavior on this
new phone signer is validation overfit rather than independent generalization.

A bounded validation-only A/B was then run on every existing Citizen, SemLex, and
local isolated validation clip for `HELLO`, `MORNING`, `NEED`, and `WHERE`—165 clips
total, with no test access. The selected isolated Stage-1 model scores 151/165; the
bare Stage-2 CTC head scores 120/165; the deployed context-adapted Stage 2 scores
117/165. Per class, the three results are respectively: `HELLO` 29/32, 19/32, 19/32;
`MORNING` 44/45, 39/45, 39/45; `NEED` 37/40, 17/40, 11/40; and `WHERE` 41/48, 45/48,
48/48. In local validation alone, the adapter changes NEED from 8/27 exact to 4/27
and emits WHERE on 19/27 NEED clips. The underlying isolated data/model is therefore
not globally trashed; the main degradation is introduced by the Stage-2 training and
deployment contract.

The Stage-2 coverage audit explains that degradation. Real Stage-2 phrase training
contains HELLO only inside 107 `HELLO HOW YOU` rows and has no real NEED target at all.
Its real validation contains HELLO only inside 27 `HELLO HOW YOU` rows and again no
NEED. By contrast, contextual ASLLRP contributes 50 real WHERE training clips and 16
WHERE validation clips, after which the explicit WHERE residual further boosts that
class. The isolated Stage-1 replay itself is reasonably populated: Citizen/SemLex/local
train counts for HELLO are 14/12/105, MORNING 14/17/149, NEED 15/14/177, and WHERE
14/18/153. This is a Stage-2 supervision/selection imbalance, not evidence that those
four isolated class folders were mislabeled.

There is also a phone temporal-boundary mismatch. `prepareStage2` anchors nonoverlapping
32-frame windows at recording frame zero and does not trim or re-anchor before
windowing. Fourteen of the 22 phone attempts contain 19--30 frames before Apple Vision
first observes a hand. Consequently, a short isolated sign is often split between the
end of a mostly idle first window and a resampled partial second window; this explains
the sensitivity to when Record was tapped and the duplicated `HELLO HELLO`/`NEED NEED`
outputs. Stored portrait/landscape metadata is not the primary cause.

No production change was made during this diagnosis. The recommended next design is
to remove the development-only HOME/WHERE residual from deployment, motion-anchor the
completed clip before forming windows, and run the already strong whole-clip Stage-1
classifier for a single detected sign while retaining Stage 2 for multi-sign clips.
Stage 2 should then be retrained/evaluated with class-balanced one-token replay, random
leading/trailing idle and window-offset augmentation, and an all-100-class isolated
validation gate in addition to genuine phrase gates. The 22 phone captures should be
given intended gloss labels and retained as new-signer diagnostic evidence, with a
subset locked before any phone-data adaptation. Incremental extraction during
recording remains explicitly deferred. No Citizen, SemLex, local, ASLLRP, or other
test split was accessed.

## 2026-08-31 19:54 PST — physical iPhone diagnostics fixed and Release app reinstalled

Two user-operated iPhone 13 runs established the first real-camera evidence for the
Flutter app. A short intended `HELLO` clip was recorded upright at 720x1280 but decoded
as `MORNING`; its report showed 2,330.3 ms Apple Vision extraction, 1,341.3 ms RGB crop
encoding, 26.5 ms median Core ML inference, and nominal thermal state. A later phrase
run emitted `HELLO HELLO HOW YOU`; recognition retained the intended phrase with one
duplicate token, but Stage 3 fell back to `Hello hello how you.` because exact reviewed
templates never delete recognized tokens. That enabled benchmark run showed 6,119.8 ms
extraction, 28.3 ms median, 32.1 ms p90, resident memory 73.6->441.6 MiB, and nominal
thermal state. These timings and memory values describe the pre-fix physical build;
they are not post-optimization measurements.

The live app at
`/Users/frnzlo/Documents/machine_learning/mobile_app/slt_mobile_app` now fixes the
camera-preview distortion at its root: Flutter no longer wraps `CameraPreview` in the
raw sensor aspect ratio a second time, and an orientation-aware cover layout preserves
geometry without stretching. Portrait controls were moved ahead of the scrollable
result so they are no longer clipped, and the benchmark line now renders a real newline.

Every attempt is copied out of iOS temporary storage before extraction to a unique
Files-visible `Documents/Diagnostics/<UTC timestamp>_<UUID>/` directory. Successful
captures retain `recording.mp4`, `report.json`, `tensor_manifest.json`, raw upright
landmarks, exact normalized model landmarks, source/window masks, hand-valid masks,
normalized hand boxes, and exact MobileCLIP2 hand embeddings as float32 NumPy files.
Failed attempts retain the video plus `error.json`. `UIFileSharingEnabled` and
`LSSupportsOpeningDocumentsInPlace` are enabled, so the folders appear under
Files -> On My iPhone -> ASL Translator -> Diagnostics. Reports are saved for every
capture rather than only benchmarks, and the old literal Swift `${...}` filename bug
can no longer overwrite prior evidence. RGB crop images are deliberately not duplicated;
the original video and saved boxes reproduce them.

The completed-file pipeline still forms sequential 32-source-frame windows, with each
window normalized/resampled to the model's fixed 32x61x5 input. Safe latency/memory
changes were applied without changing those model inputs: model/label/naturalizer
resources are cached once, camera files use AVFoundation's preferred transform rather
than a four-rotation Vision sweep, the hand observations from landmark extraction are
reused for RGB crop creation, normal one-shot inference reuses the cold output instead
of running the neural path twice, and per-frame/per-crop autorelease pools bound Apple
framework temporaries. Core ML remains configured with `computeUnits=.all`, leaving
CPU/GPU/Neural Engine scheduling to iOS. Independent parallel Vision windows were not
introduced because the pre-fix run already reached 441.6 MiB and concurrent window
buffers would increase peak memory. Live extraction during recording remains a separate
camera-stream architecture and is deferred until the post-fix completed-file benchmark
is measured.

Stage 3 gained one explicit reviewed rendering for the observed recognizer sequence
`HELLO HELLO HOW YOU` -> `Hello, how are you?`; the gloss output remains unchanged and
visible. The ordinary `HELLO HOW YOU` template already existed. The canonical and app
manifest copies match at SHA-256
`1d855ad74b2c26d68a28dd6fc55630bb00e2127ec6ab57fadc74e685b46b7716`.
This remains bounded reviewed-template naturalization, not an on-device LLM or general
ASL-to-English translator.

Validation passed: Flutter analysis has zero issues, both Flutter tests pass, all eight
focused Stage-3 naturalizer tests pass, and the unsigned arm64 Release iOS build succeeds
at 126.3 MB. The signed Release app version 1.0.0 was then installed and launched on the
connected iPhone 13 `angelo` under bundle ID `com.kokoab.sltMobileApp`. Post-fix
latency, memory stability, prediction behavior, and Files bundle contents still require
the next user-operated recording. PopSign was explicitly removed from the active plan.
No Citizen, SemLex, local, ASLLRP, PopSign, or 2M-Flores test split was accessed.

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

## 2026-08-24 21:21 PST — v17 source and evidence published to GitHub

The full eligible reorganization, v17 source, iOS projects, tests, documentation,
manifests, and compact reports were committed as
`090d3149db4f0387bdcf55b6d5e1924b8413c0f2` (`Add v17 mobile SLT pipeline and
evidence`) and pushed successfully to
`https://github.com/kokoab/sign_language_translation_system.git`, branch `master`.
The push was a fast-forward from `2a6b743` and local `master` was configured to track
`origin/master`. The separate repository-default `main` branch was not merged,
rewritten, or otherwise modified.

The commit author is the user's configured
`kokoab <francis.batiancela@intechsive.com>` and contains no co-author trailer. The
GitHub publication excludes the large/local paths documented in `.gitignore`; those
assets remain present on this Mac for iPhone builds and experiments.

## 2026-08-24 21:19 PST — physical-iPhone deployment guide and GitHub hygiene

The physical installation procedure is now documented at
`mobile_benchmark/OrientationBenchmarkV17/DEPLOY_TO_IPHONE.md`. It covers exact local
model prerequisites, Apple ID/Personal Team setup, Developer Mode, automatic signing,
unique bundle identifiers, physical-device selection, first installation, file-video
inference, JSON export, physical benchmark discipline, and common signing/device/model
failures. The project remains iOS 17.0+, file-picker based, and not a live-camera app.

Before GitHub publication, the worktree was audited rather than staged blindly. Three
local files exceed GitHub's 100 MB per-file limit, and the repository contains roughly
20 GB of datasets, model assets, Core ML packages, checkpoints, generated build trees,
tool environments, archives, and mobile build products. `.gitignore` now excludes
those reproducible/local products plus generated media and mobile build output while
retaining source, documentation, manifests, and compact evidence reports. Two
reproducible Stage-2 plan JSON files above 10 MB are also excluded. The eligible
untracked publication set is approximately 69.2 MiB across 1,112 files, with an 8.6
MiB largest file. The configured Git author remains the user's
`kokoab <francis.batiancela@intechsive.com>`; no co-author trailer will be added.

## 2026-08-24 21:00 PST — bounded Stage 3 and file-video iPhone app are deployable

The locked-100 mobile pipeline is ready for signing and interactive file-video testing
on an iPhone. The selected compact Stage-2 model and strict downstream contract remain
unchanged. The iOS app now reads arbitrary native aspect ratios/orientations, samples
at most 256 frames, retains only one 32-frame window at a time, runs Apple Vision,
creates real-pixel left/right/union hand crops, executes all three neural components in
Core ML, collapses the CTC sequence, validates the Stage-2 hashes/label mapping, and
renders bounded English. This replaces the previous interactive-path block and keeps
the video preprocessor explicitly memory bounded.

No neural Stage-3 translator was promoted. The fail-closed audit found 1,165 genuine
gloss/English pairs (999 2M-Flores `dev`, 166 NCSLGR), but zero complete sequences are
fully expressible with the locked 100-gloss vocabulary. Training by deleting OOV signs
would corrupt the target meaning. The promoted Stage 3 therefore contains 35 exact,
meaning-conservative templates and a deterministic literal fallback that never
deletes, replaces, reorders, or invents glosses. Both literal and naturalized text,
rendering mode, and fallback status are exposed. Template coverage is 85/97 local
validation rows and 0/12 ASLLRP contiguous rows; all others fall back literally. This
is a bounded naturalizer, not a general/open-domain translator.

The final iPhone 13 simulator run passed all eight expanded-canvas rotations at
0/17/37/73/90/123/180/270 degrees, each with exactly 200 timed inferences. All eight
predicted HELLO and Stage 3 rendered `Hello.`. Its final result is
`artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json`, SHA-256
`2f2c513b2f50b8f3c7a587782fe825ff727f6017738c92ef3c0f4764202f6806`.
Reports explicitly set `videoFileToGlossEndToEnd=false`,
`cameraToGlossEndToEnd=false`, `hardwarePerformanceClaim=false`, and
`thermalsInterpretable=false` because Apple Vision preprocessing occurred on the Mac
host and Core ML executed on simulator hardware.

The separate Swift-source video-to-English gate passed 8/8 HELLO rotations through
Apple Vision, all three Core ML models, CTC, and Stage 3. Maximum process RSS was
279,773,184 bytes, which is Mac-host evidence only. The unsigned generic iPhoneOS
Release build passes and produces a 110,350,170-byte arm64 app containing exactly the
three selected Core ML models. The bundled Stage-3 manifest matches source SHA-256
`68c7ce67632f66ee70fa3b3d36eb8df33ad72dc674edbf3b720e93c1240f84a6`.

One hundred fifty-seven focused tests pass. The machine-readable deployment audit
passes with zero acceptance errors at
`artifacts/reports/stage3_mobile_v17/deployment_audit.json`; its SHA-256 is
`065f42985a5285d143338520709a8211007358b353bd346954a6caf060d5fbf8`.
The complete evidence summary is
`artifacts/reports/stage3_mobile_v17/README.md`. No Citizen, SemLex, local, or
2M-Flores `devtest` split was accessed.

Claim boundary: the app is deployable for videos selected from the iPhone file picker.
A live-camera capture UI is not implemented. Physical-iPhone accuracy, latency,
memory, thermals, and ANE behavior remain unmeasured and must not be inferred from the
simulator or Mac-host results.

## 2026-08-24 18:31 PST — locked-100 Stage 2 handoff is ready for Stage 3

The requested Option A boundary is complete: the selected compact Stage-2 recognizer
emits only the frozen 100 glosses, and a future Stage 3 consumer now has a strict,
hash-pinned interface. The authoritative contract is
`active/v17/stage2_to_stage3_contract_v17.json`, SHA-256
`8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d`.
It fixes blank `0`, tokens `1...100`, label order, greedy CTC collapse, maximum eight
windows, exact required output keys, checkpoint/vocabulary hashes, empty-sequence
behavior, and fail-closed Stage-3 consumer rules. All 363 full-pipeline validation
outputs passed this contract.

The final iPhone 13 simulator suite passed at 0, 17, 37, 73, 90, 123, 180, and 270
degrees. Each condition completed exactly 200 timed inferences; all 8 conditions
predicted HELLO, and all 1,600 individual decoded iteration votes were unanimously
CTC token 15. Exact quadrant corrections were 0->0, 90->270, 180->180, and 270->90;
intermediate residual roll remained within 45 degrees. The final result is
`artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json`, SHA-256
`84b90290dff1e6e1a39aae3eab9bd074fb2bf2cd18e07e4f7267b58b13ce5cea`.

Two simulator integration faults were found and fixed before promotion. First, the
old harness corrected landmarks but cropped RGB from uncorrected rotated pixels; the
new harness creates both together after one shared v17 orientation correction.
Second, Swift `MLMultiArray` padding was allocator garbage because the padded windows
were not explicitly initialized; landmarks, validity, boxes, embeddings, and window
masks are now zero-filled exactly like the Python training/evaluation tensors. The
pre-fix run-to-run blank predictions disappeared, and both a 20-iteration regression
and the final 200-iteration suite are unanimous across all angles.

Both final unsigned Release builds pass: generic iPhoneOS and arm64/x86_64 Simulator.
Each app bundle contains exactly the three selected models—MobileCLIP2 hand image
encoder, frozen multimodal encoder, and compact context/CTC head—and no obsolete
Stage-1 package. The full raw-crop Core ML report remains exact over 363 samples, 574
windows, and 22,046 crops with zero cached-Core-ML or PyTorch decode mismatches; its
SHA-256 is `99f5ba02d3fbe624044dcf12048ccb1224f2a59dceedefb3d935c3c6463c8c5b`.

The 1,100-row independent capture pack setup audit passes with its ledger unchanged:
1,000 target and 100 OOV plans, zero errors/warnings, no model inference, and no test
access. Its provenance was repaired to pin the earlier frame-API refactor of
`model_hand_mobileclip2_v17.py`; the existing equivalence test proves that refactor
does not change pooled Stage-1 output. Ninety-six unique focused tests pass, affected
Python entry points compile, all generated JSON parses, and `git diff --check` passes.
The complete evidence summary is
`artifacts/reports/mobile_100gloss_v17/README.md`.

This readiness statement is specifically the Stage-2-to-Stage-3 gloss interface. The
existing Stage-3 translator remains a weak research model and is not promoted. The
simulator runtime also lacks Apple Vision pose weights, so simulator preprocessing was
performed by macOS Apple Vision; `cameraToGlossEndToEnd=false`, and there is no claim
about physical-iPhone latency, memory, thermals, ANE behavior, or independent capture
accuracy. No Citizen, SemLex, local, or 2M-Flores test split was accessed.

## 2026-08-24 15:08 PST — full locked-100 mobile neural path validated and integrated

Fresh decoded RGB crops now pass through all three retained Core ML packages rather
than through cached MobileCLIP2 embeddings. The complete validation gate covered 363
samples, 574 Stage-2 windows, and 22,046 valid JPEG hand crops. Fresh RGB inference
produced zero decode changes versus both the cached Core ML path and cached PyTorch
path. The retained results remain exactly 11/24 ASLLRP contiguous edits, 7/259 local
phrase edits, and 43/254 ASLLRP contextual edits. Maximum embedding deviation from the
historical float16 cache is `0.000244319`, with minimum cosine `0.999999821`.
Evidence is in
`artifacts/reports/mobile_100gloss_v17/full_coreml_validation.json`. Its timing is
Mac-host Core ML evidence only.

The iOS benchmark app now bundles the MobileCLIP2-S0 image tower, frozen multimodal
encoder, compact context/CTC head, the exact 100-label vocabulary, and all provenance
hashes. It decodes real JPEG crops in-app, runs all three neural models in Core ML,
greedily collapses blank-0 CTC output, and emits ordered gloss sequences. A 20-iteration
per-angle iPhone 13 simulator smoke passed 8/8 for HELLO at 0, 17, 37, 73, 90, 123,
180, and 270 degrees. The first attempt exposed and rejected a harness bug where RGB
crops were made before the Vision orientation correction; the corrected harness now
creates landmarks and crops together after the same v17 rotation, matching training.

The stable downstream interface is generated at
`active/v17/stage2_to_stage3_contract_v17.json`. It pins checkpoint, vocabulary, all
three Core ML tree hashes, blank `0`, tokens `1...100`, eight windows, greedy collapse,
and a strict `slt_stage2_gloss_sequence_v17` output object. Stage 3 must reject version
or hash mismatches and must not invent an unknown token or merge gloss synonyms.

Simulator Vision remains a documented limitation: the installed runtime lacks pose
Espresso weights, so macOS Apple Vision creates the corrected landmarks/crops while
the simulator runs every neural model and CTC decode. This is complete mobile-neural
functional evidence (`endToEndPipeline=true`) but not camera-to-gloss evidence
(`cameraToGlossEndToEnd=false`) and not a physical-iPhone performance claim. The final
required 200-iteration evidence run and full focused validation remain pending. No
Citizen, SemLex, local, or 2M-Flores test split was accessed.

## 2026-08-24 14:49 PST — exact MobileCLIP2 hand-crop tower exported for mobile Stage 2

The exact frozen MobileCLIP2-S0 visual tower used by the retained 100-gloss teacher is
now exported as an FP32 Core ML image model at
`artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage`, tree SHA-256
`9309548dd69a5c8e899ea00ee4f0bbe88505ed803d12520a60e5d954ff370974`, size
45,580,737 bytes. It consumes a 256x256 RGB image with pixel scale 1/255 and emits the
same normalized 512-D hand embedding used in training. The source MobileCLIP2-S0
checkpoint remains pinned at SHA-256
`ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95`.

Full class-spanning parity covered one decoded crop from every one of the 378 Citizen
validation clips. Core ML versus the unreparameterized PyTorch tower has maximum
absolute error `1.49012e-06` and minimum cosine `0.999999881`; versus the historical
float16 embedding cache it has maximum absolute error `0.000226140` and minimum cosine
`0.999999821`. The 14.91 ms median and 18.74 ms p90 are Mac-host Core ML timings only,
not iPhone, ANE, thermal, or complete-pipeline evidence. The detailed report is
`artifacts/reports/mobile_100gloss_v17/mobileclip2_image_fp32.json`. No Citizen,
SemLex, or local test split was accessed.

This closes the missing crop-to-embedding model export only. Validation of freshly
regenerated RGB embeddings through both Stage-2 Core ML packages, app integration,
orientation-safe simulator execution, and the Stage-2-to-Stage-3 contract remain open.

## 2026-08-24 13:58 PST — Stage 3 genuine-reference gate established; legacy translator rejected

The old Stage-3 evidence is invalid for promotion. `models/stage3_history_simulated.json`
is explicitly simulated, while the actual 60.5M-parameter T5-small checkpoint scored
only 0.2832 BLEU and 6.9142 chrF++ across 321 genuine reference-gloss/English pairs.
On the fixed 155-row 2M-Flores `dev` validation subset it scored 0.0390 BLEU and
4.0782 chrF++; it frequently collapsed to short synthetic-template responses.

Only text metadata was acquired from the official 2M-Flores dataset-server endpoint:
999 current `dev` rows, no video bytes, SHA-256
`6acfaeba2ef680c2b952b412fb28751045e578330e00e9e0f4afd0268ca20626`.
Every one of the 155 previously selected rows matches exactly. The other 844 rows plus
166 public NCSLGR pairs formed 1,010 genuine train rows. A deterministic 1,010-row
sample of the legacy synthetic CSV supplied equal-mass replay; all fixed validation
`(id, signer)` keys were excluded from training.

A bounded warm-start run selected epoch 2 at 0.8544 BLEU and 12.5181 chrF++ on the
unchanged 155-row genuine gate, improving from the 0.0390/4.0782 baseline. Epoch 3
regressed slightly and was rejected. The cold-reloaded package reproduces the selected
metrics exactly. Its checkpoint is
`artifacts/models/stage3_v17_reference_replay_v1/model.safetensors`, SHA-256
`25a0deb4599da88de613d70fad1ad94ca138d0c0ef6ba50efba96650e593cb82`.

Two earlier MPS attempts stopped safely before producing a checkpoint when the 40%
allocator cap detected growing variable-shape graph caches. The final run fixed both
input and target shapes, used batch size 4 and Adafactor, kept `num_workers=0`, and ran
autoregressive validation on CPU. It completed in 699.47 seconds with zero nonfinite
batches and no system memory-pressure failure.

This is a real improvement but not deployable translation quality: exact match is
0/155, and the current Stage-2 recognizer emits only 100 glosses while the genuine
sentences contain substantial out-of-vocabulary content. Stage 3 therefore remains a
research module, and no end-to-end conversational translation claim is supported.
Full evidence is in
`artifacts/reports/stage3_v17_reference_replay_v1/EXPERIMENT.md`. No Citizen, SemLex,
local, 2M-Flores `devtest`, or other test split was accessed.

Sixty-three focused Stage-2, Stage-3, signing-voice, transition, and data-contract tests
pass. All changed Python entry points compile, generated JSON parses, and
`git diff --check` passes. After training and cold reload, system-wide free memory was
63%.

## 2026-08-24 13:42 PST — exact compact Stage 2 exported and validated in Core ML

The retained compact Stage-2 graph is exported as two FP32 packages. The frozen v17
multimodal window encoder is
`artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage`, tree SHA-256
`1146b539800e6f09a743f4a8ee882c9b2cd2b01503ff3f362a11e97e8c827bb9`; the exact
context adapter plus CTC head is
`artifacts/coreml/Stage2CompactContextV17FP32.mlpackage`, tree SHA-256
`e92ba7d8b7c61c52bc776840e953c73abb6b012637991d01582d4fd64067760a`.

Combined cold validation covered 363 samples and 574 windows with zero Core ML versus
PyTorch decode mismatches. It exactly retained 11/24 ASLLRP contiguous, 7/259 local,
and 43/254 ASLLRP contextual edits. The 12.42 ms median and 12.94 ms p90 are Mac-host
Core ML timings only, not iPhone/ANE/thermal evidence. The packages consume
precomputed MobileCLIP2 hand embeddings; the crop-to-embedding MobileCLIP2 network is
not yet implemented in the iOS app. Therefore the Stage-2 Core ML graph is ready, but
camera-to-gloss mobile deployment is not. Full evidence is in
`artifacts/reports/stage2_v17_coreml_export/README.md`. No test split was accessed.

## 2026-08-24 13:18 PST — 2M-Flores temporal replay tested; compact packaging corrected

The recommended genuine-sentence experiment is complete. The rejected 448-token
2M-Flores auxiliary CTC transfer was replaced with label-free masked temporal
reconstruction and token-contrastive alignment over bounded genuine sentence crops.
The 100-sign CTC head remained frozen. A deterministic train-only 127/28 split of the
155 acquired 2M-Flores `dev` sentences selected epoch 13 after 80.79 seconds on capped
MPS. The checkpoint is
`artifacts/models/stage2_v17_2m_flores_temporal_pretrain_v1/temporal_pretrained.pth`,
SHA-256 `ba543b1fd9fa9dd5827b5c67b5f4b0b4a52748187d45513b17a87d64274519ab`.

A controlled 0/10/20/30% synthetic replay sweep selected 10%. Seed 12701 scored
12/24 ASLLRP phrase edits, 7/259 local phrase edits, and 41/254 JONATHAN contextual
edits; seeds 12702 and 12703 reproduced 12/7/41 and 12/8/41. A final label-checked
selector-oversampling experiment scanned all 39,350 rows, found 714 selector-owned
rows but only 85 exact rows, and regressed to 12/9/41. Temporal pretraining therefore
improves the compact contextual gate from 43 to 41 edits but regresses ASLLRP phrases
from 11 to 12. It fails the no-regression contract and is not promoted. The selected
accuracy-research model remains the two-head general selector at 9/24, 6/259, and
43/254 edits.

The compact-student packaging path had a real mismatch: validation used the HOME/WHERE
context residual while `best_model.pth` stored only the bare CTC head. The trainer and
a standalone packager now save the exact evaluated `Stage2ContextAdapterV17` graph and
verify a cold reload. The retained compact artifact is
`artifacts/models/stage2_v17_compact_context_student_v1/model.pth`, SHA-256
`623f9b56141643704b3562a8d2fdcebe44269985b2f618eb8f0a471e857a2cf5`, reproducing
11/24 ASLLRP, 7/259 local, and 43/254 contextual edits. The non-promoted temporal
alternative is separately packaged at SHA-256
`a2568f6d38416a41a5f9b547224c50740874bd046cfa268c2f4a58166c88c4e6` and reproduces
12/7/41.

Full evidence is in
`artifacts/reports/stage2_v17_2m_temporal_replay_v1/EXPERIMENT.md`. Stage 3 may now be
developed as a separately evaluated reference-gloss-to-English module, but current
recognition evidence does not justify a general end-to-end translation claim. Mobile
export engineering may proceed from the exact compact artifact, but v17 Stage-2 Core
ML conversion/parity, app integration, and physical-iPhone measurement remain open.
No Citizen, SemLex, local, 2M-Flores `devtest`, or consumed RIT test split was accessed.

## 2026-08-22 17:32 PST — first complete content-gated AI signing voice delivered

The promoted signing voice is **not** the unconstrained neural residual decoder.  That
route was rejected because its emitted-style AUC improved only by worsening held-out
geometry 5–8%.  Its code and diagnostic artifacts remain research-only.  The final
system is a compact, interpretable 16-D profile latent learned from 63 train-only
voices.  It factors each real trajectory against its class medoid, learns robust
per-node XYZ signing-space offsets from other glosses, mixes at least three voice
latents to create a new voice, restores mixed signer duration, and uses the frozen
transition timing/inpainting stack for phrase boundaries.  At inference, the frozen
Stage-1 landmark branch selects the strongest style strength in
`[1.0, 0.75, 0.50, 0.40, 0.25, 0.0]` that retains the requested class.  Thus style
cannot silently overwrite content.

The frozen 16-D/content-gated design was evaluated unchanged across three disjoint
folds: 21 entire held-out train-only identities and 2,673 clips.  Aggregate generated
content is 93.9394% versus 93.6775% for the unstyled medoids; spatial reconstruction
improves 15.1171%; velocity and acceleration are unchanged to floating-point noise;
and the strict same-content signer verification AUC is 0.5900.  That AUC ranks the
target signer's other-gloss profile against other signers while holding the exact
requested gloss/prototype fixed.  The hash-pinned summary is
`artifacts/reports/signing_voice_profile_signer_disjoint_summary_v17.json`, SHA-256
`535a3abc5bd70979a24507a9731879b459e7cd05d12754d4b91e28ccf7585d33`.
This is credible content/style generalization evidence, not a fluent Deaf-signer
naturalness or linguistic-correctness judgment.

The fixed-design train-all profile uses all 63 eligible identities and 3,974 examples;
no model selection was performed on it.  It is
`artifacts/models/signing_voice_profile_v17_allvoices_final/model.pth`, SHA-256
`eec0ab97a7dea26fa2c53ec4d2afd3e27f530e8395db6e6115fd838859efbcae`.
Its result report SHA-256 is
`a1cb006e7b4dc350399cce2f7c7e91ac246059dbb13b94c1a25ea8406e42ab3f`.

The requested visual evidence is a complete generated `GOOD MORNING FRIEND` phrase in
three novel voices (Aster, Cobalt, and Juniper), not a hidden interval inside a human
trajectory.  Every voice uses a distinct convex mix of three learned voices; maximum
cosine between the three decoded profiles is 0.1277 and maximum cosine to any training
profile is 0.4315.  All nine generated isolated tokens retain full style strength and
the frozen Stage-1 branch predicts all nine correctly.  Phrase lengths are 79, 81, and
88 frames, and learned boundary spans differ across voices.  The 1920x900, 30-fps,
266-frame H.264 video is `artifacts/reports/signing_voice_phrase_v17.mp4`, SHA-256
`01f083a9fba3e7480e24c7b701c3bebbfbbe0daf709a36149a1dba8847822c4d`.
The preview is `artifacts/reports/signing_voice_phrase_v17_preview.png`, SHA-256
`6c0ad7a22dc6f3e7154d41bf190dbad2771a12e7da835f427fb15d17256251ba`;
the provenance report SHA-256 is
`e2bd22040ba1cf25fd189447e3747bea48ffe0e736b596cc888a1a855ea82d01`.
This is an abstract articulated avatar driven by generated landmarks, not photorealistic
RGB synthesis.

Cold reload reproduces all three raw phrase trajectories within float16 tolerance,
repeats all nine correct content predictions, verifies every checkpoint/report/media
hash, and decodes all 266 video frames.  The cold-reload report is
`artifacts/reports/signing_voice_profile_package_cold_reload_v17.json`, SHA-256
`b6fceebcca79c83af9e741d29109ebf4d841ffbad2e85d4654396e06d9d71cf3`.
Thirty-one affected signing-voice, transition, and Stage-2 tests pass; all changed
entry points compile, final JSON parses, full video decode passes, and
`git diff --check` passes.  No Citizen, SemLex, local, How2Sign, or project validation/
test split was accessed.

The earlier transition inpainter remains separately recorded as a promising
extractor/Stage-1 hard-gloss augmentation: hide a difficult real interval, reconstruct
it from genuine context, and retain the sample only when the class label and integrity
checks remain unchanged.  It is not part of the 16-D voice profile itself and has not
yet been promoted into Stage-1 training.

## 2026-08-22 16:57 PST — direct emitted-style supervision added

The first strict content-matched fold-0 diagnostic was interrupted after epoch 7: it
improved reconstruction but emitted-style AUC remained near chance (0.526–0.532), so
it is correctly rejected as a signing voice.  The missing loss is now explicit and
mirrors the evaluation contract.  A generated gloss is pulled toward its real target
from the same signer and exact gloss; only same-gloss/different-signer batch rows act
as negatives.  The original cross-gloss real-reference loss remains responsible for
learning a content-independent conditioning style.  The new emitted-style loss has
weight 0.25 in both held-out and final train-all scripts.  Nine focused tests pass,
and fold 0 has restarted from scratch.  No interrupted checkpoint is eligible and no
sealed or project validation/test split was accessed.

## 2026-08-22 16:54 PST — emitted-style metric is now content-matched

The first three-epoch rerun was also interrupted and is ineligible after detecting a
class-frequency confound: different held-out identities cover different gloss subsets,
so comparing generated and real embeddings across arbitrary classes lets content help
identify the signer.  The final verification contract now compares each generated
gloss only against real targets of the **exact same gloss**: its aligned same-signer
target is positive and all available same-gloss/different-signer targets are negatives.
Rows without a same-gloss cross-signer negative are excluded from both sides.  This
simultaneously verifies generator output, signer style, and content control.  Eight
focused tests pass, and fold 0 has restarted from scratch.  No checkpoint from either
interrupted diagnostic run is eligible; no sealed or project validation/test split was
accessed.

## 2026-08-22 16:51 PST — generated-style verification corrected before promotion

Fold 1 was interrupted after epoch 7 because a deeper audit found that the prior style
AUC compared two **real** same-signer clips.  That proved the encoder could identify
signer manner but did not prove the generator emitted it.  Consequently, the 0.8184
style-AUC statement in the 16:46 entry below is invalidated, and its fold-0 checkpoint
is not promotion-eligible despite its still-valid reconstruction and content results.

The corrected metric now embeds the fully generated gloss, compares it with a real
same-signer target gloss, and ranks those positives against exhaustive generated-to-
real cross-signer pairs.  It separately records generated-to-conditioning-style
cosine.  Eight focused signing-voice tests pass.  All three folds will be rerun from
scratch under this stricter output-style metric before any final artifact or video is
promoted.  No sealed or project validation/test split was accessed.

## 2026-08-22 16:46 PST — signing-voice fold 0 passes the frozen gates

The corrected contrastive fold-0 experiment selected epoch 12 and is
`artifacts/models/signing_voice_v17_fold0_contrastive/best_model.pth`, SHA-256
`1f9429d3ff141a40532730c3cad8560e185153df62e88eaacc05fc8da61cba0e`.
Across seven entirely held-out train-only identities (901 examples), generated spatial
error improves 2.4475% over the class medoid, velocity error improves 5.1772%, and
acceleration error improves 6.1021%.  The frozen Stage-1 landmark branch recognizes
100% of generated glosses versus 94.4506% for the unstyled class medoids.  Exhaustive
signer-aware cross-gloss style verification uses 901 positive and 632,068 cross-signer
negative pairs: AUC improves from the untrained encoder's 0.7165 to 0.8184.  This is
credible held-out content/style/reconstruction evidence, not human naturalness.

The architecture, loss weights, stopping rule, and evaluation contract are now frozen.
Fold 1 is running unchanged; fold 2 will follow.  No sealed or project validation/test
split was accessed.

## 2026-08-22 16:38 PST — signer-aware signing-voice metric correction

The first contrastive fold-0 run was deliberately interrupted after epoch 8 because
its validation negative used the next row in a batch, which was not guaranteed to be
a different signer.  No checkpoint from that interrupted run is eligible.  Style
verification now compares every aligned same-signer/different-gloss reference-target
pair against exhaustive cross-signer pairs and records both pair counts.  A new
contract test verifies that same-signer off-diagonal pairs never enter the negative
set; all five signing-voice tests pass.  Future checkpoints also store the 100
train-only class-median observed durations required for phrase-level timing.  No
sealed or project validation/test split was accessed.

The phrase-level implementation now has a fail-closed composition contract.  It
creates every requested 32-frame gloss from the fixed class prototype plus one shared
continuous style latent, restores train-only class/signer timing, predicts each
boundary span from the generated endpoints, synthesizes the entirely missing
coarticulation interval with the frozen train-all transition inpainter, and concatenates
the result into one complete trajectory.  A “novel voice” must be a normalized convex
mix of at least three unique learned voice centroids, with no source weight above
0.60.  Seven focused tests now pass, including full-phrase boundary/timeline and novel
style-mixture contracts.  The train-all script and a three-avatar same-phrase renderer
are implemented but must not be presented as final evidence until the corrected
signer-disjoint folds, fixed-epoch train-all run, cold reload, and video validation
finish.

## 2026-08-22 16:15 PST — true content/style signing-voice experiment started

The earlier transition inpainter is explicitly retained as a separate candidate for
continuous-trajectory augmentation, including future hard-gloss Stage-1 experiments.
It must not be renamed a signing voice: it reconstructs only a missing interval inside
an existing human trajectory.

The new signing-voice task generates a complete 32-frame isolated gloss from two
separate inputs: a requested class prototype supplies content, while a reference clip
of a **different gloss from the same signer** supplies style.  The style encoder never
receives the reference label.  This makes content copying through the reference a
fail-closed data-contract violation and establishes the required content/style split
for later full-phrase composition and novel latent-style interpolation.

The existing 67-identity train-only feature pool has now been materialized in raw v17
landmark space with exact index alignment: 3,978 trajectories, all 100 classes, 1,475
Citizen official-train items, 1,115 contextual ASLLRP train items, and 1,388 SemLex
official-train items.  Sixty-three dataset-local identities have at least two distinct
classes and are eligible as style voices.  The pool is
`data/local/signing_voice_v17/train_only_landmark_pool.npz`, SHA-256
`0764c4295524d48417f2fd89058c7037c65ccb1d477223104a2a84be61435702`;
its report is `artifacts/reports/signing_voice_v17/landmark_pool.json`.

`active/v17/model_signing_voice_v17.py` implements a continuous 32-dimensional style
encoder and a content-prototype residual Transformer.  Presence/confidence and missing
nodes remain tied to the class prototype; only XYZ style motion is generated.  A
frozen selected Stage-1 landmark branch enforces content preservation.  The first
three contract tests pass.  A two-epoch smoke selected epoch zero, correctly rejecting
premature style perturbation; the predeclared fold-0 signer-disjoint pilot is now
running with seven entire identities held out (three Citizen, three SemLex, and
RACHEL).  At epoch 8 it first exceeded the medoid baseline while retaining 100%
generated-content recognition.  No sealed split or held-out project validation signer
was accessed.

## 2026-08-22 15:54 PST — transition synthesis rendered for direct inspection

A reproducible six-panel visual demonstration now shows the original human RGB,
genuine extracted landmarks, linear interpolation, learned deterministic transition,
and stochastic temperatures 0.10/0.20.  It contains three high-motion examples from
How2Sign signer 3, How2Sign signer 5, and one public channel voice proxy.  The timing
model exactly recovers the deliberately hidden spans in all three demonstrations
(8/8, 8/8, and 11/11 frames); this is a visualization sanity check using the final
train-all artifacts, not new independent accuracy evidence.

The 16.1-second 1920x1080 H.264/yuv420p MP4 is
`artifacts/reports/transition_multivoice_visual_demo_v17.mp4`, SHA-256
`5ac68b3e652ed2660efcee30a4c3130dbc877dda95991baefcc4d9f6e9e47617`.
All 483 frames decode successfully.  The preview PNG is
`artifacts/reports/transition_multivoice_visual_demo_v17_preview.png`, SHA-256
`1e3a4a77ce55a6491037c07d92dbd679a38cbd5d470935f1c3a6192e9754757f`,
and the provenance report is
`artifacts/reports/transition_multivoice_visual_demo_v17.json`, SHA-256
`74ef0f637c20703f265bebe04ee3ac1c83f09aeabb10f53603acaa07ce6730c9`.
The yellow interval in the video is the only region synthesized; visible context is
identical.  This is an abstract landmark rendering, not synthesized RGB or a human-
naturalness rating.  No sealed split was accessed.

## 2026-08-22 15:46 PST — 133-source/proxy transition-voice package complete

The held-out-fold settings are now being converted into train-all deployment
artifacts without new model selection.  The final deterministic mean used all six
How2Sign train-shard signers plus all 127 usable YouTube-ASL channel-level voice
proxies, with the frozen 90%/10% source balance, LR `5e-5`, and median 63 selected
epochs.  It trained on 4,938 How2Sign and 994 web windows and is
`artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth`,
SHA-256 `eba3bbd5086c04f099e4466ac211a7b69322a0dd1f29d680a3134982a4cc7e2a`.
This train-all checkpoint inherits the three-fold held-out evidence; its own training
loss is not an independent accuracy score.

The final 10-epoch stochastic residual layer used the same 133 sources/proxies and
source-balanced sampler.  Its residual normalization is the root of the exact
90%/10% weighted source second moments, avoiding a hidden corpus-size bias.  It is
`artifacts/models/transition_residual_diffusion_multicorpus_v17_allvoices_final/model.pth`,
SHA-256 `0618f7d6976219a6d3180eb24dffdd2570e79bef4437107e23fc348561f1f5d5`,
and pins the deterministic checkpoint hash exactly.  Recommended temperatures remain
0.10 and 0.20 from prior LOSO selection.

The all-voice timing predictor then completed the frozen 39-epoch train-all schedule
over 53,388 balanced span examples.  It is
`artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth`, SHA-256
`b752ecf1bebbb6e82ccc803e86beee40adf771fd56750da3811363b0f3c1c555`.
It inherits the three-fold elapsed-span evidence; its own training loss is not an
independent semantic-timing score.

An audit of a suspected timing-feature leak confirmed there is none: v17 channel 5 is
per-frame landmark confidence, not motion derived from an earlier hidden frame.  The
timing model receives only XYZ, presence, and confidence from the visible eight-frame
context on each side.  Twenty affected Stage-2/transition tests pass.  The completed
runs used `num_workers=0` and the 10% MPS process cap; observed RSS stayed below
0.5 GB and system free memory was 45–52%.

All three components cold-reload together from disk on a real extracted How2Sign
trajectory.  The mean is finite and preserves every visible value exactly; stochastic
outputs at 0.10/0.20 are finite, nonzero, bounded, and preserve visible context; and
the timing checkpoint emits finite nine-class logits and a valid 4–12 frame span.
The cold-reload report is
`artifacts/reports/transition_multivoice_package_cold_reload_v17.json`, SHA-256
`734576a216e69d4737c8effa01053a8d1e40aafafe2d260f9f1a468b7b84aed3`.
The canonical hash-pinned package/evidence manifest is
`artifacts/reports/transition_multivoice_package_v17.json`, SHA-256
`5e99828f31390cfb89fc18352e1051592e7e30379bb8e72a26cd963eb8422c4b`.
It records 6 controlled How2Sign signers plus 127 channel-level voice proxies, not 133
identity-verified unique people.  Model sizes are 7.7 MB mean, 8.3 MB diffusion, and
1.7 MB timing.

All 20 affected Stage-2/transition tests pass; all changed entry points compile, all
final JSON parses, all stored hashes/linkages match the bytes on disk, and
`git diff --check` passes.  This completes a generalizing landmark-space experiment,
not a human-naturalness claim: semantic prosody, RGB rendering realism, and blinded
Deaf-signer preference remain unmeasured.  No Citizen, SemLex, local, How2Sign
validation/test, 2M-Flores `devtest`, or consumed RIT test row was accessed.

## 2026-08-22 15:25 PST — signer-context timing passes three held-out folds

Natural transition timing is now measured separately from motion inpainting.  The
new task removes a genuine 4–12 frame interval from a real continuous 32-frame
trajectory and presents only eight visible frames on each side.  The model must
recover elapsed span length without receiving the mask width.  Every source window
contributes all nine balanced target lengths.  The same two-layer, signer-ID-free
context transformer, 10% web balance, seed, and 40-epoch schedule were frozen on
signer 8 and applied unchanged to signers 3 and 5.

Across 42,174 held-out How2Sign examples (the same 4,686 unseen-signer windows), exact
span accuracy is 92.3887%, macro F1 is 0.924508, within-one-frame accuracy is
96.1849%, and MAE is 0.1721 frames.  Across 5,184 channel-held-out web evaluations,
exact accuracy is 87.1528%, macro F1 is 0.876557, within-one-frame accuracy is
91.6860%, and MAE is 0.3258 frames.  Fixed eight-frame timing is 11.1111% accuracy
and 2.2222-frame MAE.  A direct boundary-distance/observed-speed rule is about 12%
accuracy and 3.63–3.73-frame MAE.  Crucially, repeating only the two endpoint poses
while removing local temporal style collapses the trained model to about 11%
accuracy and 2.85–2.88-frame MAE.  Thus genuine local rhythm, not endpoints alone,
drives the result.

The fold checkpoints are
`artifacts/models/transition_span_multicorpus_v17_h3_w010/best_model.pth`
(`fdca6a12e7e4dd64b9234e779b2ea0af9083d4cc6a82b343743ab422ec996257`),
`...h5...` (`085733a44f5219ad05e8b479d216c148e43f911407ef12f48a85c725ccedf9b4`),
and `...h8...` (`1bd5016279272449a1e94e43a39d6410e93cbd276d4e49f1cd7868f54a52ccc2`).
The aggregate report is `artifacts/reports/transition_span_loso_summary_v17.json`,
SHA-256 `e2a71e66575a55d939dd2989d9008b485e92c91b84fcd23c66d64f316fcb780b`.
Eleven focused transition/timing tests pass.  This is strong self-supervised elapsed-
span evidence; it is not semantic prosody or a human-perceptual naturalness result.
No sealed split was accessed.

## 2026-08-22 14:52 PST — 127-channel transition adaptation passes three-fold reconstruction gates

The bounded YouTube-ASL acquisition is complete without downloading the 984-hour
corpus.  It retained 128 hash-pinned 30–38 second derivatives from 128 distinct
public channels (104 train, 24 channel-disjoint internal validation), rejected 11
uncuttable candidates, occupies 72 MB, and preserves each source aspect ratio.  Two
spaced visual contact sheets showed active signing in all 24/24 sampled clips.
`active/v17/youtube_asl_transition_manifest_v17.json` has SHA-256
`4865a62dc44a4b546c130bfa21ed27763538897ad6b86ff312c9d4b8ce4f09f3`;
all 128 file hashes match and there is no role overlap.  Channels are explicitly
voice proxies, not verified one-to-one signer identities.

The fixed Apple Vision v17 extractor completed in 349 seconds with stable sub-1-GB
RSS.  It produced 127/128 usable voice-proxy archives: 103 train voices/802 valid
windows and all 24 validation voices/192 valid windows.  The sole failed train clip
had no valid hand window.  The landmark tree is 9.7 MB, SHA-256
`c2b3ce5a1963e24cdbd7189bb5c9200155dfcd4233097d850c53eae37d2e0ce8`,
and passes the enforced 80-train/16-validation breadth floor.  Extraction and audit
reports have SHA-256 `77ef1023be0ea1e8807ed441950e4e66475ef7a2deb3aac11f16ef3ca99dee59`
and `24bf947e5949e40fdb095e68055c7097209953bbf5c4e8e0d449bf66c2c0b6e5`.

Multi-corpus adaptation warm-started each exact How2Sign-only fold checkpoint and
used source-balanced sampling with a zero-regression held-out How2Sign selection
floor.  Web probabilities 10%, 25%, and 50% were tested on signer 8; 15% and 20%
refined the only promising interval.  Although 50% maximized reconstruction, it
regressed the How2Sign grouped discriminator.  Ten percent was frozen because it was
the only selection candidate improving reconstruction and both discriminator
statistics on both signer-8 domains.  That exact 10%, LR `5e-5`, seed, architecture,
and schedule were then applied unchanged to signer 3 and signer 5.

All six fold/domain reconstruction comparisons improve.  Across 4,686 held-out
How2Sign windows/971 clips, relative reconstruction improves from 18.1259% to
20.2012%, the improved-window fraction rises 68.9714% to 69.8250%, discriminator
balanced accuracy falls 61.4810% to 61.1289%, and AUC falls 0.653906 to 0.648197.
Across the same 24 held-out web channels evaluated by the three independent fold
models, reconstruction improves 8.7223% to 10.9495%, improved-window rate rises
68.2292% to 71.7014%, and AUC falls 0.580539 to 0.570041.  Web balanced accuracy is
the one mixed metric: 55.2083% to 55.3819% (+0.1736 point), so it must not be reported
as an across-the-board distribution win.  The pinned aggregate is
`artifacts/reports/transition_inpainter_multicorpus_loso_summary_v17.json`, SHA-256
`a1df378f741fe758298a57c00329c356c52d29cdb1bfe09f32d029a55a659084`.
This is strong multi-voice landmark evidence, not human-perceptual naturalness.
No sealed split was accessed.

## 2026-08-22 13:40 PST — multi-corpus selection is now warm-started and no-regression constrained

A normalized contact sheet of the first 12 independent YouTube-ASL channels confirms
active human signing in all 12, with varied people, framing, backgrounds, signing
spaces, and both one- and two-person material.  The bounded downloader remains stable
and resumable; 52/128 channel clips were complete at this timestamp, with one failed
candidate automatically covered by the discovery reserve.  The acquisition process
uses about 40 MB RSS and does not create memory pressure.

The multi-corpus trainer no longer starts each fold blindly or selects only an
unconstrained cross-domain average.  It can now warm-start from the exact proven,
fold-matched How2Sign checkpoint, rejects mismatched held-out signers/configurations,
and pins the initial checkpoint hash.  The initial held-out How2Sign relative
improvement becomes a hard model-selection floor (zero regression tolerance by
default), while channel-held-out YouTube-ASL quality contributes to selection above
that floor.  This preserves the established signer-generalization evidence while
testing whether many new channel-level voice proxies improve web generalization.
The optional motion-distribution term is exposed but remains disabled because the
prior targeted experiment regressed held-out metrics.  Eight focused transition tests
pass, the changed entry points compile, and no sealed split was accessed.

## 2026-08-22 13:28 PST — genuine web-voice expansion is in progress

The six-voice How2Sign artifact is no longer being treated as the endpoint.  A
source audit tested OpenASL first.  Its pinned public train TSV and signer boxes are
SHA-256
`e7e1559bcef5ac77d2c14c2ccfc9db54516768e296235f266b3f4f96de459a40`
and `a79b5327956670db0988bacd96aebb9229ecd5ea948b8de887cb530669152a44`.
All 2,007 eligible 4–12 second train-source videos were queried.  They resolve to
only three public channels (`Sign1News`, `The Daily Moth`, and `nad1880`).  This
matches the literature warning that OpenASL's approximately 220 signer identities
come from finer-grained source metadata that is not present in the public TSV.
Therefore OpenASL channel IDs cannot honestly prove a large voice count.  Three
visually valid clips were retained for possible domain work, but OpenASL was rejected
as the primary voice-expansion evidence.

The replacement source is the official, human-filtered YouTube-ASL video-ID release.
Its generation-pinned list contains 11,096 unique IDs and has SHA-256
`ca5622737279afc33b1f9ddfa585b5e8bd284f1be458c461e10887b03e519191`.
Unlike OpenASL, 195 distinct public channels were found after only 336 deterministic
ID probes.  Acquisition targets 128 channel-level voice proxies: 104 internal train
and 24 channel-disjoint internal validation.  One clip per channel is retained.  The
initial 2–10 second smoke sample exposed an opening title without hands; the contract
was corrected to 30–38 seconds and the same source then visibly contained continuous
signing with face and both hands.  Every derivative preserves source aspect ratio,
uses a maximum 720-pixel side, and is normalized to 30 fps.  At this timestamp 22
channel clips are complete; acquisition is resumable at
`data/local/youtube_asl_transition_subset_v17/acquisition_state.json`.

New reproducible entry points are
`scripts/acquire_openasl_transition_voices_v17.py`,
`scripts/acquire_youtube_asl_transition_voices_v17.py`,
`scripts/prepare_youtube_asl_transition_manifest_v17.py`, and the role-preserving
continuous extractor `scripts/extract_how2sign_transition_landmarks_v17.py`.
`active/v17/train_transition_inpainter_multicorpus_v17.py` implements a
signer-ID-free, source-balanced sampler and joint model selection on one unseen
How2Sign signer plus channel-disjoint YouTube-ASL voices.  It is not allowed to
replace the six-voice model unless both domains improve.  Its sampler and split
invariants are covered by two new focused tests; all eight transition tests pass.
No project sealed split was accessed.

## 2026-08-22 12:42 PST — multi-voice transition layer packaged after stochastic LOSO

The bounded stochastic residual experiment is complete across the same three
How2Sign leave-one-signer-out folds as the retained deterministic inpainter.  The
diffusion model predicts only XYZ residual motion around the frozen deterministic
mean, only inside a contiguous transition mask.  It receives visible motion context
but no signer ID.  All six domain-matched How2Sign train-shard voices participate
across the folds; the two NCSLGR voices remain a separate domain because direct equal
pooling had already regressed unseen-How2Sign evidence.

The two operating temperatures were frozen on signer 8 and applied unchanged to
signers 3 and 5.  Across 4,686 unseen-signer windows/971 source clips, deterministic
mean motion scores 61.4810% balanced accuracy / 0.652976 macro ROC AUC under the
grouped genuine-vs-generated discriminator, versus 67.9684% / 0.730314 for linear
interpolation.  Temperature 0.10 retains a 15.9245% weighted reconstruction
improvement over linear, improves 57.5758% of individual windows, is 2.6963% worse
than the deterministic mean, and scores 61.2356% / 0.644220.  Temperature 0.20 trades
more reconstruction fidelity for diversity: 12.0351% better than linear, 46.6069% of
windows improved, 7.4755% worse than the mean, and 60.3073% / 0.631665.  The pinned
fold report is
`artifacts/reports/transition_residual_diffusion_loso_summary_v17.json`, SHA-256
`f619d94f1d1977bcc5ca9f428af8192cc4a8e5bf48fdbc89320e643cbc5f0e3a`.
Temperature 0.10 is the accuracy/diversity mode; 0.20 is the stronger-diversity mode.

Fixed train-all artifacts were then fit on 4,938 windows from all six How2Sign
train-shard voices, without model selection or independent scoring on those same
rows.  The deterministic mean is
`artifacts/models/transition_inpainter_v17_all6_final/model.pth` (7.7 MB), SHA-256
`7e99aa7b3d8723d47c610230c9ddb87931809cd8f3b3ab7e888013ef5ca2a1bd`;
the 10-epoch stochastic residual layer is
`artifacts/models/transition_residual_diffusion_v17_all6_final/model.pth` (8.3 MB),
SHA-256
`05f8b7aedc69db94a6abe3b32c1d556749c4d1e627da7a6ab8b2dcd7dcd818d9`.
Both used `num_workers=0` and a 10% MPS process cap.  Peak observed RSS was about
0.49 GB for the mean fit and 0.82 GB for diffusion, with system free-memory pressure
at 45% or better; neither run leaked or approached red pressure.

A cold reload from disk on a real extracted trajectory passed at both temperatures:
outputs were finite, every visible context value was preserved exactly, and masked
XYZ deviations were nonzero and inside the hard residual bound.  The verification is
`artifacts/reports/transition_voice_artifact_cold_reload_v17.json`, SHA-256
`71fbd77fcb059cdd50e1b747b34f8a585449b4c3545c3264a09baf0c0703c0e1`.

All 15 affected Stage-2/transition unit tests pass.  Every new Python entry point
compiles, all final model/result/history and aggregate-report JSON files parse, stored
checkpoint hashes and mean-to-diffusion linkage match the bytes on disk, and
`git diff --check` passes.

This establishes a generalizing, context-conditioned **landmark transition layer**—a
useful component of the requested signing “voice.”  It is not text/gloss-conditioned,
does not render RGB, and does not establish human-perceptual naturalness.  A fluent
Deaf-signer blinded preference study on rendered held-out-signers remains mandatory
before calling it genuine human-natural signing.  It also does not change the Stage 2
recognizer's WER.  No Citizen, SemLex, local, How2Sign validation/test, 2M-Flores
`devtest`, consumed RIT test, or JONATHAN data was accessed.

## 2026-08-22 12:07 PST — full leave-one-signer-out transition evidence

Full extraction is complete.  How2Sign yielded 1,026/1,027 usable train-shard clips;
the sole exclusion, `0HfN3Ts0FxQ_18`, was retried and visually inspected and contains
no usable Apple Vision hand window (the signer is resting with hands below/in the lap).
The already acquired NCSLGR continuous subset added 166/166 usable clips from two more
human signers.  The combined tree has 1,192 archives, eight signer identities, 5,499
valid 32-frame windows, seven invalid windows excluded by saved masks, and SHA-256
`79cc83c2c2ff711f505b9af1ceca5271386287b676a98737e65e74e07c487806`.
The complete corpus audit is
`artifacts/reports/transition_inpainter_full_corpus_v17.json`.

Directly pooling the two cross-corpus NCSLGR voices was rejected for the How2Sign
held-out-signer task.  On signer 8, the eight-voice model improved reconstruction by
15.9577% and scored 62.7939% balanced accuracy / 0.666003 ROC AUC under the grouped
genuine-vs-generated discriminator.  The otherwise identical How2Sign-only model
improved by 17.6497%, improved 72.1725% of windows, and reduced discrimination to
61.8141% / 0.655865.  NCSLGR remains useful as a separate genuine domain; it must not be
pooled at equal weight without source balancing or domain-aware adaptation.

The retained residual design was then evaluated leave-one-signer-out across all three
large How2Sign voice pools, always training on the other five How2Sign signer IDs and
never using learned signer-ID embeddings.  The same seed and hyperparameters improved
all three unseen signers:

- signer 3: 840 windows/172 clips, 15.4908% reconstruction improvement (95% bootstrap
  CI 12.6802–18.3631%), discriminator 59.9405% balanced accuracy / 0.648756 AUC versus
  linear 65.2381% / 0.705816;
- signer 5: 2,060 windows/399 clips, 19.6131% (CI 17.7347–21.5042%), discriminator
  61.8204% / 0.654308 versus linear 69.8544% / 0.766070;
- signer 8: 1,786 windows/400 clips, 17.6497% (CI 15.8425–19.5685%), discriminator
  61.8141% / 0.655865 versus linear 67.0773% / 0.719056.

Across 4,686 unseen-signer windows from 971 source clips, weighted reconstruction
improves 18.1259%, 68.9714% of windows improve, discriminator balanced accuracy falls
from 67.9684% for linear interpolation to 61.4810%, and macro AUC falls from 0.730314
to 0.652976.  The pinned fold-by-fold report is
`artifacts/reports/transition_inpainter_loso_summary_v17.json`.  This is strong evidence
that context-conditioned residual motion generalizes and is closer to genuine motion;
61.48% discrimination remains above chance, so it is still **not** a human-naturalness
pass and not a rendered-video/text-to-sign system.

The full models are approximately 2.02 million parameters/7.7 MB.  Dataset preloading
is bounded at about 108 MB, MPS remains capped at 10%, and vectorizing the interpolation
prior reduced full-corpus epochs from about 13 seconds to 3–5 seconds without changing
the exact baseline (focused tests pass).  A targeted per-articulator velocity/
acceleration moment loss was also rejected: on signer 8 it regressed reconstruction
from 17.6497% to 16.0072%, reduced the improved-window fraction from 72.1725% to
68.5890%, and worsened discrimination from 61.8141%/0.655865 to
62.4020%/0.657110.

No Citizen, SemLex, local, or How2Sign sealed split, 2M-Flores `devtest`, consumed RIT
test, or JONATHAN data was accessed.  How2Sign and NCSLGR were used only for train-side
self-supervised motion reconstruction, never as CTC gloss supervision.

## 2026-08-22 11:22 PST — genuine held-out-signer transition pilot improves on interpolation

The bounded How2Sign acquisition completed successfully: 1,027/1,027 train-shard
clips, zero failures, 1,482,334,057 video bytes, 1.6801778 hours, six signer IDs, and
348 source videos.  The deterministic plan SHA-256 is
`b88f2cab457ad2bc31a9189a7050746a315aa6a23c682df60fafeb4de4e69113`;
the completed-file ledger is
`180ea6af1a7a51cd032399726b93e3bc42646e35636630f991d46a943b09f8cd`.
`active/v17/how2sign_transition_manifest_v17.json`, SHA-256
`5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24bc9f`,
contains only train-role, unlabeled self-supervision rows and explicitly records that
How2Sign validation/test and every project sealed split were untouched.

Apple Vision smoke extraction passed 6/6 clips.  The expanded pilot passed 199/199
clips with no clip failures and produced 968 valid 32-frame windows: 766 training
windows from five signers and 202 validation windows from the completely unseen
`how2sign:8` signer across 48 source clips.  All 61 v17 nodes, including all lip nodes,
are preserved.  One candidate window lacked usable hands and is excluded by the saved
validity mask.  Reports are under
`artifacts/reports/how2sign_transition_landmarks_v17/`.

The first absolute-reconstruction Transformer was rejected: at its best epoch it was
111.44% worse than two-boundary linear interpolation on the held-out signer.  The
revised model uses interpolation as an exact zero-initialized residual prior and learns
only genuine deviations.  Three independent seeds all generalized to signer 8,
improving the composite spatial/velocity/acceleration score by 18.2004%, 19.8785%, and
20.3403%.  Seed 10703 epoch 59 is the best single reconstruction model:
`artifacts/models/transition_inpainter_residual_v17_pilot/best_model.pth`, SHA-256
`d1857b2fc1272fb3ad10aeee1faf0f9bf73b9d82608ddb4e97245db7a536a5ed`.
The run took 631.2 seconds on MPS capped at 10% memory with `num_workers=0` and showed
no memory-pressure failure.

The stricter source-clip-grouped paired audit is
`artifacts/reports/transition_inpainter_naturalness_v17_ensemble3.json`.  The equal
three-seed ensemble improves per-window reconstruction by 19.4824% with a 95% bootstrap
interval of 13.9227% to 25.2311%, and improves 69.8020% of 202 windows.  A held-out
linear discriminator distinguishes genuine motion from plain interpolation at 65.3465%
balanced accuracy / 0.695741 ROC AUC, versus 59.6535% / 0.621312 for the learned
ensemble.  Thus the learned residual is materially closer to the genuine held-out
distribution, but it is still distinguishable above chance and does **not** yet pass a
human-naturalness gate.  These are landmark-reconstruction results, not RGB synthesis
or text-to-sign production results.

Full extraction of all 1,027 acquired clips is now running so that leave-one-signer-out
experiments can rotate across the three large signer pools instead of selecting on one
voice.  Primary literature supports learning temporal alignment from continuous
signing and warns that direct pose regression under-articulates motion; the pilot's
slightly worse acceleration error is consistent with that failure mode.  Residual
motion modeling (and, only if enough genuine data supports it, a bounded stochastic
motion model) is the next research direction.  Four focused inpainter tests and all 13
affected Stage-2/model tests pass.

No Citizen, SemLex, local, or How2Sign sealed split, 2M-Flores `devtest`, consumed RIT
test, or JONATHAN data was accessed for this work.

## 2026-08-22 10:52 PST — synthetic transitions falsified; genuine multi-signer pilot started

The existing feature-space interpolation/style-transfer generator does **not** pass a
natural-human-motion gate.  `scripts/audit_stage2_transition_naturalness_v17.py`
matched generated and genuine train-only ASLLRP spans by exact ordered gloss pair and,
for the strict analysis, by the same signer.  A frozen label-independent temporal
descriptor plus train-fold-only PCA/logistic classifier separated 18 strict paired
examples (36 rows) with 97.2222% balanced accuracy and 0.99691358 ROC AUC.  Two hundred
paired permutations gave a null balanced-accuracy mean of 0.47346 and p=0.004975.  A
broader genuine-signer-held-out analysis over 33 pairs (66 rows) still reached 90.9091%
balanced accuracy and 1.0 ROC AUC.  The complete report is
`artifacts/reports/stage2_v17_transition_naturalness_audit_v1.json`.  Therefore the old
generator remains recognition augmentation only; it must not be described as genuine
coarticulation, timing, or signer-style synthesis.

The phrase-agnostic selector was also distilled into full-model and head-only compact
students across bounded seeds.  The best students reached 11/24 ASLLRP, 7/259 local,
and 43/254 JONATHAN edits.  They retained the contextual-sign benefit but did not
transfer the rare ASLLRP corrections, so neither replaced the selected two-head
research model.  Parameter soups likewise bottomed out at 11/7/43.  The selected
general selector remains 9/6/43.

A genuine-data experiment is now in progress using a bounded, train-only How2Sign
subset.  The official corpus provides continuous signing by 11 signers under CC
BY-NC 4.0; a per-file mirror was pinned at commit
`cfe9b6482aa34d6f6bda1974a7b7cae822c16613`.  Metadata was downloaded and a
deterministic plan selected 1,027 clips (1.6801778 hours), 348 source videos, and all
six signer IDs present in that train shard: signer counts 24, 23, 8, 172, 400, and
400.  Selection is capped at eight clips per source video and includes every eligible
clip from the three rarest voices.  The resumable state and plan are under
`data/local/how2sign_transition_subset_v17/`; at this timestamp 597/1,027 files had
completed with zero failures and about 899 MB downloaded.  How2Sign validation and
test are untouched.

New preprocessing preserves the genuine 32-frame trajectories and all lip nodes:
`scripts/prepare_how2sign_transition_manifest_v17.py` and
`scripts/extract_how2sign_transition_landmarks_v17.py`.  A new
`TransitionInpainterV17` masks a contiguous interval and reconstructs it from visible
context.  Its style representation is the visible-frame mean and variance from the
same clip; it has no signer-ID embedding, so the planned `how2sign:8` signer-held-out
evaluation measures adaptation from context rather than identity lookup.  Training is
capped at 10% MPS allocation with `num_workers=0` and compares spatial, velocity, and
acceleration errors directly against linear interpolation.  Four focused shape,
context-preservation, interpolation, and finite-loss tests pass.  Even if it beats the
linear baseline, that will establish feature reconstruction only; a held-out
discriminator/perceptual gate is still required before any natural-motion claim.

No Citizen, SemLex, local, or How2Sign sealed split, 2M-Flores `devtest`, consumed RIT
test, or JONATHAN data was accessed for these experiments.

## 2026-08-22 10:15 PST — phrase-agnostic 63-voice Stage 2 selector promoted

The phrase-specific `FRIEND NOW` gate is no longer the best general development
design. A new accuracy-research wrapper keeps the 63-voice context-adapted primary in
control, applies a class-agnostic 90/10 primary/direct-transition logit blend plus a
+0.30 blank calibration, and permits the specialist to own a row only when its
different greedy sequence has the same length (at least two signs) and has no lower
exact full-path CTC probability under the specialist. It never inspects gloss labels,
phrase identities, signer IDs, or a phrase allowlist.

The cold-reloadable artifact is
`artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth`, SHA-256
`0782d052f0500164a2433ebfee86dcce7413c6bcffca03fae379871ece86dc3d`.
Relative to its 63-voice primary, it improves every development domain: ASLLRP genuine
phrases from 11/24 to 9/24 edits (45.8333% to 37.5% WER; 4/12 exact), local phrases
from 7/259 to 6/259 (2.3166% WER; 92/97 exact), and JONATHAN contextual signs from
44/254 to 43/254 (16.9291% WER; 213/254 exact). It has the same 58 aggregate edits as
the former phrase-gated artifact but distributes them without encoding a known phrase
and improves local validation by one edit. The full report is
`artifacts/reports/stage2_v17_general_ctc_selector_v1/validation.json`, SHA-256
`4194dccb3723273bb4a76c0134de9b43968df292d41ee53b38d4c14b95efb8f7`.

A broader generic prefix-extension rule can recover 8/24 ASLLRP edits, but it was
rejected because it regressed local phrases to 13/259 and JONATHAN to 49/254; training
data also strongly rejects that behavior. The retained selector changes only
same-length multi-sign hypotheses. Its exact CTC dynamic program matches PyTorch's
reference CTC loss, saved-artifact reload reproduces train and validation metrics, and
the primary still carries all 63 train-only style voices. The Python/data-dependent
selector is not the compact Core ML graph; distillation remains required.

The 0.10 blend and +0.30 blank bias were selected during development exploration, so
these results remain development evidence. No sealed test or 2M-Flores `devtest` was
accessed. An independent signer/capture set is still required for generalization, and
recognition WER still does not prove perceptually natural human motion.

## 2026-08-22 10:05 PST — exhaustive multi-voice transition experiment audited

The train-only frozen pool can support much broader transition coverage than the
original random synthetic plans. Its 63 usable dataset-local style voices (29 Citizen,
31 SemLex, and three ASLLRP) collectively cover every one of the 9,900 ordered pairs
of distinct locked classes; each pair has at least eight eligible voices. A new
deterministic plan assigns every ordered pair to two different eligible voices and
combines the resulting 19,800 balanced transitions with 12,000 existing
ASLLRP-transition/style-transfer rows and 6,000 Citizen replay rows. The 37,800-row
plan is `active/v17/stage2_balanced_multivoice_plan_v17.json`, SHA-256
`5cda81ce5c30c8cdec99a71b15f3bf4bdc97f6880028608f3f3d08a7dd12d68b`.
Its builder and memory-bounded trainer are
`scripts/build_stage2_balanced_multivoice_plan_v17.py` and
`active/v17/train_stage_2_balanced_multivoice_v17.py`.

A conservative adaptation pilot selected epoch zero, so it was rejected. A three-seed
scratch direct-transition experiment completed on capped MPS in 350.1 seconds without
memory-pressure failure. Its best seed (1701, epoch 2) kept ASLLRP phrases at 11/24
edits but regressed local phrases to 106/259; the other seeds reached 15/24 ASLLRP and
24/259 or 29/259 local edits. This rejects raw feature interpolation as a standalone
model even when all ordered pairs and many voices are present: breadth of synthetic
coverage is not evidence that the transitions are human-natural.

Class-agnostic context adapters, model soups, logit ensembles, token-confidence fusion,
per-frame Potts decoding, and naive direct-expert voting were also rejected because
they worsened at least one development domain. A generic calibrated length-consensus
exploration reached 9/24 ASLLRP phrase edits (37.5% WER), 7/259 local edits (2.7027%),
and 43/254 JONATHAN contextual edits (16.9291%). Unlike the retained 8/24 direct-join
artifact, it does not name or match a specific phrase, but it is still a validation
exploration and has not been promoted or packaged. The next experiment is a generic
two-head CTC sequence-probability selector with all thresholds selected strictly on
train-only phrases/context before the development validations are scored.

No Citizen, SemLex, or local test split, 2M-Flores `devtest`, or consumed RIT test was
accessed. JONATHAN remains validation-only and was not used to train or synthesize any
voice. A genuinely new signer/capture set and fluent-signer perceptual evaluation are
still required before claiming unseen-signer generalization or human-natural motion.

## 2026-08-22 09:24 PST — direct-join specialist lowers ASLLRP and JONATHAN WER

The rejected scratch direct-isolated-join model was audited as a complementary expert.
Although it is weak globally, it recognizes three of five genuine `FRIEND NOW`
validation clips exactly, while the stronger 63-voice primary recognizes none exactly.
A loadable `Stage2DirectJoinSpecialistV17` now applies a 97/3 primary/specialist logit
blend and lets the specialist own a row only when its tensor-only greedy CTC collapse
is exactly `FRIEND NOW`. The gate fired exactly three times across 109 phrase and 254
contextual validation clips; all three were true `FRIEND NOW` phrases, with no local
or contextual false trigger. The 3% global residual also corrects one additional
JONATHAN `FRIEND` item.

Relative to the previous 63-voice candidate, ASLLRP genuine phrases improve from
11/24 to 8/24 edits (45.8333% to 33.3333% WER) and exact sequence accuracy rises from
2/12 to 5/12 (16.6667% to 41.6667%). JONATHAN contextual signs improve from 44/254 to
43/254 edits (17.3228% to 16.9291% WER). Local phrases remain exactly 7/259 edits
(2.7027% WER) and 91/97 exact sequences (93.8144%). All requested development gates
therefore improve or remain unchanged.

The selected artifact is
`artifacts/models/stage2_v17_direct_join_specialist_v1/model.pth`, SHA-256
`8efd55446a13acdc1c710da1db68ff2d72ffb3db1cbcc9258c55253c0c4acba0`.
It cold-reloads through the generic Stage 2 loader and reproduces every metric.
Separate generic-loader evaluations in `phrase_reload.json` and
`contextual_reload.json` reproduce the phrase and contextual metrics. The
gate and weight are explicitly validation-tuned: weights 0.00 through 0.30 were
inspected in 0.01 increments and 0.03 was the smallest nonzero value improving the
contextual edit count. This is development evidence, not an unseen-signer estimate.

Thirteen focused tests, Python compilation, generic artifact loading, independent
phrase/contextual reloads, JSON parsing, artifact-hash verification, and
`git diff --check` pass. No Citizen, SemLex, or
local test, 2M-Flores `devtest`, consumed RIT test, or JONATHAN training/synthesis data
was accessed. Full evidence, reviewed direct-stitching literature, and limitations are
in `artifacts/reports/stage2_v17_direct_join_specialist_v1/EXPERIMENT.md`.

## 2026-08-22 09:14 PST — 63-voice style transfer improves every Stage 2 dev gate

The signer-voice pool now contains 3,978 compatible train-only trajectories from 67
dataset-local identities: 1,475 Citizen clips/32 official-training signer IDs, 1,388
exact-variant quality-gated SemLex clips/32 official-training signer IDs, and 1,115
contextual ASLLRP segments from the three allowed training signers. Requiring at least
two distinct signs leaves 63 usable style voices: 29 Citizen, 31 SemLex, and 3 ASLLRP.
The pool SHA-256 is
`ee079873023b782bbc64c9fe4c64b32f4be2cb6d95494fa3fe446443e18e6653`.
SemLex frozen encoding completed in 29.43 seconds on MPS with only 103,579,648 peak
driver bytes.

Directly composing isolated Citizen/SemLex sign transitions was rejected: its best
seed retained 45.8333% ASLLRP phrase WER but worsened local phrases to 3.4749%. A
lower-rate second-stage attempt selected epoch 0 for every seed. The successful design
therefore preserves every core transition trajectory from one genuine continuous
ASLLRP training signer and transfers only the other 60 voices' observed duration
distributions and neutral endpoint context. Its 18,000-row plan includes 6,000 native
ASLLRP sequences, 6,000 style-transferred sequences balanced at 100 per additional
voice, and 6,000 full-vocabulary Citizen replay sequences. Plan SHA-256 is
`8c24e8268c38d840a8b10a9a59caf5d9dfecd607bac69f0dc9887d7f9cb34dc4`.

Three predeclared seeds were trained from the exact v2 checkpoint with the frozen-v2
distillation teacher. Seed 4702 epoch 12 was selected. Relative to the previously
retained context-adapted candidate, it keeps ASLLRP genuine phrases at the improved
11/24 edits (45.8333% WER), improves local phrases from 11/259 to 7/259 edits (4.2471%
to 2.7027% WER), and after the already fitted HOME/WHERE context residual at the
previously selected weight 1.5 improves JONATHAN contextual signs from 46/254 to
44/254 edits (18.1102% to 17.3228% WER). The loadable artifact is
`artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth`,
SHA-256 `f2ea9d99796e71b7355657a5bbcf791bdfedddec07adb54a053d9eba4292164b`.
A cold reload reproduces all metrics and all required development gates pass.

This is the new selected **development** candidate, not independent proof of natural
coarticulation or unseen-signer generalization. The ASLLRP phrase gain is one token on
only 24 tokens, dataset-local signer IDs are not claimed to identify unique people
across corpora, and WER cannot establish perceptual naturalness or nonmanual grammar.
Full evidence and rejected variants are recorded in
`artifacts/reports/stage2_v17_multivoice/EXPERIMENT.md`.

Citizen, SemLex, and local test splits, 2M-Flores `devtest`, the already-consumed RIT
external test, and JONATHAN as a synthesis source were not accessed. The next valid
accuracy step is a new signer/capture set; any visual-naturalness claim additionally
requires rendering plus fluent-signer evaluation.

Twelve focused Stage 2/model/data tests, Python compilation, the full 18,000-row plan
audit, six JSON parses, saved-artifact cold reload, and `git diff --check` pass. The
plan audit verifies all 63 style voices, every target/source mapping, and the absence
of JONATHAN from synthesis inputs.

## 2026-08-22 00:05 PST — signer-voice/coarticulation pilot improves phrase validation

The old mixed synthetic ASLLRP plan was found to sample each token independently,
allowing signer identity to switch inside one phrase, and to force every source sign
to 32 frames regardless of its decoded source duration. A train-only signer-voice
composer now holds one signer across each synthetic sequence, restores authoritative
source timing, performs monotonic boundary trimming (maximum three frames, minimum
four retained), adds a two-frame feature bridge, and uses five frames of signer-specific
neutral endpoint context. This is a feature-level recognition baseline, not a claim
of visually or linguistically natural human motion.

The generated plan has 12,000 sequences: 6,000 Citizen replay and 6,000 ASLLRP
signer-voice compositions, exactly 2,000 for each of BENJAMIN_JAMES_BAHAN, CORY, and
RACHEL, covering all 53 available ASLLRP classes. JONATHAN remained validation-only.
Plan SHA-256 is
`f0048f83047a4a969af491eab9bd9fbb2e12b59cab316ab5fcd3f370591fe75f`.

A scratch CTC pilot was rejected: seed 1702 epoch 6 scored 50.0000% ASLLRP phrase WER
and 13.8996% local phrase WER, although it newly recognized 3/5 `FRIEND NOW` clips
exactly. A conservative three-seed adaptation from the selected v2 checkpoint, using
a frozen v2 distillation teacher, selected seed 4702 epoch 4. It improved ASLLRP
phrase validation from 12/24 to 11/24 edits (50.0000% to 45.8333% WER) and local
phrase validation from 11/259 to 8/259 edits (4.2471% to 3.0888% WER). Its raw
contextual JONATHAN result regressed from 54/254 to 58/254 edits.

Applying the existing train-only HOME/WHERE context adapter and explicitly sweeping
development residual weights 0.5, 0.75, 1.0, 1.25, and 1.5 restored contextual
validation to 46/254 edits (18.1102% WER) at weight 1.5 while preserving both phrase
gains. The combined artifact is
`artifacts/models/stage2_v17_signer_voice_context_adapted_w1p5_pilot_v1/model.pth`,
SHA-256 `62677053984e675f6e6d3d792d0551bfc36c5192aabc431a112df99ed7fd2cce`.
It is retained as an experimental development candidate, not promoted as independent
proof: the ASLLRP improvement is one token on a 24-token validation set and the
candidate/weight were selected on development validation. Full design, literature,
metrics, and limitations are recorded in
`artifacts/reports/stage2_v17_signer_voice/EXPERIMENT.md`.

No Citizen, SemLex, or local test split, 2M-Flores `devtest`, or already-consumed RIT
test was accessed. The next valid step is a learnable monotonic transition model using
more genuine train-only parent utterances, followed by one-shot evaluation on a new
signer/capture set and fluent-signer review if visual naturalness is claimed.

## 2026-08-17 01:45 PST — Stage 2 development-validation gate cleared at 18.11% WER

The selected v2 CTC checkpoint remains the base, pinned to SHA-256
`dd5f3e620acf5e911f9373a14eddba6e0c0610d422cfaa27dd9de1eacc509cc9`.
Two attempts to transfer the new 2M-Flores full-gloss supervision into its shared
temporal encoder were rejected: the full auxiliary-CTC pilot and the conservative
partial-locked-CTC pilot both selected epoch 0 because every trained epoch worsened
the 254-row ASLLRP contextual validation. MPS CTC also produced non-finite loss, so
these small Stage 2 experiments now default to CPU with explicit non-finite guards.
The 2M-Flores data and frozen features remain valid future training assets; the failed
transfer mechanism, not the corpus, is rejected.

A compact ridge context adapter was then fitted on all 1,116 ASLLRP contextual
**training** segments. Feature mode and regularization were selected only by
leave-one-training-signer-out cross-validation across `BENJAMIN_JAMES_BAHAN`, `CORY`,
and `RACHEL`; the selected configuration is `mean_std_max_delta`, alpha 1000, with
29.8330% mean and 42.7861% worst-signer CV WER. Once frozen, the standalone adapter
scored JONATHAN once at 40/254 errors, 15.7480% WER. Its artifact SHA-256 is
`41d1bcccc84f2cadfa3b8ab0d944538de26fa60a4896aad424ece8bb4115a55d`; its full
selection/result report SHA-256 is
`773cac3bf832cbf4eac102e46239179195012f5f6aed525f2b82e8d4aaa7bb13`.

The deployment candidate is not the disconnected classifier. A loadable
`Stage2ContextAdapterV17` now applies the adapter independently to every 32-frame
window as a residual prior on the existing continuous CTC logits. It changes only
the already diagnosed `HOME` and `WHERE` locked classes; blank and the other 98 class
logits remain identical to v2. At residual weight 0.5, the combined model improves
the 254-token signer-held contextual development validation from 54/254 errors
(21.2598% WER, 79.5276% sequence accuracy) to 46/254 edits (18.1102% WER,
82.6772% sequence accuracy). It exactly preserves both existing phrase-domain
metrics: local validation remains 11/259 edits (4.2471% WER, 91.7526% sequence
accuracy) and the very small ASLLRP phrase validation remains 12/24 edits (50.0% WER,
16.6667% sequence accuracy). A cold reload reproduced every metric.

The combined artifact is
`artifacts/models/stage2_v17_context_adapted_ctc_v1/model.pth`, SHA-256
`24f846176bb836eaa744b2b93405d16fbb2173942c4b0f6e131ea6e58ec1bfe3`; validation
evidence is `artifacts/reports/stage2_v17_context_adapted_ctc_v1/validation.json`,
SHA-256 `d9ce9dd0f22175954584716a4dc86dccb42a9932a9e8d56465df2122f6770302`. The target
allowlist and smallest tested passing residual weight were chosen after inspecting
development-validation errors, so 18.11% is an achieved **development-validation**
result, not independent-test evidence. A new signer/capture set is still required
before making a generalization claim.

Ten focused Stage 2/model/data tests, Python compilation, JSON parsing, artifact
reload, and `git diff --check` pass. Citizen, SemLex, and local test splits, the
2M-Flores `devtest` split, and the already-consumed RIT external test were not
accessed.

## 2026-08-17 00:59 PST — 2M-Flores multimodal frozen features complete; training unblocked

All 155 long-sentence hand-RGB archives were encoded with MobileCLIP2-S0 in 20 short
sequential workers. The run covered 2,718 windows and 122,783 valid crop views with
zero failures. Peak MPS driver allocation was flat at 65,748,992 bytes across workers.
The hand audit matches all 155 expected archives with no missing, unexpected,
non-finite, non-normalized, mask, box, source, or schema errors. The encoding and audit
report SHA-256 values are respectively
`d6d66f0a6c8ac6ec5a1c498a31d1e7cbe54a30beeff78010cd18c208c2d5806b` and
`2253af42d8e519fe9f16ef8467944e480865d440463c957519445b5d1e8caca6`.

The frozen selected Stage 1 temporal cache then completed all 155 sentences in 19.7
seconds with zero failures. Its Stage 1 checkpoint remains pinned to SHA-256
`1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`;
peak MPS current/driver allocations were 48,528,640/165,232,640 bytes. The independent
frozen-feature audit covers all 155 archives, 2,718 windows, 2,810 full-order target
tokens, feature dimension 612, and all 448 auxiliary classes with no failures. Cache
and audit report SHA-256 values are
`8cff0eb14f71b274686c07d586eaeb2448ddebafc8b49d1dbf27979a2926d152` and
`41b7116adde6c178f84ff00287fc3552f12e08fabbf06674d0241eec60e16743`.
Dual-head CTC training is now unblocked. No evaluation split was accessed during
preprocessing.

## 2026-08-17 00:28 PST — all 155 2M-Flores landmark/RGB archives pass integrity

Long-video Apple Vision landmark and hand-RGB extraction is complete for all 155
selected 2M-Flores sentences with zero failures. Native detector RSS rose on long
4K-origin videos but system memory stayed green; the run was deliberately split into
short processes after row 84 so native allocations were returned between chunks.
Every save is atomic and the resumed workers reused compatible completed archives.

The independent fail-closed audit matches all 155 expected archives with no missing,
unexpected, or schema-incompatible file. It covers 2,718 nonoverlapping temporal
windows, 32 windows without a valid landmark result, and mean hand-view validity of
94.4719%; every archive has usable landmark and hand evidence. Total archive size is
1,395,267,704 bytes under long-video schema fingerprint `277d70d19c5cbb42`. The final
extraction report SHA-256 is
`2e25e187f7de4de2eb967a73532bfe7ebcb4fcb8c39d748cb2d2a9142162e754`; the audit
SHA-256 is `084f84201e3e4dac4d2ff1eec11d0b339ad0dde6a615c48a8908ffe026546a40`.

The dual-head Stage 2 implementation now warm-starts the exact selected v2 locked
100-sign head, extends its positional table from 8 to 40 windows, and adds a
training-only 448-class full-gloss auxiliary CTC head. Its focused shape, warm-start,
long-collation, and validation-priority tests pass. MobileCLIP2 hand encoding has
started in sequential eight-file workers at an 8% MPS memory cap. No evaluation split
was accessed.

## 2026-08-17 00:07 PST — full-gloss 2M-Flores manifests locked; long-video extraction proven

The success gate remains validation-only Stage 2 WER below 20%. The current selected
v2 checkpoint is not yet sufficient because its 254-row signer-held-out ASLLRP
contextual diagnostic is 21.2598% WER. The new 2M-Flores training design therefore
keeps the deployment 100-sign CTC head separate and uses a full-gloss auxiliary CTC
head to teach shared temporal structure without adding hundreds of rare distractor
classes to the deployment decoder.

All 155 acquired `dev` sentences now have a hash-verified training manifest. The
locked 100 labels remain indices 0--99. Five explicit annotation-category tokens and
343 recurring expanded lexical glosses produce a 448-token auxiliary vocabulary.
One-off lexical items map to an explicit unknown token rather than being deleted, so
the full ordered target timing is preserved; 579 token occurrences use that unknown
category. The selected videos require as many as 1,264 source frames, 40 nonoverlapping
32-frame windows, and 37 target tokens. The old eight-window cap must not truncate
this corpus.

The auxiliary vocabulary SHA-256 is
`227c462df9c5e2689645590b22a9e321838425fe9ed4db19902cbc6c9e7f2e44`.
The training manifest SHA-256 is
`eb197bd56c1b6601b75d69c2ff6dad35e0097320e29ef17f1976c794344831f9`.
The Apple Vision/RGB extractor now accepts an explicit maximum-source-frame contract
and fails closed rather than silently truncating declared longer videos. A real
1,264-frame-cap smoke run completed row 3 in 26.0 seconds with 13 windows, zero
failures, and Stage 2 schema fingerprint `277d70d19c5cbb42`. Six focused preparation
and extraction tests, JSON validation, and Python compilation pass. 2M-Flores
`devtest` and all project evaluation splits remain untouched.

## 2026-08-16 23:17 PST — compact 2M-Flores acquisition complete and independently verified

The selected 2M-Flores `dev` acquisition is complete: 155/155 manifest rows, 155 state
records, and 155 derived video files. An independent final pass recalculated every
derived SHA-256 and matched all 155 recorded hashes. The temporary source directory is
empty. In total, 20,327,996,229 source bytes (18.9319 GiB) were individually verified;
the aspect-preserving 720p30 H.264 corpus retains 1,355,423,637 bytes (1.2623 GiB).
The maximum recorded source-to-derived duration difference is 0.158333 seconds, below
the unchanged 0.200-second gate, and each clip passed full decode verification during
acquisition.

The final acquisition state SHA-256 is
`a7b2f363f877317c7439fd2b566945fff00df13dbd4ddeef7faa27c14c64449e`.
Completion evidence is recorded at
`artifacts/reports/stage2_v17_new_dataset_search/2m_flores_acquisition_complete.json`
(SHA-256 `ff84141641e6ee9fb9b49e79a4b4f1cd008542e963fbcdc087778739d2ee426b`).
The source plan now marks acquisition complete. The next gate is to lock the expanded
gloss vocabulary and convert all complete ordered gloss sequences into a v17 Stage 2
preprocessing manifest; the 100-label matches remain selection metadata and must not
replace the full transcripts. The 2M-Flores `devtest` split and all project evaluation
splits remain untouched.
Generated JSON validation, four focused audit/selection tests, Python compilation,
and `git diff --check` pass.

## 2026-08-16 17:02 PST — 2M-Flores acquisition resumed after correct duration-gate fix

The bounded 2M-Flores acquisition was not complete when checked: it had safely stopped
after 106/155 verified clips while processing row 682. The derived duration differed
from the source *container* duration by 0.203 seconds, three milliseconds beyond the
0.200-second threshold. Frame/timestamp inspection proved this was not truncation: the
source video stream is 15.235 seconds with 914 frames near 60 fps, while the derived
stream is 15.233333 seconds with 457 frames at 30 fps. The 15.436666-second source
container includes approximately 0.202 seconds of empty trailing container time.

The acquisition verifier now compares video-stream durations and records container
duration separately, falling back to container duration only if a stream duration is
unavailable. The 0.200-second accuracy threshold was not relaxed. Python compilation,
four focused audit/selection tests, and `git diff --check` pass. Row 682 then completed
successfully under the corrected check. At 17:02 PST, acquisition has resumed at
107/155 clips, 944,704,523 retained derived bytes, with 48 rows remaining. The live
authoritative count remains `data/local/2m_flores_asl_stage2_v17/acquisition_state.json`.
No evaluation split was accessed.

## 2026-08-16 14:13 PST — compact 2M-Flores selection locked; acquisition started safely

The full 156 GB 2M-Flores `dev` split will not be bulk-downloaded. File-level size and
LFS SHA-256 metadata were resolved from the dataset's pinned revision
`b450c1a427738e78f06362fc4619674f5d74f774`. A binary minimum-byte multicover
optimization selected 155 complete sentence videos, 18.9319 GiB of cumulative source
transfer, covering all 95 available locked labels with up to five examples per label.
For labels occurring fewer than five times, every available matching sentence is
retained. Complete gloss sequences remain unchanged for an expanded Stage 2 decoder.
The selection manifest is
`data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json`, SHA-256
`92ed45d9b52c3d34146233356563363d7a8517540a6fd9df9b32bd451f27da4c`.

The resumable acquisition worker downloads exactly one selected source at a time,
checks its pinned SHA-256, performs an aspect-preserving fit within 1280x720 at 30 fps
using the macOS hardware H.264 encoder, checks duration and dimensions, fully decodes
the result, records both hashes, and then removes only the verified temporary source.
It enforces at least 12 GiB disk headroom and never uses MPS. The first complete safety
run passed: row 3's 223.7 MiB source produced a verified 16.7 MiB derived clip, its
state was recorded under `data/local/2m_flores_asl_stage2_v17/acquisition_state.json`,
and the source temporary was removed. The remaining 154 clips are safe to resume
without repeating completed work.

At 14:23 PST, continuous acquisition has reached 10/155 verified clips and 118,947,445
derived bytes; the next selected clip is in its one-file temporary download. The live,
authoritative count is the `completed_rows` map in `acquisition_state.json`, which is
atomically updated after every verified clip. Acquisition continues in the bounded
worker; this timestamp is a progress snapshot rather than a completion claim.

The actual ASL-Homework-RGBD archive is Databrary volume 1249. The RIT supplemental
page does not hyperlink it; it only states that the dataset uses Databrary. Full access
requires an authorized Databrary account, while the RIT page publicly exposes only a
sample video/ELAN/depth triplet, prompts, annotation guide, and demographics. Four
focused selection/audit tests and Python compilation pass. No reserved dataset or
project test split was accessed.

## 2026-08-16 13:57 PST — new real-gloss Stage 2 source found and audited

The new continuous-ASL dataset search found one immediately actionable supervised
source: Meta's 2M-Flores-ASL. Its videos have human-created sentence glosses plus an
additional expert harmonization pass. A metadata-only audit read all 999 rows of the
official `dev` split and deliberately did not access `devtest`. Of those rows, 811
contain at least one locked Citizen-100 lexical label and collectively cover 95/100
labels; the missing labels are `GOODBYE`, `PLEASE`, `SORRY`, `SAD`, and `TOMORROW`.
The split contains 4,388 normalized gloss tokens, so it must be trained with an
expanded Stage 2 vocabulary. Deleting out-of-vocabulary tokens to manufacture
Citizen-100-only transcripts is forbidden because it would corrupt sequence order and
CTC timing. The dataset's signer field is only a local ID (`0` on 997 rows and `1` on
two), not a global identity, so this source cannot establish a new signer-disjoint
claim.

The complete row-level audit is
`data/local/dataset_metadata/2m_flores_asl/dev_locked100_audit.json`, SHA-256
`4dcb426bab947fdd455a364ede8c7039c10518316bd031f709712f2ee18d7130`.
The ranked evidence is recorded in
`artifacts/reports/stage2_v17_new_dataset_search/NEW_STAGE2_DATASET_SHORTLIST.md`
(SHA-256 `b4a65a67bc68a119936c599124fc7b8442321a323bf76f8a759c6d965294f937`)
and `shortlist.json` (SHA-256
`b4bad7843e2d371467142db1d1d42ca81376a00f9ad3e6a6a9832b7033f0fd97`).

ASL-Homework-RGBD is the second-ranked source: 935 continuous videos from 45 signers
(24 fluent and 21 learners) with ELAN gloss/nonmanual/error annotations, but the full
volume requires authorized Databrary access. Apple's newly reported ASL STEM Wiki
annotations rank third: nearly 500 professionally annotated videos and 8,655 sign
annotations, but no downloadable official annotation bundle was located. How2Sign,
the base ASL STEM Wiki/FLEURS-ASL releases, OpenASL, YouTube-SL-25, and the ASL portion
of SignNet-1M do not currently expose verified ordered ASL gloss targets suitable for
this Stage 2 CTC trainer. Their RGB may later support translation, representation
learning, or separately governed weak supervision, but they are not substitutes for
real gloss sequences.

Raw 2M-Flores is approximately 326 GB total and its `dev` videos are approximately
156 GB, so no bulk download was started during discovery. The next safe action is a
resumable one-file-at-a-time `dev` acquisition that records source hashes, performs an
aspect-preserving 720p30 transcode, verifies duration/decode and derived hashes, then
releases each temporary source MOV. The compressed videos must remain available for
RGB crops. `active/v17/stage2_data_sources_v17.json` is updated to version 2 with this
acquisition order. Two audit tests, JSON validation, and Python compilation pass; the
Python compilation, generated JSON validation, both focused audit tests, and
`git diff --check` pass. Citizen, SemLex, local, and 2M-Flores sealed test splits were
not accessed.

## 2026-08-16 13:45 PST — v2 contextual generalization confirmed; phase v3 rejected

The 254-row JONATHAN contextual validation-only diagnostic is complete. Apple
Vision/RGB extraction, four bounded MobileCLIP2 workers, the hand-archive audit, and
the frozen-feature cache all completed with zero failures. The hand audit covers 254
archives/windows and 12,086 valid RGB views; its SHA-256 is
`ee2ab9842a6ac21426a91b48e7288299d521a53891c8bc6442249ec1dc0ad4c6`.
Peak MPS driver allocation remained 65,765,376 bytes during RGB encoding and
103,579,648 bytes during frozen-feature caching.

On these 254 signer-held-out contextual signs, v1 scores 37.4016% WER and 67.3228%
exact accuracy; v2 improves to 21.2598% WER and 79.5276% exact accuracy. The v1 and v2
evaluation report SHA-256 values are respectively
`5729162c70f361b2f6bee5341c2dc05f98ed750dd11716cecd0a8b8f7f101354` and
`e08559bcd3ccaff931134414af202b81590a38dd243844932933a4f0b46f849a`.
This independently confirms within-ASLLRP signer generalization from the contextual
replay, but it does not override the unchanged 50.0% WER on 12 real ASLLRP phrases or
the previously consumed RIT failure.

A targeted v3 experiment tested arbitrary synthetic window phase, so isolated signs
crossed 32-frame boundaries instead of always aligning perfectly. Its plan SHA-256 is
`b685737db33f918163bc71ffb61f131a56294c267b0809499fc7848f9710624b`.
All three seeds completed, but the best candidate regressed to 54.1667% ASLLRP phrase
WER, 9.2664% local WER, and 31.7165% equal-domain mean WER. It is rejected; v2 remains
the selected Stage 2 checkpoint. The evidence now localizes the remaining limitation
to scarce genuine continuous phrase supervision: only 44 ASLLRP train phrases are
available. The next defensible improvement requires new fully labelled continuous
utterances or phrases, not more isolated-window augmentation.

Eleven focused Stage 2 tests pass, including fail-closed mixed-source composition and
window-phase packing. Python compilation, generated JSON validation, and
`git diff --check` pass. The RIT external set was not rerun, and Citizen, SemLex, and
local sealed test splits were not accessed.

## 2026-08-16 13:27 PST — contextual-replay Stage 2 candidate selected on validation only

The contextual-replay v2 CTC candidate is selected at
`artifacts/models/stage2_v17_unified_ctc_v2/best_model.pth` (SHA-256
`dd5f3e620acf5e911f9373a14eddba6e0c0610d422cfaa27dd9de1eacc509cc9`).
Its locked pool combines 1,475 Citizen official-training-only signs with 1,115
one-window ASLLRP contextual train-only signs: 2,590 items total, all 100 classes from
Citizen and contextual coverage for 53 classes. One ASLLRP item requiring two windows
is excluded from isolated composition rather than truncated. Pool SHA-256 is
`a45453c70b92001070a24ebfc4aff8f58f6a04f20acabb7c07ccb53668adf462`.
The deterministic 12,000-sequence within-domain composition plan contains 6,000
Citizen and 6,000 ASLLRP sequences; its SHA-256 is
`1caab14ced170926213bd5a8889470fca804b8b2ca17f67e9407abe68474ad57`.

Three scratch CTC-head seeds used the unchanged validation-only selection rule. Seed
1703 epoch 20 wins with 4.2471% local WER and 91.7526% local exact accuracy across 97
clips, plus 50.0% signer-held-out ASLLRP WER and 16.6667% exact across the 12 JONATHAN
phrase clips. Equal-domain mean WER is 27.1236%, improving v1's 28.6680%; worst-domain
WER remains 50.0%. An independent reload reproduced every metric exactly. Therefore
v2 supersedes v1 under the predeclared worst-WER/mean-WER selection rule, but the
ASLLRP domain gap is **not** resolved and the model remains unready for mobile
promotion. The consumed RIT external set was not rerun.

Because 12 phrase clips are too small to diagnose contextual generalization reliably,
254 usable JONATHAN segmented signs spanning 34 classes were locked as a larger
validation-only diagnostic in
`active/v17/stage2_asllrp_segmented_validation_manifest_v17.json` (SHA-256
`26e14b25900d0f5f4eed00eac86d80d9eb15f21c4ae0f303ca22253ab39149c4`).
Thirty JONATHAN clips shorter than four frames are rejected, all other signers are
excluded, and all RIT rows remain excluded. Apple Vision/RGB extraction completed all
254 with zero failures; the later 13:45 entry records completed encoding and evaluation. These rows must
remain validation-only and must never enter replay training. Citizen, SemLex, and
local sealed test splits were not accessed.

## 2026-08-16 13:00 PST — train-only contextual replay features complete

The locked 1,116-row ASLLRP contextual replay set has completed the full Stage 2
preprocessing path. All 1,116 bounded MobileCLIP2 archives were written in 18 short
workers at an 8% MPS memory cap with zero failures; peak MPS driver allocation was
65,765,376 bytes. Its independent audit passes all 1,116 archives, 1,117 windows, and
53,137 valid RGB views under schema fingerprint `fd3110e2db69da2e`. The frozen
selected Stage 1 temporal cache then wrote all 1,116 feature archives with zero
failures in 85.3 seconds, at feature dimension 612 and peak MPS driver allocation
104,677,376 bytes. The frozen Stage 1 checkpoint remains pinned to SHA-256
`1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`.

These rows are training-only: BENJAMIN_JAMES_BAHAN, CORY, and RACHEL are included;
JONATHAN validation and all consumed RIT external rows remain excluded. The next
candidate may be selected only on the unchanged local+JONATHAN validation gate. RIT
must not be rerun, and no Citizen, SemLex, or local sealed test split was accessed.

## 2026-08-16 12:33 PST — first v17 Stage 2 model selected; external RIT gate fails

The first genuine v17 Stage 2 CTC model is selected at
`artifacts/models/stage2_v17_unified_ctc_v1/best_model.pth` (SHA-256
`eafeb9290fccd9ed76db03e5dd922c47d753ce7a2198716904286bb993031488`).
It uses the frozen selected Stage 1 landmark, full-face/lip, hand-RGB/MobileCLIP2, and
fusion encoders. Their per-frame outputs are cached as 612-dimensional evidence; the
trainable 3,427,373-parameter CTC head emits eight tokens per 32-frame window. Training
used 434 real train phrases plus 10,000 synthetic compositions from 1,475 Citizen
official-training-only isolated features covering all 100 classes. The synthetic pool
SHA-256 is `25e11fa3aa3f61f26680a33d3074f54112b7f62bce4322883cd0845b845f66f5`
and its plan SHA-256 is
`f595b3f1d1451d0de40d26fa671c9817f911434c8bdb49bab1289f790965a3c6`.

Three seeds were selected by worst-domain WER, then equal-domain mean WER, then exact
sequence accuracy on the unchanged 97 local and 12 signer-held-out JONATHAN validation
clips. Seed 1701 epoch 37 won. Local validation is 7.3359% WER and 83.5052% exact
sequence accuracy. Signer-held-out ASLLRP validation is 50.0% WER and 25.0% exact.
Equal-domain mean WER is 28.6680% and mean exact accuracy is 54.2526%. An independent
checkpoint reload reproduced these metrics exactly. The full result JSON SHA-256 is
`2d96577d4d79d9ee2d155cfae380a8398404b370bcf123497ae6c9cddf4983f4`.

After model selection, the 14 permanently reserved RIT spans were consumed exactly
once as external evidence and were never used for training or checkpoint selection.
The result is a hard failure: 96.4286% WER and 0/14 exact sequences. The report SHA-256
is `efcc87c87c5782b41c4af88450e8a5ee9caaf39725ea49db7fd2a70edf5862e0`.
Therefore Stage 2 is **not** ready for mobile promotion, and the strong familiar-local
score must not be presented as general phrase recognition. Those RIT rows are now
consumed and must not be used to tune or select another checkpoint.

The independent failure identifies a train-domain gap. A new train-only contextual
replay manifest has therefore been locked from the already downloaded ASLLRP segmented
signs: 1,116 clips across 53 classes from BENJAMIN_JAMES_BAHAN, CORY, and RACHEL.
JONATHAN remains held out, all 236 RIT segments remain excluded, and 83 clips shorter
than four frames are rejected. The manifest SHA-256 is
`3efa2ead3e28428fd631d54de4e946e00569a34109573e462553bca0d6f72d05`.
All 1,116 landmark/RGB archives completed with zero extraction failures; the later
13:00 entry records completion of their bounded embeddings and frozen features. This replay source may improve the
unchanged local+JONATHAN validation gate, but RIT will not be rerun or used to justify
the next selection. Citizen, SemLex, and local sealed test splits were not accessed.

## 2026-08-16 11:13 PST — Stage 2 temporal contract locked before extraction

Stage 2 work is active again; Stage 1 data acquisition is explicitly deferred. The
selected frozen base remains
`artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth`, whose
landmark branch exposes 32 temporal 256-dimensional tokens and whose hand-RGB branch
encodes 16 MobileCLIP2 frames before attention pooling. A whole phrase must **not** be
passed through the isolated v17 extractor once, because its fixed 32-frame resampling
would collapse phrase duration and sign boundaries.

The locked Stage 2 preprocessing design is therefore non-overlapping 32-source-frame
windows after one orientation correction per source video. Each window retains the
unchanged `(32, 61, 5)` landmark contract and 16 three-view hand-RGB/MobileCLIP2
samples. CTC will consume multiple temporal tokens per window above the frozen Stage 1
encoders, rather than one pooled isolated-sign prediction per phrase. This preserves
full face geometry and uses the hand-RGB evidence that fixes landmark errors. The four
lip points remain present for ASLLRP; only those four nodes are zeroed for local phrase
rows whose lip supervision is unavailable, preserving the other 11 face nodes. The
pipeline works for portrait or landscape sources through the existing v17 orientation
contract and avoids repeated Vision work on overlapping windows.

The local phrase corpus remains limited to the six strictly in-vocabulary templates;
`FOOD -> EAT` is still unapproved and phrases containing `LATE`, `TEACHER`, or `MEET`
remain excluded. ASLLRP exact target-only spans may train only from the existing
`train_candidate` partition; all RIT spans remain external evaluation. NCSLGR
utterances containing unlabelled out-of-vocabulary signs will not be falsely treated
as direct CTC targets; only exact target-only spans may enter training. Citizen,
SemLex, and local sealed test splits were not accessed.

The preprocessing rows are now locked in
`active/v17/stage2_training_manifest_v17.json` (SHA-256
`2fe324bc5d4ec9b97f1ff4aa437c0af60117088739679dead78bcf6d0cafc48c`). Local
capture batches contain 20 recordings; every fifth adequate-length recording is
validation, as explicitly allowed by the owner despite signer overlap. This yields
390 local train and 97 local validation clips. Thirty-three additional vocabulary-
valid local clips are recorded but rejected as too short to contain eight source
frames per target sign. ASLLRP contributes 44 train spans from CORY, RACHEL, and
BENJAMIN_JAMES_BAHAN and 12 signer-held-out validation spans from JONATHAN. The 14
RIT spans remain external-evaluation reserved and will not be used for training or
checkpoint selection. There is zero source-video hash or parent-utterance overlap
between active roles. The audit is
`artifacts/reports/stage2_v17_training_manifest/audit.json`; four focused fail-closed
manifest tests and Python compilation pass.

The bounded extractor in `scripts/extract_stage2_multimodal_v17.py` has completed all
543 active train/validation rows in one serialized Apple Vision process: 539 newly
written, four smoke archives resumed, and zero failures in 601.3 seconds. It produced
1,587 temporal windows totaling 639,690,288 archive bytes under schema fingerprint
`f2b206169c243a1d`. The full extraction report SHA-256 is
`a611ffba7b15992fc28b2ab0fa840dfed95830b5e8f00cfa603471103c270cf8`.
The independent archive audit checked every feature shape, finite value, target,
source/manifest hash, window range, JPEG offset, hand-valid mask, and row-specific lip
policy. All 543/543 archives pass with no missing or unexpected files. Eighty-one of
the 1,587 individual windows contain no usable landmark hand detection and remain
explicit blank windows; every phrase has at least one valid landmark window and one
valid hand crop. Mean hand-view validity is 0.661879. The audit SHA-256 is
`d35c10b109823c1d0f72c91250408c7c3d32dd9febbafcda09dc096f22e40903`.

## 2026-08-15 09:18 PST — current ASLLRP segmented-sign metadata acquired and exact clips verified

The owner obtained the current metadata for signs segmented from continuous signing,
not the separate citation-form Stage 1 datasets. The authoritative inputs are
`asllrp_sentence_signs_2025_06_28.csv` (SHA-256
`9c1641b6b95a6e6c2223c34eb101ed2e83bb1da824e18c7a21791c59b1593e86`) and
`rit_sentence_signs_2025_11_01.csv` (SHA-256
`27af833206b85ef02dbe6008173c0b1938504ba05943954f76269380c07fe6fb`). Older
2023/2024 ASLLRP snapshots remain untouched in Downloads but are superseded and were
not combined. The ASLLRP CSV has 17,519 valid rows and three malformed quoted-gesture
rows that are rejected fail-closed; RIT has 3,056 valid rows and no rejected rows.

`scripts/prepare_asllrp_continuous_citizen100_v17.py` applies the official ASL-LEX
`SignBankAnnotationID` contract rather than English-label normalization. It admits
1,719 unique segmented signs across 54/100 classes: 1,483 ASLLRP signs from four
participants across 53 classes are training candidates, while all 236 RIT signs from
two participants across 30 classes are permanently reserved for new external Stage 1
evaluation. The short clips were downloaded and independently rehashed to
`data/local/asllrp_segmented_citizen100_v17`; all 1,719 decode, all SHA-256 values are
unique, and there are zero missing, size, hash, or acquisition failures. They total
132,149,094 bytes and 476.131753 seconds. The acquisition manifest SHA-256 is
`44f50fc61c981fd774dc7eedc8c226747ca4221738c952449f44e13d7e73d622`.

These metadata expose 1,237 parent utterances containing at least one exact target
sign, but only one utterance (`WATER COLD`) has at least two tokens and no non-gesture
gloss outside the locked 100-class vocabulary. A stricter frame-level scan nevertheless
finds 70 contiguous target-only multi-sign runs across 68 parents: 56 ASLLRP training
candidates and 14 RIT external-evaluation spans, totaling 144 target tokens across 53
unique phrases. Only those 68 parent videos were downloaded; the spans were cropped at
manual first/last sign bounds plus five context frames. All 70 crops decode, have unique
SHA-256 values, and total 73.842842 seconds with zero integrity failures. The span
manifest SHA-256 is
`9e38d4ab1d289f41f402e3c9b3b5a355167c8e2783fd1ba8c91538d6932c9a4e`.

Therefore the segmented clips are high-value contextual Stage 1 data and the 70 exact
runs are valid short real-phrase Stage 2 candidates. The other parent utterances are
not falsely labeled as direct 100-class CTC training data, and no master videos were
downloaded. Stage 2 must lock either an expanded output vocabulary or a principled
partial-label objective before acquiring the other target-bearing parents. The
separate citation-form metadata is still not acquired.

The exhaustive audit is
`artifacts/reports/asllrp_continuous_citizen100_v17/audit.json` (SHA-256
`ef32b0f367f25468b88b45f2d78c21c10d2778c5d61d7c340c4740a648907147`). Four new
fail-closed parser/variant tests pass, Python compilation passes, JSON validation and
`git diff --check` pass. Citizen, SemLex, and local sealed test splits were not
accessed. The pinned Stage 2 source plan is now SHA-256
`12f13c910393a6506f2df873d7dec9c5465f3bcbdc65e422b3f038d2638d5e42`.

## 2026-08-14 15:30 PST — ASLLRP Sign Bank approved as a candidate Stage 1 source

The public ASLLRP Sign Bank catalog and its official download documentation were
checked against the frozen 100-class v17 manifest. It is a valid candidate source for
Stage 1: it contains both citation-form isolated recordings and individual signs whose
linguistic start/end frames were manually segmented from continuous utterances. The
download metadata includes main-entry, entry/variant, occurrence label, frame bounds,
handshapes, source collection, filenames, and sign type. Citation-form data and
segmented-continuous signs have separate authenticated download pages.

A preliminary public-catalog audit finds literal entry/variant-label candidates for
84/100 target labels totaling 3,158 displayed occurrences. Punctuation-normalized
matching expands the upper bound to 86/100 labels and 3,386 occurrences, but this is
not an approved training count: forms such as `#NO`, `"WHAT"`, agreement/index signs,
compounds, and lexical variants cannot be collapsed by string normalization. Counts
may also fall after unavailable DawnSignPress rows, repeated views, source duplicates,
and nonmatching target variants are excluded from the authenticated CSV downloads.

Once account access is approved, the correct protocol is to download the citation-form
and segmented-sign CSV/video bundles; hash and group all views by occurrence, source
utterance, participant, and collection; review every ASLLRP entry/variant against the
pinned Citizen/ASL-LEX target; and retain only front-view single-sign videos. Composite
videos are not training samples. A signer- or collection-held-out subset must first be
used as new cross-domain evaluation evidence for the already selected Stage 1 model;
only the remaining partition may be added to training. Any segmented sign whose parent
utterance is later used by Stage 2 must remain in the same split to prevent cross-stage
source leakage. No authenticated video, sealed split, or existing test set was accessed
for this preliminary audit.

## 2026-08-14 14:12 PST — Stage 2 source audit complete; first real external subset acquired

Stage 2 data gathering is now pinned in
`active/v17/stage2_data_sources_v17.json` (SHA-256
`b5f2d285d6f2aeb6bcc730426f531c7ba7676ec90739098587665b00c67d3b47`). The local
`data/raw_videos/PHRASES` corpus contains 780/780 valid, SHA-unique 640x480 clips
across nine fixed phrases and totals 2,226.700004 seconds (0.6185 hours). It has no
usable signer metadata. Under the frozen 100-class vocabulary, 520 clips are strictly
eligible after the safe `ME -> I` normalization. A further 60 clips require an
explicit semantic decision before treating `FOOD` as `EAT`; this mapping is not
silently approved. The remaining 200 contain out-of-vocabulary `LATE`, `TEACHER`, or
`MEET`. Existing phrase and 15,000-sequence synthetic arrays use legacy 61-node
v16-era schemas and must not be used by v17; synthetic sequences must be regenerated
from hash-pinned, train-only v17 isolated archives.

The public NCSLGR/SignStream static subset has been acquired to
`data/local/ncslgr_continuous_v17_source`: 166/166 compressed frontal videos and
166/166 frame-aligned SignStream annotation files, with all sizes and SHA-256 hashes
verified. The videos total 34,631,570 bytes and 8.8591 minutes, are 324x312, and cover
two normalized participant IDs. Of the 166 utterances, 132 contain at least one target
gloss, totaling 198 target-gloss occurrences across 17 of the 100 classes. This is
real supervised continuous-sign data but is low-resolution, narrow-coverage
supplemental evidence, not the primary Stage 2 corpus. Its manifest SHA-256 is
`c03ff8a5b13c7fa7da9b7ac211ab15b66719768a152c4b0d66b96529b6660744`.

The modern ASLLRP DAI public catalog was queried against all 100 frozen labels across
all 47 BU collections, four RIT sources, and all exposed participant records. Exact
normalized coverage is 76/100 labels with 2,313 occurrences and zero query failures.
This is the best identified broad, exact-gloss supervised source, but bulk video/XML
acquisition requires an authenticated ASLLRP DAI account; access control will not be
bypassed. Official manually realigned How2Sign train/validation metadata was also
acquired and hash-pinned: 32,906 English-aligned sentences. Its official release does
not supply the ordered gloss targets required by the current CTC design, so English
word hits are only weak-label evidence and the approximately 33 GB RGB download is
deferred pending a separately locked self-supervised or translation objective.

The exhaustive local/source report is
`artifacts/reports/stage2_v17_data_audit/audit.json` (SHA-256
`09c1dd1d191584184ff877b5862969ae4ee6b8ce96dde7210ebf1711b910992c`). Acquisition
and audit tooling lives in `scripts/acquire_ncslgr_stage2_v17.py` and
`scripts/audit_stage2_phrase_sources_v17.py`. Four focused parser/fail-closed
vocabulary tests pass; both scripts compile; all new JSON files validate. No Citizen,
SemLex, or local sealed test split was accessed. The next safe actions are full-length
v17 extraction and near-duplicate grouping of the local phrases, a v17 extraction
quality gate on NCSLGR, authenticated modern ASLLRP acquisition, and regeneration of
v17 synthetic sequences.

## 2026-08-14 13:40 PST — Stage 1 evidence accepted for Stage 2 development

The owner accepts the existing signer coverage and official validation evidence as
sufficient to begin Stage 2 development. The Citizen provenance contains 32 training
signers and five validation signers with zero identity overlap; the selected unified
classifier scores 364/378 = 96.30% on those unseen validation signers. SemLex contains
32 dataset-specific signer IDs and the local corpus adds about seven, so the training
sources collectively expose the model to roughly 70 dataset-specific signer IDs
(cross-dataset identities cannot be de-duplicated reliably). SemLex validation is not
signer-disjoint from SemLex train and local validation permits familiar signers, so
neither is being misrepresented as an independent signer gate.

Stage 2 may now proceed with the selected unified Stage 1 checkpoint frozen and
version-pinned. The already-consumed official Citizen test split will not be rerun:
its one-time 87.57% v17 landmark result remains historical evidence, and reusing it
during development would turn it into another validation set. The current unseen-
signer Citizen validation result is the promotion basis. Stage 2 must preserve
sequence-level train/validation separation and must not claim continuous-sign or
translation quality from isolated-clip accuracy alone. Citizen test, SemLex test, and
local test remain sealed for further model selection.

## 2026-08-14 13:35 PST — single unified classifier selected, exported, and simulator-gated

The predeclared three-seed unified-student experiment is complete. Seed 5101 at
epoch 15 wins the locked equal-domain validation criterion with a 94.1517% mean:
364/378 = 96.30% Citizen, 871/978 = 89.06% SemLex, and 2,812/2,896 = 97.10%
familiar-signer local validation. Top-5 is 99.21%, 96.73%, and 99.65%, respectively.
The checkpoint SHA-256 is
`1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`.
Seed 1701 reached a 94.1406% mean and seed 3407 reached 94.0732%, so neither is
promoted. The unified checkpoint is the selected **single-classifier compromise**:
one landmark encoder, one MobileCLIP2 hand-temporal encoder, and one trained fusion
head in one checkpoint. It improves the previous local 75/25 landmark/hand fusion by
11 clips while satisfying the 361/378 Citizen floor. Because its Citizen/SemLex
scores remain below the independent four-stream teacher's 370/378 and 882/978, that
teacher remains the accuracy-research reference; it has not been falsely replaced by
the smaller deployment candidate. The exact promoted landmark branch remains frozen,
so its previously measured 356/378 eight-angle floor is preserved at branch level.

The selected deployment artifact is the accuracy-first float32 Core ML package
`artifacts/coreml/Stage1UnifiedMultimodalV17FP32.mlpackage`. It contains 12,131,824
parameters, is 48,959,812 bytes (46.69 MiB), and has package-tree SHA-256
`96c35c739d55b911eae420887436de72ae8f9cb7524dc3bbb25d0007b0e6ee99`.
Core ML parity is exact at top-1 on all 378 Citizen validation rows with zero
mismatches and maximum logit difference `1.9073486328125e-06`. The smaller FP16
export was rejected because it changed one of 378 predictions. The package has four
runtime tensors—landmarks, hand embeddings, hand validity, and hand boxes. Apple
Vision landmark extraction and MobileCLIP2 RGB-hand embedding remain upstream
preprocessors; this is one classifier package, not a misleading raw-video-to-label
graph. The unsigned Release build succeeds for both the iOS simulator and generic
iPhoneOS 26.2 targets.

The automated iPhone 13 simulator gate passes all acceptance checks in suite
`orientation-v17-ios26-3-1-20260814T052728Z`. Exactly eight reports were produced for
0, 17, 37, 73, 90, 123, 180, and 270 degrees, each with 200 timed inferences; every
expanded-canvas video extracted successfully and predicted `HELLO`. Quadrant
corrections are exactly 0->0, 90->270, 180->180, and 270->90 degrees, and intermediate
residual roll never exceeds 37 degrees. The dedicated simulator is an iPhone 13
(`iPhone14,5`) on iOS 26.3.1. Unified-classifier median latency averages 14.258 ms and
the maximum per-condition p90 is 21.216 ms. These numbers are expressly simulator
evidence: `hardwarePerformanceClaim=false`, `thermalsInterpretable=false`, and
`endToEndPipeline=false`. The simulator runtime lacks the Apple Vision pose Espresso
weights, so the same host-macOS v17 Apple Vision and MobileCLIP2 preprocessors supplied
the pinned tensors. Physical-iPhone latency, ANE behavior, resident memory, thermals,
and interactive on-phone hand embedding extraction remain deferred.

Final verification is green: 122/122 affected tests pass, the 1,100-row portrait
capture-pack setup audit passes with zero errors, 19 new JSON evidence files parse,
the eight simulator report contracts were independently rechecked, Python compilation
passes, and `git diff --check` passes. The latest simulator result SHA-256 is
`aab9019d794d0ed05ff92de5744e0b2da17b0a68e14dbdf0ed352c6b574c44e2`.
The frozen candidate manifest and capture-pack provenance were refreshed only for
current source hashes; no capture review state or model selection was changed.
Citizen test, SemLex test, and local test were never accessed.

## 2026-08-14 13:18 PST — unified-student selection protocol locked

The next experiment is now predeclared as a single unified multimodal Stage-1
classifier, not another independently weighted four-checkpoint ensemble. Its runtime
inputs are the unchanged `(32,61,5)` Apple Vision tensor plus the existing
`(16,3,512)` MobileCLIP2 hand-crop embeddings, validity mask, and normalized boxes.
The model contains the exact promoted local-replay landmark encoder, the exact
retained hand encoder, and one learned fusion head in one checkpoint/Core ML graph.
The two encoders start frozen so the local corpus cannot erase mouth geometry or the
hand expert's clean-domain knowledge. Citizen train rows may additionally distill the
fixed 0.30/0.15/0.35/0.20 four-stream teacher; SemLex/local rows use only their
approved hard labels because no mouth/lower-face teacher target will be fabricated for
silent local clips. Local landmark inputs keep the established four-lip-point mask.

Fusion candidates are restricted in advance to three seeds (1701, 3407, 5101) of the
same zero-residual gated head and the same 34/33/33 source-balanced replay. Selection
uses only Citizen validation, SemLex validation diagnostics, and the approved
familiar-signer local validation set. A candidate must keep at least 361/378 Citizen
correct and the existing 356/378 eight-angle landmark robustness floor. Among eligible
candidates, the equal-domain mean of Citizen, SemLex, and local top-1 selects the
winner; Citizen correct, then SemLex correct, then local correct are fixed tie-breakers.
The existing four-stream teacher remains the accuracy reference rather than being
silently relabeled as a single model. The selected unified model will be exported to
one FP16 Core ML classifier package and exercised in the existing dedicated iPhone 13
simulator harness. Simulator timing remains Mac-host simulator evidence only; hand-RGB
embedding generation and Apple Vision extraction remain preprocessing, and no
physical-iPhone/ANE/thermal claim is permitted. Citizen test, SemLex test, and local
test remain sealed.

## 2026-08-14 13:06 PST — local multimodal challenger completed; retain general teacher

All finalized local hand features are complete and exhaustively audited. The bounded
visual-only MobileCLIP2 path produced exactly 13,381/13,381 train and 2,896/2,896
validation embeddings across all 94 admitted classes. The 27-worker bulk-train run
took 6,663.8 seconds with image batch 32, one archive and one worker at a time, a 12%
MPS process ceiling, zero failures, and zero swap. Peak Metal-driver allocation was
1,432,961,024 bytes and maximum RSS was 1,623,146,496 bytes. Run report SHA-256 is
`48bcae48fd187652b5d91f1726d43a8b88968e297dbde929153c4813a61d3213`.
The fail-closed auditor now rejects both missing and unexpected embedding inventory
members in addition to checking every archive's shape, finiteness, unit norm,
explicit missing-view zeros, source item, label, manifest hash, split, eligibility,
and sealed-test provenance. Train and validation audit SHA-256 values are
`db978e2a7b36761af46d22a874b0dbb2a26e898afb48bcd63b42ff2c7b861063`
and `f6334cc448cede0285857b6b95ca0376b0ceec136fd65a03a5ec50a173af2f10`;
both report zero errors and exact crop/embedding valid-view agreement.

The hand replay protocol was locked before optimization at SHA-256
`204281e97d6cb195a7bdcd84ffe46e5428fdbe699d2697b521a2613215e90948`:
exact source checkpoint, seed 1701, 34/33/33 Citizen/SemLex/local replay, 100-epoch
maximum, 20-epoch patience, 5e-5 peak learning rate, four warmup epochs, batch 64,
streamed `--no-cache` data, and Citizen-primary checkpoint selection with local top-1
allowed only on an exact Citizen tie. The exact smoke reproduced the source at
305/378 = 80.69% Citizen and 1,684/2,896 = 58.15% local before optimization. The
definitive run stopped after 20 stale epochs. Its live EMA reached 2,839/2,896 =
98.03% local at epoch 20, but only 284/378 = 75.13% Citizen; the best post-update
Citizen epoch was 299/378 = 79.10%. Therefore every adapted hand state was correctly
rejected. The selected output checkpoint, SHA-256
`7e920d0916842afe4fae284a383aad6c68eced940505a9d66f559f2b983b465e`,
has all 108 model tensors bit-exact to source checkpoint SHA-256
`ec16d1b14a2346fecd993d92b3c92b4965cd1204132f64514d86c109570e6d84`;
its differing file hash comes only from the new provenance/checkpoint envelope. Run
result SHA-256 is
`1ca647a9841ce18bcbed58ffb45bc48266413bee840be9015c31164f6ab9c23d`.

The final fixed-weight evaluations use the promoted mouth-safe local landmark branch,
the retained exact hand branch, and byte-unchanged frozen mouth/lower-face branches.
Per-sample z-score fusion at fixed 0.30 landmark / 0.15 mouth / 0.35 lower face /
0.20 hand reaches 366/378 = 96.83% Citizen and 887/978 = 90.70% SemLex. This is four
Citizen clips below the retained teacher's 370/378 despite improving SemLex over
882/978, so it is **not** the new general teacher. Report SHA-256 values are
`f0b9fe24652b380dfbabdfc097d41b90cabf41212114f0eb931a39df4a283ff4`
and `3d7e011af89d0e4e097c9fa27acb07bb9ba0fa04d518dc1ed762d9f0175c1654`.
On local familiar-signer validation, the separately frozen 75/25 landmark/hand fusion
reaches 2,801/2,896 = 96.72%, 99.65% top-5, and 91.00% all-100 macro F1. It repairs
28 landmark errors while breaking 17, a net gain of 11 clips over the 2,790/2,896
adapted-landmark result. Its report SHA-256 is
`89ddaf26970e860d0b2ce79335fc7ff153d5fab5120ba55f4bae5c211a78c6d6`.
The adapted landmark still clears the existing eight-angle gate with a 356/378
worst-angle floor; the retained hand/face experts were not modified by orientation
training. The existing four-stream teacher at 370/378 Citizen and 882/978 SemLex
therefore remains the accuracy research default, while the completed local challenger
is retained as domain-adaptation evidence and a future independent-capture candidate.

The full affected suite is green: 100/100 focused tests pass, including 17/17 in the
real Apple Vision host environment; both exhaustive hand audits and all generated
JSON reports validate; `git diff --check` passes. The first sandbox-only run produced
three Vision setup errors, then the required host rerun passed all three real Vision
tests; no assertion failed. Citizen test, SemLex test, and local test were never
accessed. Simulator/physical-iPhone benchmarking remains explicitly deferred per the
owner's instruction.

## 2026-08-14 09:33 PST — memory-bounded visual-only encoder proven; local hand validation complete

The local MobileCLIP2 memory regression is fixed and measured. The exact visual-only
loader now instantiates only MobileCLIP2-S0's 11,406,976-parameter FastViT image tower
on the meta device and loads the 784 official `visual.*` tensors from checkpoint
SHA-256 `ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95`.
One CPU comparison against a pre-restart MPS archive differed by at most one float16
unit (`6.103515625e-05`), consistent with CPU/Metal kernel rounding; valid masks and
boxes were exact. The bounded MPS supervisor keeps one archive and one worker live,
enforces a per-process memory fraction, records PyTorch/Metal telemetry, and exits the
Metal context after deterministic file shards. The full validation continuation ran
52 sequential 32-clip workers at image batch 16 and an 8% cap with zero failures and
zero swap. Peak Metal-driver allocation was 1,168,441,344 bytes and supervisor/child
maximum RSS was 1,108,426,752 bytes; worker peaks remained flat rather than accumulating.
Its report SHA-256 is
`99573b9d300162318a44c5dbf4ef8e49afab436819e1bc1fe8d40c0b8784119e`.

Exactly 2,896/2,896 finalized local-validation hand embedding archives now exist.
The exhaustive crop+embedding audit passes with zero errors across all 94 classes:
shape, finiteness, explicit zero missing views, unit normalization, manifest hash,
source, split, eligibility, and sealed-test provenance all match. Crop and embedding
valid-view fractions agree exactly at 0.7413458218232044. Audit report SHA-256 is
`f6334cc448cede0285857b6b95ca0376b0ceec136fd65a03a5ec50a173af2f10`.
The untouched selected hand checkpoint (SHA-256
`ec16d1b14a2346fecd993d92b3c92b4965cd1204132f64514d86c109570e6d84`)
scores 1,684/2,896 = 58.15% local top-1, 77.45% top-5, and 51.80% macro F1; this is
the frozen pre-adaptation hand baseline. Metrics SHA-256 is
`557feea178e6c2d89afe59a37b9fff7ba690a8f88b3a9120e6e0865f2c5061e0`.

A batch-32/12%-cap train smoke encoded 128 clips in 61.5 seconds at a 1.225 GiB Metal
peak and zero swap. Batch 48 was rejected despite remaining stable: it saved only 7%
wall time while increasing the Metal peak to 2.123 GiB. Bulk local-train encoding is
therefore pinned to image batch 32, one archive/worker at a time, a 12% hard cap, and
process recycling. Hand replay is additionally fail-closed unless `--no-cache` is
present. The mouth/lower-face experts remain frozen, and Citizen test, SemLex test,
and local test remain untouched.

## 2026-08-14 00:46 PST — local MPS MobileCLIP encoding prohibited after host restart

The long local MobileCLIP2 hand-embedding path is stopped. Repeated MPS encoder
processes allocated Apple-silicon unified memory through the full OpenCLIP image and
unused text towers; abnormal process termination did not reliably return the MPS
driver allocation, and the host subsequently restarted under memory pressure. This
is an infrastructure failure, not model evidence. No local MPS MobileCLIP encoding or
hand training may resume. The restart left 1,117 atomic, schema-checked local
validation embedding archives and no known partial archive because writes use a
temporary file followed by an atomic rename.

The encoder loader is being replaced with an exact visual-only loader: it constructs
the 11,406,976-parameter FastViT visual tower on PyTorch's meta device and materializes
only the 784 official `visual.*` tensors from the hash-pinned safetensors checkpoint.
It therefore never instantiates the unused text transformer or duplicate random
weights. Exact-output equivalence must be proven with a small CPU comparison before
the loader is accepted. Remaining bulk encoding and hand replay will use a bounded
non-MPS path (Kaggle GPU preferred; local CPU only for small verification), and hand
training will stream archives with `--no-cache`. Landmark replay evidence remains
valid and complete; mouth/lower-face experts remain frozen. Citizen test, SemLex test,
and local test remain untouched.

## 2026-08-14 00:20 PST — mouth-safe landmark replay clears all promotion gates

The definitive exact-checkpoint landmark replay completed on laptop MPS after 22
epochs and the predeclared 20-stale-epoch stop. It used 1,475 Citizen train, 1,388
approved SemLex train, and 13,381 finalized local train archives with exact
34/33/33 class/source-balanced replay. Only the four local lip landmarks were
masked; all other local face anchors and all Citizen/SemLex lips remained visible.
The selected source checkpoint was restored exactly at 362/378 Citizen and
1,765/2,896 mouth-masked local validation before optimization.

The strict Citizen-best checkpoint is epoch 2: 363/378 = 96.03% Citizen and
2,215/2,896 = 76.48% local. The stronger promotion-gate checkpoint is epoch 21:
361/378 = 95.50% Citizen, 2,790/2,896 = 96.34% local, and 90.55% local macro F1.
On the frozen SemLex validation diagnostic, the Citizen-best checkpoint reaches
835/978 = 85.38%, while the promotion-gate checkpoint reaches 860/978 = 87.93%,
exceeding the old selected landmark's 839/978. The promotion checkpoint also improves
the eight-angle landmark stress floor from 348/378 to 356/378: per-angle correct
counts at 0/17/37/73/90/123/180/270 degrees are
361/362/361/357/357/356/356/358. It therefore clears every predeclared landmark gate
without touching Citizen test, SemLex test, or local test and is the landmark branch
candidate for the final multimodal evaluation.

Exact checkpoint SHA-256 is
`12a74a18d71712abf525350e8120e26694d24a62234b4c268639220a37da47a1`;
the conservative Citizen-best SHA-256 is
`d7969867c70335da1455d4b5964ef12e2160eb043f732f805c065d711675544b`.
SemLex reports are under
`artifacts/reports/semlex_citizen100_val_audit/local_deep_clean_replay_ft_*`;
orientation reports are under
`artifacts/reports/stage1_v17_orientation_robustness/local_deep_clean_replay_ft_*`.
The hand-RGB MobileCLIP2 encoding phase is now active; mouth/lower-face experts remain
frozen.

## 2026-08-13 23:38 PST — local adaptation changed to exact warm-start replay; mouth policy narrowed

The owner identified that the local corpus has no dependable lip articulation and
correctly rejected masking the entire 15-node sparse face: doing so removes stable
hand-to-face reference geometry needed by signs such as `GOOD` and `THANK YOU`.
The final local landmark policy now zeros only four nodes (`mouth_left`,
`mouth_right`, `upper_lip`, `lower_lip`) while retaining the other 11 sparse
eye/brow/nose/jaw/chin anchors. Citizen and SemLex retain all 15 face nodes. The
15-node face projection contains only 10,752 of the selected landmark model's
6,791,717 parameters. Physically deleting the four lip inputs would save only 2,816
parameters (0.041% of the model) while changing the frozen schema and losing the
ability to use those lips on Citizen/SemLex. Keeping the sparse 15-node schema and
masking four inputs only where their supervision is invalid is therefore the selected
mobile/accuracy middle ground. The
unchanged selected compact checkpoint scores 1,765/2,896 = 60.95% top-1 and
2,449/2,896 = 84.57% top-5 under that exact local-mouth-masked policy; this is the
correct comparable local landmark floor for the adapted model. Exact evidence is
`artifacts/reports/local_deep_clean_v17/orientation_augmentation_only_v1_mouth_masked_baseline/metrics.json`.

Two superseded laptop runs are quarantined and must never be promoted. The original
from-scratch, unmasked local run was stopped during epoch 11 after epoch 10 reached
92.86% Citizen validation and 89.30% local familiar-signer validation, because its
local silent-mouth supervision could erase real mouth knowledge and it omitted the
proven hand-RGB complement. Its ledger is
`artifacts/models/stage1_v17_local_deep_clean_mps_v1/ABORTED.json`. The subsequent
all-face-masked run was stopped during epoch 1 because its 15-node mask destroyed
useful face-contact geometry; its ledger is
`artifacts/models/stage1_v17_local_deep_clean_face_masked_mps_v1/ABORTED.json`.

The approved best design is exact-checkpoint balanced replay adaptation. The landmark
branch starts strictly from selected checkpoint SHA-256
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`;
the hand-RGB branch starts strictly from selected checkpoint SHA-256
`ec16d1b14a2346fecd993d92b3c92b4965cd1204132f64514d86c109570e6d84`.
Both replay Citizen/SemLex/local at approximately 34/33/33 source mass, select only
on official Citizen validation, with local validation allowed only to break an exact
Citizen top-1 tie for `best_model.pth`. A separate
`best_promotion_gate_model.pth` retains the highest local-validation top-1 subject to
the already predeclared floor of 361/378 Citizen correct; it cannot be promoted until
it also passes the frozen SemLex and eight-angle orientation gates. The landmark run
is predeclared at seed 1701, 80 epochs maximum,
20-epoch patience, 5e-5 peak learning rate, four warmup epochs, the unchanged full-
circle roll augmentation, and 34/33/33 Citizen/SemLex/local replay. The hand run is
predeclared at seed 1701, 100 epochs maximum, 20-epoch patience, 5e-5 peak learning
rate, four warmup epochs, and the same replay margins. These settings may not be tuned
after seeing local-validation results. The existing mouth-RGB and lower-face-RGB
experts remain frozen and never see silent local clips. Strict exact-state/model/
schema/manifest/label-map loaders have been added and executed successfully for both
source checkpoints.

The first executable replay epoch served as a protocol smoke: it moved local
mouth-masked validation from 1,765/2,896 = 60.95% to 70.75% while Citizen moved by
exactly one clip from 362/378 to the allowed 361/378 floor. The initial trainer kept
only strict Citizen-best checkpoints and would have discarded this gate-eligible
tradeoff. It was stopped during epoch 2; no result was promoted. The dual-retention
policy above was then implemented before the definitive seeded restart so both the
unchanged Citizen-best control and a gate-eligible adaptation candidate are preserved
for SemLex/orientation evaluation.

The first eight-shard local hand-crop extraction correctly failed closed after 197
train and 201 validation outputs because legacy web containers can over-report frame
counts. Investigation proved the raw files were unchanged: the v17 landmark archives
recorded both reported and actually decoded counts. RGB extraction now reconstructs
the exact deterministic raw-frame sample used by the landmark extractor (including
its bounded reservoir path), decodes through EOF, and verifies reported count,
decoded count, and rotation metadata against landmark provenance. Focused regression
tests for 72-reported/47-decoded and 156-reported/151-decoded cases pass. The full
eight-shard extraction has resumed and safely reuses the already validated crops.
Citizen test, SemLex test, and local test remain untouched.

The resumed extraction is now complete: exactly 13,381/13,381 train and
2,896/2,896 validation crop archives across all 94 local classes. A full structural,
finite-value, explicit-missing-view, JPEG-offset, schema, manifest-hash, split,
eligibility, and sealed-test provenance audit reports zero errors. Train/validation
valid-view fractions are 73.66% and 74.13%; packed JPEG payloads total 6,034,709,430
and 1,311,159,077 bytes. Audit SHA-256 values are
`44c0aea4a8d4cab00f57f5d18789794618e0c0795e235d51ecdaa9d85f102bb1`
and `7e185bcbe9dc7cc40387f5841ba1d094daebd3ae0f302c4229e1663969674c80`.
The reproducible auditor is
`scripts/audit_local_deep_clean_hand_features_v17.py`, SHA-256
`d2afc666583fdeb68f9ca53092de80972dcc66e8e6d9d90d055c4ed325573bd7`.

The first Kaggle kernel created while the private feature dataset was still indexing
had its invalid source silently dropped by Kaggle and errored by design. It is an
infrastructure failure, not a training experiment or model result; laptop MPS is the
current critical path.

## 2026-08-13 23:01 PST — local trainval corpus finalized and Kaggle upload staged

Fresh current-v17 Apple Vision extraction is complete. Validation retained all
2,896/2,896 clips; train retained 13,381/13,382 clips. The only extraction rejection
is `HE/HE_SHE_6eb9df3e`, whose raw movie produced no hands; there were zero processing
failures. Both final splits still cover all 94 admitted local classes. Every one of
the 16,277 final train/validation archives passes the current v17 integrity audit with
zero schema or invariant errors. Final train/validation manifest SHA-256 values are
`17124e03b59dc3b2fa6031af19e86875eeff75698e62ee2bc531dbc19f2621c7`
and `ab28c5d754133e140cbbcc5a3a8ceaffd083efddc1ebd143f70441786d6dd122`;
finalization summary SHA-256 is
`caed8aaceaabc9d96fd14093772d779b1eb0c8a53eee2717549efc78ad42459a`.

A real MPS optimizer/checkpoint smoke completed through the exact strict loaders:
1,475 Citizen train, 1,388 approved SemLex train, 13,381 local train, 378 Citizen
validation, and 2,896 local validation. Provenance confirms the 0.34/0.33/0.33
class/source-balanced margins, continuous full-circle roll augmentation, current
part-wise architecture, Citizen-only checkpoint selection, and false test-access
flags. Smoke result/provenance SHA-256 values are
`72069114257b62ac359cb268d1a9d052dd82f76b1ee50aa39558310513035a54`
and `c08092f764fcd1b2768c1ee72b9c6c43717bf331b080437f0d47bc3ee248ae80`.
The full affected suite passes 73/73, including six preparation/finalization tests,
50 Stage-1 tests, and 17 extractor tests. `git diff --check` passes.

The fail-closed Kaggle package contains exactly 16,280 regular files: 13,381 train
features, 2,896 validation features, two final manifests, and the finalization
summary. It contains no local test member and no Citizen/SemLex test artifact. The
153 MiB archive SHA-256 is
`7efc93abbfec3f7c8885ecfbe17394db1a26fc7fbf9226b72dbf390154b5c6ca`;
its independently re-extracted tree SHA-256 is
`254019cd2cfa8c9e1e2461aef66708ece0b88b9574f2f915ffa6a3332f0bf56d`.
Private Kaggle datasets `kokoab/slt-v17-local-deep-clean-trainval-v1` and
`kokoab/slt-v17-stage1-local-deep-clean-code-v1` were uploaded. The code dataset is
indexed; the larger 16,280-file feature dataset is still server-side indexing, so the
CUDA kernel is correctly held rather than launched against a partial mount. No test
split was accessed.

## 2026-08-13 22:31 PST — local validation extraction frozen; compact baseline measured

Fresh Apple Vision extraction completed for all 2,896/2,896 local-validation clips
across all 94 admitted local classes with zero no-hand cases, failures, skips, or
finalization rejections. All archives pass the v17 schema/shape/finiteness/missing-data
audit. Final validation manifest SHA-256 is
`ab28c5d754133e140cbbcc5a3a8ceaffd083efddc1ebd143f70441786d6dd122`;
the audit report at `artifacts/reports/local_deep_clean_v17/VAL_V17_AUDIT.md` has
SHA-256 `2eb08e49eac169f4bd51fa30b4208ede56abac1dc039a8e5becb6c99f93c6442`.

The unchanged compact orientation checkpoint
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
scores 1,803/2,896 = 62.26% top-1 and 2,461/2,896 = 84.98% top-5 on
this explicit non-signer-disjoint familiar-signer validation set, with 61.58% macro F1
over the 94 present classes. This is the frozen pre-training local baseline: the new
challenger must exceed 1,803 correct while also satisfying Citizen, SemLex, and
eight-angle orientation gates. Exact metrics and logits are under
`artifacts/reports/local_deep_clean_v17/orientation_augmentation_only_v1_baseline/`,
with SHA-256 values
`18a43fd07b28c72fea10a7a0f636615028d3225f3ba8d276b3d0ea94a376bf36`
and `aa0a7ec466de84f066ba336ebc6a5544a355a965fe7b8fe2d10a6f0973015dab`.
Citizen test, SemLex test, and local test remain untouched. Train extraction continues
independently and had reached 7,009/13,382 archives at this checkpoint.

## 2026-08-13 22:06 PST — finalized-manifest and train-only packaging gates added

Fresh extraction remains active across four disjoint class shards; at this handoff it
had produced 2,734 of 13,382 train archives with all four Apple Vision workers healthy
and no logged extraction errors. `scripts/finalize_local_deep_clean_v17.py` now validates
every archive against the current v17 schema, emits immutable train/validation final
manifests plus a rejection ledger, and fails closed if extraction loses any of the 94
admitted classes. The training and validation loaders accept those final manifests
only when extraction completeness, schema, signer-overlap approval, and false
Citizen/SemLex test-access claims all hold. The frozen-model local evaluator now
defaults to the final validation manifest.

`scripts/package_local_deep_clean_v17_kaggle.py` stages only finalized local train and
validation features for the future private Kaggle job and explicitly forbids local
test members. Six preparation/finalization tests and the two focused finalized-loader
tests pass; affected Python files compile and focused `git diff --check` passes.
Current SHA-256 values are finalizer
`46f35d292bf0be94d8ba6c3fdac73bcccc674ccbfb30a6b3fec01a7f3fed0dce`,
packager `50d2a38ff0e5e3b2eec92ec0c484aae47bbc4959967a556cbf746e0afc3db813`,
trainer `085d6cc344aadd79dd7919b24106fa64e41bd087f4e6eb201bbce9fcf74f98cc`,
and evaluator
`e256bd2c8122e460297069ff80898756e5c1170b72afd3ee9287f801bbc0d30a`.
Neither Citizen test, SemLex test, nor the unused local test split was accessed.

The CUDA-only runner and reproducible Kaggle job builder are now implemented as
`active/v17/kaggle_stage1_local_deep_clean_runner_v17.py` and
`scripts/prepare_local_deep_clean_v17_kaggle_job.py`. The run is predeclared at
Citizen/SemLex/local source margins 0.34/0.33/0.33, seed 1701, current part-wise +
global architecture, and continuous full-circle roll augmentation. It selects only
on official Citizen validation top-1. Promotion requires all four gates: at least
361/378 Citizen validation top-1 (no more than one clip below the frozen compact
orientation model), at least 839/978 SemLex validation top-1, and strictly better
local familiar-signer validation top-1 than the frozen compact baseline, while the
eight-angle landmark-roll stress suite must retain at least the current 348/378
worst-angle score. Runner,
packager, and job-builder compile and pass `git diff --check`; their SHA-256 values are
`487d0c737ecb85e9b8c3267b3072734ee8cbedb986f05305989452635e1c384f`,
`7e1a62e9a097225a46bcf568d3d68bd7645fb53fbf528019f9d9b4602a1023a7`,
and `cf7472d35a63e71eeeaf500406edfce0ee4cd9fbbb8a44423c8131379824baa3`.
The combined preparation/training suite is 56/56 passing; the 17/17 extractor tests
also pass, including real Apple Vision rotation/mirroring and portrait/landscape
isotropy coverage.

An explicit raw-byte decontamination audit also passes before training. The 13,382
local-train and 2,896 local-validation rows have zero SHA-256-identical matches against
all 1,854 unique Citizen train/validation raw clips and all 2,448 unique SemLex
train/validation raw clips. The audit deliberately does not read either protected test
split or the unused local test split. Exact evidence is
`artifacts/reports/local_deep_clean_v17/raw_hash_overlap_audit.json`, SHA-256
`d875b76decd357ae6b630e3d7527da06eccf0f715a30e8dd3cecfd4843db873b`;
the reproducible audit script SHA-256 is
`13683961bd6178c28c4b0c34f1737b811b0a9ba5ffee341cf6b39cc022176ecc`.

## 2026-08-13 21:57 PST — 94-class local deep-clean v17 challenger prepared

The project owner approved a separate model trained with the historical local clean
data together with Citizen train and SemLex train, and explicitly accepted local
signer overlap between train/validation/test. Old tensors and old training code remain
excluded: the historical v16 manifest is used only as a row whitelist back to raw
videos, which are being re-extracted with the current orientation-safe Apple Vision
v17 extractor and will be trained by the current `active/v17` part-wise/orientation
trainer. The selected Core ML checkpoint remains untouched as the control.

The exact clean-data lineage is now resolved. The raw v16 ledger has 66,770 clips, the
first confidence-cleaned ledger has 64,767, and the final deep-cleaned ledger has
62,023 clips across 310 classes. Final deep-clean manifest SHA-256 is
`5cf1e20a48ba2188cf93abf7564676fa5562b9ef2f20051adf6c0a091b9e0f70`.
The owner's remembered smaller snapshot is also real: the older float16 manifest has
57,557 entries and SHA-256
`3bb9072364b124396b041f6b038aa5729f3d07cd8f7c1fa38e94bc145cb5cb02`.
The final 62,023-entry deep-clean list is authoritative for this new experiment.

Ninety current labels overlap the deep-clean ledger exactly. Four more are recoverable
without mixing the two sides of old merged folders: `EAT` from only the `EAT` side of
`EAT_FOOD`, `MAKE` from only the `MAKE` side of `MAKE_CREATE`, `SAME` from the old
`ALSO_SAME` folder stored as `ALSO`, and `HOME` from `HOUSE`. Six current classes have
no traceable local raw class and are not filled with semantic neighbors: `FIND`,
`HAVE`, `HUNGRY`, `LISTEN`, `SICK`, and `TALK`. Citizen and SemLex continue to supply
all 100 classes.

`scripts/prepare_local_deep_clean_v17.py` resolves all relevant historical rows back
to exact raw movies and SHA-256-hashes both raw bytes and historical features. Rows
connected by either identical raw bytes or identical historical features are assigned
to one deterministic content-group split; six cross-label duplicate rows are
quarantined. The resulting 19,111 rows cover 94 classes: 13,382 train, 2,896 local
validation, and 2,833 currently unused local test. Every split covers all 94 local
classes. Signer overlap is explicit and approved; exact duplicate-content leakage is
not. There are 16,098 canonical/raw-text-equal rows and 3,013 traceable but
variant-unverified rows. Preparation summary SHA-256 is
`d9ca1aa84898e8241942f348442b9336ff229cb2786bc945d9502979afe5cad4`;
train/validation manifest hashes are
`b19ff51b4f87ccac1a2f2ea1d191a65016b3d64b3db979e85d50d737aeed7f2f`
and `5e98ab2017ded4eccd4e883c60491016e2010fa8db5629fe44393fb09832ed02`.

The updated v17 trainer now accepts the explicit non-signer-disjoint local validation
manifest and reports its loss/top-1/top-5/macro-F1 every epoch. Checkpoint selection
remains Citizen official validation top-1; local validation is a secondary familiar-
signer-domain diagnostic, so its much larger clip count cannot mask Citizen
regressions. Both local train and validation manifests prove false Citizen/SemLex test
access. Four new preparation tests and the focused local-validation loader test pass,
and affected files compile. Current source hashes are preparation script
`d57095a1e113e51dc618a42d474e9b236d47fb88de6d404f9f7cfbf182666c54`,
trainer `4c50a8ff402b46d77d4181a960113db959a90b4202cffeaaa5b9899be9565a87`,
new tests `4bc1c01b3c906d539a94170b0d352b93976152abeb52ffbe54b0f51e134aab95`,
and updated Stage-1 tests
`9cdd760cbc086452c9c51fd5a0131c2dfeb457bd7611e6adbd82c6045883f276`.
Train/validation Apple Vision v17 extraction is active; local test remains unextracted
and unused.

## 2026-08-13 21:45 PST — local 77-class external diagnostic exposes generalization gap

At the project owner's direction, the next model diagnostic used the existing local
`data/raw_videos/ASL VIDEOS` collection instead of waiting for new phone capture. The
raw directory currently contains 66,875 files in 316 class directories and occupies
about 32 GiB. It is a mixed historical corpus: filenames identify local hash-named and
numbered sessions as well as known scraped sources, but do not contain trustworthy
per-file signer identities. The owner's estimate of about seven new recurring signers
is consistent with the historical description but cannot be mechanically verified or
used to claim a signer-disjoint split.

The evaluation therefore used the already frozen, model-independent quality shortlist
at `data/local/local_citizen100_quality_audit_q82_cap14_exact/`. Its manifest SHA-256
is `45351f760dc8c7e1d064f04296e6055676cf50a900609a2ed3cef3693a9b1a14` and contains
1,021 unique clips across the 77 current classes whose directory text exactly equals
the pinned Citizen raw gloss. Known MS-ASL/WLASL/SignASL filenames, the mixed `I`
folder, non-exact variants, weak decodes, and near-duplicate sessions remain excluded.
All 1,021 Apple Vision v17 archives again pass the integrity audit with zero errors;
prior raw-hash decontamination found zero byte-identical overlap with all Citizen rows
and all retained SemLex-train clips. The other 23 current classes are not silently
filled with aliases or different numeric variants.

The untouched selected orientation checkpoint
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
scores 690/1,021 = 67.58% top-1 and 903/1,021 = 88.44% top-5, with 53.80%
100-class macro F1. Fifty-eight classes meet the existing model-consistency screen,
17 are ambiguous, and two are high-risk. The sharpest mismatch priorities are `COME`
(0/14 top-1 and 0/14 top-5) and `SIGN` (1/14 top-1 and 3/14 top-5). Exact evidence is
under
`artifacts/reports/local_citizen100_quality_audit/orientation_augmentation_only_v1_external_diagnostic/`;
the summary SHA-256 is
`d736d8baee12b87dfb4b37780acbec98eef828dea1aea1d07116bd7bc73a6459`
and the immutable logit ledger SHA-256 is
`33dcdca9d4386b1640913b6144e7d5e3ae1bc18775ec7facd1b33802acc92f33`.

The previously selected Citizen-validation 75/25 landmark/hand-RGB fusion was then
applied unchanged, without choosing weights on this local pool. It improves the
orientation model to 738/1,021 = 72.28% top-1 and 932/1,021 = 91.28% top-5,
rescuing 51 orientation-model errors while regressing only three correct predictions.
The result SHA-256 is
`60eb2a700eed39ea77ace7433d3099c8b01fc0f117ea0e3acbf0f85626e81caa`.
This independently confirms that RGB covers meaningful landmark weaknesses, but the
remaining 280 dual failures and variant uncertainty make blind distillation or blind
use of all 60k files premature.

Decision: this local collection is now the preferred source for the next train-only
v17 supplement, restricted to the current vocabulary and exact pinned variants. It
does not replace a future signer-disjoint benchmark unless anonymous signer/session
membership is first attached to every admitted row and held-out identities are frozen
before training. The 67.58%/72.28% figures are external local-label diagnostics, not
production accuracy. No Citizen or SemLex test split was accessed, and the selected
checkpoint and Core ML package remain unchanged.

## 2026-08-13 21:27 PST — iPhone 13 simulator Core ML orientation suite passes

The automated simulator harness is complete and the dedicated virtual device is
strictly an iPhone 13: `SLT Orientation Benchmark iPhone 13`, CoreSimulator device
type `com.apple.CoreSimulator.SimDeviceType.iPhone-13`, model identifier `iPhone14,5`,
and UDID `ABE172E1-A940-4937-92D9-1C666E674060`. No iPhone 17 device was created.
Apple no longer offers the exact iOS 26.2 simulator runtime through Xcode's download
catalog, so the harness compiled against the installed iOS 26.2 SDK and used the
nearest compatible runtime, iOS 26.3.1 (`23D8133`). The runtime and virtual device
remain installed for repeatable runs.

The first literal end-to-end simulator attempt correctly failed closed at all eight
angles. Runtime diagnostics show that the iOS 26.3.1 simulator image contains the
Apple `cnn_human_pose.espresso.net` and `.shape` files but omits their matching
`.weights` file; Vision reports `Unable to setup request in
VNDetectHumanBodyPoseRequest`. The Mac framework has matching graph hashes and the
weights, confirming this is a simulator-runtime asset boundary. The physical-device
app's normal Apple Vision extraction path is unchanged. The final simulator harness
therefore performs the unchanged v17 Apple Vision extraction on the macOS host,
serializes and SHA-256-pins each `(32,61,5)` tensor, and runs only Core ML model loading
and inference inside the iPhone 13 simulator. Every report explicitly records
`extractionExecutionEnvironment: host_macos_apple_vision`, `endToEndPipeline: false`,
`hardwarePerformanceClaim: false`, `thermalsInterpretable: false`, and the missing-
weights limitation. This is not end-to-end iOS Vision evidence and makes no physical
iPhone, ANE, memory, thermal, or sustained-latency claim.

Suite `orientation-v17-ios26-3-1-20260813T132843Z` passed. It uses only Citizen's
official validation clip `020030442376253177-HELLO.mp4`, source SHA-256
`d5d3ac36b623c46b0b22a42dbaa36e5e36321bc1b5987ef719dcd96d9d63473b`,
and expanded-canvas 0/17/37/73/90/123/180/270-degree inputs without crop or
anisotropic stretch. All 8/8 host Apple Vision extractions succeeded, all 8/8 iPhone
13 simulator predictions were `HELLO`, and each report contains exactly 200 timed
inferences. Corrections were `0/0/0/270/270/270/180/90`; residual rolls were
`0/17/37/-17/0/33/0/0`, all within 45 degrees. Mean per-angle median simulator
inference was 5.5683 ms and maximum p90 was 6.0731 ms; these are Mac simulator timings
only. The selected checkpoint remains
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
and the Core ML tree remains
`1cfd5e97cb8ebb29b424b1391ceb85ed9d62e5b7e25841b86254d414ccd0fb5e`.

The final result is
`artifacts/reports/orientation_v17_simulator_benchmark/orientation-v17-ios26-3-1-20260813T132843Z/result.json`
with SHA-256
`929881c5658f3ad1e26c4db1eed83fa7f5311ef7e97b36128077b89b3ca4918d`;
its aggregate SHA-256 is
`a74212ff84a97d5a1377bfcde996641b3651076fade25b0d2a8708dac10116ce`.
The host runner, app automation/reporting code, and focused test SHA-256 hashes are
`ced997fb753ad9dc86f5458fbccb2e8e4b32c150e8f99cab04216fbdb43235e4`,
`73f5eafc272355e2d6cc41c531cff8ba519894e5104b23dd22653d4074977e36`,
and `2efc6be3e0ac6b55b24f06e476ccf4237dc238f46bdd94a87e68cbb366a3d7d1`.

Final validation passes: unsigned Release builds for both iPhone Simulator and generic
iPhoneOS; all 84 focused Stage-1, extractor, independent-capture, and simulator tests;
11 simulator evidence JSON files parsed; changed Python compilation; the 1,100-row
capture-pack setup audit with zero errors (audit SHA-256
`de885a97a09698abf42cb666522dd20c8ab826f7c6c2e964916640a59faf7732`);
and `git diff --check`. Citizen and SemLex test splits were not accessed. The app is
compiled and ready for the deferred signed install and full end-to-end run on a real
iPhone 13.

## 2026-08-13 18:51 PST — arbitrary-orientation gate passes; ready for real phone

The final automatic coarse-orientation rule supersedes the 18:20 face-band rule. It
first chooses the horizontal versus vertical anatomical-axis family from the maximum
shoulder/eye-line horizontalness, using body confidence only as a tie-break, and then
uses signed mouth-below-eyes geometry to distinguish the two opposite directions in
that family. This prevents a correctly upright clip from being rotated merely because
its shoulders temporarily disappear during signing. Python and Swift implement the
same rule. Container metadata is still applied first; the probe only chooses a
lossless quadrant, and the trained classifier covers the remaining continuous roll.
No path crops or anisotropically stretches ordinary input video.

All 175 official ASLLVD clips were re-extracted from the exact top/front-camera pixels
and inclusive annotated frame interval with the final rule. All 175 were retained as
already upright, all 175 pass the v17 integrity audit, and zero clips failed. The final
manifest SHA-256 is
`e2d6d18cb4e43e1809b35e97f980f561a04f0affacd5877ab2d0e5e666009faf`;
the audit SHA-256 is
`8809631035963351a0c6f50a349764900ae850625424a4f177f7998d4962715e`.
Before training on this source, the frozen orientation fallback independently scores
110/175 (62.86%) top-1, 152/175 (86.86%) top-5, and 60.58% macro class top-1 over
52 exact variants and six external consultants. The evidence report is
`artifacts/reports/asllvd_asllex_v17_external_baseline.json` with SHA-256
`c80ad4e1b01fcf76e09b77b073af30a74289df5136657ca280fb0f41612a041b`.

Private Kaggle feature dataset version 3 contains only those final derived features
and provenance; raw ASLLVD movies were not uploaded. Kernel
`kokoab/slt-v17-stage1-orientation-asllvd-v1` version 3 completed successfully, pins
the final manifest, and confirms false Citizen/SemLex test-access flags. Its
checkpoint SHA-256 is
`661be8d6db71df8e07c161d57cc12464566f20317620ba6005d5c54fa552b412`,
but it scores only 355/378 (93.92%) on Citizen validation. This is below the
predeclared 359/378 clean-domain floor, so the challenger is rejected immediately;
SemLex and raw-pixel errors are not used to rescue it. Kernel versions 1 and 2 remain
explicitly superseded, and all three relevant orientation kernels now report
COMPLETE. The selected phone candidate remains the continuous-roll augmentation-only
fallback checkpoint
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`.
The four-stream RGB/landmark research teacher remains unchanged at 370/378 Citizen
and 882/978 SemLex.

The frozen candidate manifest now pins final extractor SHA-256
`d049ae34d732fa504ad8702a91d3409dcf1debd8415bc5dea588d7f243138f47`;
its own SHA-256 is
`cd0f2be27cabd1b9d9eedc15d4b74cfcd01c09f552e2efba491678ca55323ee9`.
The untouched 1,100-row independent capture pack passes its setup audit with that
lock and remains inference-free and capture-pending. The final Swift orientation
pipeline and fallback Core ML model compile in unsigned Release mode for both generic
iPhoneOS arm64 and the arm64/x86_64 iPhone Simulator. Both bundles contain the
compiled 13 MiB model, frozen 100-class manifest, and exact model-provenance manifest;
all four iOS interface orientations are declared. Real-device timing, memory,
thermals, and independent capture accuracy are deliberately still unmeasured: the
bundle is the instrument that will collect those measurements on the actual phone.

The definitive raw-pixel sweep with that exact final selector completed over the
fixed 100-clip Citizen validation slice at 0, 17, 37, 73, 90, 123, 180, and 270
degrees. All 800/800 conditions extracted successfully. Correct counts are
93/96/85/94/93/91/93/93, respectively: 92.25% eight-angle mean and 85% minimum.
The 90-, 180-, and 270-degree predictions are exactly identical to upright. Every
upright clip remains at correction 0; every 90/180/270 clip receives exactly the
inverse lossless quadrant; 99/100 clips at 37 degrees remain at 0, while all 73- and
123-degree clips choose the nearer 270-degree correction. The exact report is
`artifacts/reports/stage1_v17_raw_orientation_robustness/augmentation_plus_vision_auto_axis_family/metrics.json`
with SHA-256
`c36f3ec9408158714f12b5942c816ec4a01d8c28437bbe04438adb0bc65e4c77`.

Final validation passes: all 78 focused Stage-1, extractor, and independent-capture
tests; 175/175 ASLLVD schema/integrity archives with zero errors; the 1,100-row frozen
capture-pack setup audit with zero errors; Python compilation for all new/changed
training, extraction, acquisition, finalization, evaluation, and Kaggle-runner code;
unsigned Release builds for generic iPhoneOS and the iPhone Simulator; compiled-model
and manifest bundle checks; and `git diff --check`. Core ML exhaustive parity remains
378 validation samples with zero top-1 mismatches and maximum absolute logit error
0.006403. All relevant Kaggle jobs are COMPLETE. Citizen and SemLex test splits were
not accessed.

## 2026-08-13 18:20 PST — raw-orientation gate passes; ASLLVD view contract corrected

The final automatic orientation rule gates candidates by body confidence, separates
adjacent quadrants with shoulder/eye horizontalness, treats near-equal axes as ties,
and resolves 0-versus-180 with signed mouth-below-eyes anatomy. On the fixed 100-clip
Citizen validation raw-pixel slice it extracts 100/100 at every angle and scores
93 at 0, 95 at 17, 82 at 37, 94 at 73, 93 at 90, 89 at 123, 93 at 180, and 93 at
270 degrees (91.50% eight-angle mean). Exact quadrants have identical predictions and
coverage to upright. The report is
`artifacts/reports/stage1_v17_raw_orientation_robustness/augmentation_plus_vision_auto_axis_faceband/metrics.json`
with SHA-256
`4b1a3c4af163582119d032906d45f22a7ae89c61699a3fdf8121b70f6c60006b`.
This evidence uses expanded canvases, no crop, no anisotropic stretch, and no test data.

Visual inspection of the official ASLLVD movies exposed that every downloaded movie is
an extended vertical composite: front camera on top, side camera below, plus 50 frames
of context before the workbook's annotated `Start`. The first 175-feature composite
bundle and its Kaggle v1 checkpoint are therefore superseded, not selected. The
superseded checkpoint scored only 353/378 Citizen validation and cannot satisfy the
predeclared clean-domain floor. `scripts/materialize_asllvd_front_view_v17.py` now
materializes only the exact top/front pixels and the inclusive workbook `Start..End`
interval (the official extended movie starts at `Start-50`). It never resizes or
stretches pixels and uses lossless H.264 encoding. Burned-in frame-number contact-sheet
inspection confirms the retained interval; for example WHAT/Brady contains exactly
frames 3121..3145 and 25 output frames.

The corrected front/exact feature set again produces and audits 175/175 v17 archives
with zero failures. Before any training on it, the frozen orientation fallback scores
111/175 (63.43%) top-1 and 84.57% top-5 over 52 exact variants and six external
consultants, versus 23/175 on the rejected composite/context representation. The final
manifest SHA-256 is
`602dcef30b6f3b4355b4eac8b385692f26cddf16faa447bdd1786b140cea3874`;
the model-unseen report is
`artifacts/reports/asllvd_asllex_v17_external_baseline.json` with SHA-256
`328a1c647d151544704e13a522a5175f59385330a419e972485a1270f8c2c614`.
Private Kaggle feature dataset version 2 supersedes version 1, and kernel version 2 is
RUNNING against only the corrected features. Raw ASLLVD movies were not uploaded.

## 2026-08-13 17:52 PST — 175 exact variants admitted; controlled retrain running

The official ASLLVD/ASL-LEX acquisition is complete. All 175 selected movies across
52 frozen classes and six named consultants downloaded and passed full video decoding.
Apple Vision v17 extraction produced 175/175 archives with zero no-hand or processing
failures; all 175 pass schema, finite-value, binary-presence, and missing-zero
invariants. The finalized manifest SHA-256 is
`9ebf73276a8e5e12ea33cdbc640169011ebd15170b2bb52e62d34fe6a69a9a8c`.
The audit is `artifacts/reports/ASLLVD_ASLLEX_V17_EXTRACTOR_AUDIT.md` with SHA-256
`776562fc35cad9c7b24002e56b806ea884f593144453996d6d11c8e4db841db1`.
No Citizen or SemLex test data was accessed.

Only the derived v17 landmarks and exact provenance were uploaded to the private
Kaggle dataset `kokoab/slt-v17-asllvd-asllex-features-v1`; raw ASLLVD movies were not
uploaded. A local two-batch smoke proves the strict loader admits exactly 175 samples,
52 classes, and the six known signer identities, and that requested sampler margins
converge to Citizen 0.45 / SemLex 0.45 / ASLLVD 0.10. Private T4 kernel
`kokoab/slt-v17-stage1-orientation-asllvd-v1` is RUNNING with the same seed,
architecture, optimizer protocol, and continuous full-circle augmentation as the
orientation winner. Its only training-data treatment is the new exact-variant source.
Before its result is known, replacement is constrained as follows: it must improve the
unweighted mean of Citizen and SemLex validation accuracy, lose no more than one
percentage point on either clean domain, retain at least the current 348/378 worst
nonzero landmark-roll result, and not reduce the corrected raw-pixel eight-angle mean
or extraction floor. A challenger that fails the clean gates is rejected without
using raw-pixel errors to rescue or tune it.

The first axis-prioritized automatic orientation selector is rejected. On the fixed
100-clip Citizen validation raw-pixel slice it scored only 80/100 upright because
tiny shoulder-axis differences sometimes chose 180 degrees over 0, despite improving
some intermediate-angle decisions. The corrected rule gates by body confidence,
uses an axis tolerance band to distinguish adjacent quadrants, then uses signed face
anatomy to resolve 0-versus-180. Sixteen extractor tests pass, including the real
0/17/37/73/90/123/180/270 correction sequence. A new full 800-condition raw-pixel
run is active; the rejected selector is not bundled for final phone testing.

## 2026-08-13 17:20 PST — exact ASLLVD supplement selected; failed archive route closed

The initially downloaded Rutgers ASLLRP distribution archives passed transport and
CRC validation but do not contain the newer recording IDs referenced by the current
Data-Sharing Project metadata. The exact-variant preparation therefore failed closed
at zero clips; those archives are not training data. The official BU ASLLVD workbook
provides a usable alternative with direct per-consultant movie URLs. An exact join
from each frozen Citizen class's ASL-LEX code through ASL-LEX 2.0 nonempty
`SignBankAnnotationID` to ASLLVD `Gloss Variant`, removing only the documented `+`
repetition suffix, selects 175 clips across 52 classes and six named consultants.
Selection permits at most one recording per consultant and five per class.

`scripts/prepare_asllvd_asllex_supplement_v17.py` downloads the official movies,
fully decodes them, hashes them, and records exact raw/feature provenance. The
download is resumable and active. The clips remain research-only/noncommercial and
will not be redistributed. The trainer loader independently enforces the exact
variant, signer, eligibility, tier, and false Citizen/SemLex test-access contracts.
`scripts/finalize_asllvd_asllex_supplement_v17.py` now retains extraction failures for
provenance while admitting only schema-valid v17 archives. No frozen test data was
accessed.

The current augmentation-only orientation winner has also been exported as a fallback
FP16 ML Program at `artifacts/coreml/Stage1OrientationV17.mlpackage`. It is 13.29 MiB
with package-tree SHA-256
`1cfd5e97cb8ebb29b424b1391ceb85ed9d62e5b7e25841b86254d414ccd0fb5e`.
Exhaustive parity over all 378 Citizen validation archives has zero top-1 mismatches;
the maximum logit difference is 0.006403. The measured 16.31 ms median and 21.67 ms
p90 are Mac timings only and are not phone evidence. The package and a manifest that
pins checkpoint SHA-256
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
are wired into the orientation benchmark Xcode project. The bundle will be replaced
and parity rerun only if the independent ASLLVD challenger wins the declared
development gates.

## 2026-08-13 16:55 PST — detector-space quadrant fix and official supplement acquisition

The fixed 100-clip Citizen validation raw-pixel stress completed for both orientation
retrained checkpoints. The augmentation-only model scores 93/100 at 0 degrees,
86/100 at 37, 79/100 at 90, and 20/100 at 180. The canonicalized model scores
92/100, 89/100, 73/100, and 17/100, respectively. All 800 attempted extractions
returned a usable sample. At 180 degrees Apple Vision's mean body presence is exactly
zero for both, proving the remaining inversion failure occurs before the classifier.
The augmentation-only checkpoint is the better raw-video candidate. Exact reports are
under `artifacts/reports/stage1_v17_raw_orientation_robustness/augmentation_only/`
and `canonical/`; no Citizen or SemLex test data was accessed.

`active/v17/extract_v17.py` now applies container orientation first and, in automatic
mode, probes three frames at four lossless quadrants using Apple Vision face/body
anatomy. It selects the quadrant whose mouth-to-eye vertical relationship and body
confidence are most upright, then performs the main extraction once. The continuously
roll-augmented classifier therefore sees at most a 45-degree residual roll; this is
not a portrait/landscape classifier and does not stretch aspect ratio. Explicit manual
rotations remain authoritative and can bypass the probe. Sixteen extractor tests pass,
including real Apple Vision recovery of 0-, 90-, and 180-degree source rotations. A
new full 100-clip detector-space validation run with this automatic probe is active.

The same four-quadrant anatomy probe is implemented in the generic iOS benchmark
pipeline after AVFoundation applies the file's preferred transform. The app accepts
any native aspect ratio, records the chosen correction and all orientation scores,
and still measures extraction, Core ML latency, memory, thermal state, and expected
label accuracy. Its simulator build succeeds. The model-byte measurement was also
corrected to measure the compiled model directory rather than the whole app bundle.

The ASLLRP route described at this timestamp was subsequently rejected because the
downloaded distribution archives did not contain the metadata's newer recording IDs.
The 17:20 entry records the replacement official ASLLVD acquisition and is canonical.

## 2026-08-13 16:29 PST — arbitrary-roll failure measured; two controlled retrains active

The existing compact part-wise checkpoint was stress-tested on the official Citizen
**validation** landmarks only at continuous synthetic camera rolls. It scores 366/378
(96.83%) at 0 degrees, 363/378 (96.03%) at 17, 347/378 (91.80%) at 37,
151/378 (39.95%) at 73, 40/378 (10.58%) at 90, 4/378 (1.06%) at 123,
2/378 (0.53%) at 180, and 38/378 (10.05%) at 270. This proves the previously
selected classifier was not orientation-robust even though v17 extraction already
preserves aspect ratio and honors right-angle video orientation metadata. Exact
metrics are under
`artifacts/reports/stage1_v17_orientation_robustness/partwise_original/`.

`active/v17/model_v17.py` now has an optional, parameter-free, missing-safe clip-level
camera-roll canonicalizer. It estimates an anatomical horizontal axis from
confidence-weighted shoulders with an eye-line fallback and rotates every landmark XY
channel in isotropic space. Forced onto the old checkpoint, it gives bit-stable class
predictions at 0, 17, 37, 73, 90, 123, 180, and 270 degrees, but only 295/378
(78.04%) at every angle. This is a diagnostic, not a selected result: the old model
was not trained on canonicalized inputs. Exact metrics are under
`artifacts/reports/stage1_v17_orientation_robustness/partwise_forced_canonical/`.

A real train-only Citizen clip was also synthetically rotated in pixel space before
Apple Vision. Hand detections remained nonzero at every tested angle, but coverage
degraded: observed hand frames were 23/19/20/15/16 at 0/37/90/123/180 degrees,
respectively, while body presence dropped to zero at 90 and 180. Therefore landmark
rotation alone is insufficient evidence for raw-video robustness; pixel-level
orientation stress and missing-landmark behavior must be included before handoff.
The clip was read only and no Citizen or SemLex test data was accessed.

The first Kaggle augmentation kernel failed before training because its initial code
overlay contained the new trainer but an old `model_v17.py` that lacked the active
part-wise configuration fields. The private code dataset was corrected to include the
current model and trainer, with an explicit `test_data_included:false` manifest. The
augmentation-only kernel `kokoab/slt-v17-stage1-orientation-robust-v1` version 2 and
the independent augmentation-plus-canonicalization kernel
`kokoab/slt-v17-stage1-orientation-canonical-v1` version 1 are both RUNNING on T4s.
Both retain the exact train/validation-only Citizen+SemLex protocol, class/source
balancing, architecture, and seed; neither frozen test split is present or accessed.

The Apple Vision extractor now also accepts any finite explicit clockwise correction
angle, not only 0/90/180/270. Exact right angles retain lossless transpose/flip paths;
other angles use a single affine resampling pass on an expanded canvas that contains
all four transformed corners, so pixels are neither cropped nor anisotropically
stretched. The transform and its exact floating-point angle are recorded in output
metadata. Fourteen focused extractor tests pass, including arbitrary 37-degree canvas
expansion, non-finite rejection, exact right-angle/mirror equivalence, isotropic
portrait/landscape geometry, and two real Apple Vision tests. This provides an explicit
path for detector-space stress generation and for a phone sensor-derived roll
correction; automatic container orientation metadata remains the default.

`active/v17/evaluate_raw_orientation_v17.py` now defines the corresponding fixed
detector-space evaluation: it rejects Citizen test by construction, selects only clips
already accepted in a train/validation feature inventory, applies expanded-canvas
pixel rolls, re-runs Apple Vision, and reports extraction coverage, top-1, and upright
prediction agreement for each angle. The candidate freeze was advanced only for the
intentional model/extractor runtime changes; checkpoint members, fusion weights, and
their evidence did not change. Candidate-manifest SHA-256 is now
`035bb08476b098c4a47273120cddeae9a42389b60ebc69f1eceb9b4105406ff4`.
The capture pack now pins that hash without changing its 1,100 immutable ledger rows
or schedules. The combined Stage-1, extractor, and capture workflow suite passes 74/74,
and the scoped diff check passes.

The fixed 100-clip raw-pixel validation stress has now completed for the old compact
checkpoint. Apple Vision returned an extractable sample for all 100 clips at every
tested angle, but model top-1 fell from 96/100 upright to 81/100 at 37 degrees,
4/100 at 90, and 14/100 at 180. Upright-prediction agreement was respectively
100%, 81%, 4%, and 14%. Mean hand presence stayed near 0.52--0.56, while body
presence fell from 0.385 upright to 0.099 at 90 and zero at 180. This confirms both
effects at realistic scale: the classifier lacks roll invariance, and auxiliary
landmarks become selectively missing after raw pixel rotation. Exact results are in
`artifacts/reports/stage1_v17_raw_orientation_robustness/partwise_original/metrics.json`.
The evaluator opened 100 Citizen validation videos already admitted by the frozen
validation feature inventory. It did not access Citizen or SemLex test data.

Both controlled orientation kernels completed successfully and their outputs were
pulled locally. The augmentation-only checkpoint SHA-256 is
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`;
it retained epoch 108 and completed 138 epochs. It scores 362/378 (95.77%) on clean
Citizen validation and 839/978 (85.79%) on SemLex validation. In landmark-roll stress
its top-1 is 358/378 at 37 degrees, 355/378 at 90, 350/378 at 123, and 348/378 at
180; the worst nonzero-angle prediction agreement with upright is 94.71%.

The augmentation-plus-canonicalization checkpoint SHA-256 is
`32659e40f9b26b3fd63bc25d3ad5bfb0293bc19f36648ecc2bd71afd0fbba639`;
it also completed 138 epochs. It scores 360/378 (95.24%) on clean Citizen validation
and 848/978 (86.71%) on SemLex validation. Its predictions are exactly invariant in
landmark space at all eight evaluated angles, with 360/378 top-1 and 100% prediction
agreement at each angle. Neither model replaces the prior 366/378 and 853/978 compact
clean-accuracy checkpoint on clean-domain evidence alone; raw-pixel roll stress is
running as the declared orientation selection gate. Training provenance for both
confirms Citizen train/validation plus approved SemLex train only, 50/50 source/class
balancing, seed 1701, and false Citizen/SemLex test-access flags.

## 2026-08-13 16:17 PST — orientation is a model/data contract, not an iPhone gate

The project owner clarified that deployment must work across aspect ratios and phone
orientation, including continuously rolled video rather than only portrait/landscape
categories. The earlier portrait-only collection gate was therefore incorrect as a
model requirement. The extractor evidence already supports this correction:
isotropic image coordinates, rotation-metadata handling, no aspect stretching, real
Vision rotation/mirror equivalence, and portrait/landscape letterbox tests all pass.
The capture pack now accepts all four iOS interface orientations, preserves native
aspect ratio, records the observed orientation/dimensions, and does not assign an
orientation to a class or repetition. The updated pack-manifest SHA-256 is
`fe406143a63eb576ec2e79b2aadd6dcb34a729a5db7950b5ab0a4631d93607f0`.

`active/v17/train_stage_1_v17.py` now exposes a missing-safe isotropic arbitrary-roll
transform and applies a declared mixture of continuous camera-roll augmentation. The
new default is 35% uniform full-circle roll (up to +/-180 degrees) and 65% mild roll
(+/-12 degrees); it never applies anisotropic feature scaling to imitate aspect ratio.
Provenance stores the exact probabilities/limits and the extractor aspect policy.
Tests explicitly cover invertibility at 17, 37, 123, and 180 degrees, full-circle
augmentation, missing zeros, portrait/landscape isotropic identity, metadata rotation,
and aspect-preserving RGB letterboxing. The affected 75-test suite passes, a real
Apple Vision test passes, Python compilation passes, and a two-batch CPU training
smoke completes with false Citizen/SemLex test-access flags. The training source
SHA-256 is `cac52af93542e6414d3898b538581ba2d65c94c0fd927e61cb46cf7e28f8dd3b`.

Private Kaggle code dataset `kokoab/slt-v17-stage1-orientation-code-v1` was created
with only the modified trainer and an explicit `test_data_included:false` manifest.
Private T4 job `kokoab/slt-v17-stage1-orientation-robust-v1` version 1 is RUNNING
against the previously verified train/validation-only bundle. It retains the selected
part-wise+global architecture, Citizen + exact-variant SemLex train data, 50/50
class/source-balanced sampling, seed 1701, and all prior optimization settings; the
only treatment is continuous roll augmentation. Runner SHA-256 is
`4d112358c604e2180dc88ac30ed410d1aa0114f9404ec7b13a39c5b77bdcc021`.
Neither frozen test split was accessed. The fixed four-stream teacher remains unchanged
at 370/378 Citizen and 882/978 SemLex while this robustness challenger is evaluated.

## 2026-08-13 15:55 PST — approved portrait pack built; candidates and decode gate frozen

The project owner explicitly approved all 100 pinned Citizen raw-gloss/ASL-LEX rows
in the active goal thread. The review ledger now records `review_status=approved`,
reviewer ID `project_owner`, UTC timestamp `2026-08-13T07:49:55Z`, and the exact
approval provenance on every row. Its new SHA-256 is
`58b3ee057d02bcc499d6f94f441798c552cc3d535b4f46e4c982f7c71911a92c`.
No raw gloss, class index, ASL-LEX code, entry ID, or reference link changed.

The real local capture pack is now built at
`data/local/portrait_iphone_eval_v17/` with pseudonyms S01-S05, seed 1701, two
independently randomized 100-class sessions per signer, and one 20-slot OOV session
per signer. The ledger has exactly 1,000 target plans plus 100 OOV plans and all 1,100
attempts are pending physical capture. The immutable ledger SHA-256 is
`9e9ae8b86357a0ec25920b333aa84f807f00a69e6fbed7fcf3309dad0d44e5d4`;
the pack-manifest SHA-256 is
`d7043677b0ead9e50a7096f353eed6010b04587a811b88dffb7979a37c95d6fc`.
The setup audit passes with zero errors at
`artifacts/reports/portrait_iphone_eval_v17_setup_audit.json` (SHA-256
`24bcbf355bc0cf4422a451d16656339e5eb63bcca90eff64aef5f377972b4bde`).
It reports 1,000 target plans, 100 OOV plans, 1,100 pending rows, and false test/model
access flags. It correctly reports `ready_for_first_inference:false` because capture
has not occurred.

`active/v17/portrait_iphone_candidates_v17.json` now freezes the exact six evaluation
checkpoints, fourteen runtime sources, three evidence reports, and the external
MobileCLIP2-S0 asset hash. Its SHA-256 is
`90342f7eaa80b239e18dec45be7288667852e1bbaf797f0d5a6cc9bb65dd85da`.
The fixed research teacher remains the flat-landmark/mouth/lower-face/hand composition
at 0.30/0.15/0.35/0.20 with development evidence 370/378 Citizen and 882/978 SemLex.
The compact standalone remains the part-wise+global landmark checkpoint at 366/378
and 853/978. The existing 75/25 landmark/hand fusion and the fixed-weight part-wise
teacher substitution are also pinned. The validator rejects any checkpoint, source,
evidence, member, or weight change and requires `allow_recalibration:false`.

The pre-inference audit now requires `ffprobe` plus full video-stream-only
`ffmpeg -xerror` decoding of every accepted file. It checks exact content hashes,
rotation-aware portrait dimensions, frame rate versus the ledger, and selects only
`0:v:0`; audio is never decoded. A real local MP4 smoke fully decoded 43 frames at
640x480/30 fps and reported `audio_accessed:false`. This was a tooling smoke, not a
portrait-set evaluation. Inference remains mechanically gated until every plan has
exactly one accepted attempt and all 1,100 videos pass full decode.

Ten portrait-workflow tests and the existing 43 focused Stage-1 tests pass together
(53/53). Both affected Python files compile and `git diff --check` passes. Current
script/test/guide SHA-256 hashes are
`8326339e6c1e5e62ce4069e06f2bb4f736625280ded1a0fef287488fb0bcabe9`,
`32065cc8494c41b9727ee372c1e5bb7cdc2a6e04b624ecb13d2f84bd08d85b13`,
and `3f474c8b5de9f83cd53600055e3bac081072821e40b31b163bcf2eb865679d29`.
No model inference, frozen test access, dataset deletion, Kaggle job, distillation,
or mobile benchmark occurred. The next irreducible action is physical capture of the
1,100 planned portrait-iPhone clips by the five genuinely new signers, followed by
objective ledger QC and the full pre-inference audit.

## 2026-08-13 15:46 PST — portrait-iPhone collection is executable and variant-gated

The independent portrait-iPhone protocol is now an executable, fail-closed collection
workflow rather than only prose plus an empty ledger header.
`scripts/build_portrait_iphone_eval_v17.py` has three explicit phases: generate the
100-row exact-variant review sheet, build reproducible capture schedules only after
all variants are approved, and audit either the untouched setup or the completed
pre-inference set. A valid pack contains at least five new pseudonymous signers,
two independently randomized 100-class repetitions per signer (1,000 target slots),
and the recommended 20 OOV slots per signer (100 OOV slots). It writes a 1,100-row
attempt-preserving ledger, per-session prompt schedules, and hashes of every source
and schedule. Recaptures append a numbered attempt instead of deleting an objective
failure.

The structural audit pins every class index, canonical label, exact Citizen raw gloss,
and ASL-LEX code; checks exact signer/repetition/class and OOV coverage; rejects changed
input/schedule hashes, unsafe paths, duplicate accepted paths/content hashes,
model-derived QC reasons, unresolved attempts, incomplete device metadata, nonportrait
declarations, prompts not confirmed hidden, and target rows whose performed gloss does
not exactly confirm the pinned variant. The pre-inference phase can report ready only
when exactly one objectively accepted attempt exists for every planned slot. It never
runs a model and records both model/test access as false.

The real review sheet is frozen at
`active/v17/portrait_iphone_variant_review_v17.csv` with exactly 100 pending rows and
SHA-256 `1a42ca6716305f5fdc3582e4b032554dd034a3687a102c12789fe5a6beef9d10`.
Each row links the exact local ASL-LEX entry but does not copy its reference video.
The official ASL-LEX license permits personal searches but prohibits saving,
displaying, or reusing reference videos without permission, so the workflow is
links-only (`https://asl-lex.org/download.html`). Capture is intentionally blocked
until an ASL-fluent reviewer marks every row approved with a pseudonymous reviewer ID
and timezone-aware timestamp. English-label agreement or a normalized/numeric variant
is not approval.

The expanded protocol and commands are in
`docs/guides/PORTRAIT_IPHONE_EVAL_V17.md`; the capture schema is in
`active/v17/portrait_iphone_capture_template.csv`. Six new workflow tests pass. The
existing 43 focused Stage-1 tests also pass unchanged, both new Python files compile,
and scoped `git diff --check` passes. Relevant SHA-256 hashes are script
`dc49e64c2ace80875fd0dc767cdbdeedf232aa4b3ca35084ffc217aab067dd1c`,
tests `34a5af3ff9c2e5f5e82d7812f28bcc4d69bd597ed51262ff2b4cc790f3978f31`,
guide `11cb804be7430ecbc81788b1d6ca82f682a1a360e4616c57382cfc918be5ddad`,
and ledger schema `0c960c82c96446efb9053a16b432d0b4c516296bc61cc7068986c316d505ad74`.

Kaggle remains reachable through the CLI, but no job was launched: this gate requires
genuinely new iPhone captures and human variant confirmation rather than more cloud
training. A narrow 2026 primary-source web check found no public replacement for the
capture: PopSign/PopSignAI still use the same one-handed Pixel-4A PopSign v1.0 corpus
(`https://openreview.net/forum?id=yEf8NSqTPu`,
`https://doi.org/10.1145/3742413.3789164`), FSboard is fingerspelling rather than the
100 isolated lexical variants (`https://arxiv.org/abs/2407.15806`), and ASL-100-RGBD
is Kinect capture rather than portrait iPhone
(`https://www.sign-lang.uni-hamburg.de/lrec/pub/20034.html`). None supplies the new
iPhone signers, two-handed coverage, and exact frozen variants required here.

Neither frozen test split, any model checkpoint, nor any existing dataset was accessed
or changed. The next safe action is the 100-row ASL-fluent review; after approval, run
`build-pack` with five genuinely new signer pseudonyms and immediately run the setup
audit before capture.

## 2026-08-12 05:38 PST — score mixing rejected; component ladder closed

Private Kaggle kernel `kokoab/slt-v17-stage1-attention-score-mix-v1` completed on a
Tesla T4. The 6,792,293-parameter model retained epoch 54 and early-stopped at epoch
84. Its eight initially zero score mixers learned an aggregate absolute weight of
36.54, so the treatment was active. Checkpoint SHA-256 is
`c223e73203417dadc6f07aa335549783cedb3f0cb6209a7c8a89ff96a1caf912`.

It scores 363/378 = 96.03% Citizen validation top-1, 376/378 = 99.47% top-5,
and 95.62% macro F1. SemLex validation is 845/978 = 86.40% top-1,
944/978 = 96.52% top-5, and 83.56% present-class macro F1. Relative to part-wise-only,
attention-score mixing loses three Citizen and eight SemLex top-1 clips while gaining
one Citizen and five SemLex top-5 clips. It fails both top-1 gates and is rejected;
do not test the paper's ambiguous four-layer version on the same validation sets.

The supervised and self-supervised component ladder is now closed. The only compact
landmark architecture that produced a replicated improvement on both top-1 domains is
feature-isolated part-wise temporal encoding followed by the global Squeezeformer.
Its supported seed-1701 checkpoint remains
`artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth`
(SHA-256 `5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b`),
at 366/378 Citizen and 853/978 SemLex. The fixed four-stream RGB/landmark teacher
remains the accuracy-oriented research option at 370/378 and 882/978, but it is not
the compact mobile default and its weights must not be retuned on these validation
sets. Further selection needs the independent portrait-iPhone set or genuinely new
compatible data, not more mining of the current validation errors.

Exact score-mixing reports are under
`artifacts/generated/kaggle_stage1_attention_score_mix_kokoab_result_v1/`,
`artifacts/reports/stage1_v17_attention_score_mix_v1_validation/`, and
`artifacts/reports/semlex_citizen100_val_audit/attention_score_mix_v1/`. Neither test
split was accessed and no heavy process is active.

## 2026-08-12 05:31 PST — one declared attention-score component staged

The higher-risk HTMA-inspired function is implemented without the paper's full CNN
or ambiguous four-convolution variant. Each of the eight part/global Squeezeformer
attention blocks optionally applies one independent depthwise 3x3 convolution to
each head's `T x T` pre-softmax score map. The score residual is zero-initialized,
and CPU RNG is restored after constructing the extra convolution so later baseline
weights retain their initialization sequence. The component starts numerically equal
to ordinary attention and adds only 576 parameters (6,792,293 total).

All 43 focused Stage-1 tests pass, including zero-score equivalence to the underlying
ordinary MHA and the exact 180-parameter delta for the small five-block test model.
A real two-batch/full-validation CPU smoke passed; all eight score mixers moved away
from zero and provenance kept test flags false. Source hashes are
`624e5d611dbf1bb25ee66c4b1af3b7dcccbd860dc675d7ba077c91f03dd816c4` and
`37121efa7ef7da7719daad2f45cf4da4e29f9993e9ea923d1ba472aa09dfa763`.
The staged private overlay archive under
`artifacts/generated/kaggle_stage1_attention_score_mix_overlay_v1/` has SHA-256
`c96ca041c4c55bb13957f572279219817244af4e57db75d0443cdfda511f6631`
and declares no test data. The CUDA-only runner is staged under
`artifacts/generated/kaggle_stage1_attention_score_mix_kokoab_v1/`. Private code
dataset `kokoab/slt-v17-stage1-attention-score-mix-code-v1` is ready and private
kernel `kokoab/slt-v17-stage1-attention-score-mix-v1` version 1 is RUNNING on a T4.
No local heavy process is active.

## 2026-08-12 05:26 PST — masked-pose reconstruction converges but hurts recognition

Private Kaggle kernel `kokoab/slt-v17-stage1-masked-pose-v1` completed on a Tesla T4.
Pretraining ran all 40 epochs / 1,800 steps and reduced the composite reconstruction
loss to 0.05712. It loaded 249 encoder tensors covering 6,591,808 parameters; the
temporary 78,385-parameter decoder was discarded. Pretraining checkpoint SHA-256 is
`6498048e4728ac1af36f01ba84ce061f07c49e7fbaf4c6a45f03c9d57e9d6fbf`.
Provenance confirms no validation or test access during pretraining.

The unchanged 6,791,717-parameter part-wise classifier retained fine-tune epoch 50
and early-stopped at epoch 80. Checkpoint SHA-256 is
`3aef02eaf4bbc3221e07e5a8ef64a7d4fa3613ac4ed3c455978a5d2ab1760b85`.
It scores only 360/378 = 95.24% Citizen validation top-1, 376/378 = 99.47% top-5,
and 94.79% macro F1. SemLex validation is 819/978 = 83.74% top-1,
931/978 = 95.19% top-5, and 80.64% present-class macro F1. Relative to the matched
part-wise control, this loses six Citizen and thirty-four SemLex top-1 clips. The
reconstruction objective therefore converged but learned a materially worse
discriminative initialization. Reject it and do not combine or tune its mask ratio on
the current validation sets. Revisit masked pretraining only with a substantially
larger v17-compatible unlabeled pool or a separately justified pretext objective.

Exact outputs are under
`artifacts/generated/kaggle_stage1_masked_pose_kokoab_result_v1/`,
`artifacts/reports/stage1_v17_masked_pose_finetune_v1_validation/`, and
`artifacts/reports/semlex_citizen100_val_audit/masked_pose_finetune_v1/`. Neither test
split was accessed and no heavy process is active.

## 2026-08-12 05:14 PST — zero-deployment-cost masked-pose trial staged

A SHuBERT/MS-MAE-inspired multi-stream masking component is now implemented without
their full models or external corpora. During pretraining, four-frame spans are
sampled independently for left hand, right hand, face, and body at nominal ratio
0.35. All five public v17 channels are hidden for the selected nodes; the unchanged
part-wise encoder reconstructs masked XYZ with Smooth-L1, presence with binary cross
entropy, and observed confidence with MSE. Only approved Citizen-train and SemLex-
train clips are loaded. Validation and both tests are absent from pretraining.

The temporary 78,385-parameter reconstruction decoder is discarded. Exactly 249
encoder tensors (6,591,808 parameters) are strict-loaded into the unchanged
6,791,717-parameter part-wise classifier before ordinary full fine-tuning. Thus the
existing seed-1701 part-wise run is the architecture/seed control and this treatment
adds zero inference parameters or preprocessing. The loader fails closed on model
config, schema, both manifest hashes, and the exact encoder-key set.

All 41 focused Stage-1 tests pass, including independent part-span coverage,
finite masked reconstruction, and strict encoder-only loading. A real-data two-step
pretraining smoke plus two-batch/full-validation downstream smoke passed; the latter
loaded all 249 encoder tensors, retained 6,791,717 parameters, and kept all validation
and test access flags false. Source hashes are
`015a84413d4c591b197ccbf88f1bd937ff77f461245d520455d64c8503f6d46f`,
`75ca0f66b5e632fc1a26ebdc4bb4c75621942734e38712f271e99dc5ee77aa90`, and
`f2b95df892c69ac357eaef9f78e427f3851c9e3edb356d5f155ea772c4274892`.
The staged private overlay archive under
`artifacts/generated/kaggle_stage1_masked_pose_overlay_v1/` has SHA-256
`efb9bbbbfbbfc0d54ababc067238172f78607e5671f6a07d015d1ae1b02ba559`;
its manifest declares no test data. The sequential pretrain/fine-tune runner under
`artifacts/generated/kaggle_stage1_masked_pose_kokoab_v1/` is now active. Private code
dataset `kokoab/slt-v17-stage1-masked-pose-code-v1` is ready and private kernel
`kokoab/slt-v17-stage1-masked-pose-v1` version 1 is RUNNING on a T4. No local heavy
process is active.

## 2026-08-12 05:09 PST — low-motion hand token rejected by quality control

Private Kaggle kernel `kokoab/slt-v17-stage1-static-hand-v1` completed both matched
6,825,254-parameter runs on a Tesla T4. The quality-only control retained epoch 104,
early-stopped at epoch 134, and learned residual scale 0.3251. Checkpoint SHA-256 is
`188562952ac6601c636317b3a7347b46c3d617f53809f12463c12b57f8ae40d2`.
It scores 367/378 = 97.09% Citizen validation top-1, 376/378 = 99.47% top-5,
and 96.70% macro F1. SemLex validation is 851/978 = 87.01% top-1,
939/978 = 96.01% top-5, and 84.26% present-class macro F1. Relative to part-wise,
this gains one Citizen top-1 and one top-5 clip but loses two SemLex top-1 clips and
0.18 macro-F1 points. It is a small mixed result, not a new supported winner.

The low-motion treatment retained epoch 117, early-stopped at epoch 147, and learned
residual scale 0.2863. Checkpoint SHA-256 is
`d59e6574cf7d98086e3dc20419ab6852d6cf3f398eedd441ef0cd1b3572b3648`.
It also scores 367/378 Citizen top-1 but only 374/378 top-5 and 96.86% macro F1.
SemLex falls to 840/978 = 85.89% top-1, 933/978 = 95.40% top-5, and 83.27% macro
F1. Against its exact quality control, low-motion loses eleven SemLex top-1 and six
top-5 clips while adding no Citizen top-1 clip. Reject the low-motion mechanism and
do not combine it. The supported compact landmark model remains part-wise-only at
366 Citizen / 853 SemLex because neither static branch improves both domains.

Exact artifacts are under
`artifacts/generated/kaggle_stage1_static_hand_kokoab_result_v1/`,
`artifacts/reports/stage1_v17_static_hand_*_v1_validation/`, and
`artifacts/reports/semlex_citizen100_val_audit/static_hand_*_v1/`. Neither test split
was accessed and no heavy process is active.

## 2026-08-12 04:45 PST — canonical-hand component ready as a matched pair

The next Handshape-GNN-inspired function is implemented off by default without
importing that paper's model or handshape labels. For each hand, it selects three
frames with at least 12 observed joints and pools the existing part-wise features.
The capacity control ranks frames only by mean landmark confidence; the treatment
uses the identical branch but also penalizes normalized inter-frame hand speed, with
a quality-only fallback when no reliable transition exists. Both add the same 33,537
parameters to the 6,791,717-parameter part-wise model (6,825,254 total).

The hand token enters through a zero-initialized scalar residual, so both challengers
are exactly equal to part-wise-only at initialization. This makes quality versus
low-motion the only treatment and prevents a new fusion projection from immediately
perturbing the supported backbone. All 39 focused Stage-1 tests pass, including exact
initial identity, selection behavior, absent-hand zeros, configuration isolation, and
the fixed mobile-size bound. Separate real-data two-batch/full-validation CPU smokes
passed for both modes; each residual scale moved away from zero and provenance kept
both test flags false.

The two source hashes are
`015a84413d4c591b197ccbf88f1bd937ff77f461245d520455d64c8503f6d46f`
and `4915025571521e27906c78d573f5d139de6a726947c63752fe3242d0fcb9b642`.
The staged private overlay archive under
`artifacts/generated/kaggle_stage1_static_hand_overlay_v1/` has SHA-256
`b9876e12fe19567ba577d8f99cb029ec67265aaa0f8464f99f210749ff2fb0af`
and declares no test data. The CUDA-only sequential quality/low-motion runner is under
`artifacts/generated/kaggle_stage1_static_hand_kokoab_v1/`. Private code dataset
`kokoab/slt-v17-stage1-static-hand-code-v1` is ready and private kernel
`kokoab/slt-v17-stage1-static-hand-v1` version 1 is QUEUED for a T4. No local heavy
process is active.

## 2026-08-12 04:42 PST — articulated-distance initialization rejected by its control

Private Kaggle kernel `kokoab/slt-v17-stage1-articulated-pose-v1` completed all three
stages on a Tesla T4. The 84,416-parameter geometry MLP saw 30,000 approved Citizen-
train and 30,000 approved SemLex-train frames, processed 199,597 triplets over 20
epochs, and ended at loss 0.01139. Its checkpoint SHA-256 is
`41e7f2e8c3dde10ad3c204142c9fb4e40be5fff4bf7b9a8d9c50ed9e06adad4d`.

The capacity-matched random branch retained epoch 111 and early-stopped at epoch 141.
Its checkpoint SHA-256 is
`d13df6845d1284a15d7212e4518fc7dbd50ead26538633ae822963ac74b70eba`.
It scores 366/378 = 96.83% Citizen validation top-1, 375/378 = 99.21% top-5,
and 96.56% macro F1; SemLex validation is 856/978 = 87.53% top-1,
940/978 = 96.11% top-5, and 84.79% present-class macro F1. Relative to the
6,791,717-parameter part-wise winner, the 6,958,821-parameter random branch ties both
Citizen top-1/top-5 and adds three SemLex top-1, one SemLex top-5, and 0.35 macro-F1
points. This is small cross-domain capacity signal but fails the primary improvement
gate and does not become the supported model.

The identical distance-pretrained branch retained epoch 53 and early-stopped at epoch
83. Its checkpoint SHA-256 is
`a58e1a8357b04e3267462db4e425f124722553f8c18d768c31601804e3ab12d9`.
It also scores 366/378 Citizen top-1, with 376/378 top-5 and 96.37% macro F1, but only
842/978 = 86.09% SemLex top-1, 937/978 = 95.81% top-5, and 83.41% macro F1. Thus the
paper-derived initialization loses fourteen SemLex clips and 1.37 macro-F1 points
against its exact random control while offering only one Citizen top-5 clip. Reject
the articulated-distance initialization and do not combine it. Exact outputs are
under `artifacts/generated/kaggle_stage1_articulated_pose_kokoab_result_v1/` and the
four validation reports under `artifacts/reports/`. Neither test split was accessed;
no heavy process is active.

## 2026-08-12 04:21 PST — articulated-distance component staged with required control

The next paper-derived trial is implemented as a component rather than a wholesale
model replacement. A missing-aware articulated hand distance uses length-weighted
bone-orientation differences to mine per-frame triplets from only the approved
Citizen and SemLex training splits. A 168-D wrist-relative hand vector is mapped by
an 84,416-parameter MLP to a normalized 64-D pose embedding and fused into the
replicated part-wise Squeezeformer. The downstream model has 6,958,821 parameters.

The comparison is explicitly capacity-matched: first train the added branch from
random initialization, then train the identical architecture with only that branch
initialized by the distance-preserving triplet task. Thus random versus part-wise
measures extra capacity, while pretrained versus random measures the paper-derived
geometry objective. Neither model receives test data, label-derived pretraining, or
validation feedback during pretraining.

All 37 focused Stage-1 tests pass. They cover translation invariance, missing-hand
masking, articulated distance behavior, forward/configuration isolation, and strict
branch-only preload. A real-data pretraining smoke and separate random/pretrained
two-batch downstream smokes passed with the exact 1,475 Citizen + 1,388 SemLex
training loaders and full Citizen validation. The three source hashes are
`2c6ce33509f22a45b4c5485952e233cf369986f8bee9aace634a8d3bdcef31a6`,
`fe6bddcc02bd1a9201b4c574bc14e479fc8a4fcc8fbb86f12f3829dc4212def7`,
and `00d4fb065fe168354ef85391e46e5e8ecaa7c2ec71ac073d459c5d54da532949`.

The private Kaggle overlay is staged under
`artifacts/generated/kaggle_stage1_articulated_pose_overlay_v1/`; archive SHA-256 is
`44b078f47621987dfdb9a0319f30c030211c6ac463ef8d1787ccd9330440f4cc`.
Its manifest declares no test data. The CUDA-only runner under
`artifacts/generated/kaggle_stage1_articulated_pose_kokoab_v1/` will execute geometry
pretraining, the random capacity control, and the pretrained treatment sequentially
on one T4 while preserving the same seed, data, sampler, optimizer, schedule, and
patience. Private code dataset
`kokoab/slt-v17-stage1-articulated-pose-code-v1` is ready, and private kernel
`kokoab/slt-v17-stage1-articulated-pose-v1` version 1 is RUNNING. No local heavy
process is active. The exhaustive audit's predeclared ladder now records the rejected
temporal gate and the running articulated-distance/control comparison before any
static-hand, masked-pose, or visual-symbol experiment.

## 2026-08-12 04:13 PST — temporal gate rejected on both domains

The isolated per-keypoint temporal-gate run completed on a Tesla T4, retained epoch
51, and early-stopped after 81 epochs. Its 6,792,449-parameter checkpoint SHA-256 is
`18bcf98bdb53995b7b960b8ecb73aac3f17b3593e89985342a5166b88ff346e0`.
It scores 364/378 = 96.30% Citizen validation top-1, 376/378 = 99.47% top-5,
and 95.89% macro F1. The fixed SemLex diagnostic scores 848/978 = 86.71% top-1,
943/978 = 96.42% top-5, and 83.54% present-class macro F1.

Part-wise-only scores 366/378 Citizen, 375/378 top-5, and 853/978 SemLex with
939/978 top-5 and 84.44% macro F1. The gate gains one Citizen top-5 clip and four
SemLex top-5 clips but loses two and five top-1 clips respectively and 0.90 SemLex
macro-F1 points. It fails the both-domain gate, is rejected as a winner, and will not
be combined. Exact outputs are under
`artifacts/generated/kaggle_stage1_temporal_gate_kokoab_result_v1/`,
`artifacts/reports/stage1_v17_temporal_gate_v1_validation/`, and
`artifacts/reports/semlex_citizen100_val_audit/temporal_gate_v1/`. No test data was
accessed and no heavy process is active.

## 2026-08-12 04:03 PST — part-wise gain replicated; temporal gate launched

The controlled seed-3407 part-wise replication completed on a Tesla T4, retained
epoch 33, and early-stopped after 63 epochs. Its 6,791,717-parameter checkpoint
SHA-256 is `4dab34bec5fc72684f596f775acfec638312ac88a7cfd3dd29d55084a1515447`.
It scores 362/378 = 95.77% Citizen validation top-1, 375/378 = 99.21% top-5,
and 95.38% macro F1. The fixed SemLex validation diagnostic scores 845/978 =
86.40% top-1, 946/978 = 96.73% top-5, and 84.01% present-class macro F1.

Against the matched seed-3407 flat baseline (358 Citizen, 832 SemLex), part-wise adds
exactly four Citizen and thirteen SemLex clips. This repeats the seed-1701 gains
exactly: 366 versus 362 Citizen and 853 versus 840 SemLex. Part-wise temporal
isolation is therefore a replicated architecture effect and remains the supported
compact landmark backbone; the seed-1701 checkpoint remains the best single run.
Exact replication reports are under
`artifacts/reports/stage1_v17_partwise_seed3407_v2_validation/` and
`artifacts/reports/semlex_citizen100_val_audit/partwise_seed3407_v2/`.

A SKIM-inspired per-keypoint temporal gate is now implemented off by default. It uses
independent depthwise temporal filters over each node's confidence, speed, and
acceleration, and a zero-initialized output makes the initial gate exactly identity.
It adds 732 parameters (6,792,449 total with part-wise), preserves the public
`[32,61,5]` input, and is restricted to an isolated part-wise ablation. All 34 focused
tests pass, and a real two-batch CPU train/checkpoint/full-validation smoke passed on
the exact 1,475 Citizen + 1,388 SemLex training loaders.

Private code dataset `kokoab/slt-v17-stage1-temporal-gate-code-v1` contains only the
two hash-locked source overlays; archive SHA-256 is
`6e128810bf0651d4b6ade9b3c11c3aee17e5a0dc39b6a4163f899bcf3d6a1d02` and it declares
no test data. Private kernel `kokoab/slt-v17-stage1-temporal-gate-v1` version 1 is now
running on a Tesla T4 with seed 1701 and the otherwise unchanged controlled protocol.
No test data was accessed and no local heavy process is active.

The 66,409 legacy files under `data/local/ASL_landmarks_apple_vision/` were also
checked as a possible masked-pretraining pool. They are `[32,61,10]` outputs from the
older extractor: XYZ plus precomputed Savitzky-Golay velocity/acceleration and a
part-level `ever observed` mask. A real sample confirms channel 9 is that mask while
channels 3-8 are derivatives, not v17 presence/confidence. The older pipeline also
interpolates missing frames, uses different normalization, and predates the v17
chirality/schema corrections. These files cannot be sliced or relabeled into the
v17 `[32,61,5]` contract. They may support a separate legacy pretraining study only
after an explicit adapter ablation, or their raw videos may be selectively re-extracted
with v17; they will not be silently mixed into the current model.

A component-level literature pass added one higher-cost geometry candidate from
Sartinas et al. (VISAPP 2026): pretrain a small per-frame MLP to preserve neighborhoods
under a hierarchical articulated bone-orientation distance, then concatenate only its
64-D embedding with the temporal input. Their five-run ablation shows that a random
extra branch explains much of the headline gain, so any v17 trial must include a
capacity-matched random-branch control. This is not selected ahead of the running gate;
it is a later alternative to generic masked reconstruction because it targets the
positive bone signal while differing from the rejected raw angle channel.

A second component-only review covered Varanasi et al.'s CVPRW 2026 HTMA block. The
portable function is a small 2D convolution over each attention head's `T x T` score
map before softmax, not their full 1D-CNN/MediaPipe model. It remains higher-risk:
their attention ablation uses INCLUDE's original split rather than the described
pseudo-signer grouping, and the algorithm claims four score-mixing convolutions while
the hyperparameter table says one. Retain one explicitly declared score-convolution
variant as a later Squeezeformer-attention ablation; do not treat the paper's desktop
latency or nominal mobile claim as iPhone evidence.

## 2026-08-12 03:55 PST — same-seed part-wise replication launched

Private Kaggle kernel `kokoab/slt-v17-stage1-partwise-seed3407-v2` version 1 is now
running on a Tesla T4. The runner compiled locally before launch and fail-closes on
the exact training archive/tree hashes and any test artifact. Its only treatment
relative to the existing seed-3407 flat baseline is `partwise_global` temporal
encoding with part depth 1; data, balanced sampler, optimizer schedule, maximum 160
epochs, patience 30, and seed 3407 are otherwise fixed. The predeclared comparison is
358/378 Citizen validation and 832/978 SemLex validation. No test data was accessed
and no local heavy process is active.

## 2026-08-12 03:54 PST — hand angles rejected; part-wise replication staged

The isolated flat hand-angle run completed on a Tesla T4, retained epoch 86, and
early-stopped after epoch 116 / 30 stale epochs. Its 6,486,501-parameter checkpoint
SHA-256 is `3c8d5a82ea7df1f1015fab4ae3665b9c5ee1bf2e7c085c85bc24f98fedd5ee97`.
It scores 364/378 = 96.30% Citizen validation top-1, 375/378 = 99.21% top-5, and
95.70% macro F1. On SemLex validation it scores 838/978 = 85.69% top-1,
933/978 = 95.40% top-5, and 82.96% present-class macro F1. Relative to flat, angles
gain two Citizen top-1 clips but lose two SemLex top-1 and seven SemLex top-5 clips.
They fail the both-domain gate and will not be combined with part-wise. Exact results
are under `artifacts/generated/kaggle_stage1_hand_angle_kokoab_result_v1/` and
`artifacts/reports/semlex_citizen100_val_audit/hand_angle_v1/`.

The cheap supervised component ladder therefore leaves part-wise-only as the best
single development architecture. A seed-3407 replication kernel is staged under
`artifacts/generated/kaggle_stage1_partwise_seed3407_kokoab_v2/` with the exact same
data/protocol and only the seed changed. It will be compared to the already measured
same-seed flat baseline of 358/378 Citizen and 832/978 SemLex. No test data was
accessed.

## 2026-08-12 03:45 PST — second-seed flat cross-domain baseline established

The existing clean flat seed-3407 checkpoint (358/378 = 94.71% Citizen validation)
was evaluated once on the fixed SemLex validation diagnostic for the forthcoming
architecture replication. It scores 832/978 = 85.07% top-1, 934/978 = 95.50% top-5,
and 82.47% present-class macro F1. The seed-1701 flat reference is 362/378 Citizen and
840/978 SemLex. A seed-3407 part-wise run can therefore be compared against its own
same-seed flat baseline rather than only the selected seed-1701 winner. Exact output
is under `artifacts/reports/semlex_citizen100_val_audit/d256_full_clean_seed3407/`.
No test data was accessed; the hand-angle kernel remains RUNNING.

## 2026-08-12 03:44 PST — per-part supervision mixed; angle trial launched

The corrected per-part auxiliary run completed on a Tesla T4, retained epoch 65, and
early-stopped after epoch 95 / 30 stale epochs. Its 6,818,229-parameter training
checkpoint SHA-256 is
`c1778e5952c71dd0c54f2a6706fedc1ea6398686da9481b1e0373e0448f8a534`.
Provenance confirms the missing-part skip policy, fixed total auxiliary weight 0.20,
unchanged data/sampler/seed, and false test-access flags.

It scores 365/378 = 96.56% Citizen validation top-1, 377/378 = 99.74% top-5, and
96.32% macro F1. The fixed SemLex diagnostic scores 861/978 = 88.04% top-1,
943/978 = 96.42% top-5, and 85.85% present-class macro F1. Relative to part-wise-only,
this loses one Citizen top-1 clip but gains eight SemLex top-1, four SemLex top-5, and
1.41 macro-F1 points. It therefore improves cross-domain robustness but fails the
primary Citizen gate and does not replace the 366/378 part-wise winner. Exact results
are under `artifacts/generated/kaggle_stage1_partaux_w020_kokoab_result_v2/` and
`artifacts/reports/semlex_citizen100_val_audit/partaux_w020_v1/`.

The copied writable Kaggle tree caused the first unfiltered result pull to start
downloading training artifacts; a file-pattern-limited CLI pull retrieved only the
five intended result/log files. The cloud checkpoint was intact; the earlier local
zero-byte partial is not evidence. Future overlay runners should avoid leaving the
copied base tree under `/kaggle/working` at exit or always use filtered pulls.

Private kernel `kokoab/slt-v17-stage1-hand-angle-v1` is now RUNNING. It uses the
unchanged flat d=256/depth=4 Squeezeformer and makes the 30 missing-aware cosine
finger-flexion values its only treatment; part-wise, bone, PartMix, phonology,
contrastive, and auxiliary losses are disabled. No local heavy process or test data
access is active.

## 2026-08-12 03:39 PST — absent-part audit confirms auxiliary masking is required

A complete read of the exact 1,475 Citizen + 1,388 SemLex training features found
587/523 clips with no observed left-hand node, 118/88 with no observed right-hand
node, and 82/159 with no observed body node; every clip has at least one face anchor.
Thus about 39% of training clips lack at least one hand stream, usually because the
sign is genuinely one-handed or that hand was not detected. Training every auxiliary
head on every sample would create a large impossible-label objective from all-zero
inputs. The running v2 kernel correctly skips only those absent part/sample pairs
while retaining global classification loss for every clip. No labels, files, or test
data were changed.

## 2026-08-12 03:35 PST — isolated hand-angle trial staged

Private 22.4 KiB code dataset `kokoab/slt-v17-stage1-hand-angle-code-v1` is ready;
its archive SHA-256 is
`79bf8f102835dde168485538061c3f6437965a127cf8505a2fa706b2cdcfa004`.
It pins only the tested model/trainer sources and the exact angle definition, with no
training data or test artifacts. A fail-closed global-Squeezeformer kernel is staged
under `artifacts/generated/kaggle_stage1_hand_angle_kokoab_v1/`; it will reuse and
reverify the frozen base tree, apply the two-file overlay in the writable ephemeral
volume, and run angles as the only treatment. It has not launched while the corrected
per-part auxiliary kernel is RUNNING. No local heavy process or test data access is
active.

## 2026-08-12 03:32 PST — auxiliary v2 running; hand-angle component ready

Per-part kernel version 1 passed CUDA, the complete base-tree digest, and both overlay
source hashes, then failed before training because the verified base tree was still on
Kaggle's read-only input mount. It produced no checkpoint or metric. Version 2 now
copies that already verified 41 MiB compact tree to the ephemeral working volume,
rechecks the full tree digest there, applies and rechecks only the two pinned overlay
files, and is RUNNING. This is a transport fix only; the experiment and data are
unchanged.

The next paper-derived component is implemented off by default as
`--hand-angle-features`. It derives cosine flexion angles at the three internal joints
of each of five fingers on both hands (30 angles total), zeroing a value unless its
parent, center, and child are all observed. Cosines avoid `acos` instability and add a
scale/translation/rotation-invariant handshape cue without changing the public Apple
`[32,61,5]` archive. The flat model grows by only 15,616 parameters, from 6,470,885 to
6,486,501; the part-wise model grows by 3,904, from 6,791,717 to 6,795,621. All 32
focused tests pass and a real two-batch Citizen+SemLex optimizer/checkpoint smoke
passed. This component is motivated by the Handshape-GNN joint-angle analysis and
SignRep's angle/keypoint/distance prior ablation, not by importing either full model.
It has no full accuracy result and must wait for the running auxiliary trial. No test
data was accessed.

## 2026-08-12 03:31 PST — untuned landmark pair confirms complementary errors

Aligned Citizen logits for the part-wise+bone model reproduce its Kaggle metrics
exactly. A single predeclared, untuned 50/50 per-sample-z-scored ensemble of
part-wise-only and part-wise+bone scores 367/378 = 97.09% Citizen validation top-1,
376/378 = 99.47% top-5, and 96.79% macro F1. On SemLex validation it scores
868/978 = 88.75% top-1, 949/978 = 97.03% top-5, and 84.27% 100-class macro F1.
Relative to part-wise-only this is +1 Citizen and +15 SemLex top-1 clips, proving the
bone model has complementary errors despite its weaker Citizen standalone result.

The pair is an accuracy-oriented research option, not the mobile default: it runs two
6.8M-class landmark networks and still trails the existing four-stream teacher at
370/378 Citizen and 882/978 SemLex. No weights were searched, and no additional
combinations should be mined on these validation sets. Exact reports are under
`artifacts/reports/stage1_v17_partwise_bone_equal_ensemble_validation/` and
`artifacts/reports/semlex_citizen100_val_audit/partwise_bone_equal_ensemble/`. No test
data was accessed; the missing-aware per-part kernel remains RUNNING.

## 2026-08-12 03:29 PST — missing-aware per-part supervision launched

Private kernel `kokoab/slt-v17-stage1-part-auxiliary-w020-v1` is RUNNING. Kaggle
canonicalized the title-derived slug by spelling out `part-auxiliary`; local metadata
now matches the server identity. The runner verifies the unchanged base training tree,
then the ready v2 overlay and both corrected source hashes, and runs the 6,818,229-
parameter part-wise architecture with a fixed total auxiliary weight of 0.20. Each of
the left-hand, right-hand, face, and body heads receives gloss supervision only for
clips where that part has at least one observed node. Bone, PartMix, phonology, and
contrastive objectives are disabled. The auxiliary heads are training-only and normal
inference uses the same 6,791,717-parameter path as part-wise-only. No local heavy
process or test data access is active.

## 2026-08-12 03:28 PST — bone interaction improves SemLex but fails primary gate

The part-wise+bone Kaggle run completed on a Tesla T4, retained epoch 71, and
early-stopped after epoch 101 / 30 stale epochs. Its 6,815,141-parameter checkpoint
SHA-256 is `b5a1a808f3de9cd74e081e816abf1d737cbe448f46d12ea11a4f039e7624f7bf`.
All data/objective/test-isolation provenance matches the declared interaction.

It scores 364/378 = 96.30% Citizen validation top-1, 376/378 = 99.47% top-5, and
96.01% macro F1. On SemLex validation it scores 859/978 = 87.83% top-1,
949/978 = 97.03% top-5, and 85.18% present-class macro F1. Relative to part-wise-only,
bone gains six SemLex top-1 clips and ten SemLex top-5 clips but loses two Citizen
top-1 clips and one Citizen top-5 clip. It therefore fails the predeclared rule that
the interaction must exceed part-wise on the primary Citizen gate and does not replace
the 366/378 part-wise winner. The tradeoff is still useful evidence that bone features
improve cross-domain robustness and may be valuable after independent calibration;
do not choose it solely from SemLex.

Exact artifacts are under
`artifacts/generated/kaggle_stage1_partwise_bone_kokoab_pull_v1/`; the diagnostic is
under `artifacts/reports/semlex_citizen100_val_audit/partwise_bone_v2/`. The corrected
per-part auxiliary objective now skips a part/sample loss when that anatomical stream
has no observed node, preventing legitimate one-handed clips from being punished for
an absent nondominant hand. Normal inference logits remain bit-identical, all 30 tests
pass, and a new real two-batch smoke passed. The ready v2 code overlay SHA-256 is
`2da438ffc49bdf47843802454ef4c25a392f93cda89c85e3bc70d9bfc997e49e`;
the obsolete v1 overlay must not be launched. No test data was accessed.

## 2026-08-12 03:20 PST — two more paper components triaged, not blindly adopted

The literature audit now includes a 2025 arXiv dual-reference model and a 2024
LREC-COLING keypoint-importance study. The portable dual-reference idea is to represent
hand morphology relative to each wrist while retaining wrist trajectory relative to
the body/face. v17 already retains body-relative trajectory and exposes
translation-invariant pairwise hand distances; the positive bone experiment covers
much of the missing local-geometry hypothesis. A direct wrist-relative-coordinate
stream is therefore a later cheap ablation if the current part-wise/bone ladder
plateaus, not a reason to import the paper's 46.3M graph/LSTM/optimal-transport model.

The feature-importance study reports that outer-finger bases/tips dominate while
inner finger joints and coarse face points are often under-used, with occlusion,
missing depth, data imbalance, and insufficient facial detail as plausible causes. Its
own warning is important: low model importance does not imply low linguistic value.
The project will use this only for diagnostics and quality-aware training hypotheses;
it will not delete joints or add dense face landmarks without a measured extractor
bake-off. These decisions and primary links are recorded in the exhaustive audit. The
part-wise+bone Kaggle kernel remains RUNNING; no test data was accessed.

## 2026-08-12 03:18 PST — part-wise plus bone interaction launched

Private kernel `kokoab/slt-v17-stage1-partwise-bone-v2` version 1 is RUNNING on the
same fail-closed CUDA/data-integrity path. The model has 6,815,141 parameters and
combines exactly the two independently positive components: one isolated temporal
layer per left hand/right hand/face/body before the depth-4 global Squeezeformer, plus
internally derived bone vectors/bone motion. Data, seed, sampler, optimizer, schedule,
patience, and classification loss are unchanged; PartMix, phonology, contrastive, and
per-part auxiliary objectives are disabled. This run must beat the part-wise-only
366/378 Citizen and 853/978 SemLex top-1 results to replace it. No heavy local process
or test data access is active.

## 2026-08-12 03:17 PST — bone representation passes both isolated top-1 gates

The isolated bone-only Kaggle run completed on a Tesla T4, retained epoch 50, and
early-stopped after epoch 80 / 30 stale epochs. Its 6,564,581-parameter checkpoint
SHA-256 is `e253b00dd23670960e43a39accc090bf61da1a6a403fe3f47570f4ce58deb4ff`.
Provenance confirms the exact unchanged data/sampler/seed/global architecture, with
bone vectors and bone motion as the only treatment and false test-access flags.

It scores 363/378 = 96.03% Citizen validation top-1, 377/378 = 99.74% top-5,
and 95.56% macro F1. The fixed SemLex diagnostic scores 845/978 = 86.40% top-1,
943/978 = 96.42% top-5, and 84.02% present-class macro F1. Relative to the flat
baseline this is +1 Citizen top-1 clip, +5 SemLex top-1 clips, and +3 SemLex top-5
clips. Bone representation therefore supplies genuine but smaller independent signal;
it does not replace the 366/378 Citizen, 853/978 SemLex part-wise winner.

Because part isolation and bone representation each improved both primary top-1
domains in separate runs, a part-wise+bone interaction is now scientifically
interpretable. Run that combination without PartMix or auxiliary losses before the
already staged per-part-supervision trial. Exact bone artifacts are under
`artifacts/generated/kaggle_stage1_bone_kokoab_pull_v1/`; its SemLex report is under
`artifacts/reports/semlex_citizen100_val_audit/bone_v2/`. No test data was accessed.

## 2026-08-12 03:18 PST — per-part follow-up transport staged without data reupload

The 21.6 KiB private Kaggle dataset `kokoab/slt-v17-stage1-partaux-code-v1` is ready.
It contains only the two updated v17 source files and a manifest—no landmarks, raw
video, validation predictions, or test artifacts. The archive SHA-256 is
`3839501a323b5dd2ca48a0198c317fd0b34bd7c6ab55374066dd41869a91c403`;
the manifest pins `model_v17.py` to
`045564493d189e298b2dda5571a05f2109b19d3cd5afcc0e7e57939d43735aec`
and `train_stage_1_v17.py` to
`df8e259296871c2814e2fe3424f0d371e688ea6f040148a9cb01fb6aa8454894`.

A CUDA-only kernel is staged as
`artifacts/generated/kaggle_stage1_partaux_w020_kokoab_v1/`. Its runner first verifies
the unchanged 3,253-file base training tree, then locates exactly one complete overlay,
checks both source hashes and the no-test manifest, copies only those sources into the
ephemeral Kaggle working tree, and runs part-wise + global training with fixed
per-part auxiliary weight 0.20. It is not launched while the isolated bone kernel is
RUNNING. This avoids uploading the 34 MiB training corpus again and keeps the next
treatment auditable. No local heavy process or test data access occurred.

## 2026-08-12 03:15 PST — fixed multimodal substitution does not inherit the gain

The new part-wise checkpoint's Citizen validation metrics were reproduced locally
exactly and aligned logits were saved under
`artifacts/reports/stage1_v17_partwise_v2_validation/`. Substituting those logits for
the flat landmark member while keeping the previously frozen 0.30 landmark / 0.15
mouth / 0.35 lower-face / 0.20 hand weights produces 368/378 = 97.35% Citizen
validation and 879/978 = 89.88% SemLex validation. The existing fixed flat-landmark
teacher remains better at 370/378 = 97.88% and 882/978 = 90.18% respectively.

This is not a contradiction: the part-wise model is a better standalone classifier,
but its errors and score calibration are less complementary to weights selected with
the old flat member. The current winners therefore remain separate: part-wise for the
single landmark branch, and the old four-stream composition for the multimodal
research teacher. Do not retune weights on SemLex, and do not repeatedly optimize
Citizen weights to manufacture a higher validation number. An independent portrait
set is required before choosing or recalibrating a production ensemble. Exact fixed-
weight reports are under
`artifacts/reports/stage1_v17_partwise_multimodal_teacher_fixed_validation/` and
`artifacts/reports/semlex_citizen100_val_audit/fixed_partwise_teacher_30_15_35_20/`.
No test data was accessed.

## 2026-08-12 03:13 PST — isolated bone-feature challenger launched

Private CUDA kernel `kokoab/slt-v17-stage1-bone-v2` version 1 is RUNNING. It uses the
same verified 3,253-file Citizen/SemLex train/validation tree, seed, sampling,
optimizer, schedule, patience, and unchanged global d=256/depth=4 Squeezeformer as
the superseded flat baseline. Its only treatment is internally derived missing-aware
hand/arm bone vectors plus bone motion (`--bone-features`); PartMix, part-wise temporal
encoding, phonology, contrastive learning, and per-part auxiliary supervision are all
disabled. The runner requires CUDA and rejects any link/path/test artifact before
training. This isolated run determines whether bone representation itself helps; it
does not yet test bone combined with the new part-wise winner. No local heavy process
or test data access is active.

## 2026-08-12 03:12 PST — part-wise encoder becomes the development winner

The isolated part-wise Kaggle trial completed successfully on a Tesla T4. It retained
epoch 100 and early-stopped after epoch 130 / 30 stale epochs. The checkpoint SHA-256
is `5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b`.
Provenance confirms the unchanged 1,475 Citizen + 1,388 SemLex training set,
50/50 class/source-balanced replacement sampling, seed 1701, no PartMix, no auxiliary
loss, 6,791,717 parameters, CUDA, and false Citizen/SemLex test-access flags.

The checkpoint scores 366/378 = **96.83%** Citizen validation top-1, 375/378 =
99.21% top-5, and 96.61% macro F1. The former flat winner scored 362/378 = 95.77%,
378/378 = 100% top-5, and 95.51% macro F1. The fixed SemLex-validation diagnostic
scores 853/978 = **87.22%** top-1, 939/978 = 96.01% top-5, and 84.44% present-class
macro F1, versus the flat winner's 840/978 = 85.89%, 940/978 = 96.11%, and 82.60%.
Thus feature-isolated left-hand/right-hand/face/body temporal modeling adds four
Citizen top-1 clips and thirteen SemLex top-1 clips, with three Citizen and one SemLex
additional top-5 misses. This satisfies the predeclared primary/cross-domain top-1
gate and is the new **development landmark winner**. It has not been evaluated on
either test split and is not yet an independently confirmed production model.

Exact pulled artifacts are under
`artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/`; the SemLex report is
under `artifacts/reports/semlex_citizen100_val_audit/partwise_v2/`. The result supports
the diagnosis that early whole-body flattening, not Squeezeformer's global temporal
stack itself, was a real bottleneck.

The literature-derived per-part auxiliary path is also implemented off by default.
When enabled, four training-only gloss heads supervise the isolated left hand, right
hand, face, and body streams before global fusion. Ordinary inference remains the
same fused classifier; the auxiliary heads are unused and can be pruned at export.
The full d=256 training graph adds only 26,512 parameters (6,818,229 total versus
6,791,717), all 30 focused Stage-1 tests pass, and a real two-batch Citizen + SemLex
CPU optimizer/checkpoint smoke passed at fixed auxiliary weight 0.20. This has not had
a full accuracy run. The bone-only challenger remains next so each component's effect
is established before combination. No heavy local process or test data access
occurred.

## 2026-08-12 03:08 PST — component-level literature ladder refined

The user's instruction is to mine useful functions from papers rather than treating
published models as all-or-nothing replacements. The controlled policy is now
explicit: transplant the smallest v17-compatible mechanism, test it in isolation on
the same Citizen-development/SemLex-diagnostic protocol, and retain it only if the
measured cross-domain tradeoff is favorable. No benchmark headline is treated as a
project result.

Two additional primary studies expose relevant components. A 2025 signer-independent
pose study found region decomposition alone gave only a modest gain, while training-
only gloss decoders attached to each hand/lip/torso stream produced the major ablation
gain. This motivates four tiny **per-part auxiliary gloss heads** on the existing
feature-isolated v17 encoder; their logits need not be computed or shipped at mobile
inference. A separate EMNLP 2025 Handshape-GNN study combines full hand dynamics with
a low-motion representative static frame. Its 37-handshape PopSign result and
handshape labels are not directly transferable, but a missing-aware static canonical-
hand token is a legitimate later component. It differs from the rejected global
ASL-LEX phonology objective because it changes the hand-specific representation, not
merely the label penalty on the final pooled token.

The exact updated order is: finish the running isolated part-wise trial; run the
already implemented isolated bone trial; then test per-part supervision; only then
consider a static-hand branch, gated joint/bone interaction, visual-symbol grouping,
or masked-pose pretraining. A CUDA-only bone kernel is staged locally as
`artifacts/generated/kaggle_stage1_bone_kokoab_v2/` against the same verified private
3,253-file train/validation tree, but it has not been pushed while the part-wise GPU
job is active. The part-wise kernel remains RUNNING. No heavy local process or test
data access occurred. The exhaustive audit records the component mapping and primary
paper links.

## 2026-08-12 02:27 PST — sign-specific PartMix challenger launched on Kaggle

The current d=256/depth=4 Squeezeformer remains the best model measured in this
project, not a proof of global architectural optimality. A primary-literature audit
identified three materially distinct skeleton-SLR directions that the rejected v17
graph experiment did not test: SKIM's corresponding-part mixing, P3D/Siformer's
part-wise temporal encoding, and DSTA-SLR's joint/bone/motion streams. The old graph
challenger only applied a per-frame anatomical graph before global temporal modeling.
Siformer's official MIT repository was cloned read-only at commit
`979a14ed15ed0f20afd77d447ad23c4f4107a2c3`; it uses separate left-hand,
right-hand, and body temporal encoders before fusion. DSTA-SLR's official repository
was cloned read-only at commit `e7e5ee225511488039a4f68ab3a146cf8f01312d`;
its released implementation assumes 27 joints/120 frames/CUDA and ensembles four
independently trained joint, bone, joint-motion, and bone-motion streams, so neither
its code nor checkpoint can be silently attached to the fixed Apple v17 contract.
No external weights or datasets were downloaded and no test data was accessed.

The first bounded challenger is SKIM-style one-hand PartMix because it changes only
training and adds zero inference parameters, latency, or mobile preprocessing.
`train_stage_1_v17.py` now optionally replaces exactly one complete 21-node left or
right hand with a guaranteed non-self batch donor and trains that sample with fixed
50/50 primary/donor labels. The default probability is zero, preserving the prior
training path. PartMix is deliberately rejected when contrastive or phonology losses
are enabled so this remains an isolated comparison. Provenance and per-epoch realized
mix fraction are saved. Two focused unit tests cover exact whole-hand replacement,
non-self labels, untouched face/body, missing-value preservation, the zero-probability
identity, and mixed cross-entropy. All 24 Stage-1 tests pass, the real 1,475 Citizen +
1,388 SemLex loader passed a two-batch CPU optimizer/checkpoint smoke at probability
0.5, affected scripts compile, and scoped `git diff --check` passes.

The authenticated Kaggle CLI uploaded a private 27.5 MiB archive as dataset
`francisbatiancela/slt-v17-stage1-partmix-trainval-v1`. Its SHA-256 is
`2c257e0bbc8fc5e198b445edbc472f7b44e9959e3ec60f8d08baf21bb16e9321`;
the archive contains the exact training code, 1,476 Citizen-train archives (1,475
usable after the frozen rejection ledger), 378 Citizen-validation archives, and all
1,388 approved SemLex-train archives. A full member audit found no `/test/` entry.
Private kernel version 1 at
`francisbatiancela/slt-v17-stage-1-partmix-p50` was launched. It verifies the archive
hash, refuses links/path traversal/test members, requires a real CUDA allocation, and
then runs the unchanged d=256/depth=4, seed-1701, 50/50 class/source-balanced protocol
with PartMix probability 0.5 and patience 30. If Kaggle again assigns CPU, it will
fail immediately rather than perform heavy local or mislabeled training.

Kaggle assigned that kernel its CPU-only PyTorch 2.10.0 image despite preserving
`enable_gpu:true` and `machine_shape:NvidiaTeslaT4` in the server-side metadata. The
runner therefore failed closed after 4.8 seconds with zero training batches. The CLI
is authenticated as `francisbatiancela`, whereas the user-supplied private notebook
`kokoab/notebook697ebe7d11` returns `403 kernels.get`; these are distinct Kaggle
accounts/sessions. The official forced OAuth refresh has been opened and is waiting
for browser approval so the CLI can authenticate to the already-open `kokoab`
session. Do not claim a Kaggle GPU run until the refreshed identity and CUDA device
are explicitly verified.

## 2026-08-12 02:34 PST — feature-isolated temporal challenger implemented

OAuth remains unapproved after five minutes and the CLI identity is still
`francisbatiancela`, so no second Kaggle launch was attempted. Work continued on the
next independent architecture rather than heating the laptop with a full run.

The Siformer/P3D evidence is now represented by a controlled v17-native challenger,
not by copying an incompatible upstream model. `Stage1V17Config` and the Stage-1 CLI
accept `temporal_encoder=partwise_global` and `part_depth`. The challenger projects
left hand, right hand, face, and body independently, runs a separate temporal
Squeezeformer on each stream, fuses them, and only then applies the unchanged global
Squeezeformer. Pairwise hand distances are routed only to their matching hand stream.
This directly tests feature-isolated part-wise temporal context; unlike the rejected
graph model, it cannot mix anatomy before each part has been modeled across time.
The default remains the original global architecture.

At d=256/depth=4/part-depth=1 the challenger has 6,791,717 parameters versus
6,470,885 for the winner, a 320,832 / 4.96% increase rather than the d=384 model's
2.22x expansion. A hook-level isolation test proves that changing the right-hand
nodes cannot change left-hand, face, or body pre-fusion projections. A full finite
forward test enforces fewer than eight million parameters. All 26 focused Stage-1
tests pass, affected scripts compile, scoped `git diff --check` passes, and the real
Citizen + full-clean SemLex loader completed a two-batch optimizer/checkpoint smoke.
No test data was accessed. This challenger is ready to package after the PartMix run;
it must not be combined with PartMix in its first comparison.

## 2026-08-12 02:38 PST — explicit bone/motion challenger implemented

The third predeclared sign-specific challenger is now available off by default as
`--bone-features`. It does not alter the public Apple `[32,61,5]` input or extractor.
The model internally derives directed vectors for the 42 hand/arm chains and three
body links, plus their masked temporal differences. A bone is zero whenever either
endpoint is absent; bone motion is zero when either adjacent bone is invalid. Sparse
face anchors are deliberately excluded because they are not a physical face mesh.
The feature is accepted by the flat/global and feature-isolated part-wise paths and
rejected for the graph-replacement path where it would otherwise be silently unused.

The unchanged global d=256 model grows from 6,470,885 to 6,564,581 parameters
(+93,696 / 1.45%); part-wise + bone is 6,815,141. Unit coverage verifies exact bone
direction, missing masking, temporal masking, face exclusion, and the unchanged
public input. All 28 focused Stage-1 tests pass, affected scripts compile, scoped
`git diff --check` passes, and a real Citizen + full-clean SemLex two-batch optimizer
smoke completed. Run bone-only only after the isolated PartMix and part-wise trials;
do not merge these ideas in their first experiments. The consolidated evidence,
literature mapping, bottleneck diagnosis, and experiment ladder are recorded in
`artifacts/reports/STAGE1_V17_SQUEEZEFORMER_EXHAUSTIVE_AUDIT.md`. No test data was
accessed.

## 2026-08-12 02:42 PST — complete Kaggle challenger package staged; OAuth blocked

All safe work that can precede GPU allocation is complete. The v2 Kaggle archive at
`artifacts/generated/kaggle_stage1_challengers_v2/stage1_v17_challengers_trainval_v2.tar.gz`
contains the current PartMix, feature-isolated part-wise, and bone-feature code plus
the same frozen train/validation-only data. Its SHA-256 is
`8a41f26d3393e388d176cf7648b4c3b33797d80de013fa286883508df6c82b79`.
An independent archive-member audit reports 1,476 Citizen train archives (1,475 usable
after frozen rejections), 378 Citizen validation archives, 1,388 approved SemLex train
archives, and zero `/test/` members. `PACKAGE_MANIFEST.json` records these counts.

`kaggle_stage1_challenger_runner_v17.py` is a single fail-closed CUDA runner for the
three isolated experiments. It verifies the v2 archive hash and safe members, rejects
CPU allocation, preserves the seed/data/sampler/schedule/patience protocol, and writes
separate PartMix, part-wise, or bone outputs. It compiles and scoped
`git diff --check` passes.

The official Kaggle OAuth process remains alive after more than twelve minutes, but
the browser callback has not been approved. The CLI still reports username
`francisbatiancela`; `kokoab/notebook697ebe7d11` still returns 403. This is the same
external authentication blocker across three consecutive goal turns. No further
upload, private-notebook mutation, or real GPU experiment can be performed without
the user clicking Kaggle's **Authorize/Allow** action in the Brave tab (or otherwise
authenticating the CLI as the notebook-owning account). No test data was accessed.

## 2026-08-12 02:47 PST — Kaggle OAuth fixed; controlled PartMix run launched

The user approved Kaggle OAuth and the CLI now identifies as `kokoab`. The old private
notebook has no active session, so a fresh fail-closed script kernel was used rather
than mutating stale notebook state. The verified v2 train/validation-only package was
uploaded privately as `kokoab/slt-v17-stage1-challengers-v2`; Kaggle finished indexing
it at 34,060,542 bytes. Its archive SHA/count/test-member invariants remain those in
the 02:42 handoff above.

Private kernel version 1 at `kokoab/slt-v17-stage1-partmix-p50-v2` is RUNNING. It
requests `NvidiaTeslaT4`, requires `torch.cuda.is_available()`, verifies the exact v2
archive hash, rejects unsafe links/paths and every `/test/` member, and only then runs
the isolated PartMix-p=0.5 experiment with the unchanged d=256/depth=4, seed-1701,
50/50 class/source-balanced, patience-30 protocol. No local heavy process is running.
Do not report the allocation or metrics until the kernel log proves CUDA and the
result artifacts are pulled and audited.

## 2026-08-12 02:53 PST — Kaggle v3 passed CUDA and integrity gates; training active

Kaggle automatically expanded the uploaded tarball into a dataset directory rather
than mounting the archive itself. Kernel v1 therefore failed after the CUDA gate but
before training because it looked only for the archive. The dataset-file API and a
fresh download proved Kaggle exposes the archive contents under the prefix
`stage1_v17_challengers_trainval_v2/`; no data content was changed. A deterministic
tree digest was added over every relative path and file SHA-256. The exact extracted
tree contains 3,253 files and hashes to
`990c6045244b00f409d735808b57025132653fe43a37073c34aa4e93ae96fad2`.
The runner accepts either the original archive+archive hash or the Kaggle-extracted
root+tree hash and retains the no-link/no-test checks.

Kernel v2 still assumed the traditional one-level `/kaggle/input/<slug>` mount and
failed after CUDA but before training. Kaggle's current mount is nested more deeply,
so v3 now finds the unique extracted root recursively and then verifies the same full
tree digest. Version 3 has remained RUNNING beyond both prior 7-second failures,
proving the CUDA requirement, dataset resolver, and full 3,253-file integrity gate
passed. The controlled PartMix training is active on Kaggle; the laptop is only
polling status. No test data was accessed.

## 2026-08-12 02:59 PST — PartMix completed and rejected; part-wise run launched

Kaggle kernel v3 completed successfully on a Tesla T4 in about 4.3 minutes. The
PartMix-p=0.5 checkpoint retained epoch 51 and early-stopped after 81 epochs / 30
stale epochs. Its SHA-256 is
`508b8aec656ed5e717a87fe734c451ab66dce83c9f0e5b8d5dbb5845ee06e7f9`.
The 81-epoch realized PartMix fraction ranged 47.61%-52.92% and averaged 50.11%, so
the requested treatment was actually applied. Provenance confirms 1,475 Citizen +
1,388 SemLex training clips, 50/50 expected source exposure, the frozen manifest and
schema hashes, the unchanged 6,470,885-parameter global architecture, CUDA+AMP, and
false Citizen/SemLex test access flags.

PartMix ties the current landmark winner on Citizen top-1 at 362/378 = 95.77%, but
scores 99.47% top-5 and 95.31% macro F1 versus the winner's 100.00% and 95.51%.
The predeclared SemLex-validation diagnostic scores 837/978 = 85.58% top-1,
932/978 = 95.30% top-5, and 83.45% present-class macro F1. The winner has 840/978 =
85.89%, 940/978 = 96.11%, and 82.60%. Thus PartMix improves SemLex macro F1 by 0.86
points but loses three top-1 and eight top-5 clips while providing no Citizen top-1
gain. Decision: reject PartMix as the new default and do not combine it with the next
challenger. Exact pulled artifacts are under
`artifacts/generated/kaggle_stage1_partmix_kokoab_pull_v3/`; the SemLex report is
under `artifacts/reports/semlex_citizen100_val_audit/partmix_p50_v2/`.

The next isolated experiment is now RUNNING as private kernel
`kokoab/slt-v17-stage1-partwise-v2`. It uses the same verified 3,253-file dataset
tree, seed/data/sampler/schedule/patience, CUDA gate, and d=256/depth=4 global stack,
but PartMix is disabled and one isolated Squeezeformer layer per left hand, right
hand, face, and body precedes whole-body fusion. It has 6,791,717 parameters. No test
data was accessed and no heavy local process is running.

## 2026-08-12 02:14 PST — local-landscape conclusion and SHuBERT feasibility gate

Landscape orientation is already handled correctly by the v17 supplemental path:
raw frames retain aspect ratio, hand crops follow v17 landmarks and motion timing,
and face crops use eye alignment rather than stretching. Orientation therefore does
not disqualify a clip, but it also does not make resolution irrelevant: tiny, blurred,
occluded, or off-frame hands still lose usable information. The full immutable local
pool has now been screened, not sampled: 1,021 exact-text candidates received current
landmark, MobileCLIP2 hand, and visual-speech diagnostics. The original 434 Tier-A
clips remain the strong local training set. Exactly 23 additional unused clips across
15 classes pass the conservative landmark+hand/crop-quality upgrade gate; they remain
non-training-eligible until ASL-fluent exact-variant review. The local mouth/lower-face
heads are excluded because they collapse to 2.15%-3.23% folder-label agreement on
this domain. Bulk-admitting the remaining local clips would add label and extraction
noise and over-weight a small number of recording sessions.

The official SHuBERT repository was shallow-cloned at commit
`cc1929326075bfbad7ad73159b2acf84356059bb` for a read-only integration audit; no
weights or datasets were downloaded. The 57 MiB code clone describes a four-stream
pipeline that requires YOLOv8 signer crops, MediaPipe face/hands/body, three
fine-tuned DINOv2 RGB streams, CUDA, and a custom Fairseq SHuBERT-base encoder (12
layers, width 768). It is not input-compatible with Apple Vision v17 or the current
MobileCLIP2 cache, its released inference path hard-codes CUDA and lacks variable-
length batching, and its downstream fine-tuning documentation is still `TODO`.
The README says the project is primarily MIT licensed, but the separately hosted
weight terms are not established in the repository. Decision: do not blindly
download or make SHuBERT the next mobile experiment. Preserve it as a later isolated
Kaggle frozen-teacher probe after checkpoint sizes/hashes/terms are known. Full audit:
`artifacts/reports/shubert_v17_feasibility.md`. No test data was accessed.

The next executable data/model sequence is: obtain exact-variant review for the 23
local candidates; promote only approved clips into a new immutable train-only
manifest; retrain the compact hand branch with the same Citizen/SemLex/local source
balancing; compare on Citizen validation and the unchanged SemLex validation
diagnostic with fixed fusion weights; then freeze the chosen ensemble before the
independent portrait-iPhone capture. More generic lip-reading data is not the next
lever: Auto-AVSR already supplies broad visual-speech pretraining, while exact-vocab
natural-mouthing supervision and portrait-domain validation are missing.

## 2026-08-12 01:08 PST — SemLex-validation hand-RGB audit isolated from training

The next accuracy gate is a cross-domain component audit on the already approved
978-clip SemLex validation diagnostic before adding more architecture or local data.
`extract_hand_rgb_semlex_val_v17.py` now resolves only the exact frozen validation
selection, requires `training_eligible=false` and
`evaluation_only_never_training`, proves the 978 retained/six quarantined inventory,
and rejects partial or unplanned files. Its outputs are explicitly stamped
`semlex_val`, `val_domain_diagnostic`, and non-training-eligible. The compact
MobileCLIP2 encoder can encode this source, but the Stage-1 training loader still has
no path that accepts it. A separate evaluator validates this provenance and reports
both 100-class and 98-present-class macro F1. The one-clip real Apple Vision smoke
passed in 0.33 seconds, the focused hand suite passes 12/12, and all new scripts
compile. Neither Citizen test nor SemLex test was accessed.

Landscape orientation is not by itself a reason to reject the local corpus: the hand
extractor preserves aspect ratio, follows the v17 landmark motion interval, and
normalizes per-hand/union crops. Tiny, blurred, occluded, or uncertain-variant clips
remain quality risks. The existing 434 strongest local Tier-A clips are already in
the 80.69% hand model. After the cross-domain audit, the remaining local pool will be
re-scored with the improved multimodal teacher, exact-variant and extraction gates,
and a strict source cap so local sessions cannot dominate Citizen/SemLex.

Extraction is now complete for all 978/978 SemLex validation clips across 98 classes:
zero failures, zero empty-view clips, 356.3 MiB total, and 67.45% mean validity over
the left/right/union view grid. A full metadata/schema/provenance read found zero bad
archives. The run took 157.7 seconds. Compact frozen MobileCLIP2 encoding is the next
step; these artifacts remain evaluation-only and cannot enter training.

The previously documented `mobileclip2_env` directory was absent, while its partial
pure-Python runtime layer remained. The isolated environment was reconstructed under
the same path with host Python 3.11/torch 2.13.0 and pinned OpenCLIP 3.3.0,
torchvision 0.28.0, timm 1.0.28, and OpenCV-headless 4.13.0. It reuses the existing
official cached MobileCLIP2-S0 weights. A real SemLex-validation feature smoke ran on
MPS, wrote one finite archive with schema `c54f4edc6f62b08b`, and retained the
non-training/validation-only provenance. The first two failed launches wrote no
project dataset files: one used the Vision environment without OpenCLIP and one
exposed the partial runtime's missing compiled dependencies.

All 978 compact SemLex-validation hand embeddings are now complete after 426.0
seconds on MPS. They occupy 31.0 MiB; a full finite-value, shape, zero-mask, schema,
source, split, and eligibility audit found zero bad or empty archives. The unchanged
epoch-33 multisource compact hand checkpoint scores 73.31% top-1, 91.41% top-5, and
69.75% present-class macro F1 on this diagnostic, versus its 80.69% Citizen
validation score. The older Citizen-only compact hand checkpoint scores only 51.64%
top-1, 79.75% top-5, and 48.21% present-class macro F1 on the same clips. Thus the
reviewed SemLex/local training pool adds 21.68 cross-domain top-1 points; more clean,
variant-consistent hand data is a much stronger lever than the rejected spatial
fine-tuning. The hand branch still trails the unchanged landmark model's 85.89%
SemLex result, so it remains a complementary RGB teacher rather than a landmark
replacement. Exact outputs are under
`artifacts/reports/semlex_citizen100_val_audit/hand_mobileclip2_*`. The landmark
diagnostic was rerun only to regenerate its aligned logits and reproduced 85.89%
exactly. No test data was accessed.

The equivalent SemLex-validation visual-speech path is now implemented as a separate
non-training diagnostic. It applies the unchanged full-utterance, mouth-motion,
eye-aligned mouth/lower-face extractor, adds the proven invalid-WebM-frame-count
fallback, and stamps source/split/eligibility/audio/test provenance. A dedicated
Auto-AVSR cache builder and evaluator support the existing mouth, lower-face, and
learned mouth+lower checkpoints without creating a training-loader path. Six focused
visual-speech tests pass, all new scripts compile, and a real Apple Vision one-clip
smoke passed with schema `c44cdc314b5128c7`. Full 978-clip extraction is next.

Full SemLex-validation visual-speech extraction completed 978/978 across 98 classes
in 393.0 seconds with zero failures. The archives occupy 240.0 MiB; a complete
metadata/schema/provenance read found zero bad archives and zero clips empty in any
view. Mouth, lower-face, and full-face selected-frame validity is 99.77% for each,
and all 978 used a detected mouth-motion interval rather than fallback timing. Frozen
Auto-AVSR mouth/lower feature encoding and unchanged-head evaluation are next.

Both 978-sample frozen Auto-AVSR caches completed, strict-loaded all 120 official
frontend keys, and retained visual-only/evaluation-only provenance. On SemLex
validation, the unchanged Citizen-trained mouth head scores 15.95% top-1 / 37.01%
top-5, lower face scores 15.13% / 40.08%, and learned mouth+lower scores 19.33% /
43.66%. These are well above chance but substantially below their Citizen scores of
29.63%, 26.72%, and 31.75%, respectively. The lip/lower classifiers therefore carry
real signal but are not individually domain-robust; broader or purpose-recorded
visual-speech data is still needed before production use.

Crucially, cross-domain complementarity survives without any SemLex weight tuning.
The exact Citizen-selected four-stream per-sample-z-score weights (0.30 landmark,
0.15 mouth, 0.35 lower face, 0.20 hand) improve SemLex validation from 840/978 =
85.89% landmarks alone to 882/978 = 90.18%, with 51 corrections and nine
regressions. The fixed 75/25 landmark/hand pair reaches 87.83%, and the fixed equal
landmark/learned-mouth+lower pair reaches 88.04%. This makes the 97.88% Citizen
ensemble a credible cross-domain research teacher, though not yet a production
estimate because the original weights were selected on Citizen validation and SemLex
validation is not signer-independent. Reports are under
`artifacts/reports/semlex_citizen100_val_audit/fixed_*`. Thirteen hand/fusion tests and
six visual-speech tests pass. No Citizen or SemLex test data was accessed.

Decision: use this multimodal teacher to re-score the remaining local corpus, admit
only exact-variant/high-confidence/extraction-clean clips, and retain a strict local
source cap. Do not promote the weak standalone lip branch or tune fusion weights on
SemLex validation.

Local multimodal re-mining is now isolated under a strict resolver for the immutable
1,021-clip, 77-class cap-14 exact-text audit manifest. It requires the original
non-training eligibility, exact canonical/pinned-raw text equality, and every raw and
landmark file. New hand and visual-speech extractors stamp `local_audit` and
`train_only_review_diagnostic`; neither the landmark nor hand trainers accept this
source. The MobileCLIP encoder and visual feature cache support the diagnostic source,
and the existing local landmark evaluator now preserves aligned logits. Twenty
focused hand+visual tests pass, all affected scripts compile, and real one-clip hand
and face smokes passed. The multimodal teacher will only screen unused clips for a
small source-capped review upgrade; it will not silently relabel model disagreements.

Local hand extraction completed 1,021/1,021 in 146.0 seconds with zero failures. The
77-class crop corpus occupies 434.8 MiB, has no empty clips, and averages 74.13%
validity over left/right/union views. A complete metadata/provenance read found zero
bad archives. Frozen compact MobileCLIP2 encoding is next.

All 1,021 local compact hand features completed in 536.9 seconds on MPS and occupy
35.3 MiB. A full shape/finite/zero-mask/schema/source/split audit found zero bad or
empty archives. As label-screening agreement, not held-out accuracy, the unchanged
multisource hand model matches the exact-text folder label at 70.62% top-1 / 87.07%
top-5. The current 95.77% landmark winner was rerun on the same pool and reaches
68.76% / 86.78%. This confirms the improved hand stream is independently useful for
local re-mining rather than merely echoing the landmark model. Face extraction is the
next and only heavy process.

Local visual-speech extraction completed 1,021/1,021 across 77 classes in 284.4
seconds with zero runtime failures. The 182.3 MiB corpus has 89.79% mean view
validity; 76 clips have no usable face view, 935 use mouth-motion timing, and 86 use
the explicit full-utterance fallback. Both Auto-AVSR caches are complete. Unlike
Citizen and SemLex, the Citizen-trained visual-speech heads are unusable on this
landscape local domain: mouth is 2.15% top-1, lower face 3.23%, and learned
mouth+lower 2.94%. They are rejected as local cleaning signals.

The unchanged Citizen-fixed 75/25 landmark/hand fusion reaches 751/1,021 = 73.56%
local exact-text agreement, 91.19% top-5, with 52 corrections and only three
regressions relative to landmarks. The full Citizen four-stream weights reach the
same 73.56% top-1 but only 87.46% top-5 and cause 30 regressions because the face
domain is incompatible. Therefore local re-mining uses landmark+hand only. A strict
candidate audit found 23 previously unused Tier-B clips across 15 classes after
requiring old dual-model top-5/one-top-1, current landmark top-1, current hand top-1,
fixed-pair top-1 with standardized margin at least 0.5, at least 80% observed-hand
frames, at least 80% new hand-crop validity, and a maximum two upgrades per class.
This is a small high-confidence review upgrade, not a reason to admit the 587 unused
clips wholesale or treat correlated model agreement as lexical proof.

`select_local_multimodal_upgrades_v17.py` now reproduces that gate from immutable
logit/crop ledgers and hashes every input. The frozen output contains exactly 23 clips
across 15 classes (maximum two/class) under
`artifacts/reports/local_citizen100_quality_audit/multimodal_teacher_reaudit/upgrade_selection/`.
It remains `training_eligible:false` and requires ASL-fluent exact-variant review;
model agreement was not silently converted into ground truth. Twenty focused tests
pass, all affected scripts compile, and scoped `git diff --check` passes.

The independent confirmation protocol is now frozen in
`docs/guides/PORTRAIT_IPHONE_EVAL_V17.md` with a capture ledger template at
`active/v17/portrait_iphone_capture_template.csv`. The minimum target set is five new
people × two repetitions × 100 exact variants = 1,000 portrait-iPhone clips, plus a
recommended 100 OOV clips. Prompts disappear before capture, natural mouthing is not
forced, objective QC occurs before inference, all current model/weight hashes freeze
before evaluation, and the complete set is evaluated once with signer-clustered
intervals and paired tests. It cannot be used for tuning; after use it becomes a
consumed confirmation set.

## 2026-08-11 21:34 PST — anatomical/phonological challengers implemented

The current 6,470,885-parameter flat v17 Squeezeformer remains the measured deployable
baseline at 95.77% Citizen validation top-1 and 85.89% SemLex validation top-1. The
larger d=384 run did not improve Citizen validation, and the earlier generic supervised
contrastive-loss experiments materially hurt it. More capacity or a stronger generic
penalty is therefore not the next justified move.

A targeted architecture/data audit found that the baseline flattens all 61 nodes per
frame before temporal modeling. It has no anatomical graph, explicit hand/face/body
part representation, or sign-phonology supervision. The remaining 16 Citizen
validation errors include confident minimal-pair confusions such as THANKYOU->GOOD and
ANSWER->GO; their landmark-presence statistics do not support treating them as simple
extraction failures. The strongest next landmark hypotheses are therefore explicit
spatial structure and auxiliary handshape/location/movement learning, while preserving
the already strong Squeezeformer temporal stack.

`active/v17/model_v17.py` now has two backward-compatible, opt-in capabilities:

- `spatial_encoder=graph_parts` uses a sparse normalized physical graph, an
  input-sensitive global joint-attention branch, and explicit left-hand, right-hand,
  face, body, and whole-body part pooling before the temporal Squeezeformer;
- optional training-only phonology heads predict ten ASL-LEX attributes from the same
  pooled representation. Ordinary `forward()` and the default flat state dict remain
  unchanged.

`scripts/build_v17_phonology_targets.py` joined every one of the 100 frozen manifest
ASL-LEX codes to the local official ASL-LEX 2.0 table and generated
`active/v17/citizen100_phonology.json`. The mapping and both source hashes are frozen.
Coverage is 100/100 classes for handshape, selected fingers, sign type, movement, major
and minor location, contact, repeated movement, and wrist twist; flexion covers 95/100.
Missing annotations use ignored target `-100`, never a fabricated category. The
trainer now supports the graph options and a weighted mean auxiliary phonology loss,
and records their complete provenance.

The focused Stage-1 suite passes 21/21 tests. All flat-phonology, graph-only, and
graph-plus-phonology real-data smokes passed; the combined path also completed a real
MPS optimizer/checkpoint smoke. The pre-change best flat checkpoint strict-loads with
all keys matched and retains exactly 6,470,885 parameters.

The single preregistered flat-plus-phonology run at weight 0.20 completed on the exact
1,475 Citizen + 1,388 full-clean SemLex class/source-balanced recipe. It peaked at
95.50% Citizen validation (361/378), one clip below the 95.77% baseline, and scored
85.07% top-1 / 95.30% top-5 / 81.96% macro F1 on the frozen 978-clip SemLex validation
diagnostic, below the baseline's 85.89% / 96.11% / 82.60%. This exact equal-head,
single-pooled-token auxiliary objective is rejected. It does not prove that all
phonology-aware learning is unhelpful; it proves this simple formulation is not an
accuracy win. The graph-only full run is now active on the same recipe. No Citizen or
SemLex test data was accessed.

The architecture research supports these hypotheses but does not justify copying
headline numbers blindly. SHuBERT is the strongest directly relevant ASL foundation
teacher found (roughly 1,000 hours of self-supervision over face/hand RGB plus pose),
while DSTA-SLR, VSNet, SignBERT+, BEST, MS-MAE, and recent hypergraph work independently
support dynamic spatial structure, part-aware modeling, or masked sign pretraining.
The small PhonSSM repository is **not valid benchmark evidence here**: its claimed
ASL-Citizen/official evaluations use random stratified splitting rather than the
required signer-disjoint split, and its nominal anatomical adjacency becomes dense at
initialization. Only the general graph/phonology ideas may be tested under our frozen
protocol.

The graph experiments are now resolved. A from-scratch graph/part replacement had the
same approximate total parameter count as the baseline but was dramatically slower on
MPS and reached only 78.31% Citizen validation at epoch 30, versus more than 92% for the
flat model at that stage. It was stopped rather than heating the laptop for a plainly
inferior multi-hour trajectory. A better residual formulation was then implemented:
the proven flat frame token is retained, a smaller 32-D/one-layer graph-part branch is
added through a bounded zero-initialized scalar gate, and the exact 95.77% flat
checkpoint is strict-warm-started. Its epoch-0 logits and validation metrics are
bit-for-bit identical to the flat checkpoint. Ten graph-only epochs followed by
low-rate joint fine-tuning never exceeded that epoch-0 result and stopped after the
declared 20 stale epochs. The graph replacement and zero-gated graph residual are both
rejected for accuracy. The best residual checkpoint simply preserves the zero-gate
baseline; it is not a new gain.

`--initialize-from` now enforces format, manifest, schema, and every shared architecture
field, permits only graph/phonology challenger keys to be missing, records the source
checkpoint hash, and saves an epoch-0 fallback. `--freeze-warm-start-epochs` supports
new-branch-only adaptation before joint fine-tuning. The residual equality regression
test raises the focused suite to 22/22, all affected files compile, and scoped
`git diff --check` passes. Neither Citizen test nor SemLex test was accessed. These
results support retaining the flat d=256 Squeezeformer and moving the accuracy ladder
to better mouth/face timing and broader reviewed hand-RGB information rather than
adding landmarks or generic penalties blindly.

The old hand-trimmed 16x96 mouth package remains preserved as prior evidence. A new,
separately fingerprinted full-utterance visual-speech contract is implemented in
`schema_visual_speech_v17.py` and `extract_visual_speech_v17.py`: up to 96 face-only
reference frames across the complete video, a conservative normalized mouth-shape
motion interval with context/minimum-duration and full-utterance fallback, 32 selected
frames, per-frame eye-line alignment, and 112x112 mouth, lower-face, and full-face
views. It never uses the landmark hand-active interval, audio, or any test split.
Reflected padding contains only transformed source pixels and missing views remain
explicitly invalid/zero.

Four focused visual-speech contract/model tests pass. The new corpus is complete under
`data/local/citizen100_v17/visual_speech_rgb`: 1,475 Citizen train plus 378 Citizen
validation clips, zero extraction failures, 32 unique full-utterance-selected frames
per clip, and median 100% validity for mouth, lower-face, and full-face views. Every
clip used the mouth-motion interval rather than the landmark hand-active interval.
The visually inspected contact sheet under
`artifacts/generated/visual_speech_v17_smoke/` shows stable alignment and clear
lip/lower-face/full-face motion instead of background or hand-only timing.

The official Auto-AVSR visual-only research checkpoint is preserved at
`artifacts/model_assets/models/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth` (SHA256
`fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`). Its exact
11,182,784-parameter 3D-stem/ResNet-18 frontend strict-loaded all 120 transferable
keys with no missing or unexpected keys. The official model zoo reports 3,291 hours
of visual-speech pretraining and 20.3% visual-only LRS3 WER for the full model. The
frontend code is Apache-2.0; downstream checkpoint/data terms still require a license
review before product redistribution. The temporal-kernel-one MaxPool3d was replaced
only by its mathematically identical per-frame MaxPool2d form for Apple MPS support.

All six frozen feature caches are complete for mouth, lower-face, and full-face train
and validation views. Identical class-balanced temporal heads achieved 29.63% Citizen
validation top-1 for mouth, 26.72% for lower face, and 23.02% for full face. Mouth is
more than three times the old 8.99% hand-timed MobileNet mouth result, demonstrating
that full-utterance timing and relevant pretraining matter; full face is rejected as a
primary VSR view. Per-sample standardized late fusion of the 95.77% landmark model
with mouth at 0.55/0.45 reached 364/378 = 96.30%, fixing two landmark errors with no
regressions. A coarse development grid over landmark/mouth/lower/full found
0.50/0.10/0.40/0.00 and reached 367/378 = 97.09%, fixing five errors with no
regressions; artifacts are under
`artifacts/reports/stage1_v17_landmark_auto_avsr_mouth_validation/` and
`artifacts/reports/stage1_v17_landmark_auto_avsr_mouth_lower_face_97_validation/`.
This is same-validation-selected research evidence, not an independent production
estimate. A learned mouth/lower-face fusion head is the next controlled experiment.
Citizen test, SemLex test, and audio remain untouched.

The first learned mouth/lower-face head formulation (LayerNorm before each frozen
feature projection) was stopped by patience at epoch 21 after remaining at chance
(1.59% best). A fixed-batch diagnostic proved the model and MPS gradients could learn,
while a full balanced-stream comparison exposed optimization instability. The next
run therefore matches the already proven single-view ordering—linear projection then
LayerNorm—without changing the views, data, validation protocol, or seed. The failed
checkpoint is retained as negative evidence, not reported as a data limitation.

The corrected learned mouth/lower-face model completed 100 epochs and retained its
epoch-82 checkpoint at 31.75% Citizen validation top-1, above mouth alone (29.63%) and
lower face alone (26.72%), with only 3,347,941 trainable parameters over cached
features. Equal per-sample-z-score fusion with the landmark winner reaches 365/378 =
96.56%, fixing three landmark errors and introducing no regressions. A development
weight sweep has the same 365-clip ceiling, so this cleaner learned head does not beat
the validation-tuned separate-mouth/lower ensemble at 367/378. It is retained as the
compact visual-speech candidate; the two-head late fusion remains research ceiling
evidence. The exact artifact is
`artifacts/models/stage1_v17_visual_speech_auto_avsr_mouth_lower_learned/`, with the
equal-fusion report under
`artifacts/reports/stage1_v17_landmark_auto_avsr_mouth_lower_learned_equal_validation/`.

The reviewed hand-RGB expansion is now complete. A strict train-only supplement
extractor resolves exactly 1,388 full-clean SemLex clips across 97 classes and 434
Tier-A dual-top1 local clips across 72 classes; raw paths, v17 landmark trim paths,
selection hashes, source identity, and the sealed-test contract are checked per item.
It also adds a decoded-frame-count fallback for WebM files whose container reports
negative/garbage counts; the already decoded landmark reference count remains the
contract. All 1,822 new archives use the unchanged hand schema fingerprint
`bf6508de2ea851a4`, passed a complete audit with zero corrupt/empty files, and occupy
704 MiB. SemLex left/right/union validity is 43.39%/75.20%/85.34% with 33.24% two-hand
frames; local Tier-A is 75.17%/59.79%/93.99% with 40.97% two-hand frames. Together
with 1,475 Citizen train clips, the planned reviewed RGB training pool is now 3,297.
Compact embedding extraction and a source-balanced hand retrain are the next gate;
the several-gigabyte spatial-map expansion is deferred until compact features prove
that the additional sources help. No validation/test source data was added or opened.

Frozen official MobileCLIP2-S0 compact encoding is complete for all 1,822 new hand
archives: 1,388 SemLex in 700.8 seconds and 434 local Tier-A in 259.2 seconds on MPS.
Every `[16,3,512]` archive passed schema `c54f4edc6f62b08b`, finite-value, explicit
zero-mask, source-provenance, sealed-test, and unit-norm audits; there are zero corrupt
or empty feature archives. The source-balanced trainer now loads selections by their
exact frozen manifests and samples 45% Citizen, 45% SemLex, and 10% local, with equal
class mass inside each available source. Its real 200-train/100-validation optimizer
smoke passed after fixing a smoke-only reduced-dataset/full-sampler index mismatch.
The full 3,297-sample compact hand retrain is the active next measurement.

The source-balanced compact hand retrain completed after 63 epochs and retained epoch
33 at 80.69% Citizen validation top-1 / 94.71% top-5 / 79.32% macro F1. This is a
10.32-point gain over the Citizen-only compact model (70.37%) and a 10.05-point gain
over the old Citizen-only spatial model (70.63%), proving that reviewed cross-source
volume was a major hand-RGB limitation. A fixed 75/25 landmark/hand standardized
fusion is rejected because it regresses one net clip (361 versus 362), despite the
stronger standalone hand score. The hand model's contribution is complementary only
in the broader visual ensemble: a coarse 0.05 development grid at 0.30 landmark,
0.15 mouth, 0.35 lower-face, and 0.20 hand reaches 370/378 = 97.88% top-1 and 97.59%
macro F1, fixing ten landmark errors while regressing two. This is the new research
teacher ceiling, but its weights were selected on the same Citizen validation set and
require independent portrait confirmation. Exact reports are under
`artifacts/reports/stage1_v17_hand_mobileclip2_multisource_balanced_validation/` and
`artifacts/reports/stage1_v17_multimodal_teacher_97_88_validation/`. No test was used.

Before spending roughly another 4 GiB on supplement spatial maps, the next cheapest
controlled experiment warm-starts the already cached Citizen spatial fine-tuner from
the new multisource compact checkpoint. This tests whether pre-pooling adaptation adds
value without re-encoding data or changing the validation protocol.

Both spatial warm-start formulations are now rejected, so supplement spatial-map
extraction remains cancelled. The old hard temporal shift destroyed the multisource
feature geometry: it reached only 66.93%, 68.52%, 72.22%, 73.02%, then 70.90% through
five epochs, with epoch time rising to roughly one minute, and was stopped. A better
zero-gated residual was implemented and regression-tested. Real cached spatial maps
without the shift reproduce the compact checkpoint exactly at 305/378 = 80.69%; its
epoch-0 checkpoint is saved. Conservative 2e-5 joint adaptation nevertheless fell to
75.13% and 73.28% in two epochs and was stopped. This proves the cached pre-pooling
maps are valid while rejecting temporal shift plus joint final-block adaptation as the
next accuracy move. The compact 80.69% hand checkpoint remains selected; no new
supplement spatial maps were created. End-to-end visual-speech pixel fine-tuning is
the next bounded experiment.

The local end-to-end mouth benchmark confirms this experiment belongs on GPU: five
training batches plus full validation took 28.1 seconds on MPS. A private Kaggle
dataset was therefore created by CLI under
`francisbatiancela/slt-v17-visual-speech-pixel-v1`, containing only the 1,475 Citizen
train and 378 Citizen validation visual-speech archives, current v17 code/manifest,
the official Auto-AVSR checkpoint, and the 29.63% frozen-mouth warm start. No test data
is included. The originally supplied private `kokoab/notebook697ebe7d11` is inaccessible
to the authenticated CLI OAuth account (`francisbatiancela`), so a new private GPU
kernel was pushed via CLI at
`francisbatiancela/slt-v17-auto-avsr-mouth-fine-tune`. It strict-warm-starts the proven
mouth checkpoint, saves epoch 0 as a fallback, fine-tunes the complete visual frontend
at 1e-5 and head at 3e-5, and uses patience 10. The kernel is currently running; its
output must be pulled and audited before any accuracy claim.

Kaggle GPU execution is externally blocked despite a correct and fully uploaded
private package. Five CLI kernel versions were audited: script and notebook formats,
explicit `enable_gpu`, `NvidiaTeslaT4`, and generic `Gpu` shapes were tried; the final
notebook syntax is valid and the CLI reports 30/30 GPU hours remaining. Kaggle still
scheduled CPU-only PyTorch every time. The CUDA assertion prevented accidental CPU
training, and logs for every attempt are preserved under
`artifacts/generated/kaggle_visual_speech_pixel_failed_v*/`. This is an allocator
failure, not a model/data failure. The reusable private dataset and kernel remain at
`francisbatiancela/slt-v17-visual-speech-pixel-v1` and
`francisbatiancela/slt-v17-auto-avsr-mouth-fine-tune`.

A bounded local progressive-unfreeze study then resolved end-to-end mouth adaptation.
The initial trainer's random 88-of-112 crop and four-frame deletion collapsed the
warm-start model to 19.31%. A mild regime was added (center-aligned +/-4 pixel jitter,
gentle photometric change, no temporal deletion). A second warm-start issue was also
found: the from-scratch EMA ramp deliberately forgets early weights and therefore
overwrote the pretrained solution. Warm starts now enter the mature fixed-decay EMA
regime. With both fixes, layer-4-only fine-tuning at head 1e-5/frontend 1e-6 is stable
but not beneficial: epochs 1/2/3 reached 29.37%/28.84%/28.57%, and patience retained
epoch 0 at 29.63%. Citizen-only end-to-end visual-speech adaptation is rejected; the
frozen Auto-AVSR mouth/lower-face teacher and learned cached-feature head remain the
selected visual-speech paths. The next accuracy gate is independent portrait-iPhone
confirmation and out-of-fold fusion calibration, followed by a sign-pretrained
video/context RGB teacher or targeted new exact-vocabulary recordings—not further
same-validation weight/model tuning.

## Product goal

Build a fully offline, iOS-first ASL translator that retains the highest practical
accuracy while remaining viable on low-end and medium-spec iPhones. Accuracy and
generalization come first for the initial model; distillation and aggressive model
compression are explicitly deferred until the best accurate baseline is established.

The first new isolated-sign vocabulary is 100 signs, reduced from 300. Dataset splits
must be signer-disjoint. The target minimum is at least 20 training signers and 5 test
signers, ideally at least 5 clips per person per selected sign where the source permits.
The old seven-person dataset is not acceptable evidence of generalization.

## Locked decisions

1. Do not use PopSign as the sole or primary source for a general recognizer because it
   intentionally captures one-handed smartphone variants. The recommended simpler
   v17 baseline is ASL Citizen as the sole primary training dataset, using its official
   signer-disjoint split. PopSign is retained only as an optional portrait/orientation
   audit or explicitly reviewed one-handed domain supplement.
2. Apple Vision is the selected v17 iOS hand extractor. It beat the official MediaPipe
   Hand Landmarker in the frozen quality/full-classifier bakeoff below. Keep MediaPipe
   as a separately fingerprinted research challenger; do not mix its archives with
   Apple archives or replace Apple without new signer-disjoint evidence.
3. v17 is a clean extractor/schema boundary. It will feed a new model trained from
   scratch. The 96% v16 checkpoint is not compatible with v17 features.
4. Portrait is the canonical mobile capture experience, but the extractor must accept
   portrait, landscape, square, rotation-metadata-tagged, explicitly rotated, and
   mirrored inputs without geometrically stretching them.
5. Do not distill the teacher yet. Do not claim mobile readiness based on desktop
   parameter count alone; eventually measure Core ML package size, cold start, memory,
   sustained latency, thermals, and accuracy on real iPhones.
6. PopSign is a one-handed smartphone-signing corpus. The user holds/controls the phone
   with one hand and signs with the other, and the vocabulary was selected for concepts
   that could be performed one-handed. This does not prove that every underlying ASL
   concept is inherently one-handed or that its normal conversational form is captured.
   The first PopSign-only model must be described as a one-handed isolated-sign
   recognizer, not a general ASL translator. General coverage requires reviewed
   two-handed data or new iPhone recordings in a later dataset phase.
7. For the Citizen-only 100-class baseline, use a minimum exact-variant per-class floor
   of 10 training, 3 validation, and 5 test signers. Preserve one raw gloss plus one
   ASL-LEX code per canonical class. Citizen has 35/6/11 signers overall, but do not
   claim that every class has 20 training signers.
8. Apple Vision was not locked by preference. MediaPipe Hand Landmarker was implemented,
   orientation-tested, visually audited, fully extracted on train/validation, and given
   an identical Stage 1 run. It lost 89.95% to 93.12% validation top-1 and was slower on
   the measured host. The schema therefore stays Apple Vision.
9. Do not use standard YOLO Pose as the hand extractor. Its ordinary human-pose
   keypoints do not provide the 21 finger joints needed for ASL. A custom YOLO/RTMO
   hand-pose project would require new keypoint annotations and is lower priority than
   evaluating pretrained MediaPipe Hand Landmarker or RTMPose-Hand.
10. Keep v17 Squeezeformer as the measured landmark classifier baseline. The first RGB
    accuracy challenger was refined from a generic MobileOne/MoViNet proposal to the
    stronger official MobileCLIP2-S0 mobile image tower plus a small temporal head. It
    was an accuracy comparison, not distillation; its completed result is locked below.
11. The frozen full-frame MobileCLIP2-S0 experiment and its fixed late fusion are
    complete and rejected. They reached 39.68% RGB-only versus 93.12% Apple-landmark
    validation top-1, and every tested fusion weight reduced accuracy. This rejects only
    that exact frozen/global-pooled configuration, not MobileCLIP2 as a fine-tunable
    visual backbone or a future high-resolution hand-crop stream. Do not spend the
    official test split tuning either one.
12. Model selection is now frozen on Apple Vision landmarks plus the v17 Squeezeformer.
    Its one-time official 11-signer Citizen test result is 87.57% top-1, 98.64% top-5,
    and 87.39% macro F1 on 1,247 clips. Never tune against this test result or repeatedly
    evaluate it. MobileCLIP variants and fusion remain research evidence, not runtime
    dependencies.

## v16 evidence and reason for replacement

The existing v16 Stage 1 checkpoint reports roughly 96% on its internal evaluation,
but only achieved 40.28% top-1 and 55.56% top-5 on the downloaded 72-video ASL Citizen
external audit. The audit contained 12 signs and 29 public participant IDs. All 72
videos decoded and Apple Vision extracted successfully. This external result confirms
the old accuracy is not evidence of broad signer/capture generalization.

The v16 extractor also had material correctness problems:

- Apple chirality was reversed (`VNChiralityLeft == -1`, unknown `0`, right `1`, while
  v16 treated `1` as left).
- Its per-joint Kalman gap filling could turn missing joints into fake zero-valued
  observations as other joints were filled.
- It marked all 21 joints present for any detected hand, ignored joint confidence, and
  allowed missing zero coordinates to become nonzero after centering.
- Fourteen of fifteen allocated face nodes were always zero.
- It used separate Vision handlers for hand/body and re-extracted overlapping windows.
- It could produce fractional presence masks during temporal interpolation.
- It buffered full-resolution video frames, unsafe for PopSign's portrait resolutions.
- Its “aspect distortion” augmentation conflicts with correct orientation-independent
  geometry and must not be carried into v17 training.

## v17 extractor state

**Location:** `active/v17/`

Implemented files:

- `schema_v17.py`: versioned `[32, 61, 5]` contract and schema fingerprint.
- `geometry_v17.py`: isotropic image coordinates, bounded per-joint interpolation,
  body-relative normalization, missing-value preservation, and temporal resampling.
- `extract_v17.py`: Apple Vision detection, correct chirality, orientation/mirror
  canonicalization, aspect-preserving image cap, memory-bounded frame sampling,
  hand-activity trimming, serialization, and sequential batch extraction.
- `audit_v17.py`: reproducible archive invariant checks and Markdown/CSV reporting.
- `README.md`: feature/orientation contract, setup, commands, and migration boundary.
- `src_v17/`: compatibility import and CLI wrapper.
- `test/test_v17_extractor.py`: pure geometry plus real Apple Vision regression tests.

Feature tensor contract:

- Shape: `[32 frames, 61 nodes, 5 channels]`, stored as float16.
- Nodes: 21 anatomical left hand, 21 anatomical right hand, 15 actual face landmark
  samples, and 4 upper-body points.
- Channels: body-relative X, body-relative Y, relative log-scale depth proxy, binary
  presence, and confidence.
- Missing spatial/depth/confidence values are exactly zero.
- Every archive embeds metadata, diagnostics, full schema JSON, and a schema
  fingerprint. A config mismatch is rejected on load.

Default extraction config:

- 32 output frames, at most 96 uniformly sampled source frames.
- Long image side capped at 1280 pixels without upscaling or aspect distortion.
- Body and face requested every 8 sampled frames; hand requested every sampled frame.
- Minimum joint confidence 0.15; hand gaps up to 3 frames and auxiliary gaps up to 16
  frames are linearly interpolated only when bounded by real observations.
- Leading/trailing inactivity is trimmed with 2 source-frame context.
- At least 2 detected-hand frames are required.

Orientation contract:

- OpenCV honors video rotation metadata by default.
- `--rotation 0|90|180|270` overrides incorrect/missing metadata.
- `--input-mirrored` flips stored mirrored pixels exactly once before Vision.
- Vision always receives upright, unmirrored pixels with explicit image orientation up.
- Coordinates are converted to isotropic image geometry using the longest image side.

## Validation ledger

### 2026-08-11 21:13 PST — Auxiliary RGB audit fixes the next Stage-1 ladder

The current mouth and hand RGB branches are useful proof-of-signal experiments, not
best-available algorithms. The 1.086M-parameter mouth model uses an ImageNet-pretrained
MobileNetV3-Small, 16 96x96 crops, two shallow temporal blocks, and only Citizen train
data. It reached 8.99% Citizen validation top-1 and corrected two errors from the clean
landmark winner, which confirms complementary information but not a deployable
lip-reader. More importantly, its crops are sampled only within the landmark
hand-activity interval; spoken or mouthed words can occur before or after that interval.

The next mouth experiment must rebuild the crop package over the full utterance using
face alignment and a speech/mouth-motion interval, with roughly 29-32 frames. Audio may
be used offline for VAD/transcription/forced alignment and label-quality auditing, but
the inference branch remains visual-only. Compare aligned mouth/lower-face and
full-face inputs because published VSR evidence shows extraoral facial motion can help.
Use a pretrained visual-speech frontend such as Auto-AVSR or AV-HuBERT plus a stronger
temporal head as an accuracy teacher before designing a mobile student. LRW is useful
for pretraining but is not a direct 100-class supplement: only 17 frozen vocabulary
labels exactly overlap LRW's 500 words, and its official 70 GB distribution requires
the BBC academic data-sharing agreement. LRS3 is sentence-level pretraining/mining
data, not clean isolated examples of all 100 ASL labels. A genuine exact-vocabulary lip
dataset would therefore require targeted recording or rigorously aligned mining and
must be fine-tuned on signing-domain faces rather than treated as equivalent ASL data.

The current hand RGB branch is also not the strongest available design. It was trained
only on 1,475 Citizen clips, starts from generic MobileCLIP image features, and its hard
temporal-shift/final-block fine-tuning reached 70.63% alone. Fixed late fusion with the
clean landmark winner added only one Citizen validation clip (96.03%). Before another
architecture sweep, extract the identical left/right/union RGB schema from the existing
1,388 reviewed full-clean SemLex clips and the 434 Tier-A local clips, then train with
explicit class/source balance. This raises the reviewed hand-RGB training pool from
1,475 to 3,297 without opening either sealed test. After that, evaluate a genuinely
video-native, sign-pretrained RGB teacher (SignRep/Hiera/VideoMAE-family reference) and
retain full upper-body/context information where hand-only crops lose sign location,
contact, or facial evidence. Distillation and mobile optimization remain downstream.

The controlled order is now: (1) collect an independent portrait-iPhone development
set because the 97.09% five-model ensemble weights were selected on the same 378-clip
Citizen validation set; (2) rebuild and retrain the visual-speech branch; (3) expand and
retrain the hand/context RGB branch; (4) calibrate fusion out of fold and accept it only
if both Citizen and the independent portrait set improve without damaging SemLex; and
(5) distill a validated multimodal teacher into a single practical student and measure
it on real iPhones. Do not move the accuracy effort to Stage 2 yet, and do not present
the fragile five-model validation ensemble as production evidence.

### 2026-08-10 17:34 PST — Accuracy ladder opened; local and lip experiments gated

The user authorized continued controlled experimentation toward approximately 97%
Citizen validation accuracy, including the model-screened local pool and a separate
lip-reading-style branch. Citizen test remains consumed and prohibited for tuning;
SemLex test remains sealed. A scan of the existing frozen validation logits showed
that the d=256/d=384 probability ensemble reaches only 364/378 (96.30%), two clips
over d=256 alone but three clips below 97%, while other existing model combinations
are no better. This is not enough to justify doubling mobile inference cost.

The next experiment ladder is deliberately ordered by cost: a second clean d=256 seed;
then Tier-A local supplementation at a bounded source share; then a face/lip-only
landmark branch and fusion; only afterward new mouth-pixel or hand-crop RGB work. The
second clean seed-3407 run is currently active on MPS with the identical 1,475 Citizen
+ 1,388 full-clean SemLex inputs and 50/50 class/source-balanced protocol.

`active/v17/train_stage_1_v17.py` now has a strict local-review loader and explicit
nonuniform source margins. The real three-source contract loads 1,475 Citizen, 1,388
SemLex, and exactly 434 Tier-A local clips and solves to 45%/45%/10% expected source
exposure while retaining exactly 1% expected exposure for every class. Local signer
identity remains unknown, the review manifest remains unmodified, and use requires an
explicit CLI approval flag and records the selected tiers/hashes in provenance.

`Stage1V17Config` also supports `all`, `hands`, `face`, and `mouth` model-visible node
masks inside the network, so a separate lip/face diagnostic cannot accidentally see
hand/body features and remains consistent across training, evaluation, checkpointing,
and export. The cheap branch uses the existing four v17 mouth landmarks first; those
landmarks are only a low-temporal-resolution proxy, so a weak result will trigger a
separate real-pixel mouth-crop experiment rather than a claim that lip information is
useless. Fifteen focused Stage-1 tests pass, including explicit source margins, strict
local-tier loading, and proof that face-only logits are invariant to changed hand
nodes. Relevant primary research reviewed for the ladder includes SWA
(`https://arxiv.org/abs/1803.05407`), model soups
(`https://arxiv.org/abs/2203.05482`), supervised contrastive learning
(`https://arxiv.org/abs/2004.11362`), and SAM
(`https://arxiv.org/abs/2010.01412`).

The clean seed-3407 run subsequently completed after 68 epochs at 94.71% Citizen
validation top-1 (358/378), below the seed-1701 winner's 95.77%. It is rejected as a
standalone candidate and was not evaluated on either test. The controlled Tier-A local
10% run then started from scratch with seed 1701 and the declared 45% Citizen / 45%
SemLex / 10% local source margins. The trainer also now has an opt-in supervised
contrastive objective for a later isolated ablation; its default weight is zero, so it
does not change the active local-only run. Sixteen focused Stage-1 tests pass.

The real-pixel lip route is now schema-gated separately in
`schema_mouth_rgb_v17.py` and `extract_mouth_rgb_v17.py`. It samples 16 actual video
frames only inside each archive's frozen v17 hand-active interval, runs Apple face
landmarks without hand/body requests, makes a 96x96 square lower-face/mouth crop,
stores JPEG pixels plus explicit validity/boxes/source indices, rejects test as a CLI
split, and embeds source/schema provenance. A two-clip real Citizen smoke completed
2/2 with zero failures; all 16 frames in both clips were usable. Visual inspection of
`artifacts/generated/mouth_rgb_v17_smoke/contact_sheet.jpg` shows stable alignment and
clear lip motion rather than accidental full-frame/background content. The three pure
mouth RGB tests and sixteen Stage-1 tests pass. Based on MobiVSR
(`https://arxiv.org/abs/1905.03968`), a small depthwise-separable visual-speech model is
the intended pixel baseline; AV-HuBERT (`https://arxiv.org/abs/2201.02184`) is a much
heavier research reference rather than the first mobile branch. Audio exists in most
sampled Citizen files, but the first branch will remain visual-only so it cannot win by
an audio-label shortcut that would be unavailable to a non-speaking signer.

The matching visual-only classifier is implemented separately in
`model_mouth_rgb_v17.py` and `train_stage_1_mouth_rgb_v17.py`. It is a
depthwise-separable 3D spatial/temporal network with validity-masked attention pooling,
under one million parameters, sign-safe mirror/color/temporal augmentation, EMA,
train/validation-only loading, saved aligned validation logits, and explicit
`visual_only:true`, `audio_accessed:false`, and `test_evaluated:false` checkpoint
provenance. Its forward/missing-frame test passes. This code is prepared but no full
mouth crop extraction or classifier result exists yet; do not quote the smoke as
accuracy. Meanwhile the active local-10% landmark run reached 95.50% Citizen
validation at epoch 41, already ahead of the clean baseline at the same stage but not
yet above the clean run's final 95.77% checkpoint.

The controlled Tier-A local-10% run completed after 84 epochs. Its retained epoch-54
checkpoint ties the clean winner at 95.77% Citizen validation top-1 (362/378), with
99.74% top-5 and 95.31% macro F1. Relative to the clean winner it corrects five clips
and regresses five, leaving eleven clips wrong for both. Probability ensembling the two
reaches only 363/378; the strongest existing d=256/d=384 ensemble remains 364/378
(96.30%). On the 978-clip SemLex validation diagnostic, however, local-10 reaches
86.50% top-1 / 96.22% top-5 / 84.25% present-class macro F1 versus the clean winner's
85.89% / 96.11% / 82.60%. Therefore Tier-A local data provides a genuine secondary-
domain improvement without sacrificing the primary gate, but it does not reach 97%
alone. Citizen and SemLex test remain untouched. The separate d=128/depth-2 face-only
landmark run has started on Citizen plus full-clean SemLex with no local data and no
hand/pairwise visibility.

The face-only landmark proxy completed after 35 epochs at only 2.65% Citizen
validation top-1 and is rejected. The four mouth points are too sparse and mostly
interpolated to support visual speech; this result does not reject real mouth pixels.
Full real-pixel extraction then completed for all 1,475 Citizen train and 378 Citizen
validation clips with zero failures and zero archive/schema/decode errors. Median
validity is 16/16 frames in both splits; the minima are 12/16 train and 15/16
validation. The complete crop corpus is only 62 MiB, and Citizen test was not accepted
or accessed.

The first 115k-parameter depthwise 3D classifier attempt was interrupted before one
epoch after macOS MPS took more than two minutes; this was an operator-throughput
failure, not an accuracy result. The model was changed to a shared depthwise-separable
2D frame encoder plus the same temporal depthwise head/attention. It has 104,785
parameters, uses mobile-friendly 2D/1D operators, completes a warm batch-16 forward+
backward in about 0.084 seconds on MPS, and its first real training epoch completed in
about ten seconds. A noncontiguous-MPS backward edge case found on the initial full run
was fixed with explicit contiguous boundaries and reproduced successfully on a real
augmented batch before restart. The active visual-only mouth run remains Citizen
train/validation-only with audio and test both untouched.

The 104,785-parameter mouth model trained from scratch was stopped after twelve epochs
because train loss fell while Citizen validation remained at 1-2% top-1; its pixels
alone did not provide enough data to learn signer-independent facial features. An
isolated torchvision 0.23.0 target directory was created under
`artifacts/generated/mouth_rgb_torchvision` without changing the validated venv. It
uses the torch-2.8-compatible macOS wheel and the official ImageNet MobileNetV3-Small
weight with SHA-256
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

The resulting pretrained visual-only MobileNetV3-Small mouth branch has 1,086,789
active parameters and trained for 53 epochs with a lower backbone learning rate. Its
best Citizen validation checkpoint is epoch 41 at 8.99% top-1 (34/378); late top-5
reached 21.96%. It accessed neither audio nor test. It correctly recognizes two of the
clean/local landmark models' sixteen errors: one `THANKYOU -> GOOD` and the
`ANSWER -> GO` clip. However, dense probability/z-score fusion reaches at most 363/378
with the clean model and does not improve the local model; uncertainty-gated top-k
reranking is also neutral. Therefore mouth pixels contain genuine complementary word
signal but this small Citizen-only branch is rejected for runtime/fusion and does not
reach the accuracy goal.

The next controlled run has started from scratch on the clean 1,475 Citizen + 1,388
SemLex d=256 protocol with a 0.05 supervised-contrastive loss weight and temperature
0.10. Architecture, data, 50/50 class/source sampling, augmentation, seed, validation,
and test isolation remain unchanged. This tests explicit cross-domain same-class
embedding alignment independently of local supplementation.

The fixed 0.05 supervised-contrastive run was stopped at epoch 27 after reaching only
90.48% Citizen validation versus the clean run's 93.92% by epoch 25; the persistent
contrastive constraint slowed/displaced classification learning. A second controlled
schedule linearly decayed the same contrastive weight to exactly zero after epoch 12,
preserving its small early advantage and then reverting to plain cross-entropy. It was
also stopped after epoch 22 at 89.95% versus the clean run's 93.39% by epoch 21.
Neither schedule approached the winner, so supervised contrastive learning is rejected
for this class/source sampler and batch regime rather than combined with local data.

The next clean ablation has started with `input_modality=hands`, changing only which
v17 nodes are visible inside the same d=256 model. Both hands, confidence, temporal
derivatives, and pairwise hand-shape distances remain; the chance-level face nodes and
four body nodes are masked. Data, sampler, augmentation, optimizer, schedule, seed,
validation, and test isolation are otherwise unchanged.

The hands-only ablation initially led the all-node baseline by roughly four to seven
points, but lost that advantage as training matured. It was stopped after epoch 23 at
90.21% Citizen validation versus approximately 93.92% for the clean all-node run at a
comparable epoch. This rejects permanent face/body masking: even sparse auxiliary
nodes provide useful late-stage disambiguation.

A clean source-ratio ablation then increased SemLex exposure from 50% to 60% while
holding all other d=256 settings fixed. It was stopped after epoch 24 at 90.74%
Citizen validation, materially behind the baseline curve. The reciprocal 60% Citizen /
40% SemLex run is now active. These are development-validation experiments only;
Citizen test remains consumed and prohibited, and SemLex test remains sealed.

Mirror test-time augmentation was evaluated without training on the retained clean
winner. The original view gets 362/378, the mirrored view 359/378, and both probability
and logit averaging get 360/378 (95.24%). Mirror TTA is rejected because it doubles
classifier inference while losing two correct clips. The active 60% Citizen / 40%
SemLex run reached 91.53% at epoch 22; it remains active because earlier retained runs
made material late-training gains after epoch 40.

The 60% Citizen / 40% SemLex run completed after 86 epochs. Its best checkpoint was
epoch 56 at 95.24% (360/378), 99.47% top-5, and 94.99% macro F1, so the original
50/50 clean model remains better by two clips. Citizen and SemLex test were untouched.

`evaluate_model_soup_v17.py` now performs schema/manifest/config-gated two-checkpoint
weight interpolation on Citizen validation and saves a single-model checkpoint plus
aligned logits. It normalizes legacy/default-equivalent configs before comparison, and
its state-blending unit test passes. Clean/Tier-A-local and clean/60:40 soups at weights
0, 0.25, 0.5, 0.75, and 1 never beat the clean endpoint's 362/378; intermediate soups
fell as low as 359/378. Therefore weight-space averaging is rejected for these
independently optimized trajectories.

Probability ensembles were also recomputed with aligned logits. Clean plus d=384 still
peaks at 364/378. Adding the 60:40 checkpoint reaches 365/378 (96.56%) near weights
0.42/0.28/0.30; a fixed 0.1 simplex grid and 100,000 deterministic Dirichlet samples
found no 366th correction. This remains below the 367/378 target and would require two
d=256 models plus the heavier d=384 model at runtime, so it is research evidence rather
than the mobile selection. A clean 50/50-source d=256 run with label smoothing reduced
from 0.10 to 0.05 is now active; all other settings remain fixed.

The label-smoothing-0.05 run completed after 115 epochs. Its retained epoch-85
checkpoint reaches 95.50% Citizen validation top-1 (361/378), 100% top-5, and 95.18%
macro F1. It is one clip below the 0.10 clean winner and does not increase any tested
landmark-only ensemble ceiling, so 0.10 remains the standalone setting.

The previously audited high-resolution hand-spatial cache was reused without video
re-extraction. The removed OpenCLIP package layer was restored under the isolated
`artifacts/generated/mobileclip2_runtime` target (OpenCLIP 3.3.0, timm 1.0.28 and
small dependencies); the existing exact cached MobileCLIP checkpoint was reused.
Aligned 256-D train/validation features were regenerated for the clean landmark and
70.63% hand-RGB models. A zero-initialized learned residual over the 95.77% clean
model made no validation prediction changes after 21 epochs. Fixed per-sample
z-score late fusion at 75% landmark / 25% hand RGB reaches 96.03% (363/378), a real
one-clip gain but still below target.

A development-only multimodal probability ensemble has now crossed the requested 97%
Citizen validation threshold. The simple rounded weights are 0.15 clean d=256, 0.15
clean d=384, 0.10 60:40-source d=256, 0.10 hand RGB, and 0.50 mouth RGB. It reaches
367/378 = 97.09% top-1, 99.47% top-5, and 97.06% macro F1. Relative to the clean
winner it fixes six clips, regresses one, and leaves ten wrong for both. The exact
reproducible output, member hashes, normalized weights, and aligned scores are under
`artifacts/reports/stage1_v17_multimodal_ensemble_97_validation/`, produced by the new
`evaluate_multimodal_ensemble_v17.py`.

This is a **development validation result, not production-ready evidence**. The weights
were searched on the same Citizen validation split: 1,392/10,000 small perturbations
around the selected vector retained 367 correct, so the result is locally reproducible
but not broadly weight-robust. It also requires three landmark classifiers, the hand
RGB branch, and the mouth RGB branch, which is far too heavy for the current mobile
selection. Citizen test remains consumed/prohibited and SemLex test remains sealed.
Independent confirmation must use a newly collected portrait-iPhone set; until then,
the 95.77% single d=256 clean model remains the honest deployable selection and the
97.09% ensemble remains a research teacher/candidate.

Final focused validation passes: 22/22 Stage-1 and mouth-RGB tests, including the new
model-soup and multimodal alignment/probability checks. All affected Python files
compile, and the scoped `git diff --check` passes. The large pre-existing untracked
repository reorganization remains untouched.

### 2026-08-09 18:11 PST — v17 unit and real-Vision regressions

Command:

```bash
venv/bin/python -m unittest test.test_v17_extractor -v
```

Result: 10 tests passed. Coverage includes portrait/landscape isotropic identity,
independent per-joint gap filling, exact zeros for missing nodes, binary masks after
resampling, correct known chirality, temporal assignment for unknown chirality, exact
rotate/unrotate and mirror/unmirror transforms, aspect-preserving 2592x1944 capping,
real Apple Vision feature equivalence across transformed inputs, face detection, and
schema-enforced save/load.

### 2026-08-09 18:11 PST — 72-video ASL Citizen v17 extraction

Inputs: train/validation/test, 24 videos per split, under
`data/local/ios100_audit/asl_citizen/`.

Output: `data/local/ios100_audit/landmarks_v17/`.

Result: 72/72 extracted, zero failures, zero no-hand clips. The final test split ran at
about 2.62 videos/second on the current Apple Silicon development machine.

### 2026-08-09 18:15 PST — reproducible v17 archive audit

Command:

```bash
venv/bin/python active/v17/audit_v17.py data/local/ios100_audit/landmarks_v17
```

Result: PASS. All 72 archives loaded with the current schema and passed shape, finite
value, binary-presence, missing-spatial-zero, and missing-confidence-zero invariants.
There were no schema/load errors. Median extraction time was 0.3911 seconds/video;
median detected hand-frame fraction improved from 0.4538 before activity trimming to
0.8806 afterward. Median hand/face/body presence was 0.4277/0.7812/0.4922. Shoulder
normalization was available for 64 videos and palm fallback for 8. Corrected chirality
counts were 1,145 left, 1,825 right, and 0 unknown. Outputs:
`artifacts/reports/V17_EXTRACTOR_AUDIT.md` and
`artifacts/reports/v17_extractor_audit.csv`.

### 2026-08-09 18:16 PST — token-efficient operating guide

`AGENTS.md` now contains the mandatory two-file startup rule, targeted repository map,
focused validation commands, dirty-worktree protection, and current v17/PopSign hard
gates. Project facts remain solely canonical in this file to prevent duplicated truth.

### 2026-08-09 18:33 PST — PopSign one-hand scope clarified

The official PopSign paper states that the game has the user hold/control the phone with
one hand while the other performs the sign, that the dataset focuses on one-handed
smartphone signs, and that a general recognizer would additionally require two-handed
signs and broader viewpoints. Decision: continue using PopSign as the primary v17
accuracy corpus, but constrain the first product claim accordingly and plan a reviewed
two-handed expansion later. Source:
`https://signdata.cc.gatech.edu/res/doc/popsign_v1_0/popsign_v1_0_supplemental.pdf`.

### 2026-08-09 18:36 PST — primary dataset recommendation changed

PopSign-only training was rejected as insufficiently representative of ordinary
one- and two-handed ASL. Metadata was recalculated without combining dataset identity
counts: ASL Citizen alone has 3 signs at a strict 20 train / 5 validation / 5 test
signer-per-class floor, 45 at 15/4/5, 123 at 10/4/5, and 208 at 10/3/5. Decision:
recommend a 100-sign Citizen-only baseline chosen from the 123 signs meeting 10/4/5,
preserving the official 35-train/6-validation/11-test-signer split. Repeated takes are
not required when unavailable; cross-signer diversity is the higher priority.

This entry records the initial normalized-label metadata scan. It was superseded by
the exact raw-gloss/ASL-LEX selection below, whose locked floor is 10/3/5 and whose
actual split identity counts must come from the frozen manifest rather than the
dataset-wide identity totals.

The active PopSign `test/thankyou` transfer was stopped without deleting recoverable
partials. Stored bytes: 25,165,824-byte prefix plus four range parts of 148,417,024,
33,554,432, 25,165,824, and 125,829,120 bytes under
`data/local/popsign_v17_archives/test/`. No PopSign video was extracted.

### 2026-08-09 18:47 PST — Citizen100 exact-variant manifest frozen

The user approved ASL Citizen as the sole primary dataset and confirmed the project is
personal/noncommercial. `active/v17/citizen100_seed.json` replaces the unavailable
standalone Citizen labels LOOK and SAY with EAT and WATER; SEE/TALK/TELL remain. The
reproducible builder `scripts/build_citizen100_v17.py` froze
`active/v17/citizen100_manifest.json` with 100 unique canonical labels and 100 unique
raw-gloss/ASL-LEX variants. No numeric or lexical variants are merged.

Coverage is 1,475 train, 378 validation, and 1,247 test videos (3,100 total). Per-class
signer ranges are 11–16 train, 3–5 validation, and 10–11 test. The manifest status is
`metadata_frozen_pending_asl_review`; exact ASL variant review remains necessary before
a final accuracy claim. Report: `artifacts/reports/CITIZEN100_V17_MANIFEST.md`.

`scripts/download_citizen100_v17.py` now reads only selected ZIP-member byte ranges,
decodes raw ZIP deflate, enforces official size and CRC, writes atomically, retains
official splits, and records SHA-256 provenance. Its dry-run planned 1.47 GiB transfer
and 1.59 GiB output with a 5 GiB reserve. Three focused downloader/manifest tests pass.
The 3,100-video selective download started at approximately 18:46 PST with four workers;
at this timestamp it is active with no observed failures.

The counts in this entry were superseded at 19:03 PST after replacing the accidentally
selected fingerspelling class with lexical `WHAT1`. The corrected total is 3,102.

### 2026-08-09 19:03 PST — fingerspelling guard and raw corpus audit

The first metadata rank selected raw gloss `W.H.A.T` because it had slightly more
coverage than the lexical sign. This was caught before extraction reached that class.
The 30 downloaded fingerspelling clips were moved intact—not deleted—to
`data/local/citizen100_v17/quarantine/w_h_a_t/`. The manifest builder now rejects dotted
fingerspelling whenever an eligible lexical sign exists. WHAT is pinned to raw gloss
`WHAT1`, ASL-LEX `D_02_094`.

The corrected manifest contains 3,102 videos: 1,476 train, 378 validation, and 1,248
test. Selective acquisition completed with 3,102/3,102 official ZIP size/CRC checks and
SHA-256 provenance. Three apparent finalization failures during an overlapping resumed
transfer were traced to shared `.part` filenames; all final files were valid, and temp
names now include the worker thread identity. A clean resume verified 3,102/3,102 with
zero failures.

`active/v17/audit_citizen100_raw.py` reports PASS: all 3,102 videos decode, there are 100
classes, the selected corpus contains 32 train / 5 validation / 11 test participants,
and participant overlap between every split pair is empty. All inputs are landscape:
2,982 at 640x480 and 120 at 960x540. Report:
`artifacts/reports/CITIZEN100_RAW_AUDIT.md`.

Full v17 extraction is now active for train, validation, and test. The train run resumed
from 493 schema-valid archives; no extraction/no-hand failures had appeared at this
timestamp.

### 2026-08-09 19:22 PST — full Citizen100 v17 extraction complete

Extraction completed for the entire corrected manifest. The feature root is
`data/local/citizen100_v17/landmarks/`. Valid archive counts are 1,476 train, 378
validation, and 1,247 test (3,101 total). One test source,
`test/HE/3500609473112364-HE.mp4`, was correctly rejected because all 14 frames contain
no visible hand; its contact sheet is
`artifacts/generated/v17_diagnostics/he_no_hands_contact.jpg`. The rejection is recorded
in `data/local/citizen100_v17/rejections.csv` and is not treated as an extractor bug.

`active/v17/audit_v17.py` reports PASS for all 3,101 archives: every file loads with the
current schema and satisfies shape, finite-value, binary-presence, missing-spatial-zero,
and missing-confidence-zero invariants. There are zero load/schema errors. Median
extraction time was 0.6607 seconds/video. Median detected hand-frame coverage increased
from 0.4667 before activity trimming to 0.875 afterward. Median hand/face/body presence
was 0.4465/0.8125/0.5312. Normalization used shoulders for 2,768 clips and palm fallback
for 333. Corrected chirality counts were 56,823 left, 81,899 right, and zero unknown.
Outputs: `artifacts/reports/CITIZEN100_V17_EXTRACTOR_AUDIT.md` and
`artifacts/reports/citizen100_v17_extractor_audit.csv`.

A train `SLEEP` clip with only four sampled hand detections was manually reviewed. It is
mostly idle and begins signing only at the end, so it is documented in the rejection
ledger while its raw video and already extracted feature remain preserved for traceable
review. A separate `BAD` clip with nine detections was visually valid and retained;
therefore no blanket hand-coverage threshold was introduced.

Focused validation passed 17 tests across extractor geometry/real Vision, Citizen
manifest/downloader, and the optional PopSign downloader. Python compilation, the v17
compatibility CLI help, and the scoped diff whitespace check also passed at this
milestone.

### 2026-08-09 19:47 PST — v17 Stage 1 baseline and landmark-quality gate

`active/v17/model_v17.py` and `train_stage_1_v17.py` now provide a clean Stage 1 path.
The model consumes only the archived `[B, 32, 61, 5]` contract and derives masked XYZ
velocity/acceleration plus valid hand-shape distances internally. Derivatives never
cross a missing observation. The loader uses the official split directories and the
explicit rejection ledger; effective counts are 1,475 train, 378 validation, and 1,247
untouched test archives. Augmentation preserves isotropic geometry and missing zeros,
and reflection swaps anatomical hands, face pairs, shoulders, and elbows. There is no
distillation, aspect stretching, or random video split.

The first capacity choice is d=256/depth=4 (6,470,885 parameters), not d=384/depth=6
(21,154,853 parameters). The older v16 capacity check favored 256, and 21M parameters
is unjustified for only 1,475 training clips. Capacity remains configurable for a
controlled validation ablation.

An initial training attempt revealed that fixed EMA decay 0.999 left validation
dominated by random initialization because there are only about 24 optimizer steps per
epoch. The run was stopped at epoch 10 and preserved under
`artifacts/generated/v17_stage1_aborted_ema999/`. EMA now has a step-count warm start;
the focused regression verifies its first update follows learned weights. A five-epoch
preflight improved from 2.65% to 16.67% validation top-1 under otherwise comparable
conditions.

The corrected baseline early-stopped after 74 epochs. Its best checkpoint is epoch 44
at `artifacts/models/stage1_v17_baseline/best_model.pth`: 93.12% top-1, 99.47% top-5,
and 92.53% macro F1 on 378 clips from the five official validation signers. This is a
validation result, not a final test or portrait-iPhone result. The Citizen test split
has not been evaluated. `evaluate_stage_1_v17.py` requires an explicit `--allow-test`
gate for test access.

The new missingness audit measured all 3,101 archives. Hand-active output-frame
coverage is min/p10/median/p90/max 28.12/81.25/87.50/93.75/100%; zero clips fall below
25%. When a side is active, median joint completeness is 91.67% left and 95.59% right.
Median observed-point confidence is 0.5889. Face-node presence is 78.92%; body/elbow
coverage is lower, but those nodes are auxiliary and explicitly masked.

Visual Apple Vision overlays on the lowest-, median-, and highest-coverage clips show
the detected 21-point hands aligned to visible fingers, including two-handed examples.
The lowest clip, validation SLEEP at 28.12%, is mostly idle with only a short visible
sign; it was nevertheless classified correctly. Validation accuracy by archived
hand-active coverage was 77.78% for the nine clips below 50%, 100% for six clips at
50–75%, 93.33% for 330 clips at 75–90%, and 93.94% for 33 clips at 90%+. The low bin is
small but confirms the coverage tail deserves review; it does not make training
useless. No fake tracking was reintroduced.

Evidence:

- `artifacts/reports/CITIZEN100_V17_LANDMARK_QUALITY.md`
- `artifacts/reports/citizen100_v17_landmark_quality.csv`
- `artifacts/generated/v17_diagnostics/citizen_landmark_overlay_audit.jpg`
- `artifacts/reports/stage1_v17_validation/REPORT.md`
- `artifacts/reports/stage1_v17_validation/predictions.csv`

Focused validation now totals 24 passing tests: the previous 17 plus seven Stage 1
dataset, missing-motion, full-anatomy mirror, augmentation, model-forward, real-count,
and EMA warm-start checks. The MPS optimizer/checkpoint smoke, full validation
evaluation, Python compilation, compatibility CLI help, and scoped whitespace check
also pass.

### 2026-08-09 19:50 PST — mobile algorithm alternatives reviewed

Current primary-source findings:

- Google MediaPipe Hand Landmarker has an official iOS live/video implementation using
  `MediaPipeTasksVision`. It outputs 21 image XYZ landmarks, 21 world-coordinate XYZ
  landmarks, and handedness, and uses tracking in video/live modes to reduce repeated
  palm detection. Google's published full-model Pixel 6 latency is 17.12 ms CPU and
  12.27 ms GPU. This makes it the highest-priority Apple Vision extractor challenger.
- RTMPose-s reports 70+ FPS on Snapdragon 865 for COCO body pose. MMPose also publishes
  an RTMPose-m hand model paired with SSDLite MobileNetV2. It is credible but its hand
  model/mobile iPhone cost is not established by the body benchmark, so it is the
  second extractor challenger rather than an assumed upgrade.
- MoViNet was designed for streaming mobile video with constant-memory stream buffers.
  It is the best direct RGB-video reference. Apple MobileOne reports sub-1-ms backbone
  inference on iPhone 12 for some variants and has the cleaner Core ML path, so a
  MobileOne frame encoder plus small temporal head is the preferred first RGB prototype.
- The existing Squeezeformer family remains competitive for compact temporal landmark
  modeling and has a direct PyTorch-to-Core-ML path. No reviewed evidence currently
  justifies replacing it with a generic Vision Transformer, ST-GCN, or YOLO classifier
  before controlled validation.

Sources:

- `https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/ios`
- `https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker`
- `https://arxiv.org/abs/2303.07399`
- `https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md`
- `https://research.google/pubs/movinets-mobile-video-networks-for-efficient-video-recognition/`
- `https://machinelearning.apple.com/research/mobileone`
- `https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html`

### 2026-08-09 20:00 PST — MediaPipe extractor challenger implemented

The first controlled challenger is the official MediaPipe Hand Landmarker full float16
task bundle, stored at
`artifacts/model_assets/mediapipe/hand_landmarker.task`. It is 7,819,105 bytes and its
SHA-256 is
`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
The source URL is
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`.
The model binary is part of the feature-schema fingerprint, so features from another
model or threshold configuration cannot be silently mixed.

`active/v17/extract_mediapipe_v17.py` now implements an orientation-safe hybrid
candidate: MediaPipe supplies both hands on every sampled frame and Apple Vision may
supply low-rate body/face auxiliary points without redundantly running its hand request.
Video tracking state is recreated for every clip. MediaPipe world-coordinate Z is
wrist-centered, normalized by world-coordinate palm length, interpolated only across
bounded short gaps, and used only where genuinely observed. It is never extrapolated.
The existing scale-depth proxy remains an explicit fallback elsewhere. This behavior is
implemented by `interpolate_scalar_short_gaps` in `active/v17/geometry_v17.py` and is
covered by tests. Apple archives and behavior remain unchanged.

The candidate has its own contract in `active/v17/schema_mediapipe_v17.py`, named
`slt_mediapipe_hand_apple_aux_v17`, and remains shape-compatible at `[32, 61, 5]` while
being fingerprint-incompatible with Apple archives. MediaPipe Tasks handedness agreed
with corrected Apple anatomical labels on unmirrored Citizen frames, so the Tasks API
labels are not swapped in this pipeline.

Real evidence collected before the bakeoff:

- Citizen validation MAKE: MediaPipe hybrid took 1.4698 seconds versus 0.9364 seconds
  for the existing Apple archive on this M4 host. Both covered every trimmed output
  frame; MediaPipe hand-node presence was 68.75% versus Apple's 67.58%. MediaPipe
  provided genuine world depth for 69.17% of hand-node slots. This single clip is only
  a smoke result, not an extractor verdict.
- A real 16-frame HELLO test was bit-exact after a physical 90-degree pixel rotation
  followed by canonicalization, and after horizontal mirroring followed by anatomical
  unmirroring (`max_abs_diff = 0.0` for both). Genuine world depth was nonzero.
- Three new MediaPipe tests pass: bounded scalar depth interpolation, real
  rotate/mirror equivalence plus missing-value invariants, and schema save/load plus
  mismatched-configuration rejection. The 11 existing v17 Apple extractor tests also
  still pass after the common extraction changes.

No winner is declared from these smoke tests. The official Citizen test split remains
sealed. The next gate is a deterministic 100-class train/validation quality bakeoff;
full MediaPipe extraction and classifier training are allowed only if that evidence is
competitive with Apple Vision.

### 2026-08-09 20:24 PST — frozen Apple/MediaPipe extractor bakeoff complete

`active/v17/extractor_bakeoff_manifest.json` freezes 300 train/validation-only clips:
for every one of the 100 classes, the lowest-coverage Apple training clip, the median
Apple training clip, and the lowest-coverage Apple validation clip. It contains 200
train and 100 validation entries, excludes the rejection ledger, explicitly forbids
the test split, and has entry SHA-256
`f27bd7be8bc904c36fa3d57d1c429765ec185e529e94e3297233400e396b36eb`.
`active/v17/extractor_bakeoff_v17.py` creates, validates, extracts, and reports this
protocol. The official Citizen test split was not read.

Both predeclared MediaPipe threshold variants produced 300/300 valid archives with no
failure or no-hand result:

- 0.30 thresholds: schema `79b2eb79820b2f79`, 432.99 seconds wall time.
- 0.50 thresholds: schema `69ae032129a68974`, 392.27 seconds wall time.

On the intentionally difficult 300 clips, Apple versus MediaPipe 0.50 aggregate
medians were: active output frames 87.50% versus 87.50%; hand-node presence 43.38%
versus 46.88%; bone-length CV 0.2539 versus 0.1818 (lower is more stable); extractor
time 0.6780 versus 1.2302 seconds per clip on the M4 host; and genuine detector-depth
coverage 0% versus 46.61%. Mean active output coverage was 79.52% Apple versus 82.24%
MediaPipe. MediaPipe's pre-trim source-hand detection was lower, 38.54% median versus
42.65%, so the higher output coverage partly reflects different trimming/interpolation
and is not treated as an unconditional win. MediaPipe raw confidence is whole-hand
confidence while Apple's is per-joint, so raw confidence was not compared.

The decisive visual audit is in
`artifacts/generated/v17_extractor_bakeoff/disagreements_1.jpg` and
`disagreements_2.jpg`, selected from the largest active-hand and two-hand differences.
MediaPipe generally produced cleaner, more stable skeletons and visibly recovered
useful hands in LISTEN, TRY, TOMORROW, and WEEK. However, it frequently collapsed
overlapping two-hand signs to one detected hand, especially HELP and TIME, missed a
blurred STOP hand, and falsely placed a hand skeleton on a signer's beard/chin in one
COME frame. Lowering the threshold to 0.30 did not consistently repair the overlapping
two-hand failures and sometimes reduced two-hand coverage further. Therefore 0.50 is
the strongest MediaPipe candidate, but Apple remains the incumbent until an identical
full-corpus Stage 1 validation comparison is complete.

Extractor-aware Stage 1 loading/training/evaluation is now implemented without schema
mixing. `--extractor apple|mediapipe_t50` selects an exact fingerprint; checkpoints
record it and the evaluator enforces it. MediaPipe batch extraction now writes its
schema contract and exposes only the reviewed 0.30/0.50 thresholds. The quality audit
also supports MediaPipe 0.50. Eleven focused Stage 1 and MediaPipe tests pass, including
the explicit proof that a MediaPipe archive is accepted only under its exact schema and
is rejected by the Apple loader. Python compilation, CLI help, and scoped whitespace
validation pass.

### 2026-08-09 21:02 PST — full MediaPipe 0.50 train/validation corpus ready

The 300 fingerprint-validated bakeoff archives seeded the full corpus, and resume-safe
extraction processed every remaining official train/validation raw video under schema
`69ae032129a68974`. The test split was not extracted or read.

- Train raw: 1,476. Result: 1,274 newly extracted, 200 validated seeds, two no-hand
  clips, zero failures. One no-hand clip is the already rejected malformed SLEEP clip
  `10667360637258794-SLEEP.mp4`; the only additional omission versus Apple is
  `train/BAD/008134541577476506-BAD.mp4`.
- Validation raw: 378. Result: 278 newly extracted, 100 validated seeds, zero no-hand
  clips, zero failures.
- Effective schema-checked loader counts after applying the existing rejection ledger:
  1,474 train and 378 validation. All 100 classes remain represented; per-class counts
  are 12–18 train and 3–6 validation.

`artifacts/reports/CITIZEN100_V17_MEDIAPIPE_T50_QUALITY.md` audits all 1,852 archives.
Hand-active output-frame coverage is min/p10/median/p90/max
15.62/75.00/87.50/87.50/100%; one clip is below 25%. Pre-trim source hand detection
is 8.93/26.67/41.38/54.55/100%; post-trim is
17.24/73.33/85.98/90.70/100%. MediaPipe emits all 21 joints as a hand or none, so its
reported within-hand completeness is 100% by construction and must not be mistaken for
per-joint ground-truth accuracy. A one-epoch MPS smoke used the exact MediaPipe schema,
loaded 200/100 balanced train/validation samples, completed forward/backward/checkpoint
creation, and kept test evaluation disabled.

The next active gate is an equal Stage 1 run with the same baseline seed, architecture,
augmentations, optimizer, schedule, EMA, and early stopping. Apple remains the selected
extractor unless MediaPipe produces a material signer-disjoint validation advantage
large enough to offset its slower runtime and observed overlapping-hand/false-positive
risks.

### 2026-08-09 21:17 PST — Apple Vision selected as the v17 extractor

The full equal MediaPipe Stage 1 run used 1,474 training and 378 validation clips,
6,470,885 parameters, seed 1701, and the same architecture, augmentations, optimizer,
schedule, EMA warm start, and 30-stale-epoch stopping rule as Apple. It early-stopped at
epoch 140. The best MediaPipe checkpoint is epoch 110 at
`artifacts/models/stage1_v17_mediapipe_t50/best_model.pth`: 89.95% top-1, 97.35%
top-5, and 89.81% macro F1. The corresponding Apple result remains 93.12% top-1,
99.47% top-5, and 92.53% macro F1 at epoch 44. Both use the same 378 clips from the
official five-signer validation split; neither accessed the test split.

Paired predictions show 326 clips correct for both, 12 wrong for both, 26 correct only
for Apple, and 14 correct only for MediaPipe. The exact paired two-sided p-value is
0.0807. This does not establish a population-level statistical guarantee from only five
validation signers, but the engineering decision is clear: Apple leads by 3.17 top-1
points and 2.12 top-5 points, is about 1.8 times faster on the measured M4 clip timing,
and visually handles overlapping two-hand signs better. MediaPipe's lower bone jitter
and genuine world depth do not compensate for its recognition loss, one additional
unusable training clip, overlapping-hand collapse, and observed beard/chin false
positive.

Apple Vision is therefore frozen as v17's extractor. RTMPose/ONNX is not being installed
or benchmarked now: the predeclared rule allowed a second external dependency only if
the native-versus-MediaPipe result was inconclusive, and it is not an attractive iOS
accuracy/latency trade after the native extractor won. This avoids adding a body/hand
detector stack, conversion risk, package size, and mobile runtime cost without evidence
of likely downstream gain.

Detailed final evidence:

- `artifacts/reports/EXTRACTOR_BAKEOFF_V17.md`
- `artifacts/reports/stage1_v17_mediapipe_t50_validation/REPORT.md`
- `artifacts/reports/stage1_v17_mediapipe_t50_validation/predictions.csv`
- `artifacts/generated/v17_extractor_bakeoff/disagreements_1.jpg`
- `artifacts/generated/v17_extractor_bakeoff/disagreements_2.jpg`

Operational docs now identify Apple as selected and MediaPipe as a schema-isolated
challenger. The complete focused v17 suite passes 25 tests, including real Apple and
MediaPipe rotate/mirror equivalence, missing/depth invariants, schema separation,
Citizen split/rejection contracts, model forward/backward behavior, and exact loader
counts. All v17 Python files compile and the scoped whitespace check passes.

The next RGB experiment is refined to the official Apple MobileCLIP2-S0 image encoder
plus a small temporal head, rather than a bare ImageNet MobileOne. It retains a
MobileOne-family mobile path but has substantially stronger image-language pretraining;
Apple reports an 11.4M-parameter image encoder and 1.5 ms image-encoder latency on
iPhone 12 Pro Max, and publishes an iOS demo. This is a Stage 1/RGB experiment, not a
replacement landmark extractor. It must be built in a separate Python 3.10/OpenCLIP
environment because the official project requires Python 3.10 while the validated v17
Vision environment is Python 3.9. Do not install it into the current venv or claim its
sign-recognition accuracy before the controlled train/validation run exists.
Source: `https://github.com/apple/ml-mobileclip`.

### 2026-08-09 21:38 PST — MobileCLIP2-S0 challenger environment verified

The RGB challenger uses the official OpenCLIP model name `MobileCLIP2-S0` and
pretrained tag `dfndr2b`. It is isolated from the validated Apple Vision environment
under `artifacts/generated/mobileclip2_env` with Python 3.10.20, PyTorch 2.13.0,
torchvision 0.28.0, OpenCLIP 3.3.0, and timm 1.0.28. The exact official checkpoint is
cached under `artifacts/model_assets/huggingface`; its SHA-256 is
`ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95` and the full
image-plus-text checkpoint occupies approximately 300 MiB. Only the image tower is
part of the Stage 1 experiment and eventual mobile runtime.

The loaded image tower has 11,406,976 parameters, accepts 256x256 RGB input, and emits
a 512-dimensional embedding. A finite CPU forward pass succeeded. An MPS batch-16
smoke measured 8.39 ms per frame on this M4 host after warmup; this is a desktop
development measurement, not Android or iPhone evidence. The official preprocessing is
resize to 256, center crop to 256, tensor conversion, and mean-zero/unit-standard-
deviation normalization. v17 will letterbox each upright full frame to a square before
that transform so landscape hands are not discarded by the center crop.

MobileCLIP2-S0 is architecturally usable on Android after exporting and validating the
image tower through ONNX Runtime Mobile or LiteRT/TFLite, but Apple does not publish a
ready-made Android app/package in the official repository. Android support and speed
must therefore remain unproven until conversion, operator compatibility, numerical
parity, and NNAPI/GPU/CPU benchmarks run on target phones. Apple's published iPhone
latency must not be projected onto Android.

### 2026-08-09 21:53 PST — Frozen full-frame MobileCLIP2-S0 challenger rejected

The complete frozen-RGB experiment is finished. `extract_mobileclip2_v17.py` sampled
16 upright RGB frames uniformly from each archive's frozen Apple-v17 hand-activity
interval, preserved the full aspect ratio with zero letterboxing to 256x256, and ran
only the official normalized MobileCLIP2-S0 image embeddings. It wrote 1,475 train
archives in 359.6 seconds and 378 validation archives in 95.1 seconds on MPS. Citizen
test is not an accepted CLI split and was not accessed. The RGB feature root is
`data/local/citizen100_v17/mobileclip2_s0` and occupies approximately 36 MiB.

All 1,853 archives have shape `[16, 512]`, finite float16 values, and mean embedding
norm 1.0000003 (observed range 0.9998624–1.0001177 after float16 storage). All Citizen
clips in this selected corpus are landscape. Three clips have one repeated sample
index because their reviewed hand-active interval contains fewer than 16 distinct
source positions; repetition is deterministic and does not fabricate an interpolated
image. The feature/schema fingerprint is `800d51479eb65bbb`.

The temporal challenger is a 3-block, 256-dimensional compact Squeezeformer head with
4,728,037 trainable parameters over the frozen 11,406,976-parameter image encoder.
The one-epoch 200-train/100-validation smoke completed forward, backward, evaluation,
and checkpoint creation. The equal full run used seed 1701, the official 1,475/378
signer-disjoint train/validation samples, label smoothing, AdamW, cosine scheduling,
EMA, and 30-stale-epoch early stopping. It stopped at epoch 96. The best checkpoint is
epoch 66 at `artifacts/models/stage1_v17_mobileclip2_s0/best_model.pth`:
39.68% top-1, 72.49% top-5, and 36.46% macro F1. The frozen Apple-landmark baseline
remains 93.12%, 99.47%, and 92.53% respectively on the identical 378 clips.

A predeclared, per-sample-logit-standardized late-fusion sweep used landmark weights
`[1.00, 0.75, 0.50, 0.25, 0.00]`. Top-1 results were 93.12%, 91.01%, 74.60%, 51.32%,
and 39.68%. Therefore every RGB contribution reduced validation accuracy and pure
Apple landmarks remain the selected v17 system. Do not ship, distill, quantize, or add
this exact frozen full-frame MobileCLIP2 branch to v17. Its global pooled embeddings
discarded too much fine hand-shape and motion information for this task; the
near-training-floor loss alongside poor signer-disjoint validation also indicates
severe domain/signer generalization failure. This result alone does not establish that
a hand-cropped, sign-aware, end-to-end-fine-tuned MobileCLIP2 video model would fail.

Primary artifacts:

- `artifacts/reports/stage1_v17_mobileclip2_s0_validation/REPORT.md`
- `artifacts/reports/stage1_v17_mobileclip2_s0_validation/predictions.csv`
- `artifacts/reports/stage1_v17_mobileclip2_s0_validation/logits.npz`
- `artifacts/reports/stage1_v17_late_fusion_validation.json`
- `artifacts/models/stage1_v17_mobileclip2_s0/history.json`

The isolated RGB code has explicit train/validation-only split enforcement, exact
checkpoint/schema/manifest checks, orientation-safe letterboxing, deterministic trim
mapping, rejection-ledger filtering, atomic archives, and finite/shape validation. The
five new MobileCLIP2 tests pass. Together with the Apple, MediaPipe, Stage 1, Citizen,
PopSign, and aspect-correction focused suites, 37 relevant tests pass. All v17 Python
files compile and the scoped whitespace check passes.

### 2026-08-09 21:58 PST — RGB conclusion narrowed; hand-aware route identified

The user correctly challenged treating a frozen image encoder like a landmark feature
extractor. MobileCLIP2 has no native sign-video classifier: its published objective is
image-text contrastive learning. The completed run froze the visual tower, reduced each
full frame to one globally pooled 512-D vector, and only then modeled time. A temporal
head cannot reconstruct finger articulation or local spatial layout already removed by
global pooling. The head was large enough to drive training loss near its
label-smoothed floor, so the primary failure is representation/domain generalization,
not evidence that Squeezeformer was too small.

The next legitimate RGB experiment, if pursued, is a new hand-aware schema rather than
a replacement temporal head on the old archives:

1. Use Apple Vision only to provide reliable left/right/union hand crop boxes and the
   existing landmark trajectory. Crop actual RGB pixels at high resolution; missing
   detections remain explicitly masked. Short box stabilization may select pixels but
   must never synthesize landmarks or RGB content.
2. Feed shared-weight 224–256-pixel left/right hand crops (and an overlap-aware union
   crop when hands contact) through the MobileCLIP2 visual trunk. Retain spatial feature
   maps or fine-tune the late visual stages instead of freezing and keeping only the
   global image embedding.
3. Add temporal interaction inside the visual backbone using an efficient video method
   such as Temporal Shift Module, or benchmark MoViNet-A0 on the hand-crop sequence.
   Fuse hand appearance with the frozen Apple landmark representation through a small
   gated/cross-attention residual so landmarks retain absolute position and trajectory.
4. Train signer-invariant features with class cross-entropy plus supervised contrastive
   sampling across different signers; use strong background/color/appearance
   augmentation without geometrically corrupting the hands. Fine-tune late stages first
   with a lower backbone learning rate, then unfreeze only if validation supports it.
5. Compare predeclared variants on the same sealed train/validation contract: Apple
   only; frozen hand crops; fine-tuned hand crops with temporal interaction; and
   Apple-plus-hand fusion. Test remains sealed. The RGB branch earns runtime inclusion
   only if it provides a net validation gain and later survives phone profiling.

This direction is supported by sign-specific evidence rather than analogy to ordinary
action recognition. De Coster et al. reported 82.03% validation accuracy for full-frame
VTN, 90.13% after high-resolution hand cropping, and 91.51% after restoring pose-flow
motion. SignRep (ICCV 2025) explicitly addresses the weakness of general visual
pretraining with skeleton-guided sign-specific masked pretraining of a 16-frame Hiera
video model; it is an accuracy reference, not currently a low-end-phone candidate.
Multi-stream sign work likewise reports that local hand/face RGB plus skeleton streams
improve WLASL/MS-ASL recognition. TSM and MoViNet are the relevant mobile temporal
families: TSM introduces feature-level temporal exchange without extra arithmetic or
parameters, while MoViNet is designed for memory-bounded streaming video.

Primary sources:
`https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.html`,
`https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html`,
`https://arxiv.org/abs/2106.15989`,
`https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf`, and
`https://openaccess.thecvf.com/content/CVPR2021/html/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.html`.

### 2026-08-09 22:13 PST — Hand-aware real-pixel crop corpus complete

The new branch is schema-isolated from both landmark archives and the rejected
full-frame embeddings. `extract_hand_rgb_v17.py` uses the selected Apple Vision
detector only to assign anatomical left/right hands and derive crop boxes on the same
16 raw-frame positions and frozen hand-activity interval used by the earlier RGB run.
It stores actual upright source pixels as JPEG byte blobs with explicit offsets, plus
left/right/union validity, normalized boxes, detected-joint counts, contact flags, and
source indices. Invalid views contain no JPEG, decode to exact zero, and remain
`valid=false`; no landmark, box, or RGB content is hallucinated. An overlap-aware union
view preserves two-hand contact and broader spatial context. The schema fingerprint is
`bf6508de2ea851a4`.

All allowed Citizen train/validation clips were extracted: 1,475 train in 206.9 seconds
and 378 validation in 53.9 seconds. Test is not an accepted split and was not accessed.
There are no empty clips. Aggregate valid fractions are left/right/union
42.20%/72.65%/83.49% for train and 49.92%/65.31%/84.56% for validation. Two separately
detected hands occur in 31.36%/30.67% of train/validation sampled frames; crop-box
contact occurs in 22.33%/21.30%. Similar rates across splits are a useful distribution
check. The corpus occupies approximately 669 MiB, with mean packed JPEG payloads of
378 KiB train and 398 KiB validation. A real validation contact sheet was visually
inspected: individual crops materially enlarge fingers, union crops retain body/two-hand
context, and explicitly missing boundary frames are black and masked.

New implementation files are `schema_hand_rgb_v17.py`, `extract_hand_rgb_v17.py`,
`schema_hand_mobileclip2_v17.py`, `extract_hand_mobileclip2_v17.py`,
`model_hand_mobileclip2_v17.py`, and `train_stage_1_hand_mobileclip2_v17.py`. Four crop
geometry/packing/schema tests pass. The frozen high-resolution hand-embedding extraction
and classifier smoke/full runs are the next active gate; the fine-tuned temporal visual
model must not be judged from crop availability alone.

### 2026-08-09 22:30 PST — Frozen high-resolution hand crops validate the correction

The frozen hand-crop diagnostic is complete and confirms that full-frame spatial
resolution was a major cause of the earlier failure. The official normalized
MobileCLIP2-S0 tower encoded all valid left/right/union views; invalid views remained
exact zero. The resulting 1,475/378 train/validation archives occupy approximately
62 MiB, have schema fingerprint `c54f4edc6f62b08b`, contain only finite embeddings,
and have mean valid-view norm 1.0 after float16 storage. Extraction took 536.9 seconds
for train and 116.8 seconds for validation. Test remained inaccessible.

The view-aware model adds normalized crop boxes and learned left/right/union identity,
masked per-frame view attention, temporal modeling, and supervised contrastive loss.
Its 200/100 optimizer/checkpoint smoke passed. The full seed-1701 run early-stopped at
epoch 105; the best checkpoint is epoch 75 at
`artifacts/models/stage1_v17_hand_mobileclip2_frozen/best_model.pth`: 70.37% validation
top-1, 91.27% top-5, and 69.13% macro F1. This is a +30.69-point top-1 recovery over
the frozen full-frame MobileCLIP2 result (39.68%), solely from preserving hand-scale
pixels, view identity, and crop trajectories. It still trails Apple landmarks by 22.75
points and is not a selected runtime branch.

The next gate now operates before global pooling. `extract_hand_spatial_mobileclip2_v17.py`
caches the finite `[16, 3, 512, 8, 8]` FastViT stage-3 spatial maps. The subsequent
model applies per-view temporal shift to those maps and fine-tunes MobileCLIP2's final
visual convolution/projection together with the view-aware sign head. A one-clip smoke
passed with schema `530061b1c5dfcabf`; the measured compressed size is approximately
2.4 MiB per clip, so extraction was started only after confirming roughly 19 GiB free.

No real-phone latency was measured in this session. `devicectl` lists an iPhone14,5
and iPad13,1, but both are unavailable, so desktop timings must not be relabeled as
iPhone timings. Free local space is approximately 23 GiB after the full candidate
artifacts.

### 2026-08-09 22:37 PST — Spatial experiment implementation validation

The pre-pooling path, fusion feature exporter, and zero-initialized gated fusion trainer
compile successfully with all v17 modules. A seventh focused hand/RGB unit test now
proves bit-exactly that fusion logits equal the frozen Apple logits before optimization;
all seven focused tests pass. `active/v17/README.md` was corrected to reject only the
measured frozen full-frame/global-pooling design, not MobileCLIP2 or RGB generally. The
full training spatial-map extraction remains in progress and had written 960/1,475
training clips without skips or failures at this timestamp. The test split remains
sealed.

### 2026-08-09 22:45 PST — Pre-pooling spatial corpus complete and fully audited

All 1,475 training and 378 validation clips were cached from the official
MobileCLIP2-S0 FastViT stage-3 output before global pooling. Extraction took 637.6
seconds for train and 174.5 seconds for validation on MPS, with zero skips and schema
fingerprint `530061b1c5dfcabf`. A full readback audit decompressed every archive and
found zero shape, dtype, schema, finite-value, or invalid-view-zero violations. The
valid-view fractions are 66.11% train and 66.60% validation; mean valid-map absolute
activation is 0.03233 and 0.03213 respectively. The close split statistics are a useful
distribution check. The cache occupies 3.4 GiB and leaves approximately 16 GiB free.
It is temporary training data and is not required by an eventual phone runtime. The
Citizen test split was not accessed.

### 2026-08-09 23:12 PST — Hard temporal-shift result

The optimizer/checkpoint smoke passed at batch sizes 8, 16, and 32; batch 16 was used
for the single full run as the fastest safe measured setting on the 24 GiB M4 host.
Training early-stopped after 29 epochs (15 stale). The saved epoch-14 checkpoint at
`artifacts/models/stage1_v17_hand_mobileclip2_spatial/best_model.pth` reached 70.63%
validation top-1, 88.89% top-5, and 69.41% macro F1. This is only +0.26 point over the
70.37% frozen high-resolution hand-crop model and remains -22.49 points behind the
93.12% Apple landmark model. Later epochs were unstable and did not improve top-1;
epoch 26 matched 70.37% with 91.53% top-5. The result shows that processing spatial
maps before pooling contains a small amount of additional sign signal, but hard TSM is
not a selected standalone branch. The checkpoint reports `test_evaluated=false`.

The next and final active gate is zero-initialized feature residual fusion over the
frozen Apple logits. It is accepted only if signer-disjoint validation exceeds the
Apple baseline; otherwise Apple landmarks remain the selected extractor/model. A
possible future RGB ablation is identity-initialized residual temporal mixing, because
hard TSM disrupts the frozen representation at initialization, but it is not required
to decide the present v17 extractor.

### 2026-08-09 23:15 PST — Conservative fusion result does not displace Apple

The first fusion feature export exposed and fixed a real broken interface: the Apple
model supports `return_embeddings=true`, not a `forward_features` method. The failed
attempt wrote no archive. The corrected aligned exports contain finite 256-D features,
identical item IDs/targets, and exactly reproduce 99.59%/93.1217% Apple train/validation
top-1 and 100%/70.6349% spatial-MobileCLIP train/validation top-1.

The zero-initialized gated feature residual is bit-exactly the frozen Apple logits at
initialization. Canonical seed 1701 reached 93.92% validation top-1, 98.94% top-5, and
93.46% macro F1 at epoch 9, versus Apple's 93.12%/99.47%/92.53%. It fixed nine Apple
errors but broke six Apple-correct clips: only +3/378 net. The exact two-sided paired
McNemar/binomial p-value is 0.607, so this is not statistically persuasive. Four
additional diagnostic seeds all exceeded Apple individually (93.65, 93.92, 94.18,
94.44%; five-seed mean 94.02, population SD 0.27), but averaging their logits returned
exactly to 93.12% and produced five fixes versus five regressions. This cancellation
shows that the small residual corrections are seed-sensitive despite favorable
per-seed validation checkpoint selection.

Therefore the fusion branch is retained as research evidence but is **not** the mobile
selection. Adding a 48-view MobileCLIP visual pass for a non-significant, seed-sensitive
net three-clip gain contradicts the low-end offline goal. Apple Vision landmarks plus
the v17 Squeezeformer remains the selected Stage-1 extractor/model at 93.12% signer-
disjoint validation top-1. The official test split remains sealed; none of the fusion
or robustness runs accessed it.

### 2026-08-09 23:17 PST — Frozen Apple model evaluated once on official test

After all extractor/model challengers, fusion diagnostics, and hyperparameter choices
were complete, the Apple Vision landmark model was frozen and the explicit test gate
was opened exactly once with `evaluate_stage_1_v17.py --split test --allow-test`.
Checkpoint epoch 44 achieved 87.57% top-1 (1,092/1,247 correct; Wilson 95% interval
85.62–89.29%), 98.64% top-5, and 87.39% macro F1 across the official 11-signer test
partition. This is 5.55 points below the six-signer validation top-1 and is the honest
generalization result; 93.12% must no longer be described as test accuracy.

The selected classifier has 6,470,885 parameters and its current float checkpoint is
approximately 25 MiB. This is compatible with a serious Core ML deployment attempt but
does not by itself prove low-end-phone viability; measured conversion accuracy, runtime
memory, sustained latency, and thermals remain required.

The immutable evaluation artifacts are under
`artifacts/reports/stage1_v17_test_frozen_apple/` (`metrics.json`, `logits.npz`,
`predictions.csv`, and `REPORT.md`). The most frequent reciprocal confusion is
ANSWER↔GO (7/6 clips); GOOD→THANKYOU also occurs seven times. These errors may motivate
new independently collected training data, but this test partition must not guide
further checkpoint or hyperparameter selection.

### 2026-08-09 23:19 PST — Final focused validation

All 34 `test_v17*.py` tests pass, including real Apple Vision rotation/mirror
equivalence, real MediaPipe orientation parity, schema separation, missing-value
contracts, real archive counts, hand-crop packing, TSM masking, temporal-model
forward/backward, and bit-exact fusion initialization. Three Citizen downloader tests,
three PopSign downloader tests, and four legacy aspect-correctness tests also pass: 44
focused tests total. All `active/v17`, `src_v17`, affected scripts, and v17 tests compile.
`git diff --check` passes. No hand-RGB, hand-embedding, or hand-spatial files exist under
their Citizen test output directories. Expected Python 3.9 EOL/LibreSSL and MediaPipe
delegate warnings occurred but caused no failures.

### 2026-08-10 09:25 PST — Streaming MoViNet baseline completed; Kaggle failure reconciled

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

### 2026-08-10 10:37 PST — Higher-quality external data audit identifies RIT Sign Bank lead

Official metadata and access paths were audited before acquiring another training
corpus. The evidence is consolidated in
`artifacts/reports/CITIZEN100_EXTERNAL_DATASET_AUDIT.md`. The strongest immediately
auditable video source is the newer ASLLRP/RIT isolated-sign collection: its current
official metadata has 12,197 segmented clips, 13 participant IDs, explicit main-entry
and entry/variant glosses, handshape metadata, and 60 frozen Citizen100 classes with
exact entry/variant-name matches totaling 541 clips. The segmented clips are stored in
two official range-addressable ZIP archives. One byte-range-retrieved smoke clip was
1280x720 H.264 at 30 fps and 1.7 seconds. Apple Vision v17 extracted it in 0.73 seconds
with 96.08% observed hand-frame coverage, 96.88% face/body presence, and a clean
`audit_v17.py` pass. This was a temporary audit clip, not retained project data.

RIT/ASLLRP data are research-only, noncommercial, and non-redistributable under the
Sign Bank terms. Exact text matches are only candidates: the frozen Citizen ASL-LEX
code still must be reconciled with each ASLLRP entry/variant before any clip becomes
training-eligible. A bounded three-participant-per-class audit is the next safe action;
it must remain quarantined until variant review and frozen-model triage pass.

Other official sources are weaker for the current need. ASL-100-RGBD offers 1080p RGB,
4,150 tokens, and 22 fluent/DHH signers, but only 12 direct frozen-label matches plus six
variant families and requires an authorized Databrary account. Legacy ASLLVD has 9,747
tokens from six signers and detailed variants, but its public pre-cut movies stack front
and side views in a 328x656 frame; clean citation-form downloads require login. MS-ASL
has the best secondary signer diversity: 97 name-overlap classes include 2,612 official
train clips from 110 train signer IDs, but nearly all are variable-quality YouTube
segments and exact ASL-LEX variants are not encoded. Google's Kaggle `asl-signs`
competition exposes MediaPipe landmark Parquets rather than raw RGB video, so it cannot
be re-extracted through Apple Vision v17. No dataset video beyond the two temporary
quality smokes was acquired, and PopSign/local selection remains paused.

### 2026-08-10 10:23 PST — PopSign preview audit completed; training use remains blocked

Two provenance-locked scripts and focused tests were added for a PopSign/Citizen100
variant audit: `scripts/download_popsign_citizen100_previews.py`,
`scripts/evaluate_popsign_citizen100_variant_audit.py`, and their two test modules.
The downloader found 43 exact frozen-label overlaps using only exact names plus the
declared GOODBYE->bye, MOTHER->mom, and FATHER->dad aliases. It acquired three
distinct official train-participant previews per class, 129 clips total, under
`data/local/popsign_citizen100_variant_audit/raw/`. The provenance manifest marks every
preview `training_eligible: false`: PopSign's website previews are deliberately
downsampled and speed-normalized audit media, not the original recordings. The
original PopSign v1 clips are mostly 1944x2592, so the preview quality does not imply
that the source corpus itself is low resolution.

Apple Vision v17 extraction succeeded on all 129 previews with zero no-hand or failed
clips. Frozen Citizen100 model agreement was 33.33% top-1 and 65.12% top-5. The
non-benchmark triage classified 9 labels as model-consistent, 23 as ambiguous, and 11
as high-risk; it approved no class for training because exact lexical equivalence
still requires ASL-fluent/Deaf review. The report is
`artifacts/reports/popsign_citizen100_variant_audit/REPORT.md`. All three focused
preview/audit tests pass. PopSign v2.1 pages currently claim 562 signs, but tested
official archive/preview resource URLs returned HTTP 404, so no v2.1 data was acquired
or treated as available.

### 2026-08-10 10:45 PST — RIT exact-name acquisition plan validated

`scripts/download_rit_citizen100_candidates.py` and its three focused unit tests were
added. The downloader reads the official dated RIT metadata, requires literal equality
on the ASLLRP `entry/variant gloss label`, indexes the two 1.7-1.9 GiB official ZIPs by
HTTP byte range, and retrieves only selected members with size and CRC verification.
Every result is quarantined with complete source provenance and
`training_eligible: false` until ASLLRP-to-ASL-LEX variant confirmation. The selection
distinguishes pinned Citizen raw-gloss equality from weaker canonical-label-only
equality and never normalizes or merges variant suffixes.

The focused tests pass. A live dry-run indexed both remote archives successfully and
found 292 unique candidate clips across 60 frozen classes: 249 clips in 51 classes are
literal pinned-raw-gloss matches, while 43 clips in nine classes are canonical-only
candidates because the Citizen raw gloss is a numbered/different variant. All 292
members exist in the official archives. At the user's direction, acquisition order is
RIT and ASL-100-RGBD first, then strictly quality-filtered local raw clips, and only
then another web source if coverage remains insufficient.

At the user's direction, PopSign and local-corpus candidate selection are now paused
while higher-quality external isolated-sign datasets are evaluated first. No PopSign
preview may enter training, and no local supplement has been selected or extracted.

### 2026-08-10 10:54 PST — Selective RIT acquisition completed; ASL-100 access audited

The selective RIT downloader completed 292/292 candidate transfers from the official
segmented archives with member-size, ZIP CRC, decode, and SHA-256 provenance checks.
The retained raw subset is 76,372,759 bytes (73 MiB), all clips decode as 1280x720,
and the candidates span 60 frozen classes and 13 participant IDs. The exact tiers are
249 pinned-Citizen-raw-gloss matches and 43 weaker canonical-label-only matches. Every
clip and the top-level manifest remain `training_eligible: false`; acquisition alone
does not establish lexical equivalence or suitability for training.

Databrary volume 1062 was also inspected through its current public API. The volume
metadata confirms 42 1080p RGB sequences from 22 fluent/DHH signers, but its 22 video
sessions have `releaseLevel: authorized_users`, `nativeAccessible: 0`, and blurred
session metadata without an authorized institutional login. No ASL-100-RGBD media was
downloaded. Keep it queued for authorized access; do not bypass the release gate.

The active order is now: extract and triage the acquired RIT candidates, acquire
ASL-100-RGBD if authorized access becomes available, audit local raw videos and retain
only genuinely strong, diversity-aware, train-only clips under a strict per-class cap,
then search further official web datasets only if coverage remains insufficient.

### 2026-08-10 10:56 PST — RIT v17 extraction and frozen-model triage completed

Apple Vision v17 extraction succeeded on all 292 selected RIT clips with zero no-hand
and zero failed cases. `audit_v17.py` passed all 292 archives with zero schema errors.
Extraction quality is strong: median observed-hand-frame coverage is 97.59%, median
hand-node presence is 86.91%, median face presence is 93.75%, and median body presence
is 93.75%. These measurements support the capture quality of the source but do not
establish lexical equivalence.

The frozen Citizen100 checkpoint was used only for mismatch triage, never as an
accuracy benchmark. Clip agreement is 38.36% top-1 and 60.62% top-5. Of 60 candidate
classes, 16 are model-consistent, 22 ambiguous, and 22 high-risk; zero classes were
automatically approved. The detailed immutable outputs are under
`artifacts/reports/rit_citizen100_variant_audit/`. ASL-fluent/Deaf review of the exact
ASLLRP entry/variant against each pinned Citizen ASL-LEX code remains required before
any RIT clip can enter a train-only supplement.

### 2026-08-10 11:03 PST — Local raw corpus quality shortlist completed

The local raw corpus was audited only after the RIT pass, as directed. A new
fail-closed selector and three focused tests were added in
`scripts/audit_local_citizen100_candidates.py` and
`test/test_local_citizen100_candidates.py`; all tests pass. The selector excludes all
known `msasl_`, `signasl_`, and `wlasl_` files, requires valid isolated-sign duration,
resolution, exposure, sharpness, and a conservative 0.82 composite quality floor, and
uses coarse appearance distance only to avoid near-duplicate recording sessions. It
does not infer or claim signer identity.

Of 17,179 local-style candidates inspected, 356 clips across 89 exact class folders
were shortlisted under a hard cap of four clips per class. This is intentionally much
smaller than Citizen's primary corpus so the local seven-ish recurring people/sessions
cannot dominate. Ten frozen classes have no exact local folder. The `I` folder was
visually confirmed to mix fingerspelled-I and ME/self-reference productions and is
fully quarantined, leaving 89 usable audit folders rather than 90. The shortlist has a
minimum quality score of 0.82, minimum brightness 36.83, and minimum pairwise selected
appearance distance 0.171. All clips remain `training_eligible: false`, train-only
after exact variant review, and are staged as symlinks under
`data/local/local_citizen100_quality_audit_q82/raw/`. A 356-clip visual contact sheet
is at `artifacts/reports/local_citizen100_quality_audit/shortlist_contact_sheet_q82.jpg`.

An earlier 356-symlink pre-floor audit remains under
`data/local/local_citizen100_quality_audit/` because local data must not be deleted
without explicit permission. It is superseded and must not be extracted or trained.
The q82 shortlist is now queued for Apple Vision v17 extraction and frozen-model
mismatch triage; selection itself is not approval.

### 2026-08-10 11:07 PST — Local v17 triage yields a 132-clip human-review shortlist

Apple Vision v17 extraction succeeded on all 356 q82 local candidates with zero
no-hand and zero failed clips; all 356 archives passed `audit_v17.py`. Frozen-model
agreement, used only as a mismatch screen, is 51.69% top-1 and 81.18% top-5. The
class-level result is 54 model-consistent, 27 ambiguous, and eight high-risk classes.
The high-risk labels are COME, GOODBYE, SIGN, TAKE, UNDERSTAND, WANT, WHAT, and YOUR.
No class or clip was automatically approved.

`scripts/select_local_citizen100_review_shortlist.py` and three focused tests were
added; all pass. A stricter final review shortlist keeps only pinned-raw-text-equal,
model-consistent classes and clips that are frozen-model top-5 hits with at least 80%
observed hand-frame coverage and 50% face presence, then caps each class at three.
This leaves 132 clips across 49 classes under
`data/local/local_citizen100_quality_audit_q82/review_raw/`, with provenance in
`review_shortlist.json`. They remain `training_eligible: false` and require ASL-fluent
exact ASL-LEX variant review. Trustworthy signer counts remain unavailable, so these
clips can only become a small train-only supplement and never validation/test evidence.

Because only 49/100 classes survive the conservative local review gates and RIT still
requires independent lexical review, external-data coverage is not yet sufficient.
The next action is a further official-source web pass, starting with a bounded,
quality-filtered MS-ASL train-only audit rather than bulk downloading web video.

### 2026-08-10 11:29 PST — Bounded MS-ASL gap acquisition completed

The official Microsoft MS-ASL annotation package was downloaded from Download Center,
SHA-256 `a8562008309eea4129e1bc0ed7f654a314fee195227222859657e307b6434c34`, and
retained under `data/local/dataset_metadata/msasl_official/`. Its C-UDA license and
official train/validation/test annotations are preserved. Only `MSASL_train.json` was
used; validation and test were not accessed for candidate acquisition.

`scripts/download_msasl_citizen100_gap_candidates.py` and five focused tests were
added; all pass. The downloader considers only canonical labels whose text exactly
equals the pinned Citizen raw gloss and that are absent from the 49-class strict local
review shortlist. It requires official annotation resolution of at least 640x360,
isolated segments of 0.4-8 seconds starting within the first 120 source seconds,
unique official train signer IDs per class, and at most three retained clips per
class. Attempts within a class are sequential and stop at the target, so the process
cannot download surplus clips. An isolated current yt-dlp 2026.07.04 environment is
under `artifacts/generated/msasl_download_env`; every retained segment is decode- and
SHA-256-verified.

The bounded pass made 192 attempts and retained 62 clips across 30 classes, totaling
24,710,250 bytes in provenance. Eight classes reached 3/3: BIG, FRIEND, GOOD, HAPPY,
LIKE, MOTHER, SICK, and SIGN; WE also reached 3/3, for nine total target-saturated
classes. COME, GIVE, GO, STOP, and YES had zero currently valid bounded sources. Other
classes retained one or two clips. The exact retained set is materialized as 62
symlinks under `data/local/msasl_citizen100_gap_audit/retained_raw/` and recorded in
`candidate_provenance.json`; all are `training_eligible: false` pending v17 extraction,
frozen-model mismatch triage, and ASL-fluent exact-variant review.

Nineteen successfully downloaded clips from the interrupted pre-120-second-filter run
remain in the broader `raw/` directory but are absent from current provenance and must
not be extracted or trained. They were not deleted because local data deletion requires
explicit permission. Only `retained_raw/` is the active MS-ASL audit input.

### 2026-08-10 11:35 PST — MS-ASL triage completed; remaining high-quality sources are gated

Apple Vision v17 extraction succeeded on all 62 provenance-linked MS-ASL clips with
zero no-hand and zero failed cases; all 62 archives passed `audit_v17.py`. Extraction
quality is strong: median observed-hand-frame coverage is 95.1%, median face presence
90.6%, and median body presence 87.5%. Frozen-model mismatch triage is 56.45% top-1
and 77.42% top-5, with 11 model-consistent, 13 ambiguous, and six high-risk classes.
The model-consistent classes are ANGRY, BIG, FRIEND, GOOD, HAVE, HOT, LIKE, SICK,
UNDERSTAND, WE, and WHY. Zero classes were automatically approved.

The conservative union now covers 63/100 classes: 49 exact-text/model-consistent local
classes, 11 new MS-ASL classes, and three additional exact-tier RIT classes not already
in that union (GIVE, LESS, WHEN). This is candidate coverage only, not approved training
coverage; all sources still require ASL-fluent exact-variant review.

Further official web research found no additional ungated, clearly higher-quality video
source that can be safely acquired now. ASL-LEX reference videos explicitly may not be
saved or used without permission. Sem-Lex is the best next corpus (91,148 videos,
3,149 signs, 41 Deaf participants, expert ASL-LEX/SignBank alignment), but the named
user must submit its Google access form and personally accept CC BY-NC-SA and
community-respect commitments. Purdue RVL-SLLL requires a signed license and issued
credentials. ASL-100-RGBD remains behind Databrary authorized-user access.

Additional ASLLRP metadata was audited without downloading gated video. The 2025
ASLLRP sentence metadata has 1,992 exact-name candidate tokens across 65 frozen classes;
DSP sentence metadata has 317/60, and DSP citation-form metadata has 142/65. The
official ASLLRP interface states that segmented-video downloads require a free login.
Do not use the exposed archive index to bypass that account gate. Metadata SHA-256s and
the current comparison are recorded in
`artifacts/reports/CITIZEN100_EXTERNAL_DATASET_AUDIT.md`.

Focused validation after all acquisition/audit changes passes 23/23 unit tests across
the RIT downloader/triage, local quality/diversity/review gates, and MS-ASL bounded
downloader/triage. All seven new scripts compile, and `git diff --check` passes. Final
on-disk active counts are 292 RIT raw/292 landmarks, 132 strict local review symlinks,
and 62 retained MS-ASL symlinks/62 landmarks. Free disk space is approximately 56 GiB.

### 2026-08-10 12:04 PST — Local 132-clip ceiling corrected; SemLex exact plan started

The earlier 132 local clips were not the total number of strong clips. They resulted
from a deliberately narrow four-candidate audit, a class-consistency screen, and a
three-per-class output cap. At the user's request, the same 0.82 visual-quality floor
and 0.08 appearance-diversity constraint were rerun with a seven-per-class cap, which
selected 623 visually strong candidates across 89 classes from the same 17,179 local
clips. Apple Vision v17 extracted all 623 with zero failures/no-hand cases and
`audit_v17.py` passed all 623 archives with zero errors.

Frozen-model mismatch triage on the expanded set is 52.33% top-1 and 80.26% top-5.
Class triage is 39 model-consistent, 40 ambiguous, and ten high-risk. Applying only
the exact-text, clip top-5, >=80% observed-hand-frame, >=50% face-presence, and
seven-per-class gates yields a 369-clip/77-class human-review pool. Adding the stricter
class-level model-consistency gate yields 209 clips across 38 classes. The selector now
writes both `clip_review_pool.json` (369) and `review_shortlist.json` (209) under
`data/local/local_citizen100_quality_audit_q82_cap7/`; neither is training-approved.
The 209 set is the conservative local supplement, while the wider 369 set must not be
silently promoted without ASL-fluent review.

The user supplied named-access SemLex metadata and six official Google Drive links.
CLI range inspection identified three video archives and three pose archives. Only the
23,673,462,199-byte official train video archive is in scope; SemLex val/test and all
provided pose archives are excluded. Official ASL-LEX 2.0 metadata was downloaded from
OSF under its CC BY-NC license (`signdata.csv` SHA-256
`080ecc3de4b307a04dd9b2c2583c22bc623a5c731ab869d6e3905d4a3540fbd3`). The supplied
SemLex metadata SHA-256 is
`f4250b2877e738f028e4b9517922952a350ec8952e59d53ceabdd778938fc3c3`.

`scripts/prepare_semlex_citizen100_candidates.py` performs an exact join from each
pinned Citizen ASL-LEX code through the official ASL-LEX EntryID to SemLex `asllex`
labels; English-name similarity and free-text labels are never accepted. The initial
five-signer cap (486 clips) was only a conservative starting point and was superseded
after the user questioned it. Acquisition and training balance are now separate: the
active acquisition plan retains all 1,624 exact matched clips (one clip per official
SemLex signer/class, 98 classes, 32 train signers), while the first balanced training
subset retains at most 12 distinct SemLex train signers per class (1,091 clips, 98
classes, 29 signers). The latter roughly matches Citizen's 11–16 train signers per
class while remaining below Citizen's 1,475 training clips overall. `THEY` is excluded
because Citizen pins
`they_2` while SemLex train provides `they_1`; `CHILD` is excluded because the SemLex
entry is `children`, not the pinned `child` entry. The full acquisition plan is under
`data/local/semlex_citizen100_train_audit/selection_plan.json`; the balanced subset is
under `data/local/semlex_citizen100_train_audit/balanced_cap12/selection_plan.json`.
Both remain `training_eligible:false` until extraction and quality triage complete.

`scripts/download_semlex_citizen100_candidates.py` and five focused SemLex planning /
download tests were added and pass. Google packages train as one gzip stream, so member
bytes cannot be remotely selected. A resume-safe parallel CLI range transfer is in
progress on the internal SSD; after the full transport passes range/length checks, the
script will extract and decode only the planned 1,624 WebM members and remove the
23.67 GB transport archive. Do not treat the transfer's sparse transport file as a
dataset or use any SemLex clip until selective extraction, v17 audit, and mismatch
triage complete.

Focused validation passes 15/15 tests across the local selector/triage/two-level
review outputs and SemLex exact mapping/range-member logic. All five involved scripts
compile and `git diff --check` passes. At 12:10 PST the throttled Google transport had
29/2,823 independently validated 8 MiB ranges complete (243,269,632 validated bytes)
with additional in-flight sparse-file bytes; the resume state is
`transport/train.tar.gz.ranges.json` and the CLI process remains active.

At 12:35 PST the first high-concurrency transfer process was stopped after Google read
timeouts; its 95 independently completed 8 MiB ranges were preserved. The same sparse
archive/state resumed with 16 workers and 20 per-range attempts. Before final resume,
the acquisition plan was expanded from the provisional five-signer cap to all 1,624
exact one-per-signer/class clips, while the separate 1,091-clip cap-12 training subset
was preserved. No validated range was redownloaded and eventual selective extraction
will use the full acquisition pool.

At 12:40 PST the user chose to download the official SemLex train video archive
manually. The active CLI range transfer was terminated (exit 143), and process checks
confirmed no SemLex downloader remained. At the user's explicit deletion request, the
sparse `transport/train.tar.gz`, its `train.tar.gz.ranges.json` checkpoint, and the
earlier `train_prefix_64m.tar.gz.partial` probe were permanently removed. No SemLex raw
video had yet been selectively extracted. The official metadata, full 1,624-clip
acquisition plan, and balanced 1,091-clip training plan were preserved. When the user
places a clean `train.tar.gz` locally, continue with selective extraction only; do not
start another Drive download.

### 2026-08-10 14:38 PST — SemLex train extracted and v17-audited; val/test policy clarified

The user manually downloaded the official SemLex `train.tar.gz` to
`/Users/frnzlo/Downloads/train.tar.gz`. It has the exact expected size
(23,673,462,199 bytes) and passed a full `gzip -t` integrity check. The original
Downloads archive was not modified or removed. `--skip-download` support was added to
the bounded SemLex extractor so this local-archive run made no Drive request.

The full 1,624-member exact-ASL-LEX acquisition plan was scanned against the archive.
1,499 clips across all 98 matched classes and 32 SemLex train signers decoded and were
retained (564,485,964 bytes). The other 125 are explicitly rejected: 115 metadata video
IDs are absent from the official train archive, and ten VP9 WebM members fail full-frame
decode validation. Decode-broken members are quarantined under `rejected_raw/`; missing
members have provenance but no fabricated file. The immutable retained/rejected record
is `data/local/semlex_citizen100_train_audit/download_provenance.json`.

SemLex WebM support was added to the v17 batch inventory with a regression test. Apple
Vision v17 extracted 1,499/1,499 retained clips with zero no-hand and zero failed cases;
`audit_v17.py` passed all 1,499 archives with zero schema/invariant errors. Extraction
quality medians are 86.21% observed hand frames, 50.59% hand-node presence, 78.12% face
presence, and 43.75% body presence.

Frozen Citizen model agreement is used only as a cross-domain mismatch diagnostic:
72.85% top-1 and 89.86% top-5 across 1,499 clips. Class triage is 76 model-consistent,
21 cross-domain ambiguous, and one mismatch-review priority (`TAKE`, 2/10 top-1 and
3/10 top-5). Because SemLex is expert-aligned through the exact pinned ASL-LEX entry,
ambiguous classes are not discarded merely for Citizen-model disagreement. `TAKE` is
withheld pending review. A quality-ranked maximum-12-per-class first supplement now
contains 1,058 clips across 97 classes and all 32 retained train signers, materialized
as symlinks under `balanced_raw/` and `balanced_landmarks_v17/`; provenance is
`balanced_train_candidates.json`. It remains `training_eligible:false` until the final
review/approval decision.

SemLex validation/test archives are not required to maintain a numeric split ratio.
The first controlled augmented run should train on Citizen official train plus the
balanced SemLex train-only supplement and continue selecting checkpoints on Citizen's
fixed official validation split. SemLex validation could later be acquired as a
secondary unseen-SemLex-signer diagnostic, not merged into the primary selection
metric. SemLex test should remain untouched until a final frozen model warrants a
one-time independent SemLex-domain evaluation. Never rerun the already-consumed
Citizen official test during development.

Focused validation passes 19/19 tests: 12 v17 extractor tests (including WebM
inventory), five SemLex download/planning tests, one frozen-triage aggregation test,
and one balanced-selector test. The new/modified scripts compile and `git diff --check`
passes. Current SemLex audit storage is about 567 MiB; the
separate original 22.05 GiB archive remains in Downloads.

### 2026-08-10 14:52 PST — Local A-Z fingerspelling audit passed as a separate track

The planned Citizen-train plus balanced-SemLex first run contains 2,533 training clips
(1,475 Citizen and 1,058 SemLex) for the fixed 100 lexical classes. This is enough for
a meaningful controlled augmentation experiment, but not evidence of production
robustness. None of the fixed 100 labels are alphabet classes: canonical `I` is pinned
to the lexical ASL-LEX entry `ME`, not the fingerspelled letter I. Alphabet clips must
remain a separate 26-class fingerspelling model/head; J and Z retain their motion.

The local `data/raw_videos/ASL VIDEOS/{A..Z}` corpus was visually and mechanically
audited before considering another web download. It contains genuine A-Z fingerspelling
from multiple visible people/environments but also repeated sessions, scraped-source
copies, exact `__from_MARIAH_` duplicates, and weaker clips. The conservative selector
`scripts/audit_local_alphabet_candidates.py` inspected 6,307 MP4 candidates, excludes
known duplicate/scraped/unknown sources and quality/duration failures, caps named
single-session contributions, and retained exactly 312 clips (12 per letter). Selection
provenance is `data/local/local_alphabet_quality_audit/candidate_selection.json`; raw
materialization uses symlinks and all candidates remain `training_eligible:false`.

Apple Vision v17 extracted 312/312 shortlisted alphabet clips with zero no-hand and
zero failed cases. `audit_v17.py` passed all 312 archives with zero errors. Median
observed-hand-frame coverage is 100% and median hand-node presence is 50%; the videos
are uniformly 640x480 and are often tight hand crops, so median face presence is 87.5%
while body and shoulder coverage are 0%, using the extractor's wrist normalization
fallback. The local clips are technically good enough for train-only fingerspelling
experiments, so no web alphabet dataset was downloaded. They are not a credible
validation/test set because signer identities and independence are not established;
future alphabet accuracy claims require a signer-disjoint labeled evaluation source.

### 2026-08-10 14:55 PST — SemLex validation/test acquisition deferred past augmented v1

The official metadata was re-counted for the 98 exact SemLex matches. Train contains
5,539 rows / 1,624 unique signer-class pairs from 32 signers; validation contains
1,861 rows / 984 unique signer-class pairs from 32 signers; and test contains 1,618
rows / 444 unique signer-class pairs across 95 matched classes from ten signers.
SemLex train and validation overlap on 31 of their 32 signer IDs, and 822/984 validation
signer-class pairs already occur in train. Test has zero signer overlap with either.

Therefore validation/test are not free additional training data. The controlled
augmented-v1 experiment remains Citizen official train plus the quality-balanced
SemLex-train supplement, selected on Citizen official validation. SemLex validation
may be downloaded after v1 for a secondary within-domain diagnostic and can only be
promoted into a later training version after its diagnostic role is finished. SemLex
test must remain protected until a final frozen candidate needs a one-time unseen-
SemLex-signer evaluation; sacrificing it for training is especially unsound because
the Citizen official test has already been consumed. No validation or test archive was
downloaded during this decision.

### 2026-08-10 15:14 PST — Citizen + SemLex augmented Stage 1 completed; baseline retained

The Apple Vision extraction path was already fixed before this run: 1,475 usable
Citizen-train archives, 378 fixed Citizen-validation archives, and all 1,058 selected
SemLex-train archives pass the v17 schema/invariant audit. Training does not create
physical duplicate archives. `augment_v17` produces new random variation online on
every training batch through anatomy-correct left/right reflection, masked isotropic
scale/rotation/translation, masked coordinate noise, nearest-frame temporal warping,
and masked joint dropout. Binary presence and exact missing-zero contracts remain
preserved.

`active/v17/train_stage_1_v17.py` now optionally accepts a provenance-locked SemLex
supplement. `SemLexSupplementV17Dataset` requires a `train_only` manifest, rejects any
non-train clip, validates every label/video/archive/schema, and requires the explicit
`--approve-supplement` run gate. A supplemented run cannot request Citizen test.
Checkpoints and results record both dataset hashes/counts, approval state, online
augmentation, and false test-access flags. The real combined loader reports exactly
2,533 train clips (1,475 Citizen + 1,058 SemLex), 378 Citizen validation clips, 100
classes, and 6,470,885 parameters. A two-batch MPS optimizer/checkpoint smoke passed.

The first full local attempt was stopped at epoch 20 when the user initially redirected
training to Kaggle; its recoverable files are under
`artifacts/generated/stage1_v17_citizen_semlex_interrupted_epoch20/`. No Kaggle kernel
was uploaded or started. After confirming this landmark-only run is short, the user
chose local MPS and a clean seed-1701 run restarted from scratch. It completed in about
8.6 minutes and early-stopped after 67 epochs / 30 stale epochs. The best checkpoint is
epoch 37 at `artifacts/models/stage1_v17_citizen_semlex_augmented/best_model.pth`:
92.86% top-1 (351/378), 100.00% top-5 (378/378), and 92.25% macro F1 on Citizen
validation. Citizen and SemLex test were not accessed.

The primary Citizen-only baseline remains selected: its 93.12% top-1 (352/378) is one
clip better, with 99.47% top-5 and 92.53% macro F1. The augmented challenger corrected
12 former baseline errors but regressed 13 formerly correct clips, so the one-clip net
loss is real rather than identical predictions. The result is promising cross-domain
ranking coverage but not a top-1 win; do not replace the baseline or tune against the
consumed Citizen test. The validation-only augmented report is under
`artifacts/reports/stage1_v17_citizen_semlex_augmented_validation/`.

### 2026-08-10 15:23 PST — v16 96% protocol audited against v17 generalization

The v16 96.00% number is real for its recorded protocol but is not comparable to the
v17 signer-disjoint result. Its deep-cleaned corpus has 62,023 clips and 310 classes,
split by deterministic per-class random clip shuffle into 43,263 train, 9,311
validation, and 9,449 test clips. The loader explicitly says that this mixes every
video source/signer across splits. The local corpus has approximately seven recurring
people/sessions per class, so v16 validation/test mainly measure new clips from familiar
people and capture conditions rather than new-signer generalization.

The v16 cleaning loop also leaks evaluation information. A checkpoint trained on the
random-split corpus scored all 66,770 original samples, including clips later assigned
to validation/test, and removed the bottom 3% by model self-confidence. Statistical
cleaning then removed another 2,744 coordinate/jitter/distribution/low-motion outliers;
4,747/66,770 clips (7.1%) were removed before the final split was recomputed. This can
improve real label/landmark quality, but it also favors the existing model's decision
boundary and allows future evaluation clips to influence corpus selection.

An exact SHA-256 audit of every deep-cleaned v16 `.npy` found 60,999 unique arrays among
62,023 files: 1,024 duplicate pairs remain. Of these, 444 cross split boundaries;
221 validation files and 175 test files have an exact array duplicate in training.
There are 399 cross-split same-label duplicate pairs and 45 cross-split conflicting-
label pairs. Frequent conflicting aliases include ABOUT/WHEN (59 total pairs) and
WORK/WORKER (25). This is direct evidence that the reported 96.45% validation and
96.00% test sets are not independent. The v17 Citizen features have zero exact feature
duplicates across their 3,101 archives/splits.

The v16 pool nevertheless contains useful train-only signal: 18,236 deep-cleaned clips
match 90 of the current canonical names, including 15,653 local hash-named clips. These
are not automatically exact ASL-LEX matches, independent signers, or safe validation
examples. The earlier local audit already found 623 visually strong clips across 89
folders under a seven-per-class cap; 132/209 were conservative review subsets, not the
total usable ceiling.

Generalization evidence favors v17. The frozen v16 checkpoint scored 40.28% top-1 on
72 external Citizen clips from 29 participant IDs, including 29.2% on that audit's
Citizen-test subset. Of 21 exact-variant Citizen-test clips shared with the immutable
v17 test predictions, v16 got 7/21 (33.3%) and v17 got 19/21 (90.5%). Three other audit
clips were intentionally absent from v17 because they are different variants
(`W.H.A.T`, `HOSPITAL2`, and `DRINK2`). This is retrospective reporting from existing
frozen predictions, not a new test run or a tuning signal.

The honest v17 contract remains 1,475 train clips from 32 participants, 378 validation
clips from five disjoint participants, and 1,247 usable test clips from eleven disjoint
participants, with zero participant overlap. Per-class signer ranges are 11-16 train,
3-5 validation, and 10-11 test. Its 93.12% validation and one-time 87.57% test results
therefore measure a harder and more useful question than v16's 96% random-clip score.
The 5.55-point validation/test gap also shows v17 still has a real signer-generalization
problem; it must not be hidden by reverting to the v16 split.

### 2026-08-10 15:42 PST — Balanced Citizen/SemLex challenger wins validation gate

The one predeclared sampling ablation is implemented in
`active/v17/train_stage_1_v17.py`. `class_source_balanced_weights` uses iterative
proportional fitting over existing class/source cells; the resulting replacement
sampler has exactly 1% expected exposure for each of 100 classes and 50/50 expected
Citizen/SemLex source exposure. It creates no files and changes no augmentation,
architecture, optimizer, schedule, seed, train inputs, or validation inputs. Sampling
provenance and min/max per-sample weights are stored in the checkpoint/result. A unit
test verifies both expected margins exactly, and the real-manifest MPS optimizer /
checkpoint smoke passed.

The clean full run used the same 1,475 Citizen + 1,058 SemLex train clips, 378 Citizen
validation clips, seed 1701, d=256/depth=4 model, and patience 30 as the ordinary-
shuffle challenger. It early-stopped after 76 epochs. The best checkpoint is epoch 46
at `artifacts/models/stage1_v17_citizen_semlex_balanced/best_model.pth`: 93.92% top-1
(355/378), 100.00% top-5, and 93.46% macro F1. This is +0.79 top-1 / three clips over
the selected Citizen-only baseline and +1.06 / four clips over ordinary Citizen+
SemLex shuffle. Relative to the Citizen baseline it corrected 13 former errors and
regressed ten, so the net three-clip gain is not an identical-prediction artifact.
Citizen test, SemLex validation, and SemLex test were not accessed. The validation-only
report is `artifacts/reports/stage1_v17_citizen_semlex_balanced_validation/`.

The next agreed gate is now due: acquire SemLex validation for a secondary-domain
diagnostic of the frozen Citizen-only baseline, ordinary augmented checkpoint, and
balanced augmented checkpoint. Google Drive range metadata verifies that file ID
`1VvrbYgNZe_4fWS5ZdSsHyxOuWHmhisGq` is `val.tar.gz`, exactly 8,076,365,890 bytes.
Only exact matched variants should be selectively extracted; the full archive is not
training data at this stage. The SemLex test video archive is the separate 14.66 GB
file ID `1nVjvgJhjo3lILr5S23p_PsMR7yFdQTrS` and must remain untouched.

### 2026-08-10 15:51 PST — Balanced-model local fishing yields 315 priority review clips

The user's concern is confirmed: the current signer-disjoint results establish real
learning but not production readiness. The one-time Citizen test is 87.57%, only 100
isolated classes are covered, and independent portrait-iPhone behavior, OOV rejection,
ASL variant review, licensing, and device measurements remain unresolved. The SemLex
test archive is not additional development volume: its ten signers have zero overlap
with SemLex train/validation and it must remain sealed for one final independent
cross-dataset evaluation. SemLex validation may be used once as the planned secondary
diagnostic and only then folded into a later training version if its diagnostic role is
explicitly retired.

The 623-clip, quality/diversity-capped local pool was rescored with the new balanced
Citizen+SemLex checkpoint, which was never trained on local clips. Its model-assisted
label agreement is 62.12% top-1 / 83.15% top-5, up from the independent Citizen-only
checkpoint's 52.33% / 80.26%. `scripts/select_local_citizen100_consensus.py` now joins
both immutable prediction sets, enforces the existing 80% observed-hand and 50% face
coverage gates, requires exact equality with the pinned Citizen raw gloss, and hashes
all raw files for exact deduplication. All 623 files are hash-unique.

The conservative result is 238 Tier-A clips across 70 classes where both models put
the folder label at top-1, plus 77 Tier-B clips where both include the label in top-5
and one puts it at top-1. The 315-clip A+B priority review pool covers 76 classes with
1-7 clips/class (median four); 35 classes have at least five. Another 41 Tier-C clips
have dual top-5 support only. Quarantine contains 105 extraction-quality failures, 84
clips whose folder/canonical name is not the exact pinned raw-gloss text, and 78 model
disagreements. No file was deleted and no model prediction was written back as a label.

The immutable review manifest and per-clip/class ledgers are under
`artifacts/reports/local_citizen100_quality_audit/consensus/`. They explicitly remain
`training_eligible:false`, train-only after ASL-fluent exact-variant review, because
model agreement is correlated screening evidence rather than label proof and the local
corpus has no trustworthy signer IDs. The v16 model was not used for selection because
it trained on this local domain. Fifteen focused Stage-1/consensus tests pass, both
scripts compile, and `git diff --check` passes.

### 2026-08-10 16:15 PST — Exact-only local and uncapped-clean SemLex pools expanded

The earlier 623 local clips were a diversity cap, not the total mechanically valid
inventory. At the user's direction, local selection was expanded from seven to 14 per
class while making the lexical gate stricter. The selector now supports
`--exact-pinned-raw-only` and rejects canonical-folder/pinned-raw-gloss inequality
without normalization (for example, `DRINK` cannot stand in for `DRINK2`). It retained
1,021 visually strong, appearance-diverse clips across 77 exact-text classes from
14,883 inspected local-style candidates. The 2,749 files in non-exact pinned-raw
classes were quarantined before feature extraction. The selected count is below the
1,078 theoretical cap because 57 candidates were too appearance-similar to add.

The expanded local set reused 539 schema-validated archives from the cap-seven audit
and extracted 482 new videos. Extraction completed 482/482 with zero failures and zero
no-hand clips; the full 1,021-archive v17 audit passed with zero errors. All 1,021 raw
SHA-256 values are unique. Citizen-only model agreement is 56.32% top-1 / 84.13% top-5;
the balanced Citizen+SemLex model reaches 65.03% / 85.99%. Dual-model consensus plus
the 80% observed-hand and 50% face gates yields 434 Tier-A dual-top-1 clips and 154
Tier-B dual-top-5/one-top-1 clips: 588 priority clips across 76 classes, 1-14/class
with median eight. Another 85 dual-top-5-only clips remain Tier C; 203 extraction-
quality and 145 model-disagreement clips remain quarantined. Outputs are under
`artifacts/reports/local_citizen100_quality_audit/cap14_exact_consensus/`. Local A/B
clips remain train-only candidates rather than automatically approved labels because
folder equality plus correlated model agreement cannot prove the exact ASL variant or
signer identity; ASL-fluent review remains the final gate.

Exact raw-hash decontamination found zero overlap between all 1,021 expanded local
clips and the 3,102 official Citizen provenance rows, including zero Citizen validation
or test overlap. It also found zero overlap with all 1,499 retained SemLex-train raw
hashes. Thus the local shortlist contains no byte-identical copy of either primary
source; this does not convert unknown local signer identities into an evaluation set.

The SemLex cap of 12 signers/class was also removed without overwriting the immutable
1,058-clip manifest used by earlier checkpoints. The new selector accepts cap zero as
all distinct quality-passing signers, verifies every source is SemLex train with an
exact `asllex` entry/label identity, retains the existing `TAKE` mismatch quarantine,
and requires at least 70% observed-hand frames, 30% hand-node presence, and 50% face
presence. `full_clean_train_candidates.json` contains 1,388 unique-hash clips across
97 classes and all 32 SemLex train signers, with median 14 and range 2-25 clips/class.
Relative to cap-12, 37 weaker clips were dropped and 367 stronger signer/class clips
were added, a net +330. All 1,388 feature archives pass the v17 audit and the real
Stage-1 supplement loader returns exactly 1,388 while excluding `TAKE`.

The immediately safe next controlled run is 1,475 Citizen train plus 1,388 full-clean
SemLex train = 2,863 clips with online class/source-balanced sampling and the unchanged
378-clip Citizen validation. The 588 local A/B candidates would raise the pool to
3,451 only after exact-variant human approval and require a three-source sampler so the
unknown local sessions cannot dominate. Citizen test and SemLex test remain sealed.

### 2026-08-10 16:38 PST — Full-clean SemLex run reaches 95.77% Citizen validation

The controlled full-clean experiment used exactly 1,475 Citizen train plus 1,388
quality-gated exact-ASL-LEX SemLex train clips, the fixed 378-clip Citizen validation,
seed 1701, the same d=256/depth=4 architecture and augmentation, and the same online
class/source-balanced sampler as the cap-12 winner. No local candidate entered this
run. It completed 123 epochs and stopped after the declared 30 stale epochs; a late
series of genuine improvements moved the best checkpoint to epoch 93.

The retained checkpoint at
`artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth`
achieves 95.77% top-1 (362/378), 100.00% top-5, and 95.51% macro F1. This is +1.85
points / seven clips over the 93.92% cap-12 balanced model and +2.65 points / ten clips
over the 93.12% Citizen-only baseline. Relative to cap-12 it corrected eleven clips
and regressed four; relative to Citizen-only it corrected sixteen and regressed six,
so neither gain is an identical-prediction artifact. Sixteen Citizen-validation errors
remain, led only by `THANKYOU -> GOOD` twice; every other confusion occurs once.

The checkpoint and result provenance record 2,863 train clips, the exact full-clean
manifest SHA-256 `09f22aeff491ac16a498cbf5be02eb9867ddb0b16c139b1e302571d2bd51883a`,
50/50 expected Citizen/SemLex exposure, exactly 1% expected exposure per class, and
false Citizen-test/SemLex-test access. The immutable validation report is under
`artifacts/reports/stage1_v17_citizen_semlex_full_clean_balanced_validation/`. This
checkpoint is the new validation winner, but it must not be evaluated on the already
consumed Citizen test during development. The next independent model-quality gate is
SemLex validation after its manual archive download and selective exact-variant
extraction; SemLex test remains sealed. The checkpoint SHA-256 is
`23d5b7f1b343a6b5246e4afe03c0ab99067d3dac6e355024b5c53e5c31013f8e`.
Twenty-two focused local/SemLex/Stage-1 tests pass, the affected scripts compile, and
`git diff --check` passes.

### 2026-08-10 17:19 PST — SemLex validation and d=384 mobile ablation completed

The user's manual `/Users/frnzlo/Downloads/val.tar.gz` is the exact expected
8,076,365,890-byte SemLex validation video archive. Its SHA-256 is
`6eca70a5761f2bfeea5d4f58b1ed34431f2d1be39a20947f01a27e4a22516b90`; both gzip
identification and a complete `gzip -t` passed. The original archive was preserved.
SemLex planning/extraction now accepts an explicit split and locks validation/test as
`evaluation_only_never_training`; the Stage-1 train loader still rejects non-train
supplements.

The frozen exact-ASL-LEX validation plan selected one clip per official signer/class:
984 requested clips across 98 classes and 32 validation signers. `THEY` and `CHILD`
have no exact validation entry. Selective extraction retained 978 clips and quarantined
six VP9 files that failed complete decode validation. Apple Vision extracted 978/978
with zero failures/no-hand clips, and the full v17 schema audit passed. This diagnostic
is cross-domain but not signer-independent: 31/32 SemLex validation signer identities
also occur in SemLex train.

On the identical 978-clip diagnostic, Citizen-only d=256 scores 73.72% top-1 / 88.75%
top-5 / 70.56% present-class macro F1; cap-12 SemLex d=256 scores 82.41% / 95.81% /
79.61%; and full-clean SemLex d=256 scores 85.89% / 96.11% / 82.60%. This independently
supports the larger clean SemLex train pool. Reports are under
`artifacts/reports/semlex_citizen100_val_audit/`; SemLex test remains untouched.

The controlled d=384 challenger changed only model width from the d=256 winner. It
used the same 1,475 Citizen + 1,388 full-clean SemLex train clips, fixed Citizen
validation, class/source-balanced sampler, augmentation, seed, schedule, and patience.
It has 14,338,853 parameters versus 6,470,885 (2.22x), completed 81 epochs, and retained
epoch 51 at 95.24% Citizen-validation top-1 (360/378), 99.74% top-5, and 94.90% macro
F1. The d=256 winner remains better at 95.77% / 100% / 95.51%. Paired Citizen outcomes
are four d=384 corrections and six regressions (exact McNemar p=0.754). On SemLex
validation d=384 reaches 86.71% / 96.01% / 84.52%, only eight net clips above d=256;
it has 40 corrections and 32 regressions (p=0.410). Neither difference is statistically
persuasive, and the more trustworthy primary gate favors d=256. No test split was
accessed by either run.

Both checkpoints were successfully converted through the same fixed-shape iOS 15
FP16 ML Program path with trace/manual-attention parity and matching Core ML top-1.
The d=256 package is 12.58 MiB; d=384 is 27.59 MiB (2.19x). Warm current-Mac batch-one
latency ratios for d=384 versus d=256 are 1.79x CPU-only (1.007/0.564 ms), 1.56x
CPU+Neural Engine (0.770/0.495 ms), and 1.27x CPU+GPU (6.084/4.778 ms). These are
desktop proxy measurements, not low-end/medium iPhone latency, memory, or sustained
thermal evidence. Export outputs are under `artifacts/generated/coreml_v17_comparison/`.

Decision: retain full-clean SemLex d=256 as the Stage-1 validation/mobile winner.
d=384 is rejected because it is larger/slower without a reliable accuracy gain. The
588 local A/B clips were deliberately not used in this architecture ablation because
their exact ASL variant still lacks human confirmation. The next data action is
ASL-fluent review of those local candidates; the next deployment action is real-device
Core ML latency/memory/thermal testing on target iPhones.

### 2026-08-10 09:40 PST — Direct Citizen100 augmentation exhausted; broad pretraining opened

The Citizen acquisition, manifest, raw/extractor/quality reports, Stage 1 validation
report/history, cached official split metadata, local raw inventory, and current
Microsoft release page were re-audited before attempting an accuracy-data expansion.
ASL Citizen v1.0 remains the latest official release: 83,399 videos, 2,731 signs, and
52 signers with fixed signer-disjoint splits. The 100 pinned raw-gloss/ASL-LEX pairs
match exactly 3,102 official metadata rows; all 3,102 MP4s are already local and zero
selected exact-variant files are missing. The prior downloader already records
3,102/3,102 verified members with zero failures and zero remaining output bytes.
A fresh downloader dry-run against the official archive independently reported
1.5883 GiB existing output and exactly 0.0 GiB remaining.

Therefore there is no safe additional Citizen download for augmenting the existing
100 class labels. Fourteen remaining same-name raw-gloss/ASL-LEX pairs across 13
concept names are different lexical or numeric variants. They total 188 train, 49
validation, and 161 test clips and cannot be merged into the pinned classes; W.H.A.T is
fingerspelling and its 30 clips are already quarantined. The validation log has 26
errors on 378 clips, led by I -> WE and ANSWER -> GO twice each, but Citizen contains
no further pinned exact-variant examples for those classes. No video was downloaded:
doing so would either duplicate local data, leak the consumed official test signers,
or change class semantics. Evidence and the three safe expansion choices are recorded
in `artifacts/reports/CITIZEN100_V17_EXPANSION_AUDIT.md`.

The user correctly clarified that the 3,102 selected clips are not all videos recorded
by those official Citizen signers. All other signs can be used for broad representation
pretraining without pretending they are examples of the current 100 labels. The full
official train/validation pool is 40,154/10,304 videos covering 2,731 exact raw-gloss
plus ASL-LEX pairs from 35/6 signer-disjoint participants. Compressed transfer is
21.25/5.37 GiB and retained raw video would require 22.90/5.83 GiB, exceeding the
approximately 16 GiB free host space.

`scripts/extract_citizen2731_v17.py` temporarily provided a resume-safe,
storage-bounded route:
it accepts only train/validation, range-downloads one verified official ZIP member,
checks size and CRC, extracts Apple v17 landmarks from a temporary video, saves compact
features plus source provenance, and removes the temporary transfer copy. Four bounded
download workers overlap network fetches with the single Vision detector. The frozen
pretraining manifest is `active/v17/citizen2731_pretrain_manifest.json`, SHA-256
`7c827f8d71f4dca7266070e28f4b1ed74927ad3699ac7a4c30c103fe4ea5e203`; it has 2,731
classes and preserves exact pair identity, including separate RESEARCH1/RESEARCH2
classes despite their shared Citizen code `B_03_084`. Three focused tests cover exact
pair construction, the shared-code edge case, and fail-closed test rejection.

The first 15 streamed train clips extracted successfully at schema level with zero
failures and zero no-hand results. After enabling bounded prefetch, the 12 new clips in
the resume smoke completed at 1.68 clips/second; all three earlier smoke outputs were
recognized as existing. Raw video was not retained and Citizen test was not accessed.
Seventeen focused Citizen/Apple tests pass, and `audit_v17.py` passes all 15 smoke
archives. A full 50,458-clip train/validation stream was then started based on the
interpretation that the user's request for the signers' other videos authorized broad
2,731-class pretraining. The user questioned that expanded scope, so the process was
immediately interrupted at 299 processed items. It left 298 compact train landmark
archives, one provenance-recorded no-hand result, zero retained temporary/raw videos,
and no active extraction process under `data/local/citizen2731_v17/`. At the user's
explicit request, the entire 4.5 MiB directory (301 files including metadata/events)
was moved recoverably to
`/Users/frnzlo/.Trash/SLT_citizen2731_v17_20260810_0955`. The temporary extractor,
its focused test, and the 2,731-class manifest were removed from the worktree. The
project is again scoped strictly to the frozen 100 classes.

### 2026-08-10 01:32 PST — Kaggle T4 full run launched

All eight private multipart datasets
`francisbatiancela/slt-v17-movinet-trainval-part-00` through `part-07` now report
Kaggle status `ready`. The official uploader's retry progress was proven unreliable:
connection resets caused it to reread bytes without matching server acceptance. A
bounded GCS-resumable helper was added as
`active/v17/kaggle_resumable_upload_v17.py`. It reused the official CLI's saved upload
sessions, queried authoritative server offsets, transferred 256 KiB chunks, recovered
resets without replaying the whole remainder, and never printed signed URLs. Every
90 MiB part reached the exact byte total `94,371,840`; part 07 reached its exact smaller
total. The official Kaggle CLI then finalized the dataset records, and all eight were
independently queried as `ready`.

Private kernel `francisbatiancela/slt-v17-movinet-end-to-end` version 1 was pushed with
explicit accelerator `NvidiaTeslaT4` and a 43,200-second limit. At launch, Kaggle
reported `KernelWorkerStatus.RUNNING`; the later 09:25 CLI audit above supersedes that
transient status and proves the run failed before training. The attached inputs remain
train/validation-only; the runner verifies every part SHA-256, reconstructs and verifies
the original archive SHA-256, requires CUDA, runs the fixed 5-epoch warmup plus 35-epoch
joint fine-tune, and rejects any result that claims test evaluation.

### 2026-08-10 00:39 PST — Kaggle multipart upload resumed

The original single-stream Kaggle upload was stopped after sustained throttling. The
exact 696 MiB archive was split byte-for-byte into eight private 90 MiB-or-smaller
parts, each with its own recorded SHA-256. Concatenating the local parts reproduces the
original archive SHA-256 exactly. The Kaggle runner and kernel metadata now attach all
eight private datasets, verify every part, reconstruct the archive, and verify the
whole-archive checksum before extraction. This is a transport change only; crops were
not resized, recompressed, or otherwise changed.

Eight parallel uploads reached substantial progress, but the local upload processes
were closed when the active tool call was interrupted by a user status message before
Kaggle created the dataset records. Kaggle showed no completed datasets at that point.
All eight official resumable uploads were immediately restarted. Kaggle confirmed
stored offsets for the restarted streams (for example, part 01 resumed after 27,787,263
bytes and part 07 had approximately 31.4 MiB remaining). Uploads are active again. The
CUDA kernel has **not** launched yet and no training result exists; launch remains gated
on all eight private dataset records becoming queryable.

### 2026-08-10 00:04 PST — Kaggle CUDA runner configured; upload in progress

The official Kaggle CLI 2.2.4 is installed in the isolated Python 3.11 environment
`artifacts/generated/kaggle_cli_env`, and OAuth authentication completed as Kaggle user
`francisbatiancela`. No API token was copied into the repository. A private Kaggle
dataset is being created as `francisbatiancela/slt-v17-movinet-trainval`, and the
private GPU script is configured as
`francisbatiancela/slt-v17-movinet-end-to-end` on an NVIDIA T4.

The upload bundle is
`artifacts/generated/kaggle_movinet_v17/dataset/movinet_v17_trainval.tar`, size 696 MiB,
SHA-256 `699834265f70ae6226b4692a0058b7c1ef2ea325d935941bbdf608af2b9c8bab`.
It contains only the train/validation RGB crops, train/validation Apple feature caches,
official MoViNet-A0 checkpoint, and the exact runner dependencies/code. An archive
listing audit found no `test/` directory or `landmark_test` cache. The private dataset
metadata uses license `other` and explicitly preserves the original ASL Citizen
research/noncommercial terms; the bundle is not published or relicensed.

The fail-closed cloud entrypoint is `active/v17/kaggle_movinet_runner_v17.py`. It verifies
the bundle checksum, rejects unsafe tar paths, installs the pinned CUDA environment,
requires a real CUDA TensorFlow device through the trainer, runs the fixed 5-epoch
head warmup plus 35-epoch joint fine-tune with batch size 4/patience 8, and refuses a
result whose manifest says the test split was evaluated. The Kaggle upload is currently
in progress; this section does not yet establish that the GPU kernel launched or that
full training completed.

### 2026-08-09 23:57 PST — Unfrozen MoViNet benchmark and execution boundary

The MoViNet implementation is capable of genuine end-to-end optimization of the RGB
branch: in the `joint_finetune` phase, gradients run from the fused and visual losses
through all 911,583 MoViNet-A0 backbone parameters to the three 16-frame pixel streams.
The Apple landmark encoder remains deliberately frozen, so the precise description is
"end-to-end MoViNet visual fine-tuning with jointly trained Apple/RGB fusion," not
end-to-end training of both encoders.

An unfrozen-backbone measurement was completed at batch size 4 using one real training
batch and one real validation batch. It passed the exact-Apple initialization check,
performed a joint update, evaluated, saved, reloaded, and completed in 62.8 seconds.
The result is stored under
`artifacts/generated/stage1_movinet_v17_joint_benchmark/`; its 50% metrics cover only
four validation clips and are **not accuracy evidence**. The actual splits contain
1,475 train clips and 378 validation clips, or 369 train plus 95 validation batches per
epoch at batch size 4. Even using the optimistic post-compilation throughput from the
earlier frozen smoke, the current official three-view CPU graph implies hours per epoch
and multiple days for the declared 5-warmup/35-joint schedule.

The full end-to-end run was therefore not completed locally. This was a compute-route
decision, not a model or gradient-path failure. The official Model Garden graph cannot
use this Mac's TensorFlow Metal device because its grouped/depthwise Conv3D path requires
an XLA platform Metal does not provide; no CUDA/Kaggle runner or credentials are
configured in the workspace. Do not describe the smoke or the one-update benchmark as
a completed MoViNet experiment. The proper next execution is the same fixed protocol
on Linux/CUDA, with train/validation only and the consumed Citizen test remaining
sealed. Launching a multi-day CPU job that monopolizes the personal Mac is not an
equivalent fast experiment and requires an explicit decision if no GPU becomes
available.

The trainer now accepts `--device cuda` and fails immediately unless TensorFlow is a
CUDA build with a visible GPU. Linux dependencies are pinned separately in
`active/v17/movinet_requirements_cuda.txt`; the macOS environment keeps its Metal
plugin only for reproducing the documented failure/CPU path. As of this audit,
TensorFlow's official install guide still says there is no official macOS GPU support,
and PyPI lists `tensorflow-metal==1.2.0` as the newest plugin release. Sources:
`https://www.tensorflow.org/install/pip` and
`https://pypi.org/project/tensorflow-metal/`.

### 2026-08-09 23:48 PST — Joint-training correction and sign-specialized MoViNet started

The earlier term “five-seed ensemble” described only a robustness diagnostic. The
canonical MobileCLIP fusion did train one residual head on aligned Apple and RGB
features together, but both base encoders were frozen/cached; it was **not** end-to-end
co-adaptation. MobileCLIP is optimized for image-text semantic alignment, so being a
strong image encoder does not guarantee preservation of subtle finger configuration.
The hand-crop recovery from 39.68% to 70.37% proves that crop scale mattered, while the
remaining gap shows that frozen features still lost sign-specific detail. A true joint
pixel/landmark fine-tune remains a distinct experiment and must not be conflated with
the completed feature-residual result.

A separate MoViNet-A0 experiment is now implemented in
`train_stage_1_movinet_v17.py`; it is explicitly sign-specialized rather than a generic
full-frame Kinetics head. One shared pretrained video backbone processes anatomical
left-hand, right-hand, and union/context sequences from the real Apple-selected crop
corpus. It consumes explicit missing-view masks and all 16 normalized box trajectories,
uses view identity/attention, applies sign-safe mirror/temporal/photometric/view-drop
augmentation, and jointly trains a visual-only auxiliary classifier plus cross-modal
Apple/RGB fusion. The fusion residual is zero-initialized and was verified bit-exactly
equal to the frozen Apple logits before training. The official Citizen test is rejected
by the loader and was not accessed.

The official TensorFlow 2.16.1 / Model Garden 2.16.0 implementation and Kinetics-600
MoViNet-A0 checkpoint were installed in isolated Python 3.10 environment
`artifacts/generated/movinet_env`; the checkpoint archive SHA-256 is
`7bae6c7ef74e2ff4115ad51f1fdad8718247375b420da2e35e8ee7771ac35758`. The environment
is pinned by `movinet_requirements.txt`. TensorFlow Metal 1.2 cannot execute the
official XLA-compiled grouped/depthwise Conv3D graph (`registered platform` failure),
and removing the explicit wrapper still fails in Metal's grouped Conv3D operator.
CPU execution is therefore the reproducible local training path; this host limitation
does not imply a TFLite runtime failure.

Two pure data-contract tests pass: item-ID alignment/test rejection and exact mirrored
left/right involution with missing views remaining zero. The CPU optimizer/checkpoint
smoke then passed official-weight restoration, three-stream forward, two updates,
partial validation, save/reload, and test isolation. It built a 1,867,433-parameter
joint model with a 911,583-parameter MoViNet backbone. Its 87.5% fused result is only
7/8 smoke clips and the 0% visual result is after two updates; neither is accuracy
evidence. The full train/378-clip validation run has not completed. Local CPU throughput
must be benchmarked before committing to a many-hour run or moving it to CUDA/Linux.

### 2026-08-09 — v17 smoke result

The HELLO audit clip produced finite features, 40 observed hand frames, 0.8889 detected
hand-frame fraction after trimming, 0.4346 hand-node presence, 0.875 face presence, and
0.4219 body presence. Apple Vision reported 40 right-hand and 0 left-hand observations;
this specifically verifies the old v16 chirality reversal is corrected.

## Data and storage state

- Put raw/local datasets only under `data/local/`.
- Put generated reports under `artifacts/reports/` and disposable generated outputs
  under `artifacts/generated/`.
- Current free space measured at approximately 13 GiB on 2026-08-09 23:39 PST after
  retaining the 3.4 GiB temporary spatial-feature cache and isolated MoViNet environment.
- PopSign is approximately 1.1 TB in full and must not be downloaded wholesale.
- Download one sign/split archive at a time, check available disk before transfer and
  extraction, preserve source/license provenance, and validate portrait extraction
  before expanding.
- The PopSign `thankyou` test audit download is paused with recoverable partial files;
  it must not resume unless portrait one-handed auditing is explicitly useful.
- ASL Citizen is research/noncommercial-use data. It is suitable for this research
  baseline but cannot be assumed to license a commercial shipping model.

## Environment

- Host: macOS on Apple Silicon.
- Project Python: `venv/bin/python` (Python 3.9 with system site packages).
- Apple bindings: PyObjC Vision and Quartz 11.1 in `venv/`.
- System Python does not have the required Vision bridge; run real extraction/tests
  through `venv/bin/python`.
- The completed MobileCLIP2 challenger uses its isolated Python 3.10 environment at
  `artifacts/generated/mobileclip2_env`; it is not part of the selected runtime.
- The active MoViNet research branch uses isolated Python 3.10 environment
  `artifacts/generated/movinet_env`; official TensorFlow Metal training is unsupported
  for its grouped Conv3D graph, so local training is CPU-only.

## Immediate next actions

1. Keep Apple Vision plus the v17 Squeezeformer frozen as the selected Stage-1 path.
   Do not combine MediaPipe/RGB archives with Apple training data, select another
   checkpoint from the official test errors, or rerun the official test while tuning.
2. Collect a new portrait-iPhone, signer-disjoint evaluation set. This is now the only
   valid dataset for measuring portrait capture and future model changes without
   contaminating the official Citizen test result.
3. Export the frozen Apple model and preprocessing to Core ML. Measure package size,
   memory, cold start, sustained latency, and thermals on real low-/medium-spec iPhones;
   desktop MPS timing is not a substitute.
4. Design UNKNOWN/out-of-vocabulary rejection and evaluate it on independently held-out
   nonsign/background clips before presenting the classifier as an app feature.
5. Obtain ASL-fluent review of the exact raw-gloss/ASL-LEX mappings and the most frequent
   confusions when practical. Add new training data only under a new experiment/version;
   never tune the frozen v17 selection against Citizen test predictions.
6. Keep the new MoViNet and any true end-to-end MobileCLIP/Apple co-training exploratory:
   benchmark training throughput, use only train/validation, and never reopen the
   consumed official Citizen test. A new portrait-iPhone set is required before either
   can displace the frozen v17 selection.

## Known boundary

A v17 classifier is trained and has a one-time official signer-disjoint test result of
87.57% top-1, but independent portrait-iPhone accuracy, UNKNOWN rejection behavior,
ASL variant review, Core ML export, and real-device performance are not yet established.
Do not tune on the official test, call 93.12% test accuracy, or claim production mobile
readiness.
