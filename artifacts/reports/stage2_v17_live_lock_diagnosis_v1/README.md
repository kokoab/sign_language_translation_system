# Continuous-signing lock diagnosis

**The locking difficulty is real and has three separable causes: an isolated-sign
locking workflow applied to continuous motion, temporal decoding sensitivity, and
measurable differences between training and live inputs.** The evidence does not
support “the model learns nothing,” variant mismatch as the sole cause, or a simple
lower-confidence-threshold fix. No runtime, checkpoint or training data was changed.

## Your latest webcam session

Inspected `live_reel_continuous_v17/20260910_071844_645159/history.json`, its saved
video, and the active locking/Finish code. This was the newest webcam history at
completion. Intended transcripts are not recorded, so no webcam accuracy/WER is
invented from model outputs. Visual review supports framing inspection, not expert
lexical certification of everything signed.

| Stage in the actual locking process | Count | Percentage of 66 probes |
| --- | ---: | ---: |
| Initial classifier accepted | 40 | 60.61% |
| Rejected | 26 | 39.39% |
| Reached stable-label proposal | 14 | 21.21% |
| Appended a committed sign | 10 | 15.15% |

These percentages are gate throughput, not sign accuracy or recall. There were 22
provisional-label events,3 candidate timeouts,4 keyboard Finishes and no explicit
Reset events. Finish itself clears utterance state. No results were ignored after
reset. Rejections include 25 low-score and20 low-margin flags; these overlap.

The live lock uses a growing activity-anchored isolated-sign crop: minimum 0.5s,
maximum 2.5s, reprobe every 0.12s, two consecutive accepted same-label results.
A single rejected result resets the candidate because release_hits=1. Once a crop
fails to lock, it keeps growing across subsequent movement; competing labels
change and the two-hit condition restarts. Example: the crop beginning7.560s
changed TRY → TIME → UNKNOWN/GO → HE → THEY → HE, then timed out around 10.076s.
This is evidence of unstable recognition/segmentation before final commitment.

A full multimodal verifier agreed with 10 of 14 proposals (71.43%); all10 committed.
The other four were SIGN→GO, SORRY→I, SORRY→TOMORROW and COME→DAY. GO was suppressed
as an adjacent duplicate; the others failed verification/confidence. The proposed
labels themselves are not confirmed ground truth.

Fixed-recorded-proposal counterfactuals show why lowering thresholds is insufficient:
- Changing only the commit threshold from 0.45 to 0.25 would admit DAY for the COME
  proposal; it would not recover either SORRY proposal because verifier acceptance
  failed upstream.
- Bypassing the verifier at the existing0.45 threshold would allow the recorded
  SIGN proposal. The two SORRY scores0.265/0.412 and COME0.287 would still fail.
These are conditional gate calculations, not full changed-policy simulations:
changing commits changes subsequent crop boundaries and verifier invocations.

Detected-hand coverage averaged 99.54% across classifier crops, and99.91% among
rejected crops. Thus missing all hand detections is not the main observed failure.
This does not prove the coordinates, handedness, occlusions or fine handshape are
correct. The session recorded 1120 landmark observations and207 stale-camera skips;
skips are intentional under latest-frame capture and are not207 lost signs.

## Stage2 did not drive the live locks

The continuous Reel default is `stage2_at_finish=true` and `stage2_review_only=true`.
While you are signing, Stage1 controls locking. Stage2 is processed at Finish and
cannot replace the committed transcript in this mode—even when marked stable.

| Finish | Stage1/committed output | Final Stage2 suggestion |
| --- | --- | --- |
| 1 | HELLO LESS HOW WHO | HELLO HOW YOU YOU |
| 2 | I DIFFERENT | TIME DIFFERENT |
| 3 | I DIFFERENT | HOSPITAL DIFFERENT |
| 4 | I GO | THINK READ TOMORROW TOMORROW |

Finish1 produced HELLO HOW YOU at windows3–5, then added another YOU at 6–7.
Finish4 produced I READ TOMORROW at window4, repeated TOMORROW at 5 and changed the
prefix I→THINK at 6. “Stable” merely means the last two accepted outputs agree; it
cannot distinguish a stable error. Without verified intended transcripts and
window-level original observations, these transitions must not be called exact
recognition followed by failure or attributed solely to held posture.

Measured Stage2 Finish processing took 0.684–1.923seconds in this session. The first
and fourth outputs illustrate length/context sensitivity independent of live-lock
gating. Automatically trusting the entire final CTC sequence would introduce a
different error path; the evidence does not justify that as a complete solution.

## Paired experiment: all12 documented matching-variant development phrases

Used the entire existing ASLLRP exact validation subset:12 clips,24 reference signs,
one signer JONATHAN. No clip was selected based on success. Mapped each of 24 sign
cores to the same parent occurrence and manual frame interval, using the existing
ASL-LEX/Sign Bank variant mapping. These are development diagnostics, not a new
independent test or broad signer-generalization measurement.

| Path on the same12 phrases | WER, lower is better | Whole-phrase exact accuracy |
| --- | ---: | ---: |
| Cached repaired Stage2 | 37.50% (9/24 edits) | 33.33% (4/12) |
| Raw video through current continuous Reel Stage2 | 45.83% (11/24) | 25.00% (3/12) |
| Raw video committed transcript | 79.17% (19/24) | 0.00% (0/12) |

Raw and cached Stage2 outputs agree on 7/12 clips (58.33%); five differ. All twelve
raw processes completed successfully with zero stale-frame drops. Six produced an
empty committed transcript. This demonstrates a gap between sign recognition and
what the current lock/commit flow presents as a finished phrase.

| Reference | Source clip | Cached Stage2 | Raw Stage2 | Committed |
| --- | --- | --- | --- | --- |
| NIGHT TIME | 15718738.mp4 | SCHOOL FRIEND | SCHOOL | ∅ |
| WATER COLD | 30336.mp4 | WATER COLD | WATER COLD | ∅ |
| LIKE READ | 31657946.mp4 | LIKE | LIKE | ∅ |
| FRIEND NOW | 31659623.mp4 | FRIEND | FRIEND | ∅ |
| NOW BAD | 31660216.mp4 | NOW LEARN | NOW | NOW |
| TIME FRIEND | 4236779.mp4 | FRIEND | FRIEND | FRIEND |
| FRIEND MAYBE | 7345233.mp4 | FRIEND STOP | STOP | WRITE |
| FRIEND NOW | 841111.mp4 | FRIEND NOW | FRIEND DIFFERENT | ∅ |
| FRIEND NOW | 841314.mp4 | FRIEND NOW | FRIEND NOW | ∅ |
| FRIEND NOW | 841415.mp4 | FRIEND | FRIEND | FRIEND |
| FRIEND NOW | 841516.mp4 | FRIEND DIFFERENT | FRIEND | FRIEND |
| WORK WHERE | 842935.mp4 | WORK WHERE | WORK WHERE | WORK |

## Sign-core experiment: recognition exists, boundaries/context matter

Four of 24 exact manual cores contain only 2–3 annotated source frames (roughly
67–100ms at 30fps). The existing extractor rejects fewer than 4 frames; these four
were explicitly excluded, not padded or assigned fabricated predictions. They are
READ and three NOW occurrences. This is an actual coverage limitation of isolated
core supervision; it does not prove every connected deletion comes from it.

On the20 evaluable individual cores:
- Frozen Stage1 gets16/20 correct: **80.00%**.
- Repaired Stage2, decoding each core separately, gets18/20: **90.00%**.

For the8 complete two-sign pairs only (16 tokens), apples-to-apples comparison:

| Input/decoding arrangement | WER | Exact phrases |
| --- | ---: | ---: |
| Original full cached phrases | 31.25% (5/16) | 50.00% (4/8) |
| Stage1 applied separately to each manual core, concatenate labels | 25.00% (4/16) | 50.00% (4/8) |
| Stage2 applied separately to each manual core, concatenate outputs | 12.50% (2/16) | 75.00% (6/8) |
| Manual cores as separate windows, one joint Stage2 decode | 12.50% (2/16) | 75.00% (6/8) |

The two75% conditions fail on different examples. NIGHT and TIME are individually
correct, but joint core windows produce LEARN TIME. WORK and WHERE are individually
correct, but joint core windows produce WORK WORK, despite the original full phrase
being correct. Context can help or harm; manual cropping is not a deployable fix.
These interventions change duration resampling and extraction context as well as
boundaries. They demonstrate sensitivity, not a pure proof of one CTC loss defect.
The8-phrase subset is easier than the full12; do not compare its75% directly to 25%
as if the denominator and inputs were unchanged.

## Controlled live-input diagnosis

The saved hand embeddings/boxes and fixed source-window edges were held constant.
Only fresh landmark extraction resolution and sample density were varied. Cached
inputs through the actual CoreML frozen encoder reproduce the cached predictions
on 12/12; maximum feature difference is 0.001947. Fresh1280px extraction at source
rate also reproduces all12 predictions. This points away from model conversion as
the cause of these five raw/cached disagreements.

At source rate, changing only detection resolution1280→640 raises total edits9→11
on 24 reference tokens (WER37.50%→45.83%). The affected outputs are not identical
to all raw-runtime differences: resolution is a contributing factor, not a full
explanation of the live pipeline.

One short tail becomes fewer than 4 sampled frames in the20Hz intervention, so those
lanes cannot produce that complete phrase under the unchanged extractor contract.
For the11 clips evaluable in every lane (22 tokens):

| Controlled lane, fixed cached hand evidence | WER | Exact phrases |
| --- | ---: | ---: |
| Cached / fresh1280px at source rate | 36.36% (8/22) | 36.36% (4/11) |
| Fresh640px at source rate | 45.45% (10/22) | 36.36% (4/11) |
| Fresh1280px, reduced to 20Hz | 40.91% (9/22) | 27.27% (3/11) |
| Fresh640px, reduced to 20Hz | 45.45% (10/22) | 27.27% (3/11) |

This hybrid intervention deliberately does not change hand crops or the source
window edges. Actual runtime also differs in hand-crop extraction, detector state,
sparse body/face phase, elapsed-time windows and scheduling. Their individual
contributions remain unisolated. The complete raw replays measure their combined
behavior. A first controlled probe stopped on the short tail; the resumed analysis
records it as unavailable and uses a shared cohort rather than silently filling it.

## What to do next, in order

1. **Make live recognition and commitment a continuous-sequence design.** The current
   command still locks isolated-sign guesses and only reviews CTC at Finish. Build
   and evaluate a provisional streaming Stage2 path with explicit prefix/endpoint
   stability. Do not equate repeated hypotheses with correctness or automatically
   replace every transcript with the last CTC guess. Retain an explicit review path.
2. **Make the training input contract match the measured live input contract.** Reuse
   these same development clips to measure hand/landmark extraction, window edges
   and timestamp sampling; train/re-cache with the chosen sustainable contract.
   Simply forcing1280px is not yet a live fix: latency has not been measured for it.
3. **Adapt temporal behavior using existing connected recordings.** Include native
   rate short signs, transitions, repeated signs and endings/holds with reliable
   ordered labels. Preserve true repetitions and do not blindly trim final windows.
   Track substitutions/deletions/insertions and phrase retention separately.
   The2–3-frame cores should stay in full sequences; do not discard their labels or
   stretch every tiny core into a standalone sign as a supposed general solution.
4. **Collect targeted data only when an identified gap remains.** First use current
   data. Later, a small transcript-confirmed, timestamp-preserved webcam set can
   distinguish user/domain gaps from pipeline errors. No new bulk download or
   additional ensemble is justified by this diagnosis.

A clean small-train-set fit test could test optimization capacity in a later
training experiment, but this report has not run one and does not claim the
training objective/optimizer is the single root cause. No further model was trained.

## Evidence, limitations and reproduction

- [Session counts, gate counterfactuals and complete Finish trajectories](session_analysis.json)
- [All12 raw replay commands and histories](raw_replays.json)
- [Manual-core mapping](matched_manifest.json), [core outputs](manual_core_probe.json)
- [Controlled frontend outputs](frontend_probe.json), [log including failed probe](frontend_probe.log)
- [Reconciled metrics](summary.json), [verification](verification.json), [focused tests](tests.log)
- Scripts: summarize_session.py, replay_matched.py, probe_manual_cores.py,
  probe_frontend.py and build_report.py in this directory. Replay script launches
  new diagnostic sessions; it does not modify runtime defaults. Frontend script
  resumes completed rows in its existing report. Use a separate output copy for
  a distinct experiment; preserve the original artifacts.

The saved webcam video is 910 frames at 15fps (60.67seconds). SessionRecorder writes
available frames without storing original timestamps or filling wall-clock gaps.
Therefore replaying that MP4 cannot reproduce original capture/locking timing
exactly. No webcam reference transcript was invented; no user-session WER reported.
42 focused lock, continuous Reel, live CTC and preservation tests passed.
Protected Citizen, SemLex, RIT and local tests were not evaluated. The report is a
completed bounded diagnosis, not a claim to have fixed live recognition.

Relevant code: [StableGlossLock](../../../scripts/live_reel_stage1_v17.py#L295),
[commit verifier](../../../scripts/live_reel_stage1_v17.py#L991),
[Finish selection](../../../scripts/live_reel_stage1_v17.py#L1159),
[live extraction](../../../scripts/live_stage2_ctc_v17.py#L390),
[offline extraction](../../../scripts/extract_stage2_multimodal_v17.py#L159),
[video recorder](../../../scripts/live_isolated_v17.py#L1157).
