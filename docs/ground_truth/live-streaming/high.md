# live-streaming — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

10 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-14 — O5S5 WER improvement does not pass the continuation gate

The bounded O5S5 augmentation reduced connected WER168.31→123.59%, but increased
deletions11→42 and failed familiar/isolated retention. All12checkpoints fail measured
gates; LG positive-window accuracy remains weak. Do not run the conditional confirmation
seed or export this candidate on WER alone. Runtime latency was not certified.
Report: `artifacts/reports/o5s5_augmented_v17_20260914/README.md`.

## 2026-09-12 — Stage-1 window seed 17111 rejected; no promotion or confirmation

The complete bounded 12-epoch experiment is recorded in
artifacts/reports/stage1_window_v17/README.md and selection.json. No checkpoint passes
all gates: epoch 1 fails connected/familiar/transition gates; later epochs fail
isolated retention. Best connected WER 146.13% still raises deletions from 11 to 33.
No second seed or Core ML export is justified by this run. Preserve default pipelines
and accepted artifacts. The new backend remains explicitly experimental.

One continuous validation signer, no annotated identical repetitions/long holds, and
no independent phone/background set prevent broader reliability claims. Fifteen raw
replays match cached transcripts; first-correct timing covers only 15 signs with 53.3%
misses, not the full frozen pool. Tested recent-tail replacement is not evidence of
useful spontaneous correction. HUNGRY is diagnostic-only; suppression changed none
of its 83 compared CTC outputs. Future selection must bind runtime latency evidence
to both checkpoint and frozen annotation identities. Do not restart speculative
training from these errors without a new explicit scope.

## 2026-09-10 13:19 PHT — approved live-path experiment completed; no promotion

Completed the authorized online CTC preview, shared live/training extraction, and bounded temporal adaptation experiment. Report: artifacts/reports/stage2_v17_live_matched_v1/README.md; machine evidence: verification.json, training_summary.json, prefix_baseline.json, prefix_adapted.json, online_summary.json and adapted_online_summary.json. Default runtime/checkpoints unchanged; --sequence-preview and --stage2-live-checkpoint are explicit experimental flags.

Training seed17101 completed12 epochs in998s after matched extraction. No epoch passed both original retention and matched gates. Aggregate matched-development selection chose epoch8, SHA 362eca062454e22a24f110a5d41411b3a7668bbbed9e2d895719d2c54534a397. Matched connected errors478/284 (168.31% WER) ->226/284 (79.58%,52.72% fewer); local19/259 (7.34%) ->11/259 (4.25%); exact11/24 (45.83%) ->12/24 (50%). Original retention: connected470->223, local6->10, exact9->12, contextual43->49, Citizen331/378->330/378, STEM16/21 unchanged. Current-baseline connected gate423 is stronger than prior v2 threshold542; neither fixes phrase failures. Initialization deliberately replaces nondifferentiable run veto with strongly suppressed differentiable OTHER, so epoch0 has603 original target errors; improvement is reported against repaired470, not weaker initialization. One seed only, no endpoint/hold augmentation; Stage1 frozen.

All12 adapted raw paced exact-variant replays succeeded and match cached final hypotheses12/12. Final12/24 edits (50% WER),2/12 exact; confirmed22/24 (91.67% WER),0/12 exact. Accepted updates before Finish only1/12 (baseline2/12), affected by short clips/window+compute latency. All334 adapted prefix evaluations: local final86/97 exact (88.66%) but confirmed29/97 exact (29.90%),134/259 errors (51.74% WER),5/97 wrong committed reference prefixes. OTHER spans54/225 wrong committed prefixes,26/225 conflicts. Stable agreement cannot guarantee accurate irreversible output. No protected test accessed; no ensemble added.

56 focused tests passed, including context rollover and stale/reset behavior; best/last artifacts reload with finite parameters; git diff --check passed. First trainer invocation failed before training on missing-domain evaluator assumptions; corrected matched evaluator and completed retry. Shared source fingerprints and checkpoint provenance captured in verification.json. Unrelated pre-existing changes preserved.

Next safe action is the user-requested discussion of faster Stage1 Reel locks versus continuing temporal sequence work. Evidence supports useful visual learning with unresolved boundary/commitment/latency problems, not 'nothing learned' or a need for bulk isolated data. If further temporal work is chosen, use verified transcripts and boundary/hold/repetition annotations on existing failed recordings; evaluate correct-lock coverage and delay after sign completion. Do not promote this experimental checkpoint or describe confirmation as recognition safety.

## 2026-09-10 19:01 PHT — authorized revisable transcription implementation and experiment

User authorized completing revisable sequential transcription, transition-aware temporal training and visual Finish re-evaluation, with a final report. Existing repaired baseline and v1 live-matched experiment are controls. Runtime worker owns opt-in --revisable-transcript, provisional full-hypothesis updates and visual feature replay at Finish; no premature irreversible prefix. Data worker derives conservative gap supervision from original ASLLRP annotations, excludes OTHER signs from blank supervision, preserves train/development roles. Main adds auxiliary gap blank loss and eligible observed-prefix CTC to existing live adaptation trainer; fixed Stage1 architecture and input contract. Same seed17101,12epochs,2048samples/epoch, original replay loss and optimizer as v1 for matched comparison; .25 gap and .25 eligible-prefix loss. No bulk acquisition, protected test use, baseline overwrite, or automatic promotion. Evaluate final WER/insertions/deletions, annotation-gap false emissions, partial revisions, first-output delay and final visual replay. Training may fail promotion; report evidence without calling stability or grammar a recognition guarantee.

## 2026-09-07 17:52 PHT — continuous Reel implementation authorized and first regression checks

User approved the proposed experiment and annotation audit, asking for completion
without further design questions. Compact execution plan (writing-plans workflow,
kept in this canonical handoff): reproduce buffer loss with a deterministic delayed
verifier; add a separate continuous Reel command with preserved pending frames and
prominent tentative glosses; defer exact Stage-2 observations to Finish and expose its
output only as a review candidate; run focused tests, familiar-video comparisons and
visual HUD inspection; audit public phrase annotations without downloading videos.
Work stays in the current uncommitted workspace to retain its active models/code.

Baseline 26 Reel/CTC tests passed. New deterministic real-loop tests failed before
implementation: next candidate started at verifier completion (0.95s), losing pending
frames; CTC ran before Finish; no disk queue or prominent tentative HUD existed.
Implemented `scripts/live_reel_continuous_v17.py` plus opt-in hooks in
`scripts/live_reel_stage1_v17.py` and `scripts/reel_hud_v17.py`. Original command
defaults remain unchanged. Finish windows use a private temporary-file FIFO with
exact observation serialization; no external pickle input is accepted. Only the new
command retains frames after the consumed clip, shows amber question-mark glosses,
defers CTC until Finish/Stage-1 completion, and prevents sequence suggestions from
replacing or speaking over Stage-1 output. Speech is utterance-only in this experiment.
All 31 focused tests now pass, including frame retention, delayed CTC, review-only
selection, FIFO reset/cleanup, preview-before-commit, and quit without Finish.
Real Core ML smoke/replay and visual checks are still pending; no accuracy/speed gain
claimed. Independent bounded annotation audit owns only
`artifacts/reports/continuous_reel_v17_web_audit_v1/`; protected splits stay untouched.

## 2026-09-07 17:40 PHT — Finish accepted; preserve responsive Reel feedback

User accepts Finish once per utterance, values Reel's immediate gloss feedback, and
reports delayed locks/repeated signing as its main problem. Discussion continues;
no hybrid runtime or model change has been approved. Proposed UX separates tentative
live glosses from confirmed sequence output, while retaining continuous observations.

Targeted Reel trace: defaults require a 0.50s candidate, probes spaced at least 0.12s,
two accepted proposal hits, then full verification (commit-hits default is already
one). Proposal and verification share the same outstanding future, so new proposals
wait during verification although observations continue accumulating. After emission,
the default zero transition-overlap branch calls clear_candidate() on the entire
current observation buffer, including observations newer than the classified clip.
This is a plausible mechanism for losing the beginning of the following sign, not a
confirmed explanation of the user's session without a matched replay. Threshold
lowering alone would not resolve this buffering behavior or recognition errors.

Existing cached-Reel report records median full verification 359.00 -> 304.15ms over
14 calls per lane, identical final sequences and only 2/5 exact familiar replays;
these are past offline timings, not a new live performance result. Direct Stage 2's
1.067s window cadence should not gate the fast display. Proposed scheduling must
measure combined resource contention and queue lag; background execution alone does
not establish sufficient throughput. Only handoff updated; no tests/model evaluation,
runtime edits, training, or acquisition performed in this turn.

## 2026-09-06 16:27 PST — causal recognizer selected53% local exact; YOU lexicon constraint visually inspected

Causal continuousv2 selectedepoch14: local106/200exact(53%),21.85%WER; ASLLRP0/12,
58.33%knownWER; NCSLGR2/37,96%knownWER; rollingCitizen340/378(89.95%), SemLex729/978
(74.54%). This improves local recognition but still fails open-corpus generality and
is not promoted. Matching Stage3 is live. Original beam-only context misses the true
sequence entirely on three of six video-smoke failures, so candidate-only reranking
cannot fulfill the user's fallback goal. Proposal decoding was added/tested and must
be evaluated jointly with exact CTC scores, including harmful/blank/OTHER cases.

V8 avatar comparison completed with lexical FullyOpen constraints. YOU index skeleton
now has aligned three segment directions (approximately[.475,-.207,.856]); closeup was
inspected, showing a foreshortened pointing finger rather than the former loop. This
is not user acceptance: camera-facing projection and hand mesh still require checking.
Lexicon constraints are explicitly recorded in report, and apply only to generated
poses. Pending user feedback must not be treated as agreement.

## 2026-09-04 11:50 PST — offline unified streaming experiment is promising locally, not promoted

A separate experiment was implemented without modifying the accepted Reel or Stage-1
files. `active/v17/model_unified_streaming_ctc_v17.py` adds a 111,406-parameter causal
head over the phrase-adapted Stage-1 model's 256-dimensional pooled embedding and 100
gloss logits. Its output contract is CTC blank + the locked 100 glosses + explicit
`OTHER`; greedy decoding uses no expected-phrase table, template grammar, or language
prior. `active/v17/train_unified_streaming_ctc_v17.py` is the corresponding
protected-test-refusing trainer. Three focused tests prove output shape, preservation
of the initial Stage-1 gloss path, and that future evidence cannot change an earlier
prefix.

Training used all locally compatible supervised material: 434 exact-sequence phrase
clips, 879 ASLLRP known-plus-`OTHER` spans, 2,864 isolated one-gloss replay clips, and
5,636 incomplete-prefix/transition blank samples. How2Sign and 2M-Flores were not
assigned exact 100-gloss targets because their current archives do not provide equally
pinned labels; treating their genuine unknown signing as blank would be corrupting
supervision. No protected test or external reserved set was accessed.

Four matched variants were run. Pre-pooling frame evidence over cached windows reached
46.79% exact phrase match, 26.15% WER, 58.04% isolated exact, and 14.21% boundary false
emission. Pooled non-overlapping evidence reached 55.05%, 37.46%, 67.18%, and 16.58%.
The selected pooled 32-frame rolling-window variant at a four-source-frame stride
reached 76.15% exact phrase match, 10.60% WER, 66.00% isolated exact, and 12.64% false
emission. Freezing its gloss correction failed at 26.61% exact and 45.23% WER, proving
that timing-only adaptation is insufficient for coarticulated windows.

The rolling aggregate is not evidence of general continuous recognition. It scores
85.57% exact / 5.79% WER on 97 local held-out phrase clips, but 0/12 exact / 62.50% WER
on signer-disjoint ASLLRP contiguous phrases and 17.33% exact / 75.00% known-gloss WER
on 225 ASLLRP `OTHER` spans. It also reduces the same Citizen isolated split from the
base checkpoint's 95.77% to 73.28%. All twelve ASLLRP errors and many local errors are
under-emissions, such as `FRIEND NOW -> FRIEND`, `WORK WHERE -> WORK`, and
`HELLO HOW YOU -> HELLO HOW`.

The explicit non-phrase behavior works on the narrow supervision it saw: the selected
model rejects 97.68% of local incomplete-prefix crops and 97.53% of local transition
crops; the aggregate no-false-emission rate across all boundary crops is 87.36%.
This does not cover head scratching or arbitrary daily activity because ordinary
non-sign recordings are still absent. Keep the stable Reel path for current demos.
Do not promote or build claims around the new checkpoint until the planned 30-phrase
signer-disjoint collection is added and the model passes continuous WER, isolated
retention, and real non-sign false-activation gates. Full results are in
`artifacts/reports/unified_streaming_ctc_v17_experiment_v1/README.md` and the selected
experimental checkpoint is under
`artifacts/models/unified_streaming_rolling_ctc_v17_experiment_v1/`.

## 2026-09-04 11:16 PST — unified streaming recognizer selected as next design

Read-only tracing confirms that the default Reel path is online only at the orchestration
level. It starts a growing clip from wrist motion, repeatedly resamples the accumulated
clip to 32 frames, obtains one Stage-1 label, and uses score/stability/suppression rules
to commit it. The current landmark phrase adaptation fine-tuned a 100-way whole-clip
classifier using approximately equal phrase partitions, small neighboring context, and
boundary jitter. The current unified multimodal adaptation froze both encoders and
trained only the fusion head. Neither default model was trained to emit an ordered
gloss sequence from an unsegmented stream. The optional old Stage 2 remains disabled by
default.

The existing Squeezeformer does retain a `[B,T,256]` encoded sequence before its
attention pool, but `_pool` collapses time and its classifier emits one label. Its
self-attention and centered temporal convolution are also non-causal within each clip.
Therefore, the Squeezeformer can become the shared continuous encoder, but only by
branching before pooling and training a frame/chunk-level blank-plus-100-gloss sequence
objective. This can remove the separate old Stage-2 encoder/model; it cannot remove the
need for learned alignment, blank/background evidence, prefix decoding, and sequence
evaluation. “No separate Stage 2” and “no sequence modeling” are not equivalent.

The next bounded architecture is a single **unified streaming landmark recognizer**:
reuse the isolated-pretrained Stage-1 encoder, retain the isolated classification head
as an auxiliary/replay loss, attach a 101-way CTC projection before pooling, and train
on continuous phrases with blank/background and boundary-jitter supervision. Process
16/24/32-frame rolling chunks at fixed timebase, initially without a KV cache because
the measured model cost is already small. Display revisable partial hypotheses while
the signer continues; commit only the stable common prefix after limited right context.
The Finish control ends/naturalizes the utterance, not individual signs. Compare the
same training recipe on the current Squeezeformer and a flat/6-layer Transformer.
RNN-T/Emformer remains a fallback only if CTC prefix quality or longer context is shown
to be the limiting factor.

This differs materially from the rejected 26,861-parameter causal head. That experiment
received already pooled 100-way whole-window logits and could not recover temporal
detail: it reached 44.33% exact / 40.93% WER on local validation and 0/12 exact /
62.50% WER on held-out ASLLRP. It does not reject a shared-encoder continuous model;
it rejects sequence decoding after the useful time axis has already been collapsed.

Existing offline continuous material should be tiered rather than indiscriminately
merged. Exact local/ASLLRP target sequences can supervise CTC where label variants are
locked. ASLLRP `OTHER` spans are useful for foreground/background and OOV context.
How2Sign landmark archives and the 155 long 2M-Flores clips are useful for self-supervised
temporal pretraining, but their absent or non-pinned gloss mapping prevents treating
every token as exact 100-class truth. The current streaming training set had 434 real
phrase clips but only 44 unique genuine sequences, 50 directed bigrams, and continuous
coverage of 46/100 glosses; raw clip count overstates supervised transition diversity.

The planned 30-phrase collection is enough for a meaningful first unified-streaming
experiment, not a final general continuous claim. With an average five-gloss phrase,
the ten training signers and three performances yield 900 independent phrases and about
3,600 genuine transition events, but at most 120 phrase-position bigrams and only 150
lexical positions across phrase identities. Covering every gloss in roughly three
distinct contexts would require about 300 phrase positions, approximately 60 five-gloss
phrase identities. Do not spend for a second wave yet: maximize coverage in the first
30, run a compositional holdout/error audit, then collect only missing/confusable
contexts. Add 60--90 seconds of ordinary non-sign movements per signer at negligible
recording cost; these negatives directly target false activations such as scratching
the head.

No source/model was changed and no training was started. Before weekend capture, the
exact 30-sequence list must be audited for gloss, directed-bigram, confusable-family,
one/two-hand, location, and nonmanual coverage. The repository records the 30-phrase
plan but does not contain a frozen exact 30-sequence list.

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
