# mobile-deployment — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

7 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

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
