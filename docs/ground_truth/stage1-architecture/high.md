# stage1-architecture — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

19 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## v16 evidence and reason for replacement (undated — the basis of the v17 schema boundary)

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

## 2026-09-09 20:51 PHT — two repair packages pass independent frozen development evaluation

The initial linear OTHER preservation fit retained local phrases but failed exact/
contextual gates. Increasing false-rejection training cost to 10 retained context
but still emitted spurious OTHER. Investigation showed framewise OTHER can fragment
one accepted CTC emission run. The implemented repair preserves accepted known/blank
logits and applies a single conservative OTHER decision to each complete accepted
nonblank emission run, leaving blank runs unchanged and avoiding fragmentation.
The shared margin is max negative-training run score across both seeds (0.5793641)
plus log(2), i.e. 1.2725113. The safety factor/run policy were selected after
development inspection: this is explicitly development tuning, not an independent
test or a pre-registered comparison. Failed probes and both fitting histories remain.

Fresh full model inference from self-contained reloaded packages independently
reproduces: seed1701 target485/284; seed1702 target470/284; BOTH local/exact/contextual
6/9/43, Citizen331/378 and STEM16/21. All original gates plus STEM>=15 pass. Target
deletions are13/12 vs original v2 160/167, while insertions remain333/321; baseline
603 errors had2 deletions and467 insertions. This is a conservative accuracy tradeoff,
not a claim to retain v2's217/223 total errors or solve general continuous ASL.

Packages: artifacts/models/stage2_v17_transition_repair_v3/seed_1701.pth and
seed_1702.pth; reproducible builder scripts/build_stage2_other_preservation_v17.py.
Independent evaluator scripts/evaluate_stage2_other_preservation_v17.py wrote
artifacts/reports/stage2_v17_transition_repair_v3/validation.json. Selected seed1702
has SHA700c041a2a0a32365d351d5014467eda181a8b0c5c6c50356e09576aa40f559b.

Added tested model loader/conditional emission helper/run-level veto and opt-in live
CTC support. OTHER is removed only after collapse, retaining repeated known signs
and their emission positions. Eleven focused model/decode tests pass, including
observed red-green conditional/run/OTHER-filter regressions and real package reload.
Current defaults remain unchanged until Core ML accepted-head + CPU evidence parity
and latency are checked. No new acquisitions or protected evaluations occurred.

## 2026-09-08 16:09 PHT — expanded human review complete and frozen

The 206-row expanded ASL STEM Wiki review has a final disposition for every row: 198
were explicitly touched, while the remaining 8 are source-excluded P28 rows. Human
review admitted 111 exact bounded spans across 31 locked glosses and 18 participant IDs.
Twenty-two glosses have at least two approved participants, 18 have at least three and
14 have at least four. This clears the declared 2-train/1-validation participant floor
for 18 glosses and is sufficient for a bounded signer-disjoint Stage-2 adaptation, while
the thinner glosses may contribute training supervision but cannot support their own
signer-disjoint validation claim.

The read-only frozen queue is
`artifacts/reports/asl_stem_wiki_manual_expansion_admission_v17/expert_review_queue.final.csv`
with SHA-256 `38437d0afd506d050b0b89a452ae0a9d769d72e28496d9ab3b6c92235d64650a`; exact coverage
is in `final_review_summary.json`. The localhost reviewer was stopped after freezing to
prevent accidental edits. Next safe action: materialize only these 111 spans, normalize
by timestamps without aspect-ratio distortion, construct a participant-disjoint split,
and run a small Stage-2 adaptation with existing replay and retention gates.

## 2026-09-06 10:20 PST — clean adaptation rejected; matched rolling observation implemented

The first 14-epoch clean-lineage run ended with `no epoch passed retention gates` and
wrote no model. Original isolated validation was 96.83% Citizen / 87.22% SemLex; epoch 1
was 95.24% / 86.61%, narrowly below the declared Citizen floor; later epochs drifted to
93.39% Citizen. Local weak-core accuracy improved from 59.63% to 81.11%, but that does
not override retention. The bounded follow-up uses learning rate 5e-6 (from 1.5e-5),
distillation weight 4 (from 0.8), and 20 epochs, output
`artifacts/models/stage1_v17_grounded_clean_lineage_v2`, log `stage1_clean_v2.log` in
the same continuous rebuild report directory. It is running, not accepted yet.

Added `active/v17/continuous_evidence_v17.py` and
`active/v17/train_continuous_evidence_v17.py`: shared 8/16/32-frame trailing observations,
stride 4, learned causal scale fusion, explicit blank/100-gloss/OTHER output, genuine
annotated gaps, and duration-restored rolling isolated replay. Cache rejects encoders
with the historical local phrase-adaptation provenance and pins source hashes. Local
phrases use CTC without fabricated frame labels. A read-only load found 3,935 train and
1,854 validation sequences, all CTC-feasible at stride 4. Three focused tests in
`test/test_continuous_evidence_v17.py` pass: annotated gap exclusion including OTHER,
future-invariant observation/final tail, and causal model prefixes/shared gloss mapping.
No continuous training result is available yet.

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

## 2026-08-09 20:24 PST — frozen Apple/MediaPipe extractor bakeoff complete

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

## 2026-08-09 21:17 PST — Apple Vision selected as the v17 extractor

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

## 2026-08-09 21:53 PST — Frozen full-frame MobileCLIP2-S0 challenger rejected

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

## 2026-08-09 22:30 PST — Frozen high-resolution hand crops validate the correction

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

## 2026-08-09 23:17 PST — Frozen Apple model evaluated once on official test

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

## 2026-08-09 23:57 PST — Unfrozen MoViNet benchmark and execution boundary

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
