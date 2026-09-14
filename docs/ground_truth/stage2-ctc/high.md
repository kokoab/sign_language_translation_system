# stage2-ctc — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

9 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-14 — future Stage-2 training must fix coverage and supervision geometry

The exhaustive O5S5 post-run audit is binding for future experiments. Source-balanced
sampling by source name did not cover the admitted pool: 63.75% of ASLLRP OTHER and
15.32% of O5S5 windows were never drawn in 12 epochs. Fixed 1.07s target classification
dilutes O5S5 targets to 29.1% annotated foreground on average. O5S5 is not uniquely
too fast, and exact-core pooling does not repair held-out LG, so neither slower playback
nor tighter crops alone addresses the failure. Per-class signer support, rather than six
aggregate signer IDs, is the relevant coverage statistic.

Do not reuse whole-window target CE for long mixed context, treat unverified O5S5 gaps
as blank, or repeat source-balanced random subsampling without full-pool coverage.
Preserve isolated replay while allowing sequence loss to adapt frame features. The next
candidate should use a shallow temporal CTC head on unpooled Stage-1 features, with
O5S5 exact-core supervision and blank labels only where ASLLRP annotation verifies the
gap. Evidence: `artifacts/reports/stage2_data_learnability_audit_v17/`.

## 2026-09-10 09:10 PHT — approved continuous-path implementation and matched adaptation plan

User authorized the diagnosis recommendations before a later discussion of returning
to faster Stage1 Reel locks. Preserve that scope: (1) sequence-driven provisional
preview with independently measured conservative prefix confirmation; (2) shared
training/live observation/window contract; (3) temporal adaptation on existing real
connected recordings with original retention evaluation. Do not substitute another
isolated-window stabilizer or claim general recognition fixed. Existing default and
checkpoints remain until promotion evidence. Experimental --sequence-preview runs
Stage2 during capture, bypasses Stage1 proposal/verifier work, and keeps uncertain
final tails for review; immutable prefixes need two fresh matching position-aware
hypotheses plus one window of lookahead. Duplicate evidence cannot confirm or clear
conflicts; true repeated signs retain separate emission positions. Focused tests
observed failing then passing for observer parity, online execution and prefix rules.

Shared observe_stage2_frame uses640px detections,1280px hand crops,20Hz nominal
source-time processing, sparse8-frame body/face phase and elapsed1.0667s windows.
New cache script scripts/cache_stage2_live_matched_v17.py now re-extracts all1647
admissible original real clips:1313 train (390 local,44 exact ASLLRP,879 OTHER spans)
and334 development (97 local,12 exact,225 OTHER spans). Same live frozen encoder,
hand crop implementation and saved source/video hashes; no protected split or new
bulk acquisition. Source-specific lip-node masking is removed in this new matched
contract because real runtime has no source identity. Legacy archives untouched.
Cache is resumable per source/hash/contract. Validation first provides paired input
and prefix checks while the training cache completes.

Planned bounded adaptation: preserve complete accepted primary/context/specialist
initialization, add differentiable existing OTHER evidence (no new backbone), train
Stage2 temporal weights with source-balanced CTC and sequence-level replay/retention,
compare matched inputs before/after and all original gates. Stage1 remains frozen.
Endpoint/hold augmentation will be explicit, not duplicated whole-sign windows or
fabricated neighboring glosses. Report failures and coverage as well as WER; no
promotion on a convenient subset. CoreML cached-feature parity alone is insufficient;
raw-video and online-prefix latency/contradiction evidence required before handoff.

## 2026-09-08 16:18 PHT — correction: reuse Stage 1 for temporal CTC adaptation

User challenged the implication that continuous adaptation requires relearning the
100 isolated signs or collecting every missing exact variant first. Code inspection
confirms the accepted path already reuses frozen Stage-1 landmark/hand encoders and
fusion: `model_stage2_v17.py` produces framewise 612-D evidence, then
`Stage2TemporalHeadV17` learns temporal context and CTC outputs. The accepted runtime
uses the general primary/specialist selector, not the later experimental causal TCN.
`train_stage_2_other_ctc_v17.py` already demonstrates warm-started Stage-2 training,
replay and teacher distillation; its trainable backbone is Stage 2, not Stage 1.
Its loader accepts plain/pretrained Stage-2 checkpoints, not the packaged selector
directly, so checkpoint compatibility must be respected in any implementation.

Correction to the preceding recommendation: 64/100 exact-variant continuous coverage
is a supervision/evaluation inventory, not a prerequisite to train or proof that the
remaining 36 signs cannot transfer from Stage 1. CTC learns sequence alignment without
requiring manually fixed sign boundaries; the temporal network learns contextual
recognition under that loss. Blank is a no-emission symbol, not a physical transition
class. Transfer to unseen contexts remains a measured question, not a guaranteed result.

The frozen STEM queue verifies 111 individual target intervals, not 111 fully verified
multi-sign transcripts. Tight core-only crops cannot supply the surrounding transitions
by themselves. Next safe direction: preserve the existing Stage-1 encoder and adapt the
existing Stage-2 family on admissible connected sequences with reliable ordered labels,
using reviewed cores as supplemental supervision and replay to preserve vocabulary.
Do not label unverified surrounding signing as blank or invent neighboring targets.
Measure sequence errors and retention before deciding which further collection is needed.
This turn inspected code, saved reports and review metadata only; no training, model/data
changes or protected evaluation access occurred. Only this handoff was updated.

## 2026-09-08 13:58 PHT — accepted models produced review-only boundary proposals

Added `scripts/auto_annotate_asl_stem_wiki_v17.py` and completed a bounded scan of all
71 review rows across 59 source videos. Each scan searches within five seconds of the
expected token position, reuses Apple Vision observations, ranks dense spans with the
accepted Reel landmark classifier, and reranks its five best candidates with the
accepted frozen multimodal Stage-1 encoder and Stage-2 CTC selector. The complete
machine-readable result is
`artifacts/reports/asl_stem_wiki_auto_annotation_v17/annotations.json`.

Calibration against the 21 rows already reviewed found only 5 strict successes: the
target must have model evidence and both inclusive endpoints must be within five frames
of the human bounds. No threshold could retain at least three successes while excluding
all reviewed failures. Several rejected or unsure Citizen variants still received strong
target-gloss/CTC agreement. The honest result is therefore 0 high-confidence automatic
approvals, 40 review proposals and 10 abstentions among the 50 untouched rows. No human
field was overwritten, no automatic proposal is training eligible, and no protected
Citizen/SemLex/local/RIT evaluation split was accessed.

The existing localhost reviewer now loads the report after validating every row's queue
identity. It shows the proposed span, uncalibrated evidence, Stage-1/CTC agreement and
top predictions; provides copy/play controls; and filters review proposals separately
from abstentions. Copying bounds is editable and unsaved until the reviewer explicitly
saves. The 8 UI tests, 7 scanner tests and 8 acquisition/admission tests pass; scripts
compile, browser JavaScript parses, media range serving works, and `git diff --check`
passes.

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
