# Stage 1 v17: Squeezeformer, data, landmark, and objective audit

**Frozen development evidence through:** 2026-08-12 04:13 PST  
**Selection data:** Citizen official signer-disjoint validation (378 clips)  
**Cross-domain diagnostic:** SemLex validation (978 clips)  
**Forbidden during development:** consumed Citizen test and sealed SemLex test

## Bottom line

The d=256/depth=4 **part-wise + global Squeezeformer** is the best measured landmark
model in this repository. It is not proven to be the universally best architecture. The
evidence says the largest established gain came from better clean signer/source
coverage, not model scaling or a harsher generic loss. The remaining credible model
gains are sign-specific: preserve each articulator's temporal context, augment at the
articulator level, and explicitly expose bone/motion structure. These must be tested
one at a time under the same signer-disjoint protocol.

The best research teacher is already multimodal (97.88% Citizen validation and
90.18% SemLex validation with frozen Citizen-selected weights), but it is not yet a
production model or an independent accuracy estimate. The best compact landmark
checkpoint is now 96.83% Citizen validation and 87.22% SemLex validation. The consumed
Citizen test result remains 87.57% and cannot be revisited for architecture choice.
Replacing the old flat landmark member inside the previously fixed teacher weights
does not improve that ensemble (97.35% Citizen, 89.88% SemLex), so the single-model
and multimodal winners remain separate until independent calibration data exists.

## What the project has actually measured

| Change | Citizen val top-1 | SemLex val top-1 | Decision |
| --- | ---: | ---: | --- |
| d=256 part-wise + global, same controlled protocol | **96.83** | **87.22** | New development landmark winner |
| d=256 part-wise + global, matched seed 3407 | 95.77 | 86.40 | Replicates +4 Citizen/+13 SemLex clips over matched flat seed |
| d=256 flat + bone/bone-motion, same protocol | 96.03 | 86.40 | Positive isolated component; combine with part-wise |
| d=256 part-wise + bone interaction | 96.30 | **87.83** | SemLex gain, Citizen loss vs part-wise; reject as winner |
| fixed equal part-wise + part-wise-bone logits | 97.09 | 88.75 | Best landmark ensemble; double runtime, research only |
| part-wise + per-part auxiliary loss 0.20 | 96.56 | 88.04 | SemLex gain, -1 Citizen clip; reject as winner |
| flat + 30 hand flexion angles | 96.30 | 85.69 | Citizen +2, SemLex -2; reject |
| part-wise + per-keypoint temporal gate | 96.30 | 86.71 | Below part-wise on both domains; reject |
| d=256 flat, Citizen + full-clean SemLex, balanced | 95.77 | 85.89 | Superseded flat baseline |
| d=384, same controlled data/protocol | 95.24 | **86.71** | Too large; no Citizen gain |
| alternate seed 3407 | 94.71 | — | Seed sensitivity exists |
| local Tier-A at 10% expected exposure | 95.77 | 86.50 | Cross-domain help, no Citizen gain |
| label smoothing 0.05 instead of 0.10 | 95.50 | — | Keep 0.10 |
| supervised contrastive, fixed/decayed | materially worse | — | Reject generic SupCon |
| hands-only landmark input | worse | — | Non-manual/body context matters |
| graph replacement | 78.31 | — | Reject this graph design |
| zero-gated flat + graph residual | no gain | — | Reject this graph residual |
| phonology auxiliary, weight 0.20 | 95.50 | 85.07 | Reject simple equal heads |
| mirror test-time augmentation | worse | — | Reject |
| checkpoint soups | no gain | — | Reject |
| landmark-only probability ensembles | up to 96.56 | — | Small gain, extra runtime |
| SKIM-style PartMix p=0.5 | 95.77 | 85.58 | Tie/regression; reject as default |

The strongest causal clues are now the data and part-isolation ablations. Expanding from Citizen-only training
(93.12%) to quality-gated full-clean SemLex training (95.77%) added ten net Citizen
validation clips. Increasing width from 6.47M to 14.34M parameters did not improve
Citizen validation. Preserving each articulator's temporal context before global
fusion then added another four Citizen and thirteen SemLex top-1 clips with only
4.96% more parameters. This does not prove the model is saturated, but it strongly
argues against width or a wholesale Squeezeformer replacement as the next lever.

## Is the data the problem?

Partly. The model has enough clean data to learn the 100-class task, but not enough
independent portrait-phone diversity to justify a production-generalization claim.

- Citizen supplies 1,475 usable signer-disjoint training clips; full-clean SemLex
  supplies 1,388 exact-variant, quality-gated train clips from 32 train signers.
- Citizen validation and SemLex validation differ by 9.61 top-1 points for the new
  landmark checkpoint. That gap is real domain sensitivity.
- The 1,021-clip local audit found 434 already-used Tier-A clips and only 23 new
  conservative candidates. The 23 still require ASL-fluent exact-variant review.
  Model agreement cannot establish lexical correctness.
- More raw volume without source caps would over-weight a few local sessions and can
  reduce generalization. New independent signers and portrait-phone capture are more
  valuable than repeated clips from the same signer/session.
- The frozen confirmation gate is five new people x two repetitions x 100 variants =
  1,000 clips, plus a recommended 100 out-of-vocabulary clips.

The best next data use is therefore not “admit everything.” It is exact-variant human
review of the 23 local upgrades, controlled source weighting, and the independent
portrait-iPhone set. Unlabeled local clips may later support masked-pose pretraining,
where uncertain class labels are not used, but extraction quality still needs a gate.
The 66,409 legacy Apple arrays are not a ready v17 pool: their ten channels are
XYZ/velocity/acceleration plus an `ever observed` part mask, with older normalization,
interpolation, and chirality semantics. Reusing them requires an explicit adapter or
v17 re-extraction, not taking their first five channels.

## Are the landmarks the problem?

Not in the simple sense of “too few hand joints.” Apple v17 already has all 21 joints
for each hand, plus 15 face anchors and four arm/shoulder nodes, with X/Y/depth,
presence, and confidence. The model internally derives masked velocity,
acceleration, and 66 hand-shape distances.

- Adding more nominal hand points is not available from the current Apple hand
  contract and would require a new extractor/model schema.
- Dense face landmarks are unlikely to be the first lever. Face-only landmarks scored
  2.65%, while RGB mouth/lower-face heads carry more signal but transfer poorly
  between Citizen and SemLex. The main issue is appearance/domain supervision, not
  merely face point count.
- Four body nodes omit torso/head pose detail, but isolated-sign errors are dominated
  by near-semantic/manual confusions rather than gross body-motion failures. Extra
  torso nodes are a lower-priority extractor experiment.
- The most credible missing landmark representation is **explicit bone direction and
  bone motion**, not more raw coordinates. These are algebraically recoverable from
  joints, but exposing them can improve sample efficiency and regularization.

## Is the model learning incorrectly?

The ordinary objective is already well calibrated relative to tested alternatives:
cross-entropy with label smoothing 0.10, EMA, warmup/cosine decay, geometry-valid
augmentation, and 50/50 class/source-balanced replacement sampling. Lower smoothing,
generic supervised contrastive penalties, naive phonology heads, TTA, and soups did
not improve the controlled result.

The remaining 16 Citizen validation errors are mostly singleton confusions, including
THANKYOU/GOOD, NIGHT/DOCTOR, LIKE/MY, LIKE/HUNGRY, and BAD/GOOD. This pattern does not
support simply “punishing errors harder.” A harsher loss can amplify label/domain
noise. It supports learning articulator-specific temporal detail and adding clean
examples of the exact confused variants.

## Strongest relevant studies and what they imply here

1. **SKIM / Part Mixing.** SKIM swaps a corresponding articulator such as a hand
   between samples and mixes the labels. It is sign-specific augmentation with no
   inference cost. This is the first controlled challenger now implemented at
   `--partmix-probability 0.5`. [IEEE DOI](https://doi.org/10.1109/TMM.2023.3321502)

2. **P3D.** P3D alternates part-wise and whole-body temporal Transformers and reports
   that part-specific motion context improves pose-based recognition. This directly
   addresses the current model's early flattening. A compact v17-native version is
   implemented as `--temporal-encoder partwise_global`; it has 6.79M parameters,
   only 4.96% above the winner. [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Human_Part-wise_3D_Motion_Context_Learning_for_Sign_Language_Recognition_ICCV_2023_paper.html)

3. **Siformer.** Its feature-isolated left-hand/right-hand/body encoders reinforce the
   same hypothesis and add missing-joint rectification. Its released 204-frame 2D
   input and Transformer are not checkpoint-compatible with v17; the concept, not
   the weights, is being tested.

4. **DSTA-SLR.** DSTA uses input-sensitive and domain-graph spatial branches plus
   multi-scale temporal modeling; its released evaluation ensembles joint, bone,
   joint-motion, and bone-motion streams. This supports an explicit compact bone
   feature challenger, but the upstream 27-joint/120-frame/CUDA model cannot be
   silently attached to v17. [LREC-COLING 2024 paper](https://aclanthology.org/2024.lrec-main.484/)

5. **VSNet.** VSNet dynamically fuses joints, discards weak connections, groups
   visual-symbol-like joint sets, and models their motion. It is the strongest recent
   evidence that sign-linguistic groupings can beat generic human-action skeleton
   assumptions without large pretraining. It is a later, higher-complexity challenger
   after the cleaner PartMix/part-wise tests. [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Li_VSNet_Focusing_on_the_Linguistic_Characteristics_of_Sign_Language_CVPR_2025_paper.html)

6. **BEST and SignBERT.** Both use masked pose pretraining rather than only supervised
   classification; BEST adds coupling tokenization, while SignBERT adds hand-aware
   priors. These are credible if architecture-only changes plateau and a sufficiently
   large quality-screened unlabeled v17 corpus is extracted. [BEST, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25470),
   [SignBERT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html)

7. **Dual-reference morphology/trajectory modeling.** DSLNet separates wrist-relative
   finger configuration from body/face-relative wrist trajectory. Its five-page arXiv
   report and benchmark protocol are not strong enough to justify importing the full
   46.3M model or its optimal-transport fusion, but the coordinate decomposition is a
   cheap, testable component. v17 already has body-relative coordinates and
   translation-invariant hand distances; bone features test much of the missing local
   shape signal. A direct wrist-relative-coordinate stream remains a later ablation if
   bone and part-wise interactions plateau.
   [arXiv paper](https://arxiv.org/abs/2509.08661)

8. **Keypoint-importance diagnostics.** Holmes et al. find that outer-finger tips/bases
   dominate a pose SLR model while inner finger joints and coarse facial landmarks are
   under-used, plausibly from occlusion, missing depth, and insufficient face detail.
   This supports quality-aware/missing-aware part training and RGB non-manual support;
   it does not justify deleting low-importance joints, since model importance can miss
   linguistic importance.
   [LREC-COLING 2024 paper](https://aclanthology.org/2024.lrec-main.1387/)

No paper result is directly comparable to this project's ASL-100 vocabulary,
Apple-extracted features, signer-disjoint split, or mobile constraints. Published
benchmark rank is a hypothesis source, not proof that a model will win here.

## Component-level transplant audit

The papers are not being treated as indivisible models. The useful unit is the
smallest mechanism that tests a new hypothesis inside the frozen v17 pipeline.

| Paper mechanism | v17 component interpretation | Priority / evidence |
| --- | --- | --- |
| SKIM part mixing | Replace one complete detected hand during training | Tested; rejected on top-1/top-5 gates |
| P3D/Siformer feature isolation | Separate hand/face/body temporal encoders before global fusion | Tested; new winner on both top-1 domains |
| Tran et al. region-level supervision | Training-only gloss head on each isolated part stream | High priority after isolated part-wise/bone runs; zero deployment cost when heads are dropped |
| DSTA/LGF joint-bone-motion streams | Internally derive bone and bone-motion channels, keep one model | Tested; improves both domains, smaller than part-wise |
| Handshape-GNN canonical frame | Add a static hand-shape token from reliably detected low-motion frames | Promising later component; must not replace dynamic modeling or assume the paper's handshape labels exist here |
| Handshape-GNN/SignRep angle prior | Add 30 missing-aware cosine finger-flexion channels internally | Tested; rejected because the isolated run failed the both-domain gate |
| DSLNet dual reference frames | Preserve body-relative wrist trajectory while adding wrist-relative hand coordinates | Later low-cost ablation; pairwise/bone features already cover part of this hypothesis |
| Keypoint-importance audit | Measure part/joint reliance and target missing/occluded articulators | Diagnostic only; never delete a joint solely because the current model under-uses it |
| SKIM per-keypoint temporal reweighting | Identity-initialized joint-specific gates from confidence/speed/acceleration | Tested; rejected below part-wise on both domains |
| SML multi-feature aggregation | Gated interaction of joint, motion, and bone features | Conditional: only if simple bone concatenation shows signal |
| SML self-knowledge distillation | EMA/deep-head teacher loss during training, no inference cost | Medium; isolate carefully because generic SupCon already hurt |
| Siformer kinematic hand rectification | Audit/correct impossible finger angles before classification | Audit first; upstream is 2D, hard-coded, and ignores v17 depth/missing/confidence |
| VSNet visual-symbol groups | Learn small joint-group tokens rather than a full graph replacement | Promising later architecture; higher implementation uncertainty |
| StepNet long/short part temporal context | Add multi-scale temporal branches per anatomical stream | Partly covered by Squeezeformer attention + depthwise convolution; lower novelty |
| LGF Bayesian score fusion | Tune local/global stream weights | Reject for selection: validation optimization risks overfit and adds models |
| BEST/SignBERT masked pose modeling | Pretrain the same encoder on quality-gated unlabeled v17 clips | Strong heavier follow-up after supervised component ladder |
| SHuBERT multi-stream masking | Random short spans masked independently for hand/face/body channels | Useful objective detail; published scale is about 984 hours, far beyond the current compatible pool |
| Cross-view ISLR view synthesis | Rotate 3D pose and separate viewpoint from sign semantics during training | Promising robustness component; direct synthetic-view training alone had limited gains in the source study |
| Articulated pose-distance embedding | Pretrain a tiny frame MLP to preserve hierarchical bone-orientation neighborhoods | Promising higher-cost geometry prior; require a capacity-matched random-branch control |
| HTMA attention-score mixing | Convolve each head's `T x T` score map before softmax | Higher-risk temporal challenger; source ablation is not signer-disjoint and layer count is inconsistent |

The bounded HTMA interpretation used exactly one zero-initialized,
depthwise 3x3 residual convolution per score map. It adds 576 parameters to the
part-wise model and deliberately excludes the paper's full CNN and ambiguous
four-convolution description. It reaches 363/378 Citizen and 845/978 SemLex, below
part-wise-only at 366 and 853, and is rejected despite small top-5 gains.

SML's paper exposes three separable ideas—multi-feature aggregation, adaptive
residuals, and self-knowledge distillation—rather than requiring its entire GCN.
[SML paper](https://www.sciencedirect.com/science/article/pii/S0950705124009225)
StepNet similarly motivates part-level long/short temporal modeling, but its RGB
part extractor is not needed for the landmark experiment.
[StepNet paper](https://doi.org/10.1145/3656046)

The released Siformer rectifier was inspected directly. It rotates 2D finger joints
toward hand-authored flexion/extension and abduction/adduction ranges. Applying it
blindly would discard v17 depth, missingness, and confidence semantics and may erase
valid lexical extremes. It is therefore an audit candidate, not an automatic cleanup
step. The released LGF-SLR repository currently contains only a README, not executable
model code; its useful evidence is the local-hand/global-body and joint/bone/motion
decomposition, already represented by the compact v17 experiments.

Two newer component results sharpen the ladder without requiring their complete
models. Tran et al.'s signer-independent pose ablation found that region-wise
decomposition alone was only a modest improvement, while attaching an auxiliary gloss
decoder to every region produced the largest reported step in that ablation. The
portable hypothesis is therefore **per-part training supervision**, not their
continuous permutation decoder. It can be tested with four tiny heads on the current
left-hand/right-hand/face/body streams and removed for deployment.
[ICCVW 2025 paper](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Tran_Region-Aware_Pose_Modeling_and_Permutation_Decoding_for_Signer-Independent_Sign_Language_ICCVW_2025_paper.pdf)

Carbo and Nalisnick separately model full handshape dynamics and a representative
low-motion static frame. Their large gain is on a 37-handshape PopSign task with
handshape supervision, not this 100-gloss task, so the result is not transferable as
a number. The usable component is a small static canonical-hand token or auxiliary
branch selected only from present, reliable joints; it must complement rather than
replace motion. This is distinct from the already rejected global ASL-LEX phonology
heads because it changes where the handshape representation is extracted.
[EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.1483/)

SignRep is an RGB masked-autoencoder study, not a drop-in pose classifier, but its
pretraining ablation is useful at component level: joint-angle, keypoint, and distance
priors work best together, and hand-activity-weighted temporal aggregation beats plain
averaging. v17 already implements keypoints, distances, and hand-aware pooling; the
new angle channel tests the one missing geometric prior under our own signer-disjoint
protocol. This is an inference from the paper's mechanism, not a claim that its RGB
result transfers.
[ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html)

Sartinas et al. provide a more targeted geometry component: a 64-dimensional MLP is
pretrained with triplets ranked by an articulated pose distance that recursively
aligns proximal-to-distal bone orientations. Their five-run table importantly includes
a randomly initialized embedding branch; that control already explains much of the
headline improvement, while articulated-distance initialization adds a smaller further
gain. The transferable experiment is therefore the pretrained geometric branch plus a
capacity-matched random branch—not their full Transformer and not the claimed headline
delta. This overlaps with the positive v17 bone signal but differs from simply
concatenating angles. [VISAPP 2026 paper](https://www.scitepress.org/Papers/2026/146488/146488.pdf)

Varanasi et al.'s portable submodule is **attention-score mixing**: after QK scores
are formed, a small 2D convolution mixes neighboring head/token relations before
softmax. Their table places the HTMA variants above capacity-matched ordinary MHA,
so the mechanism is separable from the full 1D-CNN model. Evidence quality is limited:
the architecture ablation uses INCLUDE's original split rather than the paper's
pseudo-signer grouping, and Algorithm 3 says four score-convolution layers while the
hyperparameter table says one. Any v17 test must therefore implement one declared
version and use the frozen signer-disjoint protocol; the paper's desktop latency and
nominal mobile suitability are not iPhone evidence.
[CVPRW 2026 paper](https://openaccess.thecvf.com/content/CVPR2026W/MSLR/papers/Varanasi_Isolated_Sign_Language_Recognition_via_MediaPipe_Landmarks_A_Case_Study_CVPRW_2026_paper.pdf)

## Predeclared experiment ladder

1. **PartMix p=0.5, unchanged Squeezeformer — completed/rejected.** The Tesla-T4 run
   retained epoch 51 and tied Citizen top-1 at 95.77%, but fell to 99.47% top-5 and
   95.31% macro F1. SemLex validation fell from 85.89%/96.11% top-1/top-5 to
   85.58%/95.30%; its present-class macro F1 rose from 82.60% to 83.45%. The realized
   mix fraction averaged 50.11%. Because both domain top-1/top-5 gates fail to improve,
   PartMix does not replace the baseline and is not combined with later challengers.
2. **Part-wise temporal + global Squeezeformer, no PartMix — completed/accepted.**
   Same data, seed, sampler, optimizer, patience, and validation; +4.96% parameters.
   It improves Citizen from 362/378 to 366/378 and SemLex from 840/978 to 853/978.
   Top-5 changes from 378 to 375 Citizen clips and 940 to 939 SemLex clips. This is
   the new development landmark winner, pending independent portrait confirmation.
3. **Explicit hand/arm bone-vector and bone-motion input — completed/positive.**
   The public `[32,61,5]` input remains fixed and missing-aware bones are internal.
   The 6,564,581-parameter model reaches 363/378 Citizen and 845/978 SemLex, versus
   362 and 840 for flat. It is a useful component, not a new winner.
4. **Part-wise + bone interaction — completed/mixed.** It reaches 364/378 Citizen and
   859/978 SemLex, versus part-wise-only at 366 and 853. SemLex top-5 improves from
   939 to 949, but the primary Citizen gate loses two clips. Do not replace part-wise;
   retain this only as cross-domain tradeoff evidence.
5. **Per-part auxiliary gloss supervision — completed/mixed.** Missing-aware heads at
   fixed weight 0.20 reach 365/378 Citizen and 861/978 SemLex. This loses one primary
   clip but gains eight cross-domain clips. Do not replace part-wise; retain the
   training-only mechanism as domain-robustness evidence.
6. **Explicit hand joint-angle channels — completed/rejected.** The 30 missing-aware
   cosine values reach 364/378 Citizen but only 838/978 SemLex. The invariant cue is
   not robust enough to combine with part-wise under the frozen gate.
7. **Per-keypoint temporal reliability gate — completed/rejected.** The
   identity-initialized confidence/speed/acceleration gate reaches 364/378 Citizen
   and 848/978 SemLex, below part-wise-only at 366 and 853. The one-clip Citizen
   top-5 gain does not offset the top-1 losses; do not combine it.
8. **Articulated pose-distance embedding — completed/rejected.** The capacity-matched
   random branch ties Citizen at 366/378 and improves SemLex from 853 to 856/978. The
   identical distance-pretrained branch also reaches 366 Citizen but falls to 842
   SemLex, fourteen clips below its random control. Extra wrist-relative capacity has
   slight cross-domain signal; the paper-derived initialization itself is rejected.
9. **Static canonical-hand branch — completed/mixed; low-motion rejected.** The
   quality-only control reaches 367/378 Citizen and 851/978 SemLex. The identical
   low-motion treatment also reaches 367 Citizen but falls to 840 SemLex, eleven clips
   below its control, and loses two Citizen top-5 clips. Reliable-frame capacity has
   weak mixed signal; low-motion selection itself is rejected.
10. **PartMix + part-wise model.** Run only if both isolated experiments help or one
   clearly helps without destabilizing SemLex. Interaction is otherwise uninterpretable.
11. **Masked-pose pretraining — completed/rejected.** The four-part span objective
   converges to reconstruction loss 0.0571, but the unchanged 6.79M fine-tuned model
   reaches only 360/378 Citizen and 819/978 SemLex, versus 366 and 853 for the matched
   part-wise control. Do not tune masks on these validation sets; a retry requires a
   substantially larger compatible pool or a different pretext objective.
12. **VSNet-like visual-symbol grouping.** Highest architectural uncertainty and Core
   ML work; attempt only if the compact ladder fails to move both domains.
13. **Attention-score mixing — completed/rejected.** One zero-initialized depthwise
   3x3 score residual per attention block reaches 363/378 Citizen and 845/978 SemLex.
   The mixers learn nonzero weights but lose three and eight top-1 clips respectively.
   Do not escalate to the source paper's inconsistent four-layer variant on the same
   validation sets.

For every run, a candidate must improve Citizen top-1 without a material SemLex
regression, remain under the mobile size/latency envelope, and never access either
test split. A one- or two-clip Citizen gain is exploratory until the independent
portrait set confirms it.

## What should not happen next

- Do not rerun Citizen test or touch SemLex test during this ladder.
- Do not claim 95.77% validation or the 97.88% teacher as production accuracy.
- Do not increase width again before testing sign-specific structure.
- Do not add arbitrary loss penalties after SupCon/phonology failures.
- Do not add dense landmarks without an extractor bake-off and measured coverage.
- Do not admit the 23 new local clips before exact-variant human review.
- Do not merge multiple new ideas into one run before their isolated effects are known.
