# iOS-First Accuracy and Dataset Plan

**Status:** Discussion and decision record  
**Date:** 2026-08-09  
**Scope:** Stage 1 isolated recognition first; iOS first; offline operation; accuracy before model compression

## Current decisions

1. Develop and validate the first mobile implementation on iOS.
2. Use Apple Vision as the on-device hand/body landmark extractor.
3. Reduce the initial vocabulary from approximately 300 signs to approximately 100 useful, well-supported signs.
4. Prioritize unseen-signer and real-phone accuracy before distillation, INT8 quantization, or shrinking the Squeezeformer.
5. Use d=256/depth=4 as the first v17 accuracy baseline; the older capacity check
   favored 256, while d=384/depth=6 is 21.15M parameters for only 1,475 clips.
6. Do not treat the current 96% Stage 1 result as evidence of generalization. The current split randomly mixes sources/signers and the source data contains only about seven recurring people.
7. Fix portrait/landscape geometry before retraining. Augmentation alone is not a sufficient correction.
8. Do not use PopSign as the sole primary corpus: it intentionally captures one-handed
   smartphone variants. Use ASL Citizen as the single primary 100-sign baseline and
   reserve PopSign for optional, explicitly scoped portrait one-hand auditing.

## Recommended first product

The first product should be an offline iOS isolated-sign recognizer:

```text
iPhone camera
  -> Apple Vision hand landmarks on sampled frames
  -> Apple Vision body anchors at a lower frequency
  -> aspect-correct landmark canonicalization
  -> 32-frame landmark ring buffer
  -> selected v17 Stage 1 Core ML model
  -> 100-sign prediction plus UNKNOWN / PLEASE REPEAT
```

Stage 2 continuous recognition and Stage 3 translation should not determine the first dataset or accuracy milestone. Stage 1 should first prove that it can recognize unseen people using real portrait-oriented phone recordings.

## Why Apple Vision is the first extractor

Apple Vision is native, offline, and already matches the current v16 training representation. It does not add a separate pose-model download to the application. The Vision hand-pose request supports two hands and produces recognized hand joints directly from a camera pixel buffer.

- Apple hand-pose API: <https://developer.apple.com/documentation/vision/detecthumanhandposerequest>
- Current extractor: `active/v16/extract_v16.py`
- Existing Stage 1 Core ML package: approximately 29 MB FP16

A quantized ONNX whole-body pose estimator is not the preferred low-end path. Weight quantization does not eliminate person detection, hand cropping, or pixel-level pose inference. A conventional body-pose model also does not provide enough finger detail for word-level sign recognition. More powerful whole-body estimators remain useful as offline teachers and data-auditing tools.

## The present 96% result is not the target metric

The Stage 1 dataset code performs a deterministic random file split per class. This intentionally mixes sources and signers across train, validation, and test. With only about seven recurring people, the model can learn signer appearance, proportions, motion habits, framing, and background regularities.

The new headline metric must be:

> Top-1 accuracy on completely unseen signers, recorded on phones that are not represented in the training set.

Additional required metrics:

- Macro top-1 and top-5 accuracy across all 100 signs
- Per-signer accuracy and worst-signer accuracy
- Per-class recall and confusion pairs
- UNKNOWN rejection accuracy
- Hand-detection coverage per video
- Portrait and landscape accuracy reported separately
- Near/far framing and bright/dim lighting accuracy
- End-to-end latency on the target iPhone, including Apple Vision
- Sustained latency and thermal behavior after at least 10 minutes

## Signer split requirement

Twenty training signers and five test signers is a much better minimum than the present data, but a validation population is also required. The test signers cannot be used for early stopping, threshold tuning, class selection, or extractor tuning.

Recommended minimum:

| Split | Signers | Takes per signer per sign | Clips for 100 signs |
| --- | ---: | ---: | ---: |
| Train | 20 | 5 | 10,000 |
| Validation | 5 | 5 | 2,500 |
| Test | 5 | 5 | 2,500 |
| **Total** | **30** |  | **15,000** |

If only 25 signers are available, use 16 train, 4 validation, and 5 locked test signers. Do not perform validation on the five test signers.

Signer diversity is more valuable than repeated takes from the same person. A source containing one take from 30 people can add more generalization value than 20 takes from seven people. Repeated takes are still useful for learning natural within-signer variation.

## Dataset recommendations

### 1. PopSign ASL v1.0 — optional one-handed portrait audit

**Recommendation:** Do not use as the sole primary corpus. Retain only for explicitly
reviewed one-handed portrait-domain experiments.

- 250 isolated ASL signs
- 47 consenting Deaf adult signers for whom ASL is their primary language
- More than 200,000 smartphone videos
- Recorded using Pixel 4A selfie cameras at an original portrait resolution of 1944x2592
- Approximately 20 examples per signer per sign before validation filtering
- Manually reviewed into intended game-sign, variant, and unrecognizable categories
- Official signer-disjoint split: 31 train, 8 validation, 8 test
- CC BY 4.0
- Approximately 1.1 TB for the complete dataset

Official resources:

- Dataset card and license: <https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/>
- Paper: <https://papers.neurips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract-Datasets_and_Benchmarks.html>
- Selective download guide: <https://signdata.cc.gatech.edu/view/guides/downloading_popsign/index.html>

Important limitation: PopSign focuses on one-handed smartphone signing because participants may hold the phone with the other hand. It is excellent for portrait geometry and mobile variability, but it must not silently replace standard two-handed variants. Use its `game` category first. Treat `variant` examples as separately reviewed data rather than merging them automatically.

The server exposes downloads by sign, category, and split, so a 100-sign subset can be downloaded without retrieving the entire 1.1 TB dataset. Each selected sign archive may still be large; process one sign at a time, retain the chosen clips and metadata, then archive or discard unneeded local copies according to the dataset license.

### 2. ASL Citizen v1.0 — primary v17 baseline

**Recommendation:** Single primary training source for the first 100-sign v17 baseline.

- 83,399 isolated ASL videos
- 2,731 signs
- 52 consented Deaf and hard-of-hearing participants
- Everyday home/webcam environments
- Anonymous user IDs supplied for signer-level splitting
- Official user-disjoint train/validation/test split
- 40,154 train, 10,304 validation, and 32,941 test videos
- 42.8 GB self-contained download
- Research-use license; read the license before redistribution or product use

Official resources:

- Download: <https://www.microsoft.com/en-us/download/details.aspx?id=105253>
- Dataset card: <https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/>
- License: <https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/>
- Paper: <https://arxiv.org/abs/2304.05934>

ASL Citizen used an ASL-first prompting procedure and filtered blank/non-signing videos, but its authors explicitly note possible regional variants, mistakes, and residual technical noise. It should still receive an automated extraction audit and a manual audit of the selected 100 signs.

ASL Citizen is mostly webcam/home-domain data, so it does not solve portrait-phone
generalization by itself. Its strengths are ordinary one- and two-handed signing,
signer diversity, consent, stable hosting, manageable size, and official signer-disjoint
splits. The frozen exact-variant Citizen100 manifest uses a 10 train / 3 validation / 5
test signer-per-class floor and contains 32/5/11 unique selected participants overall.
Its license is for noncommercial research, so a commercial model would require
separately licensed data.

### 3. MS-ASL100 — useful secondary benchmark

**Recommendation:** Secondary robustness source, not the clean primary corpus.

- Official ASL100 subset with 100 signs
- 189 signers and 5,736 videos in the published ASL100 statistics
- Official signer-independent partitions
- Unconstrained Internet video conditions
- Microsoft provides the annotation package; the original samples are URL-based

Official resources:

- Download metadata: <https://www.microsoft.com/en-us/download/details.aspx?id=100121>
- Paper: <https://www.microsoft.com/applied-sciences/uploads/publications/3/ms-asl.pdf>

The paper explicitly states that the dataset is not fully clean and that only about one quarter was manually label-verified. Source URLs may also disappear. Use available, audited clips for external validation or additional diversity, not as the sole source of truth.

### 4. WLASL100 — external benchmark only

**Recommendation:** Optional benchmark and vocabulary reference.

- Ready-defined 100-sign subset
- Signer IDs, source IDs, dialect/variation IDs, frame bounds, and official splits
- In-the-wild sources from educational websites and YouTube
- Download availability depends on external URLs
- Each class is represented by only a small number of signers compared with PopSign or ASL Citizen
- Computational-use agreement; commercial use is not allowed

Official repository and terms: <https://github.com/dxli94/WLASL>

WLASL is not the right foundation for the requested five takes per person, but it is valuable for measuring how the system behaves on unrelated Internet footage.

## Completed metadata coverage scan

The metadata-only scan was completed on 2026-08-09 using the official ASL Citizen split CSVs and PopSign public game-preview metadata. No dataset videos were downloaded.

- PopSign game metadata: 250 signs, 31 train signers, 8 validation signers, and 8 test signers
- ASL Citizen metadata: 83,399 rows, 35 train signers, 6 validation signers, and 11 test signers
- Exact formatting-normalized PopSign/ASL Citizen intersection: 208 signs
- Signs meeting the combined 20 train / 5 validation / 5 test threshold: 207
- Only `SHOWER` failed the threshold in this metadata pass
- Provisional candidate list: 100 signs
- Candidate signs already present in the current v16 vocabulary: 57
- Candidates flagged because ASL Citizen contains multiple raw labels or ASL-LEX codes: 19

Generated resources:

- Reproducible scanner: `scripts/build_ios100_dataset_coverage.py`
- Human-readable result: `artifacts/reports/IOS100_DATASET_COVERAGE_REPORT.md`
- Complete coverage table: `artifacts/reports/ios100_dataset_coverage.csv`
- Candidate details: `artifacts/reports/ios100_candidate_100.json`

The generated 100 is a coverage-ranked candidate list, not the final product vocabulary. The ranking first preserves signs already used by v16, then uses signer coverage. This means the lower portion contains well-covered but potentially low-priority child vocabulary such as animal or household-object signs. Product usefulness, one/two-handed form, and ASL lexical equivalence must be reviewed before downloading videos.

PopSign and ASL Citizen use independent anonymous participant namespaces. The report sums their signer counts as a working estimate; cross-dataset identity overlap cannot be ruled out from public metadata. Most selected signs independently have strong PopSign coverage, reducing dependence on that assumption.

## Utility-focused 100-sign proposal

A second pass replaced the coverage-only candidate ranking with an accuracy-first conversational vocabulary. The proposal contains pronouns, question words, social and safety terms, needs, common actions, emotions/states, time/place words, family terms, and language-related terms.

Generated resources:

- Editable proposal: `active/v16/ios100_vocabulary_proposal.json`
- Reproducible report builder: `scripts/build_ios100_vocabulary_report.py`
- Human-readable proposal and deficit report: `artifacts/reports/IOS100_VOCABULARY_PROPOSAL.md`
- Per-sign machine-readable coverage: `artifacts/reports/ios100_vocabulary_proposal.csv`

Current metadata result:

- 100 unique canonical signs
- 91 reuse an exact or explicitly declared current v16 class
- 98 have ASL Citizen coverage
- 44 have PopSign `game` coverage and therefore direct portrait-phone data
- 45 currently meet the working 20 train / 5 validation / 5 test signer estimate
- HELP, LOVE, and COME each need one additional training signer
- Most of the remaining gap can be filled by recording the same two additional validation signers across all 100 signs
- No proposed sign currently needs an additional test signer according to public metadata

The 100 signs are:

| Group | Signs |
| --- | --- |
| People/reference | I, YOU, WE, THEY, HE, MY, YOUR, OUR |
| Questions | WHAT, WHERE, WHEN, WHO, WHY, HOW |
| Social/safety | HELLO, GOODBYE, PLEASE, THANKYOU, SORRY, YES, NO, MAYBE, HELP |
| Needs/thought | WANT, NEED, LIKE, LOVE, KNOW, UNDERSTAND, THINK, FEEL, HAVE |
| Actions/communication | GIVE, TAKE, COME, GO, STOP, WAIT, TRY, USE, MAKE, FIND, LOOK, SEE, HEAR, LISTEN, TALK, SAY, TELL, ASK, ANSWER, LEARN, WORK, READ, WRITE, DRINK, SLEEP |
| Descriptions/states | GOOD, BAD, HAPPY, SAD, ANGRY, EXCITED, TIRED, SICK, HUNGRY, HOT, COLD, BIG, SMALL, MORE, LESS, SAME, DIFFERENT, EASY, IMPORTANT, READY |
| Time/places | NOW, TOMORROW, YESTERDAY, MORNING, NIGHT, TIME, DAY, WEEK, YEAR, HOME, SCHOOL, HOSPITAL, DOCTOR |
| People/family | FAMILY, FRIEND, MOTHER, FATHER, CHILD, MAN, WOMAN |
| Language | NAME, SIGN, LANGUAGE |

This is still a proposal, not an approved label ontology. An ASL-fluent or Deaf reviewer must approve each cross-dataset mapping and lexical variant before training. In particular, isolated deictic pronouns such as HE can depend on discourse location, and pairs such as LOOK/SEE, TALK/SAY/TELL, GOOD/THANKYOU, and WANT/LIKE require sign-level review.

The signer counts do not imply five takes per signer. PopSign usually supplies many takes per signer for the 44 overlapping signs, while ASL Citizen generally contributes broad signer coverage with far fewer takes per signer/gloss. For the other 56 signs, targeted portrait iPhone collection is still needed if five takes per person is a hard requirement. Do not discard a clean one-take sample merely because it lacks repeats: unseen-signer diversity is more valuable than many near-duplicate takes from the same seven people.

## Proposed dataset composition

The preferred accuracy-first corpus is not a blind concatenation of every available clip.

1. Choose 100 useful signs appearing in both PopSign and ASL Citizen where possible.
2. Confirm that each chosen label represents the same ASL sign variant across sources.
3. Prefer PopSign `game` videos for portrait smartphone coverage.
4. Use ASL Citizen to add more people, backgrounds, two-handed signing, and webcam framing.
5. Keep dataset-source metadata so accuracy can be reported independently for PopSign, ASL Citizen, current local data, and newly collected iPhone data.
6. Deduplicate by signer, original source, and near-identical landmark sequence.
7. Cap clips per signer per class so prolific contributors cannot dominate training.
8. Preserve official test identities. Do not move official test signers into training merely to increase counts.
9. Create a final locked iPhone test set from at least five additional people if possible. Public datasets alone do not prove performance with the application camera pipeline.

Before freezing the proposal for training, use the generated coverage table containing:

```text
gloss
lexical/variant identifier
PopSign train/val/test signer count
ASL Citizen train/val/test signer count
one-handed or two-handed
signing location
Apple Vision hand-detection rate
portrait/landscape counts
manual audit status
```

Do not mark a proposed sign training-ready until its train, validation, and test coverage meets the agreed threshold. The current rule is at least 20 train signers, 5 validation signers, and 5 test signers across the combined corpus; the deficit report identifies what still must be collected.

## Portrait/landscape failure: likely root cause

The current extractor uses Apple Vision coordinates normalized independently by image width and height. It then computes palm lengths, Euclidean distances, centering, and perspective features directly in that normalized coordinate system.

That geometry is anisotropic:

- In landscape, one normalized X unit represents many more pixels than one normalized Y unit.
- In portrait, the relationship reverses.
- The same physical handshape therefore has different angles, distances, velocities, and palm-scale behavior after the device rotates.

This is a representation bug/domain mismatch, not merely a shortage of portrait augmentation.

### Required canonicalization

Before any distance, angle, palm-scale, velocity, or Z calculation:

```text
x_pixel = x_normalized * image_width
y_pixel = y_normalized * image_height

x_canonical = (x_pixel - image_width / 2) / max(image_width, image_height)
y_canonical = (y_pixel - image_height / 2) / max(image_width, image_height)
```

Then perform wrist/body centering and isotropic palm/body scaling in canonical coordinates. An equivalent aspect-ratio-preserving square letterbox is acceptable; stretching the frame to a square is not.

The native iOS pipeline must also:

- Pass the correct `CGImagePropertyOrientation` to Vision.
- Account for front-camera mirroring exactly once.
- Extract from the camera pixel buffer, not coordinates transformed for the aspect-fill preview.
- Record width, height, camera orientation, and mirror state with every extracted sample.
- Apply the same canonicalization in offline training extraction and live inference.

After this change, all raw videos should be re-extracted. Existing `.npy` files cannot be corrected exactly unless their original width, height, orientation, and crop metadata are known.

Augmentation should remain after the geometry fix:

- Portrait and landscape aspect ratios
- Near/far framing
- Small camera roll
- Translation and crop jitter
- Brightness, contrast, and motion blur at the RGB level when raw-video training permits it
- Landmark dropout and short detection gaps

Augmentation should simulate remaining camera variation; it should not compensate for a mathematically inconsistent coordinate system.

## Implemented v17 orientation-safe extractor

The production path for all new data is now `active/v17/`, not the earlier v16 opt-in
experiment. v17 implements the required isotropic geometry plus the rest of the input
contract that the v16 patch did not solve:

- video rotation metadata is honored by default, with explicit rotation override;
- mirrored source pixels are unmirrored exactly once before Vision;
- portrait and landscape frames keep their aspect ratio and are never stretched;
- high-resolution PopSign frames are capped to a 1280-pixel long side;
- Apple hand, body, and face requests share one handler on frames where they are due;
- Apple chirality constants are mapped correctly;
- confidence and presence are tracked per joint;
- all 15 face slots contain defined facial landmarks instead of placeholder zeros;
- interpolation fills only short gaps bounded by real observations and never
  extrapolates a hand track;
- missing spatial, depth, and confidence values remain exactly zero;
- every `[32, 61, 5]` archive embeds its schema and fingerprint.

The v17 feature channels are body-relative X/Y, relative log-scale depth proxy, binary
presence, and confidence. This is intentionally incompatible with the existing v16
checkpoint; a new model must be trained from scratch or explicitly adapted.

The full contract and commands are in `active/v17/README.md`. Ten extractor tests,
including real Apple Vision rotate/mirror equivalence, pass. A separate 72-video archive
audit passed all schema, finite-value, binary-mask, and exact-missing-zero invariants;
see `artifacts/reports/V17_EXTRACTOR_AUDIT.md`.

### Superseded v16 geometry experiment

`active/v16/extract_v16.py` now has an opt-in `--aspect_correct` mode. It converts Apple Vision coordinates into isotropic image units before palm distance, shoulder distance, perspective-Z, normalization, and palm-scale calculations.

Example for a new, separately named extraction dataset:

```bash
python active/v16/extract_v16.py data/local/ios100_raw \
  --output data/local/ASL_landmarks_v16_aspect_correct \
  --aspect_correct \
  --resume
```

The option defaults to off so the current 96% checkpoint and existing landmark files retain their legacy representation. Do not mix legacy and aspect-correct `.npy` files in one training run. A model trained with the new representation must use the identical transformation in native iOS inference and should record an extractor-schema identifier in its deployment manifest.

The coordinate-level regression test is `test/test_aspect_correct_coordinates.py`. It verifies that an equivalent synthetic skeleton encoded in portrait and landscape frames becomes equal after correction, that legacy normalization remains orientation-dependent, that square inputs are unchanged, and that invalid image dimensions are rejected.

This test validated only the coordinate mathematics and remains for compatibility. v17
supersedes it. v17 has already verified identical Apple Vision features when the same
real source frames are physically rotated or mirrored and then correctly canonicalized.
The remaining orientation evidence gate is paired, independently recorded portrait and
landscape iPhone footage plus classifier accuracy after retraining.

## Measured external-video audit

A reproducible selective downloader read individual ASL Citizen members using HTTP byte ranges, avoiding the complete approximately 46 GB archive. The initial audit downloaded only 38 MB:

- 72 videos
- 12 proposed signs: HELLO, THANKYOU, GOODBYE, YOU, WHAT, HELP, LOVE, COME, GOOD, SCHOOL, HOSPITAL, and DRINK
- Two public participant IDs per sign from each official train, validation, and test split
- 29 unique public participant IDs across the selected files
- Multiple raw/ASL-LEX variants intentionally represented for WHAT, HOSPITAL, and DRINK

Measured media properties:

- All 72 videos decoded completely
- All 72 were 640x480 landscape videos
- Durations ranged from 1.03 to 7.17 seconds, with a 2.45-second median
- Frame counts ranged from 23 to 215, with a 70-frame median

This confirms that ASL Citizen is valuable for signer diversity but cannot validate portrait-phone behavior. A separate portrait source or new iPhone collection remains mandatory.

Using PyObjC 11.1 in a project-local environment, v17 produced valid `[32, 61, 5]`
tensors for all 72 videos with no extraction or no-hand failures. Median hand-node
presence was 0.4277 after activity trimming. Median detected hand-frame coverage rose
from 0.4538 before trimming to 0.8806 afterward. Shoulder normalization was usable for
64 videos, with palm-length fallback for 8.

The same videos were also extracted with the legacy coordinate schema and evaluated using the current 310-class d=384 checkpoint:

- External top-1 accuracy: **40.28%**
- External top-5 accuracy: **55.56%**
- Official Citizen train-ID sample: 45.8% top-1
- Official Citizen validation-ID sample: 45.8% top-1
- Official Citizen test-ID sample: 29.2% top-1
- GOOD: 100% top-1
- HELP: 83.3% top-1
- COME, GOODBYE, and THANKYOU: 0% top-1
- Five of six THANKYOU samples were predicted as GOOD

This is a small diagnostic, not a publishable benchmark: it covers only 12 signs, has not undergone ASL variant review, and evaluates a 310-class model rather than the proposed 100-class model. Nevertheless, the drop from the internal 96% score to 40.28% on public external signers confirms that retraining for signer/domain generalization is necessary before compression or deployment claims.

Audit resources:

- Selective downloader: `scripts/download_ios100_audit_subset.py`
- Media/extractor audit: `scripts/audit_ios100_video_sample.py`
- Current-checkpoint diagnostic: `scripts/evaluate_ios100_audit_stage1.py`
- Video provenance: `data/local/ios100_audit/asl_citizen/provenance.csv`
- Media and extraction report: `artifacts/reports/IOS100_VIDEO_AUDIT.md`
- External Stage 1 report: `artifacts/reports/IOS100_STAGE1_EXTERNAL_AUDIT.md`

## Accuracy-first model strategy

Do not distill or shrink the network yet.

1. Retain the current d=384, depth=4 Stage 1 architecture as the baseline.
2. Replace the classifier head with a 100-class head.
3. Compare full fine-tuning from the current checkpoint against training from scratch.
4. Select using unseen validation signers, never the random file split.
5. Keep the input clip length at 32 until an ablation shows another value improves unseen-signer accuracy.
6. Add an UNKNOWN/non-sign class using setup motion, no-sign hand motion, partial signs, and out-of-vocabulary signs.
7. Calibrate confidence thresholds on unseen validation signers.
8. Delay FP16-versus-INT8 and smaller-dimension experiments until the accuracy dataset and portrait pipeline are stable.

The existing FP16 Core ML conversion already preserves predictions on stored landmarks. Therefore, the immediate uncertainty is not whether the Squeezeformer can be converted. It is whether the landmark representation and training population generalize to real iPhone users.

## Suggested execution order

1. Obtain ASL-fluent/Deaf review of the proposed ontology, aliases, and variant groups.
2. Record a paired portrait/landscape iPhone test using the same signer and signs; the external landscape audit is complete, but it cannot provide paired orientation evidence.
3. Manually review the downloaded 72-video stratified sample for label and lexical consistency.
4. Freeze the final 100 and a versioned source-to-canonical label map.
5. Record at least two additional validation signers across all 100 signs and one additional training signer for HELP, LOVE, and COME; prefer five takes per sign.
6. Download only the approved ASL Citizen clips and PopSign sign archives needed by the frozen vocabulary.
7. Extract every chosen video with v17 into a schema-validated landmark directory.
8. Manually audit a statistically useful sample for every selected sign, preferably with an ASL-fluent or Deaf reviewer.
9. Train d=256/depth=4 first and increase capacity only if a controlled signer-disjoint
   validation ablation improves accuracy.
10. Evaluate on locked public-dataset test signers and a separate five-signer portrait-iPhone test set.
11. Integrate the best checkpoint into the native iOS prototype and measure full-pipeline accuracy and sustained latency.
12. Only then measure FP16, INT8, and smaller-model accuracy/latency tradeoffs.

## Definition of the next success milestone

The next milestone is not another random-split result above 96%.

It is:

> A selected 100-sign model, trained on 32 Citizen people, selected on five different
> validation people, and evaluated on 11 completely unseen Citizen people plus an
> independent portrait-iPhone set, with comparable accuracy on correctly canonicalized
> portrait and landscape recordings.

Only after that milestone should the project trade accuracy for model size or latency.
