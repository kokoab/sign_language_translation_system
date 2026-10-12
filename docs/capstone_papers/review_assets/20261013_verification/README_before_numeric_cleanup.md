# ATLAS manuscript — exact wording and figure review

Updated: **2026-10-13**. **PROPOSALS ONLY — not applied to the manuscript.**

This revision checks the October 12 deployed recognition configuration against current source/model hashes, fresh validation inference and saved native/phone records. The manuscript “Before” passages remain the exact **October 8 snapshot**, not a claim that the live Google Doc was reread today. [Working manuscript tab](https://docs.google.com/document/d/1sC2JZ23mSpnLWeEEFDm5ce2x_As0XTS7ayMGvk5uuD4/edit?tab=t.pzqwjycim1ix).

Check each proposal independently and write comments under **Your note** using its ID. Unchecked boxes are your review decisions, not failed verification checks. [Previous checklist, preserved in full](review_assets/20261013_verification/README_PLATFORM_NEUTRAL_REVIEW_20261008_archive.md). Slides and manuscript remain unchanged.

## Author decision — training-experiment WER (October 13)

- [x] Use **31.42% development-set gloss-sequence WER from the interval-recognizer training experiment** in the paper, rather than the later 12.83% decoder-development result.

**Agreed wording:**

> During interval-recognizer adaptation, the selected checkpoint recorded a development-set gloss-sequence word error rate of 31.42% under the decoder configuration used for that training experiment.

**Scope:** This is a development result recorded during model selection, not error on the training examples and not the current deployed configuration’s WER. The 95.50% isolated-sign result is also validation accuracy, not training-set accuracy; it was freshly reproduced for the current primary head. The two results concern different checkpoints/evaluation procedures and must not be presented as one matched configuration. Device measurements remain separate. The 12.83% result stays in the internal evidence record; no result files are overwritten and no old-versus-new comparison is proposed for the paper.

**Your note:**

## Verified measurements and selected reporting scope

| Proposed paper measure | Verified value | What it measures |
|---|---:|---|
| Primary recognition head validation accuracy | **95.50%** | Isolated-sign classification; freshly reproduced in PyTorch and the current FP16 export |
| Training-experiment development WER — selected for paper | **31.42%** | Selected interval-recognizer checkpoint, under that experiment’s decoder configuration |
| iPhone 13 median processing time | **31.14 ms per processed frame** | Recalculated from the two candidate phone replay runs; includes frame preparation and recognition |
| Current FP16 recognizer package | **26.56 MB** | Recounted complete package bytes; matches the bundled source package |

**Agreed interpretation:** retain 95.50% as the primary recognition component’s validation accuracy and report 31.42% separately as the interval-recognizer training experiment’s development WER. These do not describe one matched checkpoint/configuration. Do not call either end-to-end communication accuracy. The application retains the primary head and applies an auxiliary trained head selectively for transition handling; measuring that auxiliary head alone does not measure the full runtime policy.

The earlier combined 95.53% / 31.42% / 23.69 ms summary is not reinstated. Only the scoped 31.42% training-experiment result is selected here; current device measurements remain separate. No old-versus-new comparison is proposed for the paper.

## What validation, evaluation and test mean here

**Evaluation set** is a general term, not a synonym for **test set**. Validation/development data support model or setting selection. An independent test set is reserved from those decisions. The figures above are validation/development and engineering measurements, not new independent test results.

Paper prose uses **validation dataset**, **validation accuracy**, and **development-set WER**, without “pooled,” validation-set sizes or lists of dataset names. Accuracy is total correct classifications divided by total evaluated examples. Audit counts appear only in the review evidence below.

**Preserved by instruction:** existing signer-disjoint discussion, slides, participant plans, schedules, unrelated literature and separately evaluated English-generation results. No new claim that every measurement is signer-disjoint is proposed. General prose remains platform-neutral; measured hardware is attributed explicitly.

## Exact prose and table changes

### R-01 — Define each measurement

- [ ] Approve R-01

**Before — manuscript snapshot (October 8):**

> Each comparison retains its own task and measurement conditions. Recognition percentages below describe the corresponding validation studies, while the streaming comparison uses the same recorded sequences for both configurations. Base-classifier CPU timings use prepared inputs; the mobile-device measurements include the stated processing work on the device. Detailed hardware, recording membership and source records are retained in the measurement notes. This separation allows each result to answer the engineering question for which it was measured.

**After — proposed wording:**

> Recognition performance is assessed through isolated-sign classification and connected-sign sequence recognition. Validation accuracy is calculated by dividing the number of correctly classified isolated-sign examples by the total number of examples in the validation dataset. Gloss-sequence word error rate is reported separately for the selected checkpoint in the interval-recognizer adaptation experiment, using that experiment’s development set and decoder configuration. Device-processing time covers the measured frame preparation and recognition work.

**Review note:** Section 4.4. Avoid a single “system accuracy” label that merges classification, segmentation, English generation and live communication.



**Your note:**

### R-02 — Report the current primary recognition head

- [ ] Approve R-02

**Before — manuscript snapshot (October 8):**

> The Apple Vision and MediaPipe pipelines were compared across three recognition stages using the same validation set and vocabulary. The landmark-only base recognizer uses structured movement information. The combined-input recognizer adds hand-image features, while the interval-adapted recognizer is prepared for candidate sign intervals during streaming. Table 11 and Figure 11 compare isolated-sign validation accuracy at these stages.
> 
> The Apple Vision pipeline recorded higher validation Top-1 accuracy at each of the three stages in this comparison. Combining landmark and hand-image features produced 96.30% accuracy for Apple Vision inputs and 94.44% for MediaPipe inputs. After adaptation to candidate intervals, the corresponding isolated-sign validation values were 95.24% and 94.44%. These results describe the evaluated configurations rather than every possible implementation of either extractor.
> 
> Processing speed was measured on the same Apple M4 development computer using identical recordings from the same validation set. Each clip's measured stages were combined: landmark extraction and preparation, applicable hand-image feature extraction, and recognition. Initial video decoding, model loading, English generation, and speech output were excluded. The Apple Vision pipeline recorded median times of 356.1, 668.6, and 668.7 ms per clip for the landmark-only, combined-input, and interval-adapted stages, respectively. The corresponding MediaPipe results were 546.7, 809.3, and 808.6 ms. Figure 11 presents these development-computer measurements alongside recognition accuracy; device processing is evaluated separately.

**After — proposed wording:**

> The primary recognition head achieved 95.50% validation accuracy for isolated-sign classification within the selected 100-sign vocabulary. Table 11 presents this result. During connected signing, temporal segmentation and an auxiliary transition-recognition head support the selection of sign intervals and the order of accepted glosses.

**Review note:** Fresh inference reproduces this result, including the current FP16 export. It is the original head retained inside the deployed dual-output model, not the earlier 95.53% checkpoint and not the adapted head used alone. This is a proposed wording decision: if you require an accuracy specifically for the complete conditional runtime policy, leave this item unchecked until that distinct evaluation is defined and run.

**Table 11 — before (October 8 transcription):**

| Recognition stage | Apple Vision Top-1 | MediaPipe Top-1 |
|---|---:|---:|
| Landmark-only | 95.77% | 91.80% |
| Combined inputs | 96.30% | 94.44% |
| Interval-adapted | 95.24% | 94.44% |

**After: Table 11. Isolated-Sign Validation Accuracy**

| Recognition component | Validation Top-1 accuracy |
|---|---:|
| ATLAS primary recognition head | **95.50%** |

Remove the old extractor-stage comparison and its Mac timings from this passage. Do not reuse those timings for the current configuration.

**Your note:**

### R-03 — Explain the current recognition method

- [ ] Approve R-03

**Before — manuscript snapshot (October 8):**

> Landmarks provided the stronger individual input, reaching 95.50% Top-1 accuracy compared with 80.69% for hand-image features. Combining the inputs reached 96.30%, an increase of 0.80 percentage points over landmarks alone. The improvement supports complementary use of motion and appearance within the recognition component without implying that either input contributes equally.

**After — proposed wording:**

> The recognizer combines landmark motion with hand-image features. Adaptation to local signing inputs, phrase segments and candidate sign intervals prepares the model for the inputs encountered during connected signing. Training also uses partial-sign examples to reduce confident recognition of incomplete gestures. The deployed recognizer retains its primary recognition head and uses an auxiliary trained head to provide additional evidence for selected sign transitions. The segmental decoder combines this evidence with observed movement and boundary estimates before accepting glosses.

**Review note:** Move to methodology, Section 4.3.3. Remove the stale input-comparison table. The old fixed-weight fusion description is insufficient for the current adapted head. Describe the scoped transition correction, without implying universal correction of connected signs. Boundary-model distillation remains a separate method; do not attribute all improvements to distillation. Keep low-level settings, paths and experiment history out of the paper.



**Your note:**

### R-04 — Consolidate architecture evidence

- [ ] Approve R-04

**Before — manuscript snapshot (October 8):**

> The architecture comparison considered how landmark information should be organized before temporal classification. Table 13 reports earlier design-screening checkpoints; Table 14 reports a separate matched training experiment. Their Squeezeformer results therefore refer to different trained models. Both tables concern landmark-based classifiers and exclude the hand-image encoder and combined recognition model. M denotes millions of parameters, and CPU median denotes classifier inference on prepared inputs on the development Mac. A flat model processes the input together, part-wise processing first represents anatomical regions, and the graph-part alternative represents connections between landmarks.
> 
> Part-wise and global Squeezeformer processing achieved the highest Top-1 accuracy in this comparison, at 96.83%. Relative to the flat Squeezeformer, the gain was 1.06 percentage points with an additional 1.60 milliseconds of median classifier computation. The flat model was faster and achieved the highest Top-5 value, while the selected architecture prioritized the correctness of the first prediction presented by recognition.
> 
> In the matched experiment, Squeezeformer achieved 96.30% validation Top-1 accuracy versus 95.50% for the flat Transformer, a difference of 0.79 percentage points. Transformer was faster on the development Mac CPU: 2.88 versus 6.15 milliseconds. However, a separate iPhone 13 comparison of the same two classifiers in FP16 measured 0.95 milliseconds for Transformer and 0.99 milliseconds for Squeezeformer. Both retained their original validation predictions. The 0.04-millisecond difference was small relative to variation across four counterbalanced runs. Thus, Squeezeformer provided higher measured accuracy with similar FP16 classifier speed on the evaluated phone. These single-seed accuracy results do not establish general architectural superiority. Classifier timings exclude landmark extraction, hand-image encoding and streaming processing; they are separate from the system's historical 28-millisecond measurement.

**After — proposed wording:**

> The base landmark classifiers were evaluated using the same isolated-sign validation inputs. The selected part-wise and global Squeezeformer achieved the highest Top-1 accuracy among the three configurations in Table 13. These measurements concern the landmark classifier before local, multimodal and interval adaptation. They are separate from the final recognizer’s validation accuracy reported in Table 11. The results describe the evaluated checkpoints and do not establish general superiority of one model family.

**Note:** Replace Tables 13 and 14 with one compact base-classifier table below. Remove the intervening old model-family comparison paragraph, old CPU times and historical 28 ms reference. Selected historical checkpoints share an evaluation, not a newly matched from-scratch training experiment. This item explicitly proposes removing the BiLSTM, temporal CNN, graph and wider-model screening rows from the main paper.

**October 13 verification:** The recorded canonical comparison still supports these three selected component checkpoints. They are evaluated on the same component-validation inputs, not the complete final-recognizer validation dataset. No new architecture training or speed comparison was performed today. Keep this distinction; the repeated 95.50% for Transformer and the current primary head is coincidental.

**Your note:**

**R-04 table replacement — exact content**

Before: **Table 13. Earlier Landmark Architecture Screening** (current cell contents).

| Architecture | Parameters | Top-1 | Top-5 | CPU median |
|---|---:|---:|---:|---:|
| Graph-part replacement | 6.48 M | 78.31% | 95.77% | 11.42 ms |
| Wider flat Squeezeformer | 14.34 M | 95.24% | 99.74% | 7.35 ms |
| Flat Squeezeformer | 6.47 M | 95.77% | 100.00% | 4.90 ms |
| Part-wise + global Squeezeformer | 6.79 M | 96.83% | 99.21% | 6.50 ms |

Before: **Table 14. Matched Recognition Family Comparison** (current cell contents).

| Model family | Parameters | Validation top-1 | CPU median |
|---|---:|---:|---:|
| BiLSTM | 6.10 M | 89.15% | 3.39 ms |
| Temporal CNN | 5.90 M | 92.59% | 4.34 ms |
| Flat Transformer | 6.72 M | 95.50% | 2.88 ms |
| ATLAS (part-wise + global Squeezeformer) | 6.79 M | 96.30% | 6.15 ms |

After: **Table 13. Base Landmark Classifier Comparison**

| Classifier | Parameters | Validation Top-1 accuracy |
|---|---:|---:|
| Flat Transformer | 6.72 M | 95.50% |
| Flat Squeezeformer | 6.47 M | 95.77% |
| Part-wise and global Squeezeformer | 6.79 M | 96.83% |

“M” denotes millions of parameters. These figures remain component measurements; none is presented as final ATLAS accuracy. No replacement comparison chart is proposed.

### R-05 — Report the training experiment’s development WER

- [ ] Approve R-05

**Before — manuscript snapshot (October 8):**

> Individual-sign accuracy does not fully describe connected signing, where the system must also decide when a sign begins, ends and becomes stable enough to accept. The streaming comparison therefore measures sequence errors across two complete configurations on the same recordings. Boundary-guided interval classification passes predicted intervals to recognition; ATLAS streaming recognition combines the distilled boundary model, interval-adapted recognition and sequence selection.
> 
> Word error rate decreased from 39.78% to 9.68%, a difference of 30.10 percentage points. This reflects the combined configuration, rather than an isolated effect of boundary distillation. The recordings include familiar-signer and unseen-signer sequences, providing a development comparison of how the system handles connected input. The result supports coordinating interval detection, recognition and gloss commitment instead of judging the streaming workflow solely from individual-sign accuracy.

**After — proposed wording:**

> During interval-recognizer adaptation, the selected checkpoint recorded a development-set gloss-sequence word error rate of 31.42% under the decoder configuration used for that training experiment. Word error rate accounts for substitutions, deletions and insertions relative to the reference gloss sequence. This result describes the selected checkpoint’s development performance during adaptation; mobile processing is evaluated separately.

**Review note:** Reporting basis approved by the author on October 13; the checkbox above remains for review of the complete replacement passage/table. The selected epoch-four result is 71 edits / 226 reference glosses, rounded to 31.42%. It is neither training-set error nor the current deployed recognizer/decoder’s WER. Keep the later 12.83% in internal evidence only. Do not imply that 31.42% and current 95.50% come from one matched checkpoint. Remove the old 39.78%/9.68% comparison from the proposed paper passage.

**Table — before:** boundary-guided interval classification 39.78%; ATLAS streaming recognition 9.68%.

**After: Table 15. Development Performance During Interval-Recognizer Adaptation**

| Evaluated checkpoint | Development-set gloss-sequence WER |
|---|---:|
| Selected interval-recognizer checkpoint, training-experiment decoder | **31.42%** |

**Your note:**

### R-06 — Describe prototype deployment

- [ ] Approve R-06

**Before — manuscript snapshot (October 8):**

> The current ATLAS implementation should not, however, be described as a completed production mobile deployment. The evaluated implementation is Mac-based and has been tested using signer-disjoint recorded clips and controlled live testing. The current evaluation does not establish production-level Android or iOS performance, power consumption, thermal behavior, or unrestricted conversational operation.

**After — proposed wording:**

> ATLAS has been implemented as a mobile application and evaluated using recorded inputs on a development computer and an iPhone 13. These measurements establish prototype recognition and device-processing performance within the study’s defined conditions. They do not establish unrestricted conversational operation or performance across all mobile devices.

**Review note:** The normal signed release was installed and launched October 12. Today’s source and package hashes match that deployment record; this is not a new device installation or live device inventory check. Preserve neutral terminology elsewhere.



**Your note:**

### R-07 — State the actual FP16 conversion result

- [ ] Approve R-07

**Before — manuscript snapshot (October 8):**

> The FP32 and FP16 recognizer exports were evaluated using the same validation inputs and reference labels. Landmark and cached hand-image inputs were held constant to examine the recognizer's predictions separately from changes in visual extraction. The comparison found similar recognition accuracy, while the device evaluation examined the processing cost of each precision configuration.
> 
> The FP32 recognizer recorded 95.24% validation Top-1 accuracy, compared with 94.97% for FP16. Top-5 accuracy was 98.94% for both. These values describe the interval-adapted recognizer, whereas the model-family comparison reports separately trained configurations.

**After — proposed wording:**

> The FP16 recognizer export was evaluated using the same prepared landmark and hand-image inputs as the trained model. The primary recognition head retained 95.50% validation accuracy after conversion, although one predicted label changed. Equal aggregate accuracy therefore did not imply identical predictions for every input.

**Review note:** Fresh complete admitted-validation inference was run for both heads. Primary head: 3,779 correct in both implementations, with 3,956/3,957 top-1 agreement. Auxiliary head: 3,784 correct in PyTorch versus 3,782 after FP16 conversion (95.63% versus 95.58%), with 3,955/3,957 agreement. Do not claim zero conversion impact across the whole recognizer. A matched all-FP32 mobile pipeline was not built or timed in this audit.

**Before — Table 18 recognition rows:**

| Measure | FP32 | FP16 |
|---|---:|---:|
| Validation Top-1 accuracy | 95.24% | 94.97% |
| Validation Top-5 accuracy | 98.94% | 98.94% |

**After — proposed compact conversion table:**

| Component | Trained model (FP32) | Deployment export (FP16) |
|---|---:|---:|
| Primary recognition head validation accuracy | **95.50%** | **95.50%** |

These columns compare the trained PyTorch model with its export, **not two complete mobile precision configurations**. Omit the old Top-5 row. The auxiliary-head conversion difference remains disclosed here for review; if discussing conversion of the whole dual-output model in the paper, include: “The auxiliary transition head recorded 95.63% validation accuracy before conversion and 95.58% after conversion.” Do not present that head as the main classifier or average the two heads.

**Your note:**

### R-08 — Use the current package size

- [ ] Approve R-08

**Before — manuscript snapshot (October 8):**

> Model sizes compare saved FP32 inference weights with complete FP16 deployment packages, in decimal megabytes. They describe model storage rather than working memory or the size of the complete application.

**After — proposed wording:**

> The FP16 recognizer deployment package occupies 26.56 MB. This value measures the complete recognizer package in decimal megabytes, rather than working memory or the size of the complete application.

**Review note:** Recounted 26,557,148 bytes, excluding filesystem AppleDouble metadata. Package file hashes match the mobile source bundle. Old 25.51 MB and 50.28 MB packages belong to an earlier model. There is no verified matching FP32 package size for this dual-output recognizer in this review.

**Before:** recognizer saved model size 49.62 MB / 25.51 MB; boundary size 9.25 MB / 4.68 MB.

**After — current deployment entry:**

| Component | Precision | Deployment package size |
|---|---|---:|
| Recognizer | FP16 | **26.56 MB** |

Remove the boundary storage row from the recognizer table; retain the boundary component in methodology. A new matched FP32 storage column remains pending if you want to retain that comparison.

**Your note:**

### R-09 — Use the verified current phone measurement

- [ ] Approve R-09

**Before — manuscript snapshot (October 8):**

> On the iPhone 13, the current FP16 visual and recognition configuration recorded 24.1 ms per processed frame, compared with 112.6 ms for the corresponding FP32 configuration. Both used identical recorded inputs, and English generation remained FP32 and outside the timed work. The reported values are the medians of two run medians per configuration. Figure 13 compares these integrated configurations; the difference should not be attributed to numerical precision alone. Timing includes frame preparation and recognition, rather than camera capture or complete English and speech output.

**After — proposed wording:**

> On iPhone 13, the evaluated configuration recorded a median processing time of 31.14 ms per processed frame during two camera-paced replays of a development recording. Timing covered frame preparation, landmark and hand-image processing, and recognition. Camera capture, model loading, interface presentation, English generation and speech output were outside the timed work. This measurement describes frame computation rather than the delay before a gloss is displayed.

**Review note:** Recalculated across 1,041 frame measurements in the two candidate runs; this is not a median of two medians. The 95th percentile is 57.85 ms. English was resident, but its generation was not timed. The runs processed one known recording and do not establish sustained thermals or fresh-signer accuracy. No historical-versus-current timing comparison is proposed. The 31.14 ms includes more than recognizer inference, so do not label it “per-clip inference.”

**Optional table if needed; no new bar chart:**

| Device | Median processing time | Measurement scope |
|---|---:|---|
| iPhone 13 | **31.14 ms per processed frame** | Camera-paced recorded-input preparation and recognition |

Remove the old 24.1/112.6 ms comparison and the earlier proposed 23.69 ms. A current all-FP32 versus all-FP16 phone comparison remains unmeasured.

**Your note:**

### R-10 — Practical interpretation of validation accuracy

- [ ] Approve R-10

**Before — manuscript snapshot (October 8):**

> The study therefore evaluates the performance of the developed system only within the defined ASL vocabulary and testing conditions. Recognition of signs outside the selected vocabulary, other ASL expressions, full ASL grammar, Filipino Sign Language, or other sign languages is outside the scope of the present study.

**After — proposed wording:**

> The study therefore evaluates the performance of the developed system only within the defined ASL vocabulary and testing conditions. Recognition of signs outside the selected vocabulary, other ASL expressions, full ASL grammar, Filipino Sign Language, or other sign languages is outside the scope of the present study.
> 
> Isolated-sign validation accuracy measures classification of prepared sign inputs. During live use, the system must additionally locate sign boundaries and maintain a sequence of recognized signs. Consequently, isolated-sign validation accuracy does not represent end-to-end communication accuracy.

**Note:** Append this paragraph to Section 1.4 without rewriting the existing signer-disjoint discussion or participant-research provisions.

**Your note:**

### R-11 — Explain transition handling without low-level settings

- [ ] Approve R-11

**Before — proposed insertion:** The October 8 checklist had no explanation of the newly deployed auxiliary transition head and scoped boundary correction.

**After — proposed wording:**

> Closely connected signs can cause the movement of one sign to enter the recognition window of the next. ATLAS supplements its primary recognition output with an adapted transition head while retaining the primary model’s outputs. The decoder uses this evidence selectively with observed hand contact, motion and the following gesture to refine candidate intervals. This addresses the evaluated transition cases without applying the adapted head as a replacement for every recognition decision.

**Review note:** Optional explanatory paragraph after R-03 if that paragraph alone is too brief. Avoid duplicating the two. Current evidence supports the targeted MY transition repair; it does not establish that all coarticulation errors or HAVE omissions have been resolved. Keep training-arm counts, thresholds, thread settings and processor flags out of the manuscript.

**Your note:**

## Charts, figures and exact captions

All before images are the October 8 manuscript snapshots. Proposed assets are review-only.

### F-01 — Remove the outdated pipeline comparison (current Figure 11)

- [ ] Approve F-01

**Before — manuscript snapshot (October 8) image:**

![Current Figure 11: earlier stage accuracy and Mac timing](review_assets/20261008_proposals/current_figure11.png)

**After:** Remove the chart and its caption. Use the single-row primary recognition table in R-02. There is no replacement chart: its old Mac timings and extractor comparisons do not describe the newly selected recognizer.

**Your note:**

### F-02 — Remove the old phone precision chart (current Figure 13)

- [ ] Approve F-02

**Before — manuscript snapshot (October 8) image:**

![Current Figure 13: earlier all-visual precision comparison](review_assets/20261008_proposals/current_figure13.png)

**Before caption:** “Figure 13. On-Device Processing by Precision”

**After:** Remove the chart and caption. Replace the adjoining paragraph with R-09’s exact wording; report **31.14 ms per processed frame** once in the final device-performance results.

**Your note:**

### F-03 — Retain a clearly identified base-classifier training curve

- [ ] Approve F-03

**Before — Figure 14:**

![Existing base-landmark training figure](review_assets/20261008_proposals/current_figure14.png)

**Before caption:** “Figure 14. Landmark Recognition Training”

**Before discussion:**

> Figure 14 shows landmark-recognition training across the recorded epochs. The selected checkpoint at epoch 100 achieved 96.83% validation Top-1 accuracy. Selection follows validation behavior rather than assuming that the final recorded epoch must provide the best recognition.

**After — recommended chart decision:** Retain this base-classifier figure. The October 8 eight-epoch/31.42% recognizer chart is valid only as the training experiment now selected in R-05, not as the current deployed configuration’s history. Restoring that chart under an experiment-specific caption remains a separate figure decision. Do not splice histories from different training stages into one curve.

**After caption:**

> Figure 14. Base Landmark Classifier Training

**After discussion:**

> Figure 14 presents the training history of the selected part-wise and global landmark classifier. The selected checkpoint achieved 96.83% validation Top-1 accuracy in the base-classifier evaluation. This figure concerns the landmark component before subsequent adaptation and multimodal recognition; it does not represent the complete deployed recognizer’s training history.

**Review note:** Retaining a component training curve is clearer than displaying the auxiliary transition head’s near-perfect performance on a small, repeatedly replayed development recording. No new full-model training curve exists for a runtime that combines a retained head, an adapted head and decoder rules. This item changes the previous chart proposal and requires your review. The original history was checked: epoch 100 records 96.8254% Top-1. This is verified historical provenance, not a new training run.

**Your note:**

### F-04 — Remove the now-duplicate Figure 15

- [ ] Approve F-04

**Before — current manuscript image:**

![Current Figure 15: earlier interval-adaptation history](review_assets/20261008_proposals/current_figure15.png)

**Before caption:** “Figure 15. Recognition Adaptation to Sign Intervals”

**Before discussion:**

> Figure 15 follows the eight-epoch adaptation of recognition to candidate sign intervals, with epoch zero representing the starting recognizer. The sequence-error panel uses the decoder configuration designated for this adaptation study. Together, loss and sequence error show how interval training affects the inputs that streaming recognition must handle; its curve is interpreted within that training study rather than substituted for the complete-configuration comparison in Table 15.

**After:** Remove this old interval-adaptation chart and its caption/discussion from the main results. Keep F-03 as a clearly scoped component-training figure and R-05 as the single reported training-experiment development WER. Do not treat the old 27.43% curve minimum or the selected 31.42% training result as the current deployed configuration’s WER.

**Your note:**

### F-05 — Update HELLO HOW YOU from an actual new recognition trace

- [ ] Approve rerun and replacement F-05

**Before — existing timeline asset:**

![Existing HELLO HOW YOU timeline](review_assets/hello_how_you_timeline.svg)

**Before — exact current discussion:**

> The HELLO HOW YOU example in Figure 2 illustrates how the boundary and recognition components relate to a recorded sequence. The upper row shows selected intervals based on boundary estimates and decoder decisions, while the lower row associates those intervals with the recognized glosses. HELLO occupies approximately 0.67–1.17 seconds, HOW 1.40–1.67 seconds, and YOU 2.27–2.87 seconds in the saved recognition record. The video stills show one original frame near the midpoint of each interval, providing a visual reference for the signing input.

**After — proposed replacement preview:**

![Verified current-decoder HELLO HOW YOU development example](review_assets/20261013_verification/hello_timeline.png)

**After caption:**

> Figure 2. Selected Sign Intervals in a HELLO HOW YOU Development Replay

**After discussion:**

> Figure 2 illustrates the intervals selected by the recognizer and segmental decoder for a recorded HELLO HOW YOU development example. The saved output assigns HELLO to approximately 0.87–1.80 seconds, HOW to 2.07–3.00 seconds and YOU to 3.17–4.57 seconds. These intervals describe the selected sign segments rather than the time required to publish each gloss.

**Review note:** This proposal uses a **different, documented development example**, ID `022d8f1f67a525ad4ea7`, from the current saved native decoder replay. It is not a new replay of the original `ff187c3f` video. The existing original-video stills must not be placed under this new timeline. The preview is a clean timeline only; adding stills requires tracing and checking this example’s own source frames. The original clip has a canonical validation role, but its old chart came from `score_test_C.json`; no assumption about permission to reuse that evaluation pool was made and no protected test was rerun.

[Editable SVG](review_assets/20261013_verification/hello_timeline.svg) · [Saved case and output](review_assets/20261013_verification/hello_trace.json)

**Your note:**

### F-06 — Renumber and preserve unrelated material

- [ ] Approve F-06

**Before:** Existing figures/tables and their navigation entries still reference the proposed removals.

**After:** Renumber surviving tables and figures, update all in-text references and refresh the lists of figures/tables after approved changes. Figure 14/Figure 2 above are current identifiers, not a promise of final numbering. Keep the English-output table and training figure, boundary-model explanation, schedules, participant plans and other tabs unchanged.

**Your note:**

## Optional summary sentence

- [ ] Approve S-01

**Before — October 8 proposal:**

> The final ATLAS recognizer achieved 95.53% validation accuracy on the validation dataset and a development-set gloss-sequence word error rate of 31.42%. The evaluated FP16 configuration required a median processing time of 23.69 ms per frame on iPhone 13 for preparation and recognition.

**After:**

> The current primary recognition head achieved 95.50% isolated-sign validation accuracy. In the separate interval-recognizer adaptation experiment, the selected checkpoint recorded a development-set gloss-sequence word error rate of 31.42% under that experiment’s decoder configuration. Device evaluation of the current configuration recorded a median frame-processing time of 31.14 ms on iPhone 13 for preparation and recognition.

**Review note:** This sentence preserves the different measurement scopes. It does not call 95.50% the accuracy of complete live communication, claim a phone-measured WER, or imply that the training-experiment WER and current deployment describe one unchanged configuration. Use only if the component-level accuracy presentation in R-02 is accepted.

**Your note:**

## Verification record — evidence, not manuscript prose

Fresh checks on October 13:

- Current decoder, model loader, project configuration and test-source hashes match the October 12 deployment record. All candidate-manifest pinned file hashes checked today match. This confirms local source/artifact identity, not the phone’s present running state.
- Fresh original and adapted head inference used the admitted validation features with the established identity exclusions; no new split, training or protected-test inference.
- Fresh FP16 inference used the corresponding raw validation inputs in the same admitted order. The package bundled in mobile source matches the verified export byte-for-byte.
- The later decoder-development WER was recalculated from saved native hypotheses and references: 29/226 = 12.83%. This remains valid internal evidence, but the author selected the separate 31.42% training-experiment result for paper reporting. No newly generated hypotheses were substituted.
- Phone timing was recalculated from raw frame measurements; no new phone benchmark was run.
- The HELLO preview was generated from a saved current-decoder development record and visually checked.

| Head and implementation | Correct / evaluated | Validation accuracy | Top-1 agreement with PyTorch |
|---|---:|---:|---:|
| Primary head, PyTorch FP32 | 3,779 / 3,957 | 95.50% | Reference |
| Primary head, deployed FP16 export | 3,779 / 3,957 | 95.50% | 3,956 / 3,957 |
| Auxiliary adapted head, PyTorch FP32 | 3,784 / 3,957 | 95.63% | Reference |
| Auxiliary adapted head, deployed FP16 export | 3,782 / 3,957 | 95.58% | 3,955 / 3,957 |

These counts are internal review evidence, not proposed validation-set sizes for the manuscript. The two heads are not independent models whose accuracies should be averaged. Their conditional use within the decoder is evaluated separately through development sequence outputs.

**Remaining gaps — explicitly not verified by this update:**

- [ ] V-01 — If retained, build and evaluate a matching **current all-FP32 mobile configuration** for speed and package comparison. Today’s FP32 reference is PyTorch, not an all-FP32 phone deployment.
- [ ] V-02 — If an overall conditional-runtime isolated accuracy is required instead of R-02’s component result, define the exact input/segmentation protocol and evaluate it. Do not substitute the auxiliary-head percentage.
- [ ] V-03 — If video stills are wanted in F-05, identify and verify the replacement example’s own source frames before adding them.
- [ ] V-04 — Before applying the checklist, reread the live manuscript tab and reconcile any author edits since the October 8 snapshot.

No fresh independent signer accuracy, sustained thermal test, updated English-quality evaluation or complete end-to-end communication score was produced. Existing limitations are not erased by targeted development replay success.

## Evidence

- [Selected training experiment: recorded history](../../artifacts/reports/canonical_recognition_comparison_v17_20261007/downstream_recipe/chain_9683/span_recognizer/history.json)
- [Training-experiment claim verification](../../artifacts/reports/canonical_recognition_comparison_v17_20261007/PAPER_CLAIM_REVIEW.md)

- [October 13 verification: hashes, fresh head inference, conversion, WER and timing](review_assets/20261013_verification/verification.json)
- [Current deployed transition-repair report](../../artifacts/reports/my_transition_recovery_v17_20261011/REPORT.txt)
- [Deployment and cleanup-parity record](../../artifacts/reports/my_transition_recovery_v17_20261011/deployment/REPORT.txt)
- [Saved development predictions](../../artifacts/reports/my_transition_recovery_v17_20261011/merge_validation.json)
- [Development case identities and roles](../../artifacts/reports/my_transition_recovery_v17_20261011/joint_validation_cases.json)
- [Current head-training results](../../artifacts/reports/my_transition_recovery_v17_20261011/confidence_training_results.json)
- [Phone benchmark scope and measurements](../../artifacts/reports/my_transition_recovery_v17_20261011/phone_comparison_summary.json)
- [Base-classifier comparison record](../../artifacts/reports/canonical_recognition_comparison_v17_20261007/REPORT.md)

The review does not apply changes to the manuscript, slides or production application. It preserves prior evidence rather than overwriting historical results.
