# ATLAS manuscript — remaining proposals for review

Updated: 2026-10-07. **Implemented in the latest manuscript tab, “(Oct 6) Revision.” Wording, tables, diagrams, result charts, captions, and page references are updated.**

The author approved the remaining changes after the device benchmark. P-04 and P-06 are now approved with the current FP16 deployment retained. The new matched phone comparison supersedes the earlier 28 ms wording below: **24.1 ms/frame for current FP16 visual/recognition components versus 112.6 ms/frame for FP32**, with English unchanged and outside timing. No validation-set size is included in the manuscript. Historical proposals and author comments below are retained for traceability.

### Chart cleanup — completed

Removed five redundant comparison charts: recognition inputs, architecture variants,
model families, streaming recognition, and English-output quality. Their tables and
explanations remain. Retained the combined extractor accuracy/Mac-speed figure and
phone precision chart, along with system diagrams and training curves. Renumbered
remaining figures and refreshed contents/list page references. Transformer testing
has not started; it is the next discussion.

### Applied comparison updates

| Recognition stage | Apple Vision Top-1 | MediaPipe Top-1 | Apple Vision on Mac | MediaPipe on Mac |
| --- | ---: | ---: | ---: | ---: |
| Landmark-only | 95.77% | 91.80% | 356.1 ms/clip | 546.7 ms/clip |
| Combined inputs | 96.30% | 94.44% | 668.6 ms/clip | 809.3 ms/clip |
| Interval-adapted | 95.24% | 94.44% | 668.7 ms/clip | 808.6 ms/clip |

The same validation recordings were used for the Mac comparison. Times are medians of
per-clip measured stage totals, with loading and initial decoding outside timing.
The phone comparison remains separate: **24.1 versus 112.6 ms/frame on iPhone 13**.
Recognizer-only conversion accuracy remains **95.24% FP32 versus 94.97% FP16**, with
identical prepared inputs; it is not an on-phone or complete-pipeline accuracy result.
The current deployed configuration is unchanged.

Updated seven platform-neutral diagrams, the two-panel recognition/Mac-speed figure,
and the separate phone precision chart. Removed the language screening table/figure
and detailed processor/thread discussion. The manuscript contains no new technical-notes
section. Requirements include Kotlin. Slides remain deferred.

Evidence: [matched Mac comparison](../../artifacts/reports/capstone_mac_comparison_v17_20261007/REPORT.md)
and [phone precision comparison](../../artifacts/reports/phone_precision_v17_20261007/REPORT.md).

### Historical proposals and author feedback

Previously checked proposals have been removed from this checklist; their approval remains recorded in the project log. All presentation items have been removed from this review. The proposals below cover unresolved questions and new wording arising from our latest discussion.

Check a box to approve an item, or use **Your note** to request changes. IDs are stable for this review round. The implementation summary above records what has now been applied; the passages below retain the earlier review history.

Source: [Capstone 2: ATLAS — (Oct 6) Revision](https://docs.google.com/document/d/1sC2JZ23mSpnLWeEEFDm5ce2x_As0XTS7ayMGvk5uuD4/edit?tab=t.pzqwjycim1ix). Before passages below are excerpts or explicitly labeled summaries from the reviewed manuscript. Refresh its live text before applying revisions.

## Chapter 2 — introduce the extraction frameworks together

### P-01 — Joint Apple Vision and MediaPipe introduction

- [x] Approve P-01

**Before — structure:** Section 2.1.2 introduces MediaPipe Hands and Apple Vision in separate paragraphs.

**After — proposed combined introduction:**

> Frameworks such as Apple Vision and MediaPipe provide tools for extracting landmarks from camera images. These landmarks describe the positions of relevant body regions and provide structured inputs for recognition models (Apple, n.d.-a; Apple, n.d.-b; Zhang et al., 2020). Hand, facial, and upper-body observations can provide complementary information for describing signing activity.

**Note:** Responds to your C2-01 comment. Preserve the citations and following discussion of multiple body regions and part-wise learning. Do not imply that the cited MediaPipe Hands paper alone establishes face and body extraction; framework-specific capabilities can be explained where needed. This replaces the earlier proposal to retain the two separate introductions.

**Your note:**

## Chapter 3 — requirements and evaluation equipment

### R-08 — Separate device evaluation from the common-hardware comparison

- [x] Approve R-08

**Before — summary:** General mobile-device testing entries; the previous proposal added iPhone 13 and Huawei nova Y70 together as test devices.

**After — proposed requirements wording:**

> The iPhone 13 was used for the reported mobile-application processing measurement. The Apple M4 development computer provides the common hardware environment for comparing recognition performance and processing speed between the Apple Vision and MediaPipe pipelines. Device measurements and development-computer comparisons are reported separately.

**Note:** This is the intended evaluation structure. The Mac speed-comparison sentence becomes a completed-method statement only after matching measurements are verified. Keep Huawei nova Y70 in the implementation-equipment inventory if needed, but do not add a Huawei-versus-iPhone performance chart. Kotlin remains part of the already-approved software requirements. Do not label any tested device as a minimum supported specification.

**Your note:no, d not add the huawei y70 anywhere**

## Chapter 4 — simplify the explanation

### P-02 — Retain essential measurement context without a technical-notes section

- [x] Approve P-02

**Before — summary:** The deployment/results discussion repeatedly explains thread counts, processor selection, warm-up procedures, attention caching, export precision, and component timing conditions.

**After — proposed editorial treatment:**

- Remove thread counts, processor scheduling, repeated precision explanations, and runtime-specific execution mechanics from the main discussion.
- Do not add a separate technical-notes section or appendix to the submitted paper for these details.
- Keep essential context beside each result: task, evaluation data, measurement unit, hardware, and included processing work.
- Preserve existing repository measurement records for traceability; removing material from the paper does not delete evidence files.

**Note:** The chapter should explain how the system's design supports recognition, English output, and practical local execution. The new Mac comparison still needs a short measurement description; complex experimental controls can remain in the development records.

**Your note:**

### P-03 — Shorten the deployment and precision discussion

- [x] Approve P-03

**Before — affected excerpts:**

> Core ML provides the chosen deployment runtime, with processor settings that permit supported operations to use the CPU, GPU and Neural Engine.

> The original PyTorch models used one CPU thread; Core ML was permitted to select among the available processors.

**After — proposed core explanation:**

> The trained models were exported for execution within the mobile application. The evaluated configuration uses reduced-precision visual and recognition models to decrease storage requirements. Comparisons before and after conversion examined whether the exported models retained their recognition performance. The application connects these components with camera input, gloss management, English generation, and local speech output.

**Note:** Supersedes the detailed processor wording in previously checked M-04, C4-08, and C4-11. Keep one brief FP16/FP32 definition and the model-export table if they help explain the storage result. Preserve the explanation of how the components cooperate; avoid repeating this paragraph in both methodology and results.

**Your note:**

### P-04 — Keep accuracy and storage in the conversion comparison

- [x] Approve P-04

**Before — table contents:** Table 19 includes recognition accuracy, boundary-state agreement, saved model size, recognition time per batch, and boundary-model time per window.

**After — proposed table contents:**

| Measure | Before conversion | After conversion |
| --- | ---: | ---: |
| Recognizer validation Top-1 accuracy | 95.24% | 94.97% |
| Recognizer validation Top-5 accuracy | 98.94% | 98.94% |
| Recognizer saved model size | 49.62 MB | 25.51 MB |
| Boundary-model saved model size | 9.25 MB | 4.68 MB |

**Proposed accompanying text:**

> Conversion reduced the saved size of the recognition components while retaining similar validation accuracy. The recognizer's Top-1 accuracy changed from 95.24% to 94.97%, while Top-5 accuracy remained 98.94%. Its saved size decreased from 49.62 MB to 25.51 MB. The boundary model decreased from 9.25 MB to 4.68 MB and retained the original model's predicted timing states on the checked inputs.

**Essential table note:**

> Model sizes compare saved FP32 inference weights with complete FP16 deployment packages, in decimal megabytes. They describe model storage, not application size or working memory. Accuracy was evaluated using the same validation inputs before and after conversion.

**Note:** Remove the old per-batch/per-window timing rows and their detailed thread/processor explanation from the submitted paper. Keep boundary-state agreement as a short conversion-check statement, not a claim of boundary-detection accuracy. This does not remove the separately proposed Mac pipeline-speed comparison.

**Your note:is the current model using this? the conversion saved about 20mb total and that is still a small amount. lets discuss tis because i dont wantthe deployed model to be the weaker one. **

### P-05 — Remove architecture-screening complexity from English-output results

- [x] Approve P-05

**Before — summary:** The section compares 173 ms and 391 ms graph-timing estimates using untrained weights and explains attention caching and configuration differences. It also reports estimated incremental-versus-whole-sequence generation work.

**After — proposed treatment:**

- Retain the evaluated English-output quality results and their essential assessment method.
- Remove the 173 ms versus 391 ms architecture-screening table/chart from the submitted paper.
- Remove the associated untrained-weight, cache, processor-setting, and graph-timing discussion.
- Remove the 148 ms versus 208 ms estimates from the main narrative if retaining them requires another detailed explanation of graph estimates.

**Proposed functional explanation:**

> English generation prepares sentence updates as accepted glosses accumulate. Completed sentences are retained while signing continues. When the user completes the message, the application finalizes the remaining text and provides local speech output.

**Note:** Describe the implemented behavior without claiming a newly measured translation-speed improvement. Preserve how English quality was assessed; do not present automated assessment as human evaluation. This replaces the earlier suggestion to move the screening chart into a manuscript technical appendix.

**Your note:**

## Chapter 4 — updated recognition and processing-speed comparisons

### C4-02 — Show all three recognition stages

- [x] Approve C4-02

**Before:** Section 4.4.1 pairs the historical 93.12% versus 89.95% accuracy comparison with extraction times from that earlier experiment.

**After — proposed recognition chart/table:**

**Title: Recognition performance across three model stages**

| Model stage — ASL Citizen validation Top-1 | Apple Vision inputs | MediaPipe inputs |
| --- | ---: | ---: |
| Landmark-only base recognizer | 95.77% | 91.80% |
| Combined landmark and hand-image recognizer | 96.30% | 94.44% |
| Recognizer adapted to candidate sign intervals | 95.24% | 94.44% |

**Proposed explanation:**

> The comparison examines three stages of recognition development using the same ASL Citizen validation set. The landmark-only recognizer uses structured movement information. The combined-input recognizer adds hand-image features, while the interval-adapted recognizer is prepared for the candidate sign intervals encountered during streaming. The reported percentages measure isolated-sign validation performance at each stage.

**Note:** Values are documented in the October 4 rebuild report, not newly measured in this review. Do not combine these percentages with historical extraction-time values. The interval-adapted row is still isolated-sign validation accuracy, not continuous-recognition accuracy. These existing scores must pass the shared-data review in P-06 before being used in the final comparison; they are not recalculated results after a new dataset exclusion.

**Your note:do not say ASL citizen validation set. just say use the same validation set**

### P-06 — Use shared data and exclude implementation-only additions

- [x] Approve P-06

**Before — comparison scope:** The rebuild report includes ASL Citizen, SemLex, local recordings, and separate sequence evaluations. The exact “additional dataset” referred to in the latest request has not been identified by name.

**After — proposed comparison rule:**

> The comparison uses the same ASL Citizen validation recordings and vocabulary for both pipelines. Additional evaluation datasets are excluded from this comparison. Recognition results and processing times are reported for the same selected inputs.

**Review note — not manuscript text:** The October 4 report says both families used matched training-data lists and were scored on the same validation sets; it does not establish a MediaPipe-only extra dataset. Restricting the displayed comparison to ASL Citizen avoids pooling other evaluation datasets, but does not remove any influence of additional training data. If your comment refers to extra training data, identify the dataset and affected checkpoint before selecting results. Do not describe a model as trained without that dataset merely because its evaluation row is omitted.

**Implementation scope:** Verify existing data/checkpoint provenance first. This checklist authorizes no dataset deletion, retraining, or rerun of protected test sets. Use existing eligible matched results where possible. Retain the same evaluation membership for both pipelines and count failed extraction as an error rather than silently dropping difficult clips from one pipeline.

**Dataset you mean, if specific:**

**Your note::do not say ASL citizen validation set. just say use the same validation set**

### P-07 — Compare processing speed on the same Mac

- [x] Approve P-07

**Before — issue:** Historical extraction timings, model-only timings, and phone measurements cover different tasks and cannot form one matched speed chart.

**After — proposed chart design:**

**Title: Processing time on common development hardware**

- Compare Apple Vision and MediaPipe pipelines on the same Apple M4 Mac.
- Use the same ASL Citizen input clips selected for the recognition comparison.
- Show all three recognition stages alongside the corresponding accuracy chart.
- Use **median milliseconds per clip** for the isolated-sign workload, with the same input preparation and timing boundaries for both pipelines.
- Include landmark extraction, applicable hand-image feature extraction, and recognition; exclude one-time application startup/model loading and English generation.
- Keep input sampling and execution conditions comparable. Record detailed settings internally; state only the hardware, workload, and timing scope in the paper.

**Proposed chart layout:** Three groups, one per model stage; two bars per group, labeled Apple Vision and MediaPipe. Use a separate processing-time panel from the accuracy panel rather than mixing percentages and milliseconds on one axis.

**Proposed caption, after verification:**

> Median processing time per clip on the same development computer using identical input recordings. Timing includes visual-input preparation and sign recognition for each configuration.

**Note:** Timing cells remain unfilled until matching saved measurements are verified or a separately scoped benchmark is performed. Do not reuse the iPhone's 28 ms/frame here. These are pipeline comparisons on common hardware, not an operating-system speed ranking. Using the same Mac alone does not make unlike processing workloads equivalent.

**Your note:**

### P-08 — Keep the iPhone 13 evaluation separate

- [x] Approve P-08

**Before — issue:** Implementation and performance discussion repeatedly identifies a platform, potentially blurring the distinction between the mobile result and development-computer comparisons.

**After — proposed device result:**

> The selected configuration recorded a median preparation-and-recognition time of 28 ms per processed frame, measured on an iPhone 13 using recorded input.

**Proposed presentation in the paper:** One compact device-performance result, separately labeled from the Mac comparison. Identify the measured task and device beside the value.

**Note:** Retains the already-agreed timing wording as part of the new evaluation structure. No Android device-speed value is added. This result is not per clip, complete speech-response time, or proof of sustained live-camera throughput. Use “mobile application” in the surrounding system description.

**Your note:**

## Chapter 4 — narrative flow and evaluation terminology

### P-09 — Organize results around the capstone's contribution

- [x] Approve P-09

**Before — summary:** Recognition, conversion mechanics, language-model screening, and detailed timing conditions compete for attention across the results discussion.

**After — proposed reading order:**

1. Recognition performance, including the three-stage comparison and matched Mac processing-time results.
2. Streaming recognition on recorded sequences.
3. English-output quality.
4. Model conversion: retained recognition accuracy and reduced storage.
5. iPhone 13 preparation-and-recognition measurement.

**Note:** Explain what each finding means for the mobile application. Preserve relevant model-development comparisons and training curves; this is not permission to remove unrelated study content. Update section numbering and cross-references only after the final structure is agreed.

**Your note:**

### P-10 — Clarify signer-disjoint terminology where already established

- [x] Approve P-10

**Before — Section 2.1.8 excerpt:**

> Finally, evaluation must distinguish new recordings from new signers. Separating signer identities across training, validation, and testing examines recognition on people outside the corresponding learning groups, a principle used in community-sourced isolated-sign research (Desai et al., 2023).

**After — proposed definition:**

> Signer-disjoint evaluation assigns each signer exclusively to the training, validation, or test set. Consequently, recordings from the same person cannot appear across these sets. This evaluates recognition on people not encountered during training, following the signer-independent evaluation approach used in ASL Citizen (Desai et al., 2023).

**Note:** Section 4.3.1 and Table 10 already connect this principle to isolated-sign evaluation. Keep the citation and verify study-specific split claims against the underlying records. As requested, do not add the proposed familiar/unseen-signer breakdown to the streaming narrative; describe that result as “streaming recognition on recorded sequences” without applying the isolated-sign signer-disjoint label to it.

**Your note:**

## Evidence for review

These links are for checking this proposal, not a proposed technical-notes section in the submitted paper.

- [October 4 paired pipeline results](../../artifacts/reports/mediapipe_rebuild_v17_20261004/REPORT.md)
- [Mobile deployment history](../ground_truth/mobile-deployment/log.md)
- [Existing measurement provenance](ATLAS_MEASUREMENT_NOTES.md)
- [ASL Citizen paper — Desai et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf)

## Additional feedback

**Section / item:**

**Requested change:**
