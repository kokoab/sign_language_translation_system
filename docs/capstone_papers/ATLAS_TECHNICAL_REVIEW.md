# ATLAS — technical comparisons, figures, and decision rationale

Prepared 2026-09-30. Companion to [Major changes](ATLAS_MAJOR_CHANGES.md).
Technical review of system design, model comparisons, and iPhone implementation.

## System components and modular architecture

ATLAS is an iPhone application that recognizes ASL signs, forms English sentences,
and reads the text aloud. Its modular architecture divides this work among components
with defined responsibilities. The components exchange landmarks, image features,
sign labels, and text. A module may contain a trained model, an Apple framework, or
processing rules; “modular” describes this division of work.

| Component | Role in ATLAS | Information passed onward |
| --- | --- | --- |
| Apple Vision | Locates hand, face, and upper-body landmarks in camera frames. | Landmark positions and confidence values. |
| MobileCLIP2 | Represents the appearance of cropped hand images. | Numerical image features. |
| Boundary model | Estimates where signing begins, continues, and ends. | Candidate sign intervals. |
| Squeezeformer-based recognition | Uses landmark motion and hand-image features to recognize signs. | Sign predictions and recognition scores. |
| Segmental decoder | Selects a sequence of sign intervals and determines when a prediction is stable enough to display. | Accepted sign labels, called glosses. |
| T5-efficient-tiny | Converts the recognized input into English and prepares sentence updates during signing. | English text. |
| Core ML and local speech output | Run the exported models on the iPhone and read the English text aloud. | Visible text and audible speech. |

Apple Vision supplies the spatial input. MobileCLIP2 supplies complementary hand
appearance information. Squeezeformer processes changes across time so that recognition
uses motion as well as individual hand shapes. The boundary model and decoder coordinate
when candidate signs are evaluated and accepted. T5-efficient-tiny then handles English
wording. This separation makes each responsibility explicit and allows recognition,
sentence generation, and execution performance to be assessed individually.

The boundary model was trained through **knowledge distillation**: a pretrained
*Sign Language Segmentation* model supplied timing predictions used to teach ATLAS's
own boundary model. The teacher comes from the `sign-language-processing/segmentation`
project's 2026 CNN–Transformer implementation, pretrained on German Sign Language
(DGS) and using MediaPipe pose inputs. The related research is *Linguistically
Motivated Sign Language Segmentation* (Moryossef et al., 2023); the repository identifies
that paper with its 2023 version, so the 2026 implementation is cited separately.
[Model repository](https://github.com/sign-language-processing/segmentation),
[research paper](https://aclanthology.org/2023.findings-emnlp.846/).

During distillation, the teacher's weights remained fixed. ATLAS's four-layer Transformer
boundary model learned its timing outputs using Apple Vision features and uses a short
look-ahead window to identify sign intervals. The teacher supports training; the
Apple Vision model supplies boundary estimates in the application. This transfers
segmentation information, while the sign recognizer supplies the ASL vocabulary labels.
The recognizer was separately fine-tuned on candidate sign intervals to match the inputs
it receives during streaming recognition.
[Implementation and training evidence](../../artifacts/reports/segmental_decoder_v17_20260927/REPORT.md).

## Data preparation and model training

ATLAS uses isolated-sign clips to learn vocabulary labels, sign-sequence recordings
to learn recognition across connected signing, and gloss-to-English pairs to train
sentence generation. These inputs serve different purposes and retain their own labels
and preparation records.

Preparation included annotation review and manual clipping of selected recordings.
A gloss annotation identifies the sign; a temporal annotation identifies the portion
of the recording in which it occurs. Manual clipping isolates a selected portion of
a recording for use as an example. Automatic activity trimming is a separate operation:
it removes leading and trailing inactivity during isolated-sign feature preparation.
Together, these steps connect the intended labels with the relevant visual content.

For sequence training, timed annotations and known gloss sequences supply different
forms of supervision. Where only the gloss order is available, the training procedure
aligns that sequence with candidate intervals. Those automatically assigned intervals
are kept distinct from manually annotated boundaries. The boundary model learns timing
from the pretrained teacher, while the recognition model learns the sign labels.

| Method | Application in ATLAS | Purpose |
| --- | --- | --- |
| Label and recording checks | Preserve the selected vocabulary labels, verify recording identity and usable features, and retain preparation records. | Keep each example connected to its intended target. |
| Annotation and manual clipping | Review sign labels and isolate selected portions of recordings; use timed annotations where available. | Associate sign content with the relevant video interval. |
| Automatic trimming and sampling | Remove surrounding inactivity from isolated-sign inputs and sample frames into the recognizer's input format. | Represent the signing portion consistently. |
| Landmark normalization | Express positions relative to the body and use a common scale, normally shoulder width, while preserving image proportions. Missing landmarks retain presence indicators. | Reduce variation from camera placement and signer size while preserving relative hand movement. |
| Training augmentation | Vary landmark position, uniform scale, rotation, timing and visibility; mirror with the corresponding left/right landmark swap. Sign-interval training also varies candidate start and end positions. | Expose recognition to variation in appearance, motion and interval selection. |
| Regularization | Use dropout and weight decay in model training; retain isolated-sign examples and reference predictions during sign-interval adaptation. | Reduce reliance on particular training patterns and preserve learned vocabulary recognition during adaptation. |
| Data separation and model selection | Keep the isolated-sign evaluation signer-disjoint; preserve the documented recording groups for sequence studies. Use validation or designated tuning results for model selection. | Assess performance on examples separated from the corresponding training procedure. |

The landmark recognizer receives **32 sampled frames per candidate sign interval**.
Intervals can have different durations; resampling gives each one the same input length.
Each sampled frame represents **61 landmark points**: 21 per hand, 15 facial points and
4 upper-body points. Each point carries five values: body-relative X and Y, a relative
scale-based depth proxy, presence and confidence. The resulting input is 32 × 61 × 5.
The 32-frame setting specifies model input length, not camera frame rate or a requirement
to wait for 32 new camera frames before every prediction. The sequence HELLO HOW YOU
is evaluated through candidate intervals, each prepared for the recognizer in this form.
[Input schema](../../active/v17/schema_v17.py),
[interval input preparation](../../active/v17/segmental_runtime_v17.py).

**Normalization** changes the representation of an input consistently.
**Augmentation** introduces variations into training examples. **Regularization**
constrains learning: dropout temporarily omits some internal activations during training,
and weight decay discourages excessively large weights. During sign-interval adaptation,
reference predictions also encourage the recognizer to retain its isolated-sign behavior.
These methods serve different roles; their use follows each component's training recipe.

The sign-interval recognizer is trained on intervals resembling those proposed during
streaming. Slightly shifting their boundaries exposes it to variations in where a sign
is cut. Training combines these examples with isolated-sign examples, and model selection
considers sequence recognition together with retained isolated-sign performance.

Implementation evidence: [landmark preparation](../../active/v17/geometry_v17.py),
[landmark training and augmentation](../../active/v17/train_stage_1_v17.py),
[recorded training provenance](../../artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/training_data_provenance.json),
[interval preparation](../../scripts/segmental_lab_v17.py), and
[recognizer adaptation](../../scripts/train_span_recognizer_v17.py).

## Reading the comparisons

The comparisons explain three engineering questions: how accurately the system
recognizes signs, how well it generates English, and how quickly it runs on the iPhone.
Each table identifies the task and measurement setting.

**Top-1 accuracy** is the percentage of clips whose highest-scoring prediction is correct.
**Top-5 accuracy** counts clips where the correct sign appears among the five highest
predictions. **Median time** is the middle measured processing time. Lower processing
time means faster execution. Model size is reported in millions of learned parameters.

Charts and editable diagrams are in [review_assets](review_assets/).

## 1. Apple Vision versus MediaPipe

**Decision:** use Apple Vision for the iPhone landmark input.

Both landmark methods were evaluated on the same recordings, with matching recognition
model configurations and training procedures.

| Measure | Apple Vision | MediaPipe |
| --- | ---: | ---: |
| Successful clip extraction | 100% | 100% |
| Median extraction time per clip | 0.678 s | 1.230 s |
| Classifier top-1 | 93.12% | 89.95% |
| Classifier top-5 | 99.47% | 97.35% |

![Extractor comparison](review_assets/extractors.png)

Apple Vision processed clips faster and produced higher recognition accuracy in this
comparison. These results, together with its integration with iOS, support its use
for landmark extraction in ATLAS.

Apple documents the framework's hand/body pose capability; that documentation establishes
functionality, while the local comparison supports selection. [Apple, hand-pose documentation](https://developer.apple.com/documentation/vision/detecting-hand-poses-with-vision)

Local evidence: [landmark comparison evidence](../../artifacts/reports/capstone1_v17_revision_checklist_v1/README.md).

## 2. Landmarks, hand images, and combined recognition

**Decision:** combine structured landmark motion with cropped-hand visual features.

| Recognition input | Top-1 | Top-5 |
| --- | ---: | ---: |
| Hand-image features | 80.69% | 94.71% |
| Landmarks | 95.50% | 98.68% |
| Learned combination | **96.30%** | **99.21%** |

![Recognition input comparison](review_assets/modalities.png)

**Interpretation:** landmarks were the stronger individual component, while the learned
combination added 0.80 percentage point over landmarks on the same validation recordings.
ATLAS combines both inputs to use motion and hand appearance together.

Landmarks represent configuration and motion; hand-image features retain appearance
details. Multimodal sign-recognition literature supports investigating complementary
inputs, and MobileCLIP2 supplies an efficient image encoder. The measured improvement is reported in the table. [Jiang et al., 2021](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html),
[Faghri et al., 2025](https://arxiv.org/abs/2508.20691)

Local evidence: [matched component comparison](../../artifacts/reports/capstone1_v17_revision_checklist_v1/README.md).

## 3. Recognition architecture comparisons

### 3.1 Landmark architecture variants

**Decision:** retain part-wise temporal encoding followed by global Squeezeformer processing
inside the multimodal recognition component.

“Part-wise” means processing body regions, such as the hands and upper body, before
combining their information. “Global” processing considers their combined movement
over time. A flat model processes the input together, while the graph variant explicitly
represents connections between landmark points.

| Architecture | Parameters | Top-1 | Top-5 | CPU median |
| --- | ---: | ---: | ---: | ---: |
| Graph-part replacement | 6.48 M | 78.31% | 95.77% | 11.42 ms |
| Wider flat Squeezeformer | 14.34 M | 95.24% | 99.74% | 7.35 ms |
| Flat Squeezeformer | 6.47 M | 95.77% | 100.00% | **4.90 ms** |
| Part-wise + global Squeezeformer | 6.79 M | **96.83%** | 99.21% | 6.50 ms |

![Architecture variants](review_assets/architecture.png)

**Interpretation:** the part-wise model gained 1.06 top-1 points over the flat model,
at 1.60 ms additional median classifier computation. The flat model was faster;
the selected model prioritized recognition accuracy. Accuracy is measured on validation recordings. Base-model timing uses one CPU thread
and prepared landmark inputs. Full processing conditions are in the
[measurement notes](ATLAS_MEASUREMENT_NOTES.md).

Part-wise modeling has a sign-recognition research basis. Squeezeformer provides an
attention-and-convolution temporal design originally developed for speech. ATLAS adapts
that design to signing.
[Lee et al., 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Human_Part-wise_3D_Motion_Context_Learning_for_Sign_Language_Recognition_ICCV_2023_paper.html),
[Kim et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract.html)

Local evidence: [accuracy table](../../artifacts/reports/capstone1_v17_revision_checklist_v1/README.md)
and [matched timing record](../../artifacts/reports/capstone1_v17_revision_checklist_v1/architecture_latency_benchmark.json).

### 3.2 Recognition model families

Recurrent models such as BiLSTM track information through a sequence. Temporal
CNNs learn patterns across nearby frames. Transformers relate information across
frames using attention, and Squeezeformer combines attention with convolution.

The selected comparison presents BiLSTM, Temporal CNN, Flat Transformer and the ATLAS
recognition model using the same training recordings,
validation recordings, training procedure, and random seed. Numerical values below come from its
saved result file. The family comparison and the architecture-variant comparison
use separately trained checkpoints; their accuracy values describe those respective
comparisons.

| Model family | Parameters | Validation top-1 | CPU median |
| --- | ---: | ---: | ---: |
| BiLSTM | 6.10 M | 89.15% | 3.39 ms |
| Temporal CNN | 5.90 M | 92.59% | 4.34 ms |
| Flat Transformer | 6.72 M | 95.50% | **2.88 ms** |
| ATLAS (part-wise + global Squeezeformer) | 6.79 M | **96.30%** | 6.15 ms |

![Architecture family comparison](review_assets/families.png)

Squeezeformer achieved 96.30% accuracy, compared with 95.50% for the flat Transformer:
an improvement of 0.80 percentage points. Its CPU prediction
time was 6.15 ms versus 2.88 ms, an additional 3.27 ms or about 2.1 times the duration.


This is an accuracy–speed trade-off. Squeezeformer was retained to prioritize recognition
accuracy. The flat Transformer offers a competitive alternative when computation time
has greater weight. The results come from one training run per family. These CPU timings describe the base models; the Core ML measurements in Section 6
explain deployment performance separately.

Local evidence: [family experiment record](../../artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark/result.json).

## 4. Streaming recognition

A continuous camera stream contains signs, transitions, and pauses. ATLAS estimates
candidate sign intervals, recognizes their content, and selects a sequence of outputs.
The decoder combines timing estimates with recognition scores before accepting a sign.

The researchers also tried Connectionist Temporal Classification (CTC), a method that
learns sign sequences without an exact boundary annotation for every sign
([Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). In the tested
window-based implementation, the first full-window prediction required approximately
1.07 seconds of input before processing. Predictions could change as more input arrived,
and speech waited for a stable sequence. Live observations included repeated output
for a held sign and a camera-timing mismatch; separate causal CTC variants also had
substantial connected-sign validation errors. These recorded findings motivated explicit
boundary estimation and stable word commitment for ATLAS's live interaction.
[Technical evidence](../../artifacts/reports/capstone_ctc_comparison_v17_20260930/REPORT.md).

Timely feedback is relevant to the intended interaction. Kamikubo et al. (2025) report
that 90% of ten surveyed Deaf or hard-of-hearing ASL signers prioritized real-time
sign-to-spoken/written translation. Their co-design study also identified latency as a
technical consideration. This supports responsive interaction without prescribing one
response-time threshold for all signers.
[Study, Sections 4.3 and 5.4](https://users.umiacs.umd.edu/~hal3/docs/daume25collaboration.pdf).
General interface guidance also connects short response times with continuity of
interaction ([Nielsen, 1993](https://www.nngroup.com/articles/response-times-3-important-limits/)).

| Recognition configuration | Word error rate |
| --- | ---: |
| Boundary-guided interval classification | 39.78% |
| ATLAS streaming recognition | **9.68%** |

![Streaming recognition comparison](review_assets/streaming.png)

**Word error rate (WER)** counts wrong, missing, and extra sign labels relative to a
reference sequence. Lower values indicate fewer sequence errors. Boundary-guided
interval classification uses predicted boundaries to pass sign intervals to the
recognizer. ATLAS streaming recognition combines a distilled boundary model,
interval-adapted recognition, and sequence selection. Both configurations were assessed
on the same recorded sequences. The 30.10-percentage-point difference reflects these
combined system choices.

The recordings include familiar-signer and unseen-signer sequences. Processing conditions,
recording membership, and detailed results are documented in the
[measurement notes](ATLAS_MEASUREMENT_NOTES.md).

### Worked sequence: HELLO HOW YOU

![HELLO HOW YOU recognition timeline](review_assets/hello_how_you_timeline.png)

The timeline follows a recorded **HELLO HOW YOU** sequence. Apple Vision provides
landmark features, and MobileCLIP2 provides hand-image features from the same video.
Boundary estimates guide the search for sign intervals. Squeezeformer-based recognition
scores their content, and the segmental decoder selects the sequence.

The upper row shows the intervals selected using boundary estimates and decoder
decisions. The lower row associates those intervals with the recognition model’s gloss
predictions. These are the selected model intervals in the saved recognition record. They locate HELLO at approximately 0.67–1.17 seconds, HOW at 1.40–1.67 seconds,
and YOU at 2.27–2.87 seconds. They are model estimates, rather than manually annotated
sign boundaries. The diagram shows these selected intervals; the decoder also considers
other candidates before choosing an output.

The accepted gloss sequence grows from HELLO to HELLO HOW and then HELLO HOW YOU. T5-efficient-tiny uses the accumulated glosses to prepare English
updates during signing. The Finish button or held-open-palms gesture finalizes the
remaining English text for display and local speech output.

The horizontal axis denotes position within the recording. The interval bars show where
the model located each sign, with start and end times above them. Video stills below
the timeline show one original frame near the midpoint of each interval. Their timestamps
identify positions in the same recording; the stills are examples of the visual input,
while recognition uses the sampled sequence of frames. The [measurement notes](ATLAS_MEASUREMENT_NOTES.md#hello-how-you-timeline)
identify the source recording and distinguish interval timing from word-acceptance timing.

## 5. English generation: model choice and incremental output

### 5.1 English-output comparison

**Decision:** use T5-efficient-tiny with incremental sentence generation and consistency checks. The comparison uses generated gloss-to-English sessions of two to four
sentences. DeepSeek supplies the reference sentences and automatic ratings.

| Configuration | Judged fully correct | Judged wrong | BLEU | NO omitted |
| --- | ---: | ---: | ---: | ---: |
| T5-efficient-tiny: whole-sequence generation | 60% | 6% | 75 | 0/42 |
| T5-efficient-tiny: incremental generation | 60% | 8% | 74 | 0/42 |

![English output comparison](review_assets/english_quality.png)

Both configurations use the same fine-tuned T5-efficient-tiny model, with approximately
15.6 million parameters. Whole-sequence generation processes all recognized input
together. Incremental generation prepares English during signing, keeps completed
sentences after consistency checks, and processes the remaining text at Finish.
Both configurations received a 60% fully correct automatic rating.

The automatic evaluator used three categories:

| Rating | Meaning |
| --- | --- |
| Fully correct | Preserves the reference meaning, uses grammatical English, and includes the important content. |
| Mostly correct | Preserves the main meaning with a minor wording, grammar, or content error. |
| Incorrect | Changes the meaning, omits important content, invents content, or makes the meaning unclear. |

The comparison reports the first and third categories. BLEU measures wording overlap
with reference sentences; the “NO omitted” column counts examples where negation was
lost. Both configurations preserved negation in all 42 examples.

[G-Eval, Liu et al. (2023)](https://aclanthology.org/2023.emnlp-main.153/) provides a research
basis for using a language model to evaluate generated text. ATLAS uses its own
DeepSeek-based rubric, described above, rather than the original G-Eval implementation.
These scores are automatic judgments on generated gloss-to-English sessions. The same
model family produced the references and judgments; the results are not independent
human assessments. The citation supports the evaluation approach, while the project
records supply the sentence ratings.

[T5, Raffel et al. (2020)](https://jmlr.org/papers/v21/20-074.html) supports text-to-text
modeling; [Müller et al. (2023)](https://aclanthology.org/2023.acl-short.60/) discusses
gloss-based translation evaluation; [Papineni et al. (2002)](https://aclanthology.org/P02-1040/)
defines BLEU.

### 5.2 Why use a small language model?

The following is a comparison of model computation times on an idle iPhone 13. Candidate
architectures were timed with untrained weights to compare execution cost. Figures estimate a 28-output-token session from measured
graph timings. Each row uses its best measured compute setting. KV caching reuses completed calculations when generating the next word fragment.

| Candidate execution configuration | Parameters | Estimated 28-token time |
| --- | ---: | ---: |
| ATLAS tiny T5 architecture, KV cache, FP16 | 15.6 M | 173 ms |
| FLAN-T5-base, KV cache, FP16 | 248 M | 391 ms |

![Language model execution screening](review_assets/language_latency.png)

The comparison retains the tiny T5 architecture used by ATLAS and FLAN-T5-base.
These measurements compare execution cost; Section 5.1 reports English quality.
The application's English model uses FP32 Core ML with no attention cache. At its
recorded output lengths, the timing model estimates median final-generation work of
148 ms for incremental generation and 208 ms for whole-sequence generation. These
estimates combine measured processing costs with output length.

Local evidence: [quality and training report](../../artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/REPORT.md),
[device architecture benchmark](../../artifacts/reports/stage3_latency_bench_v17_20260929/REPORT.md),
[timing rows](../../artifacts/reports/stage3_latency_bench_v17_20260929/summary.json).

## 6. Converting the models for iPhone execution

### From trained models to the application

ATLAS brings together four neural components with different responsibilities: the temporal boundary model, the Squeezeformer-based recognizer, the MobileCLIP2 image encoder and T5-efficient-tiny. The boundary model estimates candidate sign timing, while the recognizer determines which signs those intervals contain. MobileCLIP2 supplies hand-appearance features that complement landmark motion, and T5-efficient-tiny turns the accepted gloss sequence into English. These components connect visual recognition with sentence generation, but they are not a single trained model.

The researchers converted these trained components to Core ML to integrate recognition and English generation into the native iPhone application and execute them locally. The models were developed in PyTorch, while the Swift application loads their exported Core ML representations and connects them to camera input, gloss management and English output. Core ML provides the chosen deployment runtime, with processor settings that permit supported operations to use the CPU, GPU and Neural Engine. Conversion therefore serves the practical purpose of bringing the trained models into the application.

The choice of numerical precision is a separate deployment decision. FP16 exports reduce the storage requirements of the recognition components, while matched comparisons examine how closely those exports preserve the original predictions and accuracy. Similar accuracy before and after conversion indicates preservation of learned recognition behavior. Execution-time measurements assess the resulting runtime performance, which can differ by component and processor rather than improve uniformly.

All four neural components have Core ML exports. The boundary model and recognizer use FP16 exports. MobileCLIP2 has FP32 and FP16 exports in the execution comparison, with FP16 used in the selected configuration. T5-efficient-tiny uses FP32 encoder and decoder exports; these are two computational parts of the same English-generation model. The following table identifies each model and its exported form.

| Neural model | Purpose | Core ML export |
| --- | --- | --- |
| Temporal boundary model | Estimates where candidate signs begin and end from landmark features. | Boundary-model export using FP16. |
| Squeezeformer-based recognizer | Identifies candidate signs using landmark motion and hand-image features. | Recognition-model export using FP16. |
| MobileCLIP2 image encoder | Extracts appearance features from cropped hand images. | Image-encoder exports using FP32 or FP16; the selected configuration uses FP16. |
| T5-efficient-tiny | Generates English from accepted gloss sequences. | Encoder and decoder exports using FP32; both belong to one language model. |

Apple Vision supplies landmarks through an existing Apple framework; it was not converted by the researchers. The segmental decoder, Gloss Management and Finish controls are application logic rather than separately exported neural models. Camera capture, the interface and speech output likewise use native application frameworks. This distinction identifies which parts of ATLAS were exported and which coordinate their operation on the device.

FP16 and FP32 describe the numerical precision used in the model exports. They do not mean that every operation in the application, or every input array, uses the same representation. ATLAS combines FP16 visual and recognition exports with FP32 English-model exports according to the component configuration. Core ML's processor settings separately determine whether supported operations may execute on the CPU, GPU or Neural Engine.

The diagram below follows the preparation of trained models for deployment and then their integration into the application. Export changes the executable representation of a model; the application still supplies preprocessing, connects model outputs and manages the interaction. The following subsection compares recognition accuracy and boundary-state agreement before and after conversion.

**Combined operation of the deployed system.** Camera frames supply landmark and hand-image information. Boundary estimates guide candidate-interval selection, and the recognizer scores those intervals using both visual inputs. The segmental decoder selects the accepted sequence, Gloss Management preserves its order, and T5-efficient-tiny prepares English as that sequence grows. Finish finalizes the remaining text for display and local speech. The models therefore cooperate within one workflow, while each retains a specific input, output and responsibility.

![Model export and iPhone integration](review_assets/deployment_flow.png)

### Recognition performance before and after conversion

Conversion was examined separately from model selection to determine how closely the exported models preserve their original behavior. The Squeezeformer recognizer was evaluated before and after conversion using the same validation examples, reference sign labels, landmark inputs and cached hand-image features. The comparison used the recognition output of the fixed-batch FP16 package loaded by the application. Holding the inputs constant isolates the recognizer conversion from changes in MobileCLIP2 feature extraction.

The boundary-model export was checked against the original model's timing-state predictions on the same tuning inputs. This measures whether conversion changes the predicted state—outside a sign, at its beginning or within it. It is an agreement measure with the original model, rather than accuracy against manually annotated sign boundaries.

| Model and measure | Before conversion: PyTorch | After conversion: Core ML FP16 |
| --- | ---: | ---: |
| Squeezeformer recognizer: validation Top-1 accuracy | 95.24% | 94.97% |
| Squeezeformer recognizer: validation Top-5 accuracy | 98.94% | 98.94% |
| Boundary model: timing-state agreement with the original model | Reference predictions | 100.00% |
| Squeezeformer recognizer: median time per batch of eight sign intervals | 27.64 ms | 11.32 ms |
| Boundary model: median time per input window | 1.39 ms | 2.00 ms |
| Squeezeformer recognizer: saved model size | 49.62 MB | 25.51 MB |
| Boundary model: saved model size | 9.25 MB | 4.68 MB |

Model size is reported in decimal megabytes (1 MB = 1,000,000 bytes). Before conversion, the measurement covers saved FP32 inference weights without optimizer state or training metadata. After conversion, it covers the complete FP16 Core ML model package, including its graph and weights. These are storage sizes for the two deployment forms, rather than working memory or the size of the complete application.

Processing time was measured on the same Apple M4 development Mac using identical prepared inputs before and after conversion. The original PyTorch models used one CPU thread; Core ML was permitted to select among the available processors. Each version was warmed up before repeated, interleaved measurements. Recognition uses a batch of eight candidate sign intervals per call, while the boundary model processes one input window per call. These timings exclude input preparation, model loading and compilation and describe execution-path differences, including processor selection, rather than the effect of FP16 precision alone.

The recognizer's Top-1 accuracy changed from 95.24% to 94.97%, a decrease of 0.26 percentage points, while Top-5 accuracy remained 98.94%. The original and exported recognizers agreed on 99.74% of their highest-scoring predictions. These values describe the interval-adapted recognizer selected for deployment; the model-family comparison uses separately trained checkpoints and therefore reports different accuracy values.

The boundary export preserved all highest-scoring timing-state predictions in its recorded conversion check. Together, the results show that conversion retained boundary-state decisions on the checked inputs and introduced a small change in the recognizer's validation accuracy. The two measures answer different questions and are not combined into a single system-accuracy score.

The recognizer's median processing time decreased from 27.64 to 11.32 milliseconds per batch of eight intervals. The boundary model changed from 1.39 to 2.00 milliseconds per window, showing that conversion did not reduce execution time for every component in this measurement. Their different input units are retained so these component timings are not mistaken for a total frame-processing time.

On the iPhone 13, the selected configuration recorded a median of **28 ms per processed frame for preparation and recognition**.

Evidence: [matched conversion check](../../artifacts/reports/capstone_conversion_v17_20260930/REPORT.md).

## 7. Recorded training curves

### Landmark recognition training

![Landmark training](review_assets/recognition_training.png)

This plots 130 recorded epochs, with selected epoch 100 marked. Its validation top-1
was 96.83%. The curve describes training of the landmark recognition component. Training and validation losses retain their logged
objective definitions.

### Recognition training on sign intervals

![Span adaptation](review_assets/span_training.png)

This plots the actual eight-epoch adaptation on 6,254 candidate sign spans. Epoch zero
is the starting recognizer. The right panel uses the tuning decoder configuration for
the adaptation study. The chart shows how loss and sequence error change during training.

### English-model fine-tuning

![English training](review_assets/english_training.png)

This plots five saved observations through optimizer step 4,539. Final recorded
validation loss is 0.2515. Lower loss indicates a closer fit to the reference outputs.
Section 5 reports the separate assessment of generated English.

## 8. iPhone system flowcharts

### Application navigation

![ATLAS iPhone application navigation](review_assets/app_flow.png)

The Home screen branches to Live, Glosses, Practice, and History. Glosses opens sign
demonstrations and allows practice of the selected sign. Practice offers random sets
of 5, 10, or 20 signs, feedback on attempts, a Skip action, and a results screen.
History lists saved session dates, durations, and generated text. Navigation returns
to Home or another activity. Live and Practice become available when the models load.

Editable source: [app_flow.mmd](review_assets/app_flow.mmd).

### Live recognition and sentence output

![System flow](review_assets/system_flow.png)

Apple Vision supplies landmarks, while MobileCLIP2 extracts hand-image features.
Streaming Sign Recognition combines these inputs and produces recognized glosses.

T5-efficient-tiny prepares English as recognized input accumulates. The single output
path then passes through the Finish button or held-open-palms gesture, finalizes the
English text, and displays and speaks the sentence. Finish also completes any remaining
recognition input before the final rendering.

Editable source: [system_flow.mmd](review_assets/system_flow.mmd).

## 9. Runtime model-decision flowchart

![Runtime decisions](review_assets/runtime_decisions.png)

The recognizer scores candidate intervals; the segmental decoder combines those scores
with temporal evidence. Output is committed when the applicable stability/timing rules
are satisfied. Otherwise the app continues observing. This combines learned models
with deterministic processing logic.

For incremental English, a first sentence is locked only when successive generations
agree, their input counts cover the buffer, and regenerating the proposed sentence's
glosses reproduces it. Finish processes the remaining tail. This consistency check
stabilizes the generated wording across updates.

Editable source: [runtime_decisions.mmd](review_assets/runtime_decisions.mmd).

## 10. Engineering-decision flowchart

![Engineering choices](review_assets/engineering_decisions.png)

This diagram summarizes the engineering choices and their supporting comparisons.

Editable source: [engineering_decisions.mmd](review_assets/engineering_decisions.mmd).

## 11. Why these decisions? Citation support for panel questions

Published work establishes a rationale; project records establish the local choice.

| Likely question | Short answer for ATLAS | Published support and its scope | Local support |
| --- | --- | --- | --- |
| Why 100 signs? | A 100-sign prototype vocabulary contains foundational signs, including WHAT, WHERE, WHEN, WHO, WHY, and HOW. The size bounds the task; its contents support the selection rationale. | [Li et al. (2020)](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.html) provides a 100-class research setting; [Woods and Rana (2023)](https://doi.org/10.3390/math11092129) examine recognition across vocabulary sizes. The project defines its own 100-sign scope. | Selected vocabulary and per-class inclusion criteria; see the vocabulary research note. |
| Why signer-disjoint splitting? | Evaluate recognition on people whose recordings were excluded from fitting and model selection. Keep every recording of a person in one split. | [Desai et al. (2023)](https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf) evaluate on users absent from training and validation. | Split manifests and signer membership checks. |
| Why Apple Vision? | It supplies the required landmarks and performed well in the project's matched extraction/classifier comparison. | [Apple documentation](https://developer.apple.com/documentation/vision/detecting-hand-poses-with-vision) establishes API capability. | Section 1: extraction timing and classifier accuracy. |
| Why include hand images? | Landmarks encode geometry; crops retain hand appearance. Combining their evidence improved the matched recognition result. | [Jiang et al. (2021)](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html) supports investigating multimodal sign recognition. | Section 2. |
| Why MobileCLIP2? | Use an efficient pretrained image encoder to represent the cropped hands. | [Faghri et al. (2025)](https://arxiv.org/abs/2508.20691) develops efficient image-text representation models. | Current hand-crop feature path and phone profiles. |
| Why Squeezeformer? | It produced the strongest validation recognition in the matched family experiment while remaining executable on the target pipeline. | [Kim et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract.html) supports the temporal architecture, originally for speech. | Sections 3.1–3.2; faster alternatives remain visible. |
| Why part-wise and global modeling? | Preserve hand, face, and body information before combining temporal context. | [Lee et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Human_Part-wise_3D_Motion_Context_Learning_for_Sign_Language_Recognition_ICCV_2023_paper.html) investigates part-specific and whole-body motion context. | Section 3.1. |
| Why segmentation and a decoder? | A camera stream needs decisions about both the identity and timing of signs. | [Zuo et al. (2024)](https://aclanthology.org/2024.emnlp-main.619/) motivates online recognition. | Streaming recognition design and supporting evidence in Section 4. |
| Why keep glosses between recognition and English? | They provide an inspectable interface and let recognition and English generation be analyzed separately. | [Müller et al. (2023)](https://aclanthology.org/2023.acl-short.60/) discusses gloss translation and the care needed in evaluating it. | Recognized sign-label sequence supplied to the English model. |
| Why T5-efficient-tiny rather than a larger model? | The fine-tuned small model met the project's generated-text quality target while keeping execution cost low. | [Raffel et al. (2020)](https://jmlr.org/papers/v21/20-074.html) supports the text-to-text formulation. | Section 5: separate quality and architecture-cost comparisons. |
| Why incremental sentences and a Finish control? | Translate stable completed text during signing and use Finish to finalize the remaining input. | Online recognition/translation is motivated by [Zuo et al. (2024)](https://aclanthology.org/2024.emnlp-main.619/); the exact sentence-locking rule is a project implementation choice. | Swift sentence-locking code and Section 5. |
| Why iPhone and local execution? | Align the app with the selected visual framework and run recognition and English generation locally. | [Apple Core ML](https://developer.apple.com/documentation/CoreML) documents local model execution and hardware integration. | Installed Swift/Core ML implementation and Section 6. |
| Why 61 nodes and 32 frames? | These specify the current anatomical representation and fixed-length recognition input. | Part-wise modeling research motivates representing individual body regions. | Input schema and implementation. Present them as configuration choices. |
| Why accuracy, WER, BLEU, and timing? | Each measures a different part: sign classification, output sequence errors, English reference overlap, and execution cost. | [Papineni et al. (2002)](https://aclanthology.org/P02-1040/) defines BLEU; [Zuo et al. (2024)](https://aclanthology.org/2024.emnlp-main.619/) reports recognition/translation metrics. | Use the corresponding table conditions. BLEU is a score, not percentage translation accuracy. |
| Why a software-quality framework? | Organize application-quality requirements separately from model metrics. | [ISO/IEC 25010:2023](https://www.iso.org/standard/78176.html) defines a product-quality model. | Application-quality requirements and evaluation design. |

### Suggested vocabulary-scope paragraph

ATLAS uses a 100-sign prototype vocabulary containing foundational signs, including
WHAT, WHERE, WHEN, WHO, WHY, and HOW. These question signs are taught in introductory
ASL activities and provide vocabulary for basic information-seeking
([Boise State University, n.d.](https://boisestate.pressbooks.pub/pathwaysasl/front-matter/introduction-page/)).
The selected vocabulary gives the prototype a defined recognition scope and includes
signs for asking about people, things, places, times, reasons, and manner.

For the recommendations section: future vocabulary expansion can address additional
communication situations identified with Deaf users, with signer-disjoint evaluation
maintained as the vocabulary grows.

See [the vocabulary research note](ATLAS_VOCABULARY_RESEARCH.md) for verified
question-sign comparisons, frequency-rating evidence, and the proposed panel answer.

### Suggested signer-separation paragraph

The study uses signer-disjoint partitioning: every signer is assigned exclusively to
training, validation, or testing, and all recordings from that signer remain within
the assigned partition. This evaluates transfer to unseen signers rather than merely
new recordings of familiar individuals, consistent with the unseen-user evaluation
described by Desai et al. (2023).

## 12. References

Publication titles retain their original wording. Descriptive data labels are used
in the ATLAS narrative; bibliographic attribution remains intact.

1. Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). *Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison*. WACV, 1459–1469. [Publisher record](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.html).
2. Woods, L. T., & Rana, Z. A. (2023). *Modelling sign language with encoder-only transformers and human pose estimation keypoint data*. Mathematics, 11(9), 2129. [DOI](https://doi.org/10.3390/math11092129).
3. Desai, A., et al. (2023). *ASL Citizen: A community-sourced dataset for advancing isolated sign language recognition*. NeurIPS Datasets and Benchmarks. [Primary paper](https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf).
4. Apple. (n.d.). *Detecting hand poses with Vision*. [Documentation](https://developer.apple.com/documentation/vision/detecting-hand-poses-with-vision).
5. Jiang, S., Sun, B., Wang, L., Bai, Y., Li, K., & Fu, Y. (2021). *Skeleton aware multi-modal sign language recognition*. CVPR Workshops, 3413–3423. [Publisher record](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html).
6. Faghri, F., et al. (2025). *MobileCLIP2: Improving multi-modal reinforced training*. [Primary preprint](https://arxiv.org/abs/2508.20691).
7. Kim, S., et al. (2022). *Squeezeformer: An efficient Transformer for automatic speech recognition*. NeurIPS, 35. [Publisher record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract.html).
8. Lee, T., Oh, Y., & Lee, K. M. (2023). *Human part-wise 3D motion context learning for sign language recognition*. ICCV, 20740–20750. [Publisher record](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Human_Part-wise_3D_Motion_Context_Learning_for_Sign_Language_Recognition_ICCV_2023_paper.html).
9. Zuo, R., Wei, F., & Mak, B. (2024). *Towards online continuous sign language recognition and translation*. EMNLP, 11050–11067. [DOI](https://doi.org/10.18653/v1/2024.emnlp-main.619).
10. Müller, M., Jiang, Z., Moryossef, A., Rios, A., & Ebling, S. (2023). *Considerations for meaningful sign language machine translation based on glosses*. ACL Short Papers, 682–693. [DOI](https://doi.org/10.18653/v1/2023.acl-short.60).
11. Raffel, C., et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text Transformer*. JMLR, 21(140), 1–67. [Publisher record](https://jmlr.org/papers/v21/20-074.html).
12. Apple. (n.d.). *Core ML*. [Documentation](https://developer.apple.com/documentation/CoreML).
13. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *Bleu: A method for automatic evaluation of machine translation*. ACL, 311–318. [DOI](https://doi.org/10.3115/1073083.1073135).
14. ISO/IEC. (2023). *ISO/IEC 25010:2023: Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model*. [Official standard record](https://www.iso.org/standard/78176.html).

15. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks*. ICML. [Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
16. Liu, Y., et al. (2023). *G-Eval: NLG evaluation using GPT-4 with better human alignment*. EMNLP, 2511–2522. [Publisher record](https://aclanthology.org/2023.emnlp-main.153/).

17. Kamikubo, R., Glasser, A., Lu, A. X., Daumé III, H., Kacorri, H., & Bragg, D. (2025). *Exploring collaboration to center the Deaf community in sign language AI*. ASSETS. [DOI](https://doi.org/10.1145/3663547.3746390).
18. Nielsen, J. (1993). *Response times: The 3 important limits*. Nielsen Norman Group. [Article](https://www.nngroup.com/articles/response-times-3-important-limits/).

19. Moryossef, A., Jiang, Z., Müller, M., Ebling, S., & Goldberg, Y. (2023). *Linguistically motivated sign language segmentation*. Findings of EMNLP, 12703–12724. [Paper](https://aclanthology.org/2023.findings-emnlp.846/).
20. Sign Language Processing. (2026). *Sign Language Segmentation: CNN–Transformer implementation*. [Repository](https://github.com/sign-language-processing/segmentation).

## 13. Code basis and reproduction

The flowcharts were checked against the current iPhone app's `LiveReelApp`,
`LiveReelEngine`, `LiveReelModels`, `LiveReelDecoder`, and `LiveReelStage3` Swift sources.
Python counterparts are [segmental runtime](../../active/v17/segmental_runtime_v17.py)
and [multimodal recognition](../../active/v17/model_unified_multimodal_v17.py).
The system uses the multi-sentence T5-efficient-tiny model.

Rebuild figures from the repository root:

```sh
venv/bin/python docs/capstone_papers/review_assets/build_review_assets.py
```

The figures are reproduced from stored results and training histories. Tables and charts provide the technical basis for manuscript revision.
