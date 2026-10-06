# ATLAS paper writing guide

Last updated: 2026-10-07

## Current author overrides — approved 2026-10-07

These decisions supersede the older iPhone-focused and technical-comparison rules below.
The working manuscript is Google Docs tab `(Oct 6) Revision` (`t.pzqwjycim1ix`);
preserve other tabs, the author's edits, and the deferred presentation.

- Use “mobile application,” “device camera,” “landmark extractor,” and “on-device
  inference” in general explanations. Do not use “Flutter-based mobile application.”
- Introduce Apple Vision and MediaPipe together with citations. Keep named frameworks
  in relevant literature and measured comparisons. Keep Swift/iOS, Kotlin/Android,
  Flutter/Dart and runtime details in hardware/software requirements.
- Keep iPhone 13 attribution with its measurements. Do not add Huawei nova Y70.
- Compare all three recognition stages on the same validation set, without stating
  its size or naming the dataset in comparison passages. Report common-Mac processing
  separately from phone processing. Exclude MediaPipe-only evaluation data; the recorded
  rebuild already uses matching training lists, so do not claim a new training exclusion.
- Retain current FP16 visual/recognition deployment. Compare it with the corresponding
  FP32 visual/recognition configuration; English remains FP32 and outside timing.
  Use the new measured 24.1 versus112.6ms/frame comparison, not the historical28ms as
  a matched baseline. Precision alone does not explain the deployment timing difference.
- Keep recognition conversion accuracy/storage with their measurement scope. Remove
  batch/window timing rows, processor/thread mechanics, language architecture screening
  and estimated language graph timings. Add no technical-notes section to the paper;
  repository evidence remains available.
- Explain signer-disjoint with Desai et al. (2023). Do not extend that claim to the
  complete streaming evaluation or add the rejected familiar/unseen subgroup breakdown.
- Results flow: recognition/common-Mac speed, streaming, English-output quality,
  export/storage, then measured phone execution. Keep diagrams and captions consistent.

## Purpose and authority

Read this guide before discussing or revising the ATLAS manuscript. It records the
author's agreed writing preferences. Update it as the discussion establishes new
decisions; do not turn suggestions into accepted rules without the author's agreement.

Use `PROJECT_GROUND_TRUTH.md`, the latest relevant topic logs, and the current iPhone
implementation to establish technical facts. This guide controls presentation; it
does not change the implementation or replace evidence. Resolve factual differences
before writing claims.

The primary style reference for every chapter is the original manuscript,
[Capstone 2_ ATLAS.md](<Capstone 2_ ATLAS.md>). Preserve its voice, narrative development
and explanatory structure while updating technical facts. The current implementation
and recorded evidence remain the basis for factual corrections.

## Document and audience

- First prepare `ATLAS_MAJOR_CHANGES.md` for quick chapter-by-chapter review and
  `ATLAS_TECHNICAL_REVIEW.md` for comparisons, charts, diagrams, and citation support.
  Edit `Capstone 2_ ATLAS_revised.md` only after the author approves the review.
  Preserve the original manuscript.
- Present the iPhone application as the main system. Describe desktop work as
  development, training, and engineering support where relevant.
- Keep the Magsaysay Future Engineers/Technologists Award audience in mind:
  explain the engineering problem, design choices, contribution, and supported results.
- Keep individual authorship and team responsibility statements out of this paper
  revision; those belong in a separate document.
- Retain the established ASL scope and campus context. Do not expand the language
  coverage through wording.
- Keep individual contribution details in the separate nomination document.

## Writing style and evidence

- Use clear, connected academic language consistent with the original manuscript.
  Explain what a component does, why it matters and how it connects to the surrounding
  discussion. Concision must preserve the development of the argument.
- Keep discussions easy to follow, with a small number of decisions at a time.
- Describe supported functionality and measured findings affirmatively. Omit
  evaluation-status commentary from the manuscript.
- Report only measurements supported by recorded evidence. State the task, evaluation
  conditions, and hardware needed to interpret each result accurately.
- Keep recognition accuracy, sequence error, English-output quality, and execution
  speed distinct. Do not relabel development measurements as independent test results.
- Include human ratings, user outcomes, or community-impact findings only when actual
  records support them. Intended benefits may be described as objectives.
- Keep checkpoint names, local paths, debugging history, rejected experiments, and
  repeated low-level settings out of the main narrative.
- Put necessary architecture details in the technical background or methodology;
  avoid repeating parameter counts and tensor dimensions across chapters.

## Author-approved voice and editing rules

- Use third person throughout the manuscript: “ATLAS,” “the system,” “this study,”
  or “the researchers,” as appropriate. Do not use “I,” “we,” “our,” or address the
  reader as “you.” Preserve the wording of direct quotations.
- Start from the corresponding original passage. Preserve clear phrasing, paragraph
  structure and transitions; revise outdated facts, unclear wording and repetition,
  and add necessary material in the same voice.
- Explain concrete actions and responsibilities. Prefer “The model recognizes signs”
  to “Sign recognition is performed through the utilization of the model.” Use active
  voice where it clarifies responsibility; passive voice remains appropriate when
  the procedure or result is the focus.
- Remove empty introductory phrases such as “It is important to note,” “It is worth
  mentioning,” and “This underscores the significance of.”
- Let features, design reasoning, and measured results establish the contribution.
  Avoid unsupported promotional descriptions such as “groundbreaking,” “revolutionary,”
  or “seamless.”
- Avoid repeated sentence templates, including “not only … but also,” forced
  three-item lists, and paragraph endings that merely repeat the opening.
- Keep technical terms consistent. Do not rename the same component “engine,”
  “framework,” and “mechanism” just to vary the wording.
- Give each paragraph a clear purpose and develop it with relevant detail. Use
  transitions to express real connections. Allow sentence length to vary naturally.
- Present supported capabilities confidently. State scope neutrally and keep necessary
  qualifications in the relevant sections instead of repeating defensive caveats.
- Maintain correct academic grammar. Do not introduce deliberate mistakes, arbitrary
  word substitutions, or blanket punctuation bans to make the prose seem human.

Editorial references: [UNC Writing Center: Style](https://writingcenter.unc.edu/tips-and-tools/style/)
and [Purdue OWL: Concision](https://owl.purdue.edu/owl/general_writing/academic_writing/conciseness/index.html).
[Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
provides observations about formulaic prose, not a test of authorship. These references
inform editing practice; they need not appear in the capstone's literature review.

## Review document links

- [Technical comparisons, charts, flowcharts, and decision rationale](ATLAS_TECHNICAL_REVIEW.md)
- [Major changes for quick review](ATLAS_MAJOR_CHANGES.md)
- [Vocabulary research and proposed rationale](ATLAS_VOCABULARY_RESEARCH.md)

## Agreed terminology

| Use | Meaning and presentation |
| --- | --- |
| Modular architecture | Cooperating functional components with distinct responsibilities; a module need not equal one trained model. |
| Streaming Sign Recognition | Accepted replacement for “Reel activity controller.” |
| Temporal segmentation | Finding candidate sign intervals within the camera stream. |
| Segmental decoding | Selecting and committing a sequence using temporal and recognition evidence. |
| Multimodal recognition | Combining landmark information and cropped-hand image features. |
| Gloss | A written label representing a recognized sign. |
| Gloss-to-English generation | Producing English from the recognized sequence. |

Do not organize the current architecture around Stage 1, Stage 2, and Stage 3. The
former CTC Stage 2 is not the active iPhone recognition path. Internal filenames may
retain historical labels without defining the manuscript's architecture.

## Component names and responsibilities

| Component | Explain its purpose as |
| --- | --- |
| Apple Vision | Detecting hand, face, and upper-body landmarks from camera images. |
| MobileCLIP2 | Extracting visual features from cropped hand images. |
| Squeezeformer | Modeling temporal information for sign recognition. |
| Multimodal recognition head | Producing recognition outputs from combined features. |
| Temporal boundary detection | Estimating the timing of candidate signing segments. |
| Segmental decoder | Selecting sign segments and managing their commitment over time. |
| T5-efficient-tiny | Generating English from recognized gloss sequences. |
| Core ML | Executing the exported models locally on the iPhone. |
| Local text-to-speech | Producing audible output from text. |

Describe recognition outputs as glosses and use **Gloss Management** for the component
that maintains the recognized sequence. Omit fingerspelling, spelling support and
letter-specific models from manuscript prose, objectives, scope, tables and diagrams.
Do not introduce them as separate topics or rename them as additional gloss capabilities.
This presentation rule does not change the implementation or imply that all recognition
outputs use one identical trained head or boundary network.

Describe incremental English generation and Finish according to current iPhone
behavior. T5 encoder and decoder exports are parts of one language model, not two
independently developed translation models.

## Data naming and citations

- Do not insert external dataset or data-source names into the manuscript draft.
  Agree on descriptive terminology with the author first.
- Approved descriptive terms: “isolated-sign data,” “sign-sequence data,” and
  “gloss-to-English pairs.”
- Generic labels must preserve distinctions between data types, evaluation roles,
  and recorded versus generated examples. They must not imply original collection
  or ownership of externally obtained material.
- Keep provenance in the internal evidence records. Discuss how attribution and
  references will be handled before finalizing the submission.
- Retain literature citations and real technology names. Bibliographic titles retain
  their original wording; descriptive data labels apply to the paper's own narrative.

## Signer allocation requested by the author

Record the requested allocation as **15 distinct signers: 10 training, 3 validation,
and 2 testing**. Each person belongs to one split only; all recordings of that person
stay in that split.

This records an author-requested allocation, not a verified description of the
experiments supplying the existing metrics. Their manifest specifies at least 10/3/5
signers per class under its official participant-disjoint split. Preserve those
results' actual provenance; do not attach the requested 10/3/2 allocation to them or
rewrite manifests. Literature supports separation by signer, not the adequacy of
this exact sample size.

## Technical comparisons to discuss

Include these six comparison topics in the technical review:

1. Landmark-only, hand-image-only, and combined recognition.
2. Apple Vision and an alternative landmark extractor.
3. Squeezeformer architectural variants.
4. Earlier streaming recognition and the current segmental approach.
5. English model choices and incremental generation.
6. On-device execution choices affecting speed and model behavior.

Use compact numerical tables with short explanations, readable comparison charts,
and actual recorded training curves. Include the system flow, runtime model decisions,
and engineering selection rationale as separate flowcharts. Use a
simple problem → alternatives → evidence → decision structure. Verify that compared
results share appropriate data, task, metric, and execution conditions. Label component
timing separately from full application timing. Historical results describe their
actual model version, not automatically the current phone implementation.

## Citation support for panel questions

For vocabulary size, signer separation, feature representation, model architecture,
segmentation, language generation, deployment, and evaluation metrics, distinguish
published rationale from local evidence. Give primary-source citations and a short
answer to “Why was this chosen?” A citation must support the specific statement;
do not use literature to imply that a local design choice is universally optimal.

## Review format

Organize major changes by chapter using current statement → proposed change → reason.
Keep the quick review short; place the detailed evidence in its linked companion.
Keep author approval of manuscript edits as the next step after those files are ready.

## Vocabulary framing — 2026-09-30 discussion

- Frame the selection as a prototype of 100 signs with foundational elements.
  Recommended wording: “a 100-sign prototype vocabulary containing foundational signs.”
- Lead with the selected vocabulary and its purpose. State the 100-sign scope neutrally;
  do not repeatedly characterize the system as lacking, restricted, or insufficient.
- Present vocabulary expansion in the recommendations as a future direction. Keep any
  necessary scope qualification concise and in its relevant section. Preserve factual
  accuracy without adding defensive caveats to every contribution or result.
- Explain the inclusion of WHAT, WHERE, WHEN, WHO, WHY, and HOW using introductory
  ASL teaching evidence. These are six question signs, not the entire question system.
- Keep the reason for a fixed size separate from the reason for choosing particular
  signs. Literature does not establish 100 as universally optimal.
- Recommend expanding vocabulary around intended communication situations and Deaf
  users' priorities, supported by suitable recordings and signer-disjoint evaluation.
- The [vocabulary research note](ATLAS_VOCABULARY_RESEARCH.md) documents two specific
  benchmark comparisons. Do not generalize these to “most datasets,” or equate
  question-label inclusion with full question comprehension.

## Technical review presentation — 2026-09-30

- Keep manuscript and technical-review prose academic. Exclude conversation history,
  author requests, approval notes, and phrases such as “the author said.” Record
  editorial decisions in this guide or the project log instead.
- Explain metrics in everyday language. Omit pre-/post-trim hand-detection figures,
  macro-F1, and McNemar statistics from the general-reader review. Retain source records.
- Explain model selection as an accuracy–speed trade-off with the actual measurements.
  Keep desktop base-model timings separate from Core ML phone measurements.
- Retain a short evidence-backed CTC selection rationale; omit CTC numerical results
  and training-overlap discussion from the review. Keep those in the evidence report.
- Describe training accurately: boundary fine-tuning was investigated; the current
  boundary student learned from a frozen teacher; the recognizer was fine-tuned.
- Explain English-quality scoring and cite evaluation research. Attribute automatic
  judgments correctly; a publication does not certify individual generated sentences.
- Name precision and hardware configurations directly. Place Finish controls at sentence
  finalization in diagrams, while showing that English generation also occurs during signing.

## iPhone figures and CTC comparison — 2026-09-30

- Present one iPhone system; refer to it as ATLAS or the system. Keep actual measurement
  hardware in linked measurement notes. Never relabel Mac measurements as iPhone measurements.
- Show percentages without exact validation-clip counts in the reader-facing review
  and charts. Preserve counts, membership, and provenance in underlying reports.
- Integrate CTC selection rationale and boundary learning into the streaming comparison;
  do not give them separate subsections. Name the specific CTC variant behind each result.
- The reader-facing streaming chart compares boundary-guided interval classification
  (39.78% WER) and ATLAS streaming recognition (9.68% WER). Describe the differences
  between these complete configurations. Preserve CTC audits separately.
- Frame the CTC choice around documented live timing, revisable predictions, held-sign
  repetition, and the distinct later CTC validation results. Avoid claiming CTC decoding
  is intrinsically slow or that all CTC models perform poorly.
- Cite Kamikubo et al. (2025) for surveyed Deaf/HoH signers' priority for real-time
  translation. Do not turn this into a universal preference for an exact delay threshold.
- The Live output flow is one path: T5-efficient-tiny incremental English generation →
  Finish button or held-open-palms gesture → finalize English text → English text and
  local speech output. Also show full app navigation with Live, Glosses, Practice, History.
- Rebuild and inspect all PNG/SVG diagrams and charts alongside Markdown edits.

## Standalone academic narrative — approved 2026-09-30

- Introduce the system, component responsibilities, and their connections before results.
  Explain modularity as a division of responsibilities, not a claim that every module
  is one independently trained model.
- Describe configurations by what they do, not by “earlier,” “older,” “updated,” or
  “later.” Readers should not need development history to interpret a comparison.
- Explain boundary-model origin and distillation in the architecture discussion. Name
  and cite the pretrained Sign Language Segmentation teacher, distinguish its2026
  implementation from the related2023paper, and describe the frozen-teacher transfer
  to Apple Vision features. This attribution is explicitly approved.
- Keep comparison methods understandable on their own. Omit confusing development
  milestones from the main review, retaining their records in supporting evidence.
- Link ATLAS_MEASUREMENT_NOTES.md for measurement hardware and provenance. Remove
  “Mac replay” from reader-facing chart labels and the main narrative.

## Figure image formatting — approved 2026-09-30

- Do not embed explanatory subtitles, captions, footnotes, provenance notes, file paths,
  sample-count commentary, or evaluation caveats in any chart, graph, or flowchart image.
- Keep figure titles, axis labels, legends, numerical values, and diagram node/edge labels.
- Put explanations and interpretation in the Markdown document outside the image.
  Preserve that discussion and ATLAS_MEASUREMENT_NOTES.md; this rule concerns image text.
- Apply the rule to every generated PNG and SVG, including training curves, comparison
  charts, and system/application flowcharts. Do not reserve empty space for removed notes.

## Preparation and worked examples — approved 2026-09-30

- Include concise preparation and training methods in the technical review and map
  the proposed additions to chapters in ATLAS_MAJOR_CHANGES.md. Keep both manuscripts
  unchanged until review approval.
- Explain annotation review, manual clipping of selected recordings, cleaning,
  normalization, augmentation and regularization through their actual roles in ATLAS.
  Use short paragraphs and a method–purpose table rather than a general ML tutorial.
- Distinguish manual annotations, automatic trimming, teacher timing predictions and
  automatically aligned training intervals. Do not describe them all as manual labels.
- Use HELLO HOW YOU for the worked sequence diagram. Distinguish recorded model
  intervals from human annotations, recording time from processing time, and measured
  data from schematic application flow. Keep explanations outside figure images.

- Keep the HELLO HOW YOU figure focused on recording time: show the selected intervals
  in a boundary-detector/decoder row, recognition labels and original video stills below.
  Keep feature-extraction lines and the English/Finish flow out of this figure; explain
  those roles in the document. Label stills with their actual recording timestamps.
- Explain the recognizer’s 32-frame landmark input per candidate interval in the methods;
  distinguish resampled model input length from camera frame rate.

## Chapter 1 revision — authorized 2026-09-30

- The author approved starting Chapter 1 with targeted edits in the separate revised
  manuscript. Preserve the original and retain clear existing wording and citations.
- Chapter 1 now covers the iPhone implementation, modular responsibilities, foundational
  vocabulary, gloss management and incremental English/Finish behavior. Other chapters
  remain unchanged for chapter-by-chapter review.
- The author requests 15 signers (10/3/2) as temporary drafting information and intends
  to reconcile it with the reports. Keep it separate from verified results; Chapter 1
  describes signer separation without attaching this allocation to measured results.

## Introduction narrative and detail placement — approved 2026-09-30

- Follow the original paper's introduction pattern: communication context → technological
  advances → remaining practical challenges → ATLAS as the proposed response.
- Preserve the introduction's narrative progression and clear original paragraphs.
  Targeted revision should update outdated facts and improve clarity without reducing
  the introduction to a list of components or a compressed technical summary.
- Retain transitions such as “however” and “despite” when they express a meaningful
  contrast. Direct academic writing still needs connections between ideas; these words
  are not prohibited and should not be removed merely to shorten the text.
- Retain a supported discussion of practical deployment challenges before introducing
  ATLAS. Explain how its design responds to those challenges without implying that
  every existing system has the same limitations. Keep relevant citations beside claims.
- Do not state the 100-sign vocabulary size or list the question signs in the Introduction
  subsection. Place vocabulary size in Purpose and Description and Scope and Limitations;
  discuss individual question signs and selection rationale in the vocabulary discussion.
- Use the original paper's structure as the starting point. Preserve clear explanations,
  transitions and citations; change organization only when it improves the argument.

## Objective structure — approved 2026-09-30

Keep exactly four specific objectives, aligned with the original order and purpose:
(1) develop the recognition and translation pipeline, (2) evaluate model performance,
(3) implement the integrated application, and (4) validate software quality. Update
technical details to match the system without splitting these into additional objectives.

## Original manuscript as the style reference — approved 2026-09-30

- Apply the original manuscript's connected, explanatory style throughout the paper,
  including the technical and methodology chapters. Match each section's purpose while
  keeping its voice and progression close to the original.
- Preserve how ideas build: the context establishes a need, the explanation develops
  it, and the next point follows from that reasoning. Keep the connections between
  problems, design choices and intended benefits explicit.
- Let paragraph length follow the explanation. Fuller paragraphs are appropriate when
  they sustain the narrative; neither brevity nor a fixed word count is the goal.
  Remove redundant wording without removing motivation, reasoning or useful context.
- Retain natural transitions and varied sentence lengths. Avoid flattening passages
  into repeated “ATLAS uses…” or “The system provides…” statements, or replacing
  developed prose with component inventories merely to make it shorter.
- Update facts within the original passage before considering a broader rewrite.
  New material should fit the surrounding argument and wording. Reorganize only where
  needed for accuracy or clarity, and preserve the original direction of the discussion.
- Keep human purpose visible through supported communication needs and intended use.
  Do not invent anecdotes, user experiences or demonstrated benefits to create narrative.
- Before accepting a revision, compare it with the original passage for voice,
  narrative development, transitions and explanatory continuity, as well as technical
  accuracy. Preserving facts alone is not enough if the argument has been flattened.

## Chapter 2 revision — authorized 2026-09-30

- The author approved Chapter 2 revision. Keep its Related Literature/Related Systems
  structure and connected explanations, updating obsolete system descriptions.
- At the author's explicit request, preserve the original vocabulary-subset passage in
  Section 2.1.1 verbatim. This exception also preserves its source names and attribution;
  it does not establish independent verification. Keep that distinction in the internal
  evidence notes and do not alter dataset or evaluation records to match the prose.
- Retain existing bibliography entries and add primary sources for new discussions.
  Chapters 1 and 2 are revised; Chapters 3 and 4 remain for subsequent review.

## Chapter 3 revision — approved 2026-09-30

- Retain the original Current System, Proposed System, Software Requirements, Hardware
  Requirements and Peopleware structure and connected explanation.
- Include the Live system flow and HELLO HOW YOU timeline in Section 3.2; keep figure
  explanations in document text. Maintain sequential figure numbering and the figure list.
- Identify iPhone 13 as the implementation and measurement device, with Mac development
  and training roles distinguished. Do not present the measured device as a verified minimum.
- Keep the original Peopleware groups: Non-Sign Language Users and Developers. Do not
  add a separate Sign Language Users group. Update component descriptions within these groups.
- Chapters 1–3 are revised. Chapter 4 remains for the next chapter review.

## Chapter 4 revision — approved 2026-09-30

- Retain Requirements Analysis and Design; add Data Preparation and Model Development,
  followed by Results and Discussion. Preserve the original connected academic style.
- Include all approved comparison charts and recorded training curves, explaining
  their purpose and results in the document. Keep the Live system flow and worked
  HELLO HOW YOU timeline in Chapter 3.
- Fill the context, data-flow, use-case and application-flow diagrams from current
  iPhone behavior. Exclude unsupported typed-response functionality.
- Preserve the Activity List, Gantt and PERT material unchanged for this revision.
- Chapters 1–4 are now revised in the separate manuscript; the original is unchanged.

- The Sashimi figure follows the supplied five-phase diagram: Planning, Designing,
  Development, Testing and Implementation, with forward and feedback arrows.

- Use the original extracted `source_images/sashimi.png` for the manuscript Sashimi
  figure. Original PERT and seven Gantt images are saved alongside it, unchanged.

- The author subsequently approved inserting the original PERT and all seven Gantt
  images into Chapter 4. Replace their image placeholders; preserve the Activity List
  and surrounding schedule prose.

- Recognition-family tables and charts show only BiLSTM, Temporal CNN, Flat Transformer
  and ATLAS (part-wise + global Squeezeformer). Preserve the full benchmark records.

- Language-model screening shows only the ATLAS tiny T5 architecture and FLAN-T5-base
  (the recorded base-model candidate). Retain the untrained-weight screening context
  and distinction from the installed English model; preserve full benchmark records.

- Explain the complete trained-model → Core ML → iPhone integration path before
  component optimization results. Include model responsibilities and verified export
  precision; distinguish native framework/application work from exported neural models.
  Name MobileCLIP2 in the phone profiles and clarify what each configuration changes.

- Lead deployment discussion with the boundary model and recognizer. Show their
  cooperation with visual features, sequence selection, gloss management and English
  output in a combined workflow. Describe combined timings only within measured scope.

- Use descriptive configuration names in phone timing tables and charts, never unexplained
  A/B/C labels. State the cooperating models and exact precision/processor differences.
  Keep separate boundary, recognizer and MobileCLIP2 settings; define what FP16 models
  includes and distinguish the FP32 English model outside the recognition timing.

- Establish each neural model and its role before Core ML settings: boundary model,
  Squeezeformer recognizer, MobileCLIP2 image encoder and T5-efficient-tiny. Explicitly
  identify their exports; distinguish Apple Vision and application logic. T5 encoder
  and decoder exports are parts of one language model, not separate trained models.

## Conversion results presentation — approved 2026-09-30

- Replace iPhone processing-profile tables/charts with before/after Core ML FP16
  recognition measures for the boundary model and Squeezeformer recognizer.
- Report recognizer accuracy against validation labels; distinguish boundary prediction
  agreement from accuracy against human timing annotations. Do not conflate them.
- Preserve exact checkpoint/export identity and matched inputs. Keep counts in evidence.
- Retain28ms preparation-and-recognition median as one sentence. This supersedes prior
  instructions to include phone processing profiles in the reader-facing documents.

- Include matched before/after model execution times with input unit and hardware.
  Distinguish recognizer batch time from boundary-window time and iPhone frame time.
  Report regressions as measured; do not imply FP16 alone caused runtime differences.

- Include before/after model storage size with an explicit MB definition and file scope.
  Distinguish saved inference weights, Core ML packages, application size and working memory.

- Lead the Core ML rationale with native iPhone integration and local model execution.
  Explain FP16 precision separately; use accuracy, size and time to assess deployment
  consequences rather than imply conversion must improve accuracy or every model’s speed.

## Final review decisions — 2026-09-30

- The supplied Google Docs schedule is the source of truth for the Activity List, Gantt and PERT material. Preserve its dates and distinguish overlapping work from finish-to-start dependencies; do not infer completion from planned dates.
- Keep the original Chapter 2 subset attribution for discussion. It remains unverified and must not be treated as resolved evidence during submission review. See `ATLAS_FINAL_REVIEW.md`.

## Google Docs layout and preservation — 2026-09-30

- Match REC Revision: Times New Roman; table top, header divider and bottom rules
  have equal thickness, with no visible internal or vertical rules.
- Preserve copied footer text and automatic page numbers; refresh contents references
  after layout changes. Use the original sideways Gantt layout.
- Preserve the Activity List as edited directly by the author; this supersedes prior
  instructions to reconcile that list with the supplied schedule.
- Introduce major sections with a short connected overview of their subsections,
  following neighboring sections. Section 4.3 now uses this approach.
