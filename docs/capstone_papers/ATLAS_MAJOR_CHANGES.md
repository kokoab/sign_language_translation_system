# ATLAS — major changes for review

Prepared 2026-09-30. **Chapters 1–4 have been revised in the separate manuscript; the original is unchanged. Chapter 4 retains the Activity List, Gantt and PERT material unchanged.**

Start here for the proposed edits. The [technical companion](ATLAS_TECHNICAL_REVIEW.md)
contains the comparison tables, charts, flowcharts, and citation-supported answers to
design questions. The [writing guide](ATLAS_WRITING_GUIDE.md) records agreed rules.

## Main direction

Present ATLAS as an **iPhone application for recognizing a fixed 100-sign ASL vocabulary
and generating English text and speech**. Explain its cooperating components with
their real names and purposes. Use “Streaming Sign Recognition” for the current
segmentation, recognition, and word-commitment process.

Keep the existing title:

**ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation**

## Chapter-by-chapter changes

| Location | Current statement or presentation | Recommended change | Reason |
| --- | --- | --- | --- |
| Front matter | Contents include a performance subsection absent from the body; diagrams appear as placeholders. | Add the selected evidence and diagrams, then regenerate contents and figure/table lists. | Align navigation with the document. |
| Chapter 1: introduction | Older Reel controller and supplemental mouth extraction appear in the system overview. | Describe Apple Vision, MobileCLIP2, multimodal Squeezeformer, temporal segmentation, segmental decoding, and T5-efficient-tiny. | Match the active iPhone path. |
| Chapter 1: purpose | Parameter counts and low-level window settings dominate the description. | Lead with camera input, sign recognition, English output, and local operation. Move implementation details to Chapter 3. | Make the contribution clear to a general technical reader. |
| Chapter 1: objectives | Objectives repeat the older controller and 0.4-second gesture. | Organize objectives around visual extraction, streaming recognition, English generation, and application evaluation. | Keep objectives stable when internal settings change. |
| Chapter 1: scope | Mac-centered scope; English generation only after Finish. | Make iPhone the main implementation; describe gloss management and incremental English generation. | Reflect current functionality while retaining the 100-sign lexical scope. |
| Chapter 1: vocabulary | Broad claims that 100 signs guarantee practical accuracy and direct comparability. | Describe a 100-sign prototype vocabulary containing foundational signs, including six question signs. Place vocabulary expansion in recommendations. | Explain the selection’s purpose and contribution with a clear, neutral scope. See [vocabulary research](ATLAS_VOCABULARY_RESEARCH.md). |
| Chapter 2: literature | Technology descriptions repeat the implementation and mix external benchmark claims with local results. | Retain citations; group literature around the decisions it supports. Keep local evidence in the technical/results discussion. | Explain why the approach was selected. |
| Chapter 2: learning concepts | Preparation and training concepts need a clear connection to the system. | Briefly explain normalization, augmentation and regularization through their roles in ATLAS. | Give readers the background needed to understand the methods. |
| Chapter 2: sequence recognition | Historical Stage and Reel terminology. | Explain temporal segmentation and segmental decoding; keep CTC as related work where useful. | The active app does not use the former CTC Stage 2. |
| Chapter 3: architecture | Five modules described with older desktop behavior. | Use Visual Extraction, Streaming Sign Recognition, Gloss Management, and English Generation and Speech Output. | Describe functional responsibilities rather than model-file count. |
| Chapter 3: recognition | Detailed separate-model descriptions could obscure shared processing. | Present one integrated recognition component with landmark and hand-image evidence. Explain shared features and combined recognition outputs. | Keep the requested level of abstraction without misrepresenting implementation. |
| Chapter 3: features | Raw X/Y/Z wording can imply measured 3D coordinates. | Explain the 32-frame input per candidate interval, its 61 landmark points, and body-relative X/Y, a relative scale-based depth proxy, presence and confidence. Distinguish input length from camera frame rate. | Match the actual input representation. |
| Chapter 3: component interaction | Architecture diagrams identify components but do not follow one signed sequence. | Add a HELLO HOW YOU timeline with boundary-guided intervals, recognition labels and original video stills against recording time. Explain feature extraction and English output in the accompanying text. | Show how the cooperating components process the same clip. |
| Chapter 3: deployment | Python/OpenCV-centered software and Mac-only hardware tables. | Distinguish development tools from the Swift/Core ML iPhone runtime and local speech output. | Explain how the delivered app executes. |
| Chapter 3: completion | 0.4-second held palms; whole-buffer translation only. | Describe the one-second open-palms control, Finish button, incremental sentence locking, and finalization of remaining text. | Match current controls and translation behavior. |
| Chapter 4: methodology | Data preparation, splitting, model comparison, and app evaluation are not clearly separated. | Use distinct subsections for data roles, signer separation, recognition development, streaming integration, English generation, and evaluation methods. | Make the study reproducible and readable. |
| Chapter 4: preparation and annotation | Preparation needs a concrete account of how examples and labels were constructed. | Describe label checks, annotation review, manual clipping of selected recordings, automatic activity trimming and frame sampling. Distinguish manually annotated boundaries from automatically aligned intervals. | Explain the preparation work and preserve the meaning of the labels. |
| Chapter 4: normalization and augmentation | Input processing and training variation need distinct explanations. | Describe body-relative coordinates, consistent scale and missing-landmark indicators; then explain training variations and shifts to candidate interval boundaries. | Connect each method to the variation it addresses. |
| Chapter 4: training and regularization | Training safeguards are not presented together. | Summarize dropout, weight decay, isolated-sign retention during interval adaptation and validation-based model selection in a method–purpose table. | Explain how training balances adaptation with retained recognition. |
| Chapter 4: comparisons | Evidence is scattered or absent. | Add the six comparison groups in the companion, each with a brief interpretation. | Tie decisions to measurements. |
| Chapter 4: diagrams | Narrative says completed video is captured; typed response appears without a matching verified flow. | Replace with current camera-stream, runtime-decision, and engineering-selection diagrams. | Align diagrams with the app. |
| Chapter 4: schedule | Activity List, Gantt and PERT material. | Retained unchanged for this revision, as agreed. | Keep scheduling changes outside the approved Chapter 4 edit. |
| References | Relevant citations exist but do not consistently support the nearby decision. | Add the primary sources in the companion and place each citation beside its supported claim. | Make the rationale easy to defend. |

## Data terminology and signer separation

Use **isolated-sign data**, **sign-sequence data**, and **gloss-to-English pairs** in the
manuscript narrative. Retain literature citations and original publication titles.

Your requested allocation is **15 signers: 10 training, 3 validation, and 2 testing**,
with each person assigned exclusively to one split. The writing guide records it as
an author-requested allocation. Existing numerical results retain their documented
split conditions; their manifest specifies a different allocation. No results are
relabelled as measurements from the requested 15-signer arrangement.

## Recommended evidence to include

| Question answered | Proposed evidence |
| --- | --- |
| How were examples prepared and models trained? | Annotation and clipping discussion, followed by a compact preparation and training table. |
| Why Apple Vision? | Matched extraction speed and downstream recognition comparison. |
| Why combine landmarks and hand images? | Same-set single-input versus combined recognition results. |
| Why Squeezeformer and part-wise processing? | Architecture variants and matched family comparison, with classifier timing. |
| Why the current streaming approach? | Matched configuration comparison and the HELLO HOW YOU worked timeline. |
| Why tiny T5 and incremental English? | Generated-session English quality, candidate execution screening, and recorded training loss. |
| Does Core ML conversion preserve recognition? | Matched recognizer validation accuracy and boundary-state agreement; one device timing statement. |

Chapter 4 now includes all approved comparison charts and training curves, with
interpretation in the surrounding text. Chapter 3 retains the Live system flow and
HELLO HOW YOU timeline. Chapter 4 adds context, data-flow and use-case diagrams,
the application flow, runtime decisions and engineering selection rationale.

## Content to keep concise

Explain purposes and decisions in the main text. Omit individual contribution details,
checkpoint filenames, debugging chronology, repeated dimensions, and exhaustive
experiment history. Preserve signer relationships, measurement hardware, metric definitions, and timing scope
in the appropriate discussion and linked measurement notes. Keep detailed counts in the
evidence records and explanatory text outside figure images.

**Next author decision:** review Chapter 4 in `Capstone 2_ ATLAS_revised.md`, particularly
the preparation methods and interpretation of the comparisons. The Activity List,
Gantt and PERT material remains outside this revision.

Chapter 4 now includes a separate model-to-iPhone deployment subsection, component
export/precision table and workflow diagram before the measured processing profiles.
The phone-profile table/chart has been replaced by matched recognition conversion
measures;28ms remains a single preparation-and-recognition timing statement.
