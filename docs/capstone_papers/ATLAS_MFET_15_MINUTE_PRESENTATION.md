# ATLAS — 15-Minute MFET Presentation

Updated September 30, 2026. Audience: engineers, scientists, and a mixed academic panel. The 13-slide allocation includes the immediate recorded demonstration and excludes Q&A. Speaking times are rehearsal targets, not a measured delivery duration.

Primary sources: [Revised manuscript](<Capstone 2_ ATLAS_revised.md>) and [measurement notes](ATLAS_MEASUREMENT_NOTES.md). The contemporary NCDA statistic is a separately cited contextual addition. Personal motivation and future collaboration plans come from the presenter.

## Presentation direction

Follow the paper's research progression: **introduction → research gap → significance → objectives → scope and limitations → methodology → results and discussion → conclusion and recommendations**. Natural slide titles carry this structure without separate section-divider slides.

Communication accessibility provides the motivation. **Computational efficiency and local mobile deployment are the main technical gap**, supported by connected-signing responsiveness and understandable English output. Significance appears early to explain why the investigation matters before stating what the study aims to achieve.

**Central research question:** How can ATLAS balance recognition accuracy, computational efficiency, and responsive English output in an on-device sign-language application?

The opening immediately reveals the actual recorded ATLAS output. Methodology starts with the system overview, then develops data preparation, model development, streaming decisions and mobile integration. Numerical outcomes follow those methods. The 28 ms/frame finding answers the deployment challenge rather than appearing as an isolated speed claim.

Within the capstone team, the presenter reports responsibility for AI/ML, the pipeline, software and mobile implementation. Express that responsibility naturally through engineering decisions. Do not allocate a separate biography slide; journalism credentials and NSPC achievement remain in the biodata.

Reference deck reviewed: `/Users/frnzlo/Downloads/Capstone Presentation.pdf` (27 pages). Preserve its research-section progression while replacing historical model, platform and evaluation claims with the revised manuscript's content.

## Timing and section overview

| Slide | Research section | Slide title | Time |
| --- | --- | --- | --- |
| 1 | Opening and introduction | A message you can see | 0:00–1:15 |
| 2 | Problem and context | Why communication accessibility matters | 1:15–2:05 |
| 3 | Research gap | Research gap: accurate recognition within mobile computing limits | 2:05–3:25 |
| 4 | Significance of the study | Why portable communication support matters | 3:25–4:25 |
| 5 | Objectives | What ATLAS aims to achieve | 4:25–5:15 |
| 6 | Scope and limitations | What this prototype covers | 5:15–6:20 |
| 7 | Methodology: system overview | How ATLAS turns movement into a message | 6:20–7:40 |
| 8 | Methodology: preparation and development | Preparing the data and developing the models | 7:40–9:00 |
| 9 | Methodology: streaming development | Separating connected signs | 9:00–10:20 |
| 10 | Methodology: implementation | From trained components to an iPhone workflow | 10:20–11:25 |
| 11 | Results and discussion | What recognition and English evaluations showed | 11:25–12:45 |
| 12 | Results and discussion | Measured execution on the iPhone | 12:45–14:00 |
| 13 | Conclusion and recommendations | A little better than yesterday | 14:00–15:00 |

## Slide 1 — A message you can see | Purpose: Opening and introduction

**Time:** 0:00–1:15 · 75 seconds

### On the slide

Initially, only the signing video. After the replay, reveal:

**ATLAS**  
*Sign recognition and English generation on an iPhone*

### Visual

Play the greeting without captions, then replay the same recording with the actual application output.

### Speaking script

*[Play the signing video. Pause briefly.]*

“Were you able to understand what I said?

I signed, ‘Hello, good morning. How are you?’

At school, I would see Deaf students signing with one another. I could see a conversation, but I could not understand it. That made me curious about how technology could help bridge a language gap.

Now, let me show you the same greeting through the system I developed.”

*[Replay with actual recognition and English output.]*

“This is ATLAS: an iPhone research prototype that recognizes supported ASL signs and generates English text and speech.”

### Recording note

Add “Recorded demonstration” discreetly. Preserve the application's actual output and timing; record the greeting before finalizing the accompanying narration.

## Slide 2 — Why communication accessibility matters | Purpose: Problem and context

**Time:** 1:15–2:05 · 50 seconds

### On the slide

**160,589**  
Registered Deaf and hard-of-hearing persons  
*Philippines · DOH data through NCDA · August 3, 2026*

**A shared space does not always mean a shared language.**

### Visual

One large statistic and source footer, followed by a simple illustration of a signer and a non-signing recipient. Avoid imagery portraying Deaf people as helpless.

### Speaking script

“The National Council on Disability Affairs reports 160,589 registered Deaf and hard-of-hearing persons, using Department of Health data as of August 2026.

This is a registry count, rather than the total number of people with hearing difficulty. It also does not tell us how many use ASL.

For this project, the problem is the communication gap when a signer and a non-signing person do not share a language.

Written or typed exchanges can help. ATLAS investigates another form of support: connecting supported signed input with English text and speech through a portable application.”

### Sources and context — presenter notes

- [NCDA official dashboard](https://ncda.gov.ph/): DOH data as of August 3, 2026; Deaf and Hard-of-Hearing category total 160,589 (80,985 female; 79,604 male). Retrieved from the indexed official page September 30, 2026; direct page retrieval was unavailable. Recheck the live dashboard before submission because it updates.
- This is a registered-disability count, not a population prevalence estimate, not restricted to ages 5+, and not an ASL-user count. Do not compare it with the 2020 census as a change in prevalence.
- Optional educational context, not a headline: [EDCOM 2, November 20, 2025](https://edcom2.gov.ph/5-million-filipino-children-with-disabilities-remain-underserved-edcom-2-study/) reports that 60% of enrolled learners with disabilities lack school special-needs education resources. This covers disabilities generally. No verified Deaf-specific dropout-cause percentage is available for this presentation.

## Slide 3 — Research gap: accurate recognition within mobile computing limits | Purpose: Research gap

**Time:** 2:05–3:25 · 80 seconds

### On the slide

**The challenge: accurate sign recognition and English generation within mobile computing limits.**

**Main technical gap — Computational cost and local deployment**  
Preserve recognition quality while keeping processing and storage manageable on a portable device.

**Supporting gaps**

- **Connected signing:** locate signs and stabilize output as signing continues.
- **English output:** turn accepted sign labels into understandable text and speech.

Small footer: manuscript §§2.1.2, 2.1.4–2.1.5, 2.2.5; Vasu et al., 2024; Faghri et al., 2025; Moryossef et al., 2023.

### Visual

Give the computational-cost gap the largest area: a video/model feeding into a phone, with processing and storage constraints alongside it. Under that, use two smaller rows for connected-sign timing and gloss-to-English output. Repeat this hierarchy on Slide 12. Avoid treating all RGB or Transformer models as equally costly.

### Speaking script

“The main technical gap is the difficulty of bringing accurate recognition and English generation into a locally running mobile application.

Processing visual appearance and movement over time requires computation. A portable device must manage that processing alongside model storage and the rest of the application.

Reducing computation is only useful if the system preserves the information needed to distinguish signs. Improving recognition also has limited practical value if the interaction becomes too slow.

Connected signing adds a second challenge: the application must locate signs and decide when a prediction is stable enough to accept.

Finally, a sequence of recognized labels still needs to become understandable English.

These challenges establish the question behind ATLAS: how can recognition accuracy, computational efficiency, and responsive output be brought together on an iPhone?”

### Evidence and claim boundaries — presenter notes

- **Computational efficiency is the primary technical gap.** “High computational cost,” “hard to deploy locally,” and “not lightweight” describe related aspects of this deployment challenge, rather than three separate research gaps.
- Manuscript §2.1.2 motivates structured landmark representations; §2.1.4 discusses temporal modeling and computational trade-offs; §2.1.5 discusses focused hand crops and efficient visual encoders. These sections provide the rationale; ATLAS's measurements assess its implementation.
- [Vasu et al., 2024, MobileCLIP](https://arxiv.org/abs/2311.17049) and [Faghri et al., 2025, MobileCLIP2](https://openreview.net/forum?id=WeF9zolng8) are the efficient-encoder references cited in the manuscript. ATLAS uses MobileCLIP2 for hand-crop features, not an RGB-free pipeline.
- Large full-frame RGB/video models and particular Vision Transformer configurations are relevant computational-cost examples when supported by their specific architectures and measurements. Do not claim every RGB or Transformer system is heavy or that ATLAS outperforms them all. The local family comparison actually found the flat Transformer faster than the selected Squeezeformer; selection prioritized accuracy within a workable processing budget.
- [Moryossef et al., 2023](https://aclanthology.org/2023.findings-emnlp.846/) supports the segmentation problem. ATLAS's boundary teacher uses a separately cited 2026 implementation. The reported CTC waiting/error experience concerns the tested ATLAS implementation, not a universal CTC limitation.
- [Camgoz et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.html) already combines recognition and translation. The supporting gap is the quality and responsiveness of the prototype's English output, not the absence of translation research.
- [Kamikubo et al., 2025](https://doi.org/10.1145/3663547.3746390) supports attention to Deaf participation and desired translation functionality. It does not validate ATLAS or prescribe a universal latency threshold.
- This is a scoped engineering research problem grounded in the reviewed literature. No matched full-frame RGB/ViT deployment comparison is claimed by the 28 ms result.

## Slide 4 — Why portable communication support matters | Purpose: Significance of the study

**Time:** 3:25–4:25 · 60 seconds

### On the slide

**For non-signing recipients**  
Support understanding of messages within the ASL prototype vocabulary.

**For the interaction**  
Bring camera input, English text, and local speech together on a portable device.

**For engineering research**  
Examine the balance between recognition quality, streaming behavior, and device execution.

### Visual

Three connected panels: recipient → portable interaction → measured engineering contribution. Use a campus image only as context; keep the emphasis on communication and the study.

### Speaking script

“The significance of the study begins with the intended recipient.

The paper identifies non-signing faculty, staff, and peers at Leyte Normal University as people the application aims to assist in understanding supported ASL messages.

Bringing recognition, English text, and speech together on a phone gives the study a practical direction. Local processing connects those functions within the device rather than requiring a cloud recognition service.

The study also contributes an engineering investigation: how visual information, temporal decisions, language generation, and mobile execution can work together, and how each part should be evaluated.

These are the intended benefits and technical contributions. Whether the application improves everyday communication for users remains a question for further collaboration and evaluation.”

### Manuscript basis — presenter notes

Synthesized from Chapter 1 §§1.1–1.2 and Chapter 3 §3.2. The paper has no separate significance subsection; do not imply it reports measured learning gains, user satisfaction, reduced dropout, or community outcomes. Philippine impact beyond the present ASL scope remains prospective.

## Slide 5 — What ATLAS aims to achieve | Purpose: Objectives

**Time:** 4:25–5:15 · 50 seconds

### On the slide

**Design, implement, and evaluate a 100-sign ASL-to-English iPhone prototype.**

1. Develop the multimodal recognition and English-generation pipeline.
2. Evaluate recognition, sequence errors, and English output.
3. Integrate streaming recognition, message completion, and local speech on iPhone.
4. Assess software quality and behavior through the stated evaluation framework.

### Visual

Four short objective cards. Reveal one at a time; reserve model names for the system overview. Add a small footer: “Manuscript §1.3.”

### Speaking script

“The paper translates that purpose into four objectives.

First, develop the multimodal recognition and English-generation pipeline.

Second, evaluate the recognition and language components using the measures appropriate to their tasks, including recognition accuracy, word error rate, and BLEU.

Third, implement the trained components in an iPhone application with streaming recognition, message completion, and local speech.

Fourth, assess software quality and behavior through the stated ISO/IEC 25010 framework and black-box and white-box testing procedures.

The results I will present focus on the recorded technical evaluations; I will distinguish those from any broader software-quality or user assessment still to be completed.”

### Manuscript basis — presenter notes

Faithful paraphrase of §1.3, including its software-quality objective. Use ISO/IEC 25010:2023. The cited manuscript material does not supply a completed aggregate ISO/user-rating result; do not imply certification or invent a score. Objective language is not evidence of completed validation.

## Slide 6 — What this prototype covers | Purpose: Scope and limitations

**Time:** 5:15–6:20 · 65 seconds

### On the slide

| Within the study | Boundaries |
| --- | --- |
| 100-sign ASL prototype vocabulary | No established FSL or unrestricted ASL coverage |
| English text and local speech on iPhone | Output depends on accepted glosses |
| Live, Glosses, Practice, and History | Tracking depends on lighting, framing, and visibility |
| Hand, face, and upper-body visual information | No general lip-reading or complete non-manual grammar interpretation |

### Visual

Actual app screenshot beside a compact “Coverage / Conditions” panel. Avoid a dense list of caveats; the table specifies content that can be simplified visually.

### Speaking script

“ATLAS has a defined scope: a 100-sign ASL prototype vocabulary, English text and local speech, and execution on an iPhone.

The application includes Live recognition, a Glosses reference, Practice activities, and History.

The selected vocabulary provides a controlled basis for developing and evaluating the system. It does not establish Filipino Sign Language recognition or unrestricted ASL translation.

Capture conditions also matter. Adequate lighting, suitable camera framing, and visible hands and upper body support tracking. Occlusion, rapid motion, and unsuitable angles can affect the input.

Facial landmarks are included, but general lip-reading and complete interpretation of non-manual grammar are outside the defined functionality.

These boundaries determine how the methods and results should be interpreted.”

### Manuscript basis — presenter notes

Based on §1.4. Isolated-sign, recorded-sequence, English-text and device profiles have distinct evaluation conditions. Retain those distinctions on results slides. ASL resource availability is the presenter-confirmed selection rationale; do not assert Philippine ASL prevalence.

## Slide 7 — How ATLAS turns movement into a message | Purpose: Methodology: system overview

**Time:** 6:20–7:40 · 80 seconds

### On the slide

**Camera → Visual evidence → Sign intervals → Recognized signs → English + speech**

Under “Visual evidence”:

- **Movement:** Apple Vision landmarks
- **Appearance:** MobileCLIP2 hand-image features

### Visual

A simple pipeline with two visual branches merging into recognition. Reveal it progressively.

### Speaking script

“ATLAS begins with two kinds of visual evidence.

Apple Vision detects landmarks on the hands, face, and upper body. These points describe positions and movement over time.

The application also prepares cropped hand images. MobileCLIP2 extracts appearance features from those images, preserving information about handshape that coordinates alone may miss.

The next challenge is timing. The boundary model estimates where candidate signs occur within the stream.

A Squeezeformer-based recognizer then examines those candidate intervals. Its temporal processing combines nearby movement patterns with relationships across the sequence of frames.

The segmental decoder uses timing and recognition evidence to select which signs to accept. Those accepted signs appear as glosses—written labels representing recognized signs.

Finally, a fine-tuned T5 model generates English from the accumulated glosses. English preparation happens as signing continues. The Finish control completes the remaining message and triggers local speech.

Each component has a specific responsibility, and the application coordinates their outputs.”

## Slide 8 — Preparing the data and developing the models | Purpose: Methodology: data preparation and model development

**Time:** 7:40–9:00 · 80 seconds

### On the slide

**Prepare → Separate → Compare → Integrate → Verify**

Three data roles:

**Individual signs · Sign sequences · Gloss-to-English pairs**

### Visual

A compact experimental workflow. Use one small real training curve as supporting evidence.

### Speaking script

“Development began with preparing the inputs for each task.

Individual-sign recordings supplied vocabulary examples. Sign-sequence recordings supported connected input. Gloss-to-English pairs supplied language-generation examples.

Preparation included label checks, annotation review, selected manual clipping, activity trimming, and consistent input sampling.

Coordinates were normalized relative to the body while preserving image proportions. Augmentation introduced controlled variation in position, scale, timing, and visibility.

For isolated signs, a person's recordings stayed within one assigned split. That separation helps assess recognition across people rather than repeated exposure to the same signer.

I compared landmark extractors, visual input combinations, and temporal model families under their documented experimental conditions. Training and validation records guided selection.

This connected data preparation with model development. The next step was to adapt recognition to the variable intervals produced by a streaming application.”

## Slide 9 — Separating connected signs | Purpose: Methodology: connected-signing development

**Time:** 9:00–10:20 · 80 seconds

### On the slide

**Where does one sign end—and the next begin?**

Earlier implementation:

**Waiting for input → Changing predictions → Delayed output**

Current approach:

**Estimate boundaries → Recognize intervals → Commit signs**

### Visual

A signing timeline with candidate intervals and accepted labels. Use the manuscript's HELLO–HOW–YOU example.

### Speaking script

“The hardest problem I encountered was separating connected signs.

When a person signs, the camera also captures movement between signs, held positions, and pauses. The system has to determine which portions belong to meaningful signs.

My earlier implementation used Connectionist Temporal Classification, or CTC. In that tested implementation, I encountered substitutions, missing signs, and unstable output as additional input arrived.

Its first full-window prediction also required approximately 1.07 seconds of input before processing. Speech depended on obtaining a stable sequence, which affected the interaction.

This was a limitation of the implementation I tested; it is not a claim that every CTC system behaves that way.

I moved toward an approach that explicitly estimates sign boundaries, recognizes candidate intervals, and decides when a sign is stable enough to accept.

The key engineering lesson was that identifying a sign and deciding when to commit it are connected but distinct problems. Both affect what the user experiences.”

## Slide 10 — From trained components to an iPhone workflow | Purpose: Methodology: mobile implementation

**Time:** 10:20–11:25 · 65 seconds

### On the slide

**Train and adapt → Export → Integrate → Verify**

- Core ML execution of the neural components.
- FP16 visual/recognition exports; FP32 English model.
- Incremental English preparation during signing.
- Finish button or held-open-palms completion, then local speech.

### Visual

A deployment flow showing trained models entering the app, then camera → accepted glosses → English → speech. Place Finish at message completion, not before all English generation.

### Speaking script

“I then brought the trained components into the iPhone application.

Core ML exports provide the execution path for the boundary model, recognizer, hand-image encoder, and English model. Apple Vision supplies landmarks through the existing framework.

The visual and recognition exports use FP16 precision, while the installed English model retains FP32. These are component-specific choices, rather than one precision setting for the entire application.

The application coordinates camera capture, model inputs, accepted glosses, English updates, and the interface.

English preparation occurs as signing continues. Selecting Finish, or holding both open palms for one second, finalizes the remaining message for text and local speech.

After integration, conversion checks and device profiling examine whether recognition behavior is retained and what the execution path costs.”

### Manuscript basis — presenter notes

Based on §§4.3.4 and 4.4.5–4.4.6. Distinguish model export from native camera, interface, decoding and speech logic. The timing results follow in Slide 12; no unsupported speedup is attributed to a single change.

## Slide 11 — What recognition and English evaluations showed | Purpose: Results and discussion: recognition and English

**Time:** 11:25–12:45 · 80 seconds

### On the slide

| Question | Result |
| --- | ---: |
| Can it identify individual signs? | **96.30%** validation Top-1* |
| How many sequence errors occur? | **9.68% WER** in recorded development replay |
| How well does it generate English? | **60% fully correct** automatic rating |

*Combined-input comparison; separately trained configurations underpin the different evaluations.*

### Visual

Three distinct cards. Keep the evaluation labels visible.

### Speaking script

“I evaluated recognition, streaming output, and English generation separately.

In the visual-input comparison, combining motion and hand appearance reached 96.30 percent validation Top-1 accuracy, compared with 95.50 percent for landmarks alone. This measures whether the model's first prediction matches the sign label.

For connected signing, word error rate counts substituted, missing, and additional signs.

On the recorded development comparison, the earlier boundary-guided configuration had 39.78 percent word error rate. The ATLAS streaming configuration achieved 9.68 percent.

That is a comparison of complete configurations on recorded inputs, including familiar-signer examples. It is not a live-phone accuracy measurement.

For English generation, 60 percent of generated sessions were automatically judged fully correct. Those judgments used generated references and a language-model evaluator.

These results show progress in different components. They also identify English generation and broader evaluation as continuing priorities.”

### Evidence notes

The 39.78% baseline is **boundary-guided interval classification**, not CTC. Do not label this chart “CTC versus ATLAS.”

The streaming comparison covers 72 recordings and 186 reference signs: 60 familiar-signer sequences with shared phrase templates and 12 unseen-signer sequences. The ATLAS result is native Swift recorded-input replay on a Mac using the iPhone implementation and settings. The full configuration changes between the two results.

English assessment uses 300 generated sessions. DeepSeek supplied references and automatic judgments; this is not human-assessed signing accuracy. See [measurement notes](ATLAS_MEASUREMENT_NOTES.md).

## Slide 12 — Measured execution on the iPhone | Purpose: Results and discussion: mobile performance

**Time:** 12:45–14:00 · 75 seconds

### On the slide

**28 ms per processed frame**  
*Median preparation + recognition · iPhone 13 · recorded-input profile*

**Core ML execution · FP16 recognition components · Incremental English generation**

Recognizer conversion check:

**95.24% → 94.97% validation Top-1**

Recognizer storage:

**49.62 MB → 25.51 MB**

### Visual

Actual phone interface with a large 28 ms callout. Below it, show the recognizer's storage reduction and matched validation accuracy before and after conversion. Keep the measurement scope directly below the headline.

### Speaking script

“The deployment results return to the main technical gap: running the system locally while balancing accuracy, processing time, and storage.

On an iPhone 13, preparation and recognition took a median of 28 milliseconds per processed frame in a recorded-input profile.

For the recognizer, saved storage changed from 49.62 megabytes of FP32 inference weights to a 25.51-megabyte FP16 Core ML package—approximately half the size.

On matched validation inputs, Top-1 accuracy changed from 95.24 to 94.97 percent. That check examines the recognizer's behavior after conversion.

These are different measures. The storage figure is not the total application size or working memory. The 28-millisecond profile is not the entire delay from signing to English speech.

Together, the results provide evidence about local execution while showing the recognition trade-off. They connect the optimization decisions to the mobile-deployment problem introduced at the start.”

### Optimization and speed evidence notes

- Say **per processed frame**, not per clip. A sign spans multiple frames; 28 ms is not the time to recognize a complete sign or translate a sentence.
- The device profile used 226 recorded-input frames per configuration on a physical iPhone 13. It does not establish sustained live-camera throughput, thermal behavior, or end-to-end latency.
- Do not convert 28 ms into a claim that the full app runs at 36 FPS; the measured processing scope and scheduling do not establish that throughput.
- Core ML provides native execution; FP16 is a separate precision choice. The profile compares configurations already using Core ML, not Core ML versus no Core ML.
- The recognizer storage comparison is FP32 inference weights versus a complete FP16 Core ML package, in decimal MB. It is not total app size or working memory.
- The conversion accuracy check holds landmark inputs and cached hand-image features constant, isolating recognizer conversion. It does not prove accuracy retention for every changed component or the entire live system.
- Incremental English generation addresses work remaining at Finish; do not attribute its benefit to the 28 ms recognition measurement.
- Describe these as supported implementation choices and measured outcomes; do not assign the entire speed result to one optimization without a matched comparison.

Source: [measurement notes](ATLAS_MEASUREMENT_NOTES.md), sections “iPhone execution and English assessment,” and revised manuscript sections 4.4.4–4.4.6.

## Slide 13 — A little better than yesterday | Purpose: Conclusion and recommendations

**Time:** 14:00–15:00 · 60 seconds

### On the slide

**A locally running, 100-sign ASL prototype**

**Next: expand vocabulary and collaborate with Deaf users.**

*“Technology should help make lives a little better than yesterday.”*

### Visual

Return to the actual opening greeting and English output. Place the two next steps beneath the phone image. Close with the presenter's statement; no separate thank-you slide.

### Speaking script

“At the beginning, I shared a simple greeting. Understanding it depended on whether we shared a language.

ATLAS brings a 100-sign ASL vocabulary, streaming recognition, and English text and speech together in a local iPhone workflow.

Its contribution is the developed system and the evidence connecting recognition quality, sequence behavior, and deployment choices. Broader communication benefits still need evaluation with users.

My next priorities are to expand the vocabulary and collaborate with Deaf users, so that development responds to the messages and interactions that matter to them. Future FSL adaptation would require appropriate data, expertise, and evaluation.

That is the direction I want to keep working toward: using technology to make lives a little better than yesterday.

Thank you.”

### Manuscript basis — presenter notes

Current scope and technical conclusions follow Chapters 1 and 4. Vocabulary expansion follows §1.4. Collaboration and the closing statement are the presenter's stated next steps, not completed study outcomes.

## Judging-criteria alignment

| Criterion | Weight | Where the presentation addresses it |
| --- | ---: | --- |
| Person | 20% | Motivation in Slide 1; responsibility and engineering judgment in Slides 8–10; direction in Slide 13. |
| Methodology | 20% | Paper-based objectives and boundaries in Slides 5–6; preparation, development, implementation and evaluation in Slides 7–12. |
| Innovation / originality / creativity | 15% | Computational trade-off in Slide 3, adapted workflow and streaming decisions in Slides 7–10, measured outcomes in Slides 11–12. |
| Quality of write-up | 15% | Clear research progression, manuscript references, traceable comparisons and correctly labeled evidence throughout. |
| Contribution / Philippine impact | 30% | Context and early significance in Slides 2–4; evaluated implementation and realistic next steps in Slides 11–13. |

## Technical details to include or move to backup

Keep in the main presentation:

- Boundary detection versus sign recognition.
- One visual-input comparison.
- Sequence results with their evaluation context.
- Mobile conversion evidence.
- The actual recorded demonstration.

Move to backup slides:

- Tensor dimensions.
- Optimizer settings.
- Complete architecture comparison tables.
- Detailed CTC mechanics.
- Full training curves.

Use the manuscript's existing figures as source material, simplifying their labels for projection. Keep numerical distinctions consistent with the revised manuscript and measurement notes. Do not combine validation recognition, recorded-development sequence results, automatic English judgments, and device processing time into a single system-accuracy claim.

## Rehearsal and remaining preparation

- Record “Hello, good morning. How are you?” and capture the actual system output.
- Insert actual application screenshots and development visuals.
- Keep the opening video and replay within Slide 1's 75-second allocation.
- Rehearse the full script with the recording and slide transitions; the time allocations are targets, not a measured delivery duration.
- Adjust pacing and pauses after rehearsal while preserving the 15-minute total.
- Keep future collaboration, vocabulary expansion, and potential FSL adaptation clearly identified as future work.

## Relationship to the earlier deck

The revised sequence retains the earlier deck's introduction, research gap, significance, objectives, scope and methodology progression. It adds current measured results and discussion, while fitting the award audience and 15-minute allocation.

- Preserve the computational-cost/local-deployment motivation. Explain the problem before naming the implementation choices.
- Replace DS-GCN-TCN, RTMW-XL and the CTC-centered desktop scope with the manuscript's Apple Vision, MobileCLIP2, Squeezeformer, boundary-based recognition and iPhone workflow.
- Present significance as intended benefit and technical value. Do not imply completed community-impact or user-satisfaction results.
- Match objectives to §1.3, including the stated software-quality assessment objective, without inventing completed ISO ratings.
- Use §1.4 for actual scope and limitations, including capture conditions and language coverage.
- Keep preparation/development separate from measured outcomes. Explain the pipeline once and use later slides to develop decisions rather than repeat it.

## Design guidance

Use restrained LNU navy and gold accents with a plain light background and a compact institutional mark. Let the content occupy more space than the old deck's wide title banner and logo column. Use a small optional section marker such as “Methodology” on Slides 7–10, while keeping natural slide titles prominent.

Give Slide 3's deployment challenge and Slide 12's evidence a shared visual motif. Use actual app screenshots and recorded output, one clear pipeline on Slide 7, and readable measurement labels on results. Put source citations in the footer and preserve detailed evidence conditions in speaker notes. Do not add section-divider or thank-you slides to the timed deck.
