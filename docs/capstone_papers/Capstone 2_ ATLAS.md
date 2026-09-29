**ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

A Capstone Project Presented to the Faculty   
of the IT and Computer Education Unit  
Leyte Normal University  
Tacloban City

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

In Fulfillment of the   
Requirements for the Degree  
Bachelor of Science in Information Technology

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**by:**

Francis Angelo C. Batiancela  
Rica Sheen G. Cabidog  
Kyle Grant G. Lapid  
Jazcel A. Esio

**October 2026**

# **Table of Contents** {#table-of-contents}

[Table of Contents	ii](#table-of-contents)

[List of Figures	iii](#list-of-figures)

[List of Tables	iv](#list-of-tables)

[**CHAPTER 1: Introduction	1**](#chapter-1)

[1.2 Purpose and Description	4](#heading=h.81cv21ylu7hs)

[1.3 Objective of the Study	6](#heading=h.9u9qumbr1d1h)

[1.4 Scope and Limitations	7](#heading=h.xj4prufmtdlb)

[**CHAPTER 2: Review of Related Literature/Related Systems	10**](#chapter-2)

[2.1 Related Literature	10](#heading=h.wgz7i1aks4r9)

[2.2 Related Systems	16](#heading=h.sutxgqv3j7bu)

[**CHAPTER 3: Technical Backgrounds	23**](#chapter-3)

[3.1 Current System	23](#3.1-current-system)

[3.2 Proposed System	24](#3.2-proposed-system)

[Software Requirements	27](#software-requirements)

[Hardware Requirements	28](#hardware-requirements)

[Peopleware	29](#peopleware)

[**CHAPTER 4: Methodology, Results, Discussion	31**](#chapter-4)

[4.1 Requirements Analysis	31](#4.1-requirements-analysis)

[Functional Requirements	31](#functional-requirements)

[Performance Requirements	32](#performance-requirements)

[Software and Hardware Requirements	34](#software-and-hardware-requirements)

[Safety Requirements	35](#safety-requirements)

[Security Requirements	35](#security-requirements)

[4.1.1 Actual Model Performance and Current Limitations	36](#heading=h.t6pqaonws72o)

[4.2 Design of Software, Systems, Product, and/or Processes	37](#4.2-design-of-software,-systems,-product,-and/or-processes)

[Systems Development Life Cycle	37](#systems-development-life-cycle)

[Planning and Requirement Analysis	39](#planning-and-requirement-analysis)

[Activity List	40](#activity-list)

[GANTT Chart	43](#heading=h.3brsbjkx3pxh)

[PERT Chart	51](#pert-chart)

[Design	53](#heading=h.yrywqlqlyo9q)

[Context Diagram	53](#context-diagram)

[Data Flow Diagram	54](#data-flow-diagram)

[Use-case	55](#use-case)

[Flowchart	55](#flowchart)

[**REFERENCES	58**](#references)

# **List of Figures** {#list-of-figures}

**Figure 1:** SDLC Modified Waterfall (Sashimi) Model	38

**Figure 2:** PERT Chart of the Proposed System	52

**Figure 3:** Context Diagram of the Proposed System	53

**Figure 4:** Data Flow Diagram of the Proposed System	54

**Figure 5:** Proposed System Use-Case Diagram	55

**Figure 6:** Proposed System Flowchart	[5](#references)7

# **List of Tables** {#list-of-tables}

**Table 1:** Software and Descriptions	27

**Table 2:** Hardware Components and Descriptions	28

**Table 3:** Functional Requirements	31

**Table 4:** Performance Requirements	32

**Table 5:** Software and Hardware Requirements	[3](#chapter-4)4

**Table 6:** Safety Requirements	35

**Table 7:** Security Requirements 	[3](#chapter-4)5

**Table 8:** Activity List	39

**Table 9:** GANTT Chart	44

# **CHAPTER 1** {#chapter-1}

This chapter presents the background of the study on the development of ATLAS, an application-based Squeezeformer–based system for sign language recognition and English translation. It discusses existing systems, recent advancements in artificial intelligence and deep learning, and the limitations of current approaches in practical communication settings. The chapter also outlines the purpose, objectives, scope, and limitations of the study, highlighting the need for a more accessible and efficient communication support system.

## **Introduction**

Communication is a fundamental human need and a key driver of educational and social inclusion. However, communication barriers persist between sign language users and individuals who do not understand sign language. According to the World Health Organization, hearing loss affects a substantial portion of the global population, limiting access to education, employment, healthcare, and everyday social participation when communication support is inadequate (World Health Organization, 2026). In many academic settings, such as Leyte Normal University (LNU), interactions between Deaf or hard-of-hearing students and non-signing faculty or peers may rely on written notes, typed text exchanges, or improvised gestures. These manual workarounds disrupt conversational flow, introduce delays, and constrain natural expression.

Recent advances in artificial intelligence, computer vision, and natural language processing offer promising avenues for automated sign language translation. A key architectural shift is the adoption of landmark-based skeletal representations over raw full-frame video processing. By tracking discrete coordinate keypoints on the hands, face, and body, landmark extraction isolates meaningful motion and spatial configurations while filtering out background noise, lighting variations, and personal appearance bias. On supported platforms, engines such as Apple's Vision framework and Google's MediaPipe provide efficient on-device keypoint estimation for real-time applications (Apple, n.d.-a; Bazarevsky et al., 2020).

However, landmark coordinates alone may omit subtle handshape details during fast motion, severe self-occlusion, or complex articulation. While full-frame video models can capture these finer visual details, they incur heavy computational and memory overhead that hinders real-time deployment on consumer hardware. Integrating complementary, lightweight visual evidence—such as cropped RGB hand images—directly addresses this coordinate ambiguity. By combining spatial landmark tracks with localized hand-crop embeddings, multimodal pipelines can preserve critical fine-grained handshape cues while maintaining low computational latency and modest memory usage.

Recognizing gestures is only the first step in bridging the communication gap. Signed languages possess distinct grammatical structures, spatial relationships, and non-manual signals that do not map word-for-word to spoken or written English (Camgoz et al., 2020; Müller et al., 2023). Translating sign language therefore requires two distinct capabilities: an efficient sequence recognition engine capable of capturing local joint dynamics alongside global temporal context, and a bounded English rephrasing of the recognized gloss buffer.

To address sequence modeling efficiency, hybrid architectures combining self-attention and convolution have emerged as strong backbones. The Squeezeformer architecture redesigns the Conformer model through a Temporal U-Net structure, simplified block organization, removal of redundant normalization, and efficient depthwise subsampling, achieving superior accuracy and lower FLOP consumption in temporal sequence tasks (Kim et al., 2022). For target language generation, the Text-to-Text Transfer Transformer (T5) architecture offers a unified sequence-to-sequence framework well-suited for converting structured gloss buffers into readable natural language text (Raffel et al., 2020; Chung et al., 2024).

Despite technological progress, many existing sign language recognition systems face practical deployment hurdles: high computational overhead, reliance on isolated pre-segmented video clips, lack of streaming activity control, or raw gloss outputs that non-signers find difficult to interpret. To address these limitations, this study proposes ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. Designed for a fixed 100-gloss vocabulary, ATLAS integrates coordinate-based landmark extraction via Apple Vision (with supplemental MediaPipe mouth tracking), a multimodal Squeezeformer recognition engine combining skeletal landmarks and MobileCLIP2 hand-crop visual embeddings, a Reel activity controller for real-time gesture verification, a gloss buffering engine, and a fine-tuned T5-efficient-tiny model for bounded English rephrasing paired with local text-to-speech output. Through this architecture, the study aims to contribute a practical assistive communication tool that converts recognized sign language sequences into fluent English text and audible speech within academic environments.

## **1.2 Purpose and Description**

The primary purpose of this study is to design, implement, and empirically evaluate ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. The proposed system assists non-sign language users—such as faculty, staff, and peers at Leyte Normal University—in understanding signed communication by translating live gesture sequences into fluent English text and spoken audio. In doing so, ATLAS addresses campus communication barriers, replacing disruptive manual workarounds like written notes with a responsive, automated assistive tool. 

ATLAS operates for a fixed 100-gloss vocabulary, avoiding computationally heavy full-frame video processing across five integrated software modules. First, the Visual Extraction Module captures live camera streams to extract 61 coordinate keypoints (X, Y, Z, presence, confidence) via Apple Vision and MediaPipe, alongside cropped hand RGB images. Second, the Multimodal Squeezeformer Recognition Module processes 32-frame coordinate tensors through a 6.79M-parameter skeletal backbone and hand crops through a 4.81M-parameter MobileCLIP2 Squeezeformer, fusing predictions via a learned 0.53M-parameter logit head. 

Third, the Streaming Activity Controller evaluates overlapping 0.50-second candidate motion windows at 0.12-second intervals for gesture proposal, verification, and duplicate suppression. Fourth, the Gloss Buffering & Utterance Completion Module appends verified signs to a visible buffer, with utterance boundaries triggered explicitly via a UI Finish button or a 0.4-second held two-open-palm gesture. Fifth, the English Rephrasing & Speech Output Module passes accumulated glosses to a 15.58M-parameter T5-efficient-tiny model, displaying natural English sentences and synthesizing local text-to-speech output. 

For sequence recognition, ATLAS adopts Squeezeformer as its core temporal backbone. Squeezeformer redesigns the hybrid Conformer architecture through a Temporal U-Net structure, simplified block organization, removal of redundant normalization, and efficient depthwise subsampling (Kim et al., 2022). These structural refinements achieve high classification accuracy with lower FLOP consumption and reduced memory overhead, making Squeezeformer well-suited for real-time gesture modeling on edge-level consumer hardware (Kim et al., 2022). 

For target language translation, ATLAS leverages a fine-tuned, lightweight T5-efficient-tiny model. Operating within a unified sequence-to-sequence framework, T5 transforms ordered intermediate gloss buffers into syntactically correct English sentences (Raffel et al., 2020; Chung et al., 2024). This rephrasing step bridges the grammatical gap between sign language spatial syntax and spoken English, delivering clear, natural sentences rather than disconnected gloss labels. 

The system functions as a streaming, segment-buffered assistive translation tool rather than a fully simultaneous interpreter. The sign language user signs naturally within the camera view while the streaming activity controller verifies candidate signs and appends them to the gloss buffer. Once the user completes their message—signaled via UI control or the held two-open-palm gesture—the system finalizes the buffer, executes T5 English rephrasing, and synthesizes spoken audio for non-signing recipients. 

To evaluate ATLAS, a structured dataset of isolated glosses, continuous phrases, and paired gloss-to-English sequences was assembled for model training and benchmark testing. In addition to technical offline performance metrics (Top-1/Top-5 accuracy, Macro-F1, BLEU, human ratings), the implementation is evaluated against the ISO/IEC 25010:2023 quality model to assess its functional suitability, performance efficiency, usability, and reliability within an academic setting.

## **1.3 Objective of the Study**

The primary objective of this study is to design, implement, and empirically evaluate ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation that recognizes live gesture sequences from a fixed 100-gloss vocabulary, appends verified signs to an active gloss buffer via real-time activity control, and translates completed sign sequences into fluent English text and synthesized speech.

Specifically, this study aims to:

* Develop a multimodal recognition and translation pipeline combining Apple Vision landmark extraction (supplemented by MediaPipe mouth keypoints), MobileCLIP2 hand-crop visual embeddings, a dual Squeezeformer sequence classifier, and a fine-tuned T5-efficient-tiny language generation model.  
* Evaluate the offline performance of the Squeezeformer recognition model and T5 English rephrasing component using technical metrics, including Top-1 accuracy, Top-5 accuracy, Macro-F1 score, BLEU, and human translation ratings across evaluation datasets.  
* Implement the trained models into a responsive system integrated with a Reel activity controller for streaming motion verification, gloss buffering, dual-mode utterance completion (UI button and 0.4-second held two-open-palm gesture), and local text-to-speech output.  
* Validate the developed system based on ISO/IEC 25010:2023 quality standards, particularly functional suitability, performance efficiency, usability, and reliability, by conducting appropriate software evaluation procedures such as Black Box testing for functional behavior and White Box testing for internal logic and processing flow.

## **1.4 Scope and Limitations**

This study focuses on the design, implementation, and empirical evaluation of ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. The scope is designed around a fixed 100-gloss American Sign Language (ASL) vocabulary to support assistive communication in academic environments such as Leyte Normal University. The evaluation covers five integrated pipeline modules: (1) dual-stream visual extraction using Apple Vision for 61 coordinate keypoints, MediaPipe for supplemental mouth landmarks, and RGB hand crops for MobileCLIP2 visual embeddings; (2) a multimodal Squeezeformer recognition engine combining skeletal and visual features; (3) a Reel streaming activity controller for real-time motion verification and duplicate suppression; (4) a gloss buffering mechanism with dual-mode utterance completion; and (5) a fine-tuned 15.58M-parameter T5-efficient-tiny model paired with local text-to-speech (TTS) synthesis.

The study is strictly bounded by American Sign Language (ASL) within a predefined 100-gloss target vocabulary and its associated phrase structures. Consequently, the system is not designed to recognize or translate other sign language systems, such as Filipino Sign Language (FSL) or regional sign variations. Out-of-vocabulary (OOV) gestures, fingerspelling, or unrepresented sign phrases cannot be processed by the recognition engine or rephrased by the language model.

ATLAS functions as a segment-buffered assistive tool rather than a fully simultaneous real-time interpreter. The system does not execute continuous, unconstrained translation while the user is actively signing. Instead, candidate gestures are verified by the Reel activity controller (processing 0.50-second candidate windows at 0.12-second intervals) and accumulated in an active gloss buffer. Translation rephrasing and speech synthesis are triggered only when an utterance boundary is explicitly established through the manual UI Finish button or the automated 0.4-second held two-open-palm gesture.

System recognition performance remains dependent on physical and environmental conditions. Effective tracking requires adequate ambient lighting, proper camera framing, and an unobstructed view of the signer's upper body and hands. Rapid motion, severe hand self-occlusion, or improper camera angles can lead to missing keypoints, coordinate jitter, or landmark swapping. Furthermore, while MediaPipe mouth landmarks serve as supplemental evidence to distinguish specific visually similar sign pairs, the system does not perform general lip-reading, facial affect analysis, or full non-manual signal interpretation.

The current evaluated implementation is a Mac-based, designed to validate the end-to-end processing pipeline on consumer edge hardware. The empirical evaluation relies on signer-disjoint recorded clips and controlled live testing; it does not evaluate full mobile deployment, low-power micro-controller hardware, or unconstrained multi-party conversational signing. Software quality validation is governed by the ISO/IEC 25010:2023 framework—focusing on functional suitability, performance efficiency, usability, and reliability—alongside technical offline metrics including Top-1 accuracy, Top-5 accuracy, Macro-F1 score, BLEU, and human translation ratings.

# **CHAPTER 2** {#chapter-2}

**REVIEW OF RELATED LITERATURE AND SYSTEMS**  
This chapter reviews existing studies, systems, and technological developments relevant to the proposed ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. It discusses the theoretical and technical foundations of sign language recognition, landmark-based visual representation, temporal sequence modeling, Squeezeformer architectures, RGB-based hand-crop features, multimodal fusion, intermediate gloss representation, and English language generation. The chapter also examines related systems and approaches that use visual, skeletal, temporal, and multimodal information for sign language recognition and translation. Furthermore, relevant developments in streaming recognition, application-oriented sign language systems, and evaluation considerations are reviewed to provide a basis for understanding the design choices of the proposed system. By examining these studies and systems, this chapter identifies the research and implementation considerations that support the development of ATLAS as a multimodal, streaming sign language recognition and English rephrasing for a fixed 100-gloss vocabulary.

**2.1 Related Literature**

**2.1.1 Sign Language Recognition and Translation**  
Sign language recognition and sign language translation are related but distinct tasks in computer vision and natural language processing. Sign language recognition focuses on identifying signs or sign sequences from visual input, whereas sign language translation involves converting recognized sign information into a target spoken or written language. Sign languages contain their own linguistic structures, including manual and non-manual components, and therefore cannot always be converted into English through direct word-for-word substitution (Yin et al., 2021; Rastgoo et al., 2021). This distinction is important when developing systems that aim not only to recognize signs but also to produce understandable English output.

Camgoz et al. (2020) demonstrated the use of Transformer-based models for joint continuous sign language recognition and translation. Their work showed the value of an intermediate gloss representation and sequence modeling in connecting visual sign input with language generation. Similarly, Yin et al. (2021) emphasized that signed languages should be considered within natural language processing research because they possess linguistic structure and meaning beyond isolated visual gestures. These studies establish that an effective sign language translation system requires both a recognition component and a language-processing component.

The selection of a defined vocabulary is also an important consideration in sign language recognition. Desai et al. (2023) introduced ASL Citizen, a community-sourced dataset of isolated American Sign Language signs collected from Deaf and hard-of-hearing contributors in a variety of natural environments. The ASL Citizen dataset contains a large number of sign instances and was developed to support sign language recognition and dictionary-related applications. Its ASL Citizen 100 subset provides a standardized set of 100 glosses selected from the larger dataset based on factors including sample density, contributor agreement, and instance frequency (Desai et al., 2023).

The selection of the ASL Citizen 100 subset itself followed a structured, Deaf-centered methodology rather than an arbitrary sampling of frequent signs. Desai et al. (2023) described a four-step process in which candidate glosses were first cross-referenced against ASL-LEX, a linguistic database that documents the phonological properties of ASL signs, including handshape, location, and movement, to ensure that every selected gloss was phonologically well defined. Candidate signs were further filtered for dictionary utility, prioritizing terms with high real-world query frequency in digital ASL dictionaries. Recordings were then elicited through standardized prompt videos performed by fluent Deaf seed signers to maintain semantic consistency across the 52 contributors, and the final 100 glosses were ranked and retained according to instance density, or the number of available video samples per gloss across diverse signers. This methodology distinguishes ASL Citizen 100 from legacy 100-gloss benchmarks such as WLASL100 (Li et al., 2020\) and MS-ASL100, which were compiled by scraping sign videos and their titles from online sources rather than through direct collaboration with Deaf contributors. Because web-scraped datasets are not filtered through a linguistic reference such as ASL-LEX, they are more susceptible to mixed dialects, informal gesture variants, and polysemic label noise, in which a single English gloss label is attached to videos representing different underlying signs; the English word “right,” for example, may be attached to videos meaning either “correct” or “direction.” ASL Citizen 100 also provides substantially higher signer diversity and sample density per class than WLASL100, whose clips average only a small number of samples per gloss, and its dataset is expected to remain reproducible over time because it was collected under explicit participant consent rather than linked to third-party video hosting that may later become unavailable.

The ASL Citizen 100 vocabulary provides the basis for the fixed 100-gloss scope adopted in ATLAS. A predetermined vocabulary establishes a controlled recognition problem and allows the system to focus on consistent classification of a defined set of American Sign Language signs. The use of a fixed vocabulary is appropriate for the present study because ATLAS is designed for a specific set of sign language inputs rather than as an unrestricted sign language interpreter. The system therefore recognizes signs within the selected 100-gloss scope, while signs outside the vocabulary remain outside the recognition capability of the current implementation. The complete list of selected glosses may be presented in the methodology or appendix to document the system's recognition scope. Section 2.1.7 further develops the justification for constraining ATLAS to this 100-gloss scope, situating the decision within the broader vocabulary-scaling literature.

In addition to vocabulary selection, intermediate representations are relevant to the translation process. Moryossef et al. (2021) described sign language translation as a process that may be divided into video-to-gloss recognition and gloss-to-text translation. This staged approach provides a structured representation between visual recognition and natural language generation. However, gloss-free approaches have also been investigated. Lin et al. (2023), for example, explored gloss-free sign language translation to reduce dependence on manually annotated glosses. These developments indicate that both gloss-based and gloss-free approaches remain relevant research directions, with the appropriate approach depending on the available data, vocabulary, and system objectives.

For ATLAS, the recognized signs are stored in a visible gloss buffer before English generation. The buffer provides an intermediate representation between the visual recognition module and the language-generation module. As signs are accepted by the streaming activity controller, their corresponding gloss labels are appended to the buffer. When the utterance is completed through the defined completion mechanism, the accumulated gloss sequence is passed to the English rephrasing component.

The Text-to-Text Transfer Transformer (T5) provides the language-generation foundation for this stage. Raffel et al. (2020) introduced T5 as a unified text-to-text framework in which different natural language processing tasks can be formulated as text generation. In ATLAS, a fine-tuned T5-efficient-tiny model is used to rephrase the recognized gloss sequence into an English sentence. Therefore, the language component of ATLAS should be understood as a bounded English rephrasing stage operating on the recognized gloss buffer rather than as an unrestricted sign-language translation model.

Overall, the literature supports a modular recognition-and-language approach for ATLAS. The recognition component identifies signs from the visual stream, the gloss buffer provides an intermediate representation, and the T5-based component generates an English representation of the completed sequence. This separation allows the recognition and language-generation components to be developed and evaluated as related but distinct parts of the overall system.

**2.1.2 Landmark-Based Visual Representation for Sign Language Recognition**  
Landmark-based visual representation provides a structured alternative to processing complete raw video frames. Instead of representing an input entirely through pixel information, landmark-based approaches describe relevant anatomical points through coordinates and related measurements. This representation allows recognition models to focus on the spatial configuration and movement of the hands, face, and body while reducing the influence of unrelated background information (Rastgoo et al., 2021). Sign language recognition is particularly suited to structured visual representations because hand configuration, movement, position, facial information, and body posture can contribute to the interpretation of a sign.

MediaPipe Hands is one example of a landmark-based approach. Zhang et al. (2020) presented MediaPipe Hands as an on-device hand-tracking pipeline that estimates hand landmarks from RGB camera input. Such structured hand representations have been used in gesture and sign recognition because they provide coordinate-based information that can be processed by temporal and spatial models.

Apple Vision also provides visual pose estimation capabilities for supported platforms, including hand and human body pose analysis. These capabilities can provide structured information about the location and movement of relevant body parts in images and video. The use of hand and body landmarks is consistent with research showing that sign recognition may benefit from representing multiple parts of the signer rather than relying exclusively on isolated hand images (Rastgoo et al., 2021; Lee et al., 2023).

Lee et al. (2023) specifically investigated part-wise motion context for sign language recognition and proposed a framework that models different human parts before combining their information into a unified representation. Their findings support the importance of preserving part-specific motion information while also learning broader whole-body context.

In ATLAS, the primary landmark representation contains 61 nodes consisting of 21 left-hand landmarks, 21 right-hand landmarks, 15 face landmarks, and 4 upper-body landmarks. Each landmark contains raw spatial coordinates together with presence and confidence information. The system also derives temporal velocity and acceleration features and selected pairwise hand-distance features to provide additional information about movement and spatial relationships.

The landmark representation therefore serves as the structured input to the Squeezeformer recognition branch. MediaPipe mouth landmarks are used only as supplemental non-manual evidence for selected recognition cases and should not be interpreted as a general lip-reading component. The combination of hand, face, and upper-body information provides ATLAS with a broader visual representation while maintaining a structured coordinate-based input for temporal modeling.

**2.1.3 Temporal and Sequence Modeling for Sign Language Recognition**  
Sign language recognition is inherently temporal because the meaning of a sign can depend on how hand and body configurations change over time. Movement direction, timing, transitions, and the relationship between consecutive frames may distinguish signs that appear similar in individual static frames. Consequently, sign recognition systems must consider ordered visual information rather than treating each frame as an independent classification problem (Rastgoo et al., 2021).

Transformer-based architectures have become relevant to sequential sign language recognition because self-attention can model relationships between elements at different positions within a sequence. Camgoz et al. (2020) demonstrated this through a Transformer-based framework for continuous sign language recognition and translation. Their work also incorporated Connectionist Temporal Classification (CTC) to connect recognition and translation within the proposed architecture.

CTC is therefore relevant as an established approach in continuous sign language research, but it is not the primary mechanism of the current ATLAS implementation. The current system uses a streaming activity controller to repeatedly propose candidate windows, verify predictions, apply stability and commit controls, and suppress duplicate recognitions. The recognized signs are then accumulated in a visible gloss buffer. Thus, the streaming behavior of ATLAS is achieved through activity and decision control rather than requiring CTC as the default recognition mechanism.

The landmark branch of ATLAS processes ordered landmark sequences consisting of 32 frames. The model receives spatial and confidence-related landmark information together with derived motion features, allowing it to learn both static configurations and temporal changes. The RGB branch also retains temporal information by sampling left-hand, right-hand, and combined-hand crops at multiple time points before applying temporal sequence modeling.

Temporal modeling is therefore central to ATLAS because the system must distinguish individual signs while also operating on an ongoing camera stream. The combination of temporal recognition and activity control allows the system to determine when a candidate sign should be proposed, verified, committed, or suppressed. This design provides a practical alternative to treating the application as a collection of independent frame-level classifications.

**2.1.4 Squeezeformer as an Efficient Temporal Architecture**  
Hybrid attention-convolution architectures have become important for sequence modeling because they combine mechanisms for capturing long-range dependencies with operations that can model local temporal structure. Squeezeformer was introduced by Kim et al. (2022) as an efficient Transformer architecture for automatic speech recognition. The architecture modifies the Conformer design through a Temporal U-Net structure, simplified block organization, reduced normalization operations, and efficient depthwise downsampling. The authors reported competitive recognition performance while maintaining computational efficiency.

Although Squeezeformer was originally developed for speech recognition, its architectural characteristics are relevant to other sequential recognition problems. Sign language recognition, like speech recognition, involves ordered information in which the interpretation of a particular observation may depend on preceding and following observations. Therefore, Squeezeformer's emphasis on efficient temporal modeling provides a theoretical basis for investigating its use in sign language recognition.

The current ATLAS landmark branch uses a part-wise and global Squeezeformer design. The landmark representation is divided into four streams corresponding to the major body-part groups used by the system. These streams are projected into a 64-dimensional representation before being integrated into a 256-dimensional global representation processed by a four-block Squeezeformer. This design allows the model to preserve part-specific information while also learning broader temporal relationships across the combined landmark sequence. The use of Squeezeformer in ATLAS should therefore be understood as an architectural adaptation rather than a claim that Squeezeformer is an established standard for sign language recognition. Kim et al. (2022) provide the foundation for its efficiency-oriented temporal design, while the current study applies those principles to a sign-language landmark sequence. The resulting architecture is intended to model local movement cues and broader temporal context within the fixed 100-gloss recognition problem.

**2.1.5 RGB Hand-Crop Features and Efficient Visual Encoders**  
Landmark representations provide structured information about the location and movement of body parts, but coordinate-based features may not fully preserve fine-grained visual appearance such as detailed handshape and local visual configuration. Research on sign language recognition has therefore explored combinations of skeletal or pose information with visual information to capture complementary characteristics of a sign (Rastgoo et al., 2021). More recent multimodal approaches have similarly investigated the combination of motion trajectories and RGB appearance features for isolated sign recognition (Renjith et al., 2026).

Cropped hand images provide a more focused visual representation than processing complete video frames. By concentrating on the hand regions involved in signing, a system can preserve appearance information while limiting the amount of unrelated background information processed by the visual encoder. This approach is particularly relevant for systems that require a balance between visual representation and computational efficiency.

MobileCLIP provides an example of an efficient visual representation model. Vasu et al. (2024) introduced MobileCLIP as a family of efficient image-text models designed to reduce the latency and memory demands associated with larger image-text models. Its lightweight design makes the model relevant to application-oriented visual processing where computational resources are limited.

ATLAS uses MobileCLIP2 as an upstream image feature extractor for cropped hand images. The system samples left-hand, right-hand, and combined-hand crops at selected time points. The resulting visual embeddings are then processed by a temporal Squeezeformer head. This design provides a complementary visual branch to the landmark branch rather than replacing landmark-based recognition.

The use of multiple hand-crop views is intended to preserve visual information that may not be completely represented by coordinates alone. The left-hand and right-hand views provide separate evidence about the hands, while the combined-hand view provides information about their relative visual configuration. The resulting RGB branch is therefore integrated with the landmark branch through the multimodal fusion component of ATLAS.

**2.1.6 Multimodal Fusion for Sign Language Recognition**  
Sign language recognition can benefit from combining different forms of visual information because individual modalities may capture different characteristics of signing. Landmark representations provide structured information about spatial configuration and motion, while RGB-based visual features can preserve appearance information such as handshape and local visual details. Sign language recognition surveys have identified multimodal approaches as an important direction because hand, body, facial, and other visual cues can contribute complementary information to recognition (Rastgoo et al., 2021).

Part-aware approaches further demonstrate the value of separating information according to body regions before combining it into a unified representation. Lee et al. (2023) showed that part-wise motion-context encoding can complement whole-body temporal modeling. This supports the idea that a recognition system can preserve local part-specific information while still learning relationships across the complete signing configuration.

More recent research has also investigated direct integration of motion and visual appearance. Renjith et al. (2026) proposed a dual-stream Transformer framework that combines skeletal motion trajectories with RGB visual features for isolated sign language recognition. Their work illustrates the complementary nature of motion and appearance information in distinguishing visually similar signs.

ATLAS applies this multimodal principle through two recognition branches. The landmark branch processes structured hand, face, and upper-body information using a part-wise and global Squeezeformer architecture. The RGB branch processes left-hand, right-hand, and combined-hand crops through MobileCLIP2 embeddings followed by temporal Squeezeformer modeling. The predictions from the two branches are then combined through a learned fusion head.

The multimodal design does not assume that one modality is always sufficient or that fusion automatically guarantees higher accuracy. Instead, it provides complementary sources of evidence that can be evaluated individually and in combination. This allows the study to compare landmark-only, RGB-only, and fused recognition performance and determine whether the additional visual branch contributes measurable value within the fixed 100-gloss vocabulary.

**2.1.7 Justification for the Fixed 100-Gloss Vocabulary Scope**  
Beyond its methodological grounding in ASL Citizen 100, the decision to constrain ATLAS to a 100-gloss vocabulary is supported by four considerations drawn from the sign language recognition literature: the relationship between vocabulary size and recognition accuracy, computational efficiency, data availability, and alignment with a standard evaluation scale.

Accuracy and vocabulary size. Recognition accuracy in isolated sign language recognition consistently declines as the number of target classes grows, even when the underlying architecture and training data collection procedure are held constant. Woods and Rana (2023) trained an encoder-only transformer on keypoint data drawn from an enhanced version of WLASL and reported top-1 accuracy of 97% on a 10-sign vocabulary, 87% on 50 signs, 83% on 100 signs, and 71% on 300 signs, a 12-point drop from 100 to 300 classes alone. A comparable pattern appears in pose-based Transformer research on the original WLASL benchmark: Boháček and Hrúz (2022) reported that their SPOTER architecture reached 63.18% top-1 accuracy on WLASL100 but only 43.78% on WLASL300. Outside the WLASL benchmark, Zheng et al. (2022) observed the same tendency in wearable-sensor-based sign recognition, noting that the accuracy of existing systems drops sharply as the number of target gestures increases because larger vocabularies contain a greater number of visually or kinematically similar signs. These findings support a 100-gloss scope as a vocabulary size at which recognition accuracy remains high enough to be practically usable, while still exceeding the very small vocabularies of 10 to 50 signs used in some prior work.

Computational efficiency. A larger vocabulary also enlarges the classification search space and increases the computational burden of both training and inference. Woods and Rana (2023) found that models with fewer than 100,000 learnable parameters could still reach 83% top-1 accuracy at the 100-gloss scale, and explicitly framed this result as evidence that a 100-class recognition problem is small enough to run on lightweight, consumer-grade hardware rather than requiring specialized equipment. This finding is directly relevant to ATLAS, whose landmark and RGB branches are designed for efficient, in-browser or on-device operation; scaling the recognition vocabulary substantially beyond 100 glosses would work against this design goal.

Data availability. Because sign language datasets require costly manual annotation and are collected from a comparatively small number of contributors, larger vocabularies tend to suffer from class imbalance, with many glosses represented by only a handful of samples. A 100-gloss scope keeps the number of target classes within a range for which datasets such as ASL Citizen 100 and WLASL100 can provide a reasonably dense number of samples per class, which is a precondition for reliably training a Squeezeformer-based classifier of the kind used in ATLAS. Extending the vocabulary to 500 or more glosses without a matching increase in data collection effort would be expected to reintroduce the class-imbalance and low-sample-density problems that the 100-gloss scope is intended to avoid.

Alignment with a standard evaluation scale. Finally, 100 glosses is a recognized benchmark scale in the sign language recognition literature, most notably through the WLASL100 subset, which is treated as the standard reporting condition by a large proportion of published isolated sign language recognition studies (Li et al., 2020). Recent systems continue to be developed and evaluated at this scale. Amiruzzaman et al. (2026) reported a lightweight, transformer-based skeleton recognition model reaching 88.48% top-5 accuracy on WLASL100 with 140.45 ms average inference latency on a consumer CPU; Syulistyo et al. (2025) evaluated a Multiple Reservoir Computing approach on WLASL100, reporting 60.35% top-1, 84.65% top-5, and 91.51% top-10 accuracy with a CPU-only inference target; and Eunice et al. (2023) reported 80.90% top-1 accuracy on WLASL100 using a pose-based Transformer with 30 ms average inference latency. These studies, discussed further in Section 2.2.6, indicate that the 100-gloss scale remains an active area of research for lightweight, edge-deployable sign language recognition systems, which situates ATLAS within a directly comparable body of published work rather than an idiosyncratic evaluation setting.

Taken together, these four considerations indicate that a fixed 100-gloss vocabulary represents a deliberate scope decision rather than an arbitrary limitation. It keeps recognition accuracy within a practically usable range, preserves the lightweight computational profile that ATLAS's landmark and RGB branches depend on, remains realistic given the amount of labeled ASL data that is available, and allows the system's reported results to be interpreted against an established body of published benchmarks.

**2.2 Related Systems**

**2.2.1 Vision-Based Sign Language Recognition Systems**  
Vision-based sign language recognition systems use cameras or video to capture visual information from a signer without requiring wearable gloves or specialized motion sensors. Vision-based approaches may analyze hand configuration, movement, body posture, facial information, and other visual characteristics associated with signing. Rastgoo et al. (2021) identified vision-based recognition as a major direction in sign language recognition research and discussed both isolated and continuous recognition approaches.

Camgoz et al. (2020) presented Sign Language Transformers, a Transformer-based framework for joint continuous sign language recognition and translation. Their system demonstrated how visual sign sequences can be processed using attention-based temporal modeling while connecting recognition and translation through CTC. The study is relevant to ATLAS because both systems treat signing as a sequence rather than as independent static images.

However, the architectures and operational designs differ. Sign Language Transformers was developed for continuous sign language recognition and translation on benchmark datasets, whereas ATLAS is designed around a fixed 100-gloss vocabulary and a streaming activity-controlled workflow. ATLAS first extracts structured landmark and cropped-hand visual information before recognition and then stores accepted predictions in a gloss buffer for later English rephrasing.

Vision-based recognition therefore provides the broader technological foundation for ATLAS, while the proposed system narrows the problem to a defined vocabulary and combines structured landmark information with localized RGB hand-crop features. This allows the study to investigate multimodal recognition within a controlled vocabulary rather than claiming unrestricted sign-language translation.

**2.2.2 Landmark-Based and Keypoint-Based Recognition Systems**  
Landmark-based and keypoint-based recognition systems transform visual input into structured coordinates representing relevant anatomical points. This approach is useful for sign language recognition because hand position, hand configuration, movement, and body relationships are important characteristics of signing. Landmark representations can also reduce the amount of irrelevant background information that must be processed by the recognition model (Rastgoo et al., 2021).

Zhang et al. (2020) introduced MediaPipe Hands as an on-device hand-tracking pipeline that estimates hand landmarks from RGB camera input. Such landmark extraction methods provide structured information that can be used as input to downstream gesture and sign recognition models.

Research has also investigated the use of multiple body parts rather than hands alone. Lee et al. (2023) proposed a part-wise motion-context framework in which motion information is first modeled according to human body parts and subsequently integrated into a whole-body representation. Their work supports the use of part-specific information for capturing movement characteristics that may be lost when all body points are processed as one undifferentiated sequence.

ATLAS extends this general landmark-based approach by combining left-hand, right-hand, face, and upper-body landmarks in its core representation. The system further derives velocity, acceleration, and selected hand-distance features before temporal modeling. Apple Vision provides the primary hand, face, and upper-body landmark information, while MediaPipe mouth landmarks are used as supplemental evidence for selected recognition cases. Thus, the landmark component of ATLAS is not itself the complete recognition system; it provides structured visual information for the downstream Squeezeformer model.

**2.2.3 Deep Learning-Based Sequential Sign Recognition Systems**  
Deep learning-based sequential sign recognition systems model ordered visual observations over time. These systems are important because sign recognition depends not only on the appearance of a hand at a particular moment but also on movement, direction, timing, and transitions between configurations. Transformer-based architectures have become particularly relevant because their attention mechanisms can model relationships across different points in a sequence.

Camgoz et al. (2020) demonstrated the use of Transformer-based sequence modeling for continuous sign language recognition and translation. Their work used CTC to connect recognition and translation, providing an example of how temporal recognition and language generation can be integrated within a sign language system.

Squeezeformer provides a different approach to efficient temporal modeling. Kim et al. (2022) introduced Squeezeformer for automatic speech recognition and designed the architecture to reduce computational overhead while retaining strong sequence-modeling capability. Although its original application was speech recognition, its temporal architecture provides a basis for adapting efficient Transformer-based modeling to other sequential recognition problems.

ATLAS applies Squeezeformer to sign language recognition through a part-wise and global landmark branch and a separate temporal RGB branch. The landmark branch processes 32-frame sequences, while the RGB branch processes temporally sampled hand-crop embeddings. Unlike systems that rely on CTC as the central streaming mechanism, the current ATLAS implementation uses a Reel activity controller for candidate proposal, verification, stability control, and duplicate suppression.

This distinction is important because ATLAS should not be described as a CTC-based system unless the CTC path is explicitly enabled and evaluated in the final implementation. Instead, its streaming behavior results from the combination of temporal recognition and activity-control mechanisms.

**2.2.4 Gloss-Based or Intermediate Representation Translation Systems**  
Gloss-based and intermediate-representation systems divide sign language translation into recognition and language-generation components. In such systems, visual sign input is first converted into a structured representation, such as glosses, before being transformed into a target spoken or written language. This approach provides an intermediate representation between visual recognition and natural language generation.

Camgoz et al. (2020) demonstrated the usefulness of gloss-level information in sign language translation through their joint recognition and translation framework. Moryossef et al. (2021) further described gloss-to-text translation as a low-resource neural machine translation problem and investigated data augmentation techniques for improving gloss-based translation. These studies demonstrate the value of separating visual recognition from subsequent language generation.

At the same time, gloss-free sign language translation has emerged as an alternative research direction. Lin et al. (2023) investigated gloss-free end-to-end sign language translation, addressing the dependence of gloss annotations in traditional pipelines. Gloss-free approaches may reduce annotation requirements, but they also represent a different system design from a controlled gloss-based architecture.

ATLAS follows the intermediate-representation approach because its recognition vocabulary is fixed at 100 glosses. Recognized signs are accumulated in a visible gloss buffer, after which the completed sequence is provided to the fine-tuned T5-efficient-tiny model for English rephrasing. The approach provides a transparent connection between recognition and language generation and allows the two components to be evaluated separately.

The English output of ATLAS should therefore be described as bounded English rephrasing of a recognized gloss sequence rather than unrestricted sign-language translation. This distinction reflects the system's fixed vocabulary, intermediate representation, and current implementation scope.

**2.2.5 Streaming and Application-Oriented Sign Language Systems**  
Application-oriented sign language recognition systems consider not only recognition accuracy but also how the recognition model operates during actual interaction. Practical systems must account for visual input conditions, temporal processing, response behavior, computational requirements, and the way recognized information is presented to users. Recent reviews have identified continuous recognition, signer variability, environmental conditions, and practical deployment as continuing challenges in sign language recognition research (Violet & Leena Sri, 2025).

Streaming recognition introduces additional challenges because the system must determine which portions of an ongoing camera stream contain meaningful signing. A system that classifies only pre-segmented clips does not necessarily solve the problem of deciding when a sign begins, when it should be accepted, or when a repeated prediction should be suppressed. Recent work on real-time isolated sign recognition has likewise emphasized the importance of efficient spatiotemporal modeling and low-latency processing for live recognition applications (Renjith et al., 2026).

ATLAS addresses this application-oriented problem through a streaming activity controller. Instead of treating every camera frame as an independent classification event, the system repeatedly evaluates candidate windows and applies proposal, verification, stability, commit, and duplicate-suppression controls. Accepted predictions are added to a visible gloss buffer, while utterance completion is explicitly established through the Finish button or the defined two-open-palm gesture.

The current ATLAS implementation should not, however, be described as a completed production mobile deployment. The evaluated implementation is Mac-based and has been tested using signer-disjoint recorded clips and controlled live testing. The current evaluation does not establish production-level Android or iOS performance, power consumption, thermal behavior, or unrestricted conversational operation.

The system is therefore better characterized as a streaming, activity-controlled assistive system. Its practical contribution lies in integrating visual extraction, multimodal recognition, streaming decision control, gloss buffering, English rephrasing, and speech output into one modular workflow. This application-oriented design provides a basis for future deployment studies while keeping the current claims consistent with the evaluated implementation.

**2.2.6 Recent 100-Gloss Word-Level ASL Recognition Systems**  
Several recently published systems illustrate how the 100-gloss scale continues to be used to explore lightweight, deployable sign language recognition, providing direct points of comparison for ATLAS's own design choices.

Amiruzzaman et al. (2026) presented a gamified, web-based vocabulary learning system for Deaf learners built around a lightweight, skeleton-based isolated sign language recognition model. Their system used a Stacked Transformer with spatial-temporal attention, integrated into a client-server web architecture, and converted webcam keypoint streams into sign glosses without processing raw video. The model reached 88.48% top-5 accuracy on WLASL100 with an average inference latency of 140.45 ms on a consumer-grade CPU, operating entirely in-browser without a dedicated GPU. This system is relevant to ATLAS because both rely on keypoint-based, rather than raw-video, recognition to keep inference lightweight, although ATLAS additionally incorporates an RGB hand-crop branch and targets streaming, activity-controlled recognition rather than a single-gloss vocabulary-learning interaction.

Syulistyo et al. (2025) proposed a low-cost isolated sign language recognition pipeline that combined MediaPipe keypoint extraction with Multiple Reservoir Computing, an approach intended to reduce the computational demands associated with conventional deep learning. Their MRC-based configuration achieved 60.35% top-1, 84.65% top-5, and 91.51% top-10 accuracy on WLASL100, with an average inference time of 5.2 seconds and a training time of 52.7 seconds on CPU, supporting its potential for low-cost edge deployment despite the higher per-inference latency compared to Transformer-based approaches. This system demonstrates that the 100-gloss scale can support alternatives to deep Transformer architectures when computational cost is the primary constraint.

Eunice et al. (2023) introduced Sign2Pose, a pose-based, Transformer-based word-level recognition system that applies key-frame extraction to remove redundant frames before classification. The system achieved 80.90% top-1 accuracy on WLASL100 and 64.21% top-1 accuracy on WLASL300, with an average inference time of 30 ms using RGB video input and pose estimation, without requiring specialized depth sensors or wearable devices. Sign2Pose is relevant to ATLAS's landmark branch because both rely on pose- or landmark-based, rather than purely appearance-based, visual representations, and because its reported accuracy drop from WLASL100 to WLASL300 provides further support for the vocabulary-size effect discussed in Section 2.1.7.

Collectively, these systems show that the 100-gloss scale continues to support active development of lightweight, real-time-oriented sign language recognition systems using keypoint, pose, and reservoir-computing approaches. ATLAS extends this line of work by combining a landmark branch and an RGB hand-crop branch within a part-wise Squeezeformer architecture, and by pairing recognition with a streaming activity controller and a bounded English rephrasing stage rather than treating recognition alone as the end objective.

# **CHAPTER 3** {#chapter-3}

**TECHNICAL BACKGROUND**  
This chapter presents the technical foundation of the proposed system. It describes the current communication methods addressed by the study and explains the architecture and technologies used in the proposed system. The chapter focuses on how visual sign language input is captured, represented, recognized, organized into a gloss sequence, and converted into understandable English output. It also describes the computational components that support the system, including landmark extraction, RGB hand-crop processing, multimodal Squeezeformer recognition, streaming activity control, gloss buffering, and T5-based English rephrasing.

## **3.1 Current System** {#3.1-current-system}

The existing communication methods addressed by this study rely primarily on manual forms of message exchange between sign language users and individuals who do not understand sign language. Within academic environments such as Leyte Normal University (LNU), communication may involve handwritten notes, typed messages through mobile note-taking applications, or other improvised forms of interaction. Although these methods allow information to be exchanged, they require the participants to manually compose and interpret messages rather than providing a direct means of bridging signed communication and English.

These communication methods can interrupt the natural flow of conversation because the sign language user may need to write or type a message instead of communicating directly through signing. The process may also introduce delays when immediate responses are needed, particularly during academic, administrative, service, or other everyday interactions. Written or typed exchanges may further result in shortened or simplified messages, which can make it more difficult to communicate ideas that would otherwise be expressed through a complete signed utterance.

Another limitation is the linguistic difference between sign languages and English. Sign languages have their own grammatical structures and may use spatial relationships, movement, facial information, and other non-manual cues that do not necessarily correspond to English on a word-for-word basis. Consequently, manually exchanging written messages does not directly address the linguistic gap between signed communication and English.

The absence of an automated assistive mechanism for recognizing sign language and producing understandable English output highlights the need for a system that can support communication more efficiently. This provides the basis for the development of ATLAS, which is designed to recognize a predefined set of American Sign Language (ASL) signs and convert completed sign sequences into English text and synthesized speech.

## **3.2 Proposed System** {#3.2-proposed-system}

ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation is a modular, streaming, activity-controlled assistive system designed to recognize a fixed vocabulary of 100 American Sign Language (ASL) glosses and convert recognized sign sequences into bounded English rephrasing. The system combines landmark-based skeletal information and localized RGB hand-crop information to provide complementary visual features for sign recognition.

The proposed system consists of five major components: (1) Visual Extraction, (2) Multimodal Squeezeformer Recognition, (3) Reel Activity Controller, (4) Gloss Buffering and Utterance Completion, and (5) English Rephrasing and Speech Output.

The Visual Extraction Module captures the signer's visual input and produces structured landmark and RGB representations. The landmark representation contains 61 nodes consisting of the left and right hands, face, and upper body. Additional motion and spatial features are derived from these landmarks to represent the movement and configuration of the signer. Localized RGB hand crops are also extracted to preserve visual details, particularly handshape information that may not be fully represented by landmarks.

The Multimodal Squeezeformer Recognition Module processes the extracted representations through two recognition branches. The first uses the landmark-based skeletal representation, while the second uses MobileCLIP2 to obtain visual embeddings from RGB hand crops. Both branches use Squeezeformer-based temporal modeling, and their outputs are combined through a learned multimodal fusion layer. This allows ATLAS to utilize both movement-based and visual information during sign recognition.

The Reel Activity Controller manages candidate sign detection within the camera stream. It uses overlapping motion windows together with proposal, verification, stability, commitment, and duplicate-suppression mechanisms to determine when a recognized sign should be accepted. This provides the system with controlled streaming behavior instead of treating every frame as an independent prediction.

The Gloss Buffering and Utterance Completion Module stores accepted signs as an ordered gloss sequence. The user can complete an utterance through the Finish button or the defined two-open-palm gesture. Once the utterance is completed, the accumulated gloss sequence is passed to the language-generation stage.

Finally, the English Rephrasing and Speech Output Module uses a fine-tuned T5-efficient-tiny model to convert the recognized gloss sequence into understandable English. The generated English text may then be presented through the application and converted into speech using local text-to-speech functionality. The language-generation process is bounded by the system's fixed vocabulary and intended phrase scope rather than functioning as unrestricted sign language translation.

Overall, ATLAS follows the workflow visual input → feature extraction → multimodal recognition → activity control → gloss buffering → English rephrasing → speech output. The system is designed as an assistive research, with its current evaluation focused on the Mac-based implementation rather than production-level Android or iOS deployment.

### Software Requirements {#software-requirements}

**Table 1**. *Software and Descriptions*

| Software  | Features | Purpose |
| :---- | :---- | :---- |
| Python | Core programming language | Used for system development |
| PyTorch | Deep learning framework | Model training and inference |
| Apple Vision | On-device hand, face, and upper-body landmark detection  | Primary visual landmark extraction for sign language input  |
| MediaPipe  | Mouth landmark detection  | Supplemental extraction of mouth landmarks for selected non-manual visual evidence  |
| OpenCV | Image and video processing | Handles camera capture, frame processing, and hand-crop preparation  |
| Squeezeformer | Efficient spatial-temporal transformer backbone | Recognition model for sign gesture classification and temporal sequence modeling  |
| MobileCLIP2  | Lightweight image embedding model  | Generates visual embeddings from left-hand, right-hand, and combined-hand RGB crops  |
| T5-efficient-tiny  | Lightweight text-to-text transformer  | Rephrases recognized gloss sequences into English text  |
| Text-to-Speech (TTS)  | Local speech synthesis  | Converts generated English text into spoken output  |

### Hardware Requirements {#hardware-requirements}

**Table 2\.** *Hardware Components and Descriptions*

| Hardware  | Features | Purpose |
| :---- | :---- | :---- |
| Mac Computer  | Multi-core processor, sufficient RAM, built-in or connected camera support  | Primary device used for system development, testing, and evaluation  |
| Camera / Webcam  | RGB video capture  | Captures live sign language input for visual feature extraction and recognition  |
| Local Storage  | Sufficient storage capacity  | Stores datasets, extracted features, trained model weights, logs, and supporting project resources  |
| Optional GPU / Accelerated Hardware  | Hardware acceleration for computational workloads  | Speeds up model training and other computationally intensive development tasks when available  |

### Peopleware {#peopleware}

*Non-Sign Language Users*  
The primary intended users of the system are non-sign language users within the Leyte Normal University community. These include students, faculty members, instructors, university administrators and staff, medical and guidance personnel, and IGP stall administrators or stall owners. The system is designed to assist these users in understanding signed messages by translating sign language input into English text through a mobile application interface. In this way, ATLAS serves as an assistive communication tool that helps reduce communication barriers between sign language users and the hearing community across academic, administrative, service, and daily campus interactions.

*Developers*  
The developers are the researchers responsible for designing, developing, and evaluating the core components of the system. Their responsibilities include collecting and preparing the sign language dataset, implementing the Squeezeformer-T5 model, developing the translation pipeline, and conducting system testing and evaluation. They are also responsible for refining the model’s performance, ensuring the reliability of the translation process, and improving the overall usability of the system based on feedback and experimental results. Through these efforts, the developers aim to create an assistive tool that is accurate, functional, and relevant to the communication needs of the Leyte Normal University community.

# **CHAPTER 4** {#chapter-4}

**METHODOLOGY, RESULTS, AND DISCUSSION**  
This chapter presents the methodology followed in the development, implementation, and evaluation of ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. It describes the requirements, development process, system design, implementation, and evaluation procedures used to determine whether the proposed system meets its intended objectives. The chapter also presents and discusses the results obtained from the recognition, translation, and software quality evaluation of the developed system.

## **4.1 Requirements Analysis** {#4.1-requirements-analysis}

The requirements analysis identifies the necessary requirements that guide the development and evaluation of ATLAS. It outlines the expected functions, performance, and overall system behavior needed to meet the objectives and intended scope of the proposed system.

### *Functional Requirements* {#functional-requirements}

**Table 3\.** *Functional Requirements*

| Requirement  | Description |
| :---- | :---- |
|  |  |
| Keypoint  Extraction | The system must extract relevant body and gesture keypoints from captured sign language input and convert them into structured coordinate-based data for further processing. |
| Sign Gesture Recognition | The system must recognize sign gestures from the extracted visual features using the multimodal Squeezeformer-based recognition model within the predefined 100-gloss vocabulary. |
| Multimodal Feature Processing | The system must process landmark-based skeletal information and RGB hand-crop visual information and combine their outputs for sign recognition. |
| Activity Detection and Verification | The system must identify and verify candidate sign activities from the camera stream while reducing duplicate or unstable predictions. |
| Gloss Buffering | The system must append verified recognized signs to an ordered gloss buffer for subsequent utterance processing. |
| Utterance Completion | The system must allow the user to complete an utterance through the designated Finish button or the defined two-open-palm gesture. |
| Gloss-to-English Rephrasing | The system must convert the accumulated gloss sequence into understandable English text using the fine-tuned T5-efficient-tiny model. |
| Speech Output | The system must provide synthesized speech from the generated English text through local text-to-speech functionality. |

### *Performance Requirements* {#performance-requirements}

**Table 4\.** *Performance Requirements*

| Requirement  | Description |
| :---- | :---- |
| Recognition Performance | The Squeezeformer-based recognition model shall achieve measurable performance in classifying the predefined 100-gloss vocabulary, evaluated using Top-1 accuracy, Top-5 accuracy, and Macro-F1 score. |
| Translation Performance | The T5-efficient-tiny rephrasing component shall generate understandable English outputs from recognized gloss sequences, evaluated using BLEU and human translation ratings. |
| Processing Efficiency | The system shall process captured sign input and perform recognition, activity verification, gloss buffering, and translation within a practical turnaround time for assisted communication. |
| Streaming Stability | The system shall maintain consistent candidate detection and verification during camera-based input while minimizing duplicate or unstable gloss predictions. |
| System Reliability | The system shall maintain stable operation during visual capture, feature extraction, recognition, gloss buffering, translation, and speech output without unexpected interruption. |
| Functional Suitability | The system shall provide the functions necessary to perform its intended sign recognition and English rephrasing tasks based on the defined system scope. |
| Usability | The system shall provide an understandable and manageable interaction flow for users performing sign input and viewing or hearing the generated English output. |
| Evaluation Compliance | The developed system shall be evaluated according to applicable criteria of ISO/IEC 25010:2023, particularly functional suitability, performance efficiency, usability, and reliability. |

### 	*Software and Hardware Requirements* {#software-and-hardware-requirements}

**Table 5\.** *Software and Hardware Requirements*

| Requirement  | Description |
| :---- | :---- |
| Software Frameworks  | The system utilizes Python and PyTorch for model development, training, and inference, together with OpenCV for image and video processing, Apple Vision for primary visual landmark extraction, MediaPipe for supplemental mouth landmark extraction, MobileCLIP2 for hand-crop visual embeddings, and T5-efficient-tiny for English rephrasing.  |
| Processing Environment  | The current implementation requires a Mac-based computing environment capable of running the visual extraction, multimodal recognition, streaming activity control, gloss buffering, and English rephrasing components.  |
| Camera Input  | The system requires a built-in or external camera capable of capturing video input for hand, face, upper-body landmark extraction, and hand-crop generation.  |
| Model Execution  | The system must support execution of the Squeezeformer-based recognition models, MobileCLIP2 visual encoder, learned fusion head, and T5-efficient-tiny model during system operation and evaluation.  |
| Storage Requirement  | The system must provide adequate local storage for datasets, extracted landmark data, RGB samples, trained model weights, logs, and other supporting project resources.  |
| Development and Training Resources  | Additional computing resources may be used to support model training, experimentation, and evaluation, depending on the computational requirements of the development environment.  |

	

### *Safety Requirements* {#safety-requirements}

**Table 6\.** *Safety Requirements* 

| Requirement  | Description |
| :---- | :---- |
| Physical User Safety | The system interface and camera-based interaction must allow users to maintain awareness of their surroundings while performing sign language input.  |
| Device Temperature Control | The system should be operated under conditions that support stable processing without excessive device heating, performance degradation, or interruption during extended use.  |
| Operational Ergonomics | Camera-based gesture capture should allow users to perform natural signing movements without requiring uncomfortable or prolonged static poses.  |

### *Security Requirements* {#security-requirements}

**Table 7\.** *Security Requirements* 

| Requirement  | Description |
| :---- | :---- |
| Data Privacy  | User sign input, extracted visual information, translated output, and other interaction data must be handled in accordance with appropriate privacy and data-protection practices.  |
| Audit Trail Protection  | System logs containing recognition events, confidence values, and reference recordings must be protected from unauthorized access, modification, or disclosure.  |
| Data Retention and Deletion  | The system should provide documented procedures for the retention and deletion of recorded interaction data and logs, particularly when these contain identifiable or sensitive information.  |
| Model Integrity  | Trained model files and supporting resources must be stored and managed in a manner that protects the integrity of the implemented recognition and English rephrasing pipeline.  |
| Confidentiality of Interaction  | Sign input, generated English output, and other interaction information must be handled within the system environment as much as possible to reduce unnecessary exposure of sensitive communication data.  |

## **4.2 Design of Software, Systems, Product, and/or Processes**  {#4.2-design-of-software,-systems,-product,-and/or-processes}

### Systems Development Life Cycle {#systems-development-life-cycle}

The project utilized the Modified Waterfall (Sashimi) Model, which is characterized by overlapping development phases that allow feedback and refinement between adjacent stages. This approach was selected because the development of ATLAS involved related activities that could be performed concurrently, particularly the refinement of the sign language recognition and English rephrasing components.

Using this model, the researchers proceeded through the major stages of planning and requirements analysis, system design, development, testing, and evaluation. The overlapping nature of the model allowed the team to refine the recognition pipeline while developing and integrating the translation component, helping ensure that the recognized gloss output was compatible with the English rephrasing process. The development process also included iterative testing and refinement of the system's visual extraction, multimodal Squeezeformer recognition, Reel activity controller, gloss buffering, T5-efficient-tiny rephrasing, and speech output components.

The development cycle concluded with system testing and evaluation based on the defined technical performance measures and selected ISO/IEC 25010:2023 quality characteristics.

*(See table on the next page)*

**Figure 1\.** *SDLC Modified Waterfall (Sashimi) Model*

### *Planning and Requirement Analysis*   {#planning-and-requirement-analysis}

This phase focused on identifying the communication needs of sign language users and non-sign language users and defining the technical and functional requirements of ATLAS. The researchers established the system scope, including the fixed 100-gloss ASL vocabulary, multimodal visual processing, Squeezeformer-based sign recognition, streaming activity control, gloss buffering, English rephrasing, and speech output.

The requirements identified during this phase served as the basis for the succeeding design and development activities. An Activity List was prepared to organize the major tasks and their dependencies, while the GANTT and PERT charts were used to plan the project schedule and monitor the sequence and overlap of development activities.

#### 

#### **Activity List** {#activity-list}

An activity list is a fundamental software project management tool that identifies all necessary tasks and their logical sequence to ensure a structured development workflow. It serves as a comprehensive guide that specifies the required resources, the estimated duration for each task, and the specific dependencies or requirements needed to move through the project phases successfully.

**Table 8\.** *Activity List*

| ID | TASKS | START | END | PREDECESSOR | DURATION (DAYS) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Phase 1: Planning |  |  |  |  |  |
| A | Project Proposal Preparation | Jan. 20 | Jan. 22 | \- | 3 |
| B | Proposal Approval and Adviser Assignment | Jan. 23 | Jan. 24 | A | 2 |
| C | Review of Related Literature | Jan. 25 | Jan. 29 | B | 5 |
| D | Identification of Research Problem | Jan. 25 | Jan. 27 | B | 3 |
| E | Chapter 1: Writing | Jan. 28 | Feb. 1 | C, D | 5 |
| F | Chapter 1: Submission | Feb. 2 | Feb. 2 | E | 1 |
| G | Chapter 1: Adviser Feedback | Feb. 3 | Feb. 4 | F | 2 |
| H | Chapter 1: Revision | Feb. 5 | Feb. 7 | G | 3 |
| I | Chapter 1: Resubmission | Feb. 8 | Feb. 8 | H | 1 |
| Phase 2: Designing |  |  |  |  |  |
| Y | Definition of System Scope and Features | Jan. 25 | Jan. 28 | B | 4 |
| Z | Dataset Planning and Design | Jan. 29 | Feb. 3 | J | 6 |
| J | Chapter 2: Writing | Feb. 9 | Feb. 13 | C, D | 5 |
| K | Chapter 2: Submission | Feb. 14 | Feb. 14 | L | 1 |
| L | Chapter 2: Adviser Feedback | Feb. 15 | Feb. 16 | M | 2 |
| M | Chapter 2: Revision | Feb. 17 | Feb. 19 | N | 3 |
| N | Chapter 2: Resubmission | Feb. 20 | Feb. 20 | O | 1 |
| O | Chapter 3: Writing | Feb. 9 | Feb. 14 | J, K | 6 |
| P | Chapter 3: Submission | Feb. 15 | Feb. 15 | Q | 1 |
| Q | Chapter 3: Adviser Feedback | Feb. 16 | Feb. 17 | R | 2 |
| R | Chapter 3: Revision | Feb. 18 | Feb. 20 | S | 3 |
| S | Chapter 3: Resubmission | Feb. 21 | Feb. 21 | T | 1 |
| T | Chapter 4: Writing | Feb. 9 | Feb. 14 | J, K | 6 |
| U | Chapter 4: Submission | Feb. 15 | Feb. 15 | V | 1 |
| V | Chapter 4: Adviser Feedback | Feb. 16 | Feb. 17 | W | 2 |
| W | Chapter 4: Revision | Feb. 18 | Feb. 20 | X | 3 |
| X | Chapter 4: Resubmission | Feb. 21 | Feb. 21 | Y | 1 |
| Phase 3: Development |  |  |  |  |  |
| AA | Data Cleaning and Preparation | Feb. 4 | Feb. 11 | K | 8 |
| AB | Data Validation and Label Checking | Feb. 12 | Feb. 17 | AA | 6 |
| AC | Feature Engineering and Processing | Feb. 18 | Feb. 25 | AB | 8 |
| AD | Initial Model Development | Feb. 26 | Mar. 7 | AC | 10 |
| AE | Model Testing and Iteration | Mar. 8 | Mar. 19 | AD | 12 |
| AF | Technical Documentation Preparation | Mar. 20 | Mar. 25 | AE | 6 |
| AG | Pre-Oral Defense | Apr. 6 | Apr. 6 | I, P, U, Z, AF | 1 |
| AH | Post Pre-Oral Revisions | Apr. 7 | Apr. 11 | AG | 5 |
| AI | Submission to Research Ethics Committee (REC) | May 4 | May 4 | AH | 1 |
| AJ | REC Initial Review Process | May 5 | Sep. 25 | AI | 144 |
| AK | REC Revisions and Compliance | Sep. 26 | Oct. 2 | AJ | 7 |
| AL | Ethics Clearance Approval | Oct. 5 | Oct. 5 | AK | 1 |
| Phase 4: Testing |  |  |  |  |  |
| AM | Mobile Application Development | Oct. 3 | Oct. 5 | AE | 3 |
| AN | System Evaluation and Usability Testing | Oct. 6 | Oct. 9 | AL, AM | 4 |
| AO | Results Analysis and Interpretation | Oct. 10 | Oct. 13 | AN | 4 |
| Phase 5: Implementation |  |  |  |  |  |
| AP | Chapter 5: Writing | Oct. 14 | Oct. 14 |  | 1 |
| AQ | Final Manuscript Editing | Oct. 12 | Oct. 16 | AO | 5 |
| AR | Chapter 5 Adviser Feedback | Oct. 15 | Oct. 15 | AP | 1 |
| AS | Formatting and Plagiarism Checking | Oct. 17 | Oct. 18 | AQ, AR | 2 |
| AT | Adviser Final Approval | Oct. 19 | Oct. 19 | AS | 1 |
| AU | Mock Final Defense | Oct. 20 | Oct. 20 | AT | 1 |
| AV | Final Defense | Oct. 21 | Oct. 23 | AU | 3 |

**GANTT Chart**  
The project was managed using a GANTT Chart to establish a clear timeline and track task dependencies from data collection to final deployment. This was paired with the overlapping nature of the Sashimi Model, where development stages such as hand-tracking and translation logic were worked on in parallel. This combined approach ensured that all milestones were met in a logical sequence while allowing for continuous feedback and immediate bug fixes throughout the development process, ensuring that the output of the recognition front-end correctly informed the translation engine during construction.

**Table 9\.** *GANTT Chart*  
*(See table on the next page)*

#### **PERT Chart** {#pert-chart}

The researchers employed a Program Evaluation and Review Technique (PERT) chart to visualize task dependencies, durations, and the critical path within the systems development life cycle. This combination allows for structured planning and continuous monitoring of the overlapping phases required to develop the Squeezeformer-T5 sign language recognition and translation system.

*(See diagram on the next page)*

**Figure 2\.** *PERT Chart of the Proposed System*  
*Design*   
The system architecture integrates skeletal landmark extraction, Squeezeformer-based sign recognition, and T5-based sign-to-English translation within a modular pipeline for sign language understanding. The current implementation follows a mobile-based workflow intended to support non-sign language users in understanding signed input through English text output. A modular approach was employed to separate feature processing, recognition, and translation, while still allowing future expansion for additional vocabularies, datasets, and deployment environments.

#### **Context Diagram**  {#context-diagram}

The context diagram illustrates the high-level relationship between the proposed system and its primary external entity, the Non-Sign Language User. It shows how the system provides translated English text output to support the user in understanding captured sign language input. The diagram focuses on the direct interaction between the system and the Non-Sign Language User within the defined system boundary.

 

**Figure 3\.** *Context Diagram of the Proposed System*

#### **Data Flow Diagram** {#data-flow-diagram}

The Data Flow Diagram (DFD) presents the flow of data within the proposed sign language recognition and English translation system. The process begins when completed sign language input is captured by the application and passed through the necessary processing stages for landmark extraction, recognition, and translation. The system uses the available gesture dataset and trained models to extract relevant hand and body keypoints through MediaPipe and Apple Vision, process the extracted landmark sequences through the Squeezeformer-based recognition model, and generate the corresponding English text output through the T5-based translation model. The translated English text is then displayed to the Non-Sign Language User to support message understanding. In addition, the system may also allow the Non-Sign Language User to enter a typed text response, which can be displayed as part of the communication support process.

  

**Figure 4\.** *Data Flow Diagram of the Proposed System*

#### **Use-case**  {#use-case}

The use case diagram illustrates the interaction between the **Non-Sign Language User** and the proposed **ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation**. It shows that the user can access the application, capture completed sign language input, process the input for recognition and translation, view the generated English text output, and enter a typed text response when necessary. This diagram presents the main functions available to the direct user of the system and reflects the system’s assistive communication workflow.  
	

	

**Figure 5\.** *Proposed System Use-Case Diagram*

#### **Flowchart** {#flowchart}

The flowchart illustrates the overall process of the proposed ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. The process starts when the application is opened and completed sign language input is captured through the device camera. The captured input is then processed through MediaPipe and Apple Vision for hand and body landmark extraction, followed by preprocessing and normalization of the extracted coordinate-based data. The processed landmark sequence is analyzed by the Squeezeformer-based recognition model to identify the corresponding sign or sign-label sequence. The recognized output is then passed to the T5-based translation model, which generates grammatically understandable English text. The translated English text is displayed to the Non-Sign Language User to support message understanding. If needed, the user may also enter a typed text response, which is then displayed by the system as part of the communication support process.

*(See diagram on the next page)*

	

**Figure 6\.** *Proposed System Flowchart*

# **REFERENCES** {#references}

Amiruzzaman, S., Batchu, R. M., Amiruzzaman, M., Ngo, L., & Dewan, M. A. A. (2026). ASL recognition and game-based interaction: A machine learning–driven, gamified and accessible vocabulary learning system for Deaf learners. Computers, 15(5), 299\. https://doi.org/10.3390/computers15050299

Apple. (n.d.-a). Detecting hand poses with Vision. Apple Developer Documentation. https://developer.apple.com/documentation/vision/detecting-hand-poses-with-vision

Apple. (n.d.-b). Detecting human body poses in images. Apple Developer Documentation. https://developer.apple.com/documentation/vision/detecting-human-body-poses-in-images

Apple. (n.d.-c). Detect body and hand pose with Vision. Apple Developer. https://developer.apple.com/videos/play/wwdc2020/10653/

Boháček, M., & Hrúz, M. (2022). Sign pose-based transformer for word-level sign language recognition. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (pp. 182–191). https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek\_Sign\_Pose-Based\_Transformer\_for\_Word-Level\_Sign\_Language\_Recognition\_WACVW\_2022\_paper.pdf

Camgoz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign Language Transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10023–10033). https://openaccess.thecvf.com/content\_CVPR\_2020/html/Camgoz\_Sign\_Language\_Transformers\_Joint\_End-to-End\_Sign\_Language\_Recognition\_and\_Translation\_CVPR\_2020\_paper.html

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Castro-Ros, A., Pellat, M., Robinson, K., … Wei, J. (2024). Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70), 1–53. https://jmlr.org/papers/v25/23-0870.html

DeGrace, P., & Stahl, L. H. (1990). Wicked problems, righteous solutions: A catalogue of modern software engineering paradigms. Yourdon Press. https://archive.org/details/wickedproblemsri0000degr

Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4690–4699). https://openaccess.thecvf.com/content\_CVPR\_2019/html/Deng\_ArcFace\_Additive\_Angular\_Margin\_Loss\_for\_Deep\_Face\_Recognition\_CVPR\_2019\_paper.html

Desai, A., Berger, L., Minakov, F. O., Milan, V., Singh, C., Pumphrey, K., Ladner, R. E., Daumé III, H., Lu, A. X., Caselli, N., & Bragg, D. (2023). ASL Citizen: A community-sourced dataset for advancing isolated sign language recognition. arXiv. https://arxiv.org/abs/2304.05934

Eunice, J., Andrew, J., Sei, Y., & Hemanth, D. J. (2023). Sign2Pose: A pose-based approach for gloss prediction using a transformer model. Sensors, 23(5), 2853\. https://doi.org/10.3390/s23052853

Faghri, F., Anasosalu Vasu, P. K., Koc, C., Shankar, V., Toshev, A., Tuzel, O., & Pouransari, H. (2025). MobileCLIP2: Improving multi-modal reinforced training. Transactions on Machine Learning Research. https://openreview.net/forum?id=WeF9zolng8

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning (pp. 369–376). https://www.cs.toronto.edu/\~graves/icml\_2006.pdf

Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for speech recognition. Proceedings of Interspeech 2020, 5036–5040. https://doi.org/10.21437/Interspeech.2020-3015

Hugging Face. (n.d.). Transformers documentation. https://huggingface.co/docs/transformers/index

International Organization for Standardization. (2023). ISO/IEC 25010:2023: Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model. https://www.iso.org/standard/78176.html

Kim, S., Gholami, A., Shaw, A., Lee, N., Mangalam, K., Malik, J., Mahoney, M. W., & Keutzer, K. (2022). Squeezeformer: An efficient Transformer for automatic speech recognition. In Advances in Neural Information Processing Systems, 35\. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract-Conference.html

Lee, T., Oh, Y., & Lee, K. M. (2023). Human part-wise 3D motion context learning for sign language recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 20740–20750). https://openaccess.thecvf.com/content/ICCV2023/html/Lee\_Human\_Part-wise\_3D\_Motion\_Context\_Learning\_for\_Sign\_Language\_Recognition\_ICCV\_2023\_paper.html

Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1459–1469). https://dxli94.github.io/WLASL/

Lin, K., Wang, X., Zhu, L., Sun, K., Zhang, B., & Yang, Y. (2023). Gloss-free end-to-end sign language translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 12904–12916). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.acl-long.722

Moryossef, A., Yin, K., Neubig, G., & Goldberg, Y. (2021). Data augmentation for sign language gloss translation. In Proceedings of the 1st International Workshop on Automatic Translation for Signed and Spoken Languages (pp. 1–11). Association for Machine Translation in the Americas. https://aclanthology.org/2021.mtsummit-at4ssl.1/

Müller, M., Jiang, Z., Moryossef, A., Rios, A., & Ebling, S. (2023). Considerations for meaningful sign language machine translation based on glosses. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 682–693). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.acl-short.60

NVIDIA. (n.d.). CUDA Toolkit documentation. NVIDIA Developer. https://docs.nvidia.com/cuda/

OpenCV. (n.d.). OpenCV documentation. https://docs.opencv.org/

Python Software Foundation. (n.d.). Python documentation. https://docs.python.org/

PyTorch. (n.d.). PyTorch documentation. https://docs.pytorch.org/docs/stable/index.html

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text Transformer. Journal of Machine Learning Research, 21(140), 1–67. https://jmlr.org/papers/v21/20-074.html

Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey. Expert Systems with Applications, 164, 113794\. https://doi.org/10.1016/j.eswa.2020.113794

Renjith, S., Varghese, A., Rashmi, M., & Poorna, S. S. (2026). Transformer-based motion-visual integrated fusion for isolated sign language recognition. Computers and Electrical Engineering, 130, 110902\. https://doi.org/10.1016/j.compeleceng.2025.110902

Renjith, S., Varghese, A., & Poorna, S. S. (2026). An efficient real-time spatio-temporal adaptive motion pattern framework for isolated sign language recognition (RT-STAMP-SLR). Discover Artificial Intelligence, 6, 713\. https://doi.org/10.1007/s44163-026-01429-3

Syulistyo, A. R., Tanaka, Y., Pramanta, D., Fuengfusin, N., & Tamukoh, H. (2025). Low-cost computation for isolated sign language video recognition with multiple reservoir computing. PLOS ONE, 20(7), e0322717. https://doi.org/10.1371/journal.pone.0322717

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818–2826). https://doi.org/10.1109/CVPR.2016.308

Violet, I. M. M., & Leena Sri, R. (2025). A comprehensive survey on recent advances and challenges in sign language recognition systems. Discover Artificial Intelligence, 5, 419\. https://doi.org/10.1007/s44163-025-00629-7

Woods, L. T., & Rana, Z. A. (2023). Modelling sign language with encoder-only transformers and human pose estimation keypoint data. Mathematics, 11(9), 2129\. https://doi.org/10.3390/math11092129

World Health Organization. (2026, March 3). Deafness and hearing loss. https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss

Yin, K., Moryossef, A., Hochgesang, J., Goldberg, Y., & Alikhani, M. (2021). Including signed languages in natural language processing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 7347–7360). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.acl-long.570

Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking. arXiv. https://arxiv.org/abs/2006.10214

Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. International Conference on Learning Representations. https://openreview.net/forum?id=r1Ddp1-Rb

Zheng, Z., Wang, Q., Yang, D., Wang, Q., et al. (2022). L-Sign: Large-vocabulary sign gestures recognition system. IEEE Transactions on Human-Machine Systems, 52(2), 290–301. https://doi.org/10.1109/THMS.2022.3146787

