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

[Table of Contents](#table-of-contents)

[List of Figures](#list-of-figures)

[List of Tables](#list-of-tables)

[CHAPTER 1: Introduction](#chapter-1)

[1.1 Introduction](#1.1-introduction)

[1.2 Purpose and Description](#1.2-purpose-and-description)

[1.3 Objective of the Study](#1.3-objective-of-the-study)

[1.4 Scope and Limitations](#1.4-scope-and-limitations)

[CHAPTER 2: Review of Related Literature and Systems](#chapter-2)

[2.1 Related Literature](#2.1-related-literature)

[2.2 Related Systems](#2.2-related-systems)

[CHAPTER 3: Technical Backgrounds](#chapter-3)

[3.1 Current System](#3.1-current-system)

[3.2 Proposed System](#3.2-proposed-system)

[Software Requirements](#software-requirements)

[Hardware Requirements](#hardware-requirements)

[Peopleware](#peopleware)

[CHAPTER 4: Methodology, Results and Discussion](#chapter-4)

[4.1 Requirements Analysis](#4.1-requirements-analysis)

[Functional Requirements](#functional-requirements)

[Performance Requirements](#performance-requirements)

[Software and Hardware Requirements](#software-and-hardware-requirements)

[Safety Requirements](#safety-requirements)

[Security Requirements](#security-requirements)

[4.2 Design of Software, Systems, Product, and/or Processes](#4.2-design)

[Systems Development Life Cycle](#systems-development-life-cycle)

[Planning and Requirement Analysis](#planning-and-requirement-analysis)

[Activity List](#activity-list)

[Gantt Chart](#gantt-chart)

[PERT Chart](#pert-chart)

[Design](#design)

[Context Diagram](#context-diagram)

[Data Flow Diagram](#data-flow-diagram)

[Use-case](#use-case)

[Flowchart](#flowchart)

[4.3 Data Preparation and Model Development](#4.3-data-preparation-and-model-development)

[4.4 Results and Discussion](#4.4-results-and-discussion)

[REFERENCES](#references)

# **List of Figures** {#list-of-figures}

**Figure 1:** ATLAS Live Recognition and English-Output Flow

**Figure 2:** HELLO HOW YOU Recognition Timeline

**Figure 3:** SDLC Modified Waterfall (Sashimi) Model

**Figure 4:** PERT Chart of the Proposed System

**Figure 5:** Context Diagram of the Proposed System

**Figure 6:** Data Flow Diagram of the Proposed System

**Figure 7:** Proposed System Use-Case Diagram

**Figure 8:** Proposed System Flowchart

**Figure 9:** Streaming Recognition Decisions

**Figure 10:** Model and Deployment Selection

**Figure 11:** Landmark Extractor Comparison

**Figure 12:** Recognition Input Comparison

**Figure 13:** Landmark Architecture Comparison

**Figure 14:** Recognition Model Family Comparison

**Figure 15:** Streaming Recognition Comparison

**Figure 16:** English-Output Comparison

**Figure 17:** Language-Model Execution Screening

**Figure 18:** Model Export and iPhone Integration

**Figure 19:** Landmark Recognition Training

**Figure 20:** Recognition Adaptation to Sign Intervals

**Figure 21:** English-Model Fine-Tuning

# **List of Tables** {#list-of-tables}

**Table 1:** Software and Descriptions

**Table 2:** Hardware Components and Descriptions

**Table 3:** Functional Requirements

**Table 4:** Performance Requirements

**Table 5:** Software and Hardware Requirements

**Table 6:** Safety Requirements

**Table 7:** Security Requirements

**Table 8:** Activity List

**Table 9:** GANTT Chart

**Table 10:** Data Preparation and Training Methods

**Table 11:** Landmark Extractor Comparison

**Table 12:** Recognition Input Comparison

**Table 13:** Landmark Architecture Comparison

**Table 14:** Recognition Model Family Comparison

**Table 15:** Streaming Recognition Comparison

**Table 16:** English-Output Comparison

**Table 17:** Language-Model Execution Screening

**Table 18:** Neural Models and Their Core ML Exports

**Table 19:** Recognition Measures Before and After Core ML Conversion

# **CHAPTER 1** {#chapter-1}

This chapter presents the background of the study on the development of ATLAS, a Squeezeformer-based system for sign language recognition and English translation on the iPhone. It discusses developments in artificial intelligence and deep learning, their application to sign language processing, and the challenges involved in bringing these approaches into practical communication settings. The chapter also outlines the purpose, objectives, scope, and limitations of the study, highlighting the need for accessible communication support within an academic environment.

## **1.1 Introduction** {#1.1-introduction}

Communication is a fundamental human need and a key driver of educational and social inclusion. However, communication barriers persist between sign language users and individuals who do not understand sign language. Hearing loss affects a substantial portion of the global population, and inadequate communication support can limit access to education, employment, healthcare, and everyday social participation (World Health Organization, 2026). In academic settings such as Leyte Normal University (LNU), interactions between Deaf or hard-of-hearing students and non-signing faculty or peers may rely on written notes, typed text exchanges, or improvised gestures. Although these approaches provide ways to exchange information, repeatedly shifting between signing and writing can interrupt conversational flow and require additional effort from both participants. The need for communication support therefore extends beyond making information available to helping people convey and understand a message during an interaction.

Recent advances in artificial intelligence, computer vision, and natural language processing offer promising avenues for automated sign language recognition and translation. One approach is the use of landmark-based skeletal representations, which describe signing through selected coordinate points on the hands, face, and upper body. By tracking these points across frames, landmark extraction captures movement and spatial relationships in a compact form that sequence models can process. Frameworks such as Apple Vision and MediaPipe provide tools for obtaining these landmarks from camera images (Apple, n.d.-a; Zhang et al., 2020). These representations allow recognition to focus on the signer's movement while reducing the amount of visual information passed to the temporal model.

However, landmark coordinates alone may omit subtle handshape details during fast motion, hand occlusion, or complex articulation. A representation of joint positions may describe where a hand moves without preserving all of the visual details needed to distinguish it from a similar sign. Cropped hand images provide complementary evidence by retaining the appearance of the signing hands. Combining landmark tracks with features extracted from these images allows a multimodal recognition model to consider both movement and handshape. This combination is particularly relevant when the system must distinguish signs using information that neither input captures fully on its own.

Recognizing signs is only the first step in bridging the communication gap. Signed languages possess grammatical structures, spatial relationships, and non-manual signals that do not map word-for-word to spoken or written English (Camgoz et al., 2020; Müller et al., 2023). A sequence of glosses, or written labels representing recognized signs, provides an intermediate representation of the visual input. However, displaying that sequence alone does not perform the language processing needed to express it in English. A recognition-and-generation approach therefore connects the identification of signs across time with a language component that produces English from the recognized sequence.

To address sequence modeling, hybrid architectures combine self-attention with convolution so that a model can relate information across frames while also capturing local patterns. Squeezeformer brings these mechanisms together in an architecture developed for efficient temporal processing (Kim et al., 2022). For target language generation, the Text-to-Text Transfer Transformer (T5) offers a unified sequence-to-sequence framework in which an input sequence is mapped to generated text (Raffel et al., 2020; Chung et al., 2024). These approaches provide complementary foundations for a system that must first interpret visual changes during signing and then express its recognized input in English.

Despite technological progress, practical deployment presents challenges beyond recognizing signs in isolated, pre-segmented video clips. A live application must process incoming frames within the device's computational capacity, locate candidate sign intervals, and determine when a prediction is ready to be accepted. It must also connect recognized glosses with English generation, since gloss sequences do not fully represent the grammar and meaning of a signed message (Müller et al., 2023). To address these challenges, this study develops ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. The application integrates Apple Vision landmarks, MobileCLIP2 hand-image features, a multimodal Squeezeformer recognizer, and Streaming Sign Recognition through boundary detection and segmental decoding. A fine-tuned T5-efficient-tiny model generates English from the accepted gloss sequence, with message completion controls and local text-to-speech output supporting the interaction. Through this architecture, the study aims to contribute an assistive communication tool that brings sign recognition and English output together on the iPhone for use in academic environments such as LNU.

## **1.2 Purpose and Description** {#1.2-purpose-and-description}

The primary purpose of this study is to design, implement, and empirically evaluate ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. The application is intended to assist non-signing faculty, staff, and peers at Leyte Normal University in understanding messages expressed through the supported American Sign Language (ASL) vocabulary. By connecting camera-based recognition with English text and spoken output, the study seeks to provide another means of communication during interactions in which the participants do not share knowledge of sign language.

ATLAS operates with a 100-sign ASL prototype vocabulary containing foundational signs and organizes its processing into four cooperating functional components. First, Visual Extraction obtains hand, face, and upper-body landmarks through Apple Vision and extracts features from cropped hand images through MobileCLIP2. These inputs represent different aspects of the same signing activity: the landmarks describe positions and movement, while the hand images contribute appearance information. Streaming Sign Recognition then brings these inputs together with temporal boundary estimates to evaluate candidate sign intervals.

Within Streaming Sign Recognition, the boundary model estimates where candidate signs occur, the Squeezeformer-based recognizer evaluates their content, and the segmental decoder selects a sequence of accepted signs. Gloss Management maintains those outputs in a visible gloss buffer, preserving the recognized order as signing continues. English Generation and Speech Output uses the accumulated glosses to prepare English text through T5-efficient-tiny. This division of responsibilities allows the visual, temporal, and language components to work together while keeping their roles clear within the application.

For sequence recognition, ATLAS adopts Squeezeformer as its temporal architecture. Its combination of attention and convolution provides a basis for processing both local changes and relationships across frames (Kim et al., 2022). In the application, this processing is applied to candidate sign intervals rather than treating each camera frame as a complete sign. Combined landmark and hand-image evidence is used to identify the interval's content, while the boundary model and decoder manage how that prediction becomes part of the recognized sequence.

For English generation, ATLAS uses a fine-tuned T5-efficient-tiny model. The text-to-text approach supplies a framework for learning the relationship between an input sequence and its English rendering (Raffel et al., 2020; Chung et al., 2024). In ATLAS, the input is the gloss sequence maintained by the recognition components. As this sequence grows, the language model prepares English updates, connecting the recognition output with text that can be presented to the non-signing recipient.

During Live use, the signer performs signs within the camera view while the system evaluates candidate intervals and appends accepted glosses to the buffer. When the message is complete, the signer selects the Finish button or holds both open palms for one second. The system then finalizes the remaining English text and provides local speech output. Core ML runs the exported models on the iPhone, allowing this interaction to take place through local processing. The Glosses reference, Practice activities, and History complement Live recognition by allowing users to inspect the vocabulary, practice supported signs, and review saved sessions within the same application.

To evaluate ATLAS, isolated-sign data, sign-sequence data, and gloss-to-English pairs serve the respective recognition and language tasks. Technical assessment considers recognition accuracy, errors in the recognized sequence, English-output quality, and processing time, so that the contribution of each part can be examined within its intended role. The ISO/IEC 25010:2023 quality framework provides criteria for assessing application behavior and quality. Together, these assessments connect model performance with the operation of the application as a communication support tool.

## **1.3 Objective of the Study** {#1.3-objective-of-the-study}

The primary objective of this study is to design, implement, and evaluate ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation, an iPhone application that recognizes a 100-sign ASL prototype vocabulary and generates English text and local speech from the recognized sequence.

Specifically, this study aims to:

* Develop a multimodal recognition and translation pipeline combining Apple Vision landmark extraction, MobileCLIP2 hand-image features, Squeezeformer-based temporal recognition, and a fine-tuned T5-efficient-tiny language generation model.
* Evaluate the performance of the recognition and English-generation components using technical metrics, including Top-1 accuracy, Top-5 accuracy, word error rate, and BLEU across their respective evaluation data.
* Implement the trained models in an iPhone application with Streaming Sign Recognition, gloss management, incremental English generation, message completion through the Finish button or one-second held-open-palms gesture, and local text-to-speech output.
* Validate the developed system using the ISO/IEC 25010:2023 quality framework and appropriate software evaluation procedures, including black-box testing for functional behavior and white-box testing for internal logic and processing flow.

## **1.4 Scope and Limitations** {#1.4-scope-and-limitations}

This study focuses on the design, implementation, and empirical evaluation of ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. Its scope centers on a 100-sign American Sign Language (ASL) prototype vocabulary containing foundational signs, with academic environments such as Leyte Normal University providing the intended context of use. The application brings together Visual Extraction, Streaming Sign Recognition, Gloss Management, and English Generation and Speech Output. These components cover the processing of camera input, the identification and acceptance of signs, and the generation of English text and local speech from the recognized sequence.

The language scope is ASL within the selected vocabulary and English output derived from recognized glosses. Consequently, the study does not establish recognition of Filipino Sign Language or coverage of all ASL expressions and grammatical features. The selected vocabulary provides a defined basis for developing and examining the prototype. Expansion can build on that basis by adding signs relevant to particular communication situations and preparing the corresponding recognition data.

ATLAS processes signing as it arrives through the camera, using boundary estimates and recognition scores to maintain a sequence of accepted glosses. English generation proceeds as the sequence accumulates, while the Finish button or one-second held-open-palms gesture provides an explicit signal to finalize the remaining message. This interaction allows the signer to control completion while the system handles recognition and English preparation during signing. The study also covers the Glosses reference, Practice activities, and History as supporting functions of the iPhone application.

System recognition performance remains dependent on physical and environmental conditions. Effective tracking requires adequate ambient lighting, suitable camera framing, and an unobstructed view of the signer's upper body and hands. Rapid motion, hand occlusion, or unsuitable camera angles can affect the landmark and image information available to the recognizer. Facial landmarks form part of the visual representation, but general lip-reading and complete interpretation of non-manual grammar remain outside the defined functionality. These conditions shape how the camera input is obtained and how recognition results are interpreted.

The implementation uses Core ML to execute the exported models locally on the iPhone. Evaluation covers isolated-sign recognition, sign-sequence processing, English generation, and device execution, with each assessment addressing a different part of the system. Signer separation is used for isolated-sign evaluation, while sequence comparisons retain their documented recording groups. English-output assessment examines text produced from gloss input, and physical-device timing measures execution on the iPhone. Keeping these assessment roles distinct allows the study to relate its findings to the corresponding task and processing conditions.

# **CHAPTER 2** {#chapter-2}

**REVIEW OF RELATED LITERATURE AND SYSTEMS**  
This chapter reviews existing studies, systems, and technological developments relevant to ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. It discusses the foundations of sign language recognition, landmark-based representation, temporal modeling, hand-image features, multimodal fusion, and English generation. The chapter also examines how these approaches are brought into applications that receive signing through a camera and present recognized information to users. Throughout the discussion, related literature provides the basis for understanding the system's design choices, while related systems illustrate different ways of connecting recognition with practical interaction.

## **2.1 Related Literature** {#2.1-related-literature}


### **2.1.1 Sign Language Recognition and Translation**

Sign language recognition and sign language translation are related but distinct tasks in computer vision and natural language processing. Sign language recognition focuses on identifying signs or sign sequences from visual input, whereas sign language translation involves converting recognized sign information into a target spoken or written language. Sign languages contain their own linguistic structures, including manual and non-manual components, and therefore cannot always be converted into English through direct word-for-word substitution (Yin et al., 2021; Rastgoo et al., 2021). This distinction is important when developing systems that aim not only to recognize signs but also to produce understandable English output.

Joint recognition and translation can connect visual sequence modeling with gloss-level supervision, as demonstrated by Sign Language Transformers (Camgoz et al., 2020). At the same time, signed-language processing must account for linguistic structure beyond isolated visual gestures (Yin et al., 2021). These considerations motivate examining recognition and language generation as related tasks, each with its own input, output, and evaluation requirements.

The selection of a defined vocabulary is also an important consideration in sign language recognition. Desai et al. (2023) introduced ASL Citizen, a community-sourced dataset of isolated American Sign Language signs collected from Deaf and hard-of-hearing contributors in a variety of natural environments. The ASL Citizen dataset contains a large number of sign instances and was developed to support sign language recognition and dictionary-related applications. Its ASL Citizen 100 subset provides a standardized set of 100 glosses selected from the larger dataset based on factors including sample density, contributor agreement, and instance frequency (Desai et al., 2023).

The selection of the ASL Citizen 100 subset itself followed a structured, Deaf-centered methodology rather than an arbitrary sampling of frequent signs. Desai et al. (2023) described a four-step process in which candidate glosses were first cross-referenced against ASL-LEX, a linguistic database that documents the phonological properties of ASL signs, including handshape, location, and movement, to ensure that every selected gloss was phonologically well defined. Candidate signs were further filtered for dictionary utility, prioritizing terms with high real-world query frequency in digital ASL dictionaries. Recordings were then elicited through standardized prompt videos performed by fluent Deaf seed signers to maintain semantic consistency across the 52 contributors, and the final 100 glosses were ranked and retained according to instance density, or the number of available video samples per gloss across diverse signers. This methodology distinguishes ASL Citizen 100 from legacy 100-gloss benchmarks such as WLASL100 (Li et al., 2020\) and MS-ASL100, which were compiled by scraping sign videos and their titles from online sources rather than through direct collaboration with Deaf contributors. Because web-scraped datasets are not filtered through a linguistic reference such as ASL-LEX, they are more susceptible to mixed dialects, informal gesture variants, and polysemic label noise, in which a single English gloss label is attached to videos representing different underlying signs; the English word “right,” for example, may be attached to videos meaning either “correct” or “direction.” ASL Citizen 100 also provides substantially higher signer diversity and sample density per class than WLASL100, whose clips average only a small number of samples per gloss, and its dataset is expected to remain reproducible over time because it was collected under explicit participant consent rather than linked to third-party video hosting that may later become unavailable.

The ASL Citizen 100 vocabulary provides the basis for the fixed 100-gloss scope adopted in ATLAS. A predetermined vocabulary establishes a controlled recognition problem and allows the system to focus on consistent classification of a defined set of American Sign Language signs. The use of a fixed vocabulary is appropriate for the present study because ATLAS is designed for a specific set of sign language inputs rather than as an unrestricted sign language interpreter. The system therefore recognizes signs within the selected 100-gloss scope, while signs outside the vocabulary remain outside the recognition capability of the current implementation. The complete list of selected glosses may be presented in the methodology or appendix to document the system's recognition scope. Section 2.1.7 further develops the justification for constraining ATLAS to this 100-gloss scope, situating the decision within the broader vocabulary-scaling literature.

In addition to vocabulary selection, intermediate representations are relevant to the translation process. Gloss-to-text translation treats the relationship between gloss sequences and written language as a learning task, with data augmentation offering one way to support training when paired examples are limited (Moryossef et al., 2021). However, gloss-free approaches also connect video directly with target-language text (Lin et al., 2023). These alternatives show that glosses are a design choice within a translation system, rather than a requirement shared by every approach.

For ATLAS, recognized signs are maintained in a visible gloss buffer that connects recognition with English generation. As the segmental decoder accepts signs, their labels are appended in order and become available to T5-efficient-tiny. English updates can therefore be prepared while signing continues. The Finish button or held-open-palms gesture signals completion and finalizes the remaining text, giving the signer an explicit way to complete the message.

The Text-to-Text Transfer Transformer (T5) provides the language-generation foundation for this component. Its text-to-text framework expresses language tasks through an input sequence and a generated output sequence (Raffel et al., 2020). In ATLAS, fine-tuning connects gloss input with English wording. Because glosses are an intermediate representation rather than a complete record of signed-language meaning, the quality of the generated text must be examined in relation to that input (Müller et al., 2023).

This recognition-and-language approach supports ATLAS's modular architecture. Visual processing identifies signs, Gloss Management maintains their sequence, and T5-efficient-tiny generates English from the accepted input. Keeping these responsibilities distinct allows errors in recognition to be examined separately from changes introduced during English generation, while the components continue to operate together within the application.

### **2.1.2 Landmark-Based Visual Representation for Sign Language Recognition**

Landmark-based visual representation provides a structured alternative to processing complete raw video frames. Instead of representing an input entirely through pixel information, landmark-based approaches describe relevant anatomical points through coordinates and related measurements. This representation allows recognition models to focus on the spatial configuration and movement of the hands, face, and body while reducing the influence of unrelated background information (Rastgoo et al., 2021). Sign language recognition is particularly suited to structured visual representations because hand configuration, movement, position, facial information, and body posture can contribute to the interpretation of a sign.

MediaPipe Hands is one example of a landmark extraction pipeline. It estimates hand landmarks from RGB camera input and is designed for on-device tracking (Zhang et al., 2020). Such representations provide coordinate-based information that downstream temporal and spatial models can use to distinguish configurations and movement.

Apple Vision also provides visual pose estimation capabilities for supported platforms, including hand and human body pose analysis. These capabilities can provide structured information about the location and movement of relevant body parts in images and video. The use of hand and body landmarks is consistent with research showing that sign recognition may benefit from representing multiple parts of the signer rather than relying exclusively on isolated hand images (Rastgoo et al., 2021; Lee et al., 2023).

Part-wise motion-context learning models information from individual body regions before combining it with broader context (Lee et al., 2023). This approach is relevant when different parts of the signer contribute distinct cues, since combining all points at the outset may obscure those relationships.

In ATLAS, Apple Vision supplies the landmark representation used by the recognition pipeline. The representation contains 61 points: 21 per hand, 15 facial points, and 4 upper-body points. Each point carries body-relative X and Y coordinates, a relative scale-based depth proxy, presence, and confidence. These values describe the available visual evidence without treating the depth proxy as a measured three-dimensional position.

The landmark representation serves as structured input to the temporal recognizer. Expressing coordinates relative to the body helps place recordings on a consistent spatial basis, while presence and confidence distinguish observed points from missing information. The hand-image branch then contributes appearance features that complement this representation. Together, the inputs connect anatomical movement with the visible configuration of the hands.

### **2.1.3 Temporal and Sequence Modeling for Sign Language Recognition**

Sign language recognition is inherently temporal because the meaning of a sign can depend on how hand and body configurations change over time. Movement direction, timing, transitions, and the relationship between consecutive frames may distinguish signs that appear similar in individual static frames. Consequently, recognition must consider ordered visual information rather than treating each frame as an independent classification problem (Rastgoo et al., 2021).

Transformer-based architectures use attention to relate observations at different positions in a sequence. Sign Language Transformers connects temporal visual processing with recognition and translation objectives, including Connectionist Temporal Classification (CTC) for gloss supervision (Camgoz et al., 2020). CTC learns a label sequence without requiring a manually supplied boundary for every label, making it relevant to recognition from unsegmented input (Graves et al., 2006).

Explicit segmentation addresses a related question: where individual signs occur within the input. Sign-language segmentation research distinguishes sign beginnings, continuation, and the surrounding sequence through temporal labels and decoding procedures (Moryossef et al., 2023). This provides a basis for separating boundary estimation from the recognition of an interval's content. The timing model proposes where to look, while the recognition model evaluates what the interval contains.

ATLAS applies this separation through a distilled boundary model, a multimodal recognizer, and a segmental decoder. Candidate intervals receive recognition scores, and the decoder combines those scores with timing information to select a sequence. Each candidate's landmark input is resampled to 32 frames, allowing intervals of different durations to be represented consistently. This is the recognizer's input length, not the camera frame rate. Temporally sampled hand-image features provide complementary evidence from the same interval.

The researchers also tried CTC-based recognition. In the tested window-based path, predictions could change as input accumulated, and speech depended on a stable sequence; live observations also exposed repeated output for a held sign. These implementation findings motivated explicit interval selection and word commitment in ATLAS. CTC remains relevant related work, while the system's live behavior is organized around boundary estimation, recognition, and segmental decoding.

### **2.1.4 Squeezeformer as an Efficient Temporal Architecture**

Hybrid attention-convolution architectures combine mechanisms for relating distant observations with operations that capture local temporal patterns. Squeezeformer was developed for automatic speech recognition and modifies the Conformer design through temporal compression and changes to its processing blocks (Kim et al., 2022). Its research contribution concerns the balance between recognition and computational cost in that setting, providing a basis for investigating related temporal designs in other tasks.

Although Squeezeformer was originally developed for speech recognition, its architectural characteristics are relevant to other sequential recognition problems. Sign language recognition, like speech recognition, involves ordered information in which the interpretation of a particular observation may depend on preceding and following observations. Therefore, Squeezeformer's emphasis on efficient temporal modeling provides a theoretical basis for investigating its use in sign language recognition.

ATLAS adapts Squeezeformer-based processing to landmark and hand-image sequences. The landmark branch preserves information from individual body regions before integrating it into a global temporal representation. This gives local movements a role within the wider sign interval, while the hand-image branch models appearance changes across sampled frames. The literature motivates this design, and the project's matched model comparisons determine its suitability for ATLAS. Selection therefore considers recognition performance together with measured processing cost; the architecture's name alone does not establish that it is the fastest option.

### **2.1.5 RGB Hand-Crop Features and Efficient Visual Encoders**

Landmark representations provide structured information about the location and movement of body parts, but coordinate-based features may not fully preserve fine-grained visual appearance such as detailed handshape and local visual configuration. Research on sign language recognition has therefore explored combinations of skeletal or pose information with visual information to capture complementary characteristics of a sign (Rastgoo et al., 2021). More recent multimodal approaches have similarly investigated the combination of motion trajectories and RGB appearance features for isolated sign recognition (Renjith, Varghese, Rashmi, et al., 2026).

Cropped hand images provide a more focused visual representation than processing complete video frames. By concentrating on the hand regions involved in signing, a system can preserve appearance information while limiting the amount of unrelated background information processed by the visual encoder. This approach is particularly relevant for systems that require a balance between visual representation and computational efficiency.

MobileCLIP is a family of efficient image-text models developed with attention to latency and memory demands (Vasu et al., 2024). MobileCLIP2 extends this line of work through improved training and knowledge transfer for visual representations (Faghri et al., 2025). These models provide a basis for obtaining image features, while their use in sign recognition requires a downstream model that learns from signing examples.

ATLAS uses MobileCLIP2 as an upstream image feature extractor for cropped hand images. The system samples left-hand, right-hand, and combined-hand crops at selected time points. The resulting visual embeddings are then processed by a temporal Squeezeformer head. This design provides a complementary visual branch to the landmark branch rather than replacing landmark-based recognition.

The use of multiple hand-crop views is intended to preserve visual information that may not be completely represented by coordinates alone. The left-hand and right-hand views provide separate evidence about the hands, while the combined-hand view provides information about their relative visual configuration. The resulting RGB branch is therefore integrated with the landmark branch through the multimodal fusion component of ATLAS.

### **2.1.6 Multimodal Fusion for Sign Language Recognition**

Sign language recognition can benefit from combining different forms of visual information because individual modalities may capture different characteristics of signing. Landmark representations provide structured information about spatial configuration and motion, while RGB-based visual features can preserve appearance information such as handshape and local visual details. Sign language recognition surveys have identified multimodal approaches as an important direction because hand, body, facial, and other visual cues can contribute complementary information to recognition (Rastgoo et al., 2021).

Part-aware approaches also motivate retaining local information before combining it with a broader representation. Motion-context learning across body regions provides one example of this principle (Lee et al., 2023). Motion and appearance fusion applies a related idea to complementary input types, using their different characteristics to support sign recognition (Renjith, Varghese, Rashmi, et al., 2026).

The combination is useful because an ambiguous observation in one branch may be clearer in the other. A hand's trajectory can distinguish movements with similar appearances, while a cropped image can preserve handshape detail that a coordinate representation omits. Fusion provides a place for the recognition model to use these cues together.

ATLAS applies this multimodal principle through two recognition branches. The landmark branch processes structured hand, face, and upper-body information using a part-wise and global Squeezeformer architecture. The RGB branch processes left-hand, right-hand, and combined-hand crops through MobileCLIP2 embeddings followed by temporal Squeezeformer modeling. The predictions from the two branches are then combined through a learned fusion head.

The multimodal design does not assume that one modality is always sufficient or that fusion automatically guarantees higher accuracy. Instead, it provides complementary sources of evidence that can be evaluated individually and in combination. This allows the study to compare landmark-only, RGB-only, and fused recognition performance and determine whether the additional visual branch contributes measurable value within the fixed 100-gloss vocabulary.

### **2.1.7 Justification for the Fixed 100-Gloss Vocabulary Scope**

The 100-sign vocabulary gives ATLAS a defined recognition task around which data preparation, model comparison, and application behavior can be organized. Its role is to establish a prototype with foundational signs and a consistent set of output labels. Vocabulary size and vocabulary content serve different purposes: the size determines the number of classes, while the selected signs determine the messages and concepts available within the prototype.

Vocabulary size is relevant to model development because adding classes changes the distinctions that a recognizer must learn. Studies using pose-based Transformers have examined recognition across vocabularies of different sizes, with lower accuracy in some larger-vocabulary conditions (Woods & Rana, 2023; Boháček & Hrúz, 2022). These results motivate evaluating the effect of vocabulary expansion alongside representation and training choices. They do not identify 100 as a universal threshold for usable recognition.

Computational efficiency is also a consideration, but the vocabulary count alone does not determine whether an application can run locally. Small Transformer models have been investigated for reduced-vocabulary recognition (Woods & Rana, 2023), while other approaches use reservoir computing to reduce training cost (Syulistyo et al., 2025). For ATLAS, the relevant design question is how the complete recognition path performs with its selected inputs and models. Device measurements therefore accompany the vocabulary decision rather than being inferred from it.

The content of the vocabulary gives the prototype its foundational focus. ATLAS includes WHAT, WHERE, WHEN, WHO, WHY, and HOW, which provide sign labels for asking about things, places, times, people, reasons, and manner. Question-sign practice appears in introductory ASL learning activities, supporting the inclusion of these concepts in a foundational selection (Boise State University, n.d.). Their presence gives the vocabulary an information-seeking purpose; it does not equate a collection of labels with complete question grammar.

Published word-level recognition research also uses 100-class evaluation settings, making that scale relevant to the literature reviewed here (Li et al., 2020; Eunice et al., 2023; Syulistyo et al., 2025). However, equal vocabulary sizes do not establish matched evaluations: label membership, signer separation, input preparation, and measurement conditions also affect results. The selected scope thus supports a defined prototype and a connection to related research, while ATLAS's own measurements remain tied to its recorded evaluation conditions.

### **2.1.8 Data Preparation and Learning Across Recordings**

A recognition model learns from the relationship between visual examples and their labels, making data preparation part of the learning method. A gloss annotation identifies a sign, whereas a temporal annotation identifies its interval in a recording. This distinction matters when preparing connected-sign examples: knowing the order of the signs is different from knowing their exact boundaries. Segmentation research provides methods for estimating those boundaries when processing continuous visual input (Moryossef et al., 2023).

Normalization and augmentation address different forms of variation. Normalization places inputs on a consistent basis, such as expressing landmark coordinates relative to the signer's body. Augmentation introduces controlled variations during training so that the model encounters more than one presentation of an example. Pose-based recognition research uses spatial normalization and transformations in preparing training inputs (Woods & Rana, 2023; Eunice et al., 2023). In ATLAS, these concepts inform landmark preparation and variations in candidate interval boundaries, alongside annotation review and manual clipping of selected recordings.

Regularization concerns how the model learns from those examples. Dropout temporarily omits internal activations during training, reducing reliance on particular combinations of features (Srivastava et al., 2014). Weight decay constrains parameter growth, with AdamW separating the decay operation from the adaptive gradient update (Loshchilov & Hutter, 2019). These methods provide training controls whose settings must be considered within the relevant model recipe.

Knowledge distillation provides another way to guide learning by using predictions from a teacher model as supervision for a student (Hinton et al., 2015). ATLAS uses this approach to transfer timing predictions from a pretrained segmentation model to its Apple Vision boundary model. The teacher supports training, while the student supplies the application with boundary estimates. Recognition adaptation separately uses sign-interval examples and retained isolated-sign examples to connect training with the intervals encountered during streaming.

Finally, evaluation must distinguish new recordings from new signers. Separating signer identities across training, validation, and testing examines recognition on people outside the corresponding learning groups, a principle used in community-sourced isolated-sign research (Desai et al., 2023). This literature supports signer separation as an evaluation choice. The exact allocation and membership belong to the study's data records, while normalization, augmentation, and regularization address learning within those groups.

## **2.2 Related Systems** {#2.2-related-systems}


### **2.2.1 Vision-Based Sign Language Recognition Systems**

Vision-based sign language recognition systems use cameras or video to capture visual information from a signer without requiring wearable gloves or specialized motion sensors. Vision-based approaches may analyze hand configuration, movement, body posture, facial information, and other visual characteristics associated with signing. Rastgoo et al. (2021) identified vision-based recognition as a major direction in sign language recognition research and discussed both isolated and continuous recognition approaches.

Sign Language Transformers processes visual sequences through attention-based modeling and connects recognition with translation through joint learning objectives (Camgoz et al., 2020). Its use of gloss supervision demonstrates a connection between visual input and written-language output, while distinguishing the two tasks within one framework. This is relevant to ATLAS because both approaches treat signing as an ordered sequence.

However, the architectures and operational designs differ. ATLAS combines landmark and cropped-hand information with an explicit boundary model and segmental decoder. Accepted glosses are maintained in a buffer used for incremental English generation. This arrangement makes the point at which a sign is accepted part of the live application design, alongside the recognition and language models.

Vision-based recognition provides the broader foundation for ATLAS's camera-based interaction. The system brings structured landmarks and localized hand images together within that approach, then connects their predictions with English text and speech. Its relevance to the literature lies in this combination of visual representation and application behavior.

### **2.2.2 Landmark-Based and Keypoint-Based Recognition Systems**

Landmark-based and keypoint-based recognition systems transform visual input into structured coordinates representing relevant anatomical points. This approach is useful for sign language recognition because hand position, hand configuration, movement, and body relationships are important characteristics of signing. Landmark representations can also reduce the amount of irrelevant background information that must be processed by the recognition model (Rastgoo et al., 2021).

MediaPipe Hands provides an on-device example of hand-landmark estimation from RGB images (Zhang et al., 2020). Such extraction methods produce structured information for a downstream recognition model; locating points and identifying a sign remain different responsibilities. Apple Vision provides the extraction framework used by ATLAS on the iPhone (Apple, n.d.-a; Apple, n.d.-b).

Research also considers information from multiple body regions. Part-wise motion-context modeling retains local movement information before integrating it into a broader representation (Lee et al., 2023). This is relevant to signs whose interpretation depends on relationships between the hands and the face or upper body, rather than on a hand configuration viewed in isolation.

ATLAS applies this landmark-based approach to left-hand, right-hand, facial, and upper-body points obtained through Apple Vision. Its body-relative representation includes presence and confidence information, allowing the recognizer to distinguish available coordinates from missing observations. The landmark component supplies structured visual evidence, which is combined with hand-image features during recognition.

### **2.2.3 Deep Learning-Based Sequential Sign Recognition Systems**

Deep learning-based sequential sign recognition systems model ordered visual observations over time. These systems are important because sign recognition depends not only on the appearance of a hand at a particular moment but also on movement, direction, timing, and transitions between configurations. Transformer-based architectures have become particularly relevant because their attention mechanisms can model relationships across different points in a sequence.

Transformer-based sequence modeling provides one route from ordered visual observations to recognition and translation outputs. Sign Language Transformers uses CTC gloss supervision alongside a translation objective, illustrating how the tasks can be learned together (Camgoz et al., 2020). This establishes a related approach to sequence processing without prescribing the same runtime design for every application.

Squeezeformer provides a temporal architecture developed for speech recognition, with changes intended to improve the balance between sequence modeling and computation (Kim et al., 2022). Its application to ATLAS involves adapting those temporal mechanisms to visual features. The adaptation must therefore be assessed through sign-recognition measurements rather than by transferring speech-recognition results directly.

ATLAS uses Squeezeformer-based temporal processing for both landmark and hand-image evidence. The landmark branch receives 32 sampled frames per candidate interval, while the visual branch processes hand-image features across selected time points. The boundary model and segmental decoder coordinate these predictions within the ongoing camera stream, allowing interval content and sequence selection to serve distinct roles.

This division also explains the system's modularity. The temporal recognizer estimates a sign's identity, while the boundary and decoding components determine the intervals considered and the outputs accepted. Improving one component can be examined through its effect on the shared recognition sequence, without treating the entire application as a single model.

### **2.2.4 Gloss-Based or Intermediate Representation Translation Systems**

Gloss-based and intermediate-representation systems divide sign language translation into recognition and language-generation components. In such systems, visual sign input is first converted into a structured representation, such as glosses, before being transformed into a target spoken or written language. This approach provides an intermediate representation between visual recognition and natural language generation.

Gloss-level information has been used to connect recognition and translation within joint learning frameworks (Camgoz et al., 2020). Gloss-to-text research also treats language generation as a separate task and investigates augmentation of its paired training examples (Moryossef et al., 2021). These approaches support examining the language component through its gloss input, while recognition is assessed against the visual sequence.

At the same time, gloss-free translation investigates learning from video and text without an intermediate gloss sequence (Lin et al., 2023). The distinction concerns the system's representation and supervision. In a gloss-based application, the intermediate labels provide a visible record of what recognition accepted; in a gloss-free approach, that record is not required by the architecture.

ATLAS follows the intermediate-representation approach by maintaining accepted signs in a visible gloss buffer. T5-efficient-tiny prepares English updates as the input accumulates, and Finish finalizes the remaining text. This arrangement gives recognition and English generation defined responsibilities and allows their outputs to be examined separately.

English-output assessment must also consider whether the generated wording preserves the supplied input. G-Eval investigates language-model evaluation using explicit criteria, providing methodological support for examining qualities beyond surface word overlap (Liu et al., 2023). ATLAS's automatic English judgments use a separate model-based rubric. Such judgments describe the evaluation procedure; they are distinct from human translation ratings and do not by themselves measure recognition of signed video.

### **2.2.5 Streaming and Application-Oriented Sign Language Systems**

Application-oriented sign language recognition considers not only recognition accuracy but also how a model operates during interaction. Visual input conditions, temporal processing, response behavior, computational requirements, and the presentation of results all affect the application's operation. Reviews identify signer variability, continuous recognition, and practical deployment as continuing research concerns (Violet & Leena Sri, 2025).

Streaming introduces a particular timing problem: an application must determine which parts of an ongoing camera stream contain signs and when the available evidence is sufficient to accept an output. Classifying an already trimmed clip does not resolve that decision. Sign-language segmentation therefore provides relevant related work, since it explicitly models sign and phrase boundaries (Moryossef et al., 2023).

The pretrained Sign Language Segmentation project supplies the teacher used in ATLAS's boundary-model training. Its 2026 CNN–Transformer implementation uses MediaPipe pose inputs and is distinct from the implementation accompanying the 2023 segmentation paper (Sign Language Processing, 2026). During ATLAS distillation, the teacher remains fixed while the student learns its timing predictions from Apple Vision features. The application uses the resulting student together with a separately adapted sign recognizer and segmental decoder. This connects the source model's timing role with the system's own input representation.

Responsiveness is also relevant to the intended interaction. A collaboration study with Deaf and hard-of-hearing ASL signers identified real-time sign-to-spoken or written translation as a priority for most of its surveyed signers and discussed latency concerns during co-design (Kamikubo et al., 2025). These findings support considering timely feedback alongside accuracy, while leaving specific response-time requirements to the application and its evaluation.

ATLAS brings these considerations into an iPhone workflow. Boundary estimates guide candidate intervals, recognition supplies sign scores, and the decoder accepts glosses into the visible buffer. T5-efficient-tiny prepares English as the sequence grows, while the Finish button or held-open-palms gesture completes the message for final text and local speech output. Core ML executes the exported models locally, connecting the recognition architecture with the application interface and its completion controls.

### **2.2.6 Recent 100-Gloss Word-Level ASL Recognition Systems**

Recent word-level systems illustrate different ways of balancing recognition, computation, and interaction at the 100-class scale. Their relevance to ATLAS lies in the design questions they address, including the representation of signing, the cost of learning temporal patterns, and the role of recognition within an application. Their numerical results describe their respective study conditions rather than a matched comparison with ATLAS.

A gamified vocabulary-learning system combines webcam-based sign entry with a word-search activity, using a skeleton-based Stacked Transformer with spatial-temporal attention (Amiruzzaman et al., 2026). Its interface and recognition backend connect a learning activity with visual input. This provides an application-oriented comparison for ATLAS's Practice and Glosses functions, while ATLAS additionally connects accepted sign sequences with English generation and local speech.

Multiple Reservoir Computing offers a different approach to the cost of sequential learning. A configuration using MediaPipe keypoints and multiple reservoirs achieved 60.35% Top-1 and 84.65% Top-5 accuracy in a 100-class isolated-sign evaluation, with a reported CPU training time of 52.7 seconds (Syulistyo et al., 2025). This example supports considering training cost alongside recognition results. It does not establish the processing time of a complete camera-to-text application.

Sign2Pose combines pose-based features, key-frame extraction, and Transformer processing for gloss prediction. Its reported Top-1 accuracy was 80.90% in a 100-class condition and 64.21% in a 300-class condition (Eunice et al., 2023). The system is relevant to ATLAS through its use of structured pose information and temporal processing, while its results also illustrate why vocabulary changes need to be evaluated under specified conditions.

These systems connect the vocabulary-scale discussion with practical model choices. ATLAS investigates a complementary combination of landmark and hand-image evidence, Squeezeformer-based temporal recognition, and explicit streaming decisions. English generation and speech output extend the workflow beyond the recognition label, while separate component and application measurements provide the basis for examining how the parts work together.

# **CHAPTER 3** {#chapter-3}

**TECHNICAL BACKGROUND**  
This chapter presents the technical foundation of the proposed system. It describes the current communication methods addressed by the study and explains the architecture and technologies used in the proposed system. The chapter focuses on how visual sign language input is captured, represented, recognized, organized into a gloss sequence, and converted into understandable English output. It also describes the computational components that support the system, including Apple Vision landmark extraction, MobileCLIP2 hand-image features, Squeezeformer-based recognition, boundary estimation, segmental decoding, gloss management, and incremental English generation.

## **3.1 Current System** {#3.1-current-system}

The existing communication methods addressed by this study rely primarily on manual forms of message exchange between sign language users and individuals who do not understand sign language. Within academic environments such as Leyte Normal University (LNU), communication may involve handwritten notes, typed messages through mobile note-taking applications, or other improvised forms of interaction. Although these methods allow information to be exchanged, they require the participants to manually compose and interpret messages rather than providing a direct means of bridging signed communication and English.

These communication methods can interrupt the natural flow of conversation because the sign language user may need to write or type a message instead of communicating directly through signing. The process may also introduce delays when immediate responses are needed, particularly during academic, administrative, service, or other everyday interactions. Written or typed exchanges may further result in shortened or simplified messages, which can make it more difficult to communicate ideas that would otherwise be expressed through a complete signed utterance.

Another limitation is the linguistic difference between sign languages and English. Sign languages have their own grammatical structures and may use spatial relationships, movement, facial information, and other non-manual cues that do not necessarily correspond to English on a word-for-word basis. Consequently, manually exchanging written messages does not directly address the linguistic gap between signed communication and English.

These communication considerations provide the basis for developing an assistive mechanism that connects sign recognition with English output. ATLAS addresses this task through a predefined ASL vocabulary, allowing accepted glosses to accumulate during signing while English text is prepared for display and local speech.

## **3.2 Proposed System** {#3.2-proposed-system}

ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation is a modular iPhone application designed to recognize a 100-sign ASL prototype vocabulary and generate English text and local speech from the recognized sequence. The system combines landmark-based information with features from cropped hand images so that recognition can draw on both movement and visual appearance. These inputs are processed within a streaming workflow that identifies candidate sign intervals, accepts glosses, and prepares English as signing continues.

The proposed system consists of four major functional components: (1) Visual Extraction, (2) Streaming Sign Recognition, (3) Gloss Management, and (4) English Generation and Speech Output. Each component has a defined responsibility and exchanges information with the next part of the workflow. A component may contain a trained model, an Apple framework, or supporting processing rules; this organization describes how the parts cooperate rather than equating every component with one model.

The Visual Extraction component captures the signer's input through the iPhone camera and prepares landmark and hand-image representations. Apple Vision locates 61 selected points: 21 on each hand, 15 on the face, and 4 on the upper body. Each point is represented by body-relative X and Y coordinates, a relative scale-based depth proxy, presence, and confidence. Expressing positions relative to the body gives the recognizer a consistent spatial representation, while presence and confidence indicate the available observations. Crops of the left hand, right hand, and both hands together are also prepared so that MobileCLIP2 can extract appearance features from the same signing activity.

Streaming Sign Recognition combines these inputs with boundary estimates to interpret the camera sequence. Its recognition model uses Squeezeformer-based temporal processing for the landmark and hand-image branches, then combines their evidence to score candidate signs. Each candidate interval is resampled into 32 landmark frames for the recognizer. This fixed input length allows intervals of different durations to be processed in a common form; it describes the model input rather than the camera frame rate. The complementary image features retain handshape information across sampled points in the interval.

The boundary model estimates candidate sign timing, while the segmental decoder uses those estimates together with recognition scores to select a sequence. ATLAS's boundary model learns through knowledge distillation from the pretrained Sign Language Segmentation teacher, whose weights remain fixed during this transfer. The resulting student uses Apple Vision features in the application, and the sign recognizer is separately adapted to candidate intervals. This division allows timing and sign identity to contribute different evidence to the same recognition decision (Moryossef et al., 2023; Sign Language Processing, 2026).

The Gloss Management component stores accepted signs as an ordered, visible gloss sequence. A candidate prediction can change while evidence is being gathered; acceptance determines when it becomes part of the output buffer. Maintaining this sequence provides the language model with the recognized input and gives the application a visible record of the signs that have been accepted. The sequence therefore connects the decisions made during visual recognition with the text prepared for the recipient.

Finally, English Generation and Speech Output uses a fine-tuned T5-efficient-tiny model to generate English from the accumulated glosses. English preparation takes place during signing, with completed sentences retained while the remaining input is processed. The Finish button or a two-open-palms gesture held for one second signals completion of the message. The system then finalizes the remaining English text and provides local speech output. Core ML executes the exported models on the iPhone, connecting the trained components with the application's camera, display, and completion controls.

Figure 1 shows how these responsibilities form the Live workflow. Visual extraction supplies the evidence used by Streaming Sign Recognition, accepted glosses become input to T5-efficient-tiny, and the completion control leads to final English text and speech. The application also provides Glosses, Practice, and History, allowing users to inspect the vocabulary, practice supported signs, and review saved sessions alongside Live recognition.

![ATLAS Live recognition and English-output flow](review_assets/system_flow.png)

**Figure 1.** *ATLAS Live Recognition and English-Output Flow*

The HELLO HOW YOU example in Figure 2 illustrates how the boundary and recognition components relate to a recorded sequence. The upper row shows selected intervals based on boundary estimates and decoder decisions, while the lower row associates those intervals with the recognized glosses. HELLO occupies approximately 0.67–1.17 seconds, HOW 1.40–1.67 seconds, and YOU 2.27–2.87 seconds in the saved recognition record. The video stills show one original frame near the midpoint of each interval, providing a visual reference for the signing input.

![HELLO HOW YOU intervals and original video frames](review_assets/hello_how_you_timeline.png)

**Figure 2.** *HELLO HOW YOU Recognition Timeline*

The horizontal axis represents position within the recording. The bars are model-selected intervals, and the stills illustrate the input rather than replacing the sequence of frames used for recognition. Boundary estimation guides the selection of candidate intervals, recognition supplies their gloss scores, and decoding selects the output sequence. This example explains how those roles work together; device processing time is measured separately from the recording positions shown here.

### Software Requirements {#software-requirements}

The software supports both model development and execution of the iPhone application. Python, PyTorch, and OpenCV are used to prepare data, train models, and carry out development work, while the application integrates native camera processing and Core ML execution. Table 1 identifies these roles so that training tools are distinguished from the software used during Live recognition.

**Table 1.** *Software and Descriptions*

| Software | Features | Purpose |
| :---- | :---- | :---- |
| Python | Development programming language | Supports data preparation, training scripts, and evaluation tools. |
| PyTorch | Deep learning framework | Trains and evaluates the recognition, boundary, and English-generation models during development. |
| OpenCV | Image and video processing | Supports recording preparation, frame inspection, and feature-extraction tools during development. |
| Flutter and Dart | Application project framework and language | Provide project infrastructure and integration with native iOS screens. |
| Swift and UIKit | Native iOS programming and interface components | Integrate Live recognition, camera controls, and native application behavior. |
| AVFoundation | Camera and media framework | Supplies camera frames and supports local speech synthesis. |
| Apple Vision | Hand, face, and body landmark detection | Extracts the selected landmarks from camera input. |
| MobileCLIP2 | Image feature extractor | Produces features from left-hand, right-hand, and combined-hand crops. |
| Squeezeformer-based recognition | Temporal processing of combined visual evidence | Recognizes candidate signs using landmark and hand-image information. |
| Boundary model and segmental decoder | Timing estimation and sequence selection | Identify candidate intervals and select glosses for acceptance. |
| T5-efficient-tiny | Text-to-text language model | Generates English from the accumulated gloss sequence. |
| Core ML | Local model execution framework | Executes the exported models on the iPhone. |
| Local text-to-speech | On-device speech synthesis | Reads the generated English text aloud. |

### Hardware Requirements {#hardware-requirements}

The hardware serves two related purposes: development of the models and operation of the application. The iPhone 13 is the implementation and device-measurement platform for the system, while the Mac supports development, model preparation, and engineering evaluation. Table 2 records these roles. The iPhone 13 identifies the device used in the study rather than a minimum supported device specification.

**Table 2.** *Hardware Components and Descriptions*

| Hardware | Features | Purpose |
| :---- | :---- | :---- |
| iPhone 13 | Built-in camera, display, speaker, and local model execution | Runs the application and provides the platform for recorded device measurements. |
| iPhone camera | RGB video capture | Captures signing input for landmark extraction and hand-image preparation. |
| iPhone display and speaker | Visual and audible output | Present recognized glosses, English text, recognition feedback, and local speech. |
| Mac computer | Development environment and supported computation | Supports application development, model preparation, conversion, and engineering evaluation. |
| Local storage | Storage for development and application resources | Holds recordings, extracted features, model files, and logs during development, and model resources and saved sessions on the device. |
| Development hardware acceleration | Supported GPU or other accelerated computation | Supports model training and computational development tasks where available. |

### Peopleware {#peopleware}

*Non-Sign Language Users*  
The primary intended users of the system are non-sign language users within the Leyte Normal University community. These include students, faculty members, instructors, university administrators and staff, medical and guidance personnel, and IGP stall administrators or stall owners. The system is designed to assist these users in understanding signed messages by presenting recognized input as English text and local speech through the mobile application. In this way, ATLAS is intended to support communication between sign language users and non-signing recipients across academic, administrative, service, and daily campus interactions.

*Developers*  
The developers are the researchers responsible for designing, developing, and evaluating the components of the system. Their responsibilities include preparing and reviewing recognition data, implementing the multimodal recognition and English-generation components, integrating the iPhone application, and conducting system testing and evaluation. They are also responsible for examining model performance, maintaining the processing workflow, and refining application behavior based on recorded findings. Through these efforts, the developers aim to create an assistive tool that is functional and relevant to the communication needs of the Leyte Normal University community.

# **CHAPTER 4** {#chapter-4}

**METHODOLOGY, RESULTS, AND DISCUSSION**  
This chapter presents the methodology followed in the development, implementation, and evaluation of ATLAS: A Squeezeformer-Based System for Sign Language Recognition and English Translation. It describes the requirements, development process, system design, implementation, and evaluation procedures used to determine whether the proposed system meets its intended objectives. The chapter also presents the recorded recognition and English-generation results, together with measurements that explain the selection of models and their execution on the iPhone. These findings connect the design decisions with the communication workflow introduced in the preceding chapters.

## **4.1 Requirements Analysis** {#4.1-requirements-analysis}

The requirements analysis identifies the necessary requirements that guide the development and evaluation of ATLAS. It outlines the expected functions, performance, and overall system behavior needed to meet the objectives and intended scope of the proposed system.

### *Functional Requirements* {#functional-requirements}

**Table 3.** *Functional Requirements*

| Requirement  | Description |
| :---- | :---- |
| Keypoint  Extraction | The system must extract relevant body and gesture keypoints from captured sign language input and convert them into structured coordinate-based data for further processing. |
| Sign Gesture Recognition | The system must recognize sign gestures from the extracted visual features using the multimodal Squeezeformer-based recognition model within the predefined 100-gloss vocabulary. |
| Multimodal Feature Processing | The system must process landmark-based skeletal information and RGB hand-crop visual information and combine their outputs for sign recognition. |
| Activity Detection and Verification | The system must identify and verify candidate sign activities from the camera stream while reducing duplicate or unstable predictions. |
| Gloss Buffering | The system must append verified recognized signs to an ordered gloss buffer for subsequent utterance processing. |
| Utterance Completion | The system must allow the user to complete an utterance through the designated Finish button or two open palms held for one second. |
| Gloss-to-English Generation | The system must prepare English sentence updates during signing using T5-efficient-tiny and finalize the remaining text when Finish is activated. |
| Speech Output | The system must provide synthesized speech from the generated English text through local text-to-speech functionality. |

### *Performance Requirements* {#performance-requirements}

**Table 4.** *Performance Requirements*

| Requirement  | Description |
| :---- | :---- |
| Recognition Performance | The Squeezeformer-based recognition model shall achieve measurable performance in classifying the predefined 100-gloss vocabulary, evaluated using Top-1 and Top-5 accuracy for individual signs and word error rate for sign sequences. |
| Translation Performance | The T5-efficient-tiny rephrasing component shall generate understandable English outputs from recognized gloss sequences, assessed using reference-based wording overlap and explicit meaning-preservation criteria. |
| Processing Efficiency | The system shall process captured sign input and perform recognition, activity verification, gloss buffering, and translation within a practical turnaround time for assisted communication. |
| Streaming Stability | The system shall maintain consistent candidate detection and verification during camera-based input while minimizing duplicate or unstable gloss predictions. |
| System Reliability | The system shall maintain stable operation during visual capture, feature extraction, recognition, gloss buffering, translation, and speech output without unexpected interruption. |
| Functional Suitability | The system shall provide the functions necessary to perform its intended sign recognition and English rephrasing tasks based on the defined system scope. |
| Usability | The system shall provide an understandable and manageable interaction flow for users performing sign input and viewing or hearing the generated English output. |
| Evaluation Compliance | The developed system shall be evaluated according to applicable criteria of ISO/IEC 25010:2023, using criteria relevant to functional suitability, performance efficiency, interaction capability, and reliability. |

### 	*Software and Hardware Requirements* {#software-and-hardware-requirements}

**Table 5.** *Software and Hardware Requirements*

| Requirement  | Description |
| :---- | :---- |
| Development Tools | Python, PyTorch and OpenCV support preparation, model training and technical evaluation. |
| Application Frameworks | The mobile project uses Flutter/Dart infrastructure with a native Swift/UIKit interface and model integration. AVFoundation supplies camera frames, and Apple Vision extracts landmarks. |
| Processing Environment | The application executes on the iPhone. The iPhone 13 is the implementation and measurement device described in this study. |
| Camera Input | The device camera captures the hands, face and upper body, supplying frames for landmark extraction and hand-image preparation. |
| Model Execution | Core ML executes MobileCLIP2, the recognition and boundary models, and T5-efficient-tiny locally. Native speech synthesis produces audible English output. |
| Local Storage | The application stores model resources and session history locally. Development storage holds training inputs, extracted features and evaluation records. |
| Development Hardware | Mac-based resources support Apple framework integration, model conversion and application builds; training resources support the corresponding model recipes. |

	

### *Safety Requirements* {#safety-requirements}

**Table 6.** *Safety Requirements* 

| Requirement  | Description |
| :---- | :---- |
| Physical User Safety | The system interface and camera-based interaction must allow users to maintain awareness of their surroundings while performing sign language input.  |
| Device Temperature Control | The system should be operated under conditions that support stable processing without excessive device heating, performance degradation, or interruption during extended use.  |
| Operational Ergonomics | Camera-based gesture capture should allow users to perform natural signing movements without requiring uncomfortable or prolonged static poses.  |

### *Security Requirements* {#security-requirements}

**Table 7.** *Security Requirements* 

| Requirement  | Description |
| :---- | :---- |
| Data Privacy  | User sign input, extracted visual information, translated output, and other interaction data must be handled in accordance with appropriate privacy and data-protection practices.  |
| Audit Trail Protection  | System logs containing recognition events, confidence values, and reference recordings must be protected from unauthorized access, modification, or disclosure.  |
| Data Retention and Deletion  | The system should provide documented procedures for the retention and deletion of recorded interaction data and logs, particularly when these contain identifiable or sensitive information.  |
| Model Integrity  | Trained model files and supporting resources must be stored and managed in a manner that protects the integrity of the implemented recognition and English rephrasing pipeline.  |
| Confidentiality of Interaction  | Sign input, generated English output, and other interaction information must be handled within the system environment as much as possible to reduce unnecessary exposure of sensitive communication data.  |

## **4.2 Design of Software, Systems, Product, and/or Processes**  {#4.2-design}

### Systems Development Life Cycle {#systems-development-life-cycle}

The project utilized the Modified Waterfall (Sashimi) Model, which is characterized by overlapping development phases that allow feedback and refinement between adjacent stages. This approach was selected because the development of ATLAS involved related activities that could be performed concurrently, particularly the refinement of the sign language recognition and English rephrasing components.

Using this model, the researchers proceeded through the major phases of planning, designing, development, testing, and implementation. The overlapping nature of the model allowed the team to refine the recognition pipeline while developing and integrating the translation component, helping ensure that the recognized gloss output was compatible with the English rephrasing process. The development process also included iterative testing and refinement of the system's visual extraction, multimodal Squeezeformer recognition, Streaming Sign Recognition component, gloss buffering, T5-efficient-tiny rephrasing, and speech output components.

Within this process, technical testing examines recognition errors, generated English and processing time, while the software-quality framework defines the characteristics to assess at application level. Feedback between these activities allows a model decision to be considered alongside its effect on the user-facing workflow.

![Sashimi model with five phases and feedback arrows](source_images/sashimi.png)

**Figure 3.** *SDLC Modified Waterfall (Sashimi) Model*

### *Planning and Requirement Analysis*   {#planning-and-requirement-analysis}

This phase focused on identifying the communication needs of sign language users and non-sign language users and defining the technical and functional requirements of ATLAS. The researchers established the system scope, including the fixed 100-gloss ASL vocabulary, multimodal visual processing, Squeezeformer-based sign recognition, streaming activity control, gloss buffering, English rephrasing, and speech output.

The requirements identified during this phase served as the basis for the succeeding design and development activities. An Activity List was prepared to organize the major tasks and their dependencies, while the GANTT and PERT charts were used to plan the project schedule and monitor the sequence and overlap of development activities.

#### **Activity List** {#activity-list}

An activity list is a fundamental software project management tool that identifies all necessary tasks and their logical sequence to ensure a structured development workflow. It serves as a comprehensive guide that specifies the required resources, the estimated duration for each task, and the specific dependencies or requirements needed to move through the project phases successfully.

**Table 8.** *Activity List*

| ID | TASKS | START | END | PREDECESSOR | DURATION (DAYS) |
| --- | --- | --- | --- | --- | --- |
| Phase 1: Planning |  |  |  |  |  |
| A | Project Proposal Preparation | Jan. 20 | Jan. 22 | — | 3 |
| B | Proposal Approval and Adviser Assignment | Jan. 23 | Jan. 24 | A | 2 |
| C | Review of Related Literature | Jan. 25 | Jan. 29 | B | 5 |
| D | Identification of Research Problem | Jan. 25 | Jan. 27 | B | 3 |
| E | Chapter 1: Writing | Jan. 28 | Feb. 1 | D | 5 |
| F | Chapter 1: Submission | Feb. 2 | Feb. 2 | E | 1 |
| G | Chapter 1: Adviser Feedback | Feb. 3 | Feb. 4 | F | 2 |
| H | Chapter 1: Revision | Feb. 5 | Feb. 7 | G | 3 |
| I | Chapter 1: Resubmission | Feb. 8 | Feb. 8 | H | 1 |
| Phase 2: Designing |  |  |  |  |  |
| Y | Definition of System Scope and Features | Jan. 25 | Jan. 28 | B | 4 |
| Z | Dataset Planning and Design | Jan. 29 | Feb. 3 | Y | 6 |
| J | Chapter 2: Writing | Feb. 9 | Feb. 13 | I | 5 |
| K | Chapter 2: Submission | Feb. 14 | Feb. 14 | J | 1 |
| L | Chapter 2: Adviser Feedback | Feb. 15 | Feb. 16 | K | 2 |
| M | Chapter 2: Revision | Feb. 17 | Feb. 19 | L | 3 |
| N | Chapter 2: Resubmission | Feb. 20 | Feb. 20 | M | 1 |
| O | Chapter 3: Writing | Feb. 9 | Feb. 14 | I | 6 |
| P | Chapter 3: Submission | Feb. 15 | Feb. 15 | O | 1 |
| Q | Chapter 3: Adviser Feedback | Feb. 16 | Feb. 17 | P | 2 |
| R | Chapter 3: Revision | Feb. 18 | Feb. 20 | Q | 3 |
| S | Chapter 3: Resubmission | Feb. 21 | Feb. 21 | R | 1 |
| T | Chapter 4: Writing | Feb. 9 | Feb. 14 | I | 6 |
| U | Chapter 4: Submission | Feb. 15 | Feb. 15 | T | 1 |
| V | Chapter 4: Adviser Feedback | Feb. 16 | Feb. 17 | U | 2 |
| W | Chapter 4: Revision | Feb. 18 | Feb. 20 | V | 3 |
| X | Chapter 4: Resubmission | Feb. 21 | Feb. 21 | W | 1 |
| Phase 3: Development |  |  |  |  |  |
| AA | Core System Development | Feb. 4 | Apr. 5 | Z | 61 |
| AB | Data Cleaning and Preparation | Feb. 4 | Feb. 11 | Z | 8 |
| AC | Data Validation and Label Checking | Feb. 12 | Feb. 17 | AB | 6 |
| AD | Feature Engineering and Processing | Feb. 18 | Feb. 25 | AC | 8 |
| AE | Initial Model Development | Feb. 26 | Mar. 7 | AD | 10 |
| AF | Model Testing and Iteration | Mar. 8 | Mar. 19 | AE | 12 |
| AG | Technical Documentation Preparation | Mar. 20 | Mar. 25 | AF | 6 |
| AH | Pre-Oral Defense | Apr. 6 | Apr. 6 | AA | 1 |
| AI | Post Pre-Oral Revisions | Apr. 7 | Apr. 11 | AH | 5 |
| AJ | Submission to Research Ethics Committee (REC) | Apr. 12 | Apr. 14 | AI | 3 |
| AK | REC Initial Review Process | Apr. 15 | Apr. 21 | AJ | 7 |
| AL | REC Revisions and Compliance | Apr. 22 | May 1 | AK | 10 |
| AM | Ethics Clearance Approval | Apr. 7 | May 11 | AH | 35 |
| BM | Survey Instrument Validation | Apr. 15 | Apr. 17 | AJ | 3 |
| BN | Participant Coordination | Apr. 18 | Apr. 25 | BM | 8 |
| Phase 4: Testing |  |  |  |  |  |
| AN | Additional Data Collection | May 12 | May 21 | AM | 10 |
| AO | Performance Evaluation and Benchmarking | May 22 | May 28 | AN | 7 |
| AR | User Interface Design and Setup | May 12 | May 25 | AP | 14 |
| AS | Initial System Demonstration | May 26 | May 29 | AR | 4 |
| AT | System Testing | May 30 | Jun. 8 | AS | 10 |
| AU | Debugging and Retesting | Jun. 9 | Jun. 18 | AT | 10 |
| AV | Evaluation Preparation | Jun. 19 | Jun. 23 | AU | 5 |
| AW | System Evaluation and Usability Testing | Jun. 24 | Jul. 4 | AV | 11 |
| BO | Results Analysis and Interpretation | Jul. 5 | Jul. 9 | AW | 5 |
| Phase 5: Implementation |  |  |  |  |  |
| AP | Mobile Application Development | May 12 | Jun. 10 | AM | 30 |
| AQ | System Integration and Finalization | Jun. 11 | Sept. 27 | AP | 109 |
| BK | System Backup and Repository Preparation | Jun. 11 | Jun. 15 | AP | 5 |
| BL | Version Control and Final Backup | Jun. 16 | Jun. 18 | BK | 3 |
| AX | Chapter 5: Writing | Jul. 5 | Jul. 12 | AR | 8 |
| AY | Chapter 5: Submission | Jul. 13 | Jul. 13 | AX | 1 |
| AZ | Chapter 5: Adviser Feedback | Jul. 14 | Jul. 15 | AY | 2 |
| BA | Chapter 5: Revision | Jul. 16 | Jul. 18 | AZ | 3 |
| BB | Chapter 5: Resubmission | Jul. 19 | Jul. 19 | BA | 1 |
| BC | Final Manuscript Editing | Aug. 1 | Aug. 7 | AW, BB | 7 |
| BD | Formatting and Plagiarism Checking | Aug. 8 | Aug. 10 | BC | 3 |
| BF | Adviser Final Approval | Aug. 8 | Aug. 10 | BC | 3 |
| BG | Presentation Script Preparation | Aug. 11 | Aug. 14 | BF | 4 |
| BH | Presentation Revision | Aug. 15 | Aug. 17 | BG | 3 |
| BJ | Final Document Checking | Aug. 18 | Aug. 20 | BC | 3 |
| BI | Mock Final Defense | Sept. 10 | Sept. 10 | BH | 1 |
| BE | Final Defense | Sept. 28 | Sept. 30 | AQ, BI, BJ, BO | 3 |

#### **Gantt Chart** {#gantt-chart}  
The Gantt chart presents the planned schedule and the overlap between project activities. It complements the Sashimi model by showing how documentation, model development and application work can proceed alongside one another. The Activity List and Gantt chart use the same source schedule, while the PERT chart shows the relationships between activities. Together, these planning tools help the researchers organize work and monitor progress; scheduled dates do not by themselves establish that an activity was completed.

**Table 9.** *GANTT Chart*

![Gantt chart, part 1](source_images/gantt_01.jpg)

![Gantt chart, part 2](source_images/gantt_02.jpg)

![Gantt chart, part 3](source_images/gantt_03.jpg)

![Gantt chart, part 4](source_images/gantt_04.jpg)

![Gantt chart, part 5](source_images/gantt_05.jpg)

![Gantt chart, part 6](source_images/gantt_06.jpg)

![Gantt chart, part 7](source_images/gantt_07.jpg)

#### **PERT Chart** {#pert-chart}

The researchers employed a Program Evaluation and Review Technique (PERT) chart to visualize task dependencies, durations, and the critical path within the systems development life cycle. This combination allows for structured planning and continuous monitoring of the overlapping phases required to develop the Squeezeformer-T5 sign language recognition and translation system.

![PERT chart](source_images/pert.png)

**Figure 4.** *PERT Chart of the Proposed System*

### *Design* {#design}

The system architecture integrates visual extraction, Streaming Sign Recognition, Gloss Management, and English Generation and Speech Output within a modular pipeline. This organization allows the application to process signing as it occurs while maintaining a clear connection between the visual input, accepted signs and resulting English. Each component has a defined responsibility, although a component may contain several cooperating models or processing rules. The diagrams below present this design from the user's interaction down to the information exchanged within the application.

#### **Context Diagram** {#context-diagram}

The context diagram illustrates the relationship between ATLAS, the device camera and the person using the application. Camera frames supply the signed input, while navigation and Finish controls determine the activity and completion of a Live session. The application returns recognized glosses, English text and local speech, and retains session records for History. These exchanges place the recognition process within the communication workflow rather than treating it as a separate video-classification task.

![ATLAS system context](review_assets/context.png)

**Figure 5.** *Context Diagram of the Proposed System*

#### **Data Flow Diagram** {#data-flow-diagram}

The Data Flow Diagram presents the transformation of camera input into readable and audible output. Apple Vision extracts landmarks, and MobileCLIP2 represents cropped-hand appearance. Streaming Sign Recognition combines these inputs with boundary estimates to recognize candidate intervals and accept a sequence of glosses. Gloss Management maintains their order so that T5-efficient-tiny can prepare English as the sequence develops. The Finish control then requests finalization of the remaining text, which is displayed and passed to local speech synthesis.

Trained model resources are installed with the application and used during these operations. Training recordings belong to model development, whereas the running application processes the camera stream. Session information passes to local History so that the generated text can be reviewed after the interaction.

![ATLAS data flow](review_assets/data_flow.png)

**Figure 6.** *Data Flow Diagram of the Proposed System*

#### **Use-case** {#use-case}

The use-case diagram illustrates the functions available to the Non-Sign Language User through the application. Live provides recognition and English output; Glosses presents sign demonstrations; Practice allows a selected sign or a random set to be attempted; and History provides access to saved sessions. Within Live, the Finish button or held-open-palms control completes the English output. Together, these functions connect communication support with opportunities to review signs and practice their production.

![ATLAS application use cases](review_assets/use_cases.png)

**Figure 7.** *Proposed System Use-Case Diagram*

#### **Flowchart** {#flowchart}

The application flowchart begins when ATLAS opens and presents Home navigation. The user may choose Live, Glosses, Practice or History. Live and Practice become available when the model resources have loaded. In Live, camera input produces recognized glosses and incremental English; Finish finalizes the remaining English before text and local speech output. Glosses provides demonstrations and an entry to practice a selected sign. Random practice offers sets of 5, 10 or 20 signs, feedback, a Skip action and a results screen. History displays saved session information and generated text. These branches return the user to navigation as the activity is completed.

![ATLAS iPhone application flow](review_assets/app_flow.png)

**Figure 8.** *Proposed System Flowchart*

## **4.3 Data Preparation and Model Development** {#4.3-data-preparation-and-model-development}

This section presents the data preparation and model development procedures used in ATLAS. It explains how recordings were cleaned and annotated, how model inputs were prepared, and how training methods supported recognition and English generation. The discussion then connects these procedures with the integration of the trained components into the iPhone application.

### **4.3.1 Preparation, Annotation and Data Separation**

The development of ATLAS required examples that represent both individual vocabulary signs and their use within a sequence. Isolated-sign clips supplied the labels for vocabulary recognition, sign-sequence recordings supplied connected signing, and gloss-to-English pairs supplied sentence-generation targets. Keeping these roles distinct allowed the researchers to prepare each input according to the task it supports, while retaining the relationship between the recording, its annotation and the resulting training example.

Preparation included label checks, annotation review and manual clipping of selected recordings. A gloss annotation identifies the sign, while a temporal annotation identifies where it occurs in a recording. Selected portions were manually clipped when needed to isolate the relevant content. Automatic activity trimming separately removed surrounding inactivity during isolated-sign preparation. For sequences with a known gloss order but no manually timed boundaries, alignment assigned the sequence to candidate intervals. These assigned intervals served as training supervision without being treated as manually annotated timing.

Signer-disjoint separation kept a person's isolated-sign recordings within the same assigned split, preventing that person's examples from appearing in both training and evaluation. This supports an assessment of recognition across people rather than repeated exposure to the same signer (Desai et al., 2023). Sequence studies retained their documented recording groups, and generated English sessions used held-out examples. Validation or designated tuning results guided model selection; the different study groups were retained when interpreting each comparison.

### **4.3.2 Input Preparation and Training Controls**

Landmark preparation expresses movement relative to the body so that changes in camera placement and signer size do not dominate the representation. Coordinates use a common scale, normally shoulder width, with a palm-based fallback when appropriate. Image proportions are preserved, and missing landmarks retain presence indicators. Each candidate sign interval is sampled into 32 frames containing 61 landmark points and five values per point: body-relative X and Y, a relative scale-based depth proxy, presence and confidence. This gives a 32 × 61 × 5 input. The fixed number describes the recognizer's input length; candidate intervals may occupy different amounts of recording time.

MobileCLIP2 supplies complementary appearance features from hand images. These features preserve visual information that may not be fully represented by landmark positions, while the landmark sequence describes motion and relative configuration. Their combination allows recognition to consider both sources of evidence (Jiang et al., 2021; Faghri et al., 2025).

Training introduces controlled variation so that the recognizer learns beyond the exact presentation of its examples. Position, uniform scale, rotation, timing and visibility can vary, and mirroring includes the corresponding left–right landmark exchange. During sign-interval adaptation, candidate boundaries also shift slightly. Regularization addresses how the model learns from these examples: dropout reduces dependence on particular internal activations, while weight decay discourages excessive parameter growth (Srivastava et al., 2014; Loshchilov & Hutter, 2019). Table 10 brings these preparation and training activities together.

**Table 10.** *Data Preparation and Training Methods*

| Method | Application in ATLAS | Purpose |
| --- | --- | --- |
| Label and recording checks | Preserve the selected vocabulary labels, verify recording identity and usable features, and retain preparation records. | Keep each example connected to its intended target. |
| Annotation and manual clipping | Review sign labels and isolate selected portions of recordings; use timed annotations where available. | Associate sign content with the relevant video interval. |
| Automatic trimming and sampling | Remove surrounding inactivity from isolated-sign inputs and sample frames into the recognizer's input format. | Represent the signing portion consistently. |
| Landmark normalization | Express positions relative to the body and use a common scale, normally shoulder width, while preserving image proportions. Missing landmarks retain presence indicators. | Reduce variation from camera placement and signer size while preserving relative hand movement. |
| Training augmentation | Vary landmark position, uniform scale, rotation, timing and visibility; mirror with the corresponding left/right landmark swap. Sign-interval training also varies candidate start and end positions. | Expose recognition to variation in appearance, motion and interval selection. |
| Regularization | Use dropout and weight decay in model training; retain isolated-sign examples and reference predictions during sign-interval adaptation. | Reduce reliance on particular training patterns and preserve learned vocabulary recognition during adaptation. |
| Data separation and model selection | Keep the isolated-sign evaluation signer-disjoint; preserve the documented recording groups for sequence studies. Use validation or designated tuning results for model selection. | Assess performance on examples separated from the corresponding training procedure. |

### **4.3.3 Recognition and Streaming Integration**

The researchers compared landmark extractors, visual input combinations, recognition architectures and model families before examining their integration into streaming recognition. Part-wise processing represents regions such as the hands and upper body before global temporal processing combines their movement. Squeezeformer provides attention and convolution within this temporal design, allowing the adapted recognizer to consider relationships across frames alongside nearby movement patterns (Lee et al., 2023; Kim et al., 2022).

Within the camera stream, recognition also requires a decision about which portion of movement belongs to a sign. ATLAS therefore uses a boundary model to suggest intervals and a segmental decoder to select and commit their recognized content. Its four-layer Transformer boundary model learned timing predictions through knowledge distillation from the pretrained Sign Language Segmentation teacher. The teacher's 2026 CNN–Transformer implementation uses MediaPipe pose inputs and was pretrained on German Sign Language; the related segmentation research and implementation are cited separately (Moryossef et al., 2023; Sign Language Processing, 2026). During this transfer, the teacher remained frozen while the ATLAS model learned from Apple Vision features. A short look-ahead supports timing decisions in the application, while the recognizer supplies the ASL vocabulary labels.

The recognizer was separately fine-tuned on candidate intervals resembling those encountered during streaming. Isolated-sign examples and reference predictions were retained during adaptation to preserve vocabulary recognition while learning from intervals with variable boundaries. The decoder then combined recognition scores with temporal evidence to form the accepted gloss sequence. Figure 9 shows how these responsibilities cooperate; the recorded HELLO HOW YOU example in Chapter 3 illustrates their operation within a clip.

![Streaming Recognition Decisions](review_assets/runtime_decisions.png)

**Figure 9.** *Streaming Recognition Decisions*

The researchers also tried Connectionist Temporal Classification (CTC), which learns sequences without requiring an exact boundary annotation for every sign (Graves et al., 2006). In the tested window-based implementation, the first full-window prediction required approximately 1.07 seconds of input before processing. Predictions could change as additional input arrived, and speech waited for a stable sequence. Live observations included repeated output for a held sign and a camera-timing mismatch, while separately trained causal CTC variants produced substantial connected-sign validation errors. These findings supported the choice of explicit boundary estimation and stable gloss commitment for this application. The decision concerns the tested implementations and their interaction behavior, rather than a general claim that CTC is inherently slow. Responsive interaction is also consistent with the real-time translation priorities reported in research involving Deaf and hard-of-hearing signers (Kamikubo et al., 2025).

### **4.3.4 English Generation and iPhone Integration**

T5-efficient-tiny learns the mapping from gloss sequences to English through text-to-text fine-tuning (Raffel et al., 2020). Incremental generation prepares sentence updates as glosses accumulate, retaining completed sentences after consistency checks. When the user selects Finish or holds both open palms for one second, the application finalizes the remaining text before presenting English and producing local speech. This arrangement distributes sentence-generation work across the interaction while keeping completion under an explicit control.

The trained neural components are exported to Core ML for local execution. FP32 represents floating-point values using 32 bits, whereas FP16 uses 16 bits and can reduce processing cost. The application uses FP16 exports of MobileCLIP2, the boundary model and the recognizer, together with FP32 exports of the English model. Precision and processor selection were measured separately from base-model architecture so that an accuracy comparison would not be confused with the effect of deployment settings. Figure 10 summarizes how the component comparisons inform these engineering choices.

![Model and Deployment Selection](review_assets/engineering_decisions.png)

**Figure 10.** *Model and Deployment Selection*

## **4.4 Results and Discussion** {#4.4-results-and-discussion}

The results are organized around the decisions that shape ATLAS: visual representation, recognition architecture, streaming output, English generation and iPhone execution. Top-1 accuracy measures how often the highest-scoring sign is correct; Top-5 accuracy measures how often the correct sign appears among the five highest-scoring predictions. Word error rate counts substitutions, missing signs and additional signs relative to the reference sequence, with lower values indicating fewer errors. Median processing time is the middle value of the recorded timings.

Each comparison retains its own task and measurement conditions. Recognition percentages below describe the corresponding validation studies, while the streaming comparison uses the same recorded sequences for both configurations. Base-classifier CPU timings use prepared inputs; the iPhone measurements include the stated processing work on the device. Detailed hardware, recording membership and source records are retained in the [measurement notes](ATLAS_MEASUREMENT_NOTES.md). This separation allows each result to answer the engineering question for which it was measured.

### **4.4.1 Landmark Extraction and Visual Inputs**

The landmark comparison used the same recordings with matching recognition configurations and training procedures. Both extractors completed feature extraction successfully, but the resulting processing time and recognition accuracy differed, as shown in Table 11 and Figure 11.

**Table 11.** *Landmark Extractor Comparison*

| Measure | Apple Vision | MediaPipe |
| --- | ---: | ---: |
| Successful clip extraction | 100% | 100% |
| Median extraction time per clip | 0.678 s | 1.230 s |
| Classifier top-1 | 93.12% | 89.95% |
| Classifier top-5 | 99.47% | 97.35% |

![Landmark Extractor Comparison](review_assets/extractors.png)

**Figure 11.** *Landmark Extractor Comparison*

Apple Vision reduced median extraction time from 1.230 to 0.678 seconds per clip and increased Top-1 recognition from 89.95% to 93.12%. These paired results support its selection alongside its integration with the iPhone's visual-processing framework. The extraction times describe clip preparation in this comparison; device frame-processing measurements are presented in Section 4.4.6.

The visual-input comparison then examined whether hand appearance contributed information beyond landmark motion. Each configuration was assessed on the same validation recordings.

**Table 12.** *Recognition Input Comparison*

| Recognition input | Top-1 | Top-5 |
| --- | ---: | ---: |
| Hand-image features | 80.69% | 94.71% |
| Landmarks | 95.50% | 98.68% |
| Learned combination | **96.30%** | **99.21%** |

![Recognition Input Comparison](review_assets/modalities.png)

**Figure 12.** *Recognition Input Comparison*

Landmarks provided the stronger individual input, reaching 95.50% Top-1 accuracy compared with 80.69% for hand-image features. Combining the inputs reached 96.30%, an increase of 0.80 percentage points over landmarks alone. The improvement supports complementary use of motion and appearance within the recognition component without implying that either input contributes equally.

### **4.4.2 Recognition Architecture and Model Families**

The architecture comparison considered how landmark information should be organized before temporal classification. Tables 13 and 14 compare landmark-based classifiers; their parameter counts exclude the separate hand-image encoder and the combined recognition model. M denotes millions of parameters, and CPU median denotes the median processing time for prepared model inputs on the development computer. A flat model processes the input together, part-wise processing first represents anatomical regions, and the graph-part alternative explicitly represents connections between landmarks.

**Table 13.** *Landmark Architecture Comparison*

| Architecture | Parameters | Top-1 | Top-5 | CPU median |
| --- | ---: | ---: | ---: | ---: |
| Graph-part replacement | 6.48 M | 78.31% | 95.77% | 11.42 ms |
| Wider flat Squeezeformer | 14.34 M | 95.24% | 99.74% | 7.35 ms |
| Flat Squeezeformer | 6.47 M | 95.77% | 100.00% | **4.90 ms** |
| Part-wise + global Squeezeformer | 6.79 M | **96.83%** | 99.21% | 6.50 ms |

![Landmark Architecture Comparison](review_assets/architecture.png)

**Figure 13.** *Landmark Architecture Comparison*

Part-wise and global Squeezeformer processing achieved the highest Top-1 accuracy in this comparison, at 96.83%. Relative to the flat Squeezeformer, the gain was 1.06 percentage points with an additional 1.60 milliseconds of median classifier computation. The flat model was faster and achieved the highest Top-5 value, while the selected architecture prioritized the correctness of the first prediction presented by recognition.

The selected family comparison presents a bidirectional long short-term memory network (BiLSTM), a temporal convolutional neural network (Temporal CNN), a Flat Transformer and the ATLAS landmark-recognition branch, evaluated using the same training recordings, validation recordings, training procedure and random seed. Recurrent models track information through a sequence; temporal convolution learns nearby frame patterns; Transformers relate frames using attention; and Squeezeformer combines attention with convolution. These families offer different balances between computation and recognition.

**Table 14.** *Recognition Model Family Comparison*

| Model family | Parameters | Validation top-1 | CPU median |
| --- | ---: | ---: | ---: |
| BiLSTM | 6.10 M | 89.15% | 3.39 ms |
| Temporal CNN | 5.90 M | 92.59% | 4.34 ms |
| Flat Transformer | 6.72 M | 95.50% | **2.88 ms** |
| ATLAS (part-wise + global Squeezeformer) | 6.79 M | **96.30%** | 6.15 ms |

![Recognition Model Family Comparison](review_assets/families.png)

**Figure 14.** *Recognition Model Family Comparison*

The Squeezeformer configuration reached 96.30% Top-1 accuracy, compared with 95.50% for the flat Transformer. Its median CPU time was 6.15 milliseconds versus 2.88 milliseconds, approximately 2.1 times the duration. Thus, its selection represents an accuracy priority with a measurable computation cost. These results come from one training run per family and identify competitive alternatives when processing cost receives greater weight. The architecture and family comparisons use separately trained checkpoints, which explains why their Squeezeformer values differ.

### **4.4.3 Streaming Recognition**

Individual-sign accuracy does not fully describe connected signing, where the system must also decide when a sign begins, ends and becomes stable enough to accept. The streaming comparison therefore measures sequence errors across two complete configurations on the same recordings. Boundary-guided interval classification passes predicted intervals to recognition; ATLAS streaming recognition combines the distilled boundary model, interval-adapted recognition and sequence selection.

**Table 15.** *Streaming Recognition Comparison*

| Recognition configuration | Word error rate |
| --- | ---: |
| Boundary-guided interval classification | 39.78% |
| ATLAS streaming recognition | **9.68%** |

![Streaming Recognition Comparison](review_assets/streaming.png)

**Figure 15.** *Streaming Recognition Comparison*

Word error rate decreased from 39.78% to 9.68%, a difference of 30.10 percentage points. This reflects the combined configuration, rather than an isolated effect of boundary distillation. The recordings include familiar-signer and unseen-signer sequences, providing a development comparison of how the system handles connected input. The result supports coordinating interval detection, recognition and gloss commitment instead of judging the streaming workflow solely from individual-sign accuracy.

### **4.4.4 English Output and Language-Model Execution**

English-output assessment used generated gloss-to-English sessions containing two to four sentences. DeepSeek supplied reference sentences and automatic judgments under a rubric that distinguishes fully correct, mostly correct and incorrect output. Fully correct output preserves the reference meaning, grammatical English and important content; mostly correct output retains the main meaning with a minor error; incorrect output changes, omits or invents important meaning. Research on language-model evaluation supports the use of explicit criteria, while the ratings here come from the study's own evaluator (Liu et al., 2023).

**Table 16.** *English-Output Comparison*

| Configuration | Judged fully correct | Judged wrong | BLEU | NO retained |
| --- | ---: | ---: | ---: | ---: |
| T5-efficient-tiny: whole-sequence generation | 60% | 6% | 75 | 100% |
| T5-efficient-tiny: incremental generation | 60% | 8% | 74 | 100% |

![English-Output Comparison](review_assets/english_quality.png)

**Figure 16.** *English-Output Comparison*

Both configurations used the same fine-tuned model with approximately 15.6 million parameters and received a 60% fully correct automatic rating. Incorrect ratings were 6% for whole-sequence generation and 8% for incremental generation. Neither configuration dropped NO in the evaluated examples containing that gloss. This measure concerns NO specifically; it does not cover every form of negation in ASL. BLEU measures overlap with reference wording, complementing the meaning-oriented rubric (Papineni et al., 2002). These are automatic judgments of generated text, with the same model family providing references and ratings; they describe English generation rather than human-assessed signing accuracy.

Execution screening examined the computation cost of candidate language-model architectures on an idle iPhone 13. The candidates used untrained weights for graph timing, and the estimates represent a 28-output-token session under each candidate's best measured processor setting. KV caching reuses attention calculations from previous output tokens. This screening provides a computation comparison, while Table 16 reports the trained model's English quality.

**Table 17.** *Language-Model Execution Screening*

| Candidate execution configuration | Parameters | Estimated 28-token time |
| --- | ---: | ---: |
| ATLAS tiny T5 architecture, KV cache, FP16 | 15.6 M | 173 ms |
| FLAN-T5-base, KV cache, FP16 | 248 M | 391 ms |

![Language-Model Execution Screening](review_assets/language_latency.png)

**Figure 17.** *Language-Model Execution Screening*

The tiny T5 architecture used by ATLAS had an estimated execution time of 173 milliseconds, compared with 391 milliseconds for FLAN-T5-base. The installed English model uses FP32 Core ML without an attention cache, so its configuration differs from the screening rows. At recorded output lengths, measured graph costs yield estimated median final-generation work of 148 milliseconds for incremental generation and 208 milliseconds for whole-sequence generation. Preparing completed sentences during signing therefore reduces the work remaining at Finish in this timing estimate.

### **4.4.5 From Trained Models to iPhone Execution**

ATLAS brings together four neural components with different responsibilities: the temporal boundary model, the Squeezeformer-based recognizer, the MobileCLIP2 image encoder and T5-efficient-tiny. The boundary model estimates candidate sign timing, while the recognizer determines which signs those intervals contain. MobileCLIP2 supplies hand-appearance features that complement landmark motion, and T5-efficient-tiny turns the accepted gloss sequence into English. These components connect visual recognition with sentence generation, but they are not a single trained model.

The researchers converted these trained components to Core ML to integrate recognition and English generation into the native iPhone application and execute them locally. The components were prepared for export through their PyTorch implementations, while the Swift application loads their exported Core ML representations and connects them to camera input, gloss management and English output. Core ML provides the chosen deployment runtime, with processor settings that permit supported operations to use the CPU, GPU and Neural Engine. Conversion therefore serves the practical purpose of bringing the trained models into the application.

The choice of numerical precision is a separate deployment decision. FP16 exports reduce the storage requirements of the recognition components, while matched comparisons examine how closely those exports preserve the original predictions and accuracy. Similar accuracy before and after conversion indicates preservation of learned recognition behavior. Execution-time measurements assess the resulting runtime performance, which can differ by component and processor rather than improve uniformly.

All four neural components have Core ML exports. The boundary model and recognizer use FP16 exports. MobileCLIP2 has both FP32 and FP16 exports, with FP16 used in the selected configuration. T5-efficient-tiny uses FP32 encoder and decoder exports; these are two computational parts of the same English-generation model. Table 18 identifies each model and its exported form.

**Table 18.** *Neural Models and Their Core ML Exports*

| Neural model | Purpose | Core ML export |
| --- | --- | --- |
| Temporal boundary model | Estimates where candidate signs begin and end from landmark features. | Boundary-model export using FP16. |
| Squeezeformer-based recognizer | Identifies candidate signs using landmark motion and hand-image features. | Recognition-model export using FP16. |
| MobileCLIP2 image encoder | Extracts appearance features from cropped hand images. | Image-encoder exports using FP32 or FP16; the selected configuration uses FP16. |
| T5-efficient-tiny | Generates English from accepted gloss sequences. | Encoder and decoder exports using FP32; both belong to one language model. |

Apple Vision supplies landmarks through an existing Apple framework; it was not converted by the researchers. The segmental decoder, Gloss Management and Finish controls are application logic rather than separately exported neural models. Camera capture, the interface and speech output likewise use native application frameworks. This distinction identifies which parts of ATLAS were exported and which coordinate their operation on the device.

FP16 and FP32 describe the numerical precision used in the model exports. They do not mean that every operation in the application, or every input array, uses the same representation. ATLAS combines FP16 visual and recognition exports with FP32 English-model exports according to the component configuration. Core ML's processor settings separately determine whether supported operations may execute on the CPU, GPU or Neural Engine.

Figure 18 follows the preparation of trained models for deployment and then their integration into the application. Export changes the executable representation of a model; the application still supplies preprocessing, connects model outputs and manages the interaction. The following subsection compares recognition accuracy and boundary-state agreement before and after conversion.

**Combined operation of the deployed system.** Camera frames supply landmark and hand-image information. Boundary estimates guide candidate-interval selection, and the recognizer scores those intervals using both visual inputs. The segmental decoder selects the accepted sequence, Gloss Management preserves its order, and T5-efficient-tiny prepares English as that sequence grows. Finish finalizes the remaining text for display and local speech. The models therefore cooperate within one workflow, while each retains a specific input, output and responsibility.

![Model export and iPhone integration](review_assets/deployment_flow.png)

**Figure 18.** *Model Export and iPhone Integration*

### **4.4.6 Recognition Performance Before and After Core ML Conversion**

Conversion was examined separately from model selection to determine how closely the exported models preserve their original behavior. The Squeezeformer recognizer was evaluated before and after conversion using the same validation examples, reference sign labels, landmark inputs and cached hand-image features. The comparison used the recognition output of the fixed-batch FP16 package loaded by the application. Holding the inputs constant isolates the recognizer conversion from changes in MobileCLIP2 feature extraction.

The boundary-model export was checked against the original model's timing-state predictions on the same tuning inputs. This measures whether conversion changes the predicted state—outside a sign, at its beginning or within it. It is an agreement measure with the original model, rather than accuracy against manually annotated sign boundaries.

**Table 19.** *Recognition Measures Before and After Core ML Conversion*

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

### **4.4.7 Training Behavior**

The training curves provide a view of how the component models learned before integration. They complement the comparison tables by showing changes across recorded training progress, while retaining the loss definitions and evaluation settings of each training procedure.

![Landmark Recognition Training](review_assets/recognition_training.png)

**Figure 19.** *Landmark Recognition Training*

Figure 19 shows landmark-recognition training across the recorded epochs. The selected checkpoint at epoch 100 achieved 96.83% validation Top-1 accuracy. Selection follows validation behavior rather than assuming that the final recorded epoch must provide the best recognition.

![Recognition Adaptation to Sign Intervals](review_assets/span_training.png)

**Figure 20.** *Recognition Adaptation to Sign Intervals*

Figure 20 follows the eight-epoch adaptation of recognition to candidate sign intervals, with epoch zero representing the starting recognizer. The sequence-error panel uses the decoder configuration designated for this adaptation study. Together, loss and sequence error show how interval training affects the inputs that streaming recognition must handle; its curve is interpreted within that training study rather than substituted for the complete-configuration comparison in Table 15.

![English-Model Fine-Tuning](review_assets/english_training.png)

**Figure 21.** *English-Model Fine-Tuning*

Figure 21 presents the saved English-model training observations through optimizer step 4,539, where the recorded validation loss was 0.2515. Lower loss indicates closer agreement with the reference training objective. The separate English-output assessment in Table 16 examines the generated sentences themselves, connecting model fitting with the meaning and wording produced for the application.

# **REFERENCES** {#references}

Amiruzzaman, S., Batchu, R. M., Amiruzzaman, M., Ngo, L., & Dewan, M. A. A. (2026). ASL recognition and game-based interaction: A machine learning–driven, gamified and accessible vocabulary learning system for Deaf learners. Computers, 15(5), 299\. https://doi.org/10.3390/computers15050299

Apple. (n.d.-a). Detecting hand poses with Vision. Apple Developer Documentation. https://developer.apple.com/documentation/vision/detecting-hand-poses-with-vision

Apple. (n.d.-b). Detecting human body poses in images. Apple Developer Documentation. https://developer.apple.com/documentation/vision/detecting-human-body-poses-in-images

Apple. (n.d.-c). Detect body and hand pose with Vision. Apple Developer. https://developer.apple.com/videos/play/wwdc2020/10653/

Boháček, M., & Hrúz, M. (2022). Sign pose-based transformer for word-level sign language recognition. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (pp. 182–191). https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek\_Sign\_Pose-Based\_Transformer\_for\_Word-Level\_Sign\_Language\_Recognition\_WACVW\_2022\_paper.pdf

Boise State University. (n.d.). Introduction. In Let’s Chat! American Sign Language (ASL). Pathways Project. https://boisestate.pressbooks.pub/pathwaysasl/front-matter/introduction-page/

Camgoz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign Language Transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10023–10033). https://openaccess.thecvf.com/content\_CVPR\_2020/html/Camgoz\_Sign\_Language\_Transformers\_Joint\_End-to-End\_Sign\_Language\_Recognition\_and\_Translation\_CVPR\_2020\_paper.html

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Castro-Ros, A., Pellat, M., Robinson, K., … Wei, J. (2024). Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70), 1–53. https://jmlr.org/papers/v25/23-0870.html

DeGrace, P., & Stahl, L. H. (1990). Wicked problems, righteous solutions: A catalogue of modern software engineering paradigms. Yourdon Press. https://archive.org/details/wickedproblemsri0000degr

Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4690–4699). https://openaccess.thecvf.com/content\_CVPR\_2019/html/Deng\_ArcFace\_Additive\_Angular\_Margin\_Loss\_for\_Deep\_Face\_Recognition\_CVPR\_2019\_paper.html

Desai, A., Berger, L., Minakov, F. O., Milan, V., Singh, C., Pumphrey, K., Ladner, R. E., Daumé III, H., Lu, A. X., Caselli, N., & Bragg, D. (2023). ASL Citizen: A community-sourced dataset for advancing isolated sign language recognition. arXiv. https://arxiv.org/abs/2304.05934

Eunice, J., Andrew, J., Sei, Y., & Hemanth, D. J. (2023). Sign2Pose: A pose-based approach for gloss prediction using a transformer model. Sensors, 23(5), 2853\. https://doi.org/10.3390/s23052853

Faghri, F., Anasosalu Vasu, P. K., Koc, C., Shankar, V., Toshev, A., Tuzel, O., & Pouransari, H. (2025). MobileCLIP2: Improving multi-modal reinforced training. Transactions on Machine Learning Research. https://openreview.net/forum?id=WeF9zolng8

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning (pp. 369–376). https://www.cs.toronto.edu/\~graves/icml\_2006.pdf

Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). Conformer: Convolution-augmented Transformer for speech recognition. Proceedings of Interspeech 2020, 5036–5040. https://doi.org/10.21437/Interspeech.2020-3015

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv. https://arxiv.org/abs/1503.02531

Hugging Face. (n.d.). Transformers documentation. https://huggingface.co/docs/transformers/index

International Organization for Standardization. (2023). ISO/IEC 25010:2023: Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model. https://www.iso.org/standard/78176.html

Kamikubo, R., Glasser, A., Lu, A. X., Daumé III, H., Kacorri, H., & Bragg, D. (2025). Exploring collaboration to center the Deaf community in sign language AI. Proceedings of ASSETS. https://doi.org/10.1145/3663547.3746390

Jiang, S., Sun, B., Wang, L., Bai, Y., Li, K., & Fu, Y. (2021). Skeleton aware multi-modal sign language recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 3413–3423). https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html

Kim, S., Gholami, A., Shaw, A., Lee, N., Mangalam, K., Malik, J., Mahoney, M. W., & Keutzer, K. (2022). Squeezeformer: An efficient Transformer for automatic speech recognition. In Advances in Neural Information Processing Systems, 35\. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract-Conference.html

Lee, T., Oh, Y., & Lee, K. M. (2023). Human part-wise 3D motion context learning for sign language recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 20740–20750). https://openaccess.thecvf.com/content/ICCV2023/html/Lee\_Human\_Part-wise\_3D\_Motion\_Context\_Learning\_for\_Sign\_Language\_Recognition\_ICCV\_2023\_paper.html

Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1459–1469). https://dxli94.github.io/WLASL/

Lin, K., Wang, X., Zhu, L., Sun, K., Zhang, B., & Yang, Y. (2023). Gloss-free end-to-end sign language translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 12904–12916). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.acl-long.722

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. Proceedings of EMNLP, 2511–2522. https://aclanthology.org/2023.emnlp-main.153/

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations. https://arxiv.org/abs/1711.05101

Moryossef, A., Jiang, Z., Müller, M., Ebling, S., & Goldberg, Y. (2023). Linguistically motivated sign language segmentation. Findings of EMNLP, 12703–12724. https://aclanthology.org/2023.findings-emnlp.846/

Moryossef, A., Yin, K., Neubig, G., & Goldberg, Y. (2021). Data augmentation for sign language gloss translation. In Proceedings of the 1st International Workshop on Automatic Translation for Signed and Spoken Languages (pp. 1–11). Association for Machine Translation in the Americas. https://aclanthology.org/2021.mtsummit-at4ssl.1/

Müller, M., Jiang, Z., Moryossef, A., Rios, A., & Ebling, S. (2023). Considerations for meaningful sign language machine translation based on glosses. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 682–693). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.acl-short.60

NVIDIA. (n.d.). CUDA Toolkit documentation. NVIDIA Developer. https://docs.nvidia.com/cuda/

OpenCV. (n.d.). OpenCV documentation. https://docs.opencv.org/

Python Software Foundation. (n.d.). Python documentation. https://docs.python.org/

PyTorch. (n.d.). PyTorch documentation. https://docs.pytorch.org/docs/stable/index.html

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). Bleu: a method for automatic evaluation of machine translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, 311–318. https://aclanthology.org/P02-1040/

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text Transformer. Journal of Machine Learning Research, 21(140), 1–67. https://jmlr.org/papers/v21/20-074.html

Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey. Expert Systems with Applications, 164, 113794\. https://doi.org/10.1016/j.eswa.2020.113794

Renjith, S., Varghese, A., & Poorna, S. S. (2026). An efficient real-time spatio-temporal adaptive motion pattern framework for isolated sign language recognition (RT-STAMP-SLR). Discover Artificial Intelligence, 6, 713\. https://doi.org/10.1007/s44163-026-01429-3

Renjith, S., Varghese, A., Rashmi, M., & Poorna, S. S. (2026). Transformer-based motion-visual integrated fusion for isolated sign language recognition. Computers and Electrical Engineering, 130, 110902\. https://doi.org/10.1016/j.compeleceng.2025.110902

Sign Language Processing. (2026). Sign Language Segmentation: CNN–Transformer implementation [Computer software]. https://github.com/sign-language-processing/segmentation

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56), 1929–1958. https://www.jmlr.org/papers/v15/srivastava14a.html

Syulistyo, A. R., Tanaka, Y., Pramanta, D., Fuengfusin, N., & Tamukoh, H. (2025). Low-cost computation for isolated sign language video recognition with multiple reservoir computing. PLOS ONE, 20(7), e0322717. https://doi.org/10.1371/journal.pone.0322717

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818–2826). https://doi.org/10.1109/CVPR.2016.308

Vasu, P. K. A., Pouransari, H., Faghri, F., Vemulapalli, R., & Tuzel, O. (2024). MobileCLIP: Fast image-text models through multi-modal reinforced training. Proceedings of CVPR. https://arxiv.org/abs/2311.17049

Violet, I. M. M., & Leena Sri, R. (2025). A comprehensive survey on recent advances and challenges in sign language recognition systems. Discover Artificial Intelligence, 5, 419\. https://doi.org/10.1007/s44163-025-00629-7

Woods, L. T., & Rana, Z. A. (2023). Modelling sign language with encoder-only transformers and human pose estimation keypoint data. Mathematics, 11(9), 2129\. https://doi.org/10.3390/math11092129

World Health Organization. (2026, March 3). Deafness and hearing loss. https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss

Yin, K., Moryossef, A., Hochgesang, J., Goldberg, Y., & Alikhani, M. (2021). Including signed languages in natural language processing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 7347–7360). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.acl-long.570

Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking. arXiv. https://arxiv.org/abs/2006.10214

Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. International Conference on Learning Representations. https://openreview.net/forum?id=r1Ddp1-Rb

Zheng, Z., Wang, Q., Yang, D., Wang, Q., et al. (2022). L-Sign: Large-vocabulary sign gestures recognition system. IEEE Transactions on Human-Machine Systems, 52(2), 290–301. https://doi.org/10.1109/THMS.2022.3146787
