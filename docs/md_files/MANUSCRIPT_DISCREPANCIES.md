# EXHAUSTIVE MANUSCRIPT DISCREPANCY REPORT (v16 System vs. Draft)

Based on a deep-dive review of all project documentation (`PAPER_REVIEW.md`, `SIMPLIFICATION_TEST_RESULTS.md`, `OPTIMIZATION_AND_MOBILE_PLAN.md`, `DEFENSE_PREPARATION.md`, and `CLAUDE.md`), the manuscript describes an older, much heavier system (v14/v15). 

The v16 system was aggressively simplified for mobile deployment. The manuscript **must** be updated to reflect these architectural simplifications to avoid contradictions during your defense.

---

## 1. CORE ARCHITECTURE & LOSS FUNCTION DISCREPANCIES (CRITICAL)

### 1.1 The Recognition Backbone (DS-GCN-TCN vs. Squeezeformer)
**What the Paper Says:** 
> "...required to develop the DS-GCN-TCN-Transformer sign language recognition and translation system."

**What the Actual System Is:**
The system completely dropped DS-GCN and TCN in v16. It now uses a custom **Squeezeformer** architecture.
*   *Why it changed (Defense point):* Simplification tests proved that the old Temporal Convolutional Network (TCN) was actively slowing down learning by 9x per step because it was redundant. The Squeezeformer efficiently combines spatial and temporal modeling using attention and depthwise convolutions, running 4.7x faster on CPU.

**Action:** Remove all mentions of "DS-GCN-TCN-Transformer" (like in the PERT chart) and replace with "Squeezeformer".

### 1.2 Loss Function (ArcFace vs. Cross-Entropy + Smoothing)
**What the Paper / Previous Drafts Said:**
Mentioned using "ArcFace" angular margin penalty for classification.

**What the Actual System Is:**
ArcFace was **removed** in v16. The system now uses standard **Cross-Entropy Loss with Label Smoothing (0.15)** and Mixup augmentation.
*   *Why it changed (Defense point):* ArcFace was needed for the old feature space, but with Squeezeformer and heavily cleaned data, standard CE with label smoothing converges better and is more stable.

**Action:** Ensure no mentions of ArcFace, angular margins, or cosine temperature scaling exist in the manuscript.

---

## 2. DATA EXTRACTION & FEATURE ENGINEERING (MAJOR SIMPLIFICATIONS)

### 2.1 Input Channel Reduction (16 Channels down to 5)
**What the Paper / Old Design Said:**
The system used a heavy 16-channel input tensor (XYZ coordinates + Velocity + Acceleration + Mask + Bone Direction + Bone Motion).

**What the Actual System Is:**
The system now uses a highly compressed **5-channel input**: `[X, Y, Z, Mask, Palm_Scale]`.
*   *Why it changed (Defense point):* Simplification tests proved that 11 of the 16 channels were just pre-computed mathematical derivatives. The Squeezeformer's internal temporal convolutions learn these dynamics automatically. Removing them reduced input size by 69% with zero accuracy loss, resulting in massive speedups.

**Action:** Explicitly state in the methodology that input features were simplified to raw 5-channel data to reduce computational overhead for mobile edge inference.

### 2.2 Signal Processing & Smoothing (1-Euro Filter removed)
**What the Paper / Old Design Said:**
Used complex post-processing: 1-Euro adaptive filters, Savitzky-Golay kinematic calculations, and explicit bone length stabilization.

**What the Actual System Is:**
Replaced entirely by a simple **Exponential Moving Average (EMA)** and a **Kalman Filter** (for gap recovery).
*   *Why it changed (Defense point):* 1-Euro had 4 tunable parameters per joint; EMA has 1. It yields the same visual stability but drops extraction time from ~300ms down to ~75ms per video.

### 2.3 Extraction Framework (Apple Vision vs. MediaPipe vs. RTMW-XL)
**What the Paper Says:**
> "...the extraction phase uses MediaPipe and Apple Vision to detect hand and body landmarks..."

**What the Actual System Is:**
The current v16 96.00% model was trained **exclusively on Apple Vision data**. 
*   *Why it changed (Defense point):* Your tests showed RTMW-XL (used in older versions) produced a **65% ghost hand rate**, destroying training data. Apple Vision was selected for its 5.6ms speed (184 FPS) and ~0% ghost hand rate. MediaPipe is acknowledged strictly as a fallback strategy for future Android cross-platform deployment, but it suffers from a 28x coordinate domain gap compared to Apple Vision.

**Action:** Clarify that Apple Vision is the primary extraction engine for the evaluated model, enabling the extreme CPU inference speeds.

---

## 3. PIPELINE STRUCTURE & TRANSLATION

### 3.1 Four Pipeline Stages (Not Three)
**What the Paper Says:**
> "...consists of three distinct stages: (1) a landmark extraction phase... (2) a Squeezeformer-based recognition stage... and (3) a T5-based translation stage..."

**What the Actual System Is:**
The system requires **four** distinct stages. You cannot go directly from isolated sign classification to continuous translation.
1. **Stage 0:** Extraction (Apple Vision -> 32-frame clips).
2. **Stage 1:** Isolated Classification (Squeezeformer -> 310 isolated classes).
3. **Stage 2:** Continuous Recognition (Shared Encoder + MultiScaleTCN + CTC Decoder -> continuous gloss sequences).
4. **Stage 3:** English Translation (Flan-T5).

**Action:** Update the objectives and methodology to explicitly include the **CTC (Connectionist Temporal Classification)** stage. Panelists will absolutely ask how you segment continuous video into words—CTC is the answer.

### 3.2 Stage 3: Flan-T5 vs. T5
**What the Paper Says:**
Refers exclusively to standard "T5 (Text-to-Text Transfer Transformer)".

**What the Actual System Is:**
The system uses **Flan-T5-Base** (and is migrating to **Flan-T5-Small**). 
*   *Why it matters (Defense point):* The "Flan" prefix means the model is **instruction-tuned**. Standard T5 requires heavy formatting; Flan-T5 understands conversational prompts out of the box (e.g., *"Translate this ASL gloss to natural conversational English: HELLO HOW YOU"*).
*   *Size Context:* Flan-T5-Small reduces the footprint from ~990MB to ~308MB, which is critical for the mobile application claim.

**Action:** Change "T5" to "Flan-T5" throughout the paper.

---

## 4. DEPLOYMENT & EVALUATION CLAIMS

### 4.1 "Mobile-Based" vs. Current Prototype State
**What the Paper Says:**
Tables 3, 4, 5, and the Scope section claim: "The system must function through a mobile-based application environment..."

**What the Actual System Is:**
The working prototype evaluated in this study runs natively on **macOS** (desktop) using Python (`inference_v16.py`). While the models have been successfully validated for mobile export (CoreML FP16 / TFLite), the actual mobile UI app is a *future integration phase*.

**Action:** Soften the language. State that the architecture is *designed and optimized* for mobile deployment (hence the aggressive simplifications), but the current evaluation is based on a macOS desktop implementation to validate the ML pipeline.

---

## 5. DATASET CLAIMS & METHODOLOGY METRICS

### 5.1 Dataset Specifics and "Data Cleaning"
**What the Paper Says:**
Vaguely refers to a "structured dataset of predefined sign language gestures."

**What the Actual System Is:**
A massive achievement of this capstone is the data pipeline. You processed 66,770 raw samples down to **62,023 highly curated training samples across 310 ASL classes**. 
*   *Why it matters:* You utilized **Confident Learning** (removing the bottom 3% of samples where the model proved the human labels were wrong) and statistical quality filters. This data cleaning alone provided a +2.7% accuracy boost.

**Action:** Add a paragraph in Chapter 4 detailing the dataset size (310 classes, 62k+ samples) and the 2-stage cleaning pipeline (confident learning + outlier removal).

### 5.2 Synthetic vs. Real Data in Stage 2
**What the Paper Says:**
Implies the system learns translation directly from the structured dataset.

**What the Actual System Is:**
Stage 2 (Continuous Recognition) is trained heavily on **Synthetic Continuous Data**—10,000 sequences created by concatenating isolated 32-frame clips using minimum-jerk biomechanical trajectory formulas (Flash & Hogan), combined with 780 real-world phrase videos.

**Action:** Panelists will ask how you obtained continuous ASL data. Be prepared to defend the synthetic concatenation + minimum-jerk transition synthesis described in your logs.
