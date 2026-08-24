# ATLAS — Optimization & Mobile Deployment Plan

## 1. Current System Summary

| Component | Details |
|---|---|
| Stage 0 (Extraction) | Apple Vision Framework, 5.4ms/frame (184 fps) hand detection, ~300ms full pipeline for 32 frames |
| Stage 1 (Classification) | DSGCNEncoderV14 + TemporalTCN + ArcFace, 5.4M params, 91.82% val accuracy, ~69ms CPU inference |
| Stage 2 (Continuous) | Frozen Stage 1 encoder + MultiScaleTCN + SequenceTransformer + CTC, 6.57% WER |
| Stage 3 (Translation) | Flan-T5-Base, 248M params, BLEU 84.2, ~400ms CPU inference |
| Total pipeline | ~370ms per sign (M4 MacBook Air) |
| App | ATLAS.app (PyInstaller, 1.6GB), macOS only |

---

## 2. Problems Identified

### 2.1 Overfitting (CONFIRMED)

**Evidence from v15 training (CE loss, no ArcFace):**

| Epoch | Train Loss | Val Loss | Val Acc | Gap |
|---|---|---|---|---|
| 10 | 5.22 | 5.22 | 27.9% | 0.00 |
| 20 | 1.13 | 1.36 | 66.0% | -0.23 |
| 30 | 0.41 | 0.88 | 83.5% | -0.47 |
| 50 | 0.11 | 0.69 | 88.9% | -0.58 |
| 100 | 0.02 | 0.73 | 91.6% | -0.71 |
| 138 | 0.01 | 0.75 | 91.8% | -0.73 |

- Train loss drops to near-zero (0.014) while val loss **rises** after epoch 50
- Model reaches 66% accuracy in just 20 epochs — learns too fast
- After epoch 30, the model is memorizing, not generalizing
- 17 regularization techniques slow it down but don't prevent it

### 2.2 Root Cause: Redundant Model Capacity

**The model has three sources of redundant capacity:**

#### A. Double Temporal Modeling (biggest problem)
- GCN blocks have `temporal_conv` inside them (kernel 3, 5, 5, 7) — temporal modeling #1
- TemporalTCN (4 dilated blocks, dilation 1, 2, 4, 8) does temporal modeling again — temporal modeling #2
- TCN alone is **3.5M parameters (65.7% of all params)**, redundant with GCN temporal conv
- The model processes the same temporal patterns twice with 2x the parameters needed

#### B. Pre-computed Derivative Features (redundant input)
The 16-channel input contains:
- XYZ (3ch) — **independent**
- Velocity (3ch) — derivative of XYZ, **computable by temporal conv**
- Acceleration (3ch) — derivative of velocity, **computable by temporal conv**
- Mask (1ch) — **independent**
- Bone direction (3ch) — **independent**
- Bone motion (3ch) — derivative of bone direction, **computable by temporal conv**

Only **7 channels are truly independent** (XYZ + bone_dir + mask). The other 9 hand the model pre-computed answers, making the task artificially easy.

#### C. High Parameter-to-Sample Ratio
- 5.4M parameters / 57,535 samples = **93.8 params per sample**
- Healthy ratio for this dataset size: 10-50
- The model has enough capacity to memorize every training sample

### 2.3 Extraction Pipeline is the Speed Bottleneck

Apple Vision hand detection: **5.4ms/frame** (fast)
Full extraction pipeline: **~300ms for 32 frames** (slow)

The 300ms comes from post-processing, not detection:
- 1-Euro adaptive low-pass filter
- Bone length stabilization (median + rescale per bone)
- Interpolation of missing detections
- Normalization (centering + scaling)
- Savitzky-Golay kinematics (velocity, acceleration)
- Bone feature computation

### 2.4 Model Too Large for Mobile

| Component | Size |
|---|---|
| Stage 1+2 model | ~50MB (FP32) |
| T5-Base | **944MB (FP32)** |
| Total app | ~1.6GB |

T5-Base (248M params) is massively overparameterized for a 310-word vocabulary translation task.

### 2.5 Platform Lock-in

- Apple Vision Framework is **iOS/macOS only**
- No Android support — limits deployment to Apple devices
- MediaPipe (cross-platform) has **different coordinate output** — 1.43 normalized mean diff vs Apple Vision, too large for model transfer

---

## 3. Proposed Solutions

### 3.1 Simplify Model Architecture (Remove TCN)

**Change:** Remove the 4-block TemporalTCN entirely.

| | Current | Proposed |
|---|---|---|
| GCN temporal conv | 4 blocks (kernel 3,5,5,7) | 4 blocks (kernel 3,5,5,7) — unchanged |
| TemporalTCN | 4 dilated blocks (3.5M params) | **REMOVED** |
| Total params | 5.4M | **~1.9M** |
| Temporal coverage | Redundant 2x | 1x (sufficient — GCN covers 17 of 32 frames) |

**Why safe:** GCN blocks already have temporal convolution inside each block. The TCN is redundant temporal processing. Removing it slows learning (model can't memorize as fast) and reduces overfitting.

**Expected impact:** -1.0 to -2.5% accuracy, but +1.0 to +2.0% from reduced overfitting → net -0.5% to 0%.

### 3.2 Reduce Input Channels (16ch → 7ch)

**Change:** Remove pre-computed derivative features from .npy files.

| Channel | Keep/Remove | Reason |
|---|---|---|
| XYZ (0-2) | **KEEP** | Primary spatial signal |
| Velocity (3-5) | REMOVE | Learnable from XYZ by temporal conv |
| Acceleration (6-8) | REMOVE | Learnable from velocity |
| Mask (9) | **KEEP** | Hand presence flag |
| Bone direction (10-12) | **KEEP** | Hand shape signal |
| Bone motion (13-15) | REMOVE | Learnable from bone direction |

**New input:** [B, 32, 61, 7] — XYZ (3) + mask (1) + bone_dir (3)

**Why safe:** Temporal convolutions in GCN blocks can learn first/second derivatives from raw XYZ. Pre-computing them just makes the task too easy for the model.

**Expected impact:** -0.5 to -1.5% accuracy, but forces model to learn temporal patterns rather than memorize pre-computed features.

### 3.3 Reduce Model Width (d_model 384 → 256)

**Change:** Reduce embedding dimension across all components.

| | Current (384) | Proposed (256) |
|---|---|---|
| GCN params | ~1.5M | ~700K |
| Node attention | 37K | 17K |
| Angle features | 193K | 130K |
| Classifier head | 120K | 67K |
| Total (without TCN) | ~1.9M | **~1.0M** |

**Why safe:** d_model=256 achieved 85% in v12 without angle features. With angle features (the primary discriminative signal), 256 should reach 87-89%.

### 3.4 Simplify Extraction Pipeline

**Change:** Replace heavy post-processing with lightweight alternatives.

| Step | Current | Proposed | Savings |
|---|---|---|---|
| Smoothing | 1-Euro adaptive filter | Simple exponential moving average | ~20ms |
| Bone stabilization | Median bone length + rescale per frame | **Remove** (Apple Vision is consistent) | ~50ms |
| Kinematics | Savitzky-Golay (polynomial fit) | **Remove** (no longer in input) | ~30ms |
| Bone features | Explicit computation | **Keep** (still in input as bone_dir) | 0ms |
| Normalization | Center + scale | **Keep** (critical for signer-invariance) | 0ms |
| Temporal resampling | Linear interpolation to 32 frames | **Keep** (fixed input size) | 0ms |

**New extraction:** detect → interpolate missing → simple EMA smooth → normalize → resample → compute bone_dir → done

**Expected time:** ~100ms (down from ~300ms)

**Requirement:** Must retrain with simplified extraction (extraction must match between training and inference).

### 3.5 Downsize T5 (Base → Small)

**Change:** Replace Flan-T5-Base (248M) with Flan-T5-Small (77M).

| | T5-Base | T5-Small |
|---|---|---|
| Parameters | 248M | **77M** |
| FP32 size | 944 MB | **308 MB** |
| INT8 quantized | ~237 MB | **~77 MB** |
| CoreML (FP16) | ~475 MB | **~155 MB** |
| CPU inference | ~400ms | **~120ms** |
| iPhone inference | Too heavy | **~80-150ms** |

**Why safe:** The translation task is trivial — 310-word vocabulary, ~500 unique patterns, short phrases. T5-Small has 77M params for ~500 patterns = 154,000 params per pattern. Massively overparameterized. Fine-tune with the same 28K training pairs and expect identical BLEU scores.

### 3.6 Keep Angle Features (NO CHANGE)

The 118 angle features (59 angles + 59 velocities) are the **highest-value component**:
- Only 193K params (3.6% of model)
- 0.90 cosine similarity across signers (signer-invariant)
- Provides the primary discriminative signal
- Without them, accuracy drops ~10-15%

**Do not remove.**

---

## 4. Combined Impact Estimate

| Metric | Current | After All Changes |
|---|---|---|
| **Val Accuracy** | 91.82% | **87.5% - 90.5%** |
| **Model size (Stage 1)** | 5.4M params | **~1.0-1.2M params** |
| **T5 size** | 248M (944 MB) | **77M (155 MB CoreML)** |
| **Total app size** | ~1.6 GB | **~200 MB** |
| **Extraction time** | ~300ms | **~100ms** |
| **Stage 1 inference** | 69ms | **~25ms** |
| **T5 inference** | 400ms | **~120ms** |
| **Total pipeline** | 370ms | **~130ms** |
| **Train-val gap** | 0.73 (unhealthy) | **~0.2 (healthy)** |
| **Params per sample** | 93.8 | **~20 (healthy)** |

**Worst case:** lose ~4% accuracy for 78% smaller model, 3x faster pipeline, healthy training curve.
**Best case:** lose ~1% because reduced overfitting offsets reduced capacity.

---

## 5. Architecture Comparison: DS-GCN vs GAT

The panelists asked about Graph Attention Networks. Here's the benchmark:

| | DS-GCN (ours) | GAT (8 heads) |
|---|---|---|
| 4-layer forward pass | **9.6ms** | **118.4ms** |
| Speed ratio | 1x | **12.3x slower** |
| Parameters | 1.17M | 393K |
| Attention memory | **0 MB** | **14.5 MB** (NxN attention matrices) |
| Accuracy gain | Baseline | +1-3% theoretical |

**GAT computes pairwise attention [B, T, 61, 61, H] at every layer** — that's why it's 12x slower despite fewer parameters. DS-GCN uses fixed adjacency with cheap per-channel scaling.

**Literature confirms:** Chen et al. (2024) and Liao et al. (2022) achieved 93-95% with GAT but noted "high computational latency, unsuitable for lightweight applications" (Zhang et al., 2023).

**Verdict:** GAT would kill real-time inference for marginal accuracy gains. DS-GCN is the correct choice for edge deployment.

---

## 6. Skeleton vs RGB Video Comparison

| Requirement | Our System (Skeleton) | Best RGB Option |
|---|---|---|
| Accuracy (300+ classes) | 91.82% | ~56-67% (WLASL-300) |
| CPU inference | 370ms | 2-5 seconds |
| Model size | 5.4M | 25-90M+ |
| Laptop/phone capable | Yes | GPU required |
| Continuous recognition | Yes (CTC) | Possible but slower |

**No published RGB-based system achieves >70% on 300-class ASL while running on a laptop CPU under 400ms.**

The skeleton approach provides a **1000x dimensionality reduction** (150K pixels → 141 skeleton values per frame), enabling CPU-only inference. This is the only viable approach for mobile edge deployment.

---

## 7. Mobile Deployment Options

### Platform Choice

| Option | Extractor | Pros | Cons |
|---|---|---|---|
| **iOS only** | Apple Vision | Fastest (5.4ms), already have training data | No Android |
| **Android only** | MediaPipe | Cross-platform | Need to retrain with MediaPipe extraction |
| **Both** | MediaPipe | Full coverage | Need to retrain, MediaPipe slightly slower |

**Apple Vision vs MediaPipe gap:** 1.43 normalized mean diff — too large for cross-use. Must pick one and train with it.

### Model Conversion

| Framework | Stage 1+2 | Stage 3 (T5) |
|---|---|---|
| iOS (CoreML) | PyTorch → CoreML (coremltools) | Flan-T5-Small → CoreML |
| Android (TFLite) | PyTorch → ONNX → TFLite | Flan-T5-Small → TFLite |

### T5 Alternatives for Older iPhones (< iPhone 15 Pro)

| Option | Size | Speed | Quality | Offline |
|---|---|---|---|---|
| **T5-Small (CoreML INT8)** | ~77MB | ~80-150ms | Identical to T5-Base | Yes |
| **Rule-based templates** | ~1MB | <1ms | 95% coverage for 310 classes | Yes |
| **Cloud API fallback** | 0MB on device | ~200ms + network | Best quality | No |
| **Hybrid (rules + cloud)** | ~1MB | <1ms / ~200ms | Best of both | Partial |

**Recommendation:** T5-Small with INT8 quantization fits on iPhone 12+ (~77MB). For iPhone 11 and below, use rule-based templates with cloud fallback.

---

## 8. Regularization Techniques Currently in Use

| Technique | Stage | Parameters | Status |
|---|---|---|---|
| Dropout | S1 | encoder=0.10, head=0.30 | ON |
| Drop-Graph (node masking) | S1 | node_drop_rate=0.05 | ON |
| DropPath (stochastic depth) | S1 | 0→0.1 linear | ON |
| Weight Decay (AdamW) | S1/S2 | 0.01 | ON |
| Label Smoothing | S1/S3 | 0.10 | ON |
| Online Augmentation | S1 | rotation 8deg, scale 0.6-1.4x, noise, speed warp | ON |
| MixUp + CutMix | S1 | alpha=0.1, cutmix_prob=0.15 | ON |
| Focal Loss | S1 | gamma=1.0 | ON |
| ArcFace angular margin | S1 | m=0.5, s=30 | ON (optional) |
| EMA | S1/S2 | decay=0.999 | ON |
| Class-balanced sampling | S1 | temperature=0.5 | ON |
| Early stopping | S1/S2 | patience=50/35 | ON |
| Focal CTC | S2 | gamma=2.0 | ON |
| CR-CTC consistency | S2 | weight=0.3 | ON |
| Encoder freeze → unfreeze | S2 | epoch 30, 0.1x LR | ON |
| Noisy gloss augmentation | S3 | 30% probability | ON |
| Balanced Softmax | S1 | — | OFF |

Despite 17 techniques, overfitting still occurs because the **root cause is redundant model capacity**, not insufficient regularization.

---

## 9. Apple Vision Benchmark (M4 MacBook Air)

| Test | Result |
|---|---|
| Per-frame hand detection | 5.4ms avg (184 fps) — **real-time** |
| Full extraction (32 frames) | ~300ms (post-processing bottleneck) |
| Sliding window (stride=16) | Pipeline 409ms < stride 533ms — **can keep up** |
| Total latency (sliding window) | ~942ms |
| Stage 1 inference | 69ms |
| Full pipeline (extract + infer) | 370ms per sign |

### Sliding Window Real-Time Approach
- Thread 1: capture + detect at 30fps (5.4ms/frame)
- Thread 2: every ~1 second, extract last 32 frames + infer
- Result: predictions every ~1 second with <1s latency
- Status: **PROVEN FEASIBLE** (tested)

---

## 10. Implementation Order (Recommended)

1. **Fine-tune T5-Small** — fastest win, reduces app from 1.6GB to ~200MB (1 day)
2. **Remove TCN from model** — biggest param reduction, requires retrain (1 day code, 4h train)
3. **Reduce to 7ch input** — simplify extraction, requires re-extract + retrain (1 day)
4. **Reduce d_model to 256** — further param reduction (retrain with step 3)
5. **Simplify extraction pipeline** — faster inference (with step 3)
6. **Convert to CoreML/TFLite** — mobile deployment (1-2 days)
7. **Build mobile app** — Swift (iOS) or React Native (both) (1-2 weeks)

Steps 2-5 can be done in a single retrain cycle on Vast.ai (~4-6 hours).
