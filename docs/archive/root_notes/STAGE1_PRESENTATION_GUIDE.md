# SLT Pipeline — Complete Presentation Guide

Use this as your script to explain **every stage** to your teacher: **Extract**, **Stage 1**, **Stage 2**, and **Stage 3**. Each section has **what to say**, the **mathematics**, and **why we did it that way**.

---

# Part A: Pipeline Overview

**Say:**  
"Our system has four main parts: (1) **Extract** turns raw videos into hand-landmark sequences; (2) **Stage 1** classifies isolated signs (one clip → one gloss); (3) **Stage 2** recognizes sequences of signs (continuous signing → gloss sequence) using CTC; (4) **Stage 3** translates gloss sequences into natural English using a T5 model. Each stage has a clear purpose and we chose the methods for specific reasons."

**Flow:**  
Raw video → **Extract** → `.npy` files [32, 42, 10] → **Stage 1** (isolated sign) → **Stage 2** (sequence of signs, CTC) → **Stage 3** (gloss → English).

---

# Part B: EXTRACT (Preprocessing)

**Purpose:** Turn raw ASL videos into a **single tensor per video**: fixed length (32 frames), 42 hand landmarks, 10 channels (position + velocity + acceleration + mask). This format is what Stage 1 and Stage 2 expect. We do **not** feed raw pixels to the neural networks.

---

## B.1 Why We Need Extract

- **Stage 1 and 2 expect skeleton data**, not video pixels. So we need a preprocessing step that turns video into landmark sequences.
- **MediaPipe Hands** gives us 21 points per hand in normalized (x, y, z). We run it on each frame (or every k-th frame), then **fill gaps**, **resample to 32 frames**, **normalize**, and **add velocity/acceleration**. That way every clip has the same shape and the same kind of features (pose + motion).

---

## B.2 Steps in Extract (and Why Each One)

### 1. Resize to 384px (longest side)

- **What:** Before running MediaPipe, we resize each frame so the longest side is 384 pixels.
- **Why:**  
  - **Speed:** Smaller images → faster MediaPipe.  
  - **Enough for hands:** Landmarks are normalized (0–1); 384px is enough for reliable hand/finger detection.  
  - **Consistency:** Every video is at a similar scale, so we avoid mixing 4K and 720p and keep behavior predictable.

### 2. Frame skipping (1.5× oversampling)

- **What:** We don’t run MediaPipe on every frame. We use `skip = max(1, total_frames // (32 * 1.5))` and only process every `skip`-th frame.
- **Why:**  
  - We only need **32 frames** per clip after resampling. Reading every frame is wasteful; sampling ~1.5× that (e.g. 48 frames from a 72-frame video) gives enough temporal coverage.  
  - When `skip > 1`, we set MediaPipe to **static_image_mode=True** so each frame is treated as an image (better for sparse frames). When `skip == 1`, we use video mode for smoother tracking.

### 3. MediaPipe + left/right sequences

- **What:** For each processed frame we get left and/or right hand landmarks (21 points × 3 coords). We store them in `l_seq` / `r_seq` and the frame indices in `l_valid` / `r_valid`.
- **Why:** We need **which** hand and **when** it was detected. Some frames may miss one or both hands; the next step fills those gaps.

### 4. Interpolate hand (`interpolate_hand`)

- **What:** We have landmarks only at frames where a hand was detected. We produce a **dense** sequence: one pose per frame for the whole video length, using linear interpolation in time.
- **Math:** For each coordinate we have values at positions `x_old = valid_indices`. We want values at `x_new = 0, 1, ..., total_frames-1`. We use `np.interp(x_new, x_old, values)` so missing frames get a blend of the nearest detected frames.
- **Why:** Downstream steps assume a **continuous** sequence over time. Gaps would break temporal resampling and kinematics. Interpolation is the simplest way to fill gaps without inventing motion.

### 5. Concatenate left + right → (T, 42, 3)

- **What:** Left hand (21 points) and right hand (21 points) are concatenated along the point dimension → 42 points per frame.
- **Why:** Stage 1 and 2 are built for **dual-hand** input (one tensor with 42 nodes). Order is fixed (e.g. left then right) so the graph and masks are consistent.

### 6. Temporal resample (`temporal_resample`) → 32 frames

- **What:** Map the current number of frames N to exactly **32** by treating time as 0→1 and interpolating every channel to 32 evenly spaced time steps.
- **Math:** `x_old = linspace(0, 1, N)`, `x_new = linspace(0, 1, 32)`. For each scalar channel: `interp(x_new, x_old, values)` → 32 values. Reshape to (32, 42, 3).
- **Why:** Stage 1 and Stage 2’s encoder expect **fixed-length** clips (32 frames). So we normalize duration: short videos are “stretched,” long ones “compressed” in time, and we always feed 32 frames. We use interpolation (not just dropping frames) to preserve the full motion from start to end.

### 7. Normalize sequence (`normalize_sequence`)

- **What:** (a) **Center:** subtract the median wrist position(s) from all landmarks (per hand). (b) **Scale:** divide by the median length of a reference bone (e.g. wrist → middle MCP).
- **Math:**  
  - Center:  \text{pose}' = \text{pose} - \text{median}(\text{wrist positions}) .  
  - Scale:  \text{pose}'' = \text{pose}' / (\text{median}(\text{bone length}) + \epsilon) .
- **Why:**  
  - **Invariance to position:** Sign meaning doesn’t depend on where the person stands. Centering removes global position.  
  - **Invariance to hand size:** Different people have different hand sizes; scaling by bone length makes the geometry comparable so the model learns shape and motion, not absolute size.

### 8. Compute kinematics (`compute_kinematics_batch`)

- **What:** From position (3 channels) we add **velocity** (central difference), **acceleration** (central difference of velocity), and a **mask** (1 where hand present, 0 otherwise). Output: (32, 42, **10**).
- **Math:**  
  - Velocity:  v_t = (x_{t+1} - x_{t-1}) / 2  (and copy at boundaries).  
  - Acceleration: same on  v .  
  - Mask: 1.0 for nodes 0–20 if left hand ever present, 1.0 for 21–41 if right hand ever present; else 0.
- **Why:**  
  - **Motion matters:** Signs are defined by movement. Velocity and acceleration make that explicit.  
  - **Mask:** One-handed signs only use one hand; the mask tells the model which nodes are valid so it doesn’t treat zeros as “both hands at origin.”

### 9. Save as float16 `.npy`

- **What:** Save the (32, 42, 10) array as `.npy` in float16.
- **Why:** Saves disk and memory; precision is enough for landmarks. Filename includes label (gloss) and a hash so we can trace back to the video and avoid reprocessing.

**Why we did Extract this way (summary):** We need a single, fixed-format representation (32, 42, 10) that is **position- and scale-invariant** and includes **motion**. Resize + skip keep cost low; interpolation and resampling give dense, fixed-length sequences; normalization and kinematics make the signal suitable for the graph-based models in Stage 1 and 2.

---

# Part C: STAGE 1 — Isolated Sign Classification

**Purpose:** Given one clip of shape (32, 42, 10), predict **one** ASL gloss (e.g. HELLO, I, THANK). This is the building block for recognizing signs; Stage 2 will later chain these into sequences.

---

## C.1 Model Input Size and Why

- **Input shape:** [B, 32, 42, 10] — batch, 32 frames, 42 nodes, 10 channels (from Extract).
- **Why 32:** Fixed clip length; all clips from Extract are already 32 frames.  
- **Why 42:** Two hands × 21 landmarks (MediaPipe); the graph is defined over these 42 nodes.  
- **Why 10:** Position (3) + velocity (3) + acceleration (3) + mask (1); pose + motion + validity.

(Preprocessing “image size” for Extract is 384px for MediaPipe; the **model** only sees the tensor above.)

---

## C.2 Epoch Convergence

- **Epoch** = one full pass over the training set. **Convergence** = loss and accuracy stabilize.
- **Training loss** (blue): Drops then plateaus ~1.0–1.2 (regularization: dropout, mixup, label smoothing make training “harder”).  
- **Validation loss** (orange): Keeps decreasing to ~0.25 → good generalization.  
- **Validation accuracy** (green): Rises to ~95–97% and stabilizes.  
- **Learning rate** (purple): Warm-up then cosine decay so we take big steps early and small steps late.

---

## C.3 Loss: Cross-Entropy (Not MAE/RMSE/MSE)

- Stage 1 is **classification**, so we use **cross-entropy**, not regression metrics.
- **Formula:**  \mathcal{L} = -\frac{1}{N} \sum_i \log p_{y_i}^{(i)} . We penalize low probability on the correct class.
- **MAE, MAPE, RMSE, MSE** are for regression; we can write their formulas for completeness but we don’t use them for Stage 1.

---

## C.4 Gradient Descent and Learning Rate

- **Update:**  \theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta} . We use **AdamW** (adaptive step sizes) and **cosine LR schedule** (warm-up + decay) for stable convergence.

---

## C.5 Confusion Matrix

- Rows = true class, columns = predicted. Diagonal = correct; off-diagonal = confusions.
- **Precision** = TP/(TP+FP), **Recall** = TP/(TP+FN), **F1** = 2·P·R/(P+R), **Accuracy** = sum(diagonal)/sum(all).
- Our matrix: high diagonal, main confusion I↔J (similar-looking signs).

**Why we did Stage 1 this way:** We need a strong **isolated sign** recognizer that works on the same [32, 42, 10] representation we use later. DS-GCN + transformer on graphs fits skeleton data; cross-entropy and accuracy are the right loss and metric for classification.

---

# Part D: STAGE 2 — Continuous Sign Recognition (CTC)

**Purpose:** Take a **sequence** of 32-frame clips (multiple signs in a row) and output the **sequence of glosses** (e.g. HELLO HOW YOU). Stage 1 only does one sign at a time; Stage 2 handles continuous signing and aligns the model output to the gloss sequence using **CTC** (Connectionist Temporal Classification).

---

## D.1 Why We Need Stage 2

- In real signing, several signs follow one another. We need a model that (1) takes variable-length input (many 32-frame clips concatenated), (2) outputs a **sequence** of labels (one per sign), and (3) does **not** require frame-level alignment (we don’t know exactly which frames belong to which sign). CTC solves the alignment problem by allowing a special “blank” token and then collapsing repeats.

---

## D.2 Architecture (and Why)

- **Encoder:** Same DS-GCN encoder as Stage 1 (same [32, 42, 10] → temporal features). We **load Stage 1’s encoder weights** and **freeze** them.
- **Why freeze:** (1) Stage 1 already learned good spatial–temporal features for signs. (2) Saves VRAM and training time. (3) We only train the **sequence** part (BiLSTM + classifier) to map encoder outputs to gloss sequences.
- **Temporal pool:** Each 32-frame clip is pooled to **4** time steps (AdaptiveAvgPool1d(4)). So one sign → 4 feature vectors.
- **Why 4:** Reduces sequence length for the LSTM while keeping enough resolution per sign. 4 is a tradeoff between detail and compute.
- **BiLSTM:** Processes the sequence of 4-vectors per sign. Bidirectional so context from both sides is used.
- **Classifier:** Linear layer from LSTM hidden size to **vocab size** (glosses + blank). Output: one logit per time step per class; CTC will interpret this as a distribution over alignments.

---

## D.3 CTC Loss (Math and Why)

- **Problem:** We have input length T (e.g. 4 × number of signs) and target sequence of length L (number of glosses). We don’t have a frame-to-gloss alignment.
- **CTC:** Introduces a **blank** token (index 0). The model outputs a distribution over (glosses + blank) at each time step. An **alignment** is a sequence of T symbols (glosses and blanks). We **marginalize** over all alignments that collapse to the target gloss sequence (by removing blanks and merging consecutive same gloss).
- **Loss:**  \mathcal{L}_{\text{CTC}} = -\log P(\text{target} \mid \text{logits}) , where the probability is the sum over all valid alignments. PyTorch’s `CTCLoss` computes this with a dynamic-programming forward pass.
- **Why CTC:** We don’t need to label each frame; we only need gloss sequences. CTC trains the model to output the right sequence in the right order while allowing variable stretch (more or fewer frames per sign).

---

## D.4 Training Data: Synthetic Sequences

- We don’t have long videos labeled with gloss sequences. So we build **synthetic** sequences: we randomly sample 2–6 glosses, load the corresponding `.npy` files (each 32 frames), **concatenate** them along time, and use the ordered list of glosses as the target.
- **Why:** (1) We have many isolated sign clips from Extract + Stage 1 data. (2) Concatenating them simulates “continuous” signing. (3) The target is well-defined (the list of gloss names). So we can train Stage 2 without hand-annotated continuous data. In real deployment, we’ll run on real continuous video; the model has learned to segment and recognize from synthetic sequences.

---

## D.5 Evaluation: WER (Word Error Rate)

- **WER** = (edit distance between reference and predicted gloss sequences) / (number of reference words), as a percentage. Edit distance = minimum insertions, deletions, substitutions to turn prediction into reference (Levenshtein).
- **Why WER:** Standard in speech/sign recognition for sequence output. Lower WER = better. We optimize CTC loss; we report WER to measure how well the decoded sequence matches the reference.

---

## D.6 Decoding (CTC)

- At inference we take **argmax** per time step, then **collapse**: remove blanks, merge consecutive same gloss. That gives the predicted gloss list.
- **Why:** Simple and matches how CTC is trained. (Beam search can be added for better decoding; our pipeline may use it in the video inference script.)

**Why we did Stage 2 this way:** We need sequence-level recognition without frame-level labels. CTC + frozen encoder + BiLSTM lets us reuse Stage 1 and train only the sequence part. Synthetic data makes training possible with the data we have; WER measures sequence accuracy.

---

# Part E: STAGE 3 — Gloss to English Translation

**Purpose:** Map a **sequence of ASL glosses** (e.g. HELLO HOW YOU) to **natural English** (e.g. “Hello, how are you?”). Stage 1 and 2 produce glosses; Stage 3 turns them into readable text.

---

## E.1 Why We Need Stage 3

- Glosses are not natural language; they’re lexical labels. Users need fluent English. So we add a **translation** step: gloss string → English sentence. We treat it as a **sequence-to-sequence** task and use a pre-trained text model (T5).

---

## E.2 Model: T5 (Seq2Seq)

- We use **T5-small** (or similar): encoder–decoder transformer, pre-trained on English. We **fine-tune** it on (gloss sequence → English) pairs.
- **Input:** A prompt like `"translate ASL gloss to English: HELLO HOW YOU"`.  
- **Output:** English sentence generated autoregressively (beam search, max length 48, etc.).
- **Why T5:** (1) Good at conditioned text generation. (2) “Translate X to Y” fits our task. (3) Small enough to train on limited data; we only need to adapt it to gloss vocabulary and style.

---

## E.3 Training Data (generate_stage3_data.py)

- We don’t have large (gloss, English) corpora. So we **generate** training pairs from rules and vocabularies: we define sentence templates (e.g. subject + verb + object), map glosses to words, apply simple grammar (e.g. conjugation, articles), and produce (gloss string, English sentence) pairs. Script: `generate_stage3_data.py`; output: e.g. CSV with columns `gloss` and `text`.
- **Why generate:** (1) No large parallel gloss–English dataset. (2) Rule-based generation gives many consistent pairs so the model learns the mapping and basic grammar. (3) We can control vocabulary and structure. Later, real paired data could be added.

---

## E.4 Loss and Training

- **Loss:** Standard **cross-entropy** on the decoder output (next-token prediction). The Hugging Face `Seq2SeqTrainer` handles this.
- **Settings:** Learning rate 3e-4, cosine scheduler, warm-up, a few epochs (e.g. 10), early stopping on eval loss. We save the best checkpoint.
- **Why these choices:** Standard for fine-tuning small encoder–decoder models; cosine and early stopping avoid overfitting on small data.

**Why we did Stage 3 this way:** We need gloss→English with limited data. T5 gives a strong prior; rule-based data generation gives us a training set; fine-tuning adapts the model to our gloss vocabulary and desired output style.

---

# Part F: Recap Table (What to Say for “Everything”)


| Stage       | Input               | Output                | Loss / Metric                      | Why this design                                                                                                                                       |
| ----------- | ------------------- | --------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Extract** | Raw video           | (32, 42, 10) per clip | N/A (preprocessing)                | Single format for all models; 384px + skip for speed; interpolate + resample for dense fixed length; normalize for invariance; kinematics for motion. |
| **Stage 1** | (32, 42, 10)        | One gloss             | Cross-entropy; Accuracy            | Isolated sign classification; fixed length; graph + transformer on skeletons; confusion matrix for per-class analysis.                                |
| **Stage 2** | Seq of (32, 42, 10) | Gloss sequence        | CTC loss; WER                      | Continuous signing without frame labels; reuse Stage 1 encoder (frozen); synthetic sequences for training.                                            |
| **Stage 3** | Gloss string        | English sentence      | Cross-entropy (decoder); eval loss | Gloss→English; T5 for seq2seq; generated data to bootstrap.                                                                                           |


---

# Part G: Charts and Confusion Matrix (Stage 1)

- **Charts (presentation_charts.png):**  
  - **Left:** Training loss (blue), validation loss (orange) → epoch convergence.  
  - **Middle:** Validation accuracy (green) → performance.  
  - **Right:** Learning rate over epochs (warm-up + cosine).
- **Confusion matrix (stage1_confusion_matrix.png):** Rows = true, columns = predicted; diagonal = correct; off-diagonal = confusions (e.g. I vs J). Use it to explain precision, recall, F1, accuracy.

Use this document as your single reference for explaining **Extract, Stage 1, Stage 2, and Stage 3**, including the **reasons** behind each design choice and the **mathematics** where the teacher asks for it.