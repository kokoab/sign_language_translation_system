# SLT Architecture & System Analysis Report
**Date:** 2026-03-14
**Role:** Elite AI Architect / ML Engineer
**Focus:** System End-to-End Accuracy, Bottlenecks, and "What-If" Analysis

---

## 1. System Mapping

The current 3-Stage pipeline is an elegant solution to the ASL data scarcity problem, operating through the following data flow:

1.  **Stage 0 (Extraction):** 
    *   **Input:** Raw Video (Variable FPS/Length).
    *   **Process:** MediaPipe Hands -> Gap Interpolation -> Temporal Resampling (32 frames) -> Wrist-Centering -> Bone-Scaling -> Kinematics (Vel/Acc) -> Masking.
    *   **Output:** `[32, 42, 10]` tensor (Float16).
2.  **Stage 1 (Isolated Classification):**
    *   **Input:** `[B, 32, 42, 10]`.
    *   **Process:** DS-GCN (Spatial Graph) + Transformer Encoder (Temporal) -> Classifier.
    *   **Output:** Isolated ASL Gloss (e.g., `HELLO`).
3.  **Stage 2 (Continuous Recognition):**
    *   **Input:** Concatenated 32-frame clips `[B, T_flat, 42, 10]`.
    *   **Process:** Frozen Stage 1 Encoder -> `AdaptiveAvgPool1d(4)` -> BiLSTM -> Linear -> CTC Decoder.
    *   **Output:** Sequence of ASL Glosses (e.g., `[HELLO, HOW, YOU]`).
4.  **Stage 3 (Translation):**
    *   **Input:** Gloss String (e.g., `"HELLO HOW YOU"`).
    *   **Process:** T5-small (Seq2Seq) Fine-tuned on synthetic grammar templates.
    *   **Output:** Natural English Sentence (e.g., `"Hello, how are you?"`).

---

## 2. System Health Check

Before diving into the scenarios, here is a critical diagnostic of your current pipeline based on the provided logs and codebase review:

*   **Critical Domain Gap (Stage 2):** Your Stage 2 model was trained on *synthetic continuous data*—perfectly aligned 32-frame chunks glued together back-to-back. Real continuous video features messy sign boundaries, variable frame lengths per sign, and "transition frames" (hands moving between resting positions and active signing). Because the BiLSTM has never seen a "transition," it is highly prone to hallucinating signs during these real-world interstitial movements.
*   **Missing CTC Language Model:** Currently, your Stage 2 inference uses Greedy Argmax decoding. This means it relies purely on the acoustic/visual model. If it predicts `[HOW, DRINK, YOU]` due to visual noise, T5 will translate that literally. A CTC Beam Search Decoder with an N-gram Language Model (LM) trained on your Stage 3 synthetic sentences would gracefully correct `DRINK` to `ARE` or drop it entirely before T5 even sees it.
*   **Overfitting Risk (Stage 3):** Your T5 model is trained heavily on the `TIME SVO` (Subject-Verb-Object) structure. While rules were patched, T5 is likely to hallucinate an SVO structure if Stage 2 outputs an out-of-distribution sequence.

---

## 3. Scenario Analysis ("What-Ifs")

### What-If 1: Train/Test Distribution Matching (Missing Frames)
**Scenario:** Ensuring live inference gap-interpolation identically matches `extract.py`.
*   **Impact:** **CRITICAL TO FIX.** If this mismatches, accuracy drops catastrophically.
*   **Analysis:** The DS-GCN relies heavily on the `[velocity, acceleration]` channels. If MediaPipe drops a frame in live inference and you feed a zero-vector (`[0,0,0]`), the velocity calculation will see a massive physical spike (teleportation), exploding the GCN's edge weights. In training, you used `scipy.interp1d` to smoothly bridge these gaps, keeping velocities natural.
*   **Implementation Strategy:** Live inference cannot interpolate the *future* if it's strictly real-time. You must introduce a **sliding buffer** (e.g., 48 frames). When the buffer is full, you look at the last 32 frames, run `interp1d` on missing internal frames (clamping the edges), compute the kinematics on the smoothed buffer, and *then* feed the network. A ~1-second latency is a required tradeoff for mathematical stability.

### What-If 2: Variable Temporal Resolution (Removing the 32-Frame Limit)
**Scenario:** Stop forcing 32 frames. Pass raw sequence length directly to Stage 1/2.
*   **Impact:** **WOULD BREAK HARDWARE / HURT ACCURACY IN CURRENT SETUP.**
*   **Analysis:** 
    1.  *Memory Explosion:* The Transformer Encoder's self-attention memory scales quadratically ($O(N^2)$). Moving from 32 frames to ~200 frames increases the attention matrix by a factor of ~39x. On Kaggle's 16GB T4/P100, you would OOM instantly unless you drop batch size to 1-2, destroying gradient stability.
    2.  *Gradient Collapse:* Padding variable sequences requires `pack_padded_sequence` for LSTMs and strict `src_key_padding_mask` for Transformers. The DS-GCN handles variable lengths okay, but temporal convolutions in the GCN blocks might misalign if not carefully padded.
    3.  *Information Dilution:* Resampling to 32 frames actually acts as a form of temporal normalization, making fast and slow signs look identical to the GCN. Removing it forces the model to learn speed invariance from scratch.
*   **Verdict:** Keep the 32-frame fixed length for Stage 1. For continuous video (Stage 2), use an overlapping sliding window (e.g., 32-frame window, 16-frame stride) rather than pushing a raw 200-frame tensor through the encoder at once.

### What-If 3: Signer Speed Augmentation
**Scenario:** Making the model robust to fast vs. slow signers.
*   **Impact:** **HIGHLY BENEFICIAL.** 
*   **Analysis:** Temporal augmentation is crucial. However, you **cannot** simply time-warp the `[32, 42, 10]` tensor. If you warp the velocity/acceleration channels linearly, they lose their physical relationship to the `XYZ` coordinates (e.g., moving twice as fast should double the velocity, not just squish the tensor).
*   **Implementation Strategy:** You must warp the raw `XYZ` positions *before* computing kinematics. 

---

## 4. Top 3 Accuracy Recommendations

Based on the constraints and architecture, here are the three highest-impact changes you can make to significantly improve final BLEU/WER scores without rewriting the architecture.

### Recommendation 1: Transition-Frame Synthesis in Stage 2 (Solves Domain Gap)
**Why:** Your Stage 2 BiLSTM fails on real videos because it never learned what "not signing" looks like. 
**How:** Modify `SyntheticCTCDataset` to randomly inject "transition noise" or blank frames between the 32-frame clips.

### Recommendation 2: Dynamic Time Warping (DTW) Augmentation (Solves Speed Variance)
**Why:** Native signers and beginners move at drastically different speeds.
**How:** Warp the time axis of the raw XYZ coordinates in `extract_augment.py` *before* `compute_kinematics_batch` is called.

### Recommendation 3: Beam Search + CTC Language Model (Fixes Hallucinations)
**Why:** Greedy CTC decoding is blind to grammar. By adding an N-gram LM trained on your Stage 3 synthetic dataset, Stage 2 will mathematically favor valid ASL gloss sequences, correcting minor vision errors before translation.

---

## 5. Implementation Snippets

### Snippet A: Transition-Frame Synthesis (For `train_stage_2.py`)
Inject this into `SyntheticCTCDataset.__getitem__` before concatenation.

```python
import numpy as np
import random

def insert_transitions(clip_list):
    """
    Injects a small random number of 'transition' frames (linear interpolation 
    between the end of Clip A and the start of Clip B) to simulate real signing.
    """
    if len(clip_list) == 1:
        return np.concatenate(clip_list, axis=0)
        
    continuous_sequence = []
    for i in range(len(clip_list) - 1):
        clip_a = clip_list[i]
        clip_b = clip_list[i + 1]
        
        continuous_sequence.append(clip_a)
        
        # 30% chance to simulate a natural transition (3 to 8 frames)
        if random.random() < 0.3:
            trans_len = random.randint(3, 8)
            # Isolate XYZ channels (0:3)
            start_pos = clip_a[-1, :, 0:3]
            end_pos = clip_b[0, :, 0:3]
            
            # Linearly interpolate positions
            alphas = np.linspace(0, 1, trans_len)[:, None, None] # [T, 1, 1]
            trans_pos = (1 - alphas) * start_pos + alphas * end_pos
            
            # Recompute dummy kinematics for transition (approximate to 0)
            trans_kinematics = np.zeros((trans_len, 42, 7), dtype=np.float32)
            trans_full = np.concatenate([trans_pos, trans_kinematics], axis=-1)
            
            continuous_sequence.append(trans_full)
            
    continuous_sequence.append(clip_list[-1])
    return np.concatenate(continuous_sequence, axis=0)
```

### Snippet B: Proper Temporal Speed Augmentation (For `extract_augment.py`)
Apply this to the `[32, 42, 3]` XYZ tensor *before* calling `compute_kinematics_batch`.

```python
import numpy as np
from scipy.interpolate import interp1d

def temporal_time_warp(xyz_sequence, max_warp=0.2):
    """
    Warps the time axis of a [T, 42, 3] position sequence simulating fast/slow signing.
    Must be applied BEFORE kinematics are computed so derivatives remain physically accurate.
    """
    T, V, C = xyz_sequence.shape
    
    # Create original time steps [0, 1, ... 31]
    orig_t = np.arange(T)
    
    # Create warped time steps (anchor endpoints, shift middle randomly)
    warp_factor = np.random.uniform(-max_warp, max_warp)
    
    # A simple quadratic warp curve
    norm_t = orig_t / (T - 1)
    warped_norm_t = norm_t + warp_factor * norm_t * (1 - norm_t)
    warped_t = warped_norm_t * (T - 1)
    
    # Ensure strictly increasing
    warped_t = np.sort(warped_t)
    
    # Reshape for interpolation
    flat_seq = xyz_sequence.reshape(T, -1)
    
    # Interpolate from warped time onto regular time grid
    interpolator = interp1d(warped_t, flat_seq, axis=0, kind='linear', fill_value='extrapolate')
    warped_flat = interpolator(orig_t)
    
    return warped_flat.reshape(T, V, C).astype(np.float32)
```

### Snippet C: Sliding Buffer for Live Inference Match
To match training distribution (What-If 1) during inference.

```python
# Concept for real-time loop
frame_buffer = [] # Store raw MediaPipe XYZ outputs

def process_live_frame(raw_xyz):
    frame_buffer.append(raw_xyz)
    
    # Wait until we have enough frames to safely interpolate
    if len(frame_buffer) == 32:
        # 1. Convert to numpy array
        xyz_arr = np.array(frame_buffer)
        
        # 2. Run your train-time `interpolate_hand` HERE on the 32 frames
        xyz_filled = interpolate_hand(xyz_arr, valid_indices, 32)
        
        # 3. Compute kinematics on the filled array
        final_tensor = compute_kinematics(xyz_filled) # -> [32, 42, 10]
        
        # 4. Pass to Stage 1/2
        # 5. Slide buffer forward (e.g., drop oldest 16 frames for overlap)
        frame_buffer = frame_buffer[16:] 
```