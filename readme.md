    # Sign Language Translator (SLT) – Transformer + GCN Hybrid
**Documentation – 001**  
**Date:** February 17, 2026  
**Status:** Optimization & Feature Refinement Phase

---

## 1. Brief Description

This project focuses on **real-time American Sign Language (ASL) alphabet recognition** using a hybrid deep-learning architecture. The system combines:

- **MediaPipe** – Hand landmark extraction  
- **Graph Convolutional Network (GCN)** – Spatial relationship modeling between hand joints  
- **Transformer Encoder** – Temporal sequence modeling across video frames  

**Goal:** Low-latency, high-accuracy inference suitable for edge devices.

---

## 2. Key Transitions & Refinements

### A. Sequence Length & Transformer Weight

**Reason for Change:**  
The original **60-frame window (~2 seconds)** caused noticeable input lag and high computational cost. The Transformer was also over-parameterized for real-time deployment.

#### Before (60 Frames)
```python
self.pos_encoder = nn.Parameter(torch.randn(1, 60, self.d_model))
# Heavy Transformer feedforward
dim_feedforward = 2048
num_layers = 6
```

#### After (30 Frames)
```python
self.pos_encoder = nn.Parameter(torch.randn(1, 30, self.d_model))
# Light Transformer feedforward
dim_feedforward = 1024
num_layers = 4
```

**Remarks:**  
Cutting the sequence length in half reduced the computational cost of self-attention by approximately **4×**, since attention complexity is **O(n²)**.

---

### B. Bone-Length Normalization

**Reason for Change:**  
Global normalization failed when the user moved farther from the camera. Hand size variations altered joint distances and confused the model.

#### Before (Global Max)
```python
norm = np.max(np.linalg.norm(input_data, axis=2))
input_data /= norm
```

#### After (Reference Bone Scaling)
```python
# Scale by distance from Wrist (0) to Middle Finger MCP (9)
ref_dist = np.mean(np.linalg.norm(input_data[:, 9, :], axis=-1))
if ref_dist > 0:
    input_data /= ref_dist
```

**Remarks:**  
Using a **reference bone** keeps perceived hand size consistent regardless of webcam distance.

---

### C. Resolving R / V / K & S / T / M / N Confusion

**Reason for Change:**  
Several ASL letters share very similar global 3D coordinates. Standard GCN layers struggled to distinguish small inter-finger gaps.

#### Before (Simple Distances)
```python
f1 = dist(data[:, :, 4], data[:, :, 8])  # Thumb–Index only
```

#### After (Enhanced Geometric Features)
```python
# Specific finger-to-finger checks
f2 = dist(data[:, :, 8], data[:, :, 12])  # Index–Middle (V vs R)
f5 = dist(data[:, :, 4], data[:, :, 12])  # Thumb–Middle (K)
f7 = dist(data[:, :, 4], data[:, :, 20])  # Thumb–Pinky (S)
```

**Remarks:**  
Adding **targeted geometric features** forces the model to focus on precise inter-finger gaps that define visually similar letters.

---

### D. Stability Logic & Variable Scoping

**Reason for Change:**  
Two inference-loop issues were identified:

- **NameError:** Variables were defined inside conditionals, causing crashes when no hand was detected.  
- **Prediction Flicker:** Rapid label switching prevented stable sentence formation.

#### Before (Unstable)
```python
if len(sequence) == WINDOW_SIZE:
    current_char = labels[idx]  # Variable created here
# Crash occurs here if hand is missing
```

#### After (Stability Buffer)
```python
current_char = "Waiting..."  # Initialization

stability_buffer.append(current_char)
if all(x == current_char for x in stability_buffer[-3:]):
    sentence.append(current_char)  # Confirmed only after 3 frames
```

**Remarks:**  
A **3-frame confirmation buffer (~0.09s)** provides a debouncing effect similar to a physical keyboard.

---

## 3. Errors Encountered & Fixes

| Error | Cause | Fix |
|------|------|-----|
| `NameError: name 'current_conf' is not defined` | Variable initialized inside a conditional block | Initialize variables at the start of the `while` loop |
| Inconsistent Shape Error | Training used 60 frames while testing used 30 | Synchronize `pos_encoder` and `WINDOW_SIZE` to 30 |
| R / U Confusion | Crossing fingers resembled joined fingers mathematically | Added Index–Middle distance (`f2`) geometric feature |

---

## 4. Final Remarks

- **Current Training Accuracy:** 99.9%  
- **Standard Sequence Window:** 30 Frames  
- **Immediate Next Step:** Dataset diversification (different hand sizes, lighting conditions, and camera distances) to maintain cross-user accuracy and real-world robustness.

---
