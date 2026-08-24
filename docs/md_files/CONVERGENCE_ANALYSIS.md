# Convergence Analysis

## Stage 1 — Isolated Sign Classification (DS-GCN-TCN v14)

### Training Configuration
- **Architecture**: DSGCNEncoderV14 (4 GCN layers with SE blocks) + Knowledge Distillation from SAM-SLR-v2 ensemble teacher
- **Dataset**: 57,535 samples, 310 ASL classes, 70/15/15 split
- **Epochs**: 200 | **LR**: 2e-4 → 5e-6 (cosine decay) | **Batch**: 256 (accum 4 = effective 1024)
- **Features**: Angle-primary (signer-invariant joint angles + bone features)

### Convergence Behavior

**Phase 1 — Rapid Learning (Epochs 1-30):**
- Val accuracy jumps from 0.62% to ~65% in the first 30 epochs
- The model quickly learns basic hand shape categories (fist vs open hand vs pointing)
- Both CE loss and KD loss contribute — the student learns from both ground truth labels and the SAM teacher's soft predictions
- Precision/Recall/F1 rise sharply from near-zero to ~60%

**Phase 2 — Refinement (Epochs 30-120):**
- Val accuracy gradually climbs from 65% to ~83%
- The model learns finer distinctions: finger configurations, hand orientation, motion patterns
- Knowledge distillation provides "dark knowledge" — the teacher's soft probabilities reveal inter-class similarities (e.g., GOOD and THANKYOU have similar softmax distributions, teaching the student about their relationship)
- Train CE loss shows characteristic instability in this phase — the model oscillates as it discovers new decision boundaries
- Val loss steadily decreases, confirming generalization is improving despite training noise

**Phase 3 — Saturation (Epochs 120-200):**
- Val accuracy plateaus around 84-85%
- Cosine LR decay brings the learning rate to minimum (5e-6)
- The model has learned the representational capacity of the angle-primary feature space
- Final push from epoch 150-167 yields the peak val accuracy of 85.32%
- After epoch 167, no further improvement — the generalization ceiling is reached

### Final Metrics
| Metric | Val | 
|--------|-----|
| Accuracy | 85.32% |
| Top-5 Accuracy | 93.21% |
| Precision | 85.73% |
| Recall | 85.25% |
| F1 Score | 85.22% |

### Understanding Precision, Recall, and F1

These three metrics measure different aspects of classification quality:

- **Precision (85.73%)**: Of all the times the model predicted a specific sign (e.g., "HELLO"), how often was it actually correct? High precision means few false positives — the model rarely says "HELLO" when it's actually "GOODBYE."

- **Recall (85.25%)**: Of all the actual instances of a sign in the dataset, how many did the model correctly identify? High recall means few false negatives — the model rarely misses a "HELLO" and calls it something else.

- **F1 Score (85.22%)**: The harmonic mean of precision and recall. It balances both metrics — a model that has high precision but low recall (or vice versa) will have a lower F1. Our F1 of 85.22% confirms the model is both precise and comprehensive.

In our case, all three metrics are nearly identical (~85%), which indicates **balanced classification** — the model performs equally well across all 310 classes. If precision were much higher than recall, it would mean the model is conservative (only predicts when very sure). If recall were higher than precision, it would mean the model is aggressive (predicts too often). The balance shows healthy learning.

For comparison:
- A random classifier on 310 classes would achieve ~0.32% on all metrics
- A majority-class classifier would get high recall on "I" (the most common class) but near-zero on everything else
- Our 85% across the board means the model learned meaningful representations for all signs, including rare ones like TEACHER (79 samples)

### Key Observations
- **Knowledge Distillation Effect**: The KD loss (train_kd) remains active throughout training, providing consistent guidance from the teacher model. The student achieves 85.32% compared to the teacher's 91.05% — a 6% gap that reflects the architectural capacity difference between DS-GCN-TCN and the larger SAM-SLR-v2 ensemble
- **Precision ≈ Recall ≈ F1**: All three metrics track closely (~85%), indicating balanced classification across all 310 classes with no systematic bias toward over-predicting or under-predicting any class
- **Top-5 at 93.21%**: The correct sign is almost always in the top 5 predictions, which is useful for downstream CTC decoding that considers multiple hypotheses

### Optimizer and Learning Rate Choices

**Why AdamW?**
AdamW (Adam with decoupled weight decay) was chosen over other optimizers for several reasons:

1. **Adaptive learning rates per parameter**: Unlike SGD which uses a single learning rate for all parameters, Adam maintains separate learning rates for each parameter based on gradient history. This is critical for our architecture because the GCN layers, attention layers, and classification head all have very different gradient magnitudes — GCN layers process sparse graph signals while attention layers process dense temporal sequences. Adam adapts to each.

2. **Decoupled weight decay**: Standard Adam applies weight decay through the gradient (L2 regularization), which interacts poorly with the adaptive learning rate. AdamW decouples weight decay from the gradient update, applying it directly to the weights. This provides more consistent regularization — important when training 310 classes with imbalanced data (79 to 487 samples per class).

3. **Stable with mixed precision**: Our training uses AMP (automatic mixed precision) for speed. AdamW's gradient scaling is more stable than SGD under FP16 gradients, reducing the risk of loss scaling issues.

**Why LR = 2e-4 with cosine decay?**

1. **Initial LR = 2e-4**: This is in the "sweet spot" for Adam-family optimizers on transformer architectures. Too high (>1e-3) causes training instability — the model oscillates and fails to converge. Too low (<1e-5) causes extremely slow convergence, wasting compute. The 2e-4 value was validated empirically: at 7e-4 with warm restarts, training showed instability; at 3e-4 without restarts, the model underfit. The 2e-4 with smooth cosine decay provided the best stability-speed tradeoff.

2. **Cosine decay schedule**: The learning rate follows a cosine curve from 2e-4 down to 5e-6 over 200 epochs. This provides:
   - **Fast early learning**: high LR in the first 50 epochs allows rapid exploration of the loss landscape
   - **Gradual refinement**: smoothly decreasing LR allows the model to settle into sharper minima
   - **No sudden drops**: unlike step-decay schedules (which cause training shocks), cosine decay is continuous and smooth
   - **Natural annealing**: the cosine shape spends more time at intermediate LR values where most learning happens, and less time at extreme values

3. **Effective batch size = 1024** (256 × 4 accumulation steps): Large batches provide more stable gradient estimates, which is important for:
   - 310-class classification (each batch needs to see diverse classes)
   - Knowledge distillation (teacher predictions are noisy, averaging over more samples reduces noise)
   - The large batch size justified the moderate learning rate — larger batches can tolerate slightly higher LR

---

## Stage 2 — Continuous Sign Recognition (CTC)

### Training Configuration
- **Architecture**: DSGCNEncoderV14 (frozen from Stage 1, unfrozen at epoch 30) + MultiScaleTCN (3-branch conv, out_tokens=4) + SequenceTransformer (4 layers, d_model=384) + CTC head (311 vocab including blank)
- **Dataset**: 10,000 synthetic continuous sequences (1-8 signs concatenated from isolated .npy) + 546 real phrase videos (3x upsampled) = 11,638 total training samples
- **Epochs**: 60 | **LR**: 5e-4 → 5e-6 (cosine warmup + cosine decay) | **Batch**: 16
- **Encoder LR**: 0.1x main LR after unfreeze at epoch 30

### Convergence Behavior

**Phase 1 — CTC Warmup (Epochs 1-5):**
- WER drops from 100% to ~38% in just 3 epochs
- This rapid initial convergence demonstrates the value of transfer learning — the pre-trained Stage 1 encoder already produces meaningful sign representations
- The TCN and SequenceTransformer start from random initialization but quickly learn to interpret the encoder's fixed features
- CTC loss drops from 15.2 to ~3.2 as the model learns basic temporal alignment (when to emit signs vs blank tokens)

**Phase 2 — Frozen Encoder Optimization (Epochs 5-30):**
- WER gradually decreases from 38% to ~14%
- The encoder remains frozen — only the MultiScaleTCN, SequenceTransformer, and CTC head parameters update
- This constraint forces the temporal layers to develop robust sequencing capabilities using the fixed sign representations
- The MultiScaleTCN learns to compress 32-frame clips into 4 informative tokens using its 3 parallel convolution branches (kernel sizes 3, 5, 9)
- The SequenceTransformer learns cross-clip attention patterns — how signs relate to each other in sequence
- Sequence accuracy climbs from ~10% to ~58%
- Val loss plateaus around 0.40-0.50, indicating the temporal layers are approaching their capacity with fixed encoder features

**Phase 3 — Encoder Fine-tuning (Epochs 30-60):**
- Encoder unfreezes at epoch 30 with 0.1x learning rate (differential LR)
- WER drops from ~14% to 10.96% — a further 22% relative improvement
- The encoder adapts its internal representations for continuous recognition — features that were optimized for isolated classification shift toward features that are more useful for CTC temporal alignment
- Training loss drops sharply as the entire model can now optimize end-to-end
- Val loss decreases to ~0.35 and stabilizes, showing the fine-tuning does not cause overfitting
- Sequence accuracy reaches 68%

### Final Metrics
| Metric | Value |
|--------|-------|
| Best WER | 10.96% |
| Word Accuracy | 89.04% |
| Sequence Accuracy | 68.0% |

### Training Strategy Rationale

**Why Freeze-then-Unfreeze?**
The two-phase training is critical for CTC convergence:

1. **Frozen phase (epochs 1-30)**: The CTC head, TCN, and SequenceTransformer all initialize randomly. Their gradients in early training are noisy and large. If these noisy gradients flowed back into the pre-trained encoder, they would corrupt the sign recognition knowledge learned from 57,535 isolated samples. Freezing prevents this catastrophic forgetting.

2. **Unfrozen phase (epochs 30-60)**: After the temporal layers have converged to a stable baseline, the encoder can safely receive gradients. The 0.1x learning rate ensures the encoder makes small, careful adjustments rather than large destructive updates. The encoder learns to produce representations that are specifically useful for CTC alignment — for example, making sign boundaries more distinct in the temporal feature space.

**Why Synthetic + Real Phrase Data?**
- Synthetic data (10,000 sequences): created by concatenating isolated .npy files. Provides diverse sign combinations but lacks natural coarticulation (transition movements between signs)
- Real phrase data (546 × 3 = 1,638): recorded continuous signing of 9 phrases. Contains natural coarticulation that only occurs in real continuous signing
- The 3x upsampling of real phrases ensures the model sees enough examples of natural transitions
- Without real phrase data, the model would only learn "clean" transitions; with it, the model learns to handle the messy reality of continuous signing

**CTC Blank Token Dynamics:**
- Early training: the model outputs almost entirely blank tokens (the safe default — predicting nothing is better than predicting wrong)
- As training progresses, the model learns to "fire" non-blank tokens at the correct temporal positions
- The monotonic alignment constraint of CTC means the model must predict signs in order without explicit alignment labels
- This is why WER drops rapidly early (learning when NOT to predict) then slowly later (learning exactly WHEN to predict each sign)

---

## Summary

| | Stage 1 (v14) | Stage 2 (CTC) |
|---|---|---|
| Task | Isolated sign → 310 classes | Continuous signing → gloss sequences |
| Input | [32, 61, 16] single clip | [N×32, 61, 16] variable clips |
| Epochs | 200 | 60 |
| Key Metric | Val Accuracy: 85.32% | WER: 10.96% |
| Convergence Speed | 30 epochs to 65% | 3 epochs to 38% WER (transfer) |
| Training Strategy | KD from SAM teacher | Freeze-then-unfreeze encoder |
| Final Architecture | 4 GCN + SE + angle features | Stage 1 encoder + TCN + Transformer + CTC |
