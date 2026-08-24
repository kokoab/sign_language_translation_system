# SLT Pipeline Optimization Plan

## Current State Analysis

### Training Chart Diagnosis (history_joint_chart.png)

The loss curve shows **three critical problems**:

1. **Extreme train loss spikes** — The raw train loss (blue) oscillates wildly between 0 and 5+ throughout all 300 epochs. This indicates training instability caused by:
   - Cosine warm restarts (T_0=50, T_mult=2) repeatedly spiking the LR back to 7e-4
   - Aggressive augmentation (mixup + cutmix + speed warp + rotation + noise + temporal masking) all applied simultaneously
   - Focal loss with gamma=1.0 amplifying gradient noise

2. **Val accuracy plateaus at ~85.5%** — The accuracy curve shows minimal improvement after epoch 120. The model converges early but warm restarts keep destabilizing it. Compare with "Model 15" which shows smooth monotonic convergence.

3. **Train-val loss gap** — Train loss (regularized) is consistently lower than val loss, suggesting the augmentation is too aggressive and the model can't generalize what it learns.

### Dataset Analysis

```
Classes:              310
Total samples:        57,535
Min samples/class:    79  (TEACHER)
Max samples/class:    487 (I)
Median:               180
Mean:                 185.6
Classes < 100:        8
Imbalance ratio:      6.2x (487/79)
```

The dataset is moderately imbalanced. The 8 classes with < 100 samples are underrepresented, which hurts their accuracy and drags down the overall score.

### Current Hyperparameters (Problems Identified)

| Parameter | Current | Problem |
|-----------|---------|---------|
| LR Schedule | CosineWarmup with restarts (T_0=50, T_mult=2) | Restarts spike LR back to 7e-4 every 50/100/200 epochs, destabilizing training |
| Learning Rate | 7e-4 | Too high for stable convergence — causes the loss spikes |
| Focal Loss gamma | 1.0 | Combined with label smoothing (0.1), creates conflicting objectives |
| Mixup alpha | 0.15 | Fine alone, but stacked with cutmix (50%) is too aggressive |
| CutMix prob | 0.5 | 50% of batches get temporal cutmix — too frequent |
| Label Smoothing | 0.10 | Combined with focal loss, prevents confident predictions |
| Head Dropout | 0.30 | Too low for 310 classes with class imbalance |
| val_every | 3 | Misses best checkpoint 2/3 of the time |
| EMA | Already used | Good — but may not be optimal decay rate |

---

## Optimization Plan

### Phase 1: Training Stability Fixes (Highest Impact — Expected +5-8%)

**1.1 Replace LR Schedule — Use Cosine Annealing WITHOUT Restarts**

The warm restarts are the #1 cause of loss spikes. Every restart resets LR to 7e-4, which is too high for an already-converged model. This is why Model 15's smooth curve looks better.

```python
# BEFORE (causes spikes):
CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=300, T_0=50, T_mult=2)

# AFTER (smooth convergence):
# Use simple cosine annealing: warmup 10 epochs, then smooth decay to near-zero
# No restarts. Single smooth curve like Model 15.
warmup_epochs = 10
scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
```

**Why:** Research shows cosine annealing without restarts gives smoother convergence and better final accuracy for GCN-based models. The restarts help escape local minima in early training but hurt in later stages when the model needs fine-grained optimization.

**1.2 Lower Peak Learning Rate**

```python
# BEFORE:
lr = 7e-4

# AFTER:
lr = 3e-4  # More stable, fewer spikes
```

**Why:** The loss spikes correlate directly with LR magnitude. At 7e-4, gradient updates are too large for the model's current loss landscape after epoch 50.

**1.3 Remove Focal Loss — Use Plain Cross-Entropy + Reduced Label Smoothing**

Focal loss with gamma=1.0 combined with label smoothing creates conflicting training signals. Focal loss says "focus on hard examples" while label smoothing says "don't be too confident about anything." Research (ICLR 2024) shows this combination degrades selective classification and can hurt overall accuracy.

```python
# BEFORE:
focal_gamma = 1.0
label_smoothing = 0.10

# AFTER:
focal_gamma = 0.0  # Disable focal loss — use standard CE
label_smoothing = 0.05  # Reduce smoothing (0.1 is too much for 310 classes)
```

**Why:** The WLASL/AUTSL SOTA methods (SML, SignFormer-GCN) use standard cross-entropy, not focal loss. Label smoothing at 0.05 provides mild regularization without preventing confident predictions.

**1.4 Reduce Augmentation Intensity**

Too many augmentations stacked together prevent the model from learning stable representations. Each augmentation adds noise; 6+ augmentations compound the noise exponentially. The model sees almost no clean data.

```python
# BEFORE (all applied every batch):
mixup_alpha = 0.15
cutmix_prob = 0.5        # 50% of batches — too aggressive
speed_warp_prob = 0.5    # 50% of batches
temporal_mask_prob = 0.3
signer_norm_prob = 0.3
rotation_deg = 10.0
noise_std = 0.003

# AFTER (gentler, fewer simultaneous):
mixup_alpha = 0.1
cutmix_prob = 0.15       # Reduced from 50% to 15%
speed_warp_prob = 0.3    # Reduced from 50% to 30%
temporal_mask_prob = 0.15 # Reduced from 30% to 15%
signer_norm_prob = 0.2   # Reduced from 30% to 20%
rotation_deg = 8.0       # Slightly reduced
noise_std = 0.002        # Slightly reduced
```

**Why:** The preprocessing paper (ACL SignLang 2024) shows that normalization alone gives 6-13% improvement. Over-augmentation fights against normalization benefits. The SML method (SOTA on WLASL) uses simple augmentations only.

---

### Phase 2: Architecture & Regularization (Expected +2-3%)

**2.1 Increase Head Dropout**

With 310 classes and some having only 79 samples, the classifier head needs stronger regularization to prevent overfitting to majority classes.

```python
# BEFORE:
head_dropout = 0.30

# AFTER:
head_dropout = 0.45
```

**2.2 Increase DropPath (Stochastic Depth)**

```python
# BEFORE:
drop_path_rate = 0.1

# AFTER:
drop_path_rate = 0.15
```

**2.3 Better EMA Decay**

The current EMA is good but check the decay rate. Standard is 0.999 for large datasets, 0.9999 for smaller ones.

```python
# Verify current EMA decay and adjust if needed:
ema_decay = 0.9995  # For 57k samples this is appropriate
```

---

### Phase 3: Training Strategy (Expected +1-2%)

**3.1 Validate Every Epoch**

Current `val_every=3` means you miss the best checkpoint 2/3 of the time. The best model might occur at epoch 147 but you only check at 147→skipped, 148→skipped, 149→checked (but model already degraded).

```python
# BEFORE:
val_every = 3

# AFTER:
val_every = 1
```

**3.2 Shorter Training with True Early Stopping**

300 epochs with warm restarts wastes compute. The model converges by epoch 120. With smooth cosine annealing, 150 epochs is sufficient.

```python
# BEFORE:
epochs = 300
patience = 50

# AFTER:
epochs = 150
patience = 25
```

**3.3 Larger Effective Batch Size**

Larger effective batch size = smoother gradients = fewer loss spikes.

```python
# BEFORE:
batch_size = 256, accum_steps = 2  # effective = 512

# AFTER:
batch_size = 256, accum_steps = 4  # effective = 1024
```

---

### Phase 4: Data Quality Improvements (Expected +2-4%)

**4.1 Address Class Imbalance**

8 classes have < 100 samples vs 487 for the most common. The weighted sampler helps but isn't enough.

Options (pick one):
- **Balanced sampling with sqrt weighting:** `temperature = 0.3` instead of 0.5 (gentler rebalancing)
- **Class-balanced loss:** Use the class-balanced focal loss from Cui et al. 2019 (not regular focal loss)
- **Oversampling minority classes** with augmented variants

**4.2 Data Cleaning Audit**

Some .npy files may have low-quality landmarks (partial hand detection, wrong hand assignment). Run a quality audit:
- Check for files where only 1 hand is detected but the sign requires 2
- Check for files with very low motion (might be misclassified or static frames)
- Check for label noise by running the trained model on training data and flagging confident mismatches

**4.3 Anchor-Point Normalization**

Research shows normalizing landmarks relative to body anchor points (wrist, shoulder) improves accuracy by 6-13%. Current normalization uses wrist + bone length, which is good but could be enhanced with:
- **Per-frame re-centering** to the midpoint between wrists
- **Hand-size invariant features** (ratios instead of absolute distances)

---

### Phase 5: Inference-Time Boosts (No Retraining Needed)

**5.1 Ensemble 4 Streams (Expected +4-6%)**

You already have 4 trained stream models:
- `stage1_joint.pth` (85.5%)
- `stage1_bone.pth`
- `stage1_velocity.pth`
- `stage1_angle.pth`

Average their softmax probabilities at inference:

```python
streams = ['joint', 'bone', 'velocity', 'angle']
all_probs = []
for stream in streams:
    model = load_model(stream)
    probs = torch.softmax(model(x), dim=-1)
    all_probs.append(probs)
ensemble_probs = torch.stack(all_probs).mean(dim=0)
prediction = ensemble_probs.argmax()
```

**Why this works:** Each stream sees different features. Joint sees positions, bone sees directions, velocity sees motion, angle sees hand shape. Errors in one stream are corrected by others.

Research: "One Model is Not Enough: Ensembles for Isolated Sign Language Recognition" (MDPI 2022) shows ensemble of 4 skeleton streams consistently adds 4-6%.

**5.2 Test-Time Augmentation (Expected +1-2%)**

Run inference on original + mirrored (left-right hand swap), average predictions:

```python
# Original prediction
probs_orig = softmax(model(x))

# Mirrored prediction (swap left/right hand landmarks)
x_mirror = x.clone()
x_mirror[:, :, 0:21] = x[:, :, 21:42]  # left ← right
x_mirror[:, :, 21:42] = x[:, :, 0:21]  # right ← left
x_mirror[:, :, :42, 0] *= -1  # flip X coordinate
probs_mirror = softmax(model(x_mirror))

final_probs = (probs_orig + probs_mirror) / 2
```

**5.3 Confidence Thresholding**

Don't output predictions below 20% confidence. Better to say "I didn't catch that" than give a wrong answer.

---

### Phase 6: Match Train/Inference Extractor (Critical for Real-World Use)

**6.1 Re-extract Training Data with rtmlib**

The #1 reason video inference fails is the domain gap between mmpose (training) and rtmlib (inference). Fix by re-extracting all 57k training videos with rtmlib.

**Time estimate:** ~48 hours in Docker on Mac, or ~8 hours on Codespace with native x86.

**6.2 Retrain Stage 1 + Stage 2**

After re-extraction, retrain with the optimized hyperparameters from Phase 1-3.

**Time estimate:** ~3.5 hours on a T4 GPU.

---

### Phase 7: Advanced Techniques (If Time Allows)

**7.1 Self-Knowledge Distillation (SKD)**

Train a teacher model (the current 85.5% model), then train a student model that learns from both the ground truth labels AND the teacher's soft predictions. This consistently adds 1-3% in skeleton-based SLR (SML paper, 2024).

**7.2 Multi-View Contrastive Learning**

Create positive pairs from augmented versions of the same sign, negative pairs from different signs. Adds a contrastive loss alongside classification loss. SC2SLR (2024) shows 2-3% improvement.

**7.3 Decoupled Graph Convolution**

Replace the standard GCN adjacency matrix with separate spatial and temporal graphs. The current DS-GCN already does partial decoupling, but full decoupling (DGCN from SML) could improve spatial feature extraction.

**7.4 Adaptive Graph Topology**

Make the adjacency matrix fully learnable instead of partially learnable (current: fixed + small learnable residual). SignFormer-GCN (2025) shows this helps capture sign-specific spatial relationships.

---

## Expected Accuracy Summary

| Phase | Change | Expected Gain | Cumulative |
|-------|--------|--------------|------------|
| Current | Baseline | - | 85.5% |
| Phase 1 | LR fix + remove focal + reduce aug | +5-8% | 90-93% |
| Phase 2 | Head dropout + droppath | +1-2% | 91-95% |
| Phase 3 | val_every=1 + larger batch | +1% | 92-96% |
| Phase 5.1 | Ensemble 4 streams | +4-6% | 94-97% |
| Phase 5.2 | Test-time augmentation | +1% | 95-97% |
| Phase 6 | rtmlib retrain | +15-20% on video | 80%+ on real video |

**Conservative estimate with Phases 1-3 + 5: 92-95%**
**With ensemble: 95-97%**
**On real video after rtmlib retrain: 80%+**

---

## Training Time Estimate

| What | Time | GPU |
|------|------|-----|
| Stage 1 retrain (150 epochs) | ~2 hours | T4 |
| Stage 2 retrain (60 epochs) | ~1.5 hours | T4 |
| Stage 3 | No retraining | - |
| Re-extraction with rtmlib | ~48 hours Mac Docker, ~8 hours Codespace | CPU |
| **Total retrain** | **~3.5 hours** | T4 |

---

## Priority Order (What To Do First)

1. **Ensemble inference** — immediate, no retraining needed, +4-6% (hour 1)
2. **Fix hyperparameters** (LR, focal loss, augmentation, val_every) — retrain needed (hour 2-4)
3. **Re-extract with rtmlib** — overnight batch (overnight)
4. **Retrain on rtmlib data** — 3.5 hours (next day)
5. **Record 180 continuous signing videos** — manual effort (day 3)
6. **Retrain Stage 2 with real videos** — 1.5 hours (day 3)
7. **Advanced techniques (SKD, contrastive)** — if time allows (day 4+)

---

## References

- [SML: Skeleton multi-feature learning with self-distillation, SOTA on WLASL/AUTSL](https://www.sciencedirect.com/science/article/abs/pii/S0950705124009225)
- [SignFormer-GCN: Spatio-temporal GCN with learnable graphs](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316298)
- [SC2SLR: Skeleton contrastive learning for SLR](https://dl.acm.org/doi/10.1145/3670105.3670173)
- [Ensemble SLR: One Model is Not Enough](https://pmc.ncbi.nlm.nih.gov/articles/PMC9269724/)
- [Hybrid Transformer achieving 99.97% on ASL Alphabet](https://www.nature.com/articles/s41598-025-06344-8)
- [TinyMSLR: Knowledge distillation, 99.75% on WLASL](https://www.nature.com/articles/s41598-026-38478-8)
- [MSKA: Multi-stream keypoint attention](https://www.sciencedirect.com/science/article/abs/pii/S0031320325002626)
- [Preprocessing keypoints: 6-13% accuracy improvement](https://aclanthology.org/2024.signlang-1.36.pdf)
- [Loss spikes in training neural networks](https://arxiv.org/pdf/2305.12133)
- [Label smoothing degrades selective classification](https://arxiv.org/html/2403.14715v1)
- [EMA improves generalization and robustness](https://arxiv.org/html/2411.18704v1)
- [Super-convergence with 1-cycle policy](https://arxiv.org/abs/1708.07120)
- [Ensemble transformer: 93.51% on WLASL](https://www.sciencedirect.com/science/article/pii/S2307187725000951)
- [rtmlib — RTMW without mmcv](https://github.com/Tau-J/rtmlib)

---

## Appendix A: Underrepresented Classes (< 100 samples)

These 8 classes need more data or stronger augmentation:

| Class | Samples | Action |
|-------|---------|--------|
| TEACHER | 79 | Record 30+ more videos |
| LOUD | 90 | Record 20+ more videos |
| WORKER | 94 | Record 15+ more videos |
| TEAM | 95 | Record 15+ more videos |
| CODE | 96 | Record 15+ more videos |
| SAD | 98 | Record 10+ more videos |
| AT | 99 | Record 10+ more videos |
| STRONG | 99 | Record 10+ more videos |

Minimum target: 120 samples per class (matches the 25th percentile).

---

## Appendix B: Recommended Continuous Signing Phrases

If adding 180 real continuous signing videos (60 per phrase), choose phrases that:
- Use signs from the existing 310-class vocabulary
- Cover common conversational patterns
- Include both underrepresented and common signs
- Vary in length (2-4 signs)

### Recommended 3 Phrases (60 videos each):

**Phrase 1: "HELLO HOW YOU" (greeting)**
- Uses: HELLO, HOW, YOU (all common signs, well-represented)
- Why: Most natural ASL greeting, essential for any demo
- Variations: different speeds, different signers, different backgrounds

**Phrase 2: "PLEASE HELP ME" (request)**
- Uses: PLEASE, HELP, I/ME (practical, high-frequency)
- Why: Demonstrates conversational use case, includes subject pronoun
- Variations: urgent vs. casual signing style

**Phrase 3: "SORRY I LATE" (apology)**
- Uses: SORRY, I, LATE (mixes common + less common signs)
- Why: Natural phrase, tests temporal sign transitions, SORRY→I has interesting hand transition
- Variations: different levels of emphasis

### Alternative Phrases (if you want more):

| Phrase | Signs | Why |
|--------|-------|-----|
| GOOD MORNING | 2 signs | Short, common greeting |
| MY NAME [fingerspell] | 2 signs + fingerspelling | Introduction pattern |
| THANKYOU FRIEND | 2 signs | Gratitude expression |
| I WANT FOOD | 3 signs | Basic need expression |
| TOMORROW SCHOOL GO | 3 signs | Future tense + location + verb |
| YESTERDAY TEACHER MEET | 3 signs | Past tense, uses underrepresented TEACHER |

### Recording Guidelines:

1. Sign naturally at conversational speed (not sign-by-sign)
2. Include the transition between signs (coarticulation)
3. Face the camera directly, good lighting
4. Record each phrase 60 times with slight variation:
   - 20x normal speed
   - 20x slightly faster
   - 20x slightly slower
5. If possible, have 2-3 different people record (diversity helps generalization)
6. Keep videos 1-3 seconds long
7. Label with the gloss sequence: "HELLO HOW YOU"

---

## Appendix C: Simplicity Check

**Things NOT to do (would add complexity without proportional accuracy gain):**

- Do NOT switch to a different pose estimator (HaMeR, ViTPose, etc.) — rtmlib RTMW is already SOTA quality
- Do NOT change the model architecture significantly — DS-GCN + Transformer is proven
- Do NOT add RGB features — skeleton-only is lighter and still competitive
- Do NOT add more transformer layers — 4-6 is already optimal for this dataset size
- Do NOT use contrastive learning unless Phase 1-5 gains are insufficient

**Things TO do (simple, high-impact):**

- Fix the LR schedule (1 line change)
- Remove focal loss (1 line change)
- Reduce augmentation (parameter changes only)
- Ensemble 4 existing streams (inference-only, no retraining)
- Re-extract with rtmlib + retrain (same code, same architecture, just new data)
- Record 180 continuous videos (manual effort, highest impact for Stage 2)
