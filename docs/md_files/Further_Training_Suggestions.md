# Further Training & Architecture Suggestions

Beyond the changes already implemented (merged classes, 47-node face-aware landmarks, focal loss, label smoothing, disambiguation data), here are additional improvements to consider after re-extraction and initial retraining.

---

## Stage 1 — After Retraining on New Data

### 1. Confusion-Pair Contrastive Loss (Advanced)
After retraining, generate a new confusion matrix. For remaining confused pairs (e.g., AT/HE, FATHER/MOTHER if face landmarks don't fully resolve them), add an auxiliary contrastive loss that pushes apart embeddings of these specific classes:

```python
# In the training loop, after the main loss:
confusion_pairs = [(idx_AT, idx_HE), (idx_FATHER, idx_MOTHER)]
contrastive_weight = 0.1
for idx_a, idx_b in confusion_pairs:
    mask_a = (y == idx_a)
    mask_b = (y == idx_b)
    if mask_a.any() and mask_b.any():
        emb_a = encoder_output[mask_a].mean(dim=0)
        emb_b = encoder_output[mask_b].mean(dim=0)
        dist = F.pairwise_distance(emb_a.unsqueeze(0), emb_b.unsqueeze(0))
        contrastive_loss = torch.clamp(1.0 - dist, min=0)
        loss += contrastive_weight * contrastive_loss
```

### 2. Curriculum Phase 3: Confusion-Pair Focus
After the current single-hand -> two-hand curriculum, add a third phase that oversamples the remaining most-confused pairs. Use the validation confusion matrix to identify them automatically.

### 3. Test-Time Augmentation (TTA)
During validation or inference, run each sample through the model 3x with slight augmentations (small rotation, scale jitter) and average the logits. This can boost accuracy 1-2% with no retraining.

---

## Stage 2 — CTC Improvements

### 4. Increase Synthetic Dataset Size
Current: 8000 train / 1000 val. With ~70k videos across 300+ classes, increase to 15000-20000 train sequences to better cover the vocabulary space. Especially important now that merged tokens like DRIVE_CAR appear.

### 5. Weighted Sequence Sampling
Weight the synthetic sequence generation to produce more sequences containing glosses that Stage 1 frequently confuses. This gives Stage 2 more practice with these challenging transitions.

### 6. Longer Training
Stage 2 currently trains for 50 epochs with patience 15. Consider increasing to 100 epochs with patience 25, since CTC convergence can be slow.

---

## Stage 3 — Translation Quality

### 7. Increase Disambiguation Dataset Weight
If the model still confuses verb/noun senses of merged tokens (e.g., "I'm driving" vs "the car" for DRIVE_CAR), increase the repetition multiplier for disambiguation data from 80x to 120x.

### 8. Context Window Tuning
The current context window passes the last 2 dialogue turns. Experiment with 3-4 turns for better disambiguation, but watch for input length overflow (MAX_INPUT_LENGTH = 96).

---

## Data Collection

### 9. Targeted Recording for Weak Classes
After retraining, identify classes with < 80% accuracy. Record 30-50 additional videos for each, focusing on:
- Different signers
- Different lighting conditions
- Different camera angles

### 10. Face Landmark Quality Check
After re-extraction with FaceMesh, check that face landmarks are detected in at least 60% of frames. Classes where face detection is consistently low may need re-recording with the signer facing the camera more directly.

---

## Inference Optimization

### 11. FaceMesh Overhead
FaceMesh adds ~3ms per frame on M4. If this is too slow for real-time, consider:
- Running FaceMesh every 3rd frame and interpolating
- Using a lighter face detector that only returns 5 keypoints

### 12. Merged Token Post-Processing
In `camera_inference.py`, add a post-processing step that maps merged tokens to the most likely English word based on surrounding context before passing to Stage 3. This can serve as a fallback if Stage 3 is slow.
