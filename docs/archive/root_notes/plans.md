# SLT Training Execution Plan — 315-Class Pipeline

## Current State

### Dataset
- **57,561 .npy files** in `ASL_landmarks_float16/`, each `[32, 47, 10]` float16
- **315 unique classes** (26 alphabet + 289 ASL signs)
- **~55,506 trainable** after excluding spatial_outlier (1,479) + no_motion (576)
- Quality: bone_cv=0.002 (excellent), coverage=100%, jitter is positional (RTMDet bbox), not structural
- 1.8 GB total on disk

### Class Distribution
| Bucket | Classes | Notes |
|--------|---------|-------|
| 100-500 samples | 303 | Healthy range |
| 50-100 samples | 8 | Acceptable |
| 10-50 samples | 1 | RIGHT (22) — borderline |
| <10 samples | 3 | HIS (4), EXCUSE (4), HER (7) — MUST FIX |

Mean: 182.7/class, Median: 179/class

### Existing Models (INCOMPATIBLE — 26 alphabet classes only)
- `weights/best_model.pth` — Stage 1, 26 classes
- `weights/stage2_best_model.pth` — Stage 2, 26-class vocab
- `weights/slt_final_t5_model/` — Stage 3 Flan-T5, 26-class vocab
- **All must be retrained from scratch** on the 315-class dataset

### Architecture Already Implemented (from previous plans.md)
All Phase 1-4 improvements are already in the code:
- Learnable adjacency residual (Phase 1A)
- Drop-graph regularization (Phase 1B)
- Bone + bone-motion features: 10→16 channels (Phase 1C)
- Multi-scale TCN in Stage 2 (Phase 2A)
- Causal masking option (Phase 2B)
- Signer normalization augmentation (Phase 3A)
- Extended geometric features: 76 dims (Phase 4A)
- Trigram LM + dynamic weighting (Phase 4B)
- Curriculum learning, EMA, focal loss, mixup — all implemented

**No architecture changes needed. This is a training execution plan.**

---

## Pre-Training: Data Preparation

### Step 0A: Fix Label Aliases in Manifest

**Problem**: 3 classes have <10 samples — will poison training.

| Class | Samples | Action |
|-------|---------|--------|
| HIS | 4 | Merge → HIS_HER (already has 341 → becomes 345) |
| HER | 7 | Merge → HIS_HER (becomes 352) |
| EXCUSE | 4 | Drop entirely (no alias target, 4 samples = noise) |

**How**: Update `manifest.json` in-place. No re-extraction needed — the .npy files are identical, we just change the label mapping.

```python
import json
m = json.load(open('ASL_landmarks_float16/manifest.json'))
updated = {}
for fname, label in m.items():
    if label == 'HIS' or label == 'HER':
        updated[fname] = 'HIS_HER'
    elif label == 'EXCUSE':
        continue  # Drop
    else:
        updated[fname] = label
json.dump(updated, open('ASL_landmarks_float16/manifest.json', 'w'), indent=2)
```

**Result**: 314 classes, ~57,549 files

**Verification**:
- No class has fewer than 15 samples
- HIS_HER count = 352
- EXCUSE not in manifest
- Total labels = 314

### Step 0B: Create Quality Exclusion List

The quality audit `quality_audit.json` stores summary stats but not per-file tier assignments. We need per-file exclusion data.

**Option A** (preferred): Re-run `verify_extraction_quality.py` with per-file output mode to generate `ASL_landmarks_float16/exclude_list.json` containing filenames flagged as `spatial_outlier` or `no_motion`.

**Option B** (lightweight): Add a fast inline check in the training dataset loader — check motion range and spatial bounds per file at load time. Skip files with zero motion or out-of-bounds landmarks.

**Files to exclude**: ~2,055 (spatial_outlier: 1,479 + no_motion: 576)
**Trainable after filtering**: ~55,494

### Step 0C: Update Hardcoded Paths

| File | Variable | Current (Kaggle) | New |
|------|----------|-------------------|-----|
| `train_stage_1.py` | `data_path` default | `/kaggle/input/datasets/kokoab/batch-1/ASL_landmarks_float16` | `ASL_landmarks_float16` |
| `train_stage_1.py` | `save_dir` default | `/kaggle/working/` | `weights/stage1/` |
| `train_stage_2.py` | `data_path` default | `/kaggle/input/datasets/kokoab/batch-1/ASL_landmarks_float16` | `ASL_landmarks_float16` |
| `train_stage_2.py` | `stage1_ckpt` default | `/kaggle/input/datasets/kokoab/model-dataset/best_model.pth` | `weights/stage1/best_model.pth` |
| `train_stage_2.py` | `save_dir` default | `/kaggle/working/` | `weights/stage2/` |
| `train_stage_3.py` | paths | Various | `weights/stage3/` |

**Also update**: `test/SLT_test.py`, `test/test_offline_pipeline.py`, `src/camera_inference.py` checkpoint paths.

**Decision**: These can be CLI args. But defaults should point to local relative paths for local/cloud training.

### Step 0D: Decide Training Platform

| Platform | GPU | VRAM | Cost | Time (all stages) |
|----------|-----|------|------|--------------------|
| **Kaggle** | 2× T4 | 16 GB each | Free (30 hrs/week) | ~16-24 hours total |
| **Vast.ai** | RTX 4090 | 24 GB | ~$0.40/hr | ~4-6 hours total |
| **Colab Pro** | A100 | 40 GB | ~$10/month | ~3-5 hours total |
| **Local Mac** | MPS | Shared | Free | ~24-48 hours (slow) |

**Recommendation**: Kaggle free tier is sufficient. Upload 1.8GB landmarks as a Kaggle dataset. Stage 1 + 2 fit comfortably on T4 (16GB). Stage 3 (Flan-T5-Base 250M params) fits with fp16.

**If Kaggle**: Default paths must use `/kaggle/input/...` and `/kaggle/working/` — which they already do. Just upload the new manifest.

---

## Stage 1: Isolated Sign Classification

### What It Trains
DS-GCN (3 blocks) + Transformer (4 layers) → 314-class classifier on individual 32-frame clips.

### Input Pipeline
```
.npy [32, 47, 10] → bone features [32, 47, 16] → online augment → model
```

### Architecture Summary
- DSGCNEncoder: 3× GCN blocks (64→128→128→256) with learnable adjacency, drop-graph, node attention
- 76 geometric features (joint angles, palm orientation, finger spread, hand-face distances)
- 4× Transformer (d=256, 8 heads, FFN=1024, DropPath)
- ClassifierHead: frame attention → MLP(256→512→314)

### Training Config (already in script)
| Parameter | Value |
|-----------|-------|
| Epochs | 200 (early stop patience=40) |
| Batch size | 256 (effective 1024 with accum=4) |
| LR | 1e-3, AdamW, cosine warmup (5 epochs) |
| Loss | Focal CE (γ=2) + label smoothing 0.15 + mixup (α=0.2) |
| Augmentation | Temporal warp 50%, rotation ±10°, scale 0.85-1.15, noise σ=0.003, signer norm 30% |
| Curriculum | Phase 1 (ep 1-50): single-hand, Phase 2 (51-100): gradual mix, Phase 3 (101+): full |
| EMA | Decay 0.999 |
| Sampler | Inverse-frequency weighted (temp=0.5) |
| Split | 70/15/15 stratified |

### What to Modify
1. **Add exclude_list filtering** in `SignDataset.__init__` — skip files in exclude list
2. **Update default paths** (or pass via CLI args)
3. Nothing else — architecture and hyperparameters are already tuned

### Expected Outcomes
- **Target**: Top-1 >80%, Top-5 >93% on 314 classes
- 26-class model was >95%, but 314 classes is 12× harder
- Bottom classes (RIGHT: 22) will have lower accuracy — focal loss + weighted sampler help
- Curriculum learning helps: alphabet (single-hand) learned first, then complex signs

### What to Look For After Training
1. **Per-class accuracy** — flag any class <40%
2. **Confusion matrix** — identify systematically confused pairs (e.g., A↔S, E↔O)
3. **Train/val gap** — if >15%, model is overfitting
4. **Curriculum effect** — accuracy should jump around epoch 50 (two-hand signs enter)
5. **Save confused class pairs** → feed to Stage 2 `confused_glosses` parameter

### Output
- `weights/stage1/best_model.pth` — best val accuracy checkpoint
- `weights/stage1/last_checkpoint.pth` — latest checkpoint (resumable)
- Contains: encoder weights, classifier weights, label_to_idx, idx_to_label, num_classes=314

---

## Stage 2: Continuous Sign Recognition (CTC)

### Dependencies
- **Requires Stage 1 `best_model.pth`** — loads frozen encoder weights

### What It Trains
Frozen Stage 1 encoder + MultiScaleTCN (temporal pooling) + BiLSTM (2-layer, bidirectional) + CTC classifier.

Variable-length sign sequences → gloss sequence prediction.

### Input Pipeline
```
Synthetic sequences: concatenate 1-8 random clips from manifest
→ bone features [T, 47, 16]
→ transition injection (35%, 4-12 frames between clips)
→ boundary jitter (±3 frames)
→ speed warp (30%)
→ model forward: frozen encoder [T, 256] → MultiScaleTCN → BiLSTM → CTC logits
```

### Training Data Generation
The script auto-generates synthetic continuous sequences by:
1. Randomly sampling 1-8 clips per sequence from the manifest
2. Concatenating with realistic transitions between signs
3. 10% single-sign (edge case), 10% long (7-8 signs), 80% standard (2-6 signs)
4. `confused_glosses` parameter oversamples hard pairs (3× weight)

**Default**: 15,000 train / 3,000 val / 3,000 test sequences

### Architecture Summary
- Frozen DSGCNEncoder (from Stage 1) — no gradients, ~50% VRAM savings
- MultiScaleTCN: 3 parallel Conv1d (kernel 3/5/9) → fuse → pool 32→4 tokens per clip
- BiLSTM: 2 layers, hidden=512 per direction → 1024-dim output
- Classifier: Linear(1024 → vocab_size=315) where vocab = `<BLANK>` (idx 0) + 314 glosses

### Training Config (already in script)
| Parameter | Value |
|-----------|-------|
| Epochs | 100 (early stop patience=25) |
| Batch size | 32 (variable-length, padded) |
| LR | 1e-3, AdamW, weight_decay=1e-4 |
| Loss | CTCLoss(blank=0, zero_infinity=True) |
| Gradient clip | 5.0 |
| EMA | Decay 0.999 |
| Augmentation | Rotation ±10°, scale 0.85-1.15, noise σ=0.003 (batch-level) |

### Critical CTC Constraints (INVIOLABLE)
- **CTC blank = index 0** — `<BLANK>` must be first in vocabulary
- **PAD ≠ blank** — padding token must NOT be index 0
- **out_lens ≥ y_lens** for every sample — assertion in training loop
- Transition injection must maintain clip/target alignment

### What to Modify
1. **Update stage1_ckpt path** → `weights/stage1/best_model.pth`
2. **Update data_path** → `ASL_landmarks_float16/`
3. **Update save_dir** → `weights/stage2/`
4. **(Optional) Pass `confused_glosses`** from Stage 1 confusion matrix analysis

### Expected Outcomes
- **Target**: WER < 35% on synthetic test set (greedy decoding)
- Real-world WER will be better with beam search + LM rescoring
- Common failure: insertions (CTC inserts extra glosses) and deletions (short signs missed)

### What to Look For
1. **WER trend** — should decrease steadily, plateau around epoch 50-70
2. **Decoded examples** — manually inspect 20 predictions vs references
3. **Assertion failures** — if `out_lens < y_lens` fires, MultiScaleTCN output count is wrong
4. **Long sequence accuracy** — 5+ sign sequences should still decode reasonably

### Output
- `weights/stage2/stage2_best_model.pth` — best WER checkpoint
- Contains: full model state, gloss_to_idx, idx_to_gloss, vocab_size=315

---

## Stage 3: Gloss-to-English Translation

### Dependencies
- **Does NOT depend on Stage 1/2 weights** — trains independently on text data
- **Requires**: `slt_stage3_dataset_v2.csv` from `generate_stage3_data_v2.py`
- **Can run in parallel with Stage 2**

### Step 3A: Generate Training Data

Run `python3 src/generate_stage3_data_v2.py` to produce `slt_stage3_dataset_v2.csv`.

**What it generates**:
- Single-word glosses: HELLO → ["Hello.", "Hi.", "Hey."]
- Multi-word templates with semantic filtering (avoids nonsense like "BUY PASSWORD")
- Dialogue sequences with context
- ≥20% questions
- Sequences up to 7-10 glosses

**CRITICAL VERIFICATION**: The vocabulary in `generate_stage3_data_v2.py` is hardcoded. It MUST match the 314 manifest classes.

```python
# After generation, verify:
import pandas as pd, json
csv = pd.read_csv('slt_stage3_dataset_v2.csv')
csv_glosses = set()
for row in csv['gloss']:
    csv_glosses.update(row.split())
manifest = json.load(open('ASL_landmarks_float16/manifest.json'))
manifest_glosses = set(manifest.values())
missing = csv_glosses - manifest_glosses
extra = manifest_glosses - csv_glosses
print(f"In CSV but not manifest: {missing}")  # Should be empty
print(f"In manifest but not CSV: {extra}")    # OK — some signs may not have templates
```

If `missing` is non-empty → the T5 model will see glosses at test time that it's never seen → fix the vocabulary in `generate_stage3_data_v2.py`.

### Step 3B: Train Flan-T5

**Model**: `google/flan-t5-base` (250M params, instruction-tuned)

| Parameter | Value |
|-----------|-------|
| Max input | 96 tokens |
| Max target | 64 tokens |
| Batch size | 32, gradient accum=2 (effective 64) |
| LR | 2e-4, cosine schedule, 200 warmup steps |
| Label smoothing | 0.1 |
| Early stopping | On validation loss |
| FP16 | Yes (CUDA) |

**Prompt format**:
```
# Without context:
"Translate this ASL gloss to natural conversational English: HELLO HOW YOU"

# With context:
"[Previous: Hello! | How are you?] Translate this ASL gloss to natural conversational English: I GOOD THANK-YOU"
```

### What to Modify
1. Verify vocabulary alignment (Step 3A verification)
2. Update save path if needed
3. No architecture changes

### Expected Outcomes
- Flan-T5-Base handles this well — it's a constrained translation task
- `TOMORROW I GO SCHOOL` → "I'm going to school tomorrow."
- Context-aware turns should produce coherent conversation flow

### Output
- `weights/stage3/slt_final_t5_model/` — fine-tuned T5 model + tokenizer

---

## Stage 4: Language Model + Integration

### Step 4A: Build N-gram Language Model

Run `python3 src/build_language_model.py`.

**Sources**: Manifest vocab + Stage 3 CSV sequences + common ASL patterns.

**Output**: `weights/gloss_bigram_lm.pkl` (trigram with Kneser-Ney smoothing)

**Verification**: LM vocab should cover all 314 manifest glosses.

### Step 4B: End-to-End Offline Test

Run `python3 test/test_offline_pipeline.py` — 100 ground-truth cases.

**Requires**: All 3 stage checkpoints + LM pickle in `weights/`.

**What it tests**: Stage 2 CTC decode → Stage 3 T5 translate → compare to expected English.

### Step 4C: Camera Inference Test

Run `python3 src/camera_inference.py` with webcam.

**Note**: Camera uses **MediaPipe** (not RTMW) for real-time extraction. Known domain gap — MediaPipe is faster but less accurate. The model was trained on RTMW landmarks. This affects accuracy but is acceptable for real-time use.

**Flow**: Camera → MediaPipe hands+face → 32-frame buffer → bone features → multi-hypothesis segmentation (N=1,2,3,4) → CTC beam search (width=25) → LM rescoring → T5 translation → display.

---

## Execution Order

```
PARALLEL PREP (do all before training):
├── Step 0A: Fix manifest aliases (HIS→HIS_HER, drop EXCUSE)
├── Step 0B: Create exclusion list from quality audit
├── Step 0C: Update hardcoded paths
└── Step 0D: Upload data to training platform (if Kaggle/cloud)

SEQUENTIAL TRAINING:
Step 1: Train Stage 1 — isolated sign classifier (314 classes)
   │    ~2-3 hours on 4090, ~8-12 hours on T4
   │    Output: weights/stage1/best_model.pth
   │
   ├── Checkpoint: analyze confusion matrix, identify confused pairs
   │
   ▼
Step 2: Train Stage 2 — CTC continuous recognition
   │    ~1-2 hours on 4090, ~4-6 hours on T4
   │    Output: weights/stage2/stage2_best_model.pth
   │
   │  PARALLEL with Step 2:
   │  ├── Step 3A: Generate Stage 3 data (CPU, <5 min)
   │  └── Step 3B: Train Stage 3 — Flan-T5 (independent of Stage 1/2)
   │       ~1-2 hours on 4090
   │       Output: weights/stage3/slt_final_t5_model/
   │
   ▼
Step 4A: Build language model (CPU, <1 min)
Step 4B: Run offline integration test
Step 4C: Run camera inference test
```

**Critical path**: Stage 1 → Stage 2 → Integration test
**Parallel path**: Stage 3 data generation + training (independent of Stage 1/2)

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| 314 classes with imbalanced data | Low accuracy on rare classes | Focal loss (γ=2) + weighted sampler (temp=0.5) + curriculum learning |
| Synthetic CTC sequences ≠ real signing | Stage 2 WER inflated | Transition injection + jitter + speed warp simulate real boundaries |
| Stage 3 vocab mismatch with manifest | T5 sees unknown glosses | Verify vocab alignment in Step 3A (CRITICAL) |
| MediaPipe ↔ RTMW domain gap at inference | Camera accuracy lower than offline | Known tradeoff — model learns relative hand shape, not absolute coordinates |
| Positional jitter in RTMW training data | Noisy absolute positions | Model uses relative features (bones, geometric features, node attention) — resilient |
| RIGHT class (22 samples) underperforming | Low per-class accuracy | Weighted sampler gives it 8× weight; focal loss focuses on hard examples |
| Kaggle session timeout (12h) | Training interrupted | Checkpoint resume is implemented — restart picks up from last epoch |

---

## Files to Modify (Summary)

| File | Change | Priority |
|------|--------|----------|
| `ASL_landmarks_float16/manifest.json` | Merge HIS/HER → HIS_HER, drop EXCUSE | Step 0A |
| `src/train_stage_1.py` | Update default paths, add exclude_list filter in SignDataset | Step 0C |
| `src/train_stage_2.py` | Update default paths (data_path, stage1_ckpt, save_dir) | Step 0C |
| `src/generate_stage3_data_v2.py` | Verify vocabulary matches 314 manifest classes | Step 3A |
| `src/train_stage_3.py` | Update paths if needed | Step 3B |
| `src/build_language_model.py` | Update paths if needed | Step 4A |
| `test/test_offline_pipeline.py` | Update checkpoint paths | Step 4B |
| `src/camera_inference.py` | Update checkpoint paths | Step 4C |

---

## Post-Training Checklist

- [ ] Stage 1: Top-1 >80%, Top-5 >93%
- [ ] Stage 1: No class <40% accuracy
- [ ] Stage 1: Train/val gap <15%
- [ ] Stage 2: WER <35% (greedy decoding)
- [ ] Stage 2: CTC constraint assertion never fires
- [ ] Stage 3: Vocab alignment verified
- [ ] Stage 3: Reasonable translations on test set
- [ ] LM: All 314 glosses in vocabulary
- [ ] Offline test: Passes >70% of 100 test cases
- [ ] Camera test: Real-time inference works end-to-end
