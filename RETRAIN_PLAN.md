# SLT Retrain Plan — Re-extract with rtmlib + Optimized Training

## Overview

Re-extract all 57k training videos with rtmlib ONNX RTMW, then retrain Stage 1 and Stage 2 with optimized hyperparameters and cleaned-up input channels. Stage 3 T5 does NOT need retraining.

---

## Step 1: Re-extract Training Data with rtmlib

### What
Extract all videos in `data/raw_videos/ASL VIDEOS/` using `docker/extract_rtmlib.py` (rtmlib ONNX RTMW). Output: new `ASL_landmarks_rtmlib/` directory with `[32, 47, 10]` float16 .npy files.

### How
```bash
# On Mac via Docker (slow — ~48 hours due to x86 emulation)
# Or on Codespace/cloud (faster — ~8-12 hours native x86)

# Create batch extraction script
python docker/extract_rtmlib_batch.py \
  --input "data/raw_videos/ASL VIDEOS" \
  --output ASL_landmarks_rtmlib \
  --manifest models/manifest.json
```

### Bugs to Pre-fix

**Bug 1: extract_rtmlib.py imports from extract.py which imports mediapipe**
- Already fixed: fake mediapipe module stub is in extract_rtmlib.py
- Verify: `python docker/extract_rtmlib.py --help` should work without mediapipe

**Bug 2: extract_rtmlib.py processes one video at a time — too slow for 57k**
- Need: batch extraction script that walks the folder tree, maps class labels from folder names, and generates manifest
- The existing `extract.py` does this with multiprocessing — need to adapt for rtmlib
- rtmlib Wholebody should be initialized ONCE and reused across all videos

**Bug 3: Subsample limit `max_process = cfg.target_frames * 3 = 96` may be too aggressive for longer videos**
- Current code drops frames from videos > 96 frames
- For training data (isolated signs), most videos are 30-90 frames — this is fine
- For continuous signing videos (180 new recordings), may need higher limit
- Fix: set `max_process = 128` for single signs

**Bug 4: rtmlib auto-downloads model on first run (~290MB)**
- Docker volume `rtmlib-cache` persists this across runs
- For batch extraction, initialize Wholebody once, pass to all extraction calls

**Bug 5: Output filename format must match existing manifest**
- Current extract.py: `{LABEL}_{video_stem}_{hash}.npy`
- extract_rtmlib.py must use the SAME format
- The manifest maps `filename → label`, so filenames must match

**Bug 6: Label aliases (DRIVE↔DRIVE_CAR, EAT↔EAT_FOOD, etc.)**
- Must apply LABEL_ALIASES from extract.py during re-extraction
- Otherwise class count changes and Stage 1 architecture mismatches

### Verification After Extraction
```bash
# Check file count matches
ls ASL_landmarks_rtmlib/*.npy | wc -l  # should be ~57k

# Check shapes
python -c "
import numpy as np, glob
files = glob.glob('ASL_landmarks_rtmlib/*.npy')
shapes = set()
for f in files[:100]:
    shapes.add(np.load(f).shape)
print('Shapes found:', shapes)  # should be {(32, 47, 10)}
"

# Check class distribution matches
python -c "
import json, os
from collections import Counter
with open('models/manifest.json') as f:
    manifest = json.load(f)
rtmlib_files = set(os.listdir('ASL_landmarks_rtmlib'))
missing = [f for f in manifest if f not in rtmlib_files]
extra = [f for f in rtmlib_files if f.endswith('.npy') and f not in manifest]
print(f'Manifest entries: {len(manifest)}')
print(f'Extracted files: {len(rtmlib_files)}')
print(f'Missing: {len(missing)}')
print(f'Extra: {len(extra)}')
"
```

---

## Step 2: Retrain Stage 1 with Optimized Hyperparameters

### Changes to `src/train_stage_1.py`

**Change 1: Data path → new extraction directory**
```python
data_path = "ASL_landmarks_rtmlib"  # instead of "ASL_landmarks_float16"
```

**Change 2: LR schedule — remove warm restarts**
```python
# BEFORE (line 788-810): CosineWarmupScheduler with T_0=50, T_mult=2
# AFTER: Simple cosine annealing, no restarts

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]
```

**Change 3: Hyperparameters**
```python
# Line ~904-911:
epochs = 150,          # was 300
lr = 3e-4,             # was 7e-4
label_smoothing = 0.05, # was 0.10
focal_gamma = 0.0,     # was 1.0 — DISABLE focal loss
mixup_alpha = 0.1,     # was 0.15
cutmix_prob = 0.15,    # was 0.5
patience = 25,         # was 50
val_every = 1,         # was 3
head_dropout = 0.45,   # was 0.30
accum_steps = 4,       # was 2 (effective batch 1024)
```

**Change 4: Augmentation reduction (in online_augment function)**
```python
# Line ~696-698:
def online_augment(x, rotation_deg=8.0, scale_lo=0.88, scale_hi=1.12, noise_std=0.002,
                   speed_warp_prob=0.3, min_speed=0.8, max_speed=1.2,
                   signer_norm_prob=0.2, temporal_mask_prob=0.15, temporal_mask_frames=3):
```

### Bugs to Pre-fix for Stage 1

**Bug 7: Dataset cache invalidation**
- `SignDataset` caches to `ds_cache.pt`. If you change the data directory to `ASL_landmarks_rtmlib`, the old cache from `ASL_landmarks_float16` will NOT be used (different path) — this is fine.
- BUT if you extract to the SAME directory, delete the cache first: `rm ASL_landmarks_rtmlib/ds_cache.pt`

**Bug 8: Manifest must match new extraction filenames**
- If re-extraction produces different filenames (e.g., hash changes), the manifest won't match
- Fix: generate a NEW manifest during extraction, or ensure filenames match exactly
- Safest: generate new manifest alongside extraction

**Bug 9: STREAM_CHANNELS references channels 10-15 (bone features)**
- Bone features are computed at load time by `compute_bone_features_np()` in train_stage_2.py
- The `SignDataset.__getitem__` in train_stage_1.py loads [32, 47, 10] then the training loop calls `compute_bone_features` to make [32, 47, 16]
- Wait — actually check: does Stage 1 training call compute_bone_features?

Let me check...

**Bug 9 (CRITICAL): Where are bone features computed for Stage 1?**

Looking at train_stage_1.py line 1156:
```python
x = online_augment(x)  # Must happen on full 16-ch tensor
```

And SignDataset returns [32, 47, 10]. But the model expects 16 channels.

The bone features must be computed INSIDE SignDataset or in the training loop. Let me check the SignDataset code...

Actually, looking at line 482-498 of train_stage_1.py:
```python
arr = np.load(data_path / fname).astype(np.float32)
if arr.shape != (32, 47, 10):
    continue
```

It loads [32, 47, 10]. Then line 1156 calls `online_augment(x)` which expects 16 channels.

The bone features must be computed somewhere between loading and augmenting. Looking at the compute_bone_features import... it's in train_stage_1.py line 161:
```python
def compute_bone_features(x):
    """Append bone direction (3ch) + bone-motion (3ch) to input tensor.
```

This is a TORCH function called inside the training loop after the data is loaded. The SignDataset stores [32, 47, 10] and bone features are added on-the-fly.

So the flow is:
1. SignDataset loads [32, 47, 10] from .npy
2. DataLoader batches it → [B, 32, 47, 10]
3. Training loop: `x = compute_bone_features(x)` → [B, 32, 47, 16]
4. Then `x = online_augment(x)` on [B, 32, 47, 16]

Wait, but I don't see compute_bone_features being called explicitly in the training loop. Let me check again...

Actually looking at the training loop (line 1154-1156):
```python
for i, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    x = online_augment(x)
```

The x here is [B, 32, 47, 10]... but online_augment expects 16 channels (line 715: `xyz = x[..., :3]`, `mask = x[..., 9:10]`).

Hmm, actually online_augment works on XYZ (channels 0-2) and mask (channel 9), rebuilds derived channels via `_rebuild_derived_channels`. Let me check that function...

Actually, `_rebuild_derived_channels` rebuilds vel, acc, mask, bone, bone_motion from XYZ. So the flow is:
1. Load [32, 47, 10]
2. online_augment takes [B, 32, 47, 10+], modifies XYZ, calls _rebuild_derived_channels which outputs [B, 32, 47, 16]
3. Model receives [B, 32, 47, 16]

So the bone features are computed INSIDE online_augment via _rebuild_derived_channels. This means the .npy files only need to store [32, 47, 10] and everything else is computed on-the-fly. Good — no changes needed to the .npy format.

### Training Command
```bash
# On GPU (Vast.ai or similar)
cd /workspace/SLT
python src/train_stage_1.py \
  --data_path ASL_landmarks_rtmlib \
  --save_dir models/output_rtmlib_joint \
  --stream_name joint \
  --epochs 150
```

### Verification After Stage 1 Training
```bash
# Check training chart — should be smooth, no spikes
python src/plot_training.py models/output_rtmlib_joint/history.json

# Expected: Val accuracy > 88%, smooth convergence by epoch 100
```

---

## Step 3: Retrain Stage 2 CTC

### Changes to `src/train_stage_2.py`

**Change 1: Data path**
```python
data_path = "ASL_landmarks_rtmlib"
```

**Change 2: Stage 1 checkpoint path**
```python
stage1_ckpt = "models/output_rtmlib_joint/best_model.pth"
```

### Bugs to Pre-fix for Stage 2

**Bug 10: Stage 2 encoder has 4 transformer layers, Stage 1 has 6**
- Stage 2's `DSGCNEncoder` is initialized with default `num_transformer_layers=4`
- Stage 1 checkpoint has 6 layers
- Stage 2 loads Stage 1 weights with `strict=False` — the extra 2 layers are silently ignored
- This is BY DESIGN — Stage 2 uses a smaller encoder. No fix needed.

**Bug 11: SyntheticCTCDataset generates sequences from individual .npy files**
- It loads files from `data_path` using the manifest
- If manifest filenames don't match the new .npy filenames → empty dataset
- Fix: ensure manifest matches or generate new manifest during extraction

**Bug 12: compute_bone_features_np is called in SyntheticCTCDataset.__getitem__**
- Line 782: `x = compute_bone_features_np(x)`
- This converts [T, 47, 10] → [T, 47, 16]
- Works the same regardless of which extractor produced the .npy — no fix needed

### Training Command
```bash
python src/train_stage_2.py \
  --data_path ASL_landmarks_rtmlib \
  --stage1_ckpt models/output_rtmlib_joint/best_model.pth \
  --save_dir models/output_rtmlib_stage2
```

---

## Step 4: Update Inference Paths

After retraining, update `docker/run_inference.py`:

```python
STAGE1_CKPT = "models/output_rtmlib_joint/best_model.pth"
STAGE2_CKPT = "models/output_rtmlib_stage2/best_model.pth"
STAGE3_DIR = "weights/slt_final_t5_model"  # unchanged
```

And `test/test_offline_pipeline.py`:
```python
STAGE2_WEIGHTS = "models/output_rtmlib_stage2/best_model.pth"
STAGE3_DIR = "weights/slt_final_t5_model"
ASL_DATA_DIR = "ASL_landmarks_rtmlib"
```

---

## Step 5: Ensemble Inference (No Retraining)

After Stage 1 joint stream is retrained, optionally retrain bone/velocity/angle streams too:
```bash
python src/train_stage_1.py --stream_name bone --data_path ASL_landmarks_rtmlib --save_dir models/output_rtmlib_bone
python src/train_stage_1.py --stream_name velocity --data_path ASL_landmarks_rtmlib --save_dir models/output_rtmlib_velocity
```

Then ensemble at inference by averaging softmax outputs from all streams.

---

## Step 6: Add 180 Continuous Signing Videos (Optional but High Impact)

### Recording
- 60 videos of "HELLO HOW YOU"
- 60 videos of "PLEASE HELP ME"
- 60 videos of "SORRY I LATE"

### Processing
1. Extract with rtmlib (same extractor as training)
2. Label each video with gloss sequence
3. Add to Stage 2 training as REAL examples alongside synthetic

### Integration into Stage 2
Create a file `continuous_labels.json`:
```json
{
  "HELLO_HOW_YOU_001.mp4": ["HELLO", "HOW", "YOU"],
  "HELLO_HOW_YOU_002.mp4": ["HELLO", "HOW", "YOU"],
  ...
}
```

Stage 2's `SyntheticCTCDataset` would need a small addition to load real continuous sequences alongside synthetic ones. The real sequences would NOT be segmented into 32-frame clips — they'd be kept as full-length extracted landmarks and treated as ground truth CTC training examples.

---

## Complete Bug Checklist

| # | Bug | Where | Fix | Status |
|---|-----|-------|-----|--------|
| 1 | mediapipe import in extract.py | extract_rtmlib.py | Fake mediapipe stub | Already fixed |
| 2 | Single-video extraction too slow | extract_rtmlib.py | Batch script needed | Need to create |
| 3 | Subsample limit too aggressive | extract_rtmlib.py line 86 | Set max_process=128 | Need to change |
| 4 | rtmlib model auto-download | Docker volume | rtmlib-cache volume in compose | Already fixed |
| 5 | Output filename format | extract_rtmlib.py | Must match manifest format | Need to verify |
| 6 | Label aliases | extract_rtmlib.py | Import LABEL_ALIASES from extract.py | Need to add |
| 7 | Dataset cache | train_stage_1.py | Delete ds_cache.pt before training | Manual step |
| 8 | Manifest mismatch | extraction | Generate new manifest during extraction | Need to create |
| 9 | Bone features computation | train_stage_1.py | Computed in online_augment/_rebuild_derived_channels | No fix needed |
| 10 | Encoder layer count mismatch | train_stage_2.py | By design (4 vs 6 layers) | No fix needed |
| 11 | SyntheticCTCDataset manifest | train_stage_2.py | Use new manifest | Point to new manifest |
| 12 | compute_bone_features_np | train_stage_2.py | Works on any [T, 47, 10] | No fix needed |
| 13 | run_inference.py model paths | docker/run_inference.py | Update paths after retrain | Manual step |
| 14 | test_offline_pipeline.py paths | test/test_offline_pipeline.py | Update paths | Manual step |
| 15 | T5 model path | everywhere | Use weights/slt_final_t5_model ALWAYS | Already identified |
| 16 | Stage 2 d_model=384 not default 256 | run_inference.py / test scripts | Pass d_model=384 explicitly | Already fixed in run_inference.py |

---

## Timeline

| Day | Task | Hours |
|-----|------|-------|
| Day 1 | Create batch extraction script, start extraction | 2h setup + overnight run |
| Day 2 | Verify extraction, apply hyperparameter changes, start Stage 1 training | 2h setup + 2h training |
| Day 2 | Start Stage 2 training after Stage 1 finishes | 1.5h training |
| Day 3 | Verify results, update inference paths, test | 2h |
| Day 3 | (Optional) Record 180 continuous videos | 2h |
| Day 4 | (Optional) Retrain Stage 2 with real videos, ensemble streams | 4h |

**Total: ~2 days for core pipeline, 4 days for everything.**

---

## Expected Results After Retrain

| Metric | Before | After |
|--------|--------|-------|
| Stage 1 val accuracy | 85.5% | 88-92% |
| Stage 1 with ensemble | N/A | 92-95% |
| CTC on ground truth concat | 95%+ | 95%+ (same) |
| Video inference (single sign) | 40-60% | 80-85% |
| Video inference (multi-sign) | 20-40% | 65-75% |
| Live webcam (record-then-process) | Not working | 75-85% |
