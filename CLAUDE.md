# SLT — Conversational Sign Language Translator

Project instructions for Claude. Read this before diving into code.

## Efficiency router (always)

1. Ask up to 3 clarifying questions if requirements are ambiguous.
2. Classify the request:
   - **Planning** (contracts, interfaces, trade-offs, "what should we do?"): use `cludev2.md` — architecture/plan only, no code.
   - **Implementation** (code edits, bug fixes, "implement this plan"): use `cludev3.md` — surgical edits, run verification.
3. If the user explicitly provides an implementation plan, skip planning and go straight to implementation.

## Token saving

- Prefer `md files/context.md` over loading large plans.
- Only load `md files/MASTER_IMPLEMENTATION_PLAN.md` when the user explicitly requests phased implementation.
- Don't paste large files; cite file paths/symbol names instead.

---

## Architecture

4-stage pipeline translating ASL webcam video → English:

1. **Stage 0** (`src/extract.py`, `extract_batch_rtmlib.py`): RTMW-l wholebody pose → `[32, 47, 10]` tensors (42 hand + 5 face landmarks, XYZ + vel + acc + mask)
2. **Stage 1** (`src/train_stage_1.py`): DS-GCN + Transformer → isolated sign classification (310 classes, d_model=384, 6 transformer layers)
3. **Stage 2** (`src/train_stage_2.py`): Frozen Stage 1 encoder (4 transformer layers, d_model=384) + MultiScaleTCN(out_tokens=4) + SequenceTransformer(4 layers) + CTC → gloss sequences
4. **Stage 3** (`src/train_stage_3.py`): Flan-T5-Base → gloss → natural English
5. **Deploy** (`src/camera_inference.py`): Real-time webcam, N-gram LM rescoring, TTA, dialogue memory

## Critical Architecture Details (DO NOT GET WRONG)

### Stage 1 Model
- `SLTStage1` in `src/train_stage_1.py`
- Encoder: `DSGCNEncoder(in_channels=16, d_model=384, nhead=8, num_transformer_layers=6)`
- Head: `ClassifierHead(d_model=384, num_classes=310, dropout=0.45)` — uses **learned frame_attn** (NOT mean pooling)
- Checkpoint saves `d_model` key for downstream detection
- Input: `[B, 32, 47, 16]` (10ch kinematics + 6ch bone features computed at load time)
- Bone features: computed by `compute_bone_features_np()` in train_stage_2.py or `compute_bone_features()` in train_stage_1.py/test_video_pipeline.py
- Optimized hyperparams: epochs=150, lr=3e-4, accum_steps=4, label_smoothing=0.05, focal_gamma=0.0 (disabled), cosine decay (NO warm restarts)

### Stage 2 Model
- `SLTStage2CTC` in `src/train_stage_2.py`
- Encoder: `DSGCNEncoder(in_channels=16, d_model=384, num_transformer_layers=4)` ← NOTE: 4 layers, not 6!
- TCN: `MultiScaleTCN(d_model=384, out_tokens=4)` with **GroupNorm(8) + GELU + AdaptiveAvgPool1d(4)**
- Seq: `SequenceTransformer(d_model=384, nhead=8, num_layers=4, dropout=0.3)` — uses **nn.ModuleList** (NOT nn.TransformerEncoder)
- Classifier: `Linear(384, 311)` (311 = 310 classes + 1 blank)
- Has `inter_ctc_proj` for intermediate CTC loss
- d_model fallback default: **384** (not 256)
- Checkpoint: `models/output/stage2_best_model.pth`
- Input: `[B, T, 47, 16]` where T is variable (multiple of 32)
- Forward splits input into 32-frame clips, each clip → **4 tokens** after TCN pooling (nc*4 total)
- Stage 2 saves to separate dir (`output_stage2`) to avoid colliding with Stage 1

### Stage 3 Model
- Flan-T5-Base fine-tuned
- **CORRECT checkpoint: `weights/slt_final_t5_model`** ← This one translates properly
- `train_stage_3.py` now saves to `slt_final_t5_model` (previously saved as `slt_conversational_t5_model`)
- Prompt format: `"Translate this ASL gloss to natural conversational English: GLOSS1 GLOSS2 ..."`

### ClassifierHead (CRITICAL — must match in ALL files)
```python
class ClassifierHead(nn.Module):
    def __init__(self, d_model=256, num_classes=29, dropout=0.4):
        super().__init__()
        self.frame_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout * 0.6), nn.Linear(d_model, num_classes))
    def forward(self, x, labels=None):
        attn = F.softmax(self.frame_attn(x).squeeze(-1), dim=1)
        return self.net((x * attn.unsqueeze(-1)).sum(dim=1))
```
**DO NOT use x.mean(dim=1) — that drops the learned frame_attn weights from checkpoint.**

### Data Format
- Raw .npy files: `[32, 47, 10]` float16 (XYZ + vel + acc + mask)
- After bone features: `[32, 47, 16]` float32 (10ch + 3ch bone direction + 3ch bone motion)
- Bone features are computed at LOAD TIME, not stored in .npy
- Geo features (76 total): computed at RUNTIME inside DSGCNEncoder._compute_geo_features()
- N_GEO_FEATURES = 76 (fingertip distances, curl ratios, angles, palm normals, spreads, face distances)

### Extraction
- **NEW**: Training data will be extracted with rtmlib ONNX RTMW-l (`extract_batch_rtmlib.py`) — closes domain gap
- Batch extraction supports `--workers 4` for multiprocessing (spawn context for CUDA safety)
- Docker inference uses rtmlib ONNX RTMW-l (same extractor as training)
- COCO-WholeBody 133 keypoints: Body 0-16, Feet 17-22, Face 23-90, LHand 91-111, RHand 112-132
- Face indices used: nose=23+30, chin=23+8, forehead=23+27, left_ear=23+0, right_ear=23+16

### Test-Time Augmentation (TTA)
- Mirror averaging: run original + hand-swapped (0-20 ↔ 21-41, X-flipped) → average softmax
- Applied in: `docker/run_inference.py` (Stage 1 + Stage 2 + sliding window), `src/camera_inference.py` (Stage 2)
- `_mirror_tta(x)` flips: X coord (ch0), vel_x (ch3), acc_x (ch6), bone_dir_x (ch10), bone_motion_x (ch13)
- Does NOT flip mask (ch9) — that would break hand presence flags

## Dataset

- 310 classes, 57,535 total samples
- 8 classes with < 100 samples: TEACHER(79), LOUD(90), WORKER(94), TEAM(95), CODE(96), SAD(98), AT(99), STRONG(99)
- Imbalance ratio: 6.2x (max 487 for 'I', min 79 for 'TEACHER')
- Split: 70% train / 15% val / 15% test (stratified)
- Label aliases: DRIVE↔DRIVE_CAR, EAT↔EAT_FOOD, ALSO_SAME→ALSO, etc. (see LABEL_ALIASES in extract.py)

## Training Results (pre-optimization, mmpose extraction)

- Stage 1 joint stream: 85.55% Top-1, 93.37% Top-5 (300 epochs, d_model=384)
- Stage 2 CTC: 2.85% WER on synthetic val (60 epochs)
- Stage 3: Flan-T5-Base on 28,333 samples

## Known Issues

1. **rtmlib vs mmpose domain gap** — FIXED by re-extracting with rtmlib (`extract_batch_rtmlib.py`)
2. **Stage 2 trained on synthetic data** — real continuous signing has coarticulation the model hasn't seen. FIX: record 180 real continuous signing videos
3. **Loss curve instability** — FIXED: removed warm restarts, lowered LR, disabled focal loss (see OPTIMIZATION_PLAN.md)
4. **Wrong T5 checkpoint name** — FIXED: `train_stage_3.py` now saves to `slt_final_t5_model`
5. **Architecture mismatches in inference scripts** — FIXED: all inference files now match training exactly (ClassifierHead frame_attn, MultiScaleTCN GroupNorm, SequenceTransformer ModuleList)

## Inviolable Constraints

| Rule | Why |
|------|-----|
| Temporal augmentation: warp XYZ first, recompute kinematics | Warping 10-ch tensor corrupts vel/acc |
| Mirror TTA: swap hand indices (0–20 ↔ 21–41) with X-flip, do NOT flip mask ch9 | Preserves hand identity |
| CTC blank = idx 0; PAD ≠ blank | Avoid target alignment errors |
| Transition injection: maintain clip/target alignment | CTC requires correct lengths |
| Stage 1 ClassifierHead uses frame_attn, NOT mean pooling | Checkpoint has learned attention weights |
| Stage 2 encoder has 4 transformer layers, not 6 | Checkpoint mismatch if wrong |
| Stage 2 TCN uses GroupNorm(8) + GELU + AdaptiveAvgPool1d(4) | BatchNorm or missing pool = wrong architecture |
| Stage 2 SequenceTransformer uses nn.ModuleList, not nn.TransformerEncoder | State dict key mismatch |
| Stage 2 d_model default fallback = 384, not 256 | Silent architecture mismatch |
| Stage 1 and Stage 2 save to DIFFERENT directories | Prevents checkpoint collision |
| Pin rtmlib + onnxruntime versions in Dockerfiles | Unpinned packages broke Vast.ai before |

## File Map

- `src/extract.py` — RTMW extraction core functions (interpolation, normalization, kinematics)
- `extract_batch_rtmlib.py` — Batch extraction with rtmlib for Vast.ai (multiprocessing, resume support)
- `src/train_stage_1.py`, `train_stage_2.py`, `train_stage_3.py` — Training
- `src/test_video_pipeline.py` — End-to-end video test (MediaPipe extraction + Stage 1/2/3)
- `src/camera_inference.py` — Live webcam inference (imports SLTStage2CTC from train_stage_2.py)
- `docker/extract_rtmlib.py` — rtmlib ONNX extraction (Docker, single video)
- `docker/run_inference.py` — .npy → Stage 1/2 → Stage 3 inference (standalone architectures)
- `docker/Dockerfile` — Docker image for local Mac inference (rtmlib + PyTorch CPU + Transformers)
- `Dockerfile` — Vast.ai training image (PyTorch + CUDA + rtmlib + onnxruntime-gpu)
- `docker-compose.yml` — Extract + inference services (local Mac)
- `test/test_offline_pipeline.py` — Offline CTC + T5 test (concatenates training .npy files)
- `src/verify_extraction_quality.py` — Quality audit of .npy files
- `OPTIMIZATION_PLAN.md` — Training optimization plan (all changes implemented)
- `RETRAIN_PLAN.md` — Re-extraction + retrain plan with bug checklist

## Model Checkpoints

| File | What | Use |
|------|------|-----|
| `models/output_joint/best_model.pth` | Stage 1 joint stream (85.5%) | Isolated sign classification |
| `models/stage1_bone.pth` | Stage 1 bone stream | Ensemble member |
| `models/stage1_velocity.pth` | Stage 1 velocity stream | Ensemble member |
| `models/stage1_angle.pth` | Stage 1 angle stream | Ensemble member |
| `models/output/stage2_best_model.pth` | Stage 2 CTC | Continuous recognition |
| `weights/slt_final_t5_model/` | Stage 3 T5 (CORRECT) | Translation |
| `models/manifest.json` | 310 class mapping | Label ↔ index |

## How To Run

```bash
# --- Mac native ---

# Run inference on training .npy files
KMP_DUPLICATE_LIB_OK=TRUE conda run -n sign_ai python docker/run_inference.py ASL_landmarks_float16/HELLO_0062d3ea_280f55.npy --output output

# Offline pipeline test (100 multi-sign sentences)
KMP_DUPLICATE_LIB_OK=TRUE python test/test_offline_pipeline.py

# Docker — extract video then infer
docker compose run -v "./sample_videos:/app/input" extract /app/input/video.mp4
KMP_DUPLICATE_LIB_OK=TRUE conda run -n sign_ai python docker/run_inference.py output/npy/video_name.npy --output output

# --- Vast.ai retrain (see VASTAI_RETRAIN_GUIDE.md for full guide) ---

# Step 1: Extract with rtmlib (multiprocessing)
python extract_batch_rtmlib.py --input "data/raw_videos/ASL VIDEOS" --output ASL_landmarks_rtmlib --device cuda --workers 4

# Step 2: Train Stage 1 (auto-detects ASL_landmarks_rtmlib)
python src/train_stage_1.py --data_path ASL_landmarks_rtmlib --save_dir models/output_rtmlib_joint

# Step 3: Train Stage 2
python src/train_stage_2.py --data_path ASL_landmarks_rtmlib --stage1_ckpt models/output_rtmlib_joint/best_model.pth --save_dir models/output_rtmlib_stage2

# Step 4: Stage 3 — NO retraining needed (uses existing slt_final_t5_model)

# Step 5: Download new checkpoints to Mac
# models/output_rtmlib_joint/best_model.pth → Stage 1
# models/output_rtmlib_stage2/best_model.pth → Stage 2
```

## CLI Arguments (for Vast.ai)

### train_stage_1.py
```
--data_path    Path to .npy directory (default: auto-detect ASL_landmarks_rtmlib or ASL_landmarks_float16)
--save_dir     Output directory (default: auto-detect)
--stream       joint|bone|velocity|bone_motion|angle (default: joint)
--epochs       150 (default)
--lr           3e-4 (default)
--batch_size   256 (default)
--accum_steps  4 (default, effective batch=1024)
--patience     25 (default)
--d_model      384 (default)
--num_layers   6 (default)
```

### train_stage_2.py
```
--data_path    Path to .npy directory (default: auto-detect)
--stage1_ckpt  Path to Stage 1 checkpoint (default: auto-detect)
--save_dir     Output directory (default: output_stage2, separate from Stage 1)
--epochs       60 (default)
--lr           5e-4 (default)
--batch_size   32 (default)
--patience     35 (default)
```
