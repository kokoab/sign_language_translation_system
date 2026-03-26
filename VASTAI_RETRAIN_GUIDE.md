# Vast.ai Retrain Guide — Complete Step-by-Step

## GPU Requirements

- **GPU:** RTX 4090 or RTX 3090 (24GB VRAM)
- **Disk:** 50GB+ (for videos + extracted data + models)
- **Docker image:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`
- **Estimated time:** ~3-4 hours on RTX 4090, ~6 hours on RTX 3090
- **Estimated cost:** ~$1-2 on RTX 4090

---

## Step 0: Push Code and Upload Data

```bash
# On your Mac — push latest code to GitHub
git add -A && git commit -m "Pre-retrain: all fixes applied" && git push

# On Vast.ai instance — clone repo
cd /workspace
git clone <your-repo-url> SLT
cd SLT
```

Upload training videos (choose one method):
```bash
# Option A: SCP from Mac (~35GB, takes a while)
scp -r "data/raw_videos/ASL VIDEOS/" vast:/workspace/SLT/data/raw_videos/

# Option B: If videos are on Google Drive, download directly on Vast.ai
# (faster — no double transfer through your Mac)
```

---

## Step 1: Install Dependencies

```bash
cd /workspace/SLT

# System deps
apt update && apt install -y libgl1-mesa-glx libglib2.0-0 ffmpeg

# Python deps (ALL PINNED)
pip install numpy==1.26.4 scipy==1.13.1 opencv-python-headless==4.10.0.84
pip install rtmlib==0.0.15 onnxruntime-gpu==1.20.1
pip install transformers==4.46.3 sentencepiece==0.2.0
pip install scikit-learn==1.5.1 matplotlib==3.9.2 seaborn==0.13.2
pip install pandas==2.2.2 inflect==7.3.1

# MediaPipe (needed for extract.py imports, installed with --no-deps to avoid conflicts)
pip install --no-deps mediapipe==0.10.14
pip install absl-py attrs flatbuffers "protobuf>=3.20,<5"

# Force numpy back (some packages overwrite it)
pip install numpy==1.26.4

# Verify
python -c "
import torch; print('CUDA:', torch.cuda.is_available())
from rtmlib import Wholebody; print('rtmlib: OK')
import numpy; print('numpy:', numpy.__version__)
print('ALL CHECKS PASSED')
"
```

---

## Step 2: Extract with rtmlib

The extraction script is already in the repo at `extract_batch_rtmlib.py` (root level). Do NOT create a new one.

```bash
cd /workspace/SLT

# GPU-accelerated extraction with 4 parallel workers (~45 min on RTX 4090)
python extract_batch_rtmlib.py \
  --input "data/raw_videos/ASL VIDEOS" \
  --output ASL_landmarks_rtmlib \
  --device cuda \
  --workers 4

# If interrupted, resume without re-extracting:
python extract_batch_rtmlib.py \
  --input "data/raw_videos/ASL VIDEOS" \
  --output ASL_landmarks_rtmlib \
  --device cuda \
  --workers 4 \
  --resume
```

### Verify extraction
```bash
python -c "
import json, glob, numpy as np
files = glob.glob('ASL_landmarks_rtmlib/*.npy')
with open('ASL_landmarks_rtmlib/manifest.json') as f:
    m = json.load(f)
print(f'Files: {len(files)}, Manifest: {len(m)} entries')
# Spot check shapes
for f in files[:5]:
    print(f'  {f}: {np.load(f).shape}')
# Should be ~57k files, all shape (32, 47, 10)
"
```

---

## Step 3: Train Stage 1

All hyperparameter optimizations are already in the code (LR schedule, augmentation, etc.). Just run:

```bash
# Delete any stale cache
rm -f ASL_landmarks_rtmlib/ds_cache.pt

# Train Stage 1 joint stream (~1.5-2 hrs on RTX 4090)
python src/train_stage_1.py \
  --data_path ASL_landmarks_rtmlib \
  --save_dir /workspace/output \
  --stream joint

# Expected: val accuracy > 88%, smooth loss curve (no spikes)
# Check training chart:
python src/plot_training.py /workspace/output/history.json
```

---

## Step 4: Train Stage 2 CTC

```bash
# Train Stage 2 (~30-45 min on RTX 4090)
python src/train_stage_2.py \
  --data_path ASL_landmarks_rtmlib \
  --stage1_ckpt /workspace/output/best_model.pth \
  --save_dir /workspace/output_stage2

# IMPORTANT: Always pass --stage1_ckpt explicitly to avoid loading wrong checkpoint
```

---

## Step 5: Stage 3 — NO Retraining Needed

Stage 3 (Flan-T5-Base) translates glosses to English. It doesn't depend on the extraction method, so the existing `weights/slt_final_t5_model` checkpoint works as-is.

If you want to retrain anyway (e.g., with more data):
```bash
python src/train_stage_3.py
# Saves to: /workspace/output/slt_final_t5_model
```

---

## Step 6: Download Results to Mac

```bash
# On your Mac — create directories
mkdir -p models/output_rtmlib_joint models/output_rtmlib_stage2

# Download checkpoints
scp vast:/workspace/SLT/output/best_model.pth models/output_rtmlib_joint/
scp vast:/workspace/SLT/output/history.json models/output_rtmlib_joint/
scp vast:/workspace/SLT/output_stage2/best_model.pth models/output_rtmlib_stage2/
scp vast:/workspace/SLT/ASL_landmarks_rtmlib/manifest.json models/manifest_rtmlib.json
```

---

## Step 7: Update Inference Paths on Mac

Update checkpoint paths in these files:

**`docker/run_inference.py`** (lines 27-30):
```python
STAGE1_CKPT = "models/output_rtmlib_joint/best_model.pth"
STAGE2_CKPT = "models/output_rtmlib_stage2/best_model.pth"
STAGE3_DIR  = "weights/slt_final_t5_model"  # unchanged
```

**`src/camera_inference.py`** (line 37):
```python
STAGE2_CKPT = "models/output_rtmlib_stage2/best_model.pth"
```

**`test/test_offline_pipeline.py`** (lines 17-19):
```python
STAGE2_WEIGHTS = "models/output_rtmlib_stage2/best_model.pth"
STAGE3_DIR = "weights/slt_final_t5_model"
ASL_DATA_DIR = "ASL_landmarks_rtmlib"  # if you downloaded extracted data
```

Then test:
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n sign_ai python docker/run_inference.py ASL_landmarks_float16/HELLO_0062d3ea_280f55.npy --output output
KMP_DUPLICATE_LIB_OK=TRUE python test/test_offline_pipeline.py
```

---

## Optional: Train Ensemble Streams

After joint stream is trained, optionally train additional streams for +4-6% accuracy via ensemble:

```bash
python src/train_stage_1.py --data_path ASL_landmarks_rtmlib --save_dir /workspace/output_bone --stream bone
python src/train_stage_1.py --data_path ASL_landmarks_rtmlib --save_dir /workspace/output_velocity --stream velocity
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'mediapipe'` during extraction | The fake mediapipe stub is built into `extract_batch_rtmlib.py` |
| `CUDA out of memory` during extraction | Reduce `--workers` to 2 or 1 |
| Extraction hangs/crashes | Use `--resume` to continue from where it stopped |
| Stage 1 loss spikes | Verify you're using the latest code (LR schedule fix is already applied) |
| Stage 2 won't load Stage 1 weights | Always pass `--stage1_ckpt` explicitly |
| Val accuracy lower than 85% | Check extraction quality: `python -c "import numpy as np; d = np.load('ASL_landmarks_rtmlib/HELLO_*.npy'); print(d.shape, d.min(), d.max())"` |
| `ds_cache.pt` stale data | Delete it: `rm ASL_landmarks_rtmlib/ds_cache.pt` |
| `pip` version conflicts | Run `pip install numpy==1.26.4` again after all installs |
