# SLT Full Pipeline Guide — Vast.ai (RTX 4090)

Extraction (4-pass with RTMPose on GPU) + Training (all 3 stages), all on one Vast.ai instance.

**Total cost: ~$1.50–3.00**

**Your data: ~33 GB raw videos, ~600 MB existing .npy files — stored on HuggingFace.**

---

## Step 1: Create Vast.ai Account

1. Go to https://vast.ai
2. Sign up with email
3. Add credits: **$5–10** via credit card or crypto
   - You'll only spend ~$2–3, the rest stays in your account

---

## Step 2: Choose a GPU Instance

Go to https://vast.ai/console/create/

**Filters to set:**

| Filter | Value | Why |
|--------|-------|-----|
| **GPU Type** | RTX 4090 | 24GB VRAM, fastest consumer GPU |
| **GPU RAM** | >=24 GB | All stages fit comfortably |
| **CPU Cores** | >=8 | More workers for parallel extraction |
| **RAM** | >=32 GB | Stage 1 loads full dataset into memory |
| **Disk** | >=100 GB | 33 GB videos + extracted .npy + weights + packages |
| **Docker Image** | `kokoab/slt-pipeline:latest` | Custom image — all deps + RTMPose pre-installed |

**Sort by price** (cheapest first). Look for **$0.25–0.40/hr**.

**What to look for:**
- Reliability score >95%
- NVMe/SSD disk (not HDD)
- Upload/download speed >200 Mbps
- 12+ CPU cores is ideal for extraction

**What to avoid:**
- Interruptible instances (your extraction + training takes hours)
- Instances with <8 CPU cores
- HDD storage (slow for .npy writes)

Click **RENT** on your chosen instance.

---

## Step 3: Connect via SSH

Vast.ai will show you an SSH command like:

```bash
ssh -p 12345 root@<VAST_IP> -L 8080:localhost:8080
```

Copy and run it in your terminal.

---

## Step 4: Verify GPU

```bash
nvidia-smi
# Should show RTX 4090 with 24GB VRAM

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Should print: CUDA: True, GPU: NVIDIA GeForce RTX 4090
```

---

## Step 5: Upload Your Data to HuggingFace (one-time, from Mac)

Skip this if already done. Your data lives on HuggingFace and gets pulled to any instance in minutes.

```python
# Run once from your Mac in a Python shell
from huggingface_hub import HfApi

api = HfApi()

# Create the private repo (only needed once)
api.create_repo(repo_id="kokoabtrue/slt-dataset", repo_type="dataset", private=True)

# Upload your data folder (~33 GB videos + .npy files)
api.upload_large_folder(
    folder_path="/Users/frnzlo/Documents/machine_learning/SLT/data/",
    repo_id="kokoabtrue/slt-dataset",
    repo_type="dataset",
)
```

**Upload speed at 50 Mbps: ~55–70 min for 33 GB.** Do this once — never upload again.

---

## Step 6: Set Up Code + Pull Data on Instance

### 6a. Upload source code from Mac (~30 sec)

Create target directories first, then copy:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Create directories on instance (scp won't create them automatically)
ssh -p <PORT> root@<VAST_IP> "mkdir -p /workspace/src /workspace/test /workspace/models"

# Upload code, tests, and models
scp -P <PORT> -r src/ root@<VAST_IP>:/workspace/src/
scp -P <PORT> -r test/ root@<VAST_IP>:/workspace/test/
scp -P <PORT> -r models/ root@<VAST_IP>:/workspace/models/
```

Replace `<PORT>` and `<VAST_IP>` with values from Vast.ai's SSH command.

### 6b. Pull data from AWS S3 (recommended — fastest, most reliable)

**One-time setup (from your Mac):**
```bash
brew install awscli
aws configure
# Enter: Access Key ID, Secret Access Key, region: us-east-1

# Upload data to S3 (~60 min first time, then incremental)
aws s3 sync "/Users/frnzlo/Documents/machine_learning/SLT/data/raw_videos/ASL VIDEOS" s3://slt-dataset-534409838815-us-east-1-an/data/ --exclude ".cache/*"
```

**On the Vast.ai instance** — pull data (~2-3 min, datacenter-to-datacenter speeds):
```bash
pip install awscli
aws configure  # same Access Key + Secret Key, region: us-east-1

# Download dataset
aws s3 sync s3://slt-dataset-534409838815-us-east-1-an/data/ /workspace/data/

# Create symlink for extract.py (expects data/raw_videos/ASL VIDEOS/)
mkdir -p /workspace/data/raw_videos
ln -s /workspace/data /workspace/data/raw_videos/ASL\ VIDEOS
```

**Cost:** Storage ~$0.78/month. Download ~$3.06 per full pull (34 GB × $0.09/GB).

### 6c. Alternative: HuggingFace

```bash
cd /workspace
python -c "from huggingface_hub import login; login()"
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='kokoabtrue/slt-dataset', repo_type='dataset', local_dir='/workspace/data/')
"

# Create symlink
mkdir -p /workspace/data/raw_videos
ln -s /workspace/data /workspace/data/raw_videos/ASL\ VIDEOS
```

**Note:** With 66K+ small files, HuggingFace may rate-limit (HTTP 429). S3 is more reliable.

---

## Step 7: Run Extraction (4-Pass with RTMPose on GPU)

Use `tmux` so extraction survives SSH disconnects:

```bash
tmux new -s pipeline

cd /workspace

# Run extraction with RTMPose GPU acceleration
# Adjust workers based on your instance's CPU count:
#   8 CPU cores  → --workers 6
#   12 CPU cores → --workers 10
#   16 CPU cores → --workers 14
python3 src/extract_do.py --workers 10
```

**What happens during extraction:**
- Pass 1: MediaPipe video mode (CPU)
- Pass 2: MediaPipe static mode (CPU)
- Pass 3: MediaPipe Tasks API (CPU)
- Pass 4: **RTMPose-WholeBody on RTX 4090 (GPU)** — rescues fists, body contact, occlusion

**Expected: ~50k videos in ~1.5–2 hours.**

The pipeline auto-skips your existing .npy files (the ones you uploaded).

Monitor progress (open second SSH session):

```bash
ssh -p <PORT> root@<VAST_IP>
watch -n 10 "ls /workspace/ASL_landmarks_float16/*.npy | wc -l"
```

**To disconnect safely:** Press `Ctrl+B`, then `D`. Extraction continues.
**To reconnect:** `tmux attach -t pipeline`

---

## Step 8: Verify Extraction

Once extraction finishes:

```bash
# Check results
python3 -c "
import numpy as np, os, random
files = [f for f in os.listdir('ASL_landmarks_float16') if f.endswith('.npy')]
print(f'Total files: {len(files)}')
for f in random.sample(files, min(5, len(files))):
    arr = np.load(f'ASL_landmarks_float16/{f}')
    print(f'  {f}: shape={arr.shape}, dtype={arr.dtype}')
"

# Run quality audit
python3 src/verify_extraction_quality.py --samples 1000
```

Check the stats output for `rtmpose=N` to see how many detections RTMPose rescued.

---

## Step 9: Train Stage 1 — Isolated Sign Classification

```bash
# Still in tmux session
python3 src/train_stage_1.py
```

**What to expect:**
- DS-GCN (3 layers) + Transformer (4 layers, 8 heads)
- 200 max epochs, early stopping patience=40
- Curriculum: single-hand first → gradual mix → full dataset
- **Time on RTX 4090: ~1–2 hours** (early stops ~epoch 60–80)
- Output: `weights/SLT_Stage1_Results/`

Monitor GPU (second SSH session):
```bash
watch -n 2 nvidia-smi
```

GPU utilization should be 60–90%.

---

## Step 10: Train Stage 2 — Continuous Gloss Recognition

```bash
python3 src/train_stage_2.py
```

**What to expect:**
- Frozen Stage 1 encoder + BiLSTM + CTC
- 100 max epochs, early stopping patience=25
- Generates 15k synthetic continuous sequences
- **Time on RTX 4090: ~30–60 min**
- Output: `weights/` (Stage 2 checkpoint)

---

## Step 11: Train Stage 3 — Gloss-to-English Translation

```bash
python3 src/train_stage_3.py
```

**What to expect:**
- Flan-T5-Base (250M params), full fine-tune, 10 epochs
- First run downloads model from HuggingFace (~500 MB)
- **Time on RTX 4090: ~10–15 min**
- Output: `weights/` (Stage 3 checkpoint)

---

## Step 12: Download Everything

On your **Mac**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Download extracted landmarks (new + old)
rsync -avz --progress -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/ASL_landmarks_float16/ ASL_landmarks_float16/

# Download trained weights
rsync -avz --progress -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/weights/ weights/
```

---

## Step 13: DESTROY THE INSTANCE

**Do this immediately.** Every hour costs $0.25–0.40.

1. Go to https://vast.ai/console/instances/
2. Click the **trash icon** next to your instance
3. Confirm destruction

---

## Step 14: Verify Locally

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Regenerate manifest
python3 src/extract.py

# Run tests
python3 test/SLT_test.py

# Test camera inference
python3 src/camera_inference.py
```

---

## Cost Summary

| Step | Time (RTX 4090) | Cost (~$0.35/hr) |
|------|----------------|-----------------|
| HF data pull + code upload (instance idle) | ~10–15 min | ~$0.08 |
| Extraction (50k videos, 4-pass GPU) | ~1.5–2 hrs | ~$0.60 |
| Stage 1 training | ~1–2 hrs | ~$0.50 |
| Stage 2 training | ~30–60 min | ~$0.25 |
| Stage 3 training | ~10–15 min | ~$0.08 |
| **Total** | **~3.5–5 hrs** | **~$1.20–2.00** |

---

## Time Breakdown

| Step | Time |
|------|------|
| Create account + rent instance | ~5 min |
| Pull data from S3 + upload src/ | ~5–10 min |
| Extraction (50k videos, 4-pass + RTMPose GPU) | ~1.5–2 hrs |
| Quality audit | ~5 min |
| Stage 1 training | ~1–2 hrs |
| Stage 2 training | ~30–60 min |
| Stage 3 training | ~10–15 min |
| Download results + weights | ~10 min |
| **Total wall-clock** | **~3.5–5 hrs** |

---

## SSH Cheat Sheet

```bash
# Connect
ssh -p <PORT> root@<VAST_IP>

# Create directories + upload code from Mac
ssh -p <PORT> root@<VAST_IP> "mkdir -p /workspace/src /workspace/test /workspace/models"
scp -P <PORT> -r src/ root@<VAST_IP>:/workspace/src/
scp -P <PORT> -r test/ root@<VAST_IP>:/workspace/test/
scp -P <PORT> -r models/ root@<VAST_IP>:/workspace/models/

# Pull data from S3 (run ON the instance, ~2-3 min)
pip install awscli
aws configure  # Access Key + Secret Key, region: us-east-1
aws s3 sync s3://slt-dataset-534409838815-us-east-1-an/data/ /workspace/data/
mkdir -p /workspace/data/raw_videos
ln -s /workspace/data /workspace/data/raw_videos/ASL\ VIDEOS

# Download results back to Mac
rsync -avz --progress -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/ASL_landmarks_float16/ ASL_landmarks_float16/
rsync -avz --progress -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/weights/ weights/

# tmux (keeps processes running when SSH disconnects)
tmux new -s pipeline        # start
# Ctrl+B, then D             # detach
tmux attach -t pipeline     # reattach
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ssh: Connection refused` | Instance still booting — wait 30 sec, retry |
| `CUDA out of memory` | Shouldn't happen on 24GB 4090. If it does, reduce batch size in training script |
| `nvidia-smi: command not found` | Make sure you selected `kokoab/slt-pipeline:latest` as the Docker image |
| `rsync: connection unexpectedly closed` | Network blip — re-run rsync, it resumes automatically |
| Instance interrupted | Re-rent, re-upload src/, re-pull data from HuggingFace, re-run. Scripts skip completed work |
| Training stops with `Killed` | OOM (system RAM). Rent instance with >=32 GB RAM |
| RTMPose not using GPU | Check `nvidia-smi` — should show mmpose process using VRAM |
| `huggingface-cli: command not found` | Use `python -c "from huggingface_hub import login; login()"` instead |
| HuggingFace download slow / HTTP 429 | Install `hf_transfer` and use `HF_HUB_ENABLE_HF_TRANSFER=1`. If still rate-limited, use tar + scp fallback (see Step 6c) |
| `scp: realpath ... No such file` | Create the directory first: `ssh -p <PORT> root@<VAST_IP> "mkdir -p /workspace/<dir>"` |
| `extract_do.py: No module named 'extract'` | Make sure you're in `/workspace` and `src/` folder is there |
| Extraction slower than expected | Check worker count matches CPU cores. Too many workers + GPU contention = slower |

---

## Expected Extraction Improvement (4-pass vs MediaPipe only)

| Metric | Before (MediaPipe only) | After (+ RTMPose on 4090) |
|--------|------------------------|--------------------------|
| Overall fail rate | ~34% | ~15-20% |
| LOVE fail rate | 85% | ~15-25% |
| CARRY fail rate | 80% | ~20-30% |
| DONT fail rate | 95% | ~30-40% |
| Tier A+B data | ~51% | ~65-70% |
| Usable dataset | ~19k files | ~25-27k files |
