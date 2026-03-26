# SLT Extraction Guide — DigitalOcean ($200 Student Credits)

Extraction on a DO CPU-Optimized droplet. For training, see `VASTAI_GUIDE.md`.

**Extraction cost: ~$1–3 from your $200 free credits**

---

## PART 1: Extraction on CPU Droplet (~$1.50)

### 1.1 Redeem GitHub Student Pack Credits

1. Go to https://education.github.com/pack
2. Find **DigitalOcean** in the partner list
3. Click "Get access" — you'll get **$200 in credits**
4. Create a DigitalOcean account (or link existing one)
5. Credits should appear in your billing dashboard

### 1.2 Create a CPU Droplet

1. Log into https://cloud.digitalocean.com
2. Click **Create → Droplets**
3. Settings:
   - **Region:** San Francisco or closest to you
   - **Image:** Ubuntu 24.04 (LTS) x64
   - **Droplet type:** Dedicated CPU
   - **Plan:** `c-16` — 16 vCPUs / 32 GB RAM / 50 GB disk ($0.476/hr)
   - **Authentication:** SSH key (recommended) or password
4. Click **Create Droplet**
5. Copy the IP address shown on the dashboard

### 1.3 SSH Into the Droplet

```bash
ssh root@<DROPLET_IP>
```

### 1.4 Install Dependencies

```bash
apt update && apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
python3 -m venv /opt/slt
source /opt/slt/bin/activate
pip install opencv-python-headless mediapipe numpy
```

### 1.5 Upload Code + Videos

On your **Mac** (new terminal):

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Upload source code (~instant)
scp -r src/ root@<DROPLET_IP>:/opt/slt/src/

# Upload raw videos (~15 GB)
# Option A: Direct (if you have decent upload speed)
rsync -avz --progress "data/raw_videos/ASL VIDEOS/" "root@<DROPLET_IP>:/opt/slt/data/raw_videos/ASL VIDEOS/"

# Option B: Compress first (faster if upload < 50 Mbps)
tar czf videos.tar.gz -C data/raw_videos "ASL VIDEOS"
scp videos.tar.gz root@<DROPLET_IP>:/opt/slt/
# Then on droplet:
# cd /opt/slt && mkdir -p data/raw_videos && tar xzf videos.tar.gz -C data/raw_videos/
```

Upload time estimates (for ~15 GB of videos):
- 10 Mbps upload → ~3.5 hrs (compress first!)
- 50 Mbps upload → ~40 min
- 100+ Mbps upload → ~20 min

### 1.5b Upload Existing .npy Files (skip already-extracted videos)

If you already have some extracted `.npy` files locally, upload them **before** running extraction. The script auto-detects and skips videos that already have a matching `.npy` file.

```bash
# Upload existing extractions (~600 MB for ~6.5k files, takes 2-5 min)
rsync -avz --progress ASL_landmarks_float16/ root@<DROPLET_IP>:/opt/slt/ASL_landmarks_float16/
```

When you run extraction, you'll see:
```
Skipping 6565 already-processed videos.
Found 93435 remaining videos. Processing with 46 workers...
```

### 1.6 Run Extraction

On the droplet:

```bash
source /opt/slt/bin/activate
cd /opt/slt
python3 src/extract_do.py
```

- Uses all 14 available workers (no 9-worker cap)
- Expected: **~2–3 hours for 30k videos**

Monitor progress (another SSH session):
```bash
ls /opt/slt/ASL_landmarks_float16/*.npy | wc -l
```

### 1.7 Download Results to Your Mac

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT
rsync -avz --progress "root@<DROPLET_IP>:/opt/slt/ASL_landmarks_float16/" ASL_landmarks_float16/
```

### 1.8 Verify

```bash
python3 -c "
import numpy as np, os, random
files = [f for f in os.listdir('ASL_landmarks_float16') if f.endswith('.npy')]
print(f'Total files: {len(files)}')
for f in random.sample(files, min(5, len(files))):
    arr = np.load(f'ASL_landmarks_float16/{f}')
    print(f'  {f}: shape={arr.shape}, dtype={arr.dtype}')
"
```

Expected: shape `(32, 47, 10)`, dtype `float16`.

### 1.9 Regenerate Manifest Locally

```bash
python3 src/extract.py
# Output: "All videos are already processed!" + writes manifest.json
```

### 1.10 DESTROY THE CPU DROPLET

1. https://cloud.digitalocean.com/droplets → your droplet → **Destroy**
2. Confirm

---

## Verify Extraction

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT
python3 test/SLT_test.py
```

---

## Cost Summary

| Plan | Time for 100k videos | Cost |
|------|---------------------|------|
| CPU-Optimized 16 vCPU ($0.48/hr) | ~6–8 hrs | ~$3–4 |
| CPU-Optimized 32 vCPU ($0.95/hr) | ~3–4 hrs | ~$3–4 |
| **CPU-Optimized 48 vCPU ($1.43/hr)** | **~70–80 min** | **~$1.70–2** |

**Remaining credits after extraction: ~$198 for future use.**

---

## SSH Cheat Sheet

```bash
ssh root@<DROPLET_IP>
scp -r src/ root@<DROPLET_IP>:/opt/slt/src/
rsync -avz --progress ASL_landmarks_float16/ root@<DROPLET_IP>:/opt/slt/ASL_landmarks_float16/
rsync -avz --progress "data/raw_videos/ASL VIDEOS/" "root@<DROPLET_IP>:/opt/slt/data/raw_videos/ASL VIDEOS/"
rsync -avz --progress "root@<DROPLET_IP>:/opt/slt/ASL_landmarks_float16/" ASL_landmarks_float16/
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ssh: Connection refused` | Droplet still booting — wait 30 sec, retry |
| `No space left on device` | Resize disk in DO dashboard (Settings → Resize) |
| `ModuleNotFoundError: mediapipe` | Activate venv: `source /opt/slt/bin/activate` |
| `rsync: connection unexpectedly closed` | Network blip — re-run rsync, it resumes automatically |
| Extraction seems stuck | Check file count: `ls /opt/slt/ASL_landmarks_float16/*.npy \| wc -l` |

---

## Next Step: Training

See **VASTAI_GUIDE.md** for training all 3 stages on Vast.ai (RTX 3090, ~$0.70–1.30 total).

Once DO GPU access is approved, you can also train on DO GPU droplets (H200, ~$3.44/hr) for ~5x faster training.
