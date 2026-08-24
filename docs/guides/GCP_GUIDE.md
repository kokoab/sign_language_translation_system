# SLT Extraction Guide — Google Cloud ($300 Free Trial)

Extract 50k videos on a high-CPU VM. Total cost: **~$2–3 from $300 free credits**.

---

## Step 1: Sign Up for Google Cloud

1. Go to https://cloud.google.com/free
2. Click **Get started for free**
3. Sign in with your Google account
4. Enter billing info (credit card required, but you won't be charged)
5. You get **$300 in free credits** valid for 90 days

---

## Step 2: Create a VM

1. Go to https://console.cloud.google.com/compute/instances
2. Click **Create Instance**
3. Settings:
   - **Name:** `slt-extraction`
   - **Region:** `us-central1` (cheapest) or closest to you
   - **Zone:** any available
   - **Machine configuration:**
     - Series: **C3** (or C3D, E2 if C3 unavailable)
     - Machine type: **c3-highcpu-44** (44 vCPU, 88 GB RAM)
     - If quota doesn't allow 44 vCPU, try **c3-highcpu-22** (22 vCPU, 44 GB RAM)
   - **Boot disk:** Click **Change**
     - OS: **Ubuntu 24.04 LTS**
     - Size: **100 GB** (need room for videos + .npy output)
     - Type: **SSD persistent disk**
   - **Firewall:** Leave defaults
4. Click **Create**
5. Wait ~30 seconds for it to boot

**Pricing:**
- c3-highcpu-44: ~$1.50/hr
- c3-highcpu-22: ~$0.75/hr

---

## Step 3: SSH Into the VM

Easiest way — click the **SSH** button next to your instance in the console. It opens a browser terminal.

Or from your Mac terminal:

```bash
gcloud compute ssh slt-extraction --zone=us-central1-a
```

(Install gcloud CLI first if needed: `brew install google-cloud-sdk`)

---

## Step 4: Install Dependencies

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
python3 -m venv ~/slt-env
source ~/slt-env/bin/activate
pip install opencv-python-headless mediapipe numpy
```

---

## Step 5: Upload Code

From your **Mac terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Option A: gcloud SCP (easiest)
gcloud compute scp --recurse src/ slt-extraction:~/src/ --zone=us-central1-a

# Option B: regular SCP (if you know the external IP)
scp -r src/ <USERNAME>@<VM_IP>:~/src/
```

---

## Step 6: Upload Existing .npy Files

Skip already-extracted videos by uploading your local .npy files first:

```bash
# Compress first (~600 MB → ~400 MB compressed, faster upload)
cd /Users/frnzlo/Documents/machine_learning/SLT
tar czf landmarks.tar.gz ASL_landmarks_float16/

gcloud compute scp landmarks.tar.gz slt-extraction:~/ --zone=us-central1-a
```

Then on the **VM**:

```bash
cd ~
tar xzf landmarks.tar.gz
rm landmarks.tar.gz
```

---

## Step 7: Upload Raw Videos

This is the slowest step (~15 GB+). Options:

**Option A: gcloud SCP (simple, depends on upload speed)**

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Compress first
tar czf videos.tar.gz -C data/raw_videos "ASL VIDEOS"

# Upload
gcloud compute scp videos.tar.gz slt-extraction:~/ --zone=us-central1-a
```

Then on the **VM**:

```bash
cd ~
mkdir -p data/raw_videos
tar xzf videos.tar.gz -C data/raw_videos/
rm videos.tar.gz
```

**Option B: Google Drive (if videos are already there)**

```bash
# On the VM
pip install gdown
gdown --folder <GOOGLE_DRIVE_FOLDER_URL> -O ~/data/raw_videos/
```

**Option C: Google Cloud Storage bucket (fastest for large files)**

From your **Mac**:

```bash
# Create a bucket and upload
gsutil mb gs://slt-videos-temp
gsutil -m cp -r "data/raw_videos/ASL VIDEOS" gs://slt-videos-temp/

# Then on the VM — download is blazing fast (internal network):
gsutil -m cp -r gs://slt-videos-temp/"ASL VIDEOS" ~/data/raw_videos/
```

This is the fastest method — GCS to VM transfer happens at **10+ Gbps** on Google's internal network.

---

## Step 8: Run Extraction

On the VM:

```bash
source ~/slt-env/bin/activate
cd ~

# 44 vCPU machine
python3 src/extract_do.py --workers 42

# 22 vCPU machine
python3 src/extract_do.py --workers 20
```

Expected output:
```
Skipping 15000 already-processed videos.
Found 50000 remaining videos. Processing with 42 workers...
```

**Time estimates:**
- 44 vCPU (42 workers): **~1.5–2 hrs** for 50k videos
- 22 vCPU (20 workers): **~3–4 hrs** for 50k videos

Monitor progress (open a second SSH session):
```bash
watch -n 10 "ls ~/ASL_landmarks_float16/*.npy | wc -l"
```

**Keep your SSH session alive.** If it disconnects, the process dies. Use `tmux` or `screen`:

```bash
# Start a tmux session (persists even if SSH disconnects)
tmux new -s extract

# Run extraction inside tmux
source ~/slt-env/bin/activate
cd ~
python3 src/extract_do.py --workers 42

# Detach: press Ctrl+B, then D
# Reattach later: tmux attach -t extract
```

---

## Step 9: Download Results

Once extraction finishes, from your **Mac terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Option A: gcloud SCP
gcloud compute scp --recurse slt-extraction:~/ASL_landmarks_float16/ ASL_landmarks_float16/ --zone=us-central1-a

# Option B: Compress first (faster download)
# On VM: tar czf results.tar.gz ASL_landmarks_float16/
gcloud compute scp slt-extraction:~/results.tar.gz . --zone=us-central1-a
tar xzf results.tar.gz
```

---

## Step 10: Verify

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

---

## Step 11: Regenerate Manifest Locally

```bash
python3 src/extract.py
# Output: "All videos are already processed!" + writes manifest.json
```

---

## Step 12: DELETE THE VM

**Do this immediately.** c3-highcpu-44 costs $1.50/hr.

1. Go to https://console.cloud.google.com/compute/instances
2. Check the box next to `slt-extraction`
3. Click **Delete** at the top
4. Confirm

Also delete the GCS bucket if you used Option C:
```bash
gsutil rm -r gs://slt-videos-temp
```

---

## Cost Summary

| Item | Cost |
|------|------|
| VM (c3-highcpu-44, ~2 hrs) | ~$3 |
| GCS bucket (temporary, ~15 GB) | ~$0.03 |
| Network egress (download results) | Free within trial |
| **Total** | **~$3** from $300 free credits |

**Remaining credits: ~$297 for future use (valid 90 days).**

---

## Quota Issues

If GCP says you don't have enough quota for 44 vCPUs:

1. Try a smaller machine: `c3-highcpu-22` (22 vCPU) — still 3–4 hrs
2. Or request a quota increase:
   - Go to https://console.cloud.google.com/iam-admin/quotas
   - Search for "CPUs" in your region
   - Click **Edit Quotas** → request 48
   - Usually approved within minutes for free trial accounts

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Quota exceeded` for CPUs | Try a different region, or request quota increase (see above) |
| SSH disconnects mid-extraction | Use `tmux` (see Step 8) — extraction continues in background |
| `No space left on device` | Increase boot disk size in console (VM → Edit → Boot disk) |
| `ModuleNotFoundError: mediapipe` | Activate venv: `source ~/slt-env/bin/activate` |
| Upload is too slow | Use GCS bucket method (Option C in Step 7) — internal transfer is 10+ Gbps |
| `gcloud: command not found` | Install: `brew install google-cloud-sdk`, then `gcloud init` |

---

## Next Step: Training

See **VASTAI_GUIDE.md** for training all 3 stages on a Vast.ai RTX 3090 (~$0.70–1.30 total).
