# SLT Extraction Guide — GitHub Codespaces (Free)

Extract 50k videos on a 16-core or 32-core Codespace. Total cost: **$0**.

---

## Step 1: Create a Codespace

1. Go to your SLT repo on GitHub (push it if you haven't)
2. Click **Code → Codespaces → Create codespace on master**
3. **Before creating**, click the **...** menu → **New with options**
4. Select machine type:
   - **16-core / 64 GB RAM** — safe choice, ~2.5 hrs for 50k videos
   - **32-core / 128 GB RAM** — if available, ~1 hr for 50k videos
5. Click **Create codespace**

If 32-core isn't listed, 16-core is fine.

**Core-hour budget (Student Pack: 180 core-hrs/month):**

| Machine | Time for 50k | Core-hrs used | % of monthly quota |
|---------|-------------|---------------|-------------------|
| 16-core | ~2.5 hrs | ~40 | 22% |
| 32-core | ~1 hr | ~32 | 18% |

---

## Step 2: Install Dependencies

The Codespace opens a VS Code terminal. Run:

```bash
sudo apt update && sudo apt install -y libgl1-mesa-glx libglib2.0-0
pip install opencv-python-headless mediapipe numpy
```

---

## Step 3: Upload Existing .npy Files

Upload your already-extracted files so the script skips them.

**Option A: From your Mac terminal (outside the Codespace)**

```bash
# Get the Codespace SSH info: gh codespace ssh --config
# Then use the Codespace name shown

cd /Users/frnzlo/Documents/machine_learning/SLT

# Upload existing .npy files (~600 MB)
gh codespace cp -e ASL_landmarks_float16/ "remote:/workspaces/SLT/ASL_landmarks_float16/"
```

**Option B: If your repo has Git LFS or the files are small enough**

If your .npy files are already committed or on cloud storage, pull them directly in the Codespace.

**Option C: Upload via Codespace UI**

In VS Code (Codespace), right-click the file explorer → **Upload** — but this is slow for many files.

---

## Step 4: Upload Raw Videos

Your raw videos (~15 GB+) need to be accessible in the Codespace. Options:

**Option A: GitHub CLI (recommended)**

From your **Mac terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Compress videos first (saves upload time)
tar czf videos.tar.gz -C data/raw_videos "ASL VIDEOS"

# Copy to Codespace
gh codespace cp -e videos.tar.gz "remote:/workspaces/SLT/"
```

Then in the **Codespace terminal**:

```bash
cd /workspaces/SLT
mkdir -p data/raw_videos
tar xzf videos.tar.gz -C data/raw_videos/
rm videos.tar.gz  # free up disk space
```

**Option B: Download from cloud storage**

If your videos are on Google Drive, Dropbox, or S3:

```bash
# Example with gdown (Google Drive)
pip install gdown
gdown --folder <GOOGLE_DRIVE_FOLDER_URL> -O data/raw_videos/
```

---

## Step 5: Run Extraction

```bash
cd /workspaces/SLT

# 16-core Codespace
python3 src/extract_do.py --workers 14

# 32-core Codespace
python3 src/extract_do.py --workers 30
```

The script will print:
```
Skipping 6565 already-processed videos.
Found 50000 remaining videos. Processing with 14 workers...
```

Monitor progress (open a second terminal in VS Code):
```bash
watch -n 10 "ls /workspaces/SLT/ASL_landmarks_float16/*.npy | wc -l"
```

**IMPORTANT:** Keep the browser tab open. Codespaces will idle-timeout after 30 min of inactivity. To prevent this:
- Keep the terminal visible and active
- Or set a longer timeout: Settings → Codespaces → Default idle timeout → **240 minutes**

---

## Step 6: Download Results

Once extraction finishes, download the .npy files to your Mac.

From your **Mac terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

gh codespace cp -e "remote:/workspaces/SLT/ASL_landmarks_float16/" ASL_landmarks_float16/
```

---

## Step 7: Verify

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

## Step 8: Regenerate Manifest Locally

```bash
python3 src/extract.py
# Output: "All videos are already processed!" + writes manifest.json
```

---

## Step 9: Stop the Codespace

**Do this immediately after downloading.** Running Codespaces eat core-hours even if idle.

1. In VS Code: **Codespaces** menu (bottom-left) → **Stop Current Codespace**
2. Or: https://github.com/codespaces → **...** → **Stop codespace**
3. You can **delete** it after downloading results to reclaim storage

---

## Time Estimates for 50k Videos

| Machine | Workers | Time | Core-hrs |
|---------|---------|------|----------|
| 16-core | 14 | ~2.5 hrs | ~40 |
| 32-core | 30 | ~55 min | ~30 |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| 32-core not available | Use 16-core — still finishes in ~2.5 hrs |
| Codespace timed out / disconnected | Restart it — your files are still there. Re-run `extract_do.py`, it skips already-done files |
| `No space left on device` | Codespaces have 32 GB disk by default. Delete `videos.tar.gz` after extracting. If still tight, use a `devcontainer.json` to request more disk |
| `gh codespace cp` is slow | Compress files first: `tar czf results.tar.gz ASL_landmarks_float16/`, then copy the single archive |
| `ModuleNotFoundError: mediapipe` | Re-run: `pip install opencv-python-headless mediapipe numpy` |
| Upload is taking forever | Use Google Drive or S3 as an intermediary — upload from Mac to cloud, download from cloud to Codespace |

---

## Next Step: Training

See **VASTAI_GUIDE.md** for training all 3 stages on a Vast.ai RTX 3090 (~$0.70–1.30 total).
