# SLT Extraction Guide — Azure for Students (Free $100 Credits)

Extract 50k videos on a 48-core Azure VM. Total cost: **$0** (from free student credits).

---

## Step 1: Activate Azure for Students

1. Go to https://azure.microsoft.com/en-us/free/students/
2. Click **Start for free**
3. Sign in with your **school email** (the one linked to your GitHub Student Pack)
4. Verify student status — no credit card needed
5. You get **$100 in free credits** valid for 12 months

---

## Step 2: Create a VM

1. Go to https://portal.azure.com
2. Click **Create a resource → Virtual Machine**
3. **Basics tab:**
   - **Subscription:** Azure for Students
   - **Resource group:** Click **Create new** → name it `slt-extraction`
   - **VM name:** `slt-extract`
   - **Region:** `(US) East US` or closest to you
   - **Image:** **Ubuntu 24.04 LTS - x64 Gen2**
   - **Size:** Click **See all sizes** → search for one of these:

| Size | vCPUs | RAM | Price | Time for 50k |
|------|-------|-----|-------|-------------|
| **Standard_F48s_v2** | 48 | 96 GB | ~$1.52/hr | ~1.5 hrs |
| Standard_F32s_v2 | 32 | 64 GB | ~$1.01/hr | ~2–3 hrs |
| Standard_F16s_v2 | 16 | 32 GB | ~$0.51/hr | ~4–5 hrs |

   Pick the largest available. If F48s_v2 shows a quota error, try F32s_v2.

   - **Authentication:** SSH public key (recommended)
     - Username: `azureuser`
     - SSH public key: paste your Mac's public key (`cat ~/.ssh/id_rsa.pub`)
     - Or choose **Password** if easier
4. **Disks tab:**
   - OS disk type: **Standard SSD**
   - OS disk size: **128 GB** (need room for videos + output)
5. **Networking tab:** Leave defaults
6. **Review + create → Create**
7. Wait ~1 min for deployment, then click **Go to resource**
8. Copy the **Public IP address**

---

## Step 3: SSH Into the VM

```bash
ssh azureuser@<VM_IP>
```

If using password, enter it when prompted.

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
scp -r src/ azureuser@<VM_IP>:~/src/
```

---

## Step 6: Upload Existing .npy Files

```bash
# Compress first
cd /Users/frnzlo/Documents/machine_learning/SLT
tar czf landmarks.tar.gz ASL_landmarks_float16/

scp landmarks.tar.gz azureuser@<VM_IP>:~/
```

On the **VM**:

```bash
cd ~
tar xzf landmarks.tar.gz
rm landmarks.tar.gz
```

---

## Step 7: Upload Raw Videos

**Option A: Direct SCP (simple)**

```bash
# On your Mac — compress first
cd /Users/frnzlo/Documents/machine_learning/SLT
tar czf videos.tar.gz -C data/raw_videos "ASL VIDEOS"

scp videos.tar.gz azureuser@<VM_IP>:~/
```

On the **VM**:

```bash
cd ~
mkdir -p data/raw_videos
tar xzf videos.tar.gz -C data/raw_videos/
rm videos.tar.gz
```

**Option B: Azure Blob Storage (faster for large uploads)**

```bash
# On your Mac — install Azure CLI
brew install azure-cli
az login

# Create storage account + container
az storage account create --name slttemp --resource-group slt-extraction --location eastus --sku Standard_LRS
az storage container create --name videos --account-name slttemp

# Upload videos (uses parallel upload)
az storage blob upload-batch --account-name slttemp --destination videos --source "data/raw_videos/ASL VIDEOS" --destination-path "ASL VIDEOS"
```

On the **VM**:

```bash
# Download from blob (internal network, very fast)
sudo apt install -y azure-cli
az login
mkdir -p ~/data/raw_videos
az storage blob download-batch --account-name slttemp --source videos --destination ~/data/raw_videos/
```

---

## Step 8: Run Extraction

On the VM, use `tmux` so extraction survives SSH disconnects:

```bash
# Install tmux
sudo apt install -y tmux

# Start tmux session
tmux new -s extract

# Activate env and run
source ~/slt-env/bin/activate
cd ~

# Pick workers based on your VM size:
python3 src/extract_do.py --workers 46    # F48s_v2 (48 vCPU)
python3 src/extract_do.py --workers 30    # F32s_v2 (32 vCPU)
python3 src/extract_do.py --workers 14    # F16s_v2 (16 vCPU)

# Detach from tmux: press Ctrl+B, then D
# You can now close the terminal. Extraction continues.
# Reattach later: tmux attach -t extract
```

Monitor progress (second SSH session):

```bash
watch -n 10 "ls ~/ASL_landmarks_float16/*.npy | wc -l"
```

---

## Step 9: Download Results

Once extraction finishes, from your **Mac terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Option A: Direct SCP
scp -r azureuser@<VM_IP>:~/ASL_landmarks_float16/ ASL_landmarks_float16/

# Option B: Compress first (faster)
# On VM: tar czf results.tar.gz ASL_landmarks_float16/
scp azureuser@<VM_IP>:~/results.tar.gz .
tar xzf results.tar.gz
rm results.tar.gz
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

## Step 12: DELETE EVERYTHING

**Do this immediately.** F48s_v2 costs $1.52/hr.

### Delete the VM:

1. Go to https://portal.azure.com → **Resource groups** → `slt-extraction`
2. Click **Delete resource group**
3. Type the name to confirm → **Delete**

This deletes the VM, disk, network, and everything in the group.

### Delete blob storage (if used):

```bash
az storage account delete --name slttemp --resource-group slt-extraction --yes
```

---

## Cost Summary

| Item | Cost |
|------|------|
| VM (F48s_v2, ~2 hrs) | ~$3 |
| Disk (128 GB SSD, ~2 hrs) | ~$0.01 |
| Blob storage (temporary) | ~$0.01 |
| **Total** | **~$3 from $100 free credits** |

**Remaining credits: ~$97 for future use (valid 12 months).**

---

## Quota Issues

If F48s_v2 isn't available or shows a quota error:

1. Try a **different region** (West US, North Europe, etc.)
2. Try a smaller size: F32s_v2 or F16s_v2
3. Request a quota increase:
   - Portal → **Subscriptions** → **Usage + quotas**
   - Search for "Fsv2" → **Request increase**
   - Usually approved in minutes for student accounts

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Quota exceeded` | Try different region or smaller VM. See quota section above |
| SSH disconnects mid-extraction | Use `tmux` (see Step 8). Extraction continues in background |
| `No space left on device` | Resize disk: Portal → VM → Disks → Resize |
| `ModuleNotFoundError: mediapipe` | Activate venv: `source ~/slt-env/bin/activate` |
| Upload too slow | Use Azure Blob Storage method (Option B in Step 7) |
| VM won't start | Student subscription may have region restrictions. Try `East US` or `West Europe` |
| Can't verify student status | Use the same email linked to your GitHub Student Pack |

---

## Next Step: Training

See **VASTAI_GUIDE.md** for training all 3 stages on a Vast.ai RTX 3090 (~$0.70–1.30 total).
