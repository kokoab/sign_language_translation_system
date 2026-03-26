# SLT Full Pipeline Guide — AWS ($60 Credits)

Extraction (64-core spot) + Training (A10G GPU spot), all on AWS. Total cost: **~$2.50–5**.

All steps use the AWS Console (browser GUI) unless there's no alternative.

---

## PART 1: Setup (One-Time)

### 1.1 Create a Key Pair

1. Open https://console.aws.amazon.com/ec2/
2. In the left sidebar, click **Key Pairs** (under Network & Security)
3. Click the orange **Create key pair** button
4. Fill in:
   - Name: `slt-key`
   - Key pair type: **RSA**
   - Private key file format: **.pem**
5. Click **Create key pair**
6. Your browser downloads `slt-key.pem` automatically
7. Open Terminal on your Mac and run:

```bash
mv ~/Downloads/slt-key.pem ~/.ssh/slt-key.pem
chmod 400 ~/.ssh/slt-key.pem
```

### 1.2 Create a Security Group

1. In the EC2 left sidebar, click **Security Groups** (under Network & Security)
2. Click **Create security group**
3. Fill in:
   - Security group name: `slt-sg`
   - Description: `SSH access for SLT`
   - VPC: leave the default selected
4. Under **Inbound rules**, click **Add rule**
   - Type: **SSH**
   - Source: **My IP** (auto-fills your current IP)
5. Click **Create security group**

---

## PART 2: Extraction on Spot Instance (~$0.50)

### 2.1 Launch the Instance

1. Go to https://console.aws.amazon.com/ec2/
2. Click the orange **Launch instances** button

**Name and tags:**
- Name: `slt-extraction`

**Application and OS Images:**
1. Click **Browse more AMIs**
2. In the search bar, type `Ubuntu 24.04`
3. Select **Ubuntu Server 24.04 LTS** — select **64-bit (x86)** (NOT Arm — RTMPose requires x86)
4. Click **Select**

**Instance type:**
1. Click the dropdown, type `c6i.16xlarge` in the search box
2. Select **c6i.16xlarge** (64 vCPU, 128 GB RAM, x86 Intel)
3. If not listed, try `c7a.12xlarge` (48 vCPU, AMD) or `c6i.8xlarge` (32 vCPU)

**Key pair:**
- Select `slt-key` from the dropdown

**Network settings:**
1. Click **Edit**
2. Under **Firewall (security groups)**, select **Select existing security group**
3. Pick `slt-sg` from the dropdown

**Configure storage:**
- Change the size to **200** GB (RTMPose + mmpose packages need ~3 GB extra)
- Change type to **gp3**

**Advanced details (expand this section):**
1. Scroll down and expand **Advanced details**
2. Find **Purchasing option** section
3. Check the box: **Request Spot Instances**
4. Leave all other spot settings as default

**Launch:**
1. Click **Launch instance**
2. Click the blue instance ID link that appears
3. Wait for **Instance state** to show **Running** (refresh if needed)
4. Copy the **Public IPv4 address** from the details panel

### 2.2 Connect to the Instance

You need Terminal for SSH — there's no GUI alternative for this:

```bash
ssh -i ~/.ssh/slt-key.pem ubuntu@<PASTE_IP_HERE>
```

Type `yes` when asked about the fingerprint.

### 2.3 Install Dependencies

Run these commands on the instance:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 tmux
python3 -m venv ~/slt-env
source ~/slt-env/bin/activate
pip install opencv-python-headless mediapipe numpy scipy

# RTMPose-WholeBody (4th detection pass — much better for fists/occlusion)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mmengine mmcv mmdet mmpose

# Download RTMPose model checkpoint (~170 MB)
python3 models/download_rtmpose.py
```

**Note:** The RTMPose install adds ~2 GB of dependencies but significantly improves hand detection for signs like LOVE, CARRY, DONT. If you skip this step, extraction still works with MediaPipe-only (3-pass instead of 4-pass).

### 2.4 Upload Your Files

These commands must be run from your **Mac Terminal** (open a new tab). No GUI alternative for file transfer to EC2.

**Your data: ~33 GB raw videos + ~600 MB .npy files. At 50 Mbps upload:**

| What | Raw Size | Compressed | Upload Time (50 Mbps) |
|------|----------|-----------|----------------------|
| Source code + models | ~200 MB | — | ~30 sec |
| Existing .npy files | ~600 MB | ~400 MB | ~1 min |
| Raw videos | ~33 GB | ~20-22 GB | **~55-70 min** |
| **Total** | | | **~60-75 min** |

**Tip:** Start the upload before going to bed. Compress + upload in one command, and the instance just idles (spot pricing is cheap).

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# 1. Upload source code + model file (~30 sec)
scp -i ~/.ssh/slt-key.pem -r src/ ubuntu@<IP>:~/src/
scp -i ~/.ssh/slt-key.pem -r models/ ubuntu@<IP>:~/models/

# 2. Upload existing .npy files (~1 min)
tar czf landmarks.tar.gz ASL_landmarks_float16/
scp -i ~/.ssh/slt-key.pem landmarks.tar.gz ubuntu@<IP>:~/

# 3. Upload raw videos — compress first to save ~30% transfer time
#    Compression takes ~10-15 min locally, but saves ~30 min of upload
tar czf videos.tar.gz -C data/raw_videos "ASL VIDEOS"
scp -i ~/.ssh/slt-key.pem videos.tar.gz ubuntu@<IP>:~/
```

Then back on the **instance** (your SSH tab):

```bash
# Unpack landmarks
tar xzf landmarks.tar.gz && rm landmarks.tar.gz

# Unpack videos (~5 min for 20 GB compressed)
mkdir -p data/raw_videos
tar xzf videos.tar.gz -C data/raw_videos/ && rm videos.tar.gz
```

**Alternative: Use S3 for video upload (recommended — can resume if interrupted)**

SCP doesn't resume if your connection drops. S3 multipart upload does. For 33 GB over WiFi, S3 is safer:

1. Open https://console.aws.amazon.com/s3/
2. Click **Create bucket**
   - Bucket name: `slt-videos-temp-12345` (add random numbers — must be globally unique)
   - Region: same as your instance (e.g. US East N. Virginia)
   - Click **Create bucket**
3. Click into your new bucket
4. Click **Upload** → **Add folder** → select your `ASL VIDEOS` folder → click **Upload**
5. Wait for upload to complete (~60-75 min at 50 Mbps)

**Or use AWS CLI for faster parallel upload** (from your Mac):

```bash
# Install AWS CLI if not already
brew install awscli
aws configure  # enter your credentials

# Parallel upload — faster than single-stream SCP
aws s3 sync "data/raw_videos/ASL VIDEOS" s3://slt-videos-temp-12345/ --quiet
```

Then on the **instance**, pull from S3 (internal network = **blazing fast**, ~2 min for 33 GB):

```bash
sudo apt install -y awscli
aws configure
# Enter your Access Key ID, Secret Access Key, region (us-east-1), format (json)
# Find your keys at: https://console.aws.amazon.com/iam/ → Users → your user → Security credentials → Create access key

mkdir -p ~/data/raw_videos/"ASL VIDEOS"
aws s3 sync s3://slt-videos-temp-12345/ ~/data/raw_videos/"ASL VIDEOS"/ --quiet
```

### 2.5 Run Extraction

On the instance:

```bash
# Start tmux (keeps extraction running even if you disconnect)
tmux new -s extract

source ~/slt-env/bin/activate
cd ~

# For c7g.16xlarge (64 cores):
python3 src/extract_do.py --workers 62

# For c7g.8xlarge (32 cores):
python3 src/extract_do.py --workers 30
```

**To disconnect safely:** Press `Ctrl+B`, then `D`. You can close your terminal — extraction continues.

**To reconnect later:**

```bash
ssh -i ~/.ssh/slt-key.pem ubuntu@<IP>
tmux attach -t extract
```

**Time: ~1–1.5 hrs for 50k videos on 62 workers.**

### 2.6 Download Results

Once extraction finishes, from your **Mac Terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Compress on instance first (run this via SSH)
ssh -i ~/.ssh/slt-key.pem ubuntu@<IP> "tar czf ~/results.tar.gz ASL_landmarks_float16/"

# Download to your Mac
scp -i ~/.ssh/slt-key.pem ubuntu@<IP>:~/results.tar.gz .
tar xzf results.tar.gz && rm results.tar.gz
```

### 2.7 Terminate the Instance

1. Go to https://console.aws.amazon.com/ec2/ → **Instances**
2. Check the box next to `slt-extraction`
3. Click **Instance state** (dropdown at top) → **Terminate instance**
4. Click **Terminate** to confirm

The instance state will change to **Terminated**. You stop being charged immediately.

---

## PART 3: Training on GPU Spot Instance (~$1.50–2)

### 3.1 Launch a GPU Instance

1. Go to https://console.aws.amazon.com/ec2/ → **Launch instances**

**Name and tags:**
- Name: `slt-training`

**Application and OS Images:**
1. Click **Browse more AMIs**
2. Click the **AWS Marketplace AMIs** tab on the left
3. Search: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch`
4. Select the latest **Ubuntu** version → click **Continue**
5. This AMI has NVIDIA drivers + CUDA + PyTorch pre-installed

If you can't find it, just use **Ubuntu 22.04 LTS x86** and install drivers manually (see troubleshooting).

**Instance type:**
1. Search for `g5.xlarge`
2. Select **g5.xlarge** (1x A10G GPU, 24GB VRAM, 4 vCPU, 16 GB RAM)
3. Fallback: `g4dn.xlarge` (T4 16GB — cheaper but slower)

**Key pair:** `slt-key`

**Network settings:**
1. Click **Edit** → **Select existing security group** → `slt-sg`

**Configure storage:** **100 GB** gp3

**Advanced details:**
1. Expand section
2. Check **Request Spot Instances**

**Launch:**
1. Click **Launch instance**
2. Click instance ID → wait for **Running**
3. Copy **Public IPv4 address**

### 3.2 Connect and Verify GPU

```bash
ssh -i ~/.ssh/slt-key.pem ubuntu@<GPU_IP>

nvidia-smi
# Should show: A10G, 24GB VRAM
```

### 3.3 Install Dependencies

```bash
# If using Deep Learning AMI (PyTorch already installed):
pip install mediapipe opencv-python-headless numpy transformers datasets pandas inflect scikit-learn matplotlib seaborn

# If using plain Ubuntu (no Deep Learning AMI):
sudo apt update && sudo apt install -y python3-pip python3-venv nvidia-driver-535 nvidia-cuda-toolkit tmux
sudo reboot
# Wait 30 sec, SSH back in, then:
python3 -m venv ~/slt-env
source ~/slt-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python-headless numpy transformers datasets pandas inflect scikit-learn matplotlib seaborn
```

### 3.4 Upload Code + Extracted Data

From your **Mac Terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# Upload source code
scp -i ~/.ssh/slt-key.pem -r src/ ubuntu@<GPU_IP>:~/src/

# Upload extracted landmarks
tar czf landmarks.tar.gz ASL_landmarks_float16/
scp -i ~/.ssh/slt-key.pem landmarks.tar.gz ubuntu@<GPU_IP>:~/
```

On the **GPU instance**:

```bash
tar xzf landmarks.tar.gz && rm landmarks.tar.gz
```

### 3.5 Train All Stages

```bash
tmux new -s train
cd ~

# If using plain Ubuntu:
source ~/slt-env/bin/activate

# Stage 1: ~1.5–3 hrs on A10G (early stops ~epoch 60–80)
python3 src/train_stage_1.py

# Stage 2: ~45 min–1.5 hrs on A10G
python3 src/train_stage_2.py

# Stage 3: ~15–20 min on A10G
python3 src/train_stage_3.py
```

**To disconnect safely:** `Ctrl+B`, then `D`
**To reconnect:** `tmux attach -t train`

Monitor GPU (open second SSH):

```bash
ssh -i ~/.ssh/slt-key.pem ubuntu@<GPU_IP>
watch -n 2 nvidia-smi
```

### 3.6 Download Weights

From your **Mac Terminal**:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT
scp -i ~/.ssh/slt-key.pem -r ubuntu@<GPU_IP>:~/weights/ weights/
```

### 3.7 Terminate the Instance

1. Go to https://console.aws.amazon.com/ec2/ → **Instances**
2. Check the box next to `slt-training`
3. **Instance state → Terminate instance**
4. Click **Terminate**

---

## PART 4: Verify Locally

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT
python3 src/extract.py        # regenerate manifest
python3 test/SLT_test.py      # run tests
python3 src/camera_inference.py  # test webcam
```

---

## PART 5: Clean Up

### Delete S3 Bucket (if used)

1. Go to https://console.aws.amazon.com/s3/
2. Click on `slt-videos-temp-12345`
3. Click **Empty** → type `permanently delete` → **Empty**
4. Go back, select the bucket → click **Delete** → type bucket name → **Delete**

### Delete Security Group

1. Go to https://console.aws.amazon.com/ec2/ → **Security Groups**
2. Select `slt-sg`
3. **Actions → Delete security groups** → **Delete**

### Delete Key Pair

1. Go to EC2 → **Key Pairs**
2. Select `slt-key`
3. **Actions → Delete** → type `Delete` → confirm

On your Mac:

```bash
rm ~/.ssh/slt-key.pem
```

---

## Cost Summary

| Step | Instance | Hours | Spot Price | Cost |
|------|----------|-------|-----------|------|
| Extraction (50k videos) | c6i.16xlarge (64 core, x86) | ~2–2.5 hrs | ~$0.40/hr | ~$0.80–1 |
| Training (all 3 stages) | g5.xlarge (A10G GPU) | ~3–5 hrs | ~$0.40/hr | ~$1.50–2 |
| S3 storage (temporary) | 15 GB | ~3 hrs | — | ~$0.01 |
| **Total** | | | | **~$2.50–3** |

**Remaining credits: ~$57.**

**Note:** Extraction is slightly slower per-video than before (~2.5 sec vs ~1.8 sec) because of the 4th RTMPose pass, but the overall fail rate drops from ~34% to ~15-20%, meaning more usable data.

---

## Expected Extraction Improvement (4-pass vs 3-pass)

| Metric | Before (MediaPipe only) | After (+ RTMPose) |
|--------|------------------------|-------------------|
| Overall fail rate | ~34% | ~15-20% |
| LOVE fail rate | 85% | ~15-25% |
| CARRY fail rate | 80% | ~20-30% |
| DONT fail rate | 95% | ~30-40% |
| Usable dataset size | ~19k | ~25-27k |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Instance type not available | In the instance type search, it might say "not available in this zone". Click a different **Availability Zone** under Network settings |
| `MaxSpotInstanceCountExceeded` | Open https://console.aws.amazon.com/servicequotas/ → search **EC2** → find "All Standard Spot Instance Requests" → click **Request increase on account level** |
| Spot instance terminated mid-job | Relaunch from console, re-upload code, re-run. Scripts skip completed work |
| SSH times out | Check security group has SSH rule for **My IP**. Your IP may have changed — update the inbound rule |
| `nvidia-smi: command not found` | Use Deep Learning AMI, or: `sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit` then `sudo reboot` |
| `No space left on device` | EC2 → **Volumes** (left sidebar) → select the volume → **Actions → Modify volume** → increase size. Then on instance: `sudo growpart /dev/xvda 1 && sudo resize2fs /dev/xvda1` |
| Deep Learning AMI not found | In AMI search, make sure you clicked **AWS Marketplace AMIs** tab, not Community AMIs |
| Training `Killed` | OOM. Use `g5.2xlarge` (32 GB RAM) instead of `g5.xlarge` (16 GB RAM) |
| RTMPose install fails | Try: `pip install mmengine` first, then `pip install mmcv mmdet mmpose` one at a time. ARM instances (c7g) don't support mmcv — use x86 (c6i, c7a) |
| RTMPose model download fails | Re-run `python3 models/download_rtmpose.py`. If still fails, extraction works without it (falls back to 3-pass MediaPipe) |
| `_detect_pass_rtmpose returned empty` | Normal if mmpose not installed — fallback to MediaPipe. Check the stats output: `rtmpose=0` means it wasn't used |
