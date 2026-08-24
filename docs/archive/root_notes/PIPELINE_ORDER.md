# SLT Pipeline — Complete Execution Order

Step-by-step order for the full pipeline: data collection, extraction, verification, and training.

---

## Phase 1: Download Supplementary Data (Mac)

Download additional videos for classes with high extraction failure rates.

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# 1. WLASL — critical classes (>50% fail), then supplement (>30% fail)
python3 src/download_wlasl.py --priority p0
python3 src/download_wlasl.py --priority p1

# 2. MSASL — Microsoft ASL dataset
python3 src/download_msasl.py --priority p0

# 3. SignASL.org — clear, consistent clips
python3 src/download_signasl.py --priority p0

# 4. YouTube search — for classes not in any dataset (DONT, FALL, etc.)
python3 src/download_other_sources.py --download --classes DONT FALL LOW LOOK EXAM
```

All videos go to `data/raw_videos/ASL VIDEOS/{CLASS}/` automatically.

---

## Phase 2: Compress and Upload to Vast.ai (Mac)

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# 1. Compress videos (~10-15 min, saves ~30% upload time)
tar czf videos.tar.gz -C data/raw_videos "ASL VIDEOS"

# 2. Compress existing .npy files
tar czf landmarks.tar.gz ASL_landmarks_float16/

# 3. Upload everything to Vast.ai
scp -P <PORT> -r src/ root@<VAST_IP>:/workspace/src/
scp -P <PORT> -r models/ root@<VAST_IP>:/workspace/models/
scp -P <PORT> landmarks.tar.gz root@<VAST_IP>:/workspace/
scp -P <PORT> videos.tar.gz root@<VAST_IP>:/workspace/
```

Upload time: ~60-75 min at 50 Mbps for ~33 GB.

---

## Phase 3: Setup Vast.ai Instance

```bash
# 1. Install core dependencies
pip install mediapipe opencv-python-headless numpy scipy

# 2. Install RTMPose (4th detection pass — uses GPU)
pip install mmengine mmcv mmdet mmpose

# 3. Install training dependencies
pip install transformers datasets pandas inflect scikit-learn matplotlib seaborn

# 4. Download RTMPose model checkpoint (~170 MB)
python3 models/download_rtmpose.py

# 5. Unpack data
cd /workspace
tar xzf landmarks.tar.gz && rm landmarks.tar.gz
mkdir -p data/raw_videos && tar xzf videos.tar.gz -C data/raw_videos/ && rm videos.tar.gz
```

---

## Phase 4: Extract (Vast.ai)

```bash
# 1. Start tmux (survives SSH disconnects)
tmux new -s pipeline

# 2. Run 4-pass extraction (MediaPipe x3 + RTMPose GPU)
#    Adjust workers based on CPU cores: 8 cores → 6, 12 cores → 10, 16 cores → 14
cd /workspace
python3 src/extract_do.py --workers 10
```

Expected: ~50k videos in ~1.5-2 hours on RTX 4090.

Skips existing .npy files automatically.

Monitor progress (second SSH session):
```bash
watch -n 10 "ls /workspace/ASL_landmarks_float16/*.npy | wc -l"
```

tmux controls: `Ctrl+B, D` to detach. `tmux attach -t pipeline` to reattach.

---

## Phase 5: Verify Extraction (Vast.ai)

```bash
# 1. Run comprehensive quality audit
python3 src/verify_extraction_quality.py

# What to check in the output:
#   - Tier A+B should be >60% (was 51% before improvements)
#   - high_jitter should be <10% (was 29.6% before smoothing)
#   - Sign variants section — shows classes with mixed signs (e.g. FALL = fall down + autumn)
#   - RTMPose rescue count — how many detections RTMPose saved that MediaPipe missed
#   - Duplicates — should be ~0 after label dedup fix

# 2. (Optional) Run relaxed pass on remaining failures
python3 src/extract_relaxed.py --workers 10

# 3. (Optional) Re-run audit to verify relaxed pass improvement
python3 src/verify_extraction_quality.py
```

---

## Phase 6: Train (Vast.ai, same instance)

```bash
# 1. Stage 1 — Isolated sign classification
#    RTX 4090: ~1-2 hours (early stops ~epoch 60-80)
python3 src/train_stage_1.py

# 2. Stage 2 — Continuous gloss recognition (CTC)
#    RTX 4090: ~30-60 min
python3 src/train_stage_2.py

# 3. Stage 3 — Gloss to English translation (Flan-T5)
#    RTX 4090: ~10-15 min
python3 src/train_stage_3.py
```

Monitor GPU: `watch -n 2 nvidia-smi` (second SSH session).

---

## Phase 7: Download Results (Mac)

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# 1. Download extracted landmarks (new + existing)
rsync -avz --progress -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/ASL_landmarks_float16/ ASL_landmarks_float16/

# 2. Download trained weights
rsync -avz --progress -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/weights/ weights/

# 3. Download quality audit results
scp -P <PORT> root@<VAST_IP>:/workspace/ASL_landmarks_float16/quality_audit.json ASL_landmarks_float16/

# 4. DESTROY the Vast.ai instance immediately
#    Go to https://vast.ai/console/instances/ → trash icon → confirm
```

---

## Phase 8: Verify Locally (Mac)

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

# 1. Regenerate manifest from downloaded .npy files
python3 src/extract.py

# 2. Run quality audit locally (optional — already done on Vast.ai)
python3 src/verify_extraction_quality.py

# 3. Run test suite
python3 test/SLT_test.py

# 4. Test camera inference
python3 src/camera_inference.py
```

---

## Script Reference

| # | Script | Where | Purpose |
|---|--------|-------|---------|
| 1 | `src/download_wlasl.py` | Mac | Download WLASL videos for matching classes |
| 2 | `src/download_msasl.py` | Mac | Download MSASL videos for matching classes |
| 3 | `src/download_signasl.py` | Mac | Download SignASL.org video clips |
| 4 | `src/download_other_sources.py` | Mac | YouTube search for missing classes |
| 5 | `models/download_rtmpose.py` | Vast.ai | Download RTMPose-WholeBody model |
| 6 | `src/extract_do.py` | Vast.ai | Main extraction (4-pass, no worker cap) |
| 7 | `src/verify_extraction_quality.py` | Both | Quality audit (jitter, bones, variants, tiers) |
| 8 | `src/extract_relaxed.py` | Vast.ai | Retry failed videos with relaxed thresholds |
| 9 | `src/train_stage_1.py` | Vast.ai | Train isolated sign classifier |
| 10 | `src/train_stage_2.py` | Vast.ai | Train CTC continuous recognition |
| 11 | `src/train_stage_3.py` | Vast.ai | Train Flan-T5 gloss→English |
| 12 | `src/extract.py` | Mac | Regenerate manifest (skips existing files) |
| 13 | `src/camera_inference.py` | Mac | Test real-time webcam inference |

---

## Time Estimates (RTX 4090 on Vast.ai)

| Phase | Time |
|-------|------|
| Phase 1: Download data | ~30-60 min (while doing other things) |
| Phase 2: Upload to Vast.ai | ~60-75 min |
| Phase 3: Setup instance | ~10-15 min |
| Phase 4: Extraction (50k videos) | ~1.5-2 hrs |
| Phase 5: Quality audit | ~5-10 min |
| Phase 6: Training (all 3 stages) | ~2-3 hrs |
| Phase 7: Download results | ~10 min |
| Phase 8: Local verification | ~5 min |
| **Total** | **~5-7 hrs** |

---

## Cost Estimate (Vast.ai RTX 4090, ~$0.35/hr)

| Item | Cost |
|------|------|
| Instance idle during upload | ~$0.40 |
| Extraction | ~$0.60 |
| Training | ~$0.80 |
| **Total** | **~$1.50-2.50** |
