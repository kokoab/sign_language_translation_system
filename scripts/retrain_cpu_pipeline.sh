#!/bin/bash
set -e

echo "============================================"
echo "  SLT CPU-Matched Retrain Pipeline"
echo "  Started: $(date)"
echo "  Old models will NOT be overwritten."
echo "============================================"

cd /workspace/SLT

# Step 1: Re-extract isolated signs with CPU ONNX
echo ""
echo "[Step 1/4] Re-extracting isolated signs (CPU ONNX, 6 workers)..."
echo "  Input:  data/raw_videos/ASL VIDEOS"
echo "  Output: ASL_landmarks_cpu (NEW - separate from ASL_landmarks_v2)"

python /workspace/extract_cpu.py \
    --input "data/raw_videos/ASL VIDEOS" \
    --output ASL_landmarks_cpu \
    --workers 6 \
    --resume

echo "[Step 1 done] $(ls ASL_landmarks_cpu/*.npy 2>/dev/null | wc -l) files"

# Step 1a: Generate manifest.json from extracted files
echo "  Generating manifest.json..."
python3 -c "
import os, json
npy_dir = 'ASL_landmarks_cpu'
manifest = {}
for f in sorted(os.listdir(npy_dir)):
    if not f.endswith('.npy'): continue
    parts = f.replace('.npy', '').rsplit('_', 2)
    if len(parts) >= 3:
        manifest[f] = parts[0]
    elif len(parts) == 2:
        manifest[f] = parts[0]
with open(os.path.join(npy_dir, 'manifest.json'), 'w') as fp:
    json.dump(manifest, fp)
print(f'Manifest: {len(manifest)} files, {len(set(manifest.values()))} classes')
"

# Step 1b: Re-extract phrases with CPU ONNX
echo ""
echo "[Step 1b/4] Re-extracting phrases (CPU ONNX)..."
echo "  Input:  data/raw_videos/phrases"
echo "  Output: ASL_phrases_cpu (NEW)"

python /workspace/extract_phrases_cpu.py data/raw_videos/phrases ASL_phrases_cpu

# Generate phrase manifest
echo "  Generating phrase manifest.json..."
python3 -c "
import os, json
npy_dir = 'ASL_phrases_cpu'
PHRASE_GLOSSES = {
    'GOOD_MORNING': 'GOOD MORNING',
    'HELLO_HOW_YOU': 'HELLO HOW YOU',
    'PLEASE_HELP_I': 'PLEASE HELP I',
    'PLEASE_HELP_ME': 'PLEASE HELP I',
    'SORRY_I_LATE': 'SORRY I LATE',
    'MY_NAME': 'MY NAME',
    'YESTERDAY_TEACHER_MEET': 'YESTERDAY TEACHER MEET',
    'THANKYOU_FRIEND': 'THANKYOU FRIEND',
    'TOMORROW_SCHOOL_GO': 'TOMORROW SCHOOL GO',
    'I_WANT_FOOD': 'I WANT FOOD',
}
manifest = {}
for f in sorted(os.listdir(npy_dir)):
    if not f.endswith('.npy'): continue
    for phrase_key, gloss_str in PHRASE_GLOSSES.items():
        if f.startswith(phrase_key):
            manifest[f] = gloss_str
            break
with open(os.path.join(npy_dir, 'manifest.json'), 'w') as fp:
    json.dump(manifest, fp)
print(f'Phrase manifest: {len(manifest)} files, {len(set(manifest.values()))} phrases')
"

echo "[Step 1b done] $(ls ASL_phrases_cpu/*.npy 2>/dev/null | wc -l) phrase files"

# Step 2: Retrain Stage 1 v14
echo ""
echo "[Step 2/4] Retraining Stage 1 v14..."
echo "  Data:   ASL_landmarks_cpu"
echo "  Output: /workspace/output_v14_cpu (NEW - separate from output_v14)"

python src/train_v14.py \
    --data_path ASL_landmarks_cpu \
    --save_dir /workspace/output_v14_cpu \
    --epochs 150 \
    --lr 3e-4 \
    --batch_size 256 \
    --accum_steps 4 \
    --patience 25

echo "[Step 2 done]"

# Step 3: Retrain Stage 2
echo ""
echo "[Step 3/4] Retraining Stage 2 v14..."
echo "  Data:     ASL_landmarks_cpu"
echo "  Stage1:   /workspace/output_v14_cpu/best_model.pth"
echo "  Phrases:  ASL_phrases_cpu"
echo "  Output:   /workspace/output_stage2_v14_cpu (NEW)"

python src/train_stage_2.py \
    --data_path ASL_landmarks_cpu \
    --stage1_ckpt /workspace/output_v14_cpu/best_model.pth \
    --save_dir /workspace/output_stage2_v14_cpu \
    --phrase_data ASL_phrases_cpu \
    --batch_size 16 \
    --no_cr_ctc \
    --epochs 60

echo "[Step 3 done]"

# Step 4: Backup
echo ""
echo "[Step 4/4] Backing up to S3..."
aws s3 cp /workspace/output_v14_cpu/best_model.pth s3://slt-dataset-534409838815-us-east-1-an/models/v14_cpu_best.pth
aws s3 cp /workspace/output_v14_cpu/history.json s3://slt-dataset-534409838815-us-east-1-an/models/v14_cpu_history.json
aws s3 cp /workspace/output_stage2_v14_cpu/stage2_best_model.pth s3://slt-dataset-534409838815-us-east-1-an/models/stage2_v14_cpu_best.pth

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE: $(date)"
echo ""
echo "  Old models (preserved):"
echo "    /workspace/output_v14/best_model.pth"
echo "    /workspace/output_stage2_v14/stage2_best_model.pth"
echo ""
echo "  New CPU-matched models:"
echo "    /workspace/output_v14_cpu/best_model.pth"
echo "    /workspace/output_stage2_v14_cpu/stage2_best_model.pth"
echo ""
echo "  Download to Mac:"
echo "    aws s3 cp s3://slt-dataset-534409838815-us-east-1-an/models/v14_cpu_best.pth models/output_v14/best_model.pth"
echo "    aws s3 cp s3://slt-dataset-534409838815-us-east-1-an/models/stage2_v14_cpu_best.pth models/output_stage2_v13/stage2_best_model.pth"
echo "============================================"
