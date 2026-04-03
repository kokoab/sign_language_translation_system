#!/bin/bash
# Full pipeline: clean data → retrain joint → train ensemble → SWA
# Run on Vast.ai after initial joint training is complete
#
# Usage: bash scripts/run_pipeline.sh

set -e

export TORCHDYNAMO_DISABLE=1
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))"):$LD_LIBRARY_PATH

DATA_PATH="ASL_landmarks_rtmlib"

echo "============================================"
echo "PHASE 1: DATA CLEANING"
echo "============================================"

python scripts/clean_labels.py \
    --checkpoint /workspace/output_v2/best_model.pth \
    --data_path $DATA_PATH \
    --output $DATA_PATH/manifest_cleaned.json \
    --remove_pct 3.0

# Replace manifest with cleaned version
cp $DATA_PATH/manifest.json $DATA_PATH/manifest_original.json
cp $DATA_PATH/manifest_cleaned.json $DATA_PATH/manifest.json
echo "Manifest replaced with cleaned version"

echo ""
echo "============================================"
echo "PHASE 2: RETRAIN JOINT ON CLEAN DATA"
echo "============================================"

rm -f $DATA_PATH/ds_cache.pt

python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_clean_joint \
    --stream joint

echo ""
echo "============================================"
echo "PHASE 3: TRAIN ENSEMBLE STREAMS"
echo "============================================"

# Bone
rm -f $DATA_PATH/ds_cache.pt
python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_clean_bone \
    --stream bone

# Velocity
rm -f $DATA_PATH/ds_cache.pt
python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_clean_velocity \
    --stream velocity

# Angle
rm -f $DATA_PATH/ds_cache.pt
python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_clean_angle \
    --stream angle

echo ""
echo "============================================"
echo "PHASE 4: TRAIN STAGE 2 CTC"
echo "============================================"

python src/train_stage_2.py \
    --data_path $DATA_PATH \
    --stage1_ckpt /workspace/output_clean_joint/best_model.pth \
    --save_dir /workspace/output_clean_stage2

echo ""
echo "============================================"
echo "ALL PHASES COMPLETE"
echo "============================================"
echo ""
echo "Checkpoints to download:"
echo "  Joint:    /workspace/output_clean_joint/best_model.pth"
echo "  Bone:     /workspace/output_clean_bone/best_model.pth"
echo "  Velocity: /workspace/output_clean_velocity/best_model.pth"
echo "  Angle:    /workspace/output_clean_angle/best_model.pth"
echo "  Stage 2:  /workspace/output_clean_stage2/best_model.pth"
echo "  Manifest: $DATA_PATH/manifest.json (cleaned)"
