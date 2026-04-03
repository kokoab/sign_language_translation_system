#!/bin/bash
# Train all ensemble streams on Vast.ai
# Run AFTER joint stream training is complete
#
# Usage: bash scripts/train_ensemble.sh

set -e

export TORCHDYNAMO_DISABLE=1
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))"):$LD_LIBRARY_PATH

DATA_PATH="ASL_landmarks_rtmlib"
MANIFEST_PATH="$DATA_PATH/manifest.json"

echo "============================================"
echo "SLT Ensemble Training — All Streams"
echo "============================================"
echo "Data: $DATA_PATH"
echo ""

# Use cleaned manifest if available
if [ -f "$DATA_PATH/manifest_cleaned.json" ]; then
    echo "Found cleaned manifest — copying as primary manifest"
    cp "$DATA_PATH/manifest.json" "$DATA_PATH/manifest_original.json"
    cp "$DATA_PATH/manifest_cleaned.json" "$DATA_PATH/manifest.json"
fi

# Train bone stream
echo ""
echo "========== BONE STREAM =========="
rm -f $DATA_PATH/ds_cache.pt
python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_bone \
    --stream bone

# Train velocity stream
echo ""
echo "========== VELOCITY STREAM =========="
rm -f $DATA_PATH/ds_cache.pt
python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_velocity \
    --stream velocity

# Train angle stream
echo ""
echo "========== ANGLE STREAM =========="
rm -f $DATA_PATH/ds_cache.pt
python src/train_stage_1.py \
    --data_path $DATA_PATH \
    --save_dir /workspace/output_angle \
    --stream angle

echo ""
echo "============================================"
echo "ALL STREAMS COMPLETE"
echo "============================================"
echo "Checkpoints:"
echo "  Joint:    /workspace/output_v2/best_model.pth"
echo "  Bone:     /workspace/output_bone/best_model.pth"
echo "  Velocity: /workspace/output_velocity/best_model.pth"
echo "  Angle:    /workspace/output_angle/best_model.pth"
