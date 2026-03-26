#!/bin/bash
# SLT Full Pipeline Training: Stage 1 (6-stream) + Stage 2 (CTC) + Stage 3 (Translation)
# Usage: bash src/train_all_stages.sh [VAST_API_KEY]
# Runs on 2x RTX 4070S/4080, ~4-5 hours total.

set +e  # Don't exit on errors (background processes return non-zero on kill)
cd /workspace

VAST_API_KEY="${1:-}"

# ================================================================
#  ENVIRONMENT FIXES (Vast.ai container quirks)
# ================================================================
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true
rm -f /workspace/ASL_landmarks_float16/manifest.json 2>/dev/null || true  # Use root manifest only

echo "============================================================"
echo "  SLT Full Pipeline Training"
echo "  Stage 1: 6-stream ensemble (~3h)"
echo "  Stage 2: Transformer + CTC (~1.5h)"
echo "  Stage 3: Flan-T5 translation (~1h)"
echo "============================================================"

# Verify data
NPY_COUNT=$(find /workspace/ASL_landmarks_float16/ -name "*.npy" 2>/dev/null | wc -l)
echo "Data: ${NPY_COUNT} .npy files"
if [ "$NPY_COUNT" -lt 50000 ]; then
    echo "ERROR: Not enough data files. Expected 56000+, found ${NPY_COUNT}"
    exit 1
fi

# ================================================================
#  STAGE 1: Multi-Stream Ensemble (2 GPUs in parallel)
# ================================================================
echo ""
echo "[STAGE 1] Starting 6-stream ensemble training..."

# GPU 0: Joint (300ep, d=384, batch=128) then Bone Motion (150ep, d=192)
(CUDA_VISIBLE_DEVICES=0 python src/train_stage_1.py \
    --stream joint --d_model 384 --num_layers 6 --epochs 300 \
    --batch_size 128 --patience 50 --save_dir output_joint --balanced_softmax && \
CUDA_VISIBLE_DEVICES=0 python src/train_stage_1.py \
    --stream bone_motion --d_model 192 --num_layers 4 --epochs 150 \
    --batch_size 128 --patience 30 --save_dir output_bone_motion --balanced_softmax \
) > gpu0_stage1.log 2>&1 &
PID_GPU0=$!

# GPU 1: Bone (150ep, d=256, ArcFace) then Velocity (120ep, ArcFace) then Angle (100ep) then Angle Motion (80ep)
(CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream bone --d_model 256 --num_layers 4 --epochs 150 \
    --batch_size 128 --patience 30 --save_dir output_bone --balanced_softmax --arcface && \
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream velocity --d_model 256 --num_layers 4 --epochs 120 \
    --batch_size 128 --patience 30 --save_dir output_velocity --balanced_softmax --arcface && \
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream angle --d_model 192 --num_layers 4 --epochs 100 \
    --batch_size 128 --patience 30 --save_dir output_angle --balanced_softmax && \
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream angle_motion --d_model 192 --num_layers 4 --epochs 80 \
    --batch_size 128 --patience 30 --save_dir output_angle_motion --balanced_softmax \
) > gpu1_stage1.log 2>&1 &
PID_GPU1=$!

echo "  GPU 0 PID: $PID_GPU0 | GPU 1 PID: $PID_GPU1"
echo "  Monitor: tail -f gpu0_stage1.log gpu1_stage1.log"

wait $PID_GPU0 $PID_GPU1
echo "[STAGE 1] All streams complete."

# Ensemble evaluation
echo "[STAGE 1] Running ensemble evaluation..."
python src/ensemble_eval.py \
    --streams joint:output_joint/best_model.pth,bone:output_bone/best_model.pth,velocity:output_velocity/best_model.pth,bone_motion:output_bone_motion/best_model.pth,angle:output_angle/best_model.pth,angle_motion:output_angle_motion/best_model.pth \
    --optimize_weights --test --output ensemble_results.json \
    2>&1 | tee ensemble_eval.log
echo "[STAGE 1] DONE."

# ================================================================
#  STAGE 2 + STAGE 3 (parallel on 2 GPUs)
# ================================================================
echo ""
echo "[STAGE 2+3] Starting Stage 2 (GPU 0) and Stage 3 (GPU 1) in parallel..."

(CUDA_VISIBLE_DEVICES=0 python src/train_stage_2.py \
    --stage1_ckpt output_joint/best_model.pth \
) > gpu0_stage2.log 2>&1 &
PID_S2=$!

(CUDA_VISIBLE_DEVICES=1 python src/train_stage_3.py \
) > gpu1_stage3.log 2>&1 &
PID_S3=$!

echo "  Stage 2 PID: $PID_S2 | Stage 3 PID: $PID_S3"
wait $PID_S2 $PID_S3
echo "[STAGE 2+3] DONE."

# ================================================================
#  PACKAGE RESULTS
# ================================================================
echo ""
echo "[PACKAGE] Packaging all results..."

tar czf /workspace/slt_all_results.tar.gz \
    output_joint/best_model.pth output_joint/history.json \
    output_bone/best_model.pth output_bone/history.json \
    output_velocity/best_model.pth output_velocity/history.json \
    output_bone_motion/best_model.pth output_bone_motion/history.json \
    output_angle/best_model.pth output_angle/history.json \
    output_angle_motion/best_model.pth output_angle_motion/history.json \
    output/stage2_best_model.pth output/stage2_history.json \
    output/asl_flan_t5_results/ output/slt_conversational_t5_model/ \
    ensemble_results.json \
    gpu0_stage1.log gpu1_stage1.log gpu0_stage2.log gpu1_stage3.log \
    ensemble_eval.log \
    2>/dev/null || echo "  Warning: some files missing from archive"

echo "  Results: /workspace/slt_all_results.tar.gz"

# ================================================================
#  AUTO-PAUSE INSTANCE
# ================================================================
if [ -n "$VAST_API_KEY" ]; then
    echo "[AUTO-PAUSE] Pausing Vast.ai instance..."
    pip install vastai -q 2>/dev/null
    vastai set api-key "$VAST_API_KEY" 2>/dev/null
    INSTANCE_ID="${VAST_CONTAINERLABEL:-$(cat /etc/hostname 2>/dev/null || hostname)}"
    vastai stop instance "$INSTANCE_ID" 2>/dev/null || echo "  Could not auto-pause."
else
    echo "[INFO] No API key. Pause manually from vast.ai console."
fi

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE"
echo "  Download: scp -P PORT root@IP:/workspace/slt_all_results.tar.gz ./"
echo "============================================================"
