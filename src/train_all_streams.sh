#!/bin/bash
# SLT Stage 1 — Train all 6 streams in parallel on 2 GPUs
# Usage: bash src/train_all_streams.sh

cd /workspace

echo "=== Starting 6-stream ensemble training ==="
echo "GPU 0: joint -> bone_motion"
echo "GPU 1: bone -> velocity -> angle -> angle_motion"
echo ""

# GPU 0: joint (main) then bone_motion
CUDA_VISIBLE_DEVICES=0 python src/train_stage_1.py \
    --stream joint --d_model 384 --num_layers 6 --epochs 200 \
    --batch_size 256 --save_dir output_joint --balanced_softmax && \
CUDA_VISIBLE_DEVICES=0 python src/train_stage_1.py \
    --stream bone_motion --d_model 192 --num_layers 4 --epochs 100 \
    --patience 30 --save_dir output_bone_motion --balanced_softmax \
> gpu0.log 2>&1 &

# GPU 1: bone -> velocity -> angle -> angle_motion
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream bone --d_model 256 --num_layers 4 --epochs 150 \
    --patience 30 --save_dir output_bone --balanced_softmax --arcface && \
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream velocity --d_model 256 --num_layers 4 --epochs 120 \
    --patience 30 --save_dir output_velocity --balanced_softmax --arcface && \
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream angle --d_model 192 --num_layers 4 --epochs 100 \
    --patience 30 --save_dir output_angle --balanced_softmax && \
CUDA_VISIBLE_DEVICES=1 python src/train_stage_1.py \
    --stream angle_motion --d_model 192 --num_layers 4 --epochs 80 \
    --patience 30 --save_dir output_angle_motion --balanced_softmax \
> gpu1.log 2>&1 &

echo "Both GPUs running. Logs: gpu0.log, gpu1.log"
echo "Monitor with: tail -f gpu0.log gpu1.log"
echo ""

# Wait for both to finish
wait

echo ""
echo "=== All streams complete. Running ensemble evaluation ==="
python src/ensemble_eval.py \
    --streams joint:output_joint/best_model.pth,bone:output_bone/best_model.pth,velocity:output_velocity/best_model.pth,bone_motion:output_bone_motion/best_model.pth,angle:output_angle/best_model.pth,angle_motion:output_angle_motion/best_model.pth \
    --optimize_weights --test

echo "=== Done! ==="
