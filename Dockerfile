FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libportaudio2 ffmpeg \
    tmux git wget rsync openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy
RUN pip install --no-cache-dir numpy==1.26.4

# rtmlib for extraction (replaces mmpose dependency hell)
RUN pip install --no-cache-dir rtmlib==0.0.15 onnxruntime-gpu==1.20.1

# MediaPipe fallback
RUN pip install --no-cache-dir --no-deps mediapipe==0.10.14 && \
    pip install --no-cache-dir absl-py attrs flatbuffers "protobuf>=3.20,<5" sounddevice

# OpenCV headless
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# ML + training packages
RUN pip install --no-cache-dir \
    scipy==1.13.1 \
    transformers==4.46.3 \
    datasets==2.21.0 \
    pandas==2.2.2 \
    inflect==7.3.1 \
    scikit-learn==1.5.1 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    sentencepiece==0.2.0

# Force numpy back
RUN pip install --no-cache-dir numpy==1.26.4

# Verify
RUN python -c "\
import numpy; print('numpy:', numpy.__version__); \
import cv2; print('opencv:', cv2.__version__); \
import torch; print('torch:', torch.__version__); \
print('CUDA:', torch.cuda.is_available()); \
from rtmlib import Wholebody; print('rtmlib: OK'); \
import transformers; print('transformers:', transformers.__version__); \
print('ALL CHECKS PASSED')"

WORKDIR /workspace
