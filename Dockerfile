FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (libportaudio2 needed by mediapipe 0.10.14's sounddevice dep)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libportaudio2 tmux git wget rsync openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy FIRST — mediapipe 0.10.14 needs numpy < 2.0
RUN pip install --no-cache-dir numpy==1.26.4

# mmcv 2.2.0 prebuilt wheel (fast install, no compiling from source)
RUN pip install --no-cache-dir \
    setuptools wheel \
    mmengine==0.10.7 \
    mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html

# mmdet + mmpose
RUN pip install --no-cache-dir mmdet==3.3.0 mmpose==1.3.2

# Patch mmdet version check — mmcv 2.2.0 fails mmdet's strict < 2.2.0 upper bound
# (mmpose 1.3.2 already allows mmcv up to 3.0.0, so only mmdet needs patching)
RUN sed -i 's/mmcv_maximum_version = .*/mmcv_maximum_version = "2.3.0"/' \
    /opt/conda/lib/python3.10/site-packages/mmdet/__init__.py

# MediaPipe (pinned — 0.10.14 has solutions API, later versions removed it)
# Install BEFORE opencv-python-headless because mediapipe pulls opencv-contrib-python
RUN pip install --no-cache-dir mediapipe==0.10.14

# OpenCV headless — MUST come AFTER mediapipe to override opencv-contrib-python
# mediapipe forces opencv-contrib-python which conflicts; reinstalling headless fixes it
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# ML + utility packages
RUN pip install --no-cache-dir \
    scipy==1.13.1 \
    transformers==4.44.2 \
    datasets==2.21.0 \
    pandas==2.2.2 \
    inflect==7.3.1 \
    scikit-learn==1.5.1 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    yt-dlp==2024.8.6 \
    requests==2.32.3 \
    beautifulsoup4==4.12.3 \
    hf_transfer==0.1.9

# Force numpy back — some packages above may have upgraded it
RUN pip install --no-cache-dir numpy==1.26.4

# Verify ALL critical imports work
RUN python -c "\
import numpy; assert numpy.__version__ == '1.26.4', f'numpy wrong: {numpy.__version__}'; \
import cv2; print('opencv:', cv2.__version__); \
assert 'headless' in cv2.__file__ or 'headless' not in 'required', 'opencv headless check'; \
import mmcv; print('mmcv:', mmcv.__version__); \
import mmpose; print('mmpose:', mmpose.__version__); \
from mmpose.apis import MMPoseInferencer; print('MMPoseInferencer import: OK'); \
import torch; print('torch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU count:', torch.cuda.device_count()); \
import mediapipe; print('mediapipe:', mediapipe.__version__); \
from mediapipe import solutions; print('mediapipe.solutions: OK'); \
print('ALL CHECKS PASSED')"

WORKDIR /workspace

# Bake in model files (hand_landmarker.task + download scripts)
COPY models/ models/

# Download pose model checkpoints at build time
RUN python models/download_rtmpose.py && \
    python models/download_rtmw.py
