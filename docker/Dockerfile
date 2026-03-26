# SLT Pipeline — RTMW extraction (via rtmlib) + Stage 2 CTC + Stage 3 T5
# Single image, two entrypoints via docker-compose
#
# Build:  docker compose build
# Run:    docker compose run extract /app/input/video.mp4
#         docker compose run inference /app/npy/

FROM continuumio/miniconda3:24.7.1-0

ENV OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/opt/conda/lib

# System deps via conda (no apt — avoids Docker Desktop signing bugs)
RUN conda config --set remote_read_timeout_secs 600 && \
    conda config --set remote_max_retries 5
RUN conda install -y -c conda-forge python=3.10 numpy=1.26.4 libstdcxx-ng>=12
RUN conda install -y -c conda-forge scipy=1.13.1 ffmpeg
RUN conda install -y -c conda-forge opencv=4.10.0 && conda clean -afy

# rtmlib — runs RTMW via ONNX, no mmcv/mmpose/mmdet needed
RUN pip install --no-cache-dir rtmlib==0.0.15 onnxruntime==1.20.1

# PyTorch CPU (for Stage 1 + Stage 2 inference)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Transformers for Stage 3 T5
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    sentencepiece==0.2.0

# Force numpy back
RUN pip install --no-cache-dir numpy==1.26.4

# Verify
RUN python -c "\
import numpy; print('numpy:', numpy.__version__); \
import cv2; print('cv2:', cv2.__version__); \
import torch; print('torch:', torch.__version__); \
from rtmlib import Wholebody; print('rtmlib: OK'); \
import onnxruntime; print('onnxruntime:', onnxruntime.__version__); \
import transformers; print('transformers:', transformers.__version__); \
print('ALL PASSED')"

WORKDIR /app

# Copy source
COPY src/extract.py /app/src/extract.py
COPY models/hand_landmarker.task /app/models/hand_landmarker.task
COPY docker/extract_rtmlib.py /app/extract_rtmlib.py
COPY docker/run_inference.py /app/run_inference.py
