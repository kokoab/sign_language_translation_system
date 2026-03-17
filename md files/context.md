# Project: Continuous Sign Language Translator (ASL to English)
**Current Status:** All three pipeline stages are fully trained and tested successfully. Moving to real-time inference integration.

## Architecture Overview
The system is a 3-stage pipeline that translates live webcam video of American Sign Language (ASL) into grammatically correct English sentences.

### Stage 0: Extraction (`extract.py`)
- **Tool:** MediaPipe Holistic (42 points: 21 per hand).
- **Format:** 32-frame windows, extracting `(X, Y, Z)` coordinates, velocity, acceleration, and visibility masks.
- **Output:** Tensor shape `[32, 42, 10]`.

### Stage 1: Isolated Sign Classification (`train_stage_1.py`)
- **Architecture:** DS-GCN (Spatial Graph Convolution) + Transformer Encoder + Linear Classifier.
- **Purpose:** Learns the spatial geometry of hands to classify isolated signs.
- **Status:** Trained and saved weights.

### Stage 2: Continuous Sign Recognition (`train_stage_2.py` / `SLT_test.py`)
- **Architecture:** Frozen DS-GCN (loaded from Stage 1) + BiLSTM (Temporal) + CTC Decoder.
- **Purpose:** Takes variable-length frame sequences and decodes them into a sequence of ASL Glosses (e.g., `[YESTERDAY, DOCTOR, DRIVE, CAR]`).
- **Status:** Trained, tested, and validated using Word Error Rate (WER). Checkpoints saved as `slt_stage2_best.pth`.

### Stage 3: NLP Translation (`train_stage_3.py` / `stage3_test.py`)
- **Architecture:** HuggingFace `t5-small` (Seq2Seq Transformer).
- **Purpose:** Translates raw ASL Gloss into natural, conjugated English (e.g., `YESTERDAY DOCTOR DRIVE CAR` -> `"Yesterday, the doctor drove the car."`).
- **Data:** Fine-tuned on a heavily structured, synthetically generated 30,000-sentence dataset (`generate_stage3_data.py`).
- **Status:** Trained and evaluated (100% exact match on test cases). Saved as a local HuggingFace model in `weights/slt_final_t5_model`.

## Immediate Next Step
Combine extraction, Stage 2 (PyTorch), and Stage 3 (Transformers) into a single `realtime_inference.py` script that reads from a webcam (OpenCV), runs the sliding window inference, decodes the CTC gloss, and translates it to English via T5.