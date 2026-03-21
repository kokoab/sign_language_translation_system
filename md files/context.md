# Project: Conversational Sign Language Translator (ASL to English)
**Current Status:** All three pipeline stages are fully implemented and optimized for conversational context. Ready for final end-to-end extraction and training runs.

## Architecture Overview
The system is a 3-stage pipeline that translates live webcam video of American Sign Language (ASL) into natural, context-aware conversational English sentences.

### Stage 0: Extraction (`extract.py`)
- **Tool:** MediaPipe Hands (42 points: 21 per hand) + FaceMesh (5 reference points: nose, chin, forehead, ears), aligned to strict `0.80` confidence.
- **Format:** Optimized 1.5× oversampling, 384px resizing. Extracts `(X, Y, Z)` coordinates, then safely computes velocity, acceleration, and visibility masks.
- **Output:** Tensor shape `[32, 47, 10]`.

### Stage 1: Isolated Sign Classification (`train_stage_1.py`)
- **Architecture:** DS-GCN (Spatial Graph) + Transformer Encoder + Classifier.
- **Enhancements:** Dynamic Temporal Speed Warping (recomputing kinematics on the fly), Mixup augmentation, and 0.10 label smoothing.
- **Purpose:** Learns the spatial-temporal geometry of hands to classify isolated signs robustly.

### Stage 2: Continuous Sign Recognition (`train_stage_2.py`)
- **Architecture:** Frozen DS-GCN Encoder (from Stage 1) + BiLSTM + CTC Decoder.
- **Enhancements:** Synthetic transition frame injection (simulating real pauses/movement between signs) and segment boundary jitter.
- **Purpose:** Decodes variable-length sequences into ASL Glosses (e.g. `[HELLO, HOW, YOU]`).

### Stage 3: NLP Translation (`train_stage_3.py`)
- **Architecture:** HuggingFace `google/flan-t5-base` (250M parameter Seq2Seq).
- **Purpose:** Translates ASL Gloss into natural conversational English using previous dialogue history as context.
- **Data:** Trained on an enhanced 50k+ dataset (`generate_stage3_data_v2.py`) containing 20%+ questions, single-word glosses, paraphrases, and multi-turn dialogue pairs.

### Deploy
- `camera_inference.py` — real-time webcam, integrates N-Gram LM (`build_language_model.py`) for CTC beam search rescoring, hand-count priors, multi-turn conversation memory.

---

## Inter-Stage Contracts (Inviolable)
| Boundary | Shape | Dtype | Rule |
|----------|-------|-------|------|
| Stage 0 → 1 | `[B, 32, 47, 10]` | float32 | confidence mask ≥ 0.80; zero-pad occluded hands/face |
| Stage 1 → 2 | `[B, T, hidden]` | float32 | encoder frozen |
| Stage 2 → 3 | `List[str]` glosses | — | strip blank tokens (idx 0) before passing |
| Stage 3 → UI | `str` translation | — | cap `max_new_tokens`; never block UI thread |

**CTC Rules:** Blank = idx 0. PAD ≠ blank. `input_lengths` / `target_lengths` computed before padding. After temporal augmentation, verify `input_lengths >= target_lengths`.

---

## Master Plan — Improvement Priorities
*(Consensus: data first, then training alignment, then inference.)*

| Priority | Task | Impact | Phase |
|----------|------|--------|-------|
| **P0** | Stage 3 dataset: 15–20% questions, single-word glosses, paraphrases | HIGH | 2 |
| **P0** | Stage 2 transition frame injection (recompute kinematics from XYZ) | HIGH | 3 |
| **P1** | MediaPipe confidence alignment (0.80 train + inference) | MEDIUM | 1 |
| **P1** | Temporal speed augmentation (warp XYZ first, recompute kinematics) | MEDIUM | 3 |
| **P1** | Dialogue context window in camera_inference | HIGH | 5 |
| **P2** | N-gram LM for beam search (diverse training data, not templates only) | MEDIUM | 4 |
| **P2** | Segment boundary jitter | LOW | 3 |

**Critical Constraints (Accuracy Critique):**
1. Temporal augmentation: warp XYZ first, then **recompute kinematics** — never warp the 10-channel tensor directly.
2. Mirror augmentation: swap hand indices (0–20 ↔ 21–41) with X-flip.
3. Transition injection: maintain clip/target alignment for CTC.
4. LM: train on diverse gloss sequences, not templates only.
5. Stage 2: ensure `T % 32 == 0` or handle variable lengths in `forward()`.

---

## Immediate Next Step
Run the full training pipeline, then deploy via `camera_inference.py`. For detailed Phase 1–6 specs and code snippets, see `md files/MASTER_IMPLEMENTATION_PLAN.md`.
