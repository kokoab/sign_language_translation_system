# SLT — Sign Language Translator (Agent Brief)

**One-line:** 4-stage pipeline: MediaPipe hands → DS-GCN+Transformer (Stage 1) → BiLSTM+CTC (Stage 2) → Flan-T5 (Stage 3) → real-time camera inference.

## Quick reference

| Stage | File | Purpose |
|-------|------|---------|
| 0 | `src/extract.py` | MediaPipe landmark extraction → `[32, 42, 10]` |
| 1 | `src/train_stage_1.py` | Isolated sign classification (DS-GCN + Transformer) |
| 2 | `src/train_stage_2.py` | Continuous recognition (frozen encoder + BiLSTM + CTC) |
| 3 | `src/train_stage_3.py` | Gloss → English (Flan-T5, conversational) |
| Deploy | `src/camera_inference.py` | Live webcam, LM rescoring, dialogue memory |
| Data | `src/generate_stage3_data_v2.py` | Stage 3 dataset; `src/build_language_model.py` | N-gram LM |

## Context files (use these first)

- **`md files/context.md`** — Architecture, contracts, priorities, critical constraints. Use instead of loading full training files.
- **`md files/MASTER_IMPLEMENTATION_PLAN.md`** — Detailed phases + code snippets. Load only when implementing phased changes.

## Critical rules

- Stage 0→1: confidence mask ≥ 0.80; `[32, 42, 10]` output.
- CTC: blank idx 0; `input_lengths`/`target_lengths` before padding.
- Temporal augmentation: warp XYZ first, **recompute kinematics** — never warp 10-channel tensor.
- Mirror: swap hand indices (0–20 ↔ 21–41) with X-flip.

## Tests

- `test/SLT_test.py` — Stage 2 batch WER.
- `test/test_offline_pipeline.py` — E2E Stage 2 → Stage 3.
