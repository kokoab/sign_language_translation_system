# SLT — Conversational Sign Language Translator

Project instructions for Claude. Read this before diving into code.

## Efficiency router (always)

1. Ask up to 3 clarifying questions if requirements are ambiguous.
2. Classify the request:
   - **Planning** (contracts, interfaces, trade-offs, “what should we do?”): use `cludev2.md` — architecture/plan only, no code.
   - **Implementation** (code edits, bug fixes, “implement this plan”): use `cludev3.md` — surgical edits, run verification.
3. If the user explicitly provides an implementation plan, skip planning and go straight to implementation.

## Token saving

- Prefer `md files/context.md` over loading large plans.
- Only load `md files/MASTER_IMPLEMENTATION_PLAN.md` when the user explicitly requests phased implementation.
- Don’t paste large files; cite file paths/symbol names instead.

---

## Architecture

4-stage pipeline translating ASL webcam video → English:

1. **Stage 0** (`src/extract.py`): MediaPipe Hands + FaceMesh → `[32, 47, 10]` tensors (42 hand + 5 face landmarks, XYZ + vel + acc + mask)
2. **Stage 1** (`src/train_stage_1.py`): DS-GCN + Transformer → isolated sign classification
3. **Stage 2** (`src/train_stage_2.py`): Frozen encoder + BiLSTM + CTC → gloss sequences
4. **Stage 3** (`src/train_stage_3.py`): Flan-T5 → gloss → natural English (context-aware)
5. **Deploy** (`src/camera_inference.py`): Real-time webcam, N-gram LM rescoring, dialogue memory

## Context loading

- **For architecture/contracts/priorities:** Read `md files/context.md` — no need for MASTER_IMPLEMENTATION_PLAN unless implementing phased changes.
- **For phased implementation details:** Read `md files/MASTER_IMPLEMENTATION_PLAN.md`.

## Inviolable constraints

| Rule | Why |
|------|-----|
| Temporal augmentation: warp XYZ first, recompute kinematics | Warping 10-ch tensor corrupts vel/acc |
| Mirror: swap hand indices (0–20 ↔ 21–41) with X-flip | Preserves hand identity |
| CTC blank = idx 0; PAD ≠ blank | Avoid target alignment errors |
| Transition injection: maintain clip/target alignment | CTC requires correct lengths |
| Detection conf ≥ 0.70 / tracking ≥ 0.65 train + inference | Align train/inference distribution (RTMPose uses 0.30 internally) |

## File map

- `src/extract.py` — Extraction
- `src/train_stage_1.py`, `train_stage_2.py`, `train_stage_3.py` — Training
- `src/camera_inference.py` — Inference
- `src/generate_stage3_data_v2.py` — Stage 3 data; `build_language_model.py` — LM
- `test/SLT_test.py`, `test/test_offline_pipeline.py` — Tests
