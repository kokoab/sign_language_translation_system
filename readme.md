# SLT - Sign Language Translator

This repo is now organized around the **v16 Squeezeformer pipeline** for ASL landmark recognition and continuous gloss decoding.

The older DS-GCN/MediaPipe code is still preserved because some tests, scripts, and historical experiments depend on it, but new work should start from the v16 paths below.

## Current Pipeline

| Stage | Current path | Purpose |
| --- | --- | --- |
| Stage 0 | `active/v16/extract_v16.py` | Apple Vision-style landmark extraction and v16 feature formatting |
| Stage 1 | `active/v16/train_stage_1_v16.py` | Isolated sign classification with Squeezeformer |
| Stage 2 | `active/v16/train_stage_2_v16_fixed.py` | Continuous recognition with CTC and anti-template-memorization fixes |
| Inference | `active/v16/inference_v16.py` | Unified video inference with trimming, sliding windows, TTA, and CTC beam search |
| Compatibility | `src_v16/*.py` | Thin wrappers for existing scripts and mobile export tools |

See `CURRENT_PIPELINE.md` for commands and cleanup notes.

## Legacy Areas

- `src/` keeps the older DS-GCN/Transformer implementation and desktop/demo utilities.
- `legacy/copies/` contains duplicate `* copy.py` files moved out of the active tree.
- `docs/archive/` is for older planning and review notes that may describe pre-v16 architecture.
- `artifacts/` is for generated metrics, reports, charts, and exported outputs.
- Large local datasets live under `data/local/`; root dataset names are compatibility symlinks.
- Model assets live under `artifacts/model_assets/`; `models` and `weights` are compatibility symlinks.
- Batch extraction implementations live under `scripts/extraction/`; root extractor files are compatibility wrappers.

## Quick Commands

```bash
# v16 inference through the canonical path
KMP_DUPLICATE_LIB_OK=TRUE python active/v16/inference_v16.py path/to/video.mp4

# v16 inference through the compatibility path
KMP_DUPLICATE_LIB_OK=TRUE python src_v16/inference_v16.py path/to/video.mp4

# v16 Stage 1 training
python active/v16/train_stage_1_v16.py \
  --data_path src_v16/ASL_landmarks_v16 \
  --save_dir models/output_v16_d384_aug \
  --manifest models/manifest_v16.json

# v16 Stage 2 training
python active/v16/train_stage_2_v16_fixed.py \
  --data_path src_v16/ASL_landmarks_v16 \
  --phrase_data src_v16/ASL_phrases_v16 \
  --stage1_ckpt src_v16/output_v16_d384/best_model.pth \
  --manifest models/manifest_v16.json
```

## Notes

- Prefer `active/v16/` for code changes.
- Prefer `src_v16/` only when maintaining older scripts that already import it.
- Prefer canonical storage folders for new outputs: `data/local/`, `artifacts/generated/`, and `artifacts/reports/`.
