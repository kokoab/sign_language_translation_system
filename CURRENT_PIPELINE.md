# Current SLT Pipeline

## Summary

The project is in a controlled v16-to-v17 transition:

- **All new dataset extraction uses v17 Apple Vision.** It is orientation-safe,
  confidence-aware, schema-versioned, and intended for the new 100-sign ASL Citizen
  baseline. Apple won a controlled full-corpus comparison against MediaPipe 0.50 by
  93.12% versus 89.95% validation top-1 and remains the selected extractor.
- **v17 now has a research Stage 1 candidate** at 93.12% on the official five-signer
  validation split. Its official test and portrait-iPhone accuracy remain sealed or
  unmeasured, so it is not yet a production mobile claim.
- v16 retains the existing Stage 2/inference path. v16 checkpoints and v17 feature
  archives are intentionally incompatible.

The older DS-GCN pipeline remains in `src/` only for compatibility and historical
reference. Read `PROJECT_GROUND_TRUTH.md` for the timestamped state and evidence.

## Active code

| Area | Canonical path | Compatibility path |
| --- | --- | --- |
| New extraction/schema/audit | `active/v17/` | `src_v17/extract_v17.py` |
| v17 Stage 1 model | `active/v17/model_v17.py` | `src_v17/model_v17.py` |
| v17 Stage 1 training | `active/v17/train_stage_1_v17.py` | `src_v17/train_stage_1_v17.py` |
| v16 model definitions | `active/v16/model_v16.py` | `src_v16/model_v16.py` |
| Old checkpoint extraction | `active/v16/extract_v16.py` | `src_v16/extract_v16.py` |
| Inference CLI | `active/v16/inference_v16.py` | `src_v16/inference_v16.py` |
| Stage 1 training | `active/v16/train_stage_1_v16.py` | `src_v16/train_stage_1_v16.py` |
| Stage 2 training | `active/v16/train_stage_2_v16_fixed.py` | `src_v16/train_stage_2_v16_fixed.py` |

`src_v16/` wrappers are intentionally kept so existing scripts, mobile export utilities,
and notebooks do not break. `src_v17/` provides v17 extractor/model/training wrappers.

## New v17 Stage 1 commands

```bash
venv/bin/python active/v17/train_stage_1_v17.py \
  --output artifacts/models/stage1_v17_baseline
```

Validation-only checkpoint analysis:

```bash
venv/bin/python active/v17/evaluate_stage_1_v17.py \
  artifacts/models/stage1_v17_baseline/best_model.pth
```

The test evaluator is fail-closed unless `--split test --allow-test` is explicit.

## New v17 extraction commands

```bash
venv/bin/python active/v17/extract_v17.py path/to/video.mp4 \
  --output data/local/ASL_landmarks_v17/example.v17.npz
```

```bash
venv/bin/python active/v17/extract_v17.py data/local/asl_citizen_v17/train \
  --output data/local/asl_citizen_landmarks_v17/train
```

```bash
venv/bin/python active/v17/audit_v17.py data/local/popsign_landmarks_v17
```

The default orientation mode honors video rotation metadata. Use explicit
`--rotation 90|180|270` only for incorrectly tagged sources, and `--input-mirrored`
only when the stored pixels themselves are mirrored.

## Existing v16 model commands

```bash
KMP_DUPLICATE_LIB_OK=TRUE python active/v16/inference_v16.py path/to/video.mp4
```

```bash
python active/v16/train_stage_1_v16.py \
  --data_path src_v16/ASL_landmarks_v16 \
  --save_dir models/output_v16_d384_aug \
  --manifest models/manifest_v16.json
```

```bash
python active/v16/train_stage_2_v16_fixed.py \
  --data_path src_v16/ASL_landmarks_v16 \
  --phrase_data src_v16/ASL_phrases_v16 \
  --stage1_ckpt src_v16/output_v16_d384/best_model.pth \
  --manifest models/manifest_v16.json
```

## Data acquisition

ASL Citizen is now the recommended sole primary source for the first v17 baseline. Use
its official signer-disjoint train/validation/test partitions and select 100 signs with
one exact raw-gloss/ASL-LEX variant and at least 10/3/5 signers per split. The frozen
manifest and selective downloader are:

```bash
venv/bin/python scripts/build_citizen100_v17.py
venv/bin/python scripts/download_citizen100_v17.py --dry-run
venv/bin/python scripts/download_citizen100_v17.py --workers 4
```

The current corpus has 3,102 raw videos and 3,101 valid v17 archives; one official HE
clip contains no visible hands and is explicitly rejected. A second incompletely
segmented SLEEP training clip is preserved but listed in the rejection manifest. ASL
Citizen is research/noncommercial-use data; do not assume it licenses a commercial
shipping model.

`scripts/download_popsign_v17.py` remains a tested utility for optional one-handed
portrait audits. PopSign must not define general two-handed ASL coverage, and its paused
partial `thankyou` download should not resume unless that audit is explicitly needed.

## Legacy and artifacts

- `src/` is the legacy DS-GCN/Transformer path. It remains importable.
- `legacy/copies/` holds duplicate files moved out of `src/`.
- `data/local/` holds local datasets; root dataset names are symlinks for compatibility.
- `artifacts/model_assets/` holds `models/` and `weights/`; root names are symlinks.
- `artifacts/generated/` holds build/runtime output directories.
- `artifacts/reports/` holds generated charts, reports, metrics, logs, and CSV outputs.
- `docs/archive/` is reserved for stale or pre-v16 documentation.
- `docs/md_files/` holds the historical markdown note collection; `md files/` is a compatibility symlink.
- `scripts/extraction/` holds batch extractor implementations; root extractor files are wrappers.

## Cleanup rule

Do not delete datasets, checkpoints, or historical reports until the active v16 commands and required tests have been verified after each cleanup step.
