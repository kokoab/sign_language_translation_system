# Documentation & artifact file map

Quick index: **where to find** architecture notes, benchmarks, metrics (e.g. **96%**), training plans, and related **JSON/checkpoint** paths. Paths are relative to the repo root unless noted.

---

## 1. Architecture: DS-GCN / TCN (v14-v15) vs Squeezeformer (v16)

For current work, start with `PROJECT_GROUND_TRUTH.md`, `CURRENT_PIPELINE.md`, and
`active/v17/README.md`. v17 is canonical for new extraction and Stage 1 research;
v16 retains the existing Stage 2/inference pipeline.

Current code lives under `active/v16/`. The `src_v16/*.py` files are compatibility wrappers for older scripts and notebooks; large local v16 datasets/checkpoints may still live under `src_v16/`.

| Topic | Primary doc | Also useful |
|--------|-------------|-------------|
| **v17 orientation-safe extraction contract** | `active/v17/README.md` | `artifacts/reports/V17_EXTRACTOR_AUDIT.md` |
| **v17 full landmark missingness audit** | `artifacts/reports/CITIZEN100_V17_LANDMARK_QUALITY.md` | `artifacts/generated/v17_diagnostics/citizen_landmark_overlay_audit.jpg` |
| **v17 Stage 1 validation result** | `artifacts/reports/stage1_v17_validation/REPORT.md` | `artifacts/models/stage1_v17_baseline/result.json` |
| **Full DS-GCN-V14 + TemporalTCN + ArcFace + Stage 2** | `md files/PIPELINE_TECHNICAL_DISSECTION.md` | `md files/DEFENSE_PREPARATION.md` (Q&A, MultiScaleTCN) |
| **v16 Squeezeformer replaces DS-GCN-TCN** | `md files/MANUSCRIPT_DISCREPANCIES.md` | `md files/PAPER_REVIEW.md` (v15 vs v16, tables) |
| **High-level pipeline (may still say DS-GCN)** | `md files/context.md` | Prefer v16 sources above; `context.md` can lag v16 naming |
| **Code** | `src/train_stage_1.py`, `src/train_stage_2.py` (DSGCN era) | `active/v16/model_v16.py`, `active/v16/train_stage_1_v16.py`, `active/v16/train_stage_2_v16_fixed.py` |

---

## 2. Simplification, TCN removal, mobile sizing

| Topic | Primary doc |
|--------|-------------|
| **Params, timing, 30-step smoke tests, remove TCN, 7ch, d=256** | `md files/SIMPLIFICATION_TEST_RESULTS.md` |
| **Redundant TCN, 16ch→7ch, DS-GCN vs GAT, skeleton vs RGB** | `md files/OPTIMIZATION_AND_MOBILE_PLAN.md` |
| **Mobile export, conversion, Stage 1 checkpoint naming** | `md files/MOBILE_DEPLOYMENT_PLAN.md` |
| **Conversion / Core ML / TFLite notes** | `mobile_export/reports/CONVERSION_BRIEFING.md`, `mobile_export/reports/CONVERSION_TECHNICAL_DETAILS.md` |

---

## 3. Stage 1 test accuracy **96.00%** (and related numbers)

| Kind | Location |
|------|----------|
| **Narrative + Run 8b tables, cleaning impact, paper numbers** | `md files/PAPER_REVIEW.md` (sections on Stage 1 results, §4 numbers, architecture tables) |
| **v16 trained on Apple Vision only** | `md files/MANUSCRIPT_DISCREPANCIES.md` |
| **Deployment summary line** | `md files/MOBILE_DEPLOYMENT_PLAN.md` |
| **Roadmap assumption (~96%)** | `md files/NEXT_TRAINING_AND_EVAL_ROADMAP.md` |
| **Serialized metrics** | `src_v16/output_v16_d384/eval_results.json` (e.g. test **accuracy 96.0**), `artifacts/metrics/output_v16/eval_results.json` |
| **Export / sanity checks** | `mobile_export/reports/CONVERSION_BRIEFING.md`, `mobile_export/reports/BRIEFING_UPDATE_500.md`, `mobile_export/artifacts/stage1_baseline.json` (`checkpoint_val_acc` ~96.44) |

---

## 4. Training reviews, convergence, Stage 2

| Topic | File |
|--------|------|
| Stage 1 training review (DSGCN era, smoke tests) | `md files/STAGE1_TRAINING_REVIEW.md` |
| Stage 2 CTC review | `md files/STAGE2_CTC_REVIEW_R2.md` |
| Convergence / LR / Stage 2 phases | `md files/CONVERGENCE_ANALYSIS.md` |
| Optimization plan (repo root, training hyperparams) | `OPTIMIZATION_PLAN.md` |

---

## 5. Extractors, gaps, pairing strategies

| Topic | File |
|--------|------|
| Apple Vision vs MediaPipe vs RTMW, benchmarks, GISLR tricks smoke tests | `md files/SIMPLIFICATION_TEST_RESULTS.md` (later sections) |
| Fresh extract pipeline / shared encoder drift | `md files/Fresh_Extract_Pipeline_Review.md` |
| Extract + augment alignment | `md files/EXTRACT_AUGMENT_REVIEW.md` |

---

## 6. Meta / system analyses

| File | Role |
|------|------|
| `md files/Comprehensive_System_Analysis_Report.md` | Broad system narrative |
| `md files/Expert_System_Analysis_Claude.md` | Long-form analysis (includes DS-GCN references) |
| `md files/E2E_vs_Modular_Explanation.md` | Modular vs end-to-end testing framing |
| `md files/SESSION_BRIEFING.md` | Recent v16 inference / training operational notes |

---

## 7. Planning docs (not architecture proofs)

| File | Role |
|------|------|
| `docs/md_files/IOS_FIRST_ACCURACY_AND_DATA_PLAN.md` | Current iOS-first direction, signer-disjoint dataset requirements, portrait canonicalization, and online ASL dataset review |
| `artifacts/reports/IOS100_DATASET_COVERAGE_REPORT.md` | Metadata-derived 100-sign candidate list with exact PopSign and ASL Citizen signer coverage |
| `artifacts/reports/IOS100_VOCABULARY_PROPOSAL.md` | Utility-focused proposed 100, source coverage, signer deficits, and required ASL review |
| `artifacts/reports/IOS100_VIDEO_AUDIT.md` | Measured decode, resolution, orientation, and Apple Vision extraction results for the 72-video external sample |
| `artifacts/reports/IOS100_STAGE1_EXTERNAL_AUDIT.md` | Current 96%-checkpoint diagnostic on 72 external ASL Citizen videos |
| `active/v17/citizen100_manifest.json` | Frozen 100-class canonical-to-exact-Citizen-ASL-LEX mapping |
| `artifacts/reports/CITIZEN100_V17_MANIFEST.md` | Human-readable exact-variant coverage report |
| `artifacts/reports/CITIZEN100_RAW_AUDIT.md` | Raw 3,102-video decode, count, and signer-disjointness audit |
| `artifacts/reports/CITIZEN100_V17_EXTRACTOR_AUDIT.md` | Full 3,101-archive v17 invariant and coverage audit |
| `md files/NEXT_TRAINING_AND_EVAL_ROADMAP.md` | Next experiments, signer-held-out, fingerspelling track |
| `md files/MASTER_IMPLEMENTATION_PLAN.md` | Phased implementation (load when doing phased work) |

---

## 8. Repo rules & entry points

| File | Role |
|------|------|
| `CLAUDE.md` | Architecture, checkpoints, inviolable constraints |
| `AGENTS.md` | Short file map + `md files/context.md` pointer |
| `md files/context.md` | Architecture summary (verify against v16 if naming matters) |

---

## 9. Optional: local Claude Code settings (not project docs)

User-level editor/agent config may live under **`~/.claude/`** (e.g. `settings.json` for hooks, model, marketplaces). That directory is **not** part of this git repo and does not replace the markdown above.

---

## 10. Quick “I need…” lookup

| I need… | Go to… |
|---------|--------|
| DS-GCN + TCN internals | `PIPELINE_TECHNICAL_DISSECTION.md` |
| Why v16 is Squeezeformer | `MANUSCRIPT_DISCREPANCIES.md`, `PAPER_REVIEW.md` |
| Where **96%** is stated + eval JSON | §3 above |
| TCN removal evidence / smoke tests | `SIMPLIFICATION_TEST_RESULTS.md` |
| DS-GCN vs GAT speed argument | `OPTIMIZATION_AND_MOBILE_PLAN.md` §5 |
| Defense talking points | `DEFENSE_PREPARATION.md` |
| What to train next | `NEXT_TRAINING_AND_EVAL_ROADMAP.md` |

---

*This map is a navigation aid; always prefer the linked source file for exact numbers and cited runs.*
