# Stage 2 Data-Path Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task.

**Goal:** Determine whether Stage 2 fails because of video quality, landmark extraction, timing/cropping, supervision construction, dataset coverage, or the temporal model contract before replacing data or architecture.

**Architecture:** Reuse existing manifests, checkpoints, extracted NPZ files, and local phrase recordings. Produce one read-only audit that compares corpora at each data boundary and a report that separates measured defects from rejected hypotheses. Do not train, alter datasets, or access the Citizen test split.

**Tech Stack:** Python standard library, NumPy, OpenCV, existing v17 loaders/models, existing report artifacts.

**Spec:** `PROJECT_GROUND_TRUTH.md` plus the user's 2026-09-21 request to investigate all locally available evidence, including local phrases.

## Global Constraints

- Preserve the locked 100-sign vocabulary and official signer-disjoint splits.
- Do not access or rerun the consumed Citizen test gate.
- Read large artifacts selectively; never load a result file over 1 MB as text.
- Treat ASLLRP source boundaries according to their published linguistic convention.
- Make no training or runtime change until the root cause is measured.
- Preserve unrelated uncommitted changes.

## Review Focus

- Low-resolution videos may reduce hand detection or fine handshape fidelity despite acceptable aggregate coverage.
- Crop-local timestamps may be offset from video frames even when annotations are correct.
- Reconstructed motion channels may differ between training and live rolling windows.
- Local phrases may fail because of signer/domain shift rather than transition learning.
- Aggregate accuracy may hide a few low-coverage glosses or one-signer classes.

---

### Task 1: Trace the complete data contract

**Files:**
- Create: `artifacts/reports/stage2_data_path_audit_20260921/audit.py`
- Create: `artifacts/reports/stage2_data_path_audit_20260921/inventory.json`

**Interfaces:**
- Consumes: current Stage 2 manifests, v17 NPZ schema, inference and training loaders.
- Produces: corpus/file/schema inventory and explicit video→annotation→landmark→window mappings.

- [x] Identify the exact manifests, checkpoints, feature roots, and live/training callers used by current Stage 2.
- [x] Assert that protected Citizen test paths are absent.
- [x] Record schema, timing, sampling, normalization, handedness, and window construction differences.
- [x] Run the audit against a small sample and verify every referenced path and timestamp range resolves.

### Task 2: Measure video and landmark sufficiency

**Files:**
- Modify: `artifacts/reports/stage2_data_path_audit_20260921/audit.py`
- Create: `artifacts/reports/stage2_data_path_audit_20260921/metrics.json`

**Interfaces:**
- Consumes: Task 1 inventory and local ASLLRP, O5S5, isolated, and local-phrase videos/features.
- Produces: per-corpus resolution/FPS, hand/face/body coverage, landmark confidence, temporal gaps, motion continuity, and usable-sign-frame metrics.

- [x] Measure video resolution, FPS, duration, and compression indicators without decoding protected splits.
- [x] Measure landmark presence and timing coverage inside signs, at boundaries, and in context.
- [x] Compare low-resolution and higher-resolution/local phrase corpora using the same metrics.
- [x] Verify a stratified visual gallery for the best, median, and worst landmark cases.

### Task 3: Isolate annotation, preprocessing, and model-contract failures

**Files:**
- Modify: `artifacts/reports/stage2_data_path_audit_20260921/audit.py`
- Create: `artifacts/reports/stage2_data_path_audit_20260921/diagnosis.json`

**Interfaces:**
- Consumes: Tasks 1–2 metrics and existing frozen-checkpoint evaluations.
- Produces: ranked hypotheses with evidence for/against each and corpus-specific replacement criteria.

- [x] Reconstruct source-to-crop and crop-to-landmark clocks and quantify edge errors.
- [x] Compare exact-core, context-window, continuous, held/repeat, and local-phrase results by corpus.
- [x] Separate landmark visibility failure from signer/domain shift, label coverage, window dilution, and decoder failure.
- [x] Test whether slowing/resampling changes information content or only repeats the same landmarks.

### Task 4: Publish the decision report

**Files:**
- Create: `artifacts/reports/stage2_data_path_audit_20260921/REPORT.md`
- Create: `artifacts/reports/stage2_data_path_audit_20260921/videos.html`
- Create: `artifacts/reports/stage2_data_path_audit_20260921/verification.json`
- Modify: `PROJECT_GROUND_TRUTH.md`
- Modify: `docs/ground_truth/data-sources/log.md`
- Modify: `docs/ground_truth/live-streaming/log.md`

**Interfaces:**
- Consumes: all measured evidence.
- Produces: a concrete keep/repair/replace decision for each corpus and the next single experiment, if justified.

- [x] Write findings with counts and limitations; do not infer causality from aggregate correlations.
- [x] Build a review page linking source video, landmark overlay, annotation, and model outcome for representative failures.
- [x] Run audit self-checks, Python compilation, report consistency checks, large-artifact indexing, and `git diff --check`.
- [x] Record the measured decision in the canonical project state and topic logs.
