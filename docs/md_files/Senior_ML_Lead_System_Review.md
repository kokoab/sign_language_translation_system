# Senior ML/Engineering Lead — SLT System Review

**Date:** 2026-03-21
**Scope:** Full pipeline audit (Stage 0 extraction → Stage 1 classification → Stage 2 CTC → Stage 3 translation → Deploy)
**Reviewed files:** `src/extract.py`, `src/train_stage_1.py`, `src/train_stage_2.py`, `src/train_stage_3.py`, `src/camera_inference.py`, `src/verify_stage1_label_integrity.py`, `src/verify_stage2_label_integrity.py`, `src/merge_dwight_into_asl_videos.py`, `src/merge_jaz_into_asl_videos.py`, `src/merge_new_into_asl_videos.py`, `md files/context.md`, `verification_outputs/stage1_label_integrity_mismatches_all.csv`

---

## Top 10 Issues (Ranked by Severity)

---

### 1. CRITICAL — Composite/Alias Labels Missing from Stage-1 Vocabulary

**What:** Merge scripts (e.g., `merge_dwight_into_asl_videos.py:22-29`) map source labels like `MARKET` and `STORE` into composite folder `MARKET_STORE`. However, Stage-1 was trained on an older vocabulary that contains the individual labels (`ALSO`, `SAME`, `HE`, `SHE`, `MARKET`, `STORE`) but **not** their merged composite forms (`ALSO_SAME`, `HE_SHE`, `MARKET_STORE`). Verification confirms: all composite labels show **0% match rate** across 1,171+ samples (ALSO_SAME: 237, MARKET_STORE: 200, US_WE: 169, FEW_SEVERAL: 157, I_ME: 106, HE_SHE: 102).

**Why it matters:** These labels represent real, common signs. The model physically cannot predict them — they don't exist in its output softmax. Every composite-labeled sample is a guaranteed training/evaluation error. This inflates reported mismatch rates and corrupts loss gradients during training.

**How to validate:**
- `python -c "import json; v=json.load(open('ASL_landmarks_float16/manifest.json')); labels=set(v.values()); composites=['ALSO_SAME','MARKET_STORE','US_WE','FEW_SEVERAL','I_ME','HE_SHE']; print([c for c in composites if c not in labels])"` — check which composites are in the manifest but absent from the model vocab.
- Load the Stage-1 checkpoint's `idx_to_label` dict and verify these composites are missing.

**Proposed fix (two options, pick one):**
- **Option A — Synonym collapse (recommended):** Before training, canonicalize all labels. Choose one representative per synonym group (e.g., `ALSO_SAME → ALSO`, `MARKET_STORE → STORE`, `I_ME → I`). Apply this mapping in `extract.py`'s `LABEL_ALIASES` (line 35) and in manifest generation. Retrain Stage 1 + Stage 2 with clean vocab.
- **Option B — Expand vocabulary:** Add composite labels to the training vocab. Requires relabeling or ensuring folders contain enough data. Less clean — doubles concept-space for identical signs.

---

### 2. CRITICAL — Stage-2 Frame Dropping When Sequence Length Not Divisible by 32

**What:** In `train_stage_2.py:211-212`, the forward pass computes `num_clips = valid_x.size(0) // 32` then `clips = valid_x.view(num_clips, 32, 47, 10)`. If the sequence has 100 frames, only 96 are used — 4 frames silently discarded.

**Why it matters:** CTC loss aligns token-level predictions to ground-truth gloss sequences. Dropping frames reduces the output token count (`num_clips * 4`) without adjusting targets. In edge cases, the output sequence becomes shorter than the target sequence, causing CTC loss to return `inf` (caught by `zero_infinity=True` but effectively wasting that sample's gradient). Worse, the dropped tail frames may contain the final sign's discriminative motion.

**How to validate:**
- Add a counter in `forward()`: `dropped = valid_x.size(0) - num_clips * 32; if dropped > 0: print(f"Dropped {dropped} frames")`
- Check how often `zero_infinity` triggers by logging per-sample CTC loss values.

**Proposed fix:** Zero-pad the last clip to 32 frames before encoding. In `forward()` after line 211:
```
remainder = valid_x.size(0) % 32
if remainder > 0:
    pad = torch.zeros(32 - remainder, 47, 10, device=valid_x.device)
    valid_x = torch.cat([valid_x, pad], dim=0)
    num_clips += 1
```
Update the mask channel for padded frames to 0 so the model can learn to ignore them.

---

### 3. HIGH — Alphabet/Fingerspelling Labels (A_A, B_B, ..., Z_Z) All 0% Match

**What:** Verification shows 619 samples across 26 alphabet labels (A_A, B_B, ..., Z_Z, plus Q_KYU) with **100% mismatch**. These use a `{LETTER}_{LETTER}` naming convention that doesn't exist in Stage-1 vocabulary.

**Why it matters:** If these represent fingerspelling data intended for training, they contribute only noise. If they're test/evaluation data, they invalidate any accuracy metric computed over the full dataset. Either way, ~1.6% of the dataset is guaranteed garbage.

**How to validate:**
- Inspect the actual video content in folders like `ASL VIDEOS/A_A/`, `ASL VIDEOS/B_B/` — are these single-letter fingerspelling clips?
- Check if the Stage-1 vocab contains single-letter labels (`A`, `B`, `C`, ...).

**Proposed fix:**
- If fingerspelling: Rename folders to single letters (`A_A → A`), add to vocab, retrain. Or exclude from sign-level training and handle fingerspelling as a separate sub-problem.
- If duplicate/test data: Remove from training set entirely.

---

### 4. HIGH — LM Rescoring Never Activates in Inference

**What:** `camera_inference.py` defines two separate LM systems:
1. `NgramLanguageModel` loaded via `load_language_model()` into `LANGUAGE_MODEL` (JSON-based, lines 142-154) — **never called** in `main()`.
2. `GlossNGramLM` loaded via `_load_gloss_lm()` into `GLOSS_LM` (pickle-based, lines 243-254) — loaded at line 625.

The CTC beam rescoring at line 523 checks `if LANGUAGE_MODEL is not None` — but `LANGUAGE_MODEL` is never assigned because `load_language_model()` is never invoked. The actually-loaded `GLOSS_LM` is unused during rescoring.

**Why it matters:** LM rescoring is a major accuracy lever for CTC-decoded gloss sequences. Without it, the system relies purely on acoustic scores, which are noisy. This is a shipped dead-code bug.

**How to validate:**
- Add `print("LM rescoring active")` inside the `if LANGUAGE_MODEL` branch — it will never fire.
- Grep for `LANGUAGE_MODEL` vs `GLOSS_LM` usage.

**Proposed fix:** Unify to a single LM. Either:
- Replace `LANGUAGE_MODEL` check with `GLOSS_LM` at line 523, or
- Call `load_language_model()` in `main()` and remove the pickle path, or
- Delete the dead `NgramLanguageModel` class and route `_load_gloss_lm()` → `LANGUAGE_MODEL`.

---

### 5. HIGH — No Mask Gating for Non-Informative Face Landmarks

**What:** `extract.py:175-178` sets `mask = 1.0` for face nodes whenever face detection succeeds, and `mask = 0.0` when it doesn't. But the model architectures (DS-GCN in Stage 1, BiLSTM in Stage 2) do **not** use the mask channel to gate features. The mask is just another input channel — the model must implicitly learn to ignore masked-out nodes.

**Why it matters:** Many signs don't involve the face. When face landmarks are detected but non-informative (signer visible but sign is hand-only), the face XYZ values contain random head-position noise that the model has no mechanism to suppress. The graph structure in Stage 1 (`train_stage_1.py:59-67`) explicitly connects wrists and fingertips to face nodes via 17 edges — so face noise propagates into hand representations via GCN message passing.

**How to validate:**
- Ablation: Train Stage-1 with face nodes zeroed out entirely. Compare accuracy. If accuracy doesn't drop (or improves), face features are noise, not signal.
- Inspect geometric features: `train_stage_1.py:177-191` computes hand-to-face distances. Check variance of these features across the dataset.

**Proposed fix (two tiers):**
- **Quick:** In `DSGCNBlock`, multiply face node features by their mask channel before GCN aggregation: `x[:, :, 42:47, :] *= x[:, :, 42:47, 9:10]`. This zeros out undetected faces and dampens non-informative ones.
- **Better:** Add learnable face-gate: `face_gate = sigmoid(linear(face_features))` that learns when face is informative. Apply before GCN aggregation.

---

### 6. HIGH — Train/Inference Feature Extraction Mismatches

**What:** Several parameter differences between training (`extract.py`) and inference (`camera_inference.py`):

| Parameter | extract.py | camera_inference.py | test_video_pipeline.py |
|-----------|-----------|---------------------|----------------------|
| MIN_RAW_FRAMES | 8 (line 75) | 8 (line 49) | 5 (line 50) |
| MODEL_COMPLEXITY | 1 (line 79) | 1 (line 57) | 0 (line 53) |

The `test_video_pipeline.py` uses lower thresholds. Any results from the test pipeline are from a different distribution than training/deployment.

Additionally, `camera_inference.py`'s normalization (`normalize_sequence`, lines 297-326) and kinematics (`compute_kinematics`, lines 328-346) are reimplemented rather than imported from `extract.py`. Any drift between these copies creates silent distribution shift.

**Why it matters:** Distribution mismatch between train and inference is a top cause of accuracy degradation in production ML systems. Even small differences in landmark detection (MODEL_COMPLEXITY 0 vs 1) change the XYZ distribution.

**How to validate:**
- Diff the normalization and kinematics functions between `extract.py` and `camera_inference.py` line-by-line.
- Run the same video through both pipelines, compare output tensors numerically.

**Proposed fix:**
- **Immediate:** Refactor shared functions (`normalize_sequence`, `compute_kinematics`, `temporal_resample`, `interpolate_hand/face`) into a shared module (e.g., `src/feature_utils.py`). Import everywhere.
- **Immediate:** Align `test_video_pipeline.py` constants with `extract.py` and `camera_inference.py`.

---

### 7. MEDIUM — Stage-3 Training Data / Stage-2 Output Domain Gap

**What:** Stage-3 (`train_stage_3.py`) trains on CSV data containing curated gloss sequences (e.g., `"I GO STORE"`). Stage-2 outputs CTC-decoded gloss sequences that may contain:
- Insertion errors (extra glosses)
- Deletion errors (missing glosses)
- Substitution errors (wrong glosses)
- Glosses not in Stage-3's training set

Stage-3 never sees noisy/erroneous gloss input during training.

**Why it matters:** At inference time, Stage-3 receives imperfect Stage-2 output but was trained on clean input. This is a classic cascading-error / exposure-bias problem. Even moderate WER from Stage-2 (e.g., 15-25%) can cause Stage-3 to produce nonsensical translations.

**How to validate:**
- Feed Stage-2's actual decoded outputs (on validation set) into Stage-3. Compare BLEU/ROUGE against feeding clean glosses. The gap measures cascading error impact.
- Check vocab overlap: what percentage of Stage-2 decoded glosses appear in Stage-3 training data?

**Proposed fix:**
- **Medium effort:** Data augmentation for Stage-3: randomly corrupt training gloss sequences (insert/delete/substitute with Stage-2-like error rates) so the model learns robustness.
- **Better:** Use Stage-2 decoded outputs (with errors) as Stage-3 training inputs instead of clean glosses. This is "scheduled sampling" / "professor forcing" adapted for pipeline training.
- **Long-term:** End-to-end fine-tuning of Stage 2 + 3 jointly.

---

### 8. MEDIUM — Verification Only Measures Stage-1 Top-1 Accuracy

**What:** `verify_stage1_label_integrity.py` checks if `argmax(model(x)) == expected_label`. This is a single metric (top-1 accuracy) that misses:
- Top-5 accuracy (was the correct label in the top 5?)
- Confusion patterns (which labels confuse with which?)
- Confidence calibration (is the model confident when wrong?)
- Per-class recall (are some classes systematically missed?)

The 5.41% mismatch includes the composite-label problem, inflating the number. True model errors are lower but unknown.

**Why it matters:** Without deeper metrics, you can't distinguish between "model is bad" and "labels are wrong" and "vocabulary is misaligned." Decision-making about what to fix first requires finer-grained diagnostics.

**How to validate:** Already have the CSV data — just needs richer analysis.

**Proposed fix:**
- **Quick:** Post-process existing CSV to compute: (a) confusion matrix, (b) top-5 accuracy, (c) per-class precision/recall/F1, (d) separate metrics for composite vs non-composite labels.
- **Quick:** Filter out known vocabulary mismatches (composite labels, alphabet labels) before computing "true" model accuracy.
- **Medium:** Add calibration curve (confidence vs accuracy bucketed by prediction confidence).

---

### 9. MEDIUM — CTC Blank/PAD Handling Not Explicitly Validated at Boundaries

**What:** CTC blank is correctly set to index 0 (`train_stage_2.py:660`, `camera_inference.py:462`). PAD is separate. But during Stage-2 training, the `SyntheticCTCDataset` constructs variable-length sequences that are then padded to batch max length. The `input_lengths` tensor (`train_stage_2.py:738`) tracks valid (non-padded) output tokens, but there's no assertion that `input_lengths[b] >= target_lengths[b]` for all batch elements — a CTC requirement.

**Why it matters:** If any sample has `input_length < target_length`, CTC loss returns `inf` for that sample. With `zero_infinity=True`, the gradient is zeroed — the sample is silently wasted. If this happens systematically for certain label sequences, those patterns never train.

**How to validate:**
- Add assertion in training loop: `assert (input_lengths >= target_lengths).all(), f"CTC violation: {input_lengths} vs {target_lengths}"`
- Log how often `zero_infinity` triggers (CTC loss per sample, count inf values).

**Proposed fix:**
- Add the assertion above before `ctc_loss()` call.
- If violations occur: increase minimum sequence length or reduce target length by dropping longest sequences.

---

### 10. LOW-MEDIUM — Stage-1 Curriculum Learning Threshold Is Fragile

**What:** `train_stage_1.py:226-235` (`is_single_hand()`) determines if a clip uses one or two hands by checking `mask.max() > 0.5` for left (indices 0-20) and right (indices 21-41) hand channels. The threshold 0.5 is arbitrary — mask values are binary (0.0 or 1.0 from extract.py), but after augmentation (noise addition at line 496: `noise_std=0.003`), the mask channel could drift.

**Why it matters:** If curriculum learning misclassifies two-hand signs as single-hand (or vice versa), Phase 1 training sees the wrong subset. This corrupts the learning progression. The effect is subtle — slightly worse convergence rather than outright failure.

**How to validate:**
- Check mask channel values after augmentation: `x_aug[:, :, :, 9].min(), x_aug[:, :, :, 9].max()`. If noise pushes 0.0 masks above 0.5, curriculum is broken.
- Count how many samples switch category after augmentation.

**Proposed fix:**
- In `online_augment()`, explicitly skip channel 9 from noise injection (currently, line 496 applies noise to `x[:, :, :, :9]` — **this is already correct**, noise is only applied to channels 0-8). So the actual risk is low. Verify and document.
- Use `>= 0.5` instead of `> 0.5` for consistency, though practically equivalent with binary masks.

---

## Prioritized Improvement Plan

### Phase 1: Quick Wins (No Retraining, 1-3 Days)

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 1.1 | Fix LM rescoring dead code — unify to single LM and wire it into beam search | `camera_inference.py:523` | Enables LM rescoring; immediate accuracy boost on multi-gloss sequences |
| 1.2 | Add CTC input_length >= target_length assertion | `train_stage_2.py` before CTC loss call | Catches silent training failures |
| 1.3 | Compute richer verification metrics from existing CSV | New script or notebook | Separates true model errors from vocab mismatches; informs what to fix next |
| 1.4 | Align test_video_pipeline.py constants (MIN_RAW_FRAMES=8, MODEL_COMPLEXITY=1) | `test_video_pipeline.py:50,53` | Eliminates test/deploy distribution mismatch |
| 1.5 | Pad last clip to 32 frames in Stage-2 forward pass | `train_stage_2.py:211-212` | Eliminates frame dropping; may improve WER |

### Phase 2: Medium Effort (Requires Retraining, 1-2 Weeks)

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 2.1 | Canonicalize composite labels — pick one label per synonym group, update LABEL_ALIASES, regenerate manifest, retrain Stage 1 + 2 | `extract.py:35-44`, manifest, Stage 1 + 2 training | Removes ~1,171 guaranteed-wrong samples; cleaner vocab |
| 2.2 | Decide on alphabet labels — include as fingerspelling or exclude | Data folders, manifest | Removes ~619 guaranteed-wrong samples |
| 2.3 | Add mask gating for face landmarks in DS-GCN | `train_stage_1.py` DSGCNBlock | Reduces face-noise propagation; may improve accuracy on hand-only signs |
| 2.4 | Refactor shared feature functions into `src/feature_utils.py` | `extract.py`, `camera_inference.py`, `test_video_pipeline.py` | Eliminates code drift; single source of truth |
| 2.5 | Augment Stage-3 training data with noisy gloss sequences (simulating Stage-2 errors) | `train_stage_3.py`, data generation | Reduces cascading error; more robust translation |

### Phase 3: Longer-Term (Architecture Changes, 2-4 Weeks)

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 3.1 | Joint Stage 2+3 fine-tuning with end-to-end gradients (unfreeze encoder partially) | `train_stage_2.py`, `train_stage_3.py` | Reduces cascading error at the architecture level |
| 3.2 | Replace static 32-frame clips with variable-length encoding (e.g., adaptive pooling per sign) | `extract.py`, `train_stage_1.py`, `train_stage_2.py` | Better handling of signs with variable duration |
| 3.3 | Add explicit fingerspelling sub-model or character-level decoder branch | New module | Handles alphabet/fingerspelling as separate problem |
| 3.4 | Implement confidence-based selective prediction (reject low-confidence predictions) | `camera_inference.py` | Reduces hallucinated translations in production |
| 3.5 | Add BLEU/ROUGE/BERTScore to Stage-3 evaluation | `train_stage_3.py` | Better generation quality tracking than loss alone |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Label canonicalization changes break downstream scripts | High | Medium | Create mapping table first; update all consumers atomically; run full verification suite after |
| Retraining with fixed vocab degrades accuracy on previously-correct classes | Medium | High | Compare per-class metrics before/after; keep old checkpoint as fallback |
| Mask gating removes useful face signal for face-touch signs (e.g., THINK, KNOW) | Medium | Medium | Ablation: compare face-gated vs ungated on face-touch sign subset specifically |
| Stage-2 clip padding introduces artifacts at sequence boundaries | Low | Medium | Set mask=0 for padded frames; verify CTC loss ignores them via input_lengths |
| LM rescoring fix introduces bias toward frequent glosses | Medium | Low | Tune LM_WEIGHT on held-out set; compare with/without LM on rare vs frequent glosses |
| Refactoring shared code introduces subtle numerical differences | Low | High | Bit-exact test: run same video through old and new paths, assert tensor equality |
| Stage-3 noise augmentation makes model too tolerant of errors | Low | Low | Control noise rate; evaluate on both clean and noisy inputs |

---

## Minimum Tests Before Claiming Improvements

### After Phase 1 (no retraining):
1. **LM rescoring activation test:** Run inference on 10+ multi-gloss sequences; confirm LM scores appear in logs and beam reranking changes output.
2. **Frame dropping test:** Feed a 100-frame sequence through Stage-2; confirm all 100 frames contribute (4 clips, last one padded, 16 output tokens).
3. **CTC assertion test:** Run one full training epoch of Stage-2; confirm no assertion failures.
4. **Verification recompute:** Rerun `verify_stage1_label_integrity.py`, filter out composite+alphabet labels, report "true model" top-1 and top-5 accuracy.

### After Phase 2 (with retraining):
5. **Vocab integrity:** `assert set(manifest_labels) == set(model_vocab)` — no label exists in data but not in model, or vice versa.
6. **Per-class regression test:** Compare per-class F1 before/after on a held-out test set. Flag any class with >5% F1 drop.
7. **Stage-2 WER:** Report word error rate on synthetic CTC validation set. Target: <20% WER.
8. **Stage-2 → Stage-3 cascade test:** Feed Stage-2 decoded outputs through Stage-3. Report BLEU on 100+ sentence pairs. Compare against clean-gloss baseline.
9. **Face gating ablation:** Report Stage-1 accuracy with and without face gating, broken down by: (a) all signs, (b) face-touch signs only, (c) hand-only signs only.
10. **End-to-end smoke test:** Record 10 live webcam clips of known signs → verify correct English output.

### Ongoing:
11. **Confusion matrix tracking:** After each retrain, generate and diff confusion matrices. New confusions = regression.
12. **Data integrity CI:** Run `verify_stage1_label_integrity.py` and `verify_stage2_label_integrity.py` as part of any training pipeline; fail if mismatch rate exceeds threshold (e.g., 3% after fixes).

---

## Appendix: Key Code References

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Label aliases | `src/extract.py` | 35-44 | Current synonym mapping (6 groups) |
| Kinematics computation | `src/extract.py` | 163-180 | Central difference vel/acc |
| Face landmark indices | `src/extract.py` | 66-70 | Nose=42, Chin=43, Forehead=44, L_Ear=45, R_Ear=46 |
| DS-GCN graph edges | `src/train_stage_1.py` | 36-67 | Hand, face, and hand-face connections |
| Geometric features | `src/train_stage_1.py` | 161-194 | 34 hand+face geometric features |
| Curriculum detection | `src/train_stage_1.py` | 226-235 | is_single_hand() mask threshold |
| Online augmentation | `src/train_stage_1.py` | 452-498 | Speed warp → rotation → scale+noise |
| Stage-2 frame dropping | `src/train_stage_2.py` | 211-212 | Integer division truncation |
| CTC loss setup | `src/train_stage_2.py` | 660, 692 | blank=0, zero_infinity=True |
| Synthetic transitions | `src/train_stage_2.py` | 378-412 | 4-12 frame interpolated transitions |
| CTC beam search | `src/camera_inference.py` | 462-504 | beam_width=25, blank=0 |
| LM rescoring (broken) | `src/camera_inference.py` | 521-531 | Checks LANGUAGE_MODEL (never loaded) |
| Multi-hypothesis | `src/camera_inference.py` | 538-573 | Try N=1-4 signs, hand-count prior |
| Stage-3 context prompt | `src/train_stage_3.py` | 150-165 | Up to 4 previous dialogue turns |
| Merge label mapping | `src/merge_dwight_into_asl_videos.py` | 22-29 | EXPLICIT_CLASS_MAPPING |
| Verification mismatch modes | `src/verify_stage2_label_integrity.py` | 180-196 | strict, all_tokens, contains |
