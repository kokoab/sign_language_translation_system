# SLT Pipeline Review — Fresh Extract-and-Train Audit

**Date:** 2026-03-21
**Scope:** Full pipeline correctness review before re-extraction and retraining from scratch.
**Reviewer context:** All `.npy` outputs deleted; fresh run: extract -> Stage 1 -> Stage 2 -> Stage 3 data gen -> Stage 3 -> build LM -> deploy.

---

## TOP 10 ISSUES (Severity-Ordered)

---

### ISSUE 1 — LM Build/Load Format Mismatch (CRITICAL, will crash at deploy)

**What:** `build_language_model.py` saves a JSON trigram LM to `models/gloss_lm.json` (line 188). `camera_inference.py` loads a **pickle** bigram LM from `weights/gloss_bigram_lm.pkl` (line 39) using a completely different class (`GlossNGramLM` vs `NgramLanguageModel`). These two classes have incompatible APIs:
- `build_language_model.py:NgramLanguageModel.log_prob(word, context)` takes a **list** as context.
- `camera_inference.py:GlossNGramLM.log_prob(gloss, prev_gloss)` takes a **single string** as prev_gloss.
- Different smoothing schemes (Laplace vs additive).
- Different serialization formats (JSON with `eval()` deserialization vs pickle).
- Different n-gram order (trigram vs bigram).

**Why it matters:** The LM at deploy time will either not load (wrong path/format) or score sequences incorrectly. LM rescoring is the primary mechanism for correcting CTC beam search errors.

**How to validate:**
```bash
python src/build_language_model.py
python -c "from src.camera_inference import _load_gloss_lm; _load_gloss_lm()"
# Will fail: path mismatch and format mismatch
```

**Proposed fix:** Unify on a single LM class. Options:
1. Make `build_language_model.py` output the pickle format that `camera_inference.py` expects (a `GlossNGramLM`-compatible dict with `unigram_counts`, `bigram_counts`, `vocab`, `total_unigrams`, `smoothing`), saved to `weights/gloss_bigram_lm.pkl`.
2. Or update `camera_inference.py` to load the JSON trigram format.

Recommendation: Option 1 is simpler — add a `save_pickle()` method to `build_language_model.py` that outputs the exact dict structure `GlossNGramLM.load()` expects.

---

### ISSUE 2 — LM Vocab Uses Stale Composite Labels (CRITICAL, will produce OOV at deploy)

**What:** `build_language_model.py` hardcodes common patterns using **composite/alias labels** that extraction no longer produces:
- Lines 137-158: `"ALSO_SAME"`, `"HE_SHE"`, `"I_ME"`, `"US_WE"`, `"MARKET_STORE"`, `"DRIVE_CAR"`, `"EAT_FOOD"`, `"MAKE_CREATE"`, `"HARD_DIFFICULT"`

But `extract.py` LABEL_ALIASES (lines 35-58) maps these to canonical labels:
- `ALSO_SAME` -> `ALSO`, `HE_SHE` -> `HE`, `I_ME` -> `I`, `US_WE` -> `WE`, `MARKET_STORE` -> `STORE`

So the LM will contain glosses that never appear in the Stage 2 vocabulary (which is built from the manifest). At inference time, LM scores for actual decoded glosses (`ALSO`, `HE`, `WE`, `STORE`) will fall back to uniform smoothing, making the LM useless for these signs.

**Why it matters:** The LM provides zero benefit for any gloss that was alias-collapsed, and the hardcoded patterns will pollute the probability distribution.

**How to validate:** After fresh extraction, diff the `set(manifest.values())` against the LM's `vocab`. Any mismatch = broken rescoring for those glosses.

**Proposed fix:**
1. `build_language_model.py` should read the manifest and derive its vocab from `set(manifest.values())` — not hardcode composite labels.
2. Hardcoded patterns should use the **canonical** label names (e.g., `["ALSO"]` not `["ALSO_SAME"]`, `["HE", "GO"]` not `["HE_SHE", "GO"]`).
3. Better: generate patterns programmatically from the Stage 3 CSV glosses, which should already use canonical labels.

---

### ISSUE 3 — Verification Scripts Expect 42-Node Shape, Extraction Produces 47-Node (CRITICAL, verification will crash)

**What:** Both `verify_stage1_label_integrity.py` (line 116, 128-129) and `verify_stage2_label_integrity.py` (line 142, 153-154) hardcode:
```python
xs = np.empty((B, 32, 42, 10), dtype=np.float32)
if arr.shape != (32, 42, 10):
    raise ValueError(...)
```

But `extract.py` produces `(32, 47, 10)` tensors (42 hand + 5 face landmarks). Every file will fail the shape check.

**Why it matters:** You cannot run any verification after fresh extraction until these scripts are updated. This blocks the entire verification methodology.

**How to validate:** Run either verification script after extraction — it will immediately crash.

**Proposed fix:** Change both verification scripts from `(32, 42, 10)` to `(32, 47, 10)` in the shape allocation and assertion.

---

### ISSUE 4 — LABEL_ALIASES Inconsistency Creates Ghost Mismatches in Verification

**What:** The CSV output (line 44-50 of the mismatch CSV) shows `ALSO_SAME` as an expected_label predicted as `ALSO`. The verification script (line 27-31) parses the label from the `.npy` filename prefix. But `extract.py` applies `LABEL_ALIASES` to convert folder names to canonical labels before saving. So:
- Folder name: `ALSO_SAME` -> applied alias -> saved as `ALSO_*.npy`
- Verification parses filename: gets `ALSO` as expected label
- Model predicts: `ALSO` -> match!

**However**, the current CSV shows `ALSO_SAME` as the expected label, meaning the old extraction did NOT apply aliases consistently, or the verification was run against old (pre-alias) extractions.

Since you're re-extracting from scratch, this should be resolved IF aliases are applied. But there's a subtlety: some video folders may still be named with composite labels (`ALSO_SAME/`, `HE_SHE/`, etc.), AND individual folders (`ALSO/`, `SAME/`, `HE/`, `SHE/`) may also exist. This means:
- Videos in `ALSO_SAME/` folder get label `ALSO` (via alias)
- Videos in `ALSO/` folder get label `ALSO` (direct)
- Videos in `SAME/` folder get label `ALSO` (via alias)
- All three map correctly, but only if folders exist and aliases are comprehensive.

**Why it matters:** If any folder name is missing from `LABEL_ALIASES`, it becomes its own class, creating a label that Stage 2 must decode but that has inconsistent semantics.

**How to validate:**
```bash
ls "data/raw_videos/ASL VIDEOS/" | sort > /tmp/folders.txt
# Check each folder name against LABEL_ALIASES keys
```

**Proposed fix:** After extraction, add a pre-training sanity check:
1. Load `manifest.json`
2. Assert no label appears that's also a key in `LABEL_ALIASES` (meaning the alias wasn't applied)
3. Print the full label distribution to catch surprises

---

### ISSUE 5 — Stage 1 `temporal_speed_warp` Comment Says `[B, T, 42, 3]` but Input is `[B, T, 47, 10]`

**What:** `train_stage_1.py:temporal_speed_warp` (line 334-395) — the docstring says `xyz_tensor: [B, T, 42, 3]` but the actual input at line 474 is `x[..., :3]` from a `[B, T, 47, 10]` tensor, meaning it's actually `[B, T, 47, 3]`. The function works correctly regardless (it's shape-agnostic via reshaping), but the misleading docstring could cause errors in future maintenance.

More importantly: in `online_augment` (line 455-502), the speed warp extracts `xyz = x[..., :3]` and `mask = x[..., 9:10]`, warps XYZ, recomputes kinematics, and reconstructs. This is correct — warp XYZ first, recompute kinematics — matching the inviolable constraint.

**Why it matters:** Low severity for now, but misleading docs increase maintenance risk.

**How to validate:** Code review (already done). The implementation is correct.

**Proposed fix:** Update the docstring from `[B, T, 42, 3]` to `[B, T, N, 3]` where `N=47`.

---

### ISSUE 6 — Stage 2 Encoder Duplication Drift Risk

**What:** `train_stage_2.py` duplicates the entire `DSGCNEncoder` class (lines 39-174) from `train_stage_1.py`. This is a copy-paste pattern. If either file is updated independently, the encoder architectures diverge, and Stage 2 won't load Stage 1's pretrained encoder weights.

**Why it matters:** A single-character difference in any layer name, shape, or initialization will cause a silent weight mismatch or a crash when loading `encoder_state_dict`.

**How to validate:**
```bash
diff <(grep -n "class DSGCNEncoder" -A 100 src/train_stage_1.py) \
     <(grep -n "class DSGCNEncoder" -A 100 src/train_stage_2.py)
```

**Proposed fix:** Long-term: factor `DSGCNEncoder` into a shared module (e.g., `src/models/encoder.py`) and import in both stages. Short-term: add a hash-based assertion at Stage 2 load time that compares encoder parameter names/shapes against the checkpoint.

---

### ISSUE 7 — Stage 2 Online Augmentation Ignores Variable Sequence Lengths

**What:** `train_stage_2.py:online_augment` (line 523-557) applies rotation, scale, and noise to `x_pad` which is a padded tensor `[B, max_T, 47, 10]`. The rotation and noise are applied uniformly across the entire padded tensor, including zero-padded regions. This means:
1. Rotation rotates zero vectors to non-zero vectors in padded frames.
2. Noise adds random values to padded frames that should be zero.
3. The speed warp loop at line 535-543 warps padded regions.

The model later reads `x[b, :x_lens[b]]`, so the padded region is ultimately ignored in the forward pass. However, the augmentation wastes compute on padding, and the speed warp with `temporal_speed_warp_np` applied to the full padded tensor produces incorrect warping because it treats padding as real frames.

**Why it matters:** The speed warp (line 534-543) is the dangerous one. If applied to the full padded sequence (which may be much longer than the actual data), it corrupts the temporal structure. Currently the warp probability is 0.3 at batch level on top of per-clip 0.3 in the dataset, which compounds augmentation.

**How to validate:** Check whether the per-sample speed warp at lines 535-543 uses the padded length or actual length. It uses `x[b, :, :, :3]` (full length), not `x[b, :x_lens[b]]`.

**Proposed fix:** Either:
1. Move batch-level augmentation before padding in `collate_ctc`, or
2. Pass `x_lens` to `online_augment` and only warp/rotate within valid regions, or
3. Remove the batch-level speed warp entirely (the dataset already does per-clip warping at 0.3 probability).

Option 3 is simplest and avoids the double-warp issue.

---

### ISSUE 8 — Geo Features Compute Hand-to-Face Distances on Zero-Masked Face Nodes

**What:** In `DSGCNEncoder._compute_geo_features` (Stage 1 line 164-194, Stage 2 line 147-161), face-related features (wrist-to-nose, wrist-to-chin, etc.) are computed from `xyz[:,:,NOSE_NODE]`, etc. When face landmarks are not detected, these nodes are zero-vectors (no face detected = `mask[:, 42:47, 0] = 0.0` but XYZ is still zero).

The GCN face gating at line 201 (`h[:, :, 42:47, :] = h[:, :, 42:47, :] * face_mask`) gates the **projected features** but not the raw XYZ used for geo features. So the `face_feats` list computes distances from hand wrists to origin (0,0,0) rather than to actual face positions, producing meaningless but non-zero values.

**Why it matters:** For clips without face detection (which the audit suggests is common), the geo features inject noise. The 10 face geo features (out of 34 total) carry garbage values, degrading the geo projection layer.

**How to validate:** Count the fraction of training clips where `face_mask.max() == 0`. If significant (>20%), the noise from face geo features is substantial.

**Proposed fix:** Gate the face geo features by the face mask:
```python
# After computing face_feats, gate them
face_mask_flat = face_mask[:, :, 0, 0]  # [B, T] — 1 if any face node detected
for i in range(len(face_feats)):
    face_feats[i] = face_feats[i] * face_mask_flat
```

---

### ISSUE 9 — `build_language_model.py` Uses `eval()` for Deserialization (Security Risk)

**What:** `NgramLanguageModel.load()` at line 82:
```python
model.ngram_counts = {eval(k): v for k, v in data['ngram_counts'].items()}
model.context_counts = {eval(k): v for k, v in data['context_counts'].items()}
```

This uses `eval()` on strings from a JSON file to reconstruct tuple keys. If the JSON file is ever modified (accidentally or maliciously), `eval()` can execute arbitrary Python code.

**Why it matters:** Security vulnerability. If the model file is shared or downloaded from an untrusted source, this is an arbitrary code execution vector.

**How to validate:** Code review (confirmed).

**Proposed fix:** Replace `eval()` with `ast.literal_eval()`:
```python
import ast
model.ngram_counts = {ast.literal_eval(k): v for k, v in data['ngram_counts'].items()}
```

---

### ISSUE 10 — Stage 3 Training Data Vocab May Not Align with Stage 2 Output Glosses

**What:** Stage 3 trains on `slt_stage3_dataset_v2.csv` generated by `generate_stage3_data_v2.py`. The gloss sequences in this CSV must use the same canonical labels that Stage 2 outputs. Stage 2's gloss vocabulary comes from `manifest.json` (which uses alias-resolved labels). But if `generate_stage3_data_v2.py` was generated before the alias resolution was added to `extract.py`, it may contain stale composite labels.

**Why it matters:** At deploy time, Stage 2 decodes glosses like `["ALSO", "HE", "WE"]`, but if Stage 3 was trained on `["ALSO_SAME", "HE_SHE", "US_WE"]`, the T5 model won't recognize the canonical forms and will produce degraded translations.

**How to validate:**
```python
import pandas as pd, json
df = pd.read_csv("slt_stage3_dataset_v2.csv")
stage3_glosses = set()
for g in df["gloss"]:
    stage3_glosses.update(g.split())
with open("ASL_landmarks_float16/manifest.json") as f:
    manifest_glosses = set(json.load(f).values())
print("In Stage3 but not manifest:", stage3_glosses - manifest_glosses)
print("In manifest but not Stage3:", manifest_glosses - stage3_glosses)
```

**Proposed fix:** Re-generate Stage 3 data AFTER fresh extraction, using the canonical labels from the new manifest. `generate_stage3_data_v2.py` should read the manifest to get the current label set.

---

## PRIORITIZED IMPROVEMENT PLAN

### Phase 1: Quick Wins (pre-extraction / no retraining needed)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 1.1 | Fix verification scripts: `(32,42,10)` -> `(32,47,10)` | `verify_stage1_label_integrity.py`, `verify_stage2_label_integrity.py` | 5 min |
| 1.2 | Add pre-extraction manifest sanity check script | New: `src/validate_manifest.py` | 30 min |
| 1.3 | Fix `build_language_model.py`: unify format to pickle, use canonical labels, read vocab from manifest | `src/build_language_model.py` | 1-2 hrs |
| 1.4 | Replace `eval()` with `ast.literal_eval()` | `src/build_language_model.py:82-83` | 5 min |
| 1.5 | Update `temporal_speed_warp` docstring from 42 to 47 | `src/train_stage_1.py:340` | 2 min |

### Phase 2: Medium Effort (during/after extraction, before training)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 2.1 | Gate face geo features by face mask in `_compute_geo_features` | `train_stage_1.py`, `train_stage_2.py` (both encoder copies) | 30 min |
| 2.2 | Remove batch-level speed warp from Stage 2 `online_augment` (dataset already does per-clip) | `train_stage_2.py:534-543` | 15 min |
| 2.3 | Re-generate Stage 3 training CSV using canonical labels from new manifest | `generate_stage3_data_v2.py` | 1 hr |
| 2.4 | Add post-extraction label distribution logging & per-class sample count check | `extract.py` (add to `_write_manifest`) | 30 min |
| 2.5 | Run `validate_manifest.py` after extraction to catch alias failures | CI/workflow step | 10 min |

### Phase 3: Longer-Term (after initial training validates)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 3.1 | Factor DSGCNEncoder into shared `src/models/encoder.py` to eliminate copy-paste drift | `train_stage_1.py`, `train_stage_2.py`, new `src/models/encoder.py` | 2-3 hrs |
| 3.2 | Build LM from actual Stage 2 decoded outputs (not hardcoded patterns) to get realistic bigram stats | `build_language_model.py` | 2 hrs |
| 3.3 | Add confused-pair analysis: for each class with >30% mismatch, check if it's a labeling error or a genuine visual ambiguity (e.g., ABOUT/WHEN) | New verification analysis | 3 hrs |
| 3.4 | Consider learnable face gate (scalar per-frame) instead of binary mask to handle partial face detection | `DSGCNEncoder` | 2 hrs |
| 3.5 | Add end-to-end smoke test: extract 5 videos -> train Stage 1 (3 epochs) -> train Stage 2 (3 epochs) -> build LM -> run camera_inference on a saved video | New `tests/test_e2e_smoke.py` | 3 hrs |

---

## RISK REGISTER FOR FRESH EXTRACT-AND-TRAIN RUN

| Risk | Likelihood | Impact | Detection | Mitigation |
|------|-----------|--------|-----------|------------|
| **LABEL_ALIASES miss a folder name** -> ghost class with 1-5 samples | Medium | HIGH (CTC blank predictions for that class) | Post-extraction: check all labels in manifest appear >10 times; no label is also a key in LABEL_ALIASES | Review `ls "data/raw_videos/ASL VIDEOS/"` against LABEL_ALIASES before extraction |
| **Face detection rate too low** -> face geo features are noise | Medium | MEDIUM (degrades geo projection) | Log face detection % per class during extraction | Gate face geo features (Issue 8 fix). Log `face_ever` ratio in manifest |
| **Stage 1 checkpoint not compatible with Stage 2 encoder copy** | Low (if code unchanged) | HIGH (Stage 2 training crashes or silently loads wrong weights) | Check `encoder_state_dict` keys match between Stage 1 ckpt and Stage 2 model | Diff the two encoder class definitions before starting Stage 2 |
| **build_language_model.py runs before manifest exists** | Medium | HIGH (LM has no real vocab) | Script will print "No gloss sequences found" | Run LM build AFTER extraction. Add manifest existence check with clear error |
| **Stage 3 CSV uses stale labels** | High (if not regenerated) | HIGH (T5 trained on wrong gloss vocab) | Run vocab diff script (see Issue 10) | Always regenerate Stage 3 CSV after fresh extraction |
| **MediaPipe version difference** between extraction and inference environments | Low | MEDIUM (landmark coordinate drift) | Pin `mediapipe` version in `requirements.txt` | Add version logging to extract.py output |
| **Augmented .npy files counted as separate samples in stratified split** -> data leakage | Medium (if augmentation enabled in extract.py) | HIGH (inflated val/test accuracy) | Check if augmented variants of same video appear in both train and val | Currently augmentation is disabled in extract.py (line 97). If enabled: split by video stem before augmentation |
| **CTC `input_lengths < target_lengths`** for long sequences | Low | HIGH (training crash) | The assert at `train_stage_2.py:749` will catch this | Already handled. But test with max_len=8 sequences to be safe |

---

## MINIMUM TESTS BEFORE CLAIMING IMPROVEMENTS

### After Extraction
1. **Shape check:** `python -c "import numpy as np, glob; files=glob.glob('ASL_landmarks_float16/*.npy'); arr=np.load(files[0]); assert arr.shape == (32,47,10), arr.shape; print(f'OK: {len(files)} files, shape {arr.shape}')"`
2. **Manifest consistency:** Every `.npy` file in the output dir has an entry in `manifest.json`, and every manifest entry maps to a label that is NOT a key in `LABEL_ALIASES`.
3. **Label distribution:** No label has fewer than 5 samples. Print class counts sorted ascending.
4. **Face detection rate:** Log percentage of clips where `face_ever=True`. Expected: 60-90% depending on video quality.

### After Stage 1 Training
5. **Verification script:** Run updated `verify_stage1_label_integrity.py` (with 47-node fix). Expect mismatch rate < 10% for clean labels.
6. **Confusion matrix:** Check top-10 confused pairs. If a pair like ABOUT/WHEN shows >50% cross-confusion, consider merging or relabeling.
7. **Test set accuracy:** Should be within 3% of validation accuracy (no train/val leakage).

### After Stage 2 Training
8. **WER on test set:** Should be < 50% (synthetic data). On held-out real sequences if available.
9. **CTC decode sanity:** Decode 10 synthetic sequences and manually check gloss output.

### After LM Build
10. **Vocab alignment:** `assert set(lm.vocab) == set(manifest_labels)` — the LM vocab must exactly match the Stage 2 gloss vocabulary.
11. **Format check:** `camera_inference.py._load_gloss_lm()` loads without error.

### After Stage 3 Training
12. **Inference test:** Run the test cases in `train_stage_3.py` (lines 309-325) using canonical gloss labels.
13. **OOV check:** No gloss from Stage 2 output should be OOV in the Stage 3 training data.

### End-to-End
14. **Camera inference smoke test:** Run `camera_inference.py`, sign 3 known signs, verify the pipeline produces sensible output at each stage.

---

## DEPENDENCY GRAPH FOR FRESH RUN

```
extract.py (Stage 0)
    |
    v
manifest.json  <-- validate_manifest.py (new)
    |
    +---> train_stage_1.py (Stage 1)
    |         |
    |         v
    |     best_model.pth
    |         |
    |         +---> verify_stage1_label_integrity.py (fix: 47 nodes)
    |         |
    |         +---> train_stage_2.py (Stage 2, loads Stage 1 encoder)
    |                   |
    |                   v
    |               stage2_best_model.pth (contains gloss_to_idx, idx_to_gloss)
    |
    +---> generate_stage3_data_v2.py (must use canonical labels from manifest!)
    |         |
    |         v
    |     slt_stage3_dataset_v2.csv
    |         |
    |         +---> train_stage_3.py (Stage 3)
    |                   |
    |                   v
    |               slt_conversational_t5_model/
    |
    +---> build_language_model.py (must use canonical labels from manifest!)
              |
              v
          weights/gloss_bigram_lm.pkl  (fix: format + vocab alignment)
              |
              v
          camera_inference.py (loads Stage 2 ckpt + T5 model + LM + manifest)
```

**Critical ordering:**
1. Extract first (produces manifest.json)
2. Validate manifest
3. Train Stage 1 (needs manifest + .npy files)
4. Train Stage 2 (needs Stage 1 checkpoint + manifest + .npy files)
5. Generate Stage 3 data (needs manifest for canonical labels)
6. Train Stage 3 (needs Stage 3 CSV)
7. Build LM (needs manifest for canonical labels)
8. Deploy (needs all artifacts)

Steps 5 and 7 can run in parallel after step 1. Steps 5-6 and 3-4 are independent chains that both depend on step 1.
