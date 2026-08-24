# SLT Data & Training Improvement Suggestions

Based on analysis of the extraction pipeline, training scripts, raw video folder structure, and the confusion patterns you reported.

---

## Problem 1: Duplicate Videos Across Separate Class Folders

**Root cause (critical):** Many video files are physically duplicated across class folders that have visually identical or near-identical ASL signs.

| Pair | Shared Video Files | Why |
|------|--------------------|-----|
| DRIVE / CAR | 90 identical files | Same ASL handshape; context distinguishes meaning |
| HARD / DIFFICULT | 60 identical files | Same sign in ASL |
| MAKE / CREATE | 90 identical files | Same sign in ASL |

This is the **#1 reason** the model confuses these pairs. The same exact video is labeled "DRIVE" in one folder and "CAR" in the other. No model can learn to distinguish them because they are literally the same input with different labels.

### Fix: Merge synonym classes

These pairs are **not distinct signs in ASL** -- they are the same sign used in different English contexts. The model should not be asked to distinguish them at Stage 1.

**Recommended merges:**

| Merge Into | Absorb | Rationale |
|------------|--------|-----------|
| `DRIVE_CAR` or `CAR` | DRIVE + CAR | Same sign |
| `HARD_DIFFICULT` or `HARD` | HARD + DIFFICULT | Same sign |
| `MAKE_CREATE` or `MAKE` | MAKE + CREATE | Same sign |
| `EAT_FOOD` or `EAT` | EAT + FOOD | Visually very similar |
| `WRITE_FIX` | WRITE + FIX | If truly confusing (verify visually) |

**Implementation:** Either:
1. **Physical merge:** Move all videos from both folders into one folder. Simpler.
2. **Label alias map in `extract.py`:** Add a dict that maps folder names to canonical labels before saving. Example:

```python
LABEL_ALIASES = {
    "DRIVE": "DRIVE_CAR",
    "CAR": "DRIVE_CAR",
    "HARD": "HARD_DIFFICULT",
    "DIFFICULT": "HARD_DIFFICULT",
    "MAKE": "MAKE_CREATE",
    "CREATE": "MAKE_CREATE",
    "EAT": "EAT_FOOD",
    "FOOD": "EAT_FOOD",
}
```

Then at line 303 in `extract.py`:
```python
label = Path(root).name
label = LABEL_ALIASES.get(label, label)  # Canonicalize
```

**Stage 3 handles disambiguation.** The merged label `DRIVE_CAR` gets passed to Stage 2 as a gloss, and Stage 3 (Flan-T5) uses sentence context to decide whether to output "drive" or "car" in English. This is the correct level for that distinction.

---

## Problem 2: Composite Folder Names Split Incorrectly

**Root cause:** Folders named `ALSO_SAME`, `MARKET_STORE`, `US_WE`, `FEW_SEVERAL`, `I_ME`, `HE_SHE`, `HIS_HER` use underscores to denote the same-sign pair. `extract.py` uses `Path(root).name` which preserves the full folder name (e.g., `ALSO_SAME`). However, the manifest shows only the first part (`ALSO`, not `ALSO_SAME`).

This means somewhere in your pipeline a split on `_` is happening, losing the second synonym. The manifest maps:
- `ALSO_SAME` folder -> label `ALSO` (loses `SAME`)
- `HE_SHE` folder -> label `HE` (loses `SHE`)
- `I_ME` folder -> label `I` (loses `ME`)

### Fix: Decide on canonical labels for these

**Option A (Recommended):** Keep composite names as-is and let Stage 3 disambiguate:
- `ALSO_SAME` stays as class `ALSO_SAME`
- Stage 3 learns to translate `ALSO_SAME` -> "also" or "same" based on context

**Option B:** Pick one canonical label per pair:
- `ALSO_SAME` -> `SAME`
- `MARKET_STORE` -> `STORE`
- `US_WE` -> `WE`
- `FEW_SEVERAL` -> `FEW`
- `I_ME` -> `I`
- `HE_SHE` -> `HE`
- `HIS_HER` -> `HIS`

Add these to the same `LABEL_ALIASES` dict.

**Either way, fix the manifest builder** so it doesn't silently split on underscores.

---

## Problem 3: AT -> HE Confusion

AT is being predicted as HE with high confidence (0.94+). These are genuinely different ASL signs, so this is likely a **data quality issue** -- the AT folder may contain mislabeled videos that are actually showing the sign for HE, or the signs are visually too similar at the current extraction resolution.

### Fix:
1. **Manually audit the AT folder.** Watch 10-15 of the most confidently-mispredicted videos. If they look like HE, remove or relabel them.
2. If AT and HE are genuinely visually similar, consider merging them and disambiguating in Stage 3.
3. Check `FATHER` / `MOTHER` confusion similarly -- these differ only by chin vs forehead contact, which MediaPipe hand-only tracking may not capture since it doesn't track face landmarks.

---

## Problem 4: MediaPipe Hands Cannot Distinguish Signs Requiring Face/Body Context

Several confusing pairs (FATHER/MOTHER, AT/HE) differ by where the hand touches the face or body. MediaPipe Hands only tracks **hand landmarks** (21 points per hand) -- it does **not** provide face or body position relative to the hands.

### Longer-term fix options (in order of effort):

1. **Merge visually-identical-to-hands pairs** (lowest effort, do this now)
2. **Add MediaPipe Holistic or Face Mesh** to capture the hand's position relative to the face. This would add ~5 face reference points (nose, chin, forehead, left ear, right ear) to the feature vector, changing the tensor shape from `[32, 42, 10]` to `[32, 47, 10]`. This is a significant architecture change but would resolve many confusion pairs.
3. **Add relative-position features** in extraction: compute the hand centroid's Y position relative to the frame center as an extra channel. Signs near the forehead (FATHER) vs chin (MOTHER) would then differ. This is lighter than full Holistic but less general.

---

## Problem 5: 320 Classes May Be Too Many for Current Data Volume

With ~39k extracted samples across 320 classes, you have ~122 samples per class on average. Given that many classes share duplicate videos with other classes, effective samples per class is even lower. This is thin.

### Recommendations:
1. **Merge synonym pairs (as above) to reduce to ~300-305 classes.** This immediately increases samples-per-class.
2. **Drop classes with < 50 unique videos** after deduplication. Move them to a "future" set.
3. **Consider a two-tier vocabulary:**
   - Tier 1 (~100 most common signs): High accuracy target
   - Tier 2 (remaining ~200): Acceptable lower accuracy, merged aggressively

---

## Problem 6: Extraction-Level Improvements

### 6a. Add a manifest during extraction (not after)

`extract.py` should write the manifest as it processes videos, not rely on filename parsing later. Add at the end of `run_pipeline()`:

```python
manifest = {}
for f in os.listdir(out_path):
    if f.endswith('.npy'):
        # Parse label from filename prefix
        label = ... # extract from naming convention
        manifest[f] = label
with open(out_path / 'manifest.json', 'w') as fp:
    json.dump(manifest, fp, indent=2)
```

This eliminates label-parsing bugs downstream.

### 6b. Increase `min_raw_frames` for better quality control

Currently `min_raw_frames = 5`. Very short videos (< 10 frames) tend to produce poor landmarks. Consider raising to `8` or `10` to filter out low-quality clips that add noise.

### 6c. Log per-class extraction success rate

Add a counter that tracks how many videos passed vs failed per label. Classes with high failure rates (> 30%) likely have data quality issues.

---

## Training-Level Suggestions

### Stage 1 (`train_stage_1.py`)

**7a. Class-aware hard-negative mining:**
The current `WeightedRandomSampler` balances class frequency but doesn't address inter-class confusion. Consider adding a **confusion-aware loss** or **focal loss** that upweights the loss on frequently-confused pairs (DRIVE/CAR, HARD/DIFFICULT). After merging synonyms this becomes less critical, but still useful for remaining confusion pairs.

**7b. Increase label smoothing for merged classes:**
After merging synonym pairs, the remaining confusions (e.g., AT/HE) may benefit from higher label smoothing (0.15-0.20) to prevent overconfident wrong predictions.

**7c. Curriculum learning adjustment:**
The current curriculum focuses on single-hand vs two-hand signs. After fixing the data issues, consider a curriculum that trains easy-to-distinguish signs first (signs with unique hand shapes) and introduces visually-similar pairs later.

**7d. Add confusion-pair contrastive loss (optional, advanced):**
Add an auxiliary loss that explicitly pushes apart embeddings of known confusion pairs. This requires identifying which classes are commonly confused (you already have this data from the mismatch analysis).

### Stage 2 (`train_stage_2.py`)

**7e. Ensure merged labels propagate correctly:**
The CTC vocabulary must reflect the merged label set. If you merge DRIVE+CAR into DRIVE_CAR, the `gloss_to_idx` mapping must include `DRIVE_CAR` (not separate entries for DRIVE and CAR).

**7f. Increase synthetic dataset variety:**
`SyntheticCTCDataset` generates sequences by randomly sampling glosses. After merging, verify that merged labels appear correctly in the generated sequences. Also consider weighted sampling that produces more sequences containing commonly-confused glosses.

### Stage 3 (`train_stage_3.py`)

**7g. Add merged-label disambiguation training data:**
`generate_stage3_data_v2.py` needs new entries that map merged glosses to context-appropriate English:
```
DRIVE_CAR -> "I'm driving" (when followed by location verb)
DRIVE_CAR -> "the car" (when preceded by adjective or determiner context)
HARD_DIFFICULT -> "it's hard" / "that's difficult"
MAKE_CREATE -> "I'll make it" / "I need to create it"
EAT_FOOD -> "I'm eating" / "some food"
```

This is where the English disambiguation happens. Stage 3 should be given enough context-dependent examples to learn when "DRIVE_CAR" means "drive" vs "car."

**7h. Add composite labels to the vocabulary lists in `generate_stage3_data_v2.py`:**
The current vocabulary lists (verbs_dict, nouns, etc.) reference "DRIVE", "CAR", etc. as separate words. After merging, update these to use the merged tokens, and add template generators that handle disambiguation.

### Language Model (`build_language_model.py`)

**7i. Train LM on merged vocabulary:**
After merging classes, the N-gram LM must be retrained with the new vocabulary. Merged tokens like `DRIVE_CAR` should appear in the LM training sequences.

---

## Recommended Action Order

1. **Identify and merge synonym pairs** (DRIVE/CAR, HARD/DIFFICULT, MAKE/CREATE, EAT/FOOD) -- add `LABEL_ALIASES` to `extract.py`
2. **Fix composite folder label parsing** (ALSO_SAME, HE_SHE, etc.) -- decide canonical labels
3. **Re-run extraction** with the fixed labels
4. **Manually audit AT folder** for mislabeled videos
5. **Update Stage 3 data generator** with merged-label disambiguation examples
6. **Retrain Stage 1** on the cleaned dataset
7. **Retrain Stages 2 and 3** with the updated vocabulary
8. **Re-evaluate confusion matrix** -- remaining confusions after merge are the real problems to solve

---

## Summary of Root Causes

| Confusion Pattern | Root Cause | Fix |
|---|---|---|
| DRIVE <-> CAR | 90 identical videos in both folders | Merge classes |
| HARD <-> DIFFICULT | 60 identical videos in both folders | Merge classes |
| MAKE <-> CREATE | 90 identical videos in both folders | Merge classes |
| EAT -> FOOD | Very similar ASL signs | Merge classes |
| ALSO_SAME mismatch | Underscore label split bug | Fix label parsing |
| AT -> HE | Mislabeled data or face-contact difference invisible to hand-only tracking | Audit data; consider merge or adding face reference points |
| FATHER <-> MOTHER | Forehead vs chin contact; invisible to hand-only tracking | Add face landmarks or merge |
