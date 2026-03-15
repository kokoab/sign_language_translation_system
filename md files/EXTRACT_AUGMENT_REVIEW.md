# SLT Pipeline v4.2 — Code Review

---

## Review Round 1 (Original Code)

### 1. Critical Bugs (Will Crash Training)

#### 1a. Duplicate Hand Labels Break `interp1d`

MediaPipe can classify both detected hands with the same label (e.g., two "Left" in one frame). When this happens, the same `frame_idx` is appended to `l_valid` twice. `scipy.interpolate.interp1d` requires **strictly monotonic** x-values and will raise a `ValueError`.

**File:** `process_dataset`, inner detection loop (~line 152–160)

```python
# Fix: take only the highest-confidence detection per label per frame
if h_label == "Left":
    if not l_valid or l_valid[-1] != frame_idx:
        l_seq.append(coords); l_valid.append(frame_idx)
    # else: duplicate left in same frame — skip
```

**Status: FIXED in v4.2**

#### 1b. Missing `cap.release()` on Early Skip

When `total_frames < cfg.min_raw_frames`, the loop hits `continue` without releasing the `VideoCapture`. This leaks file descriptors and on large datasets will eventually hit the OS limit and crash.

```python
if total_frames < cfg.min_raw_frames:
    cap.release()          # ← add this
    total_skipped += 1
    continue
```

**Status: FIXED in v4.2** — Restructured so `cap.release()` is called inside the `with` block before skip checks.

---

### 2. Mathematical / Theoretical Flaws

#### 2a. Centering Origin Shifts Between Frames — False Velocity Spikes

This was the most architecturally significant issue. In the original `normalize_sequence`, the centering reference switched dynamically per frame:

| Condition | Center |
|---|---|
| Both hands | `(L_WRIST + R_WRIST) / 2` |
| Left only | `L_WRIST` |
| Right only | `R_WRIST` |

If frame `i` had only the left hand and frame `i+1` had both hands, all landmarks suddenly shifted by **half the inter-hand distance**, creating phantom velocity spikes and spatial discontinuities.

**Status: FIXED in v4.2** — Replaced with global sequence centering using `np.median` of all non-zero wrist positions. See Round 2 notes for assessment.

#### 2b. Rotation Augmentation Interacts Poorly with Mixed Centering

Even with a centering fix, the 3D rotation in `generate_augmentations` applies a single rotation matrix uniformly across all frames. With per-frame centering, the rotation axis origin was inconsistent across frames. With ±10° range, this was **tolerable** but mathematically impure.

**Status: RESOLVED by 2a fix** — Global centering means the rotation origin is now consistent across all frames.

#### 2c. `compute_kinematics` Boundary Formulas Are Asymmetric in Order

The interior used a **second-order** central difference (`/2.0`), but the boundaries used **first-order** forward/backward differences. This created a systematic 2x magnitude difference at boundary frames.

```python
vel[0]  = vel[1]
vel[-1] = vel[-2]
```

**Status: FIXED in v4.2**

---

### 3. Hidden Edge Cases

#### 3a. `CAP_PROP_FRAME_COUNT` vs Actual Frames Read

`total_frames` from OpenCV can be **wrong** for variable-frame-rate or certain codec/container combos.

```python
total_frames = frame_idx  # trust actual reads over metadata
```

**Status: FIXED in v4.2**

#### 3b. MediaPipe Tracker State Leaks Between Videos

With `static_image_mode=False`, MediaPipe maintains internal tracking state between videos.

**Status: FIXED in v4.2** — Detector is now instantiated per-video via `with` context manager.

---

### 4. DS-GCN / Transformer Alignment Concerns

#### 4a. Zero Sentinels and GCN Bias Terms

If the DS-GCN has **bias terms** or **batch normalization**, the zero sentinel will not remain zero after the first layer: `output = W @ 0 + b = b ≠ 0`.

**Status: FIXED in v4.2** — Added a binary presence mask channel (dim 10) so the downstream model can gate sentinel nodes.

#### 4b. Feature Dimension Ordering for the GCN

Output shape is `(32, 42, 10)` — (T, N, C). Many GCN implementations expect **(N, C, T)** or **(batch, C, T, N)**. Double-check your `DS-GCN` input spec.

**Status: Acknowledged — verify downstream.**

---

### 5. Performance Bottlenecks

#### 5a. `normalize_sequence` Python Loop

**Status: FIXED in v4.2** — Replaced with vectorized operations.

#### 5b. Augmentations Computed Serially

The 15 augmentations per video are independent and could be parallelized.

**Status: Open — low priority for small datasets.**

---

### 6. Minor Issues

| Issue | Status |
|---|---|
| `np.random.seed(42)` is global | **FIXED** — Now uses `np.random.default_rng()` |
| Hardcoded `42` and `3` in `temporal_resample` reshape | **FIXED** — Uses `N, P, C = seq.shape` |
| Log says "16 samples saved" always | **FIXED** — Uses `len(aug_variants) + 1` |
| No file collision guard | **FIXED** — Added `v_idx` suffix |

---
---

## Review Round 2 (Updated Code v4.2)

### R2-1. CRITICAL REGRESSION: 1-Frame Fallback Removed

The original code had a special case in `interpolate_hand` for single-detection hands:

```python
if len(valid_indices) == 1:
    filled = np.tile(flat[0], (total_frames, 1))
    return filled.reshape(total_frames, 21, 3).astype(np.float32)
```

**This was removed in the update.** `scipy.interpolate.interp1d` with `kind='linear'` requires **at least 2 data points**. With 1 valid index, it raises:

```
ValueError: x and y arrays must have at least 2 entries
```

**When this triggers:** A hand detected in exactly 1 frame. The `max_missing_ratio` check only validates the *dominant* hand (via `max(len(l_valid), len(r_valid))`). The secondary hand can have `len(valid) == 1` and still pass validation. Example: 10-frame video, left hand detected in 8 frames, right hand detected in 1 frame — passes the 40% threshold but crashes on the right hand's `interp1d`.

**Fix — restore the fallback:**

```python
def interpolate_hand(hand_seq: np.ndarray, valid_indices: list, total_frames: int) -> np.ndarray:
    if not valid_indices:
        return np.zeros((total_frames, 21, 3), dtype=np.float32)

    flat = hand_seq.reshape(len(hand_seq), -1)

    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, 21, 3).astype(np.float32)

    f_interp = interp1d(
        valid_indices, flat, axis=0, kind='linear',
        bounds_error=False, fill_value=(flat[0], flat[-1])
    )
    return f_interp(np.arange(total_frames)).reshape(total_frames, 21, 3).astype(np.float32)
```

---

### R2-2. MODERATE: `np.median([])` Crash in Bone Length Filtering

In `normalize_sequence`, bone lengths are filtered before computing the median:

```python
median_scale = np.median([b for b in bone_lengths if b > 1e-6]) + 1e-8
```

If **all** bone lengths are ≤ 1e-6 (possible with severely corrupted detections or extreme miniature hand poses), the list comprehension returns `[]`, and `np.median([])` raises:

```
ValueError: cannot convert float NaN to integer
```

**Fix — guard the filter:**

```python
filtered = [b for b in bone_lengths if b > 1e-6]
if filtered:
    median_scale = np.median(filtered) + 1e-8
    norm_seq /= median_scale
```

---

### R2-3. MINOR: Global Centering — Design Assessment

The global centering approach (`np.median` of all non-zero wrist positions across time) is a sound fix for the original per-frame discontinuity issue. Specific notes:

**Strengths:**
- Eliminates per-frame centering discontinuities entirely — no more phantom velocity spikes
- Median is robust to outlier wrist positions from bad detections
- Consistent rotation origin across all frames (resolves 2b)

**Tradeoff to be aware of:**
- If the signer translates significantly during the video (walks, leans), the global center is the spatial median of their trajectory. Frames at the extremes will have larger absolute coordinates. For typical ASL dataset videos (stationary signer, fixed camera), this is fine. If you later add data with significant signer movement, consider per-frame centering with temporal smoothing instead.

**No action needed** for current use case.

---

### R2-4. MINOR: Mask Channel Is Sequence-Level, Not Frame-Level

In `compute_kinematics`, the presence mask is set per-sequence:

```python
if l_ever: mask[:, 0:21, 0] = 1.0
```

This means every frame gets `mask=1.0` for a hand that was detected **at least once**, including frames that were originally missing and filled by interpolation. This is a design choice, not a bug — the interpolated frames represent "best-estimate" poses and the model should process them. But be aware:

- The DS-GCN cannot distinguish "real detection" from "interpolated guess" at the frame level.
- If you later need frame-level confidence, you'd need to pass the original `l_valid`/`r_valid` indices through and build a per-frame mask.

**No action needed** unless frame-level masking becomes a requirement.

---

### R2-5. INFO: `v_idx` Collision Guard Depends on File System Order

The collision suffix `v_idx` is the enumeration index from `os.walk`, which does **not** guarantee consistent ordering across runs or platforms. If files are added/removed between runs, the same video gets different suffixes. For a one-time extraction this is fine, but for reproducible pipelines consider using a hash of the filename:

```python
import hashlib
suffix = hashlib.md5(video_name.encode()).hexdigest()[:6]
stem = f"{Path(video_name).stem}_{suffix}"
```

**Status: FIXED in v4.3** — Now uses MD5 hash of the filename.

---

## Review Round 3 (Updated Code v4.3)

All Round 2 issues have been addressed. No new bugs found.

### Fixes verified:
- **R2-1 (1-frame fallback):** Restored correctly. The `len(valid_indices) == 1` branch tiles the single frame across all `total_frames`, preventing the `interp1d` crash.
- **R2-2 (`np.median([])` crash):** Guarded with a nested `if filtered_bones:` check. If all bone lengths are ≤ 1e-6, normalization gracefully skips scaling rather than crashing.
- **R2-5 (Collision guard):** Switched from `v_idx` to `hashlib.md5(video_name.encode()).hexdigest()[:6]`, giving deterministic, filesystem-order-independent suffixes.

### Notes for downstream integration:
- **Output shape:** `(32, 42, 10)` — channels are `[x, y, z, vx, vy, vz, ax, ay, az, mask]`.
- **Mask channel (dim 9):** `1.0` = hand detected at least once in sequence, `0.0` = never detected (sentinel). This is sequence-level, not frame-level. Use it to gate GCN bias terms post first layer: `output = gcn(x) * mask`.
- **Coordinate frame:** All frames centered on the global median wrist position. Rotation augmentations are origin-consistent.

---

## Final Summary

| Priority | Item | Status |
|---|---|---|
| **P0** | Duplicate hand label guard (1a) | **FIXED** |
| **P0** | `cap.release()` on skip (1b) | **FIXED** |
| **P0** | Use `frame_idx` as `total_frames` (3a) | **FIXED** |
| **P0** | 1-frame fallback regression (R2-1) | **FIXED** |
| **P1** | Consistent centering origin (2a) | **FIXED** (global centering) |
| **P1** | GCN bias/sentinel interaction (4a) | **FIXED** (mask channel) |
| **P1** | `np.median([])` crash (R2-2) | **FIXED** |
| **P2** | Tracker state leak (3b) | **FIXED** |
| **P2** | Kinematics boundary order (2c) | **FIXED** |
| **P2** | File collision guard (R2-5) | **FIXED** (MD5 hash) |
| **P3** | Vectorize normalization (5a) | **FIXED** |
| **P3** | Local RNG, log correctness (6) | **FIXED** |
| **P3** | Frame-level vs sequence-level mask (R2-4) | **Open — monitor** |

**Pipeline status: Production-ready for DS-GCN → Transformer CTC architecture.**
