# Extract & Augment v4.0 — Expert Analysis (Revised)

**Pipeline:** `extract_augment.py` (Two-Hand WORDS Pipeline)
**Architecture:** MediaPipe Hands → DS-GCN → Transformer Encoder (CTC) → mT5-small Decoder
**Target:** Stage 1 Feature Extraction & Augmentation for DS-GCN
**Date:** 2026-03-14

---

## Revision Note

Issues 1 and 2 from the original analysis have been **withdrawn**. Boundary clamping
(freezing a hand at its last known position when it drops out) is the correct strategy
for the DS-GCN + Transformer + CTC stack. A hard zero-out would create spatial
discontinuities that damage GCN edge features and produce false CTC alignment boundaries
in Stage 2.

The `np.any()` presence check works correctly under the clamping strategy:
- **Never-detected hand** → `interpolate_hand` returns pure zeros → `np.any()` = `False`
- **Partially-detected hand** → boundary clamp fills all frames → `np.any()` = `True`

---

## Issue 1 — WITHDRAWN: Presence Mask Not Needed

**Original claim:** `np.any()` is flawed; propagate a boolean visibility mask.

**Why it was wrong:** The proposed mask would force frames outside first/last detection
back to `[0.0, 0.0, 0.0]`, creating a spatial teleport. The DS-GCN computes features
from joint distances — a sudden teleport to the origin creates massive edge-weight spikes across the
graph. Boundary clamping (freezing the hand in place) produces a stationary subgraph with
near-zero deltas, which is physically plausible and safe for training.

**Verdict:** No fix needed. Current implementation is correct.

---

## Issue 2 — WITHDRAWN: Kinematic Spikes Not a Problem with Clamping

**Original claim:** Central-difference velocity creates phantom spikes at absence boundaries.

**Why it was wrong:** With boundary clamping, the hand freezes — it doesn't jump to zero.
Velocity at the clamped boundary is near-zero (frozen hand = no movement), which is the
correct physical signal. The spike scenario only occurs with hard zero-out, which we are
not doing.

**Verdict:** No fix needed. Current implementation is correct.

---

## Issue 3 (BUG — Still Valid): Jitter Augmentation Corrupts Never-Detected Hands

If a hand is never detected in the entire video, `interpolate_hand` correctly returns
all zeros. But jitter augmentation adds noise to joints 1-20 (only the wrist is protected):

```python
jitter[:, L_WRIST, :] = 0.0  # Only zeroes the wrist
jitter[:, R_WRIST, :] = 0.0
aug = base_data + jitter      # Joints 1-20 of a never-detected hand get noise
```

The DS-GCN now sees 20 joints hovering near zero with random noise instead of clean zeros.
The model can't cleanly distinguish "absent hand" from "hand at origin with micro-movement."

**Fix applied:** Sequence-level `np.any` check after jitter to re-zero never-detected hands.

```python
# After adding jitter:
l_ever = np.any(base_data[:, 0:21, :])
r_ever = np.any(base_data[:, 21:42, :])
if not l_ever: aug[:, 0:21, :] = 0.0
if not r_ever: aug[:, 21:42, :] = 0.0
```

---

## Issue 4 (Edge Case — Still Valid): Temporal Shift Wraps Frames Circularly

`np.roll` wraps the end of the sequence to the beginning. For a sign like "hello," frames
from the end-of-gesture appear at the start. This creates a physically impossible
discontinuity that is especially harmful for Stage 2's Transformer Encoder + CTC, which
relies on monotonic temporal alignment.

**Fix applied:** Edge-pad instead of circular wrap.

```python
if shift > 0:
    shifted = np.concatenate([np.tile(base_data[0:1], (shift, 1, 1)), base_data[:-shift]], axis=0)
elif shift < 0:
    shifted = np.concatenate([base_data[-shift:], np.tile(base_data[-1:], (-shift, 1, 1))], axis=0)
else:
    shifted = base_data.copy()
```

---

## Issue 5 (Confirmed Sound): Bone Scale After Center-Subtraction

Bone length computation after center subtraction is translation-invariant:

```
norm((MCP - center) - (WRIST - center)) = norm(MCP - WRIST)  ✓
```

**Mathematically sound. No fix needed.**

---

## Issue 6 (Edge Case — Low): MediaPipe Label Mirroring

MediaPipe's `"Left"` label refers to the hand on the left side of the *image*, which is
the signer's right hand in a front-facing camera. This is consistent across all videos
recorded with the same camera orientation, so the model will learn correctly. Only becomes
a problem if mixing selfie-mode and standard-mode recordings.

**Optional config flag if needed later:**

```python
mirror_handedness: bool = False  # Set True for front-facing camera recordings
```

---

## Revised Summary Table

| # | Severity | Issue | Status | Impact |
|---|----------|-------|--------|--------|
| 1 | ~~Critical~~ | ~~Presence mask~~ | **WITHDRAWN** | Clamping is correct for DS-GCN |
| 2 | ~~Critical~~ | ~~Kinematic spikes~~ | **WITHDRAWN** | Clamping produces near-zero vel at boundaries |
| 3 | **High** | Jitter corrupts never-detected hands | **FIXED** | DS-GCN can't distinguish noise from absence |
| 4 | **Medium** | `np.roll` wraps circularly | **FIXED** | Unphysical discontinuity hurts CTC alignment |
| 5 | None | Bone scaling correctness | Sound | Translation-invariant ✓ |
| 6 | Low | Mirror label ambiguity | Noted | Only if mixing camera orientations |

## Core Takeaway

The boundary clamping strategy (freeze hand at last known position) is the correct
approach for the DS-GCN → Transformer → mT5 architecture. The zero-sentinel (`0.0`)
is only used for hands that are **never detected** in the entire video, not for
per-frame absence. Issues 3 and 4 were the only real bugs — both have been patched.
