# Gemini Optimization Suggestions — Review

Assessment of optimization suggestions from Gemini for `extract_augment.py`.

Constraint: **No quality compromise.**

---

# Round 1 — v5.2 → v5.3 Suggestions

## 1. Initialize MediaPipe Once Per Worker — Gemini Rejects

**Agree with rejection.** We fixed this exact issue in B2 (see BUNDLING_REVIEW.md). A persistent per-worker detector leaks tracking state from video N into video N+1, producing phantom detections in the first 1-2 frames. Per-video instantiation via `with` context manager is correct.

---

## 2. `model_complexity=0` — Gemini Rejects

**Agree with rejection.** The `model_complexity=1` model has measurably better 21-landmark accuracy. With the "no compromise" constraint, this is off the table.

---

## 3. `chunksize` in `executor.map()` — Gemini Accepts

**Agree. Do this.**

With `chunksize=1` (default), each of the ~45K videos requires a separate IPC round-trip (serialize task tuple → pipe → deserialize). With `chunksize=50`, workers receive 50 tasks at once, reducing IPC calls from 45K to ~900. Zero quality impact — purely a dispatch optimization.

---

## 4. Replace `scipy.interp1d` with `numpy.interp` — Gemini Accepts

**Correct in principle, but the speedup is negligible.**

- `scipy.interp1d` has Python-level object construction overhead. For small arrays (~48 frames × 63 columns), this overhead dominates actual interpolation.
- `numpy.interp` is a single C function call with no object construction. But it only handles 1D, so a column loop is needed.
- Boundary behavior matches exactly: `numpy.interp` clamps to edge values by default, identical to `fill_value=(flat[0], flat[-1])`.

The real bottleneck is MediaPipe inference (~80%) and video decode (~15%). Interpolation is <1%.

The one genuine benefit: **dropping the `scipy` import**. With macOS `spawn` multiprocessing, each worker re-imports all modules at startup. `import scipy` adds ~200-500ms per worker — 1.6-4s of wasted startup time eliminated.

### Round 1 Summary

| Suggestion | Do it? | Quality impact | Speed impact |
|---|---|---|---|
| Per-worker MediaPipe init | **No** | Tracker state leak (B2) | — |
| `model_complexity=0` | **No** | Reduced landmark accuracy | — |
| `chunksize=50` | **Yes** | None | Reduces 45K IPC calls to ~900 |
| `numpy.interp` | **Optional** | None (byte-identical output) | Negligible per-video; eliminates scipy import overhead |

**Neither accepted suggestion compromises quality. Both produce byte-identical output.**

---

# Round 2 — v5.3 → v5.4 Suggestions

## 5. Fast-Fail Check on Frame Count — ACCEPT

```python
if 0 < total_est < cfg.min_raw_frames:
    cap.release()
    return 0
```

Skips obviously useless videos before spinning up MediaPipe. The `0 <` guard is critical — `CAP_PROP_FRAME_COUNT` returns 0 or -1 for some codecs (flagged in our v4.2 review), and those must NOT be skipped. With that guard, the only risk is a valid video whose metadata *incorrectly* reports 1-4 frames — extremely rare.

**No quality impact.** Worth doing.

---

## 6. Largest-First Sorting by File Size — ACCEPT

```python
all_videos.sort(key=lambda x: x[4], reverse=True)
```

Classic LPT (Longest Processing Time) scheduling. Without sorting, workers finish easy videos early, then one worker gets stuck on the last long video while 6 sit idle. Sorting by file size descending ensures long jobs start early and short jobs fill gaps at the end.

Extra `os.path.getsize()` per file adds ~1-2s to the scan phase across 45K — negligible. Requires adding `file_size` as 5th tuple element and unpacking with `_` in the worker.

**No quality impact.** Worth doing.

---

## 7. Dynamic Chunksize Formula — REJECT (use fixed 50)

```python
# v5.4's formula:
chunk = max(1, len(all_videos) // (safe_workers * 4))
# For 45K videos, 7 workers: 45000 // 28 = 1607
```

**This is too large.** Three problems:

1. **Kills load balancing.** Only ~28 chunks for 7 workers (~4 chunks each). If one chunk is disproportionately slow, that worker falls behind with no redistribution.
2. **Kills progress reporting.** `executor.map` returns results per-chunk. Your progress counter freezes for minutes at a time while each 1607-video chunk completes.
3. **Partially defeats the sort.** The first chunk contains videos 0-1606 (all the biggest files). Any imbalance *within* that chunk can't be redistributed.

**Fix: Use a fixed `chunksize=50`.** This gives 900 chunks — plenty granular for load balancing, 900x fewer IPC calls than default, and progress updates every ~50 videos.

---

## 8. Gemini's 5 Text Suggestions — Assessment

| # | Suggestion | Verdict | Reason |
|---|---|---|---|
| 1 | Fast-fail on frame count | **Yes** | Safe with `0 <` guard, already in v5.4 code |
| 2 | "Prefetching & Memory Pinning" | **Buzzwords** | Python's ProcessPoolExecutor doesn't prefetch. M4 unified memory doesn't need pinning. The useful part (sorting) is already covered by #6 above |
| 3 | "Replace cv2.cvtColor with Direct to RGB" | **Nonsense** | OpenCV VideoCapture has no "read as RGB" mode. Gemini credits `rgb.flags.writeable = False` as this optimization — that line is already in v5.2 and is a MediaPipe hint, not an RGB optimization |
| 4 | Check for VideoToolbox HW decoding | **Worth checking** | Not a code change — it's a build config. Run `python -c "import cv2; print(cv2.getBuildInformation())"` and look for `VideoToolbox: YES`. If present, the M4 Media Engine decodes H.264/H.265 at near-zero CPU cost. Could speed up the ~15% of time spent on video decode |
| 5 | NVMe burst writing | **Gemini rejects its own suggestion** | Agree. One-file-at-a-time with resume logic is safer for long runs |

---

## Round 2 Summary

| Change | Do it? | Quality impact | Speed impact |
|---|---|---|---|
| Fast-fail check | **Yes** | None | Skips dead-weight videos before MediaPipe init |
| Largest-first sorting | **Yes** | None | Prevents idle workers at end of run |
| Dynamic chunksize formula | **No — use fixed 50** | None either way | Formula produces 1607, which kills load balancing and progress reporting |
| "Prefetching & Memory Pinning" | **No** | N/A | Buzzwords with no actionable code change |
| "Direct to RGB" | **No** | N/A | Not a real thing in OpenCV |
| Check VideoToolbox | **Worth a 1-line check** | None | Potential ~10-15% overall speedup if not already enabled |
| NVMe burst writing | **No** | Riskier on crash | Gemini agrees |
