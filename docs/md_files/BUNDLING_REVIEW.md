# SLT Bundling Review (v4.5 → v4.6 → v4.8)

## Answers to Your Questions

### Q1: Is the 4D bundling approach sound?

**Yes.** `(16, 32, 42, 10)` is purely a storage optimization. The data is byte-identical to 16 individual `(32, 42, 10)` files. The `idx // 16` / `idx % 16` indexing in `__getitem__` is mathematically correct. No information is lost or distorted.

### Q2: Will this compromise data quality, shuffling, or accuracy?

**The bundling itself does not.** PyTorch's `DataLoader(shuffle=True)` generates a random permutation over `[0, N*16)`. Index 499,203 (video 31,200 variant 3) can appear right after index 12,048 (video 753 variant 0). Shuffling is fully effective — it operates on the virtual index space, not the file layout.

**However, a new bug in v4.5 does compromise augmentation diversity.** See B1 below. **Fixed in v4.6.**

### Q3: Hidden drawbacks with this DataLoader strategy?

**Read amplification.** Each `np.load()` call reads the full 860KB bundle to extract one 54KB variant (16x amplification). With DataLoader workers and OS page cache this is usually fine, but you can eliminate it:

```python
bundle = np.load(self.file_list[file_idx], mmap_mode='r')
return torch.from_numpy(bundle[variant_idx].copy())
```

`mmap_mode='r'` memory-maps the file. The OS only pages in the requested 54KB slice. The `.copy()` is required because PyTorch tensors can't wrap read-only mmap buffers.

**No RAM concern.** Even without mmap, 860KB per load is negligible. With 4 DataLoader workers, peak overhead is ~3.4MB.

---

## New Bugs Introduced in v4.5

### B1. CRITICAL: All Workers Share Identical RNG State

```python
rng = np.random.default_rng(CFG.seed)  # module-level global
```

macOS uses `spawn` as the default multiprocessing start method (Python 3.8+). Each spawned worker **re-imports the module**, re-executing `rng = np.random.default_rng(42)`. Every worker starts with the **exact same RNG state**.

Result: if worker 0 processes video A as its 5th task and worker 3 processes video Z as its 5th task, **both videos get identical augmentation parameters** (same rotation angles, same jitter noise, same warp curves). With `cpu_count()` workers, your effective augmentation diversity drops roughly by a factor of `num_workers`.

**Fix — derive a per-video RNG inside the worker:**

```python
def process_single_video(task_info):
    root, video_name, label, cfg = task_info
    video_seed = int(hashlib.md5(video_name.encode()).hexdigest(), 16) % (2**32)
    local_rng = np.random.default_rng(cfg.seed + video_seed)
    # pass local_rng to generate_augmentations and temporal_shift_pad
```

This is deterministic (same video always gets the same augmentations) and unique per video regardless of worker assignment.

**Status: FIXED in v4.6** — Per-video `local_rng` derived from `hashlib.md5(video_name)`, passed to all augmentation functions.

### B2. MODERATE: Tracker State Leak Reintroduced

v4.3 fixed the MediaPipe tracker leak by recreating the detector per video (`with` context manager). v4.5 reverts this with a lazily-initialized persistent detector per worker:

```python
if not hasattr(process_single_video, "detector"):
    process_single_video.detector = mp.solutions.hands.Hands(...)
```

This means tracking state from video N leaks into video N+1 within the same worker. The first 1-2 frames of each video may have phantom detections carried from the previous video's last frame. Over 45,000 videos, this is a measurable source of noise.

**Fix — recreate per video, or accept the tradeoff and note it.** Per-video recreation adds ~50ms overhead per video. Over 45K videos, that's ~37 minutes of added time. If this is acceptable:

```python
with mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=cfg.max_num_hands,
    min_detection_confidence=cfg.min_detection_conf, model_complexity=cfg.model_complexity
) as detector:
    # use detector for this video
```

If not, the alternative is `static_image_mode=True` which is stateless but slower per frame.

**Status: FIXED in v4.6** — Detector recreated per video via `with` context manager. Also restored `min_tracking_confidence` parameter that was missing in v4.5.

### B3. MODERATE: No Error Handling in Worker

If any video causes an exception (corrupt file, MediaPipe segfault, disk full), `executor.map` re-raises it in the main process and **terminates the entire pipeline**. 44,999 successfully processed videos are lost if video 45,000 crashes.

**Fix — wrap the worker in try/except:**

```python
def process_single_video(task_info):
    try:
        # ... existing logic ...
    except Exception as e:
        log.error(f"Failed {task_info[2]}/{task_info[1]}: {e}")
        return 0
```

**Status: FIXED in v4.6** — Worker wrapped in try/except, logs the error, returns 0.

### B4. MINOR: Hardcoded Bundle Size

```python
bundle = np.zeros((16, cfg.target_frames, 42, 10), dtype=np.float32)
```

If you change augmentation counts (e.g., add 3 more rotations), this `16` silently produces zero-filled padding or an index overflow. Derive it:

```python
aug_variants = generate_augmentations(normalized, cfg, l_ever, r_ever)
bundle = np.zeros((1 + len(aug_variants), cfg.target_frames, 42, 10), dtype=np.float32)
```

The PyTorch Dataset's `__len__` must then also read the bundle shape dynamically rather than assuming 16.

**Status: FIXED in v4.6** — Bundle sized from `1 + len(aug_variants)`.

### B5. MINOR: `cpu_count()` May Be Aggressive

```python
max_workers=multiprocessing.cpu_count()
```

On an M4 Air (10 cores), this spawns 10 workers each holding a MediaPipe model (~200-300MB) + OpenCV VideoCapture + numpy buffers. Peak RAM could hit 4-5GB just for workers. Combined with the OS and other processes, this may trigger swap on a 16GB machine. Consider:

```python
max_workers=min(multiprocessing.cpu_count() - 1, 6)
```

**Status: FIXED in v4.6** — Uses `min(cpu_count() - 2, 6)`.

---

## Review Round 2 (v4.6)

All five v4.5 bugs have been addressed. One minor edge case noted:

### B6. TRIVIAL: `safe_workers` Can Be Zero on Low-Core Machines

```python
safe_workers = min(multiprocessing.cpu_count() - 2, 6)
```

On a 2-core machine, `cpu_count() - 2 = 0`. `ProcessPoolExecutor(max_workers=0)` raises `ValueError`. Not a concern on your M4 Air (10 cores → 6 workers), but fragile if someone runs this on a CI server or VM:

```python
safe_workers = max(1, min(multiprocessing.cpu_count() - 2, 6))
```

### Additional note from v4.5 → v4.6 diff

v4.5 was **missing `min_tracking_confidence`** in the MediaPipe Hands constructor. v4.6 correctly passes it. This was a silent regression — MediaPipe would have used its default (0.5) instead of your configured 0.65, potentially accepting lower-quality tracking between frames.

---

## Final Summary

| Item | Severity | Status |
|---|---|---|
| Bundling math correctness | — | **Sound** |
| Shuffle effectiveness | — | **Unaffected** |
| RAM / training speed | — | **Fine** (use `mmap_mode='r'` for bonus) |
| Identical RNG across workers (B1) | **Critical** | **FIXED** |
| Tracker state leak (B2) | Moderate | **FIXED** |
| No worker error handling (B3) | Moderate | **FIXED** |
| Hardcoded bundle size (B4) | Minor | **FIXED** |
| Aggressive worker count (B5) | Minor | **FIXED** |
| `safe_workers` zero on 2-core (B6) | Trivial | **FIXED** (v4.8 uses `max(1, ...)`) |

**Pipeline status (v4.6): Production-ready for bundled extraction on M4 Air.**

---
---

## Review Round 3 (v4.8)

v4.8 adds three features: batched kinematics, resume-on-restart, and worker count tuning for 24GB RAM. B1–B5 fixes from v4.6 are preserved. B6 is fixed. One new bug found.

### B7. CRITICAL: Resume Check Matches on Hash Alone — Skips Unprocessed Videos

The resume logic extracts only the 6-char hash suffix from existing output filenames:

```python
file_hash = f.replace('.npy', '').split('_')[-1]
done_files.add(file_hash)
```

Then skips any video whose filename hashes to a value already in `done_files`:

```python
file_hash = hashlib.md5(f.encode()).hexdigest()[:6]
if file_hash not in done_files:
    all_videos.append(...)
```

The hash is derived from `video_name` alone (e.g., `001.mp4`), **not** from `label + video_name`. In a typical ASL dataset:

```
ASL VIDEOS/
  HELLO/001.mp4
  GOODBYE/001.mp4
  THANK_YOU/001.mp4
  ...325 classes, each with 001.mp4
```

All `001.mp4` files hash to the same value. On a full run this doesn't matter — all videos are queued before any are processed. But on **resume after interruption**:

1. Run starts, processes 5,000 videos including `HELLO/001.mp4` → saves `HELLO_001_abc123.npy`
2. Run is interrupted
3. Resume: `done_files` contains hash `abc123`
4. `GOODBYE/001.mp4` also hashes to `abc123` → **falsely skipped**
5. All remaining classes' `001.mp4` are also skipped

With 325 classes sharing the same filenames, a single interruption could silently skip up to 324 videos per shared filename. This compounds: if the dataset has 100 shared filenames across 325 classes, up to 32,400 videos could be lost on resume.

**Fix — match on the full output filename, not just the hash:**

```python
# Build set of existing output filenames (without extension)
done_files = set()
for f in os.listdir(out_path):
    if f.endswith('.npy'):
        done_files.add(f.replace('.npy', ''))

# When queuing videos, compute the exact save name
for f in files:
    if f.lower().endswith(('.mp4', '.mov')):
        file_hash = hashlib.md5(f.encode()).hexdigest()[:6]
        save_name = f"{label}_{Path(f).stem}_{file_hash}"
        if save_name not in done_files:
            all_videos.append((root, f, label, CFG))
```

This is zero-cost (set lookup is O(1)) and eliminates false skips entirely.

### Batched Kinematics — Verified Correct

`compute_kinematics_batch` is a clean vectorization of the original per-variant loop. The batch dimension is correctly handled on axis 0, temporal slicing operates on axis 1, and the output shape `(B, F, P, 10)` matches the bundle format. Combined with `np.stack`, the integration is clean:

```python
all_variants = np.stack([normalized] + aug_variants, axis=0)  # (16, 32, 42, 3)
bundle = compute_kinematics_batch(all_variants, l_ever, r_ever)  # (16, 32, 42, 10)
```

No issues.

### Worker Count — Acceptable with Caveat

```python
safe_workers = max(1, min(multiprocessing.cpu_count() - 1, 9))
```

On M4 Air (10 cores): 9 workers. This fixes B6 (`max(1, ...)`). With 24GB RAM, memory is fine. However, 9 out of 10 cores fully utilized on a **fanless** machine will cause sustained thermal throttling. The original problem statement cites thermal throttling as a factor. Monitor temps — if throttling is severe, drop to 6-7 workers. This is a tuning decision, not a code bug.

---

## Review Round 4 (v4.8 revised)

B7 fix verified. The resume logic now stores the full `{label}_{stem}_{hash}` as the key:

```python
done_files.add(f.replace('.npy', ''))           # e.g. "HELLO_001_abc123"
expected_save_name = f"{label}_{stem}_{file_hash}"  # reconstructed per video
```

`HELLO/001.mp4` → key `HELLO_001_abc123`, `GOODBYE/001.mp4` → key `GOODBYE_001_abc123`. Distinct keys, no false skips across classes. The key reconstruction in `run_pipeline` matches the save name format in `process_single_video`. Correct.

No new issues found.

---

## Final Summary

| Item | Severity | Status |
|---|---|---|
| Bundling math correctness | — | **Sound** |
| Shuffle effectiveness | — | **Unaffected** |
| RAM / training speed | — | **Fine** (use `mmap_mode='r'` for bonus) |
| Identical RNG across workers (B1) | **Critical** | **FIXED** |
| Tracker state leak (B2) | Moderate | **FIXED** |
| No worker error handling (B3) | Moderate | **FIXED** |
| Hardcoded bundle size (B4) | Minor | **FIXED** |
| Aggressive worker count (B5) | Minor | **FIXED** |
| `safe_workers` zero on 2-core (B6) | Trivial | **FIXED** |
| Resume skips on hash collision (B7) | **Critical** | **FIXED** |
| Batched kinematics | — | **Correct** |

**Pipeline status: Production-ready.**
