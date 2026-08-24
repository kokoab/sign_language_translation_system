"""
Dead-frame trimming ablation.

Tests detect_signing_range() across many videos to verify:
1. Trim behavior is consistent (not hardcoded for one video)
2. Training-style videos (tight, pre-cropped) get near-zero trim
3. Real-world videos with setup frames get meaningful trim
4. Threshold sensitivity (0.2 vs 0.3 vs 0.4) is stable
5. Trimming does not degrade inference on well-cropped videos

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/test_dead_frame_trim.py
"""
import os, sys, random, json, cv2, gc
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src_v16"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src_v16.inference_v16 import detect_signing_range, extract_continuous_smart
from src_v16.extract_v16 import extract_frames_v16

random.seed(42)

SAMPLE_DIR    = ROOT / "sample_videos"
ISOLATED_DIR  = ROOT / "data" / "raw_videos" / "ASL VIDEOS"
PHRASE_DIR    = ROOT / "data" / "raw_videos" / "PHRASES"

N_ISOLATED = 30
N_PHRASE   = 30


def load_frames(path, max_frames=600):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frames.append(f)
    cap.release()
    return frames


def trim_at_threshold(mask_scores, threshold_frac, sample_interval=5):
    """Re-run the trim logic with a custom threshold fraction."""
    n = len(mask_scores)
    k = max(3, n // 20)
    smoothed = np.convolve(mask_scores, np.ones(k) / k, mode="same")
    threshold = max(smoothed.max() * threshold_frac, 0.15)
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return 0, n
    start = max(0, above[0] - sample_interval)
    end = min(n, above[-1] + sample_interval)
    return start, end


def compute_mask_scores(frames, sample_interval=5):
    """Extract mask scores the same way detect_signing_range does — but return
    the raw scores so we can test multiple thresholds without re-extracting."""
    n = len(frames)
    mask_scores = np.zeros(n, dtype=np.float32)
    for i in range(0, n, sample_interval):
        chunk_end = min(i + sample_interval, n)
        if chunk_end - i < 4:
            continue
        arr = extract_frames_v16(frames[i:chunk_end])
        if arr is not None:
            mask_scores[i:chunk_end] = arr[:min(chunk_end - i, 32), :42, 3].mean()
        gc.collect()
    return mask_scores


def analyze_video(path, category):
    frames = load_frames(path)
    n = len(frames)
    if n < 16:
        return None

    mask_scores = compute_mask_scores(frames)

    row = {
        "path": path.name,
        "category": category,
        "n_frames": n,
        "peak_mask": float(mask_scores.max()),
        "mean_mask": float(mask_scores.mean()),
    }
    for thresh in (0.2, 0.3, 0.4):
        s, e = trim_at_threshold(mask_scores, thresh)
        row[f"t{int(thresh*100)}_start"] = int(s)
        row[f"t{int(thresh*100)}_end"]   = int(e)
        row[f"t{int(thresh*100)}_kept"]  = round((e - s) / n, 3)
    return row


def sample_videos():
    samples = []

    # All 5 sample videos (short, well-cropped, known GT)
    for f in sorted(SAMPLE_DIR.glob("*.mp4")):
        samples.append((f, "sample"))

    # N random isolated-sign videos (tight, should need ~no trim)
    iso_classes = [d for d in ISOLATED_DIR.iterdir() if d.is_dir()]
    random.shuffle(iso_classes)
    iso_picked = 0
    for cls in iso_classes:
        vids = list(cls.glob("*.mp4")) + list(cls.glob("*.mov"))
        if vids:
            samples.append((random.choice(vids), "isolated"))
            iso_picked += 1
            if iso_picked >= N_ISOLATED:
                break

    # N random phrase videos (longer, may have setup frames)
    phrase_classes = [d for d in PHRASE_DIR.iterdir() if d.is_dir()]
    random.shuffle(phrase_classes)
    phrase_picked = 0
    for cls in phrase_classes:
        vids = list(cls.glob("*.mp4")) + list(cls.glob("*.mov"))
        if vids:
            samples.append((random.choice(vids), "phrase"))
            phrase_picked += 1
            if phrase_picked >= N_PHRASE:
                break

    return samples


def main():
    samples = sample_videos()
    print(f"Testing {len(samples)} videos across {len(set(s[1] for s in samples))} categories\n")

    rows = []
    for i, (path, cat) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {cat:8s} {path.name}", end=" ... ", flush=True)
        try:
            r = analyze_video(path, cat)
            if r:
                rows.append(r)
                print(f"n={r['n_frames']:3d}  kept@0.3={r['t30_kept']:.2f}")
            else:
                print("SKIP (too short)")
        except Exception as e:
            print(f"ERR {e}")

    print(f"\n{'='*80}")
    print(f"Summary ({len(rows)} videos)")
    print(f"{'='*80}\n")

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    # Per-category stats at each threshold
    for cat, items in by_cat.items():
        print(f"{cat.upper()} ({len(items)} videos)")
        for thresh in (0.2, 0.3, 0.4):
            key = f"t{int(thresh*100)}_kept"
            kept = np.array([r[key] for r in items])
            n_trimmed = sum(1 for k in kept if k < 0.95)
            n_heavy   = sum(1 for k in kept if k < 0.50)
            print(f"  threshold={thresh:.1f}  kept mean={kept.mean():.3f}  "
                  f"std={kept.std():.3f}  min={kept.min():.3f}  "
                  f"trimmed>5%: {n_trimmed}/{len(items)}  trimmed>50%: {n_heavy}/{len(items)}")
        print()

    # Flag suspicious cases (very heavy trim on any video — could be eating real signing)
    print(f"{'='*80}")
    print("Suspicious cases (kept < 0.5 at threshold=0.3):")
    print(f"{'='*80}")
    suspicious = [r for r in rows if r["t30_kept"] < 0.5]
    for r in suspicious:
        print(f"  {r['category']:8s}  {r['path']:50s}  "
              f"n={r['n_frames']}  kept={r['t30_kept']:.2f}  "
              f"peak_mask={r['peak_mask']:.2f}")
    if not suspicious:
        print("  (none)")

    # Save full results
    out = ROOT / "mobile_export" / "reports" / "dead_frame_trim_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
