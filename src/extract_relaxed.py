"""
Relaxed extraction pass — runs ONLY on videos that failed the standard extraction.
Saves output with same naming convention so manifest picks them up normally.

Usage:
    python3 src/extract_relaxed.py
    python3 src/extract_relaxed.py --workers 46
"""

import argparse
import multiprocessing
import os
import hashlib
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

from extract import (
    process_single_video,
    PipelineConfig,
    LABEL_ALIASES,
    _write_manifest,
    log,
)


# Relaxed config: lower thresholds to rescue borderline videos
RELAXED_CFG = PipelineConfig(
    min_raw_frames=6,           # was 8
    max_missing_ratio=0.55,     # was 0.40 (now allows hands in only 45% of frames)
    min_detection_conf=0.70,    # was 0.80 (accept lower-confidence detections)
    min_tracking_conf=0.70,     # was 0.80
)


def run_pipeline(num_workers=None):
    out_path = Path(RELAXED_CFG.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find already-extracted files (from standard + any previous relaxed runs)
    done_base_files = set()
    for f in os.listdir(out_path):
        if f.endswith('.npy'):
            base_name = f.replace('.npy', '')
            for suffix in ['_fast', '_slow', '_mirror']:
                if suffix in base_name:
                    base_name = base_name.replace(suffix, '')
                    break
            done_base_files.add(base_name)

    # Build list of videos that DON'T have a successful extraction yet
    failed_videos = []
    for root, _, files in os.walk(RELAXED_CFG.raw_video_dir):
        label = Path(root).name
        label = LABEL_ALIASES.get(label, label)
        for f in files:
            if f.lower().endswith(('.mp4', '.mov')):
                file_hash = hashlib.md5(f.encode()).hexdigest()[:6]
                stem = Path(f).stem
                expected_save_name = f"{label}_{stem}_{file_hash}"
                if expected_save_name not in done_base_files:
                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)
                    failed_videos.append((root, f, label, RELAXED_CFG, file_size))

    already_done = len(done_base_files)
    log.info(f"Standard extraction already has {already_done} files.")
    log.info(f"Found {len(failed_videos)} previously-failed videos to retry with relaxed settings.")

    if len(failed_videos) == 0:
        log.info("No failed videos to retry!")
        _write_manifest(out_path)
        return

    failed_videos.sort(key=lambda x: x[4], reverse=True)

    if num_workers is not None:
        safe_workers = max(1, num_workers)
    else:
        safe_workers = max(1, multiprocessing.cpu_count() - 2)
    chunk_size = 50

    log.info(f"Retrying {len(failed_videos)} videos with relaxed settings "
             f"(min_raw_frames={RELAXED_CFG.min_raw_frames}, "
             f"max_missing={RELAXED_CFG.max_missing_ratio}, "
             f"min_conf={RELAXED_CFG.min_detection_conf})")
    log.info(f"Using {safe_workers} workers (Chunksize: {chunk_size})...")

    class_success = Counter()
    class_fail = Counter()
    class_l_video = Counter()
    class_l_static = Counter()
    class_r_video = Counter()
    class_r_static = Counter()
    class_face_video = Counter()
    class_face_static = Counter()
    class_face_detected = Counter()
    class_coverage_sum = defaultdict(float)

    saved_count = 0
    t_start = time.time()

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        for result in executor.map(process_single_video, failed_videos, chunksize=chunk_size):
            count, label, success, diag = result
            saved_count += count
            if success:
                class_success[label] += 1
                if diag:
                    if diag['l_pass'] == 'video': class_l_video[label] += 1
                    else: class_l_static[label] += 1
                    if diag['r_pass'] == 'video': class_r_video[label] += 1
                    else: class_r_static[label] += 1
                    if diag['face_pass'] == 'video': class_face_video[label] += 1
                    else: class_face_static[label] += 1
                    if diag['face_detected']: class_face_detected[label] += 1
                    class_coverage_sum[label] += diag['detection_coverage']
            else:
                class_fail[label] += 1
            if saved_count > 0 and saved_count % 100 == 0:
                print(f"  -> Rescued {saved_count} sequences...", end="\r")

    elapsed = time.time() - t_start
    log.info(f"\nRELAXED PASS COMPLETE. Rescued {saved_count} additional videos "
             f"in {elapsed/60:.1f} min ({elapsed/max(saved_count,1):.2f} s/video).")

    all_labels = sorted(set(list(class_success.keys()) + list(class_fail.keys())))

    # Show rescue rates for previously-failed classes
    log.info("Rescue rates by class:")
    for label in sorted(all_labels, key=lambda l: class_success[l], reverse=True):
        s = class_success[label]
        f = class_fail[label]
        if s > 0:
            log.info(f"  {label}: rescued {s}/{s+f} ({100*s/(s+f):.0f}%)")

    still_failing = [(l, class_fail[l]) for l in all_labels if class_fail[l] > 0 and class_success[l] == 0]
    if still_failing:
        log.warning("Classes with 0 rescues (MediaPipe fundamentally can't detect these signs):")
        for label, f in sorted(still_failing, key=lambda x: -x[1]):
            log.warning(f"  {label}: {f} still failed")

    stats = {}
    for label in all_labels:
        s = class_success[label]
        f = class_fail[label]
        stats[label] = {
            'rescued': s, 'still_failed': f,
            'rescue_rate': round(s / max(s + f, 1), 3),
            'avg_detection_coverage': round(class_coverage_sum[label] / max(s, 1), 3),
        }
    stats_path = out_path / 'relaxed_extraction_stats.json'
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp, indent=2)
    log.info(f"Relaxed stats saved: {stats_path}")

    _write_manifest(out_path)
    log.info(f"Total dataset size: {already_done + saved_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLT Relaxed Extraction (retry failed videos)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 2)")
    args = parser.parse_args()
    run_pipeline(num_workers=args.workers)
