"""
Cloud extraction runner — multi-GPU support + lifted worker cap for high-core-count servers.

Auto-detects all available GPUs and assigns workers round-robin across them.
Default: 2 workers per GPU (latency hiding while one batch is on GPU, next prepares on CPU).

Usage:
    python3 src/extract_do.py                          # auto-detect GPUs + cores
    python3 src/extract_do.py --workers 16             # explicit worker count
    python3 src/extract_do.py --workers 16 --shard 0/3 # process shard 0 of 3 (for multi-instance)
"""

import argparse
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Required for CUDA in child processes
from extract import (
    process_single_video,
    set_worker_gpu,
    PipelineConfig,
    CFG,
    LABEL_ALIASES,
    _write_manifest,
    log,
)

# ── Worker initializer: pins each worker to exactly one GPU ──
_gpu_counter = None  # Shared counter for round-robin GPU assignment

def _worker_init(counter, num_gpus):
    """Each worker gets a unique GPU via atomic counter. Pins CUDA device
    BEFORE any model loading to prevent CUDA context on wrong GPUs."""
    with counter.get_lock():
        gpu_id = counter.value % num_gpus
        counter.value += 1
    set_worker_gpu(gpu_id)


def run_pipeline(num_workers=None, shard=None):
    import os
    import time
    import hashlib
    import json
    from pathlib import Path
    from collections import Counter, defaultdict
    import concurrent.futures

    out_path = Path(CFG.output_dir); out_path.mkdir(parents=True, exist_ok=True)

    done_base_files = set()
    for f in os.listdir(out_path):
        if f.endswith('.npy'):
            base_name = f.replace('.npy', '')
            for suffix in ['_fast', '_slow', '_mirror']:
                if suffix in base_name:
                    base_name = base_name.replace(suffix, '')
                    break
            done_base_files.add(base_name)

    all_videos = []
    for root, _, files in os.walk(CFG.raw_video_dir):
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
                    all_videos.append((root, f, label, CFG, file_size))

    total_skipped = len(done_base_files)
    if total_skipped > 0:
        log.info(f"Skipping {total_skipped} already-processed videos.")

    if len(all_videos) == 0:
        log.info("All videos are already processed!")
        _write_manifest(out_path)
        return

    all_videos.sort(key=lambda x: x[4], reverse=True)

    # Sharding: split work across multiple instances
    if shard is not None:
        shard_idx, shard_total = map(int, shard.split('/'))
        all_videos = [v for i, v in enumerate(all_videos) if i % shard_total == shard_idx]
        log.info(f"Shard {shard_idx}/{shard_total}: processing {len(all_videos)} videos")

    # ── Multi-GPU detection ──
    import torch
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_workers is not None:
        safe_workers = max(1, num_workers)
    elif gpu_count > 0 and CFG.prefer_rtmw:
        # 2 workers per GPU: ~6GB model each = ~12GB/24GB, leaves headroom for inference
        safe_workers = gpu_count * 2
        log.info(f"Detected {gpu_count} GPU(s) — using {safe_workers} workers (2 per GPU)")
    else:
        safe_workers = max(1, multiprocessing.cpu_count() - 2)

    # No per-task GPU assignment needed — workers are pinned to GPUs via initializer
    log.info(f"Found {len(all_videos)} remaining videos. Processing with {safe_workers} workers...")

    class_success = Counter()
    class_fail = Counter()
    pass_names = ['rtmw', 'video', 'static', 'tasks']
    class_l_pass = {p: Counter() for p in pass_names}
    class_r_pass = {p: Counter() for p in pass_names}
    class_face_detected = Counter()
    class_coverage_sum = defaultdict(float)

    saved_count = 0
    completed_videos = 0
    total_videos = len(all_videos)
    t_start = time.time()

    # Create shared counter for GPU assignment in worker initializer
    gpu_init_counter = multiprocessing.Value('i', 0)
    executor_kwargs = {}
    if gpu_count > 0 and CFG.prefer_rtmw:
        executor_kwargs = dict(initializer=_worker_init, initargs=(gpu_init_counter, gpu_count))

    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers, **executor_kwargs) as executor:
        futures = {executor.submit(process_single_video, v): v for v in all_videos}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=600)  # 10 min max per video (large files)
            except Exception as e:
                video_info = futures[future]
                log.warning(f"Worker crashed on {video_info[1]}: {e} — skipping")
                completed_videos += 1
                continue
            count, label, success, diag = result
            saved_count += count
            completed_videos += 1
            if success:
                class_success[label] += 1
                if diag:
                    l_winner = diag.get('l_pass', 'video')
                    r_winner = diag.get('r_pass', 'video')
                    if l_winner in class_l_pass:
                        class_l_pass[l_winner][label] += 1
                    if r_winner in class_r_pass:
                        class_r_pass[r_winner][label] += 1
                    if diag['face_detected']:
                        class_face_detected[label] += 1
                    class_coverage_sum[label] += diag['detection_coverage']
            else:
                class_fail[label] += 1
            # Progress with rate tracking
            if completed_videos % 50 == 0 or completed_videos == total_videos:
                elapsed_min = (time.time() - t_start) / 60
                rate = completed_videos / max(elapsed_min, 0.01)
                remaining = (total_videos - completed_videos) / max(rate, 0.01)
                log.info(f"Progress: {completed_videos}/{total_videos} ({rate:.0f}/min) | "
                         f"Saved: {saved_count} | ETA: {remaining:.1f} min")

    elapsed = time.time() - t_start
    log.info(f"\nDATASET COMPLETE. Successfully processed {saved_count} new videos in {elapsed/60:.1f} min ({elapsed/max(saved_count,1):.2f} s/video).")

    all_labels = sorted(set(list(class_success.keys()) + list(class_fail.keys())))
    high_fail_classes = []
    for label in all_labels:
        s = class_success[label]
        f = class_fail[label]
        total = s + f
        if total > 0:
            fail_rate = f / total
            if fail_rate > 0.30:
                high_fail_classes.append((label, s, f, fail_rate))

    if high_fail_classes:
        log.warning("Classes with >30% extraction failure rate:")
        for label, s, f, rate in sorted(high_fail_classes, key=lambda x: -x[3]):
            log.warning(f"  {label}: {s} ok / {f} fail ({rate*100:.0f}% fail)")

    total_success = sum(class_success.values())
    total_face_det = sum(class_face_detected.values())
    if total_success > 0:
        log.info(f"Cascade stats ({total_success} clips):")
        for p in pass_names:
            l_wins = sum(class_l_pass[p].values())
            r_wins = sum(class_r_pass[p].values())
            if l_wins > 0 or r_wins > 0:
                log.info(f"  {p:>8s}: L-hand won {l_wins}/{total_success} ({100*l_wins/total_success:.0f}%), "
                         f"R-hand won {r_wins}/{total_success} ({100*r_wins/total_success:.0f}%)")
        log.info(f"  Face detected: {total_face_det}/{total_success} ({100*total_face_det/total_success:.0f}%)")

    stats = {}
    for label in all_labels:
        s = class_success[label]
        f = class_fail[label]
        per_pass = {}
        for p in pass_names:
            per_pass[f'l_{p}_wins'] = class_l_pass[p][label]
            per_pass[f'r_{p}_wins'] = class_r_pass[p][label]
        stats[label] = {
            'success': s, 'fail': f,
            'fail_rate': round(f / max(s + f, 1), 3),
            **per_pass,
            'face_detected': class_face_detected[label],
            'avg_detection_coverage': round(class_coverage_sum[label] / max(s, 1), 3),
        }
    stats_path = out_path / 'extraction_stats.json'
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp, indent=2)
    log.info(f"Extraction stats saved: {stats_path}")

    _write_manifest(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLT Extraction (Cloud — Multi-GPU)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: 2 per GPU)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard index/total, e.g. '0/3' for first of 3 instances")
    args = parser.parse_args()
    run_pipeline(num_workers=args.workers, shard=args.shard)
