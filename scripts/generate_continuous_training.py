"""
Generate synthetic continuous training data for Stage 2.
Concatenates raw video frames from multiple signs, then extracts
using extract_frames_continuous — matching inference exactly.

Output: ASL_continuous_synthetic/ with [N*32, 61, 10] .npy files + manifest.json

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/generate_continuous_training.py --count 15000 --workers 16
"""
import os, sys, glob, json, random, time, argparse, hashlib, subprocess
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract import LABEL_ALIASES


def build_video_index(video_dir):
    """Build index: class_name → list of video paths."""
    index = {}
    for cls_dir in sorted(os.listdir(video_dir)):
        cls_path = os.path.join(video_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        canonical = LABEL_ALIASES.get(cls_dir, cls_dir)
        vids = sorted(glob.glob(os.path.join(cls_path, '*.mp4')))
        if vids:
            if canonical not in index:
                index[canonical] = []
            index[canonical].extend(vids)
    return index


def generate_one_sequence(index, min_signs=1, max_signs=8):
    """Pick random signs, load their frames, concatenate.
    Returns (frames_list, gloss_string) or (None, None)."""
    classes = list(index.keys())
    n_signs = random.randint(min_signs, max_signs)

    # Weight distribution: 1% single, 10% long (7-8), rest standard
    r = random.random()
    if r < 0.01:
        n_signs = 1
    elif r < 0.11:
        n_signs = random.randint(7, max_signs)
    else:
        n_signs = random.randint(2, 6)

    selected = random.choices(classes, k=n_signs)
    all_frames = []
    glosses = []

    for cls in selected:
        vid = random.choice(index[cls])
        cap = cv2.VideoCapture(vid)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        if len(frames) < 8:
            continue
        all_frames.extend(frames)
        glosses.append(cls)

    if not glosses or not all_frames:
        return None, None

    return all_frames, ' '.join(glosses)


def worker_generate(task_json_path, output_dir, video_dir):
    """Worker: generate sequences from task file."""
    from extract_apple_vision import extract_frames_continuous

    with open(task_json_path) as f:
        tasks = json.load(f)

    index = build_video_index(video_dir)
    results = {}
    generated = 0

    for task in tasks:
        seed = task['seed']
        random.seed(seed)

        frames, gloss_str = generate_one_sequence(index)
        if frames is None:
            continue

        data = extract_frames_continuous(frames)
        if data is None:
            continue

        hash_str = hashlib.md5(f"{gloss_str}_{seed}".encode()).hexdigest()[:8]
        out_name = f"syn_{seed:06d}_{hash_str}.npy"
        np.save(os.path.join(output_dir, out_name), data)
        results[out_name] = gloss_str
        generated += 1

    # Save results
    result_path = task_json_path.replace('.json', '_result.json')
    with open(result_path, 'w') as f:
        json.dump(results, f)

    return generated


if __name__ == "__main__":
    # Handle --_worker flag FIRST (before argparse)
    if '--_worker' in sys.argv:
        idx = sys.argv.index('--_worker')
        task_path = sys.argv[idx + 1]
        output_dir = sys.argv[idx + 2]
        video_dir = sys.argv[idx + 3]
        n = worker_generate(task_path, output_dir, video_dir)
        print(f"Worker done: {n} sequences")
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data/raw_videos/ASL VIDEOS")
    parser.add_argument("--output", default="ASL_continuous_synthetic")
    parser.add_argument("--count", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"=== Generating {args.count} synthetic continuous sequences ===")
    print(f"Video dir: {args.video_dir}")
    print(f"Output:    {args.output}")
    print(f"Workers:   {args.workers}")

    # Split tasks across workers
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='slt_gen_')
    tasks_per_worker = args.count // args.workers
    procs = []

    t0 = time.time()
    for wi in range(args.workers):
        start = wi * tasks_per_worker
        end = start + tasks_per_worker if wi < args.workers - 1 else args.count
        tasks = [{'seed': i} for i in range(start, end)]

        task_path = os.path.join(tmpdir, f'tasks_{wi}.json')
        with open(task_path, 'w') as f:
            json.dump(tasks, f)

        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        p = subprocess.Popen(
            [sys.executable, __file__, '--_worker', task_path, args.output, args.video_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        procs.append((p, task_path, len(tasks)))
        print(f"  Worker {wi}: {len(tasks)} sequences (PID {p.pid})")

    # Wait for all
    manifest = {}
    total_done = 0
    for p, task_path, n_tasks in procs:
        p.wait()
        result_path = task_path.replace('.json', '_result.json')
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            manifest.update(results)
        total_done += n_tasks
        elapsed = time.time() - t0
        print(f"  Progress: {total_done}/{args.count} | {len(manifest)} generated | {elapsed/60:.1f}m")

    # Save manifest
    with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"Generated: {len(manifest)} sequences")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output: {args.output}/manifest.json")
