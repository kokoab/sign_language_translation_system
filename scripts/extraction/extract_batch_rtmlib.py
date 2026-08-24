"""
Batch extract all training videos with rtmlib ONNX RTMW.
Walks data/raw_videos/ASL VIDEOS/{CLASS}/{video}.mp4
Outputs ASL_landmarks_rtmlib/{CLASS}_{video}_{hash}.npy

Usage:
    python extract_batch_rtmlib.py --input "data/raw_videos/ASL VIDEOS" --output ASL_landmarks_rtmlib --device cuda
    python extract_batch_rtmlib.py --input "data/raw_videos/ASL VIDEOS" --output ASL_landmarks_rtmlib --device cuda --workers 4
"""
import os, sys, glob, json, hashlib, warnings, time, argparse
from multiprocessing import Pool, Manager, current_process
import multiprocessing
warnings.filterwarnings('ignore')
import numpy as np
import cv2

# Fake mediapipe to avoid import error from extract.py
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from extract import (
    PipelineConfig, NUM_NODES, LABEL_ALIASES,
    interpolate_hand, interpolate_face, temporal_resample,
    normalize_sequence, compute_kinematics_batch,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)

from rtmlib import Wholebody

# COCO-WholeBody 133 keypoint indices
_LHAND_START = 91
_RHAND_START = 112
_FACE_NOSE = 23 + 30
_FACE_CHIN = 23 + 8
_FACE_FOREHEAD = 23 + 27
_FACE_LEFT_EAR = 23 + 0
_FACE_RIGHT_EAR = 23 + 16
_FACE_INDICES = [_FACE_NOSE, _FACE_CHIN, _FACE_FOREHEAD, _FACE_LEFT_EAR, _FACE_RIGHT_EAR]

HAND_CONF = 0.25
FACE_CONF = 0.25


def extract_video(video_path, wholebody, cfg):
    """Extract one video → [32, 47, 10] float16 or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames < cfg.min_raw_frames:
        return None

    # Subsample if too long
    max_process = 128
    indices = list(range(total_frames))
    if total_frames > max_process:
        step = total_frames / max_process
        indices = [int(i * step) for i in range(max_process)]
        frames = [frames[i] for i in indices]

    l_seq, r_seq, face_seq = [], [], []
    l_valid, r_valid, face_valid = [], [], []

    for i, bgr in enumerate(frames):
        fi = indices[i]
        h, w = bgr.shape[:2]

        try:
            # Skip YOLOX detection — use full-frame bbox (single person ASL videos)
            full_bbox = np.array([[0, 0, w, h, 1.0]], dtype=np.float32)
            kps, scs = wholebody.pose_model(bgr, bboxes=full_bbox)
        except Exception:
            continue

        if kps is None or len(kps) == 0:
            continue

        if kps.ndim == 3:
            person_idx = scs.mean(axis=1).argmax()
            kps = kps[person_idx]
            scs = scs[person_idx]

        if len(kps) < 133:
            continue

        # Left hand
        l_scores = scs[_LHAND_START:_LHAND_START + 21]
        if l_scores.mean() >= HAND_CONF:
            coords = [[kps[_LHAND_START + j][0] / w, kps[_LHAND_START + j][1] / h, 0.0] for j in range(21)]
            if not l_valid or l_valid[-1] != fi:
                l_seq.append(coords)
                l_valid.append(fi)

        # Right hand
        r_scores = scs[_RHAND_START:_RHAND_START + 21]
        if r_scores.mean() >= HAND_CONF:
            coords = [[kps[_RHAND_START + j][0] / w, kps[_RHAND_START + j][1] / h, 0.0] for j in range(21)]
            if not r_valid or r_valid[-1] != fi:
                r_seq.append(coords)
                r_valid.append(fi)

        # Face
        face_coords = []
        face_ok = True
        for fidx in _FACE_INDICES:
            if fidx < len(kps) and scs[fidx] >= FACE_CONF:
                face_coords.append([kps[fidx][0] / w, kps[fidx][1] / h, 0.0])
            else:
                face_ok = False
                break
        if face_ok and len(face_coords) == 5:
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_coords)
                face_valid.append(fi)

    if not l_valid and not r_valid:
        return None

    # Temporal outlier rejection
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)

    # Interpolation
    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
    face_full = interpolate_face(np.array(face_seq) if face_seq else np.zeros((0, 5, 3)), face_valid, total_frames)

    combined = np.concatenate([l_full, r_full, face_full], axis=1)

    # Process: resample → filter → stabilize → normalize → kinematics
    resampled = temporal_resample(combined, cfg.target_frames)
    smoothed = one_euro_filter(resampled[:, :, :3])
    resampled[:, :, :3] = smoothed
    if l_ever:
        resampled = stabilize_bones(resampled, 0, 21)
    if r_ever:
        resampled = stabilize_bones(resampled, 21, 42)
    normalized = normalize_sequence(resampled, l_ever, r_ever)

    T = cfg.target_frames
    mask = np.zeros((1, T, NUM_NODES, 1), dtype=np.float32)
    if l_ever: mask[0, :, 0:21, 0] = 1.0
    if r_ever: mask[0, :, 21:42, 0] = 1.0
    if face_ever: mask[0, :, 42:47, 0] = 1.0

    data = compute_kinematics_batch(
        normalized[np.newaxis, ...], l_ever, r_ever, face_ever,
        per_frame_mask=mask
    ).squeeze(0).astype(np.float16)

    if data.shape != (32, 47, 10):
        return None

    return data


def _worker_init(device):
    """Each worker initializes its own Wholebody instance."""
    global _worker_wholebody, _worker_cfg
    pid = current_process().name
    print("  [{}] Initializing RTMW-l (rtmlib, {})...".format(pid, device))
    _worker_wholebody = Wholebody(to_openpose=False, mode='performance', backend='onnxruntime', device=device)
    _worker_cfg = PipelineConfig()
    print("  [{}] Ready.".format(pid))


def _worker_extract(task):
    """Process a single video. Returns (out_name, label, data_or_none, error_or_none)."""
    video_path, out_name, canonical_label = task
    global _worker_wholebody, _worker_cfg
    try:
        data = extract_video(video_path, _worker_wholebody, _worker_cfg)
        return (out_name, canonical_label, data, None)
    except Exception as e:
        return (out_name, canonical_label, None, str(e)[:80])


def main():
    parser = argparse.ArgumentParser(description="Batch extract training videos with rtmlib RTMW")
    parser.add_argument("--input", default="data/raw_videos/ASL VIDEOS", help="Root video directory")
    parser.add_argument("--output", default="ASL_landmarks_rtmlib", help="Output .npy directory")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--resume", action="store_true", help="Skip already-extracted files")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (each gets own RTMW instance)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    manifest = {}
    stats = {"total": 0, "success": 0, "fail": 0, "skipped": 0}

    # Load existing manifest if resuming
    manifest_path = os.path.join(args.output, "manifest.json")
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print("Resuming: {} files already extracted".format(len(manifest)))

    # Build task list
    input_dir = args.input
    class_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    print("Found {} class folders".format(len(class_dirs)))

    # Collect all (video_path, out_name, label) tasks, skip already done
    all_tasks = []
    class_labels = {}  # out_name → label for skipped files
    for class_name in class_dirs:
        canonical_label = LABEL_ALIASES.get(class_name, class_name)
        class_path = os.path.join(input_dir, class_name)
        videos = sorted(glob.glob(os.path.join(class_path, "*.mp4")) +
                       glob.glob(os.path.join(class_path, "*.mov")) +
                       glob.glob(os.path.join(class_path, "*.MP4")))

        for video_path in videos:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            hash_str = hashlib.md5("{}_{}".format(canonical_label, stem).encode()).hexdigest()[:6]
            out_name = "{}_{}_{}.npy".format(canonical_label, stem, hash_str)
            out_path = os.path.join(args.output, out_name)

            if out_name in manifest or (args.resume and os.path.exists(out_path)):
                if out_name not in manifest:
                    manifest[out_name] = canonical_label
                stats["skipped"] += 1
                stats["success"] += 1
                stats["total"] += 1
            else:
                all_tasks.append((video_path, out_name, canonical_label))
                stats["total"] += 1

    pending = len(all_tasks)
    print("Total: {} videos, {} to extract, {} skipped".format(stats["total"], pending, stats["skipped"]))

    if pending == 0:
        print("Nothing to do.")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return

    t0 = time.time()
    done = 0

    if args.workers <= 1:
        # Single-process mode (original behavior)
        print("Initializing RTMW-l (rtmlib, {})...".format(args.device))
        wholebody = Wholebody(to_openpose=False, mode='performance', backend='onnxruntime', device=args.device)
        cfg = PipelineConfig()
        print("Ready. Extracting {} videos...".format(pending))

        for i, (video_path, out_name, canonical_label) in enumerate(all_tasks):
            out_path = os.path.join(args.output, out_name)
            try:
                data = extract_video(video_path, wholebody, cfg)
                if data is not None:
                    np.save(out_path, data)
                    manifest[out_name] = canonical_label
                    stats["success"] += 1
                else:
                    stats["fail"] += 1
            except Exception as e:
                stats["fail"] += 1
                if stats["fail"] <= 20:
                    print("  ERROR {}: {}".format(os.path.basename(video_path), str(e)[:80]))

            done += 1
            if done % 100 == 0 or done == pending:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                eta = (pending - done) / max(rate, 0.01)
                print("[{:>6}/{:>6}] ok:{:>6} fail:{:>4} | {:.1f} vid/s | ETA {:.0f}m".format(
                    done, pending, stats["success"], stats["fail"], rate, eta / 60))

            # Save manifest every 500 videos
            if done % 500 == 0:
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
    else:
        # Multi-process mode
        print("Launching {} workers...".format(args.workers))
        # Use 'spawn' to avoid CUDA fork issues
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.device,)) as pool:
            # imap_unordered for best throughput — results come back as they finish
            for out_name, canonical_label, data, error in pool.imap_unordered(_worker_extract, all_tasks, chunksize=4):
                out_path = os.path.join(args.output, out_name)
                if data is not None:
                    np.save(out_path, data)
                    manifest[out_name] = canonical_label
                    stats["success"] += 1
                elif error:
                    stats["fail"] += 1
                    if stats["fail"] <= 20:
                        print("  ERROR {}: {}".format(out_name, error))
                else:
                    stats["fail"] += 1

                done += 1
                if done % 100 == 0 or done == pending:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1)
                    eta = (pending - done) / max(rate, 0.01)
                    print("[{:>6}/{:>6}] ok:{:>6} fail:{:>4} | {:.1f} vid/s | ETA {:.0f}m".format(
                        done, pending, stats["success"], stats["fail"], rate, eta / 60))

                # Save manifest every 500 videos
                if done % 500 == 0:
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)

    # Final save
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("Total videos:  {}".format(stats["total"]))
    print("Extracted:     {}".format(stats["success"]))
    print("Failed:        {}".format(stats["fail"]))
    print("Skipped:       {}".format(stats["skipped"]))
    print("Manifest:      {} ({} entries)".format(manifest_path, len(manifest)))
    print("Time:          {:.0f} min".format(elapsed / 60))
    print("Rate:          {:.1f} videos/sec".format(stats["total"] / max(elapsed, 1)))


if __name__ == "__main__":
    main()
