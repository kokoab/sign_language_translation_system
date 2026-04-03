"""
CoreML extraction for SLT — Mac M4 Neural Engine.
Uses CoreMLExecutionProvider with batch_size=1 (required for CoreML stability).
Produces identical results across runs (deterministic).

Output: ASL_landmarks_coreml/ with [32, 61, 10] .npy files + manifest.json
Also saves hand crops for future hybrid CNN training.

Usage:
    python scripts/extract_coreml.py --workers 6
"""
import os, sys, glob, hashlib, warnings, time, argparse, json
import multiprocessing as mp
warnings.filterwarnings('ignore')
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_batch_fast_v2 import (
    preprocess_frame, decode_simcc_batch, postprocess_keypoints,
    interpolate_generic, normalize_sequence_v2, compute_kinematics_v2,
    _FACE_INDICES, _BODY_INDICES, _LHAND_START, _RHAND_START,
    NUM_NODES_V2, NUM_FACE_V2, NUM_BODY_V2,
    HAND_CONF, FACE_CONF, BODY_CONF, PipelineConfig,
)
from extract import (
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)

LABEL_ALIASES = {
    "DRIVE": "DRIVE_CAR", "CAR": "DRIVE_CAR",
    "HARD": "HARD_DIFFICULT", "DIFFICULT": "HARD_DIFFICULT",
    "MAKE": "MAKE_CREATE", "CREATE": "MAKE_CREATE",
    "EAT": "EAT_FOOD", "FOOD": "EAT_FOOD",
    "ALSO_SAME": "ALSO", "SAME": "ALSO",
    "MARKET_STORE": "STORE", "MARKET": "STORE",
    "US_WE": "WE", "US": "WE",
    "FEW_SEVERAL": "FEW", "SEVERAL": "FEW",
    "I_ME": "I", "ME": "I",
    "HE_SHE": "HE", "SHE": "HE",
    "HIS": "HIS_HER", "HER": "HIS_HER",
    "EXCUSE": "EXCUSE ME",
    "RIGHT": "RIGHT (ADJECTIVE)",
    "TODAY": "NOW",
}


def batch_inference_single(session, frames, centers, scales):
    """Process one frame at a time — required for CoreML stability."""
    all_kps, all_scs = [], []
    for i in range(len(frames)):
        out = session.run(None, {'input': frames[i:i+1]})
        locs, scores = decode_simcc_batch(out[0], out[1])
        kps, scs = postprocess_keypoints(locs, scores, centers[i:i+1], scales[i:i+1])
        all_kps.append(kps[0])
        all_scs.append(scs[0])
    return np.stack(all_kps), np.stack(all_scs)


def extract_one(args):
    video_path, npy_dir, crop_dir, rtmw_path, det_path, use_coreml = args
    try:
        import onnxruntime as ort
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if use_coreml else ['CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # 4 threads per worker (M4: 2 workers × 4 = 8 cores)
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(rtmw_path, sess_options=opts, providers=providers)

        from rtmlib import YOLOX
        det = YOLOX(det_path, model_input_size=(640, 640))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        total_frames = len(frames)
        if total_frames < 8:
            return None

        max_process = 128
        indices = list(range(total_frames))
        if total_frames > max_process:
            step = total_frames / max_process
            indices = [int(i * step) for i in range(max_process)]
            frames = [frames[i] for i in indices]

        h0, w0 = frames[0].shape[:2]
        bboxes = det(frames[0])
        if bboxes is not None and len(bboxes) > 0:
            bbox = bboxes[0][:4].astype(np.float32)
        else:
            bbox = np.array([0, 0, w0, h0], dtype=np.float32)

        # Save hand crop (middle frame, centered on person bbox)
        if crop_dir:
            mid = len(frames) // 2
            x1, y1, x2, y2 = bbox.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_size = max(x2 - x1, y2 - y1) // 2
            cy1 = max(0, cy - crop_size)
            cy2 = min(h0, cy + crop_size)
            cx1 = max(0, cx - crop_size)
            cx2 = min(w0, cx + crop_size)
            crop = frames[mid][cy1:cy2, cx1:cx2]
            if crop.size > 0:
                # Pad to square (preserve aspect ratio) then resize
                ch, cw = crop.shape[:2]
                max_side = max(ch, cw)
                padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                y_off = (max_side - ch) // 2
                x_off = (max_side - cw) // 2
                padded[y_off:y_off+ch, x_off:x_off+cw] = crop
                crop = cv2.resize(padded, (128, 128))
                class_name = os.path.basename(os.path.dirname(video_path))
                class_name = LABEL_ALIASES.get(class_name, class_name)
                vid_hash = os.path.splitext(os.path.basename(video_path))[0]
                crop_path = os.path.join(crop_dir, f"{class_name}_{vid_hash}.jpg")
                cv2.imwrite(crop_path, crop)

        preprocessed, centers, scales = [], [], []
        for bgr in frames:
            img, c, s = preprocess_frame(bgr, bbox=bbox)
            preprocessed.append(img)
            centers.append(c)
            scales.append(s)
        prep = np.stack(preprocessed).astype(np.float32)
        centers = np.stack(centers)
        scales = np.stack(scales)

        # CoreML: batch=1, CPU: batch=32
        if use_coreml:
            all_kps, all_scores = batch_inference_single(session, prep, centers, scales)
        else:
            from extract_batch_fast_v2 import batch_inference
            all_kps, all_scores = batch_inference(session, prep, centers, scales)

        l_seq, r_seq, face_seq, body_seq = [], [], [], []
        l_valid, r_valid, face_valid, body_valid = [], [], [], []

        for i in range(len(frames)):
            fi = indices[i] if total_frames > max_process else i
            kps = all_kps[i]
            scs = all_scores[i]
            h, w = frames[i].shape[:2]
            if len(kps) < 133:
                continue
            if scs[_LHAND_START:_LHAND_START + 21].mean() >= HAND_CONF:
                coords = [[kps[_LHAND_START + j][0] / w, kps[_LHAND_START + j][1] / h, 0.0] for j in range(21)]
                if not l_valid or l_valid[-1] != fi:
                    l_seq.append(coords); l_valid.append(fi)
            if scs[_RHAND_START:_RHAND_START + 21].mean() >= HAND_CONF:
                coords = [[kps[_RHAND_START + j][0] / w, kps[_RHAND_START + j][1] / h, 0.0] for j in range(21)]
                if not r_valid or r_valid[-1] != fi:
                    r_seq.append(coords); r_valid.append(fi)
            fc = []
            for fi_idx in _FACE_INDICES:
                if fi_idx < len(kps) and scs[fi_idx] >= FACE_CONF:
                    fc.append([kps[fi_idx][0] / w, kps[fi_idx][1] / h, 0.0])
                else:
                    break
            if len(fc) == NUM_FACE_V2:
                if not face_valid or face_valid[-1] != fi:
                    face_seq.append(fc); face_valid.append(fi)
            bc = []
            for bi in _BODY_INDICES:
                if bi < len(kps) and scs[bi] >= BODY_CONF:
                    bc.append([kps[bi][0] / w, kps[bi][1] / h, 0.0])
                else:
                    break
            if len(bc) == NUM_BODY_V2:
                if not body_valid or body_valid[-1] != fi:
                    body_seq.append(bc); body_valid.append(fi)

        if not l_valid and not r_valid:
            return None
        if l_valid:
            l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
        if r_valid:
            r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
        if not l_valid and not r_valid:
            return None

        l_ever, r_ever = bool(l_valid), bool(r_valid)
        face_ever, body_ever = bool(face_valid), bool(body_valid)

        l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
        r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
        face_full = interpolate_generic(
            np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
            face_valid, total_frames, NUM_FACE_V2)
        body_full = interpolate_generic(
            np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
            body_valid, total_frames, NUM_BODY_V2)

        combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)
        resampled = temporal_resample(combined, 32)
        resampled[:, :, :3] = one_euro_filter(resampled[:, :, :3])
        if l_ever:
            resampled = stabilize_bones(resampled, 0, 21)
        if r_ever:
            resampled = stabilize_bones(resampled, 21, 42)
        normalized = normalize_sequence_v2(resampled, l_ever, r_ever)
        data = compute_kinematics_v2(normalized, l_ever, r_ever, face_ever, body_ever)
        data = data.astype(np.float16)

        if data.shape != (32, NUM_NODES_V2, 10):
            return None

        class_name = os.path.basename(os.path.dirname(video_path))
        class_name = LABEL_ALIASES.get(class_name, class_name)
        vid_hash = os.path.splitext(os.path.basename(video_path))[0]
        out_hash = hashlib.md5(data.tobytes()).hexdigest()[:6]
        out_name = f"{class_name}_{vid_hash}_{out_hash}.npy"
        np.save(os.path.join(npy_dir, out_name), data)
        return out_name
    except Exception as e:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoreML extraction for SLT")
    parser.add_argument("--input", default="data/raw_videos/ASL VIDEOS")
    parser.add_argument("--output", default="ASL_landmarks_coreml")
    parser.add_argument("--crops", default="ASL_hand_crops")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true", default=True,
                        help="Use CPU (default, fast with multiprocessing). CoreML is for single-process only.")
    args = parser.parse_args()

    rtmw_path = glob.glob(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_*.onnx"))[0]
    det_path = glob.glob(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/yolox_m_*.onnx"))[0]
    use_coreml = not args.cpu

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.crops, exist_ok=True)
    existing = set(os.listdir(args.output)) if args.resume else set()

    tasks = []
    for class_dir in sorted(os.listdir(args.input)):
        class_path = os.path.join(args.input, class_dir)
        if not os.path.isdir(class_path):
            continue
        for vid in sorted(os.listdir(class_path)):
            if not vid.endswith('.mp4'):
                continue
            vid_hash = os.path.splitext(vid)[0]
            canonical = LABEL_ALIASES.get(class_dir, class_dir)
            if args.resume and any(f.startswith(f"{canonical}_{vid_hash}_") for f in existing):
                continue
            tasks.append((os.path.join(class_path, vid), args.output, args.crops,
                         rtmw_path, det_path, use_coreml))

    total_tasks = len(tasks)
    provider = "CoreML (Neural Engine)" if use_coreml else "CPU"
    print(f"Extracting {total_tasks} videos with {args.workers} workers")
    print(f"Provider: {provider}")
    print(f"Output: {args.output} (keypoints) + {args.crops} (hand crops)")
    print()

    done = 0
    failed = 0
    t0 = time.time()

    # CoreML doesn't work well with multiprocessing fork — use spawn
    ctx = mp.get_context('spawn') if use_coreml and args.workers > 1 else mp
    with ctx.Pool(args.workers) as pool:
        for result in pool.imap_unordered(extract_one, tasks):
            done += 1
            if result is None:
                failed += 1
            if done % 100 == 0 or done == 10:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total_tasks - done) / rate
                eta_h = int(eta // 3600)
                eta_m = int((eta % 3600) // 60)
                pct = done / total_tasks * 100
                print(f"  {done}/{total_tasks} ({pct:.1f}%) | {rate:.1f} vids/s | "
                      f"ETA: {eta_h}h{eta_m:02d}m | failed: {failed}", flush=True)

    elapsed = time.time() - t0
    npy_count = len([f for f in os.listdir(args.output) if f.endswith('.npy')])
    crop_count = len([f for f in os.listdir(args.crops) if f.endswith('.jpg')])

    # Generate manifest.json
    manifest = {}
    for f in sorted(os.listdir(args.output)):
        if not f.endswith('.npy'):
            continue
        parts = f.replace('.npy', '').rsplit('_', 2)
        if len(parts) >= 3:
            manifest[f] = parts[0]
        elif len(parts) == 2:
            manifest[f] = parts[0]
    manifest_path = os.path.join(args.output, 'manifest.json')
    with open(manifest_path, 'w') as fp:
        json.dump(manifest, fp, indent=2)

    unique_classes = sorted(set(manifest.values()))
    print(f"\nDone: {npy_count} .npy files, {crop_count} hand crops in {elapsed / 60:.1f} minutes")
    print(f"Failed: {failed}")
    print(f"Manifest: {len(manifest)} files, {len(unique_classes)} classes")
    print(f"Saved to: {args.output}/manifest.json")
