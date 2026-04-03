"""
CPU-only ONNX extraction for SLT.
Matches Mac CPU inference exactly by using CPUExecutionProvider.
"""
import os, sys, glob, hashlib, warnings, time, argparse
import multiprocessing as mp
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import onnxruntime as ort

# Add SLT root and src to path (works regardless of where script is located)
_slt_root = os.environ.get('SLT_ROOT', os.getcwd())
sys.path.insert(0, _slt_root)
sys.path.insert(0, os.path.join(_slt_root, 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_batch_fast_v2 import (
    preprocess_frame, batch_inference, interpolate_generic,
    normalize_sequence_v2, compute_kinematics_v2,
    _FACE_INDICES, _BODY_INDICES, _LHAND_START, _RHAND_START,
    NUM_NODES_V2, NUM_FACE_V2, NUM_BODY_V2,
    HAND_CONF, FACE_CONF, BODY_CONF, PipelineConfig,
)
from extract import (
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)


def extract_one(args):
    video_path, output_dir, rtmw_path, det_path = args
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(rtmw_path, sess_options=opts, providers=['CPUExecutionProvider'])
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

        preprocessed, centers, scales = [], [], []
        for bgr in frames:
            img, c, s = preprocess_frame(bgr, bbox=bbox)
            preprocessed.append(img)
            centers.append(c)
            scales.append(s)
        centers = np.stack(centers)
        scales = np.stack(scales)

        all_kps, all_scores = [], []
        bs = 32
        for start in range(0, len(preprocessed), bs):
            end = min(start + bs, len(preprocessed))
            batch = np.stack(preprocessed[start:end]).astype(np.float32)
            kps, scs = batch_inference(session, batch, centers[start:end], scales[start:end])
            all_kps.append(kps)
            all_scores.append(scs)
        all_kps = np.concatenate(all_kps)
        all_scores = np.concatenate(all_scores)

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
                    l_seq.append(coords)
                    l_valid.append(fi)
            if scs[_RHAND_START:_RHAND_START + 21].mean() >= HAND_CONF:
                coords = [[kps[_RHAND_START + j][0] / w, kps[_RHAND_START + j][1] / h, 0.0] for j in range(21)]
                if not r_valid or r_valid[-1] != fi:
                    r_seq.append(coords)
                    r_valid.append(fi)
            fc = []
            for fi_idx in _FACE_INDICES:
                if fi_idx < len(kps) and scs[fi_idx] >= FACE_CONF:
                    fc.append([kps[fi_idx][0] / w, kps[fi_idx][1] / h, 0.0])
                else:
                    break
            if len(fc) == NUM_FACE_V2:
                if not face_valid or face_valid[-1] != fi:
                    face_seq.append(fc)
                    face_valid.append(fi)
            bc = []
            for bi in _BODY_INDICES:
                if bi < len(kps) and scs[bi] >= BODY_CONF:
                    bc.append([kps[bi][0] / w, kps[bi][1] / h, 0.0])
                else:
                    break
            if len(bc) == NUM_BODY_V2:
                if not body_valid or body_valid[-1] != fi:
                    body_seq.append(bc)
                    body_valid.append(fi)

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
        vid_hash = os.path.splitext(os.path.basename(video_path))[0]
        out_hash = hashlib.md5(data.tobytes()).hexdigest()[:6]
        out_name = f"{class_name}_{vid_hash}_{out_hash}.npy"
        np.save(os.path.join(output_dir, out_name), data)
        return out_name
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only ONNX extraction")
    parser.add_argument("--input", default="data/raw_videos/ASL VIDEOS")
    parser.add_argument("--output", default="ASL_landmarks_cpu")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rtmw_path = glob.glob(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_*.onnx"))[0]
    det_path = glob.glob(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/yolox_m_*.onnx"))[0]

    os.makedirs(args.output, exist_ok=True)
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
            if args.resume and any(f.startswith(f"{class_dir}_{vid_hash}_") for f in existing):
                continue
            tasks.append((os.path.join(class_path, vid), args.output, rtmw_path, det_path))

    total_tasks = len(tasks)
    print(f"Extracting {total_tasks} videos with {args.workers} CPU workers...")
    print(f"ONNX provider: CPUExecutionProvider (matching Mac inference)")

    done = 0
    failed = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
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
                print(f"  {done}/{total_tasks} ({pct:.1f}%) | {rate:.1f} vids/s | ETA: {eta_h}h{eta_m:02d}m | failed: {failed}", flush=True)

    elapsed = time.time() - t0
    total_files = len([f for f in os.listdir(args.output) if f.endswith('.npy')])
    print(f"\nDone: {total_files} files in {elapsed / 60:.1f} minutes ({failed} failed)")
