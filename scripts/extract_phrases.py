"""
Extract continuous phrase videos for Stage 2 CTC fine-tuning.
Uses EXACT same pipeline as extract_batch_fast_v2.py (YOLOX-m + RTMW-xl 384x288).
61 nodes: 21 LHand + 21 RHand + 15 Face + 4 Body.
Outputs: [N*32, 61, 10] per video (split into 32-frame clips, each processed independently).

Usage:
    python scripts/extract_phrases.py --input data/raw_videos/PHRASES --output ASL_phrases_extracted
"""
import os, sys, json, glob, argparse, hashlib
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract import (
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
    PipelineConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_batch_fast_v2 import (
    interpolate_generic, normalize_sequence_v2, compute_kinematics_v2,
    preprocess_frame, batch_inference,
    _FACE_INDICES, _BODY_INDICES,
    NUM_NODES_V2, NUM_FACE_V2, NUM_BODY_V2,
    HAND_CONF, FACE_CONF, BODY_CONF,
)
import onnxruntime as ort

_LHAND_START = 91
_RHAND_START = 112

# Gloss mapping for phrases
PHRASE_GLOSSES = {
    'GOOD_MORNING': 'GOOD MORNING',
    'HELLO_HOW_YOU': 'HELLO HOW YOU',
    'I_WANT_FOOD': 'I WANT EAT_FOOD',
    'MY_NAME': 'MY NAME',
    'PLEASE_HELP_ME': 'PLEASE HELP ME',
    'SORRY_I_LATE': 'SORRY I LATE',
    'THANKYOU_FRIEND': 'THANK-YOU FRIEND',
    'TOMORROW_SCHOOL_GO': 'TOMORROW SCHOOL GO',
    'YESTERDAY_TEACHER_MEET': 'YESTERDAY TEACHER MEET',
}


def extract_video_phrases(video_path, ort_session, det_model):
    """Extract using EXACT same pipeline as training (extract_batch_fast_v2).
    Returns [N*32, 61, 10] — variable number of 32-frame clips."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    total_frames = len(raw_frames)
    if total_frames < 8:
        return None

    # Subsample if too long (same as extract_batch_fast_v2)
    max_process = 256  # higher limit for phrases (longer videos)
    indices = list(range(total_frames))
    if total_frames > max_process:
        step = total_frames / max_process
        indices = [int(i * step) for i in range(max_process)]
        raw_frames = [raw_frames[i] for i in indices]

    # YOLOX detection on first frame — fixed bbox for all frames (SAME AS TRAINING)
    h0, w0 = raw_frames[0].shape[:2]
    if det_model is not None:
        bboxes = det_model(raw_frames[0])
        if bboxes is not None and len(bboxes) > 0:
            bbox = bboxes[0][:4]
        else:
            bbox = np.array([0, 0, w0, h0], dtype=np.float32)
    else:
        bbox = np.array([0, 0, w0, h0], dtype=np.float32)

    # Preprocess all frames with SAME bbox (SAME AS TRAINING)
    preprocessed, centers, scales_list = [], [], []
    for bgr in raw_frames:
        img, center, scale = preprocess_frame(bgr, bbox=bbox)
        preprocessed.append(img)
        centers.append(center)
        scales_list.append(scale)
    centers = np.stack(centers)
    scales_arr = np.stack(scales_list)

    # Batch ONNX inference (SAME AS TRAINING)
    all_kps, all_scores = [], []
    bs = 32
    for start in range(0, len(preprocessed), bs):
        end = min(start + bs, len(preprocessed))
        batch = np.stack(preprocessed[start:end]).astype(np.float32)
        kps, scs = batch_inference(ort_session, batch, centers[start:end], scales_arr[start:end])
        all_kps.append(kps)
        all_scores.append(scs)
    all_kps = np.concatenate(all_kps, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # Parse keypoints into sequences (SAME AS TRAINING)
    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []

    for i in range(len(raw_frames)):
        fi = indices[i] if total_frames > max_process else i
        kps = all_kps[i]
        scs = all_scores[i]
        h, w = raw_frames[i].shape[:2]

        if len(kps) < 133:
            continue

        # Left hand
        l_scs = scs[_LHAND_START:_LHAND_START+21]
        if l_scs.mean() >= HAND_CONF:
            coords = [[kps[_LHAND_START+j][0]/w, kps[_LHAND_START+j][1]/h, 0.0] for j in range(21)]
            l_seq.append(coords); l_valid.append(fi)

        # Right hand
        r_scs = scs[_RHAND_START:_RHAND_START+21]
        if r_scs.mean() >= HAND_CONF:
            coords = [[kps[_RHAND_START+j][0]/w, kps[_RHAND_START+j][1]/h, 0.0] for j in range(21)]
            r_seq.append(coords); r_valid.append(fi)

        # Face (15 nodes)
        face_coords = []
        for fi_idx in _FACE_INDICES:
            if fi_idx < len(kps) and scs[fi_idx] >= FACE_CONF:
                face_coords.append([kps[fi_idx][0]/w, kps[fi_idx][1]/h, 0.0])
            else:
                break
        if len(face_coords) == NUM_FACE_V2:
            face_seq.append(face_coords); face_valid.append(fi)

        # Body (4 nodes)
        body_coords = []
        for bi in _BODY_INDICES:
            if bi < len(kps) and scs[bi] >= BODY_CONF:
                body_coords.append([kps[bi][0]/w, kps[bi][1]/h, 0.0])
            else:
                break
        if len(body_coords) == NUM_BODY_V2:
            body_seq.append(body_coords); body_valid.append(fi)

    if not l_valid and not r_valid:
        return None

    # Temporal outlier rejection (SAME AS TRAINING)
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)

    if not l_valid and not r_valid:
        return None

    l_ever = bool(l_valid)
    r_ever = bool(r_valid)
    face_ever = bool(face_valid)
    body_ever = bool(body_valid)

    # Interpolate to full frame count (SAME AS TRAINING)
    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
                              l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
                              r_valid, total_frames)
    face_full = interpolate_generic(np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
                                    face_valid, total_frames, NUM_FACE_V2)
    body_full = interpolate_generic(np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
                                    body_valid, total_frames, NUM_BODY_V2)

    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)  # [T, 61, 3]

    # Split into 32-frame clips, process each independently (SAME AS TRAINING)
    T = combined.shape[0]
    num_clips = T // 32
    remainder = T % 32

    if num_clips == 0:
        pad = np.zeros((32 - T, 61, 3), dtype=np.float32)
        combined = np.concatenate([combined, pad], axis=0)
        num_clips = 1
    elif remainder > 0:
        pad = np.zeros((32 - remainder, 61, 3), dtype=np.float32)
        combined = np.concatenate([combined, pad], axis=0)
        num_clips += 1

    clips_processed = []
    for ci in range(num_clips):
        seg = combined[ci*32:(ci+1)*32].copy()

        # Per-clip processing (SAME AS TRAINING)
        smoothed = one_euro_filter(seg[:, :, :3])
        seg[:, :, :3] = smoothed
        if l_ever:
            seg = stabilize_bones(seg, 0, 21)
        if r_ever:
            seg = stabilize_bones(seg, 21, 42)
        normed = normalize_sequence_v2(seg, l_ever, r_ever)
        clip_10ch = compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever)
        clips_processed.append(clip_10ch)

    result = np.concatenate(clips_processed, axis=0)  # [N*32, 61, 10]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw_videos/PHRASES')
    parser.add_argument('--output', default='ASL_phrases_extracted')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load EXACT same models as training extraction
    model_path = os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx")
    if not os.path.exists(model_path):
        print(f"RTMW-xl model not found at {model_path}")
        print("Download: rtmlib Wholebody(mode='performance') to cache it")
        return

    print("Loading RTMW-xl 384x288 (same as training)...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    print(f"  Provider: {ort_session.get_providers()[0]}")

    # Load YOLOX-m detector (same as training)
    det_model = None
    try:
        from rtmlib import Wholebody
        _wb = Wholebody(to_openpose=False, mode='performance', backend='onnxruntime', device='cuda')
        det_model = _wb.det_model
        print("  YOLOX-m detector loaded")
    except Exception as e:
        print(f"  WARNING: Could not load YOLOX-m ({e}), using full-frame bbox")

    # Find phrase directories
    phrase_dirs = sorted([d for d in os.listdir(args.input)
                         if os.path.isdir(os.path.join(args.input, d))])
    print(f"\nFound {len(phrase_dirs)} phrases: {phrase_dirs}")

    manifest = {}
    total, success, failed = 0, 0, 0

    for phrase_dir in phrase_dirs:
        glosses = PHRASE_GLOSSES.get(phrase_dir, phrase_dir.replace('_', ' '))
        phrase_path = os.path.join(args.input, phrase_dir)
        videos = sorted(glob.glob(os.path.join(phrase_path, '*.mp4')) +
                       glob.glob(os.path.join(phrase_path, '*.mov')) +
                       glob.glob(os.path.join(phrase_path, '*.webm')))

        print(f"\n[{phrase_dir}] {len(videos)} videos -> glosses: '{glosses}'")

        for vi, vpath in enumerate(videos):
            total += 1
            result = extract_video_phrases(vpath, ort_session, det_model)

            if result is None:
                failed += 1
                continue

            vid_hash = hashlib.md5(vpath.encode()).hexdigest()[:8]
            fname = f"{phrase_dir}_{vi:04d}_{vid_hash}.npy"
            np.save(os.path.join(args.output, fname), result.astype(np.float16))
            manifest[fname] = glosses
            success += 1

            if (vi + 1) % 10 == 0:
                print(f"  [{vi+1}/{len(videos)}] shape: {result.shape}")

    with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {success}/{total} extracted, {failed} failed")
    print(f"Output: {args.output}/ ({len(manifest)} files)")

    if manifest:
        shapes = []
        for fname in list(manifest.keys())[:20]:
            arr = np.load(os.path.join(args.output, fname))
            shapes.append(arr.shape)
        frames = [s[0] for s in shapes]
        print(f"Frame counts: min={min(frames)}, max={max(frames)}, nodes={shapes[0][1]}, channels={shapes[0][2]}")


if __name__ == "__main__":
    main()
