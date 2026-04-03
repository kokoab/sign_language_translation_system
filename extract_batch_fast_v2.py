"""
Fast batch extraction v2 — includes body keypoints + expanded face landmarks.
61 nodes: 21 LHand + 21 RHand + 15 Face + 2 Shoulders + 2 Elbows
Output: [32, 61, 10] float16

Usage:
    python extract_batch_fast_v2.py --input /workspace/data --output ASL_landmarks_v2 --batch_size 32
"""
import os, sys, glob, json, hashlib, warnings, time, argparse
warnings.filterwarnings('ignore')
import numpy as np
import cv2
# No torch needed — extraction is pure numpy + onnxruntime

# Fake mediapipe to avoid import error from extract.py
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from extract import (
    PipelineConfig, LABEL_ALIASES,
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)


def interpolate_generic(seq, valid_indices, total_frames, num_nodes):
    """Interpolate N landmarks across frames. Generic version for any node count."""
    if not valid_indices:
        return np.zeros((total_frames, num_nodes, 3), dtype=np.float32)
    flat = seq.reshape(len(seq), -1)
    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, num_nodes, 3).astype(np.float32)
    x_new = np.arange(total_frames, dtype=np.float64)
    xp = np.array(valid_indices, dtype=np.float64)
    result = np.column_stack([
        np.interp(x_new, xp, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(total_frames, num_nodes, 3).astype(np.float32)

import onnxruntime as ort

# ============================================================
# COCO-WholeBody 133 keypoint indices
# ============================================================

# Body keypoints we want
_BODY_L_SHOULDER = 5
_BODY_R_SHOULDER = 6
_BODY_L_ELBOW = 7
_BODY_R_ELBOW = 8

# Hand keypoints (same as before)
_LHAND_START = 91
_RHAND_START = 112

# Face keypoints — expanded from 5 to 15
# Original 5:
_FACE_NOSE = 23 + 30       # tip of nose
_FACE_CHIN = 23 + 8        # chin
_FACE_FOREHEAD = 23 + 27   # between eyebrows / forehead
_FACE_LEFT_EAR = 23 + 0    # left jaw/ear
_FACE_RIGHT_EAR = 23 + 16  # right jaw/ear
# New 10:
_FACE_L_MOUTH = 23 + 48    # left mouth corner
_FACE_R_MOUTH = 23 + 54    # right mouth corner
_FACE_UPPER_LIP = 23 + 51  # upper lip center
_FACE_LOWER_LIP = 23 + 57  # lower lip center
_FACE_L_EYEBROW_IN = 23 + 17   # left eyebrow inner
_FACE_L_EYEBROW_OUT = 23 + 21  # left eyebrow outer
_FACE_R_EYEBROW_IN = 23 + 22   # right eyebrow inner
_FACE_R_EYEBROW_OUT = 23 + 26  # right eyebrow outer
_FACE_L_EYE = 23 + 36      # left eye inner corner
_FACE_R_EYE = 23 + 45      # right eye outer corner

# All face indices in order (15 total)
_FACE_INDICES = [
    _FACE_NOSE, _FACE_CHIN, _FACE_FOREHEAD, _FACE_LEFT_EAR, _FACE_RIGHT_EAR,
    _FACE_L_MOUTH, _FACE_R_MOUTH, _FACE_UPPER_LIP, _FACE_LOWER_LIP,
    _FACE_L_EYEBROW_IN, _FACE_L_EYEBROW_OUT, _FACE_R_EYEBROW_IN, _FACE_R_EYEBROW_OUT,
    _FACE_L_EYE, _FACE_R_EYE,
]

# Body indices we extract (4 total)
_BODY_INDICES = [_BODY_L_SHOULDER, _BODY_R_SHOULDER, _BODY_L_ELBOW, _BODY_R_ELBOW]

# New node layout (61 total):
# 0-20:  Left hand (21)
# 21-41: Right hand (21)
# 42-56: Face (15)
# 57:    Left shoulder
# 58:    Right shoulder
# 59:    Left elbow
# 60:    Right elbow
NUM_NODES_V2 = 61
NUM_FACE_V2 = 15
NUM_BODY_V2 = 4

HAND_CONF = 0.25
FACE_CONF = 0.25
BODY_CONF = 0.30

# RTMW input size
INPUT_H = 384
INPUT_W = 288


def standardize_bbox(bbox, frame_h, frame_w):
    """Standardize YOLOX bbox to reduce domain gap from varying crops.

    Uses detected person center but enforces a consistent crop scale,
    so RTMW always sees the person at a similar resolution regardless
    of camera distance or YOLOX detection jitter.
    """
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Standard crop: 80% of frame width, 90% of frame height
    # This matches typical training video proportions where the signer
    # fills most of the frame
    std_w = frame_w * 0.80
    std_h = frame_h * 0.90

    # Center on detected person, clamp to frame
    sx1 = max(0, cx - std_w / 2)
    sy1 = max(0, cy - std_h / 2)
    sx2 = min(frame_w, cx + std_w / 2)
    sy2 = min(frame_h, cy + std_h / 2)

    return np.array([sx1, sy1, sx2, sy2], dtype=np.float32)


# ============================================================
# Preprocessing (same as v1 — rtmlib-identical affine transform)
# ============================================================

def _bbox_xyxy2cs(bbox, padding=1.25):
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale

def _get_warp_matrix(center, scale, rot, output_size):
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array([0., src_w * -0.5])
    src_dir = np.array([src_dir[0] * cs - src_dir[1] * sn, src_dir[0] * sn + src_dir[1] * cs])
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    direction = src[0, :] - src[1, :]
    src[2, :] = src[1, :] + np.array([-direction[1], direction[0]])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
    direction = dst[0, :] - dst[1, :]
    dst[2, :] = dst[1, :] + np.array([-direction[1], direction[0]])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def preprocess_frame(bgr, bbox=None):
    h, w = bgr.shape[:2]
    if bbox is None:
        bbox = np.array([0, 0, w, h], dtype=np.float32)
    center, scale = _bbox_xyxy2cs(bbox, padding=1.25)
    aspect_ratio = INPUT_W / INPUT_H
    if scale[0] > scale[1] * aspect_ratio:
        scale = np.array([scale[0], scale[0] / aspect_ratio], dtype=np.float32)
    else:
        scale = np.array([scale[1] * aspect_ratio, scale[1]], dtype=np.float32)
    warp_mat = _get_warp_matrix(center, scale, 0, (INPUT_W, INPUT_H))
    img = cv2.warpAffine(bgr, warp_mat, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    img = ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
    return img, center, scale

def decode_simcc_batch(simcc_x, simcc_y):
    x_locs = simcc_x.argmax(axis=2).astype(np.float32) / 2.0
    y_locs = simcc_y.argmax(axis=2).astype(np.float32) / 2.0
    x_conf = simcc_x.max(axis=2)
    y_conf = simcc_y.max(axis=2)
    scores = np.minimum(x_conf, y_conf)
    locs = np.stack([x_locs, y_locs], axis=-1)
    return locs, scores

def postprocess_keypoints(locs, scores, centers, scales):
    model_size = np.array([[INPUT_W, INPUT_H]], dtype=np.float32)
    kps = locs / model_size * scales[:, np.newaxis, :] + centers[:, np.newaxis, :] - scales[:, np.newaxis, :] / 2
    return kps, scores

def batch_inference(session, frames_preprocessed, centers, scales):
    outputs = session.run(None, {'input': frames_preprocessed})
    simcc_x, simcc_y = outputs[0], outputs[1]
    locs, scores = decode_simcc_batch(simcc_x, simcc_y)
    keypoints, scores = postprocess_keypoints(locs, scores, centers, scales)
    return keypoints, scores


# ============================================================
# Custom normalize/kinematics for 61-node layout
# ============================================================

def normalize_sequence_v2(seq, l_ever, r_ever):
    """Normalize [T, 61, 3] sequence: center on wrists, scale by hand size."""
    L_WRIST, R_WRIST = 0, 21
    L_MIDDLE_MCP, R_MIDDLE_MCP = 9, 30

    valid_wrists = []
    if l_ever: valid_wrists.append(seq[:, L_WRIST, :])
    if r_ever: valid_wrists.append(seq[:, R_WRIST, :])

    if valid_wrists:
        all_w = np.concatenate(valid_wrists, axis=0)
        nonzero = all_w[np.linalg.norm(all_w, axis=-1) > 1e-6]
        center = np.median(nonzero, axis=0) if len(nonzero) > 0 else np.zeros(3)
    else:
        center = np.zeros(3)

    seq = seq.copy()
    seq[:, :, :] -= center

    # Scale by hand size
    scales = []
    for wr, mc in [(L_WRIST, L_MIDDLE_MCP), (R_WRIST, R_MIDDLE_MCP)]:
        dists = np.linalg.norm(seq[:, mc, :] - seq[:, wr, :], axis=-1)
        valid = dists[dists > 1e-6]
        if len(valid) > 0:
            scales.append(np.median(valid))
    if scales:
        scale_factor = np.mean(scales)
        if scale_factor > 1e-6:
            seq /= scale_factor

    return seq


def compute_kinematics_v2(seq, l_ever, r_ever, face_ever, body_ever):
    """Compute [32, 61, 10] from [32, 61, 3] XYZ.
    Channels: xyz(3) + vel(3) + acc(3) + mask(1) = 10"""
    from scipy.signal import savgol_filter

    T, N, _ = seq.shape  # N=61
    xyz = seq.copy()

    # Velocity and acceleration via savgol
    vel = np.zeros_like(xyz)
    acc = np.zeros_like(xyz)
    wl = min(7, T if T % 2 == 1 else T - 1)
    if wl >= 3 and T >= wl:
        for p in range(N):
            for c in range(3):
                vel[:, p, c] = savgol_filter(xyz[:, p, c], window_length=wl, polyorder=2, deriv=1)
                acc[:, p, c] = savgol_filter(xyz[:, p, c], window_length=wl, polyorder=2, deriv=2)

    # Mask
    mask = np.zeros((T, N, 1), dtype=np.float32)
    if l_ever: mask[:, 0:21, 0] = 1.0
    if r_ever: mask[:, 21:42, 0] = 1.0
    if face_ever: mask[:, 42:57, 0] = 1.0
    if body_ever: mask[:, 57:61, 0] = 1.0

    return np.concatenate([xyz, vel, acc, mask], axis=-1).astype(np.float32)


# ============================================================
# Video extraction
# ============================================================

def extract_video(video_path, session, cfg, batch_size=32, det_model=None):
    """Extract one video → [32, 61, 10] float16 or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    total_frames = len(raw_frames)
    if total_frames < cfg.min_raw_frames:
        return None

    # Subsample if too long
    max_process = 128
    indices = list(range(total_frames))
    if total_frames > max_process:
        step = total_frames / max_process
        indices = [int(i * step) for i in range(max_process)]
        raw_frames = [raw_frames[i] for i in indices]

    # Detect person bbox on first frame
    h0, w0 = raw_frames[0].shape[:2]
    if det_model is not None:
        bboxes = det_model(raw_frames[0])
        if bboxes is not None and len(bboxes) > 0:
            bbox = bboxes[0][:4].astype(np.float32)
        else:
            bbox = np.array([0, 0, w0, h0], dtype=np.float32)
    else:
        bbox = np.array([0, 0, w0, h0], dtype=np.float32)

    # Preprocess all frames
    preprocessed = []
    centers = []
    scales = []
    for bgr in raw_frames:
        img, center, scale = preprocess_frame(bgr, bbox=bbox)
        preprocessed.append(img)
        centers.append(center)
        scales.append(scale)

    centers = np.stack(centers)
    scales = np.stack(scales)

    # Batch GPU inference
    all_kps = []
    all_scores = []
    for start in range(0, len(preprocessed), batch_size):
        end = min(start + batch_size, len(preprocessed))
        batch = np.stack(preprocessed[start:end]).astype(np.float32)
        batch_centers = centers[start:end]
        batch_scales = scales[start:end]
        kps, scs = batch_inference(session, batch, batch_centers, batch_scales)
        all_kps.append(kps)
        all_scores.append(scs)

    all_kps = np.concatenate(all_kps, axis=0)      # [N, 133, 2]
    all_scores = np.concatenate(all_scores, axis=0)  # [N, 133]

    # Parse keypoints into hand/face/body sequences
    l_seq, r_seq = [], []
    l_valid, r_valid = [], []
    face_seq, face_valid = [], []
    body_seq, body_valid = [], []

    for i in range(len(raw_frames)):
        fi = indices[i] if total_frames > max_process else i
        kps = all_kps[i]
        scs = all_scores[i]
        h, w = raw_frames[i].shape[:2]

        if len(kps) < 133:
            continue

        # Left hand
        l_scores_mean = scs[_LHAND_START:_LHAND_START + 21].mean()
        if l_scores_mean >= HAND_CONF:
            coords = [[kps[_LHAND_START + j][0] / w, kps[_LHAND_START + j][1] / h, 0.0] for j in range(21)]
            if not l_valid or l_valid[-1] != fi:
                l_seq.append(coords)
                l_valid.append(fi)

        # Right hand
        r_scores_mean = scs[_RHAND_START:_RHAND_START + 21].mean()
        if r_scores_mean >= HAND_CONF:
            coords = [[kps[_RHAND_START + j][0] / w, kps[_RHAND_START + j][1] / h, 0.0] for j in range(21)]
            if not r_valid or r_valid[-1] != fi:
                r_seq.append(coords)
                r_valid.append(fi)

        # Face (15 points)
        face_coords = []
        face_ok = True
        for fidx in _FACE_INDICES:
            if fidx < len(kps) and scs[fidx] >= FACE_CONF:
                face_coords.append([kps[fidx][0] / w, kps[fidx][1] / h, 0.0])
            else:
                face_ok = False
                break
        if face_ok and len(face_coords) == NUM_FACE_V2:
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_coords)
                face_valid.append(fi)

        # Body (4 points: L/R shoulder, L/R elbow)
        body_coords = []
        body_ok = True
        for bidx in _BODY_INDICES:
            if bidx < len(kps) and scs[bidx] >= BODY_CONF:
                body_coords.append([kps[bidx][0] / w, kps[bidx][1] / h, 0.0])
            else:
                body_ok = False
                break
        if body_ok and len(body_coords) == NUM_BODY_V2:
            if not body_valid or body_valid[-1] != fi:
                body_seq.append(body_coords)
                body_valid.append(fi)

    if not l_valid and not r_valid:
        return None

    # Temporal outlier rejection (hands only — body/face are stable)
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)
    body_ever = bool(body_valid)

    # Interpolation
    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
    face_full = interpolate_generic(np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)), face_valid, total_frames, NUM_FACE_V2)

    # Body interpolation (same approach as face)
    if body_seq:
        body_arr = np.array(body_seq)
    else:
        body_arr = np.zeros((0, NUM_BODY_V2, 3))
    body_full = interpolate_generic(body_arr, body_valid, total_frames, NUM_BODY_V2)

    # Combine: [T, 61, 3]
    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

    # Process: resample → filter → stabilize → normalize → kinematics
    resampled = temporal_resample(combined, cfg.target_frames)
    # One-euro filter on XYZ
    smoothed = one_euro_filter(resampled[:, :, :3])
    resampled[:, :, :3] = smoothed
    # Stabilize hand bones
    if l_ever:
        resampled = stabilize_bones(resampled, 0, 21)
    if r_ever:
        resampled = stabilize_bones(resampled, 21, 42)
    # Normalize
    normalized = normalize_sequence_v2(resampled, l_ever, r_ever)

    # Compute kinematics → [32, 61, 10]
    data = compute_kinematics_v2(normalized, l_ever, r_ever, face_ever, body_ever)
    data = data.astype(np.float16)

    if data.shape != (32, NUM_NODES_V2, 10):
        return None

    return data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fast batch extraction v2 — body + face keypoints")
    parser.add_argument("--input", default="data/raw_videos/ASL VIDEOS", help="Root video directory")
    parser.add_argument("--output", default="ASL_landmarks_v2", help="Output .npy directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Frames per GPU batch")
    parser.add_argument("--resume", action="store_true", help="Skip already-extracted files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize ONNX Runtime with CUDA
    model_path = os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx")
    if not os.path.exists(model_path):
        print("Downloading RTMW model first...")
        from rtmlib import Wholebody
        wb = Wholebody(to_openpose=False, mode='performance', backend='onnxruntime', device='cuda')
        del wb
        print("Model downloaded.")

    print("Initializing RTMW with CUDA batch inference (batch_size={})...".format(args.batch_size))
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    actual_provider = session.get_providers()[0]
    print("Using provider: {}".format(actual_provider))

    # Initialize YOLOX detector
    from rtmlib import Wholebody
    print("Initializing YOLOX detector...")
    _wb = Wholebody(to_openpose=False, mode='performance', backend='onnxruntime', device='cuda')
    det_model = _wb.det_model
    print("Ready.")

    cfg = PipelineConfig()
    manifest = {}
    stats = {"total": 0, "success": 0, "fail": 0, "skipped": 0}

    # Load existing manifest if resuming
    manifest_path = os.path.join(args.output, "manifest.json")
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print("Resuming: {} files already extracted".format(len(manifest)))

    # Walk all class folders
    input_dir = args.input
    class_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    print("Found {} class folders".format(len(class_dirs)))
    print("Output format: [32, {}, 10] (61 nodes: 42 hand + 15 face + 4 body)".format(NUM_NODES_V2))

    t0 = time.time()

    for ci, class_name in enumerate(class_dirs):
        canonical_label = LABEL_ALIASES.get(class_name, class_name)
        class_path = os.path.join(input_dir, class_name)
        videos = sorted(glob.glob(os.path.join(class_path, "*.mp4")) +
                       glob.glob(os.path.join(class_path, "*.mov")) +
                       glob.glob(os.path.join(class_path, "*.MP4")))

        for vi, video_path in enumerate(videos):
            stats["total"] += 1
            stem = os.path.splitext(os.path.basename(video_path))[0]
            hash_str = hashlib.md5("{}_{}".format(canonical_label, stem).encode()).hexdigest()[:6]
            out_name = "{}_{}_{}.npy".format(canonical_label, stem, hash_str)
            out_path = os.path.join(args.output, out_name)

            if out_name in manifest or (args.resume and os.path.exists(out_path)):
                if out_name not in manifest:
                    manifest[out_name] = canonical_label
                stats["skipped"] += 1
                stats["success"] += 1
                continue

            try:
                data = extract_video(video_path, session, cfg, args.batch_size, det_model=det_model)
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

        # Progress report per class
        elapsed = time.time() - t0
        rate = (stats["success"] + stats["fail"] - stats["skipped"]) / max(elapsed, 1)
        remaining = stats["total"] - stats["success"] - stats["fail"]
        # Estimate remaining from total videos in all remaining classes
        eta = (len(class_dirs) - ci - 1) * (elapsed / max(ci + 1, 1))
        print("[{:>3}/{:>3}] {:20s} | ok:{:>6} fail:{:>4} skip:{:>5} | {:.1f} vid/s | ETA {:.0f}m".format(
            ci + 1, len(class_dirs), canonical_label[:20],
            stats["success"], stats["fail"], stats["skipped"], rate, eta / 60))

        # Save manifest periodically
        if (ci + 1) % 5 == 0:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    # Final save
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE (v2 — 61 nodes)")
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
