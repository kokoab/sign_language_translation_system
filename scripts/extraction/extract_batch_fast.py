"""
Fast batch extraction with RTMW — GPU batch inference.
Processes multiple frames per GPU call for 10-20x speedup over rtmlib's frame-by-frame approach.

Usage:
    python extract_batch_fast.py --input /workspace/data --output ASL_landmarks_rtmlib --batch_size 32
"""
import os, sys, glob, json, hashlib, warnings, time, argparse
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

import onnxruntime as ort

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

# RTMW input size
INPUT_H = 384
INPUT_W = 288


def _bbox_xyxy2cs(bbox, padding=1.25):
    """Convert bbox [x1,y1,x2,y2] to center, scale. Matches rtmlib exactly."""
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def _get_warp_matrix(center, scale, rot, output_size):
    """Get affine warp matrix. Matches rtmlib exactly."""
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
    """Preprocess a frame for RTMW using rtmlib-identical affine transform.
    Returns preprocessed [3, 384, 288] float32 and (center, scale) for postprocess."""
    h, w = bgr.shape[:2]
    if bbox is None:
        bbox = np.array([0, 0, w, h], dtype=np.float32)
    center, scale = _bbox_xyxy2cs(bbox, padding=1.25)

    # Reshape scale to fixed aspect ratio (matches rtmlib)
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
    """Decode SimCC heatmaps to keypoint coordinates in model input space.
    simcc_x: [N, 133, W*2], simcc_y: [N, 133, H*2]
    Returns: locs [N, 133, 2] in input pixel coords, scores [N, 133]"""
    x_locs = simcc_x.argmax(axis=2).astype(np.float32) / 2.0  # [N, 133]
    y_locs = simcc_y.argmax(axis=2).astype(np.float32) / 2.0  # [N, 133]
    x_conf = simcc_x.max(axis=2)
    y_conf = simcc_y.max(axis=2)
    scores = np.minimum(x_conf, y_conf)
    locs = np.stack([x_locs, y_locs], axis=-1)  # [N, 133, 2]
    return locs, scores


def postprocess_keypoints(locs, scores, centers, scales):
    """Rescale keypoints from model input space to original image coords.
    Matches rtmlib postprocess exactly.
    locs: [N, 133, 2], centers: [N, 2], scales: [N, 2]
    Returns: keypoints [N, 133, 2] in original image coords"""
    model_size = np.array([[INPUT_W, INPUT_H]], dtype=np.float32)  # [1, 2]
    # keypoints = locs / model_input_size * scale + center - scale/2
    kps = locs / model_size * scales[:, np.newaxis, :] + centers[:, np.newaxis, :] - scales[:, np.newaxis, :] / 2
    return kps, scores


def batch_inference(session, frames_preprocessed, centers, scales):
    """Run RTMW on a batch of preprocessed frames.
    Returns: keypoints [N, 133, 2] in original coords, scores [N, 133]"""
    outputs = session.run(None, {'input': frames_preprocessed})
    simcc_x, simcc_y = outputs[0], outputs[1]
    locs, scores = decode_simcc_batch(simcc_x, simcc_y)
    keypoints, scores = postprocess_keypoints(locs, scores, centers, scales)
    return keypoints, scores


def extract_video(video_path, session, cfg, batch_size=32, det_model=None):
    """Extract one video → [32, 47, 10] float16 or None.
    Uses YOLOX detection on first frame, then batch RTMW inference on all frames."""
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

    # Detect person bbox on first frame using YOLOX (run once per video)
    h0, w0 = raw_frames[0].shape[:2]
    if det_model is not None:
        bboxes = det_model(raw_frames[0])
        if bboxes is not None and len(bboxes) > 0:
            bbox = bboxes[0][:4]  # Use first detected person
        else:
            bbox = np.array([0, 0, w0, h0], dtype=np.float32)
    else:
        bbox = np.array([0, 0, w0, h0], dtype=np.float32)

    # Preprocess all frames using the detected bbox (rtmlib-identical affine transform)
    preprocessed = []
    centers = []
    scales = []
    for bgr in raw_frames:
        img, center, scale = preprocess_frame(bgr, bbox=bbox)
        preprocessed.append(img)
        centers.append(center)
        scales.append(scale)

    centers = np.stack(centers)  # [N, 2]
    scales = np.stack(scales)    # [N, 2]

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

    all_kps = np.concatenate(all_kps, axis=0)      # [N, 133, 2] in original image coords
    all_scores = np.concatenate(all_scores, axis=0)  # [N, 133]

    # Parse keypoints into hand/face sequences
    l_seq, r_seq, face_seq = [], [], []
    l_valid, r_valid, face_valid = [], [], []

    for i in range(len(raw_frames)):
        fi = indices[i] if total_frames > max_process else i
        kps = all_kps[i]    # [133, 2] in original pixel coords
        scs = all_scores[i]  # [133]
        h, w = raw_frames[i].shape[:2]

        if len(kps) < 133:
            continue

        # Normalize to [0, 1] range (same as rtmlib extraction in extract_batch_rtmlib.py)
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


def main():
    parser = argparse.ArgumentParser(description="Fast batch extraction with RTMW GPU batch inference")
    parser.add_argument("--input", default="data/raw_videos/ASL VIDEOS", help="Root video directory")
    parser.add_argument("--output", default="ASL_landmarks_rtmlib", help="Output .npy directory")
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

    # Initialize YOLOX detector (run once per video for person bbox)
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
        rate = (stats["success"] + stats["fail"]) / max(elapsed, 1)
        remaining = sum(1 for d in class_dirs[ci+1:] for _ in glob.glob(os.path.join(input_dir, d, "*.mp4")))
        eta = remaining / max(rate, 0.01)
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
