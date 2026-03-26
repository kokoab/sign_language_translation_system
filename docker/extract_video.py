"""
Extract landmarks from video(s) using RTMW (inside Docker container).
Produces .npy files identical to training data format: [32, 47, 10] float16.

Usage (from host):
    docker run --rm -v /path/to/video.mp4:/app/input/video.mp4 \
               -v /path/to/output:/app/output \
               slt-extract /app/input/video.mp4

    # Or extract all videos in a folder:
    docker run --rm -v /path/to/videos:/app/input \
               -v /path/to/output:/app/output \
               slt-extract /app/input/
"""
import sys, os, glob, warnings, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

sys.path.insert(0, '/app/src')
from extract import (
    PipelineConfig, NUM_NODES,
    interpolate_hand, interpolate_face, temporal_resample,
    normalize_sequence, compute_kinematics_batch,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
    FACE_LANDMARK_INDICES,
    _detect_pass_rtmw, _detect_pass, _get_rtmw_inferencer,
)
import cv2
import mediapipe as mp


def extract_single_video(video_path, output_dir, label=None):
    """Extract landmarks from a single video using RTMW CPU path.
    Produces a .npy file matching training data format exactly."""

    basename = os.path.splitext(os.path.basename(video_path))[0]
    if label is None:
        label = basename.replace(' ', '_')

    print(f"\n[EXTRACT] {os.path.basename(video_path)}")

    cfg = PipelineConfig()
    cfg.prefer_rtmw = True

    # Read all frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None

    frames_rgb = []
    frame_indices = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        frame_indices.append(idx)
        idx += 1
    cap.release()

    total_frames = len(frames_rgb)
    if total_frames < cfg.min_raw_frames:
        print(f"  ERROR: Too few frames ({total_frames})")
        return None
    print(f"  Frames: {total_frames}")

    # Subsample if too many
    max_process = cfg.target_frames * 3
    if total_frames > max_process:
        step = total_frames / max_process
        selected = [int(i * step) for i in range(max_process)]
        frames_rgb = [frames_rgb[i] for i in selected]
        frame_indices = [frame_indices[i] for i in selected]

    processed_count = len(frames_rgb)

    # Try RTMW first (CPU — this is why we use Docker)
    rw_l_seq, rw_r_seq, rw_face_seq = [], [], []
    rw_l_valid, rw_r_valid, rw_face_valid = [], [], []

    try:
        print("  Running RTMW-l (CPU)...")
        rw_l_seq, rw_r_seq, rw_face_seq, rw_l_valid, rw_r_valid, rw_face_valid = \
            _detect_pass_rtmw(frames_rgb, frame_indices, cfg, device=None)
        rw_coverage = max(len(rw_l_valid), len(rw_r_valid)) / max(processed_count, 1)
        print(f"  RTMW: L={len(rw_l_valid)} R={len(rw_r_valid)} Face={len(rw_face_valid)} ({rw_coverage:.0%})")
    except Exception as e:
        print(f"  RTMW failed: {e}")
        rw_coverage = 0

    # Fallback to MediaPipe if RTMW coverage < 80%
    v_l_seq, v_r_seq, v_face_seq = [], [], []
    v_l_valid, v_r_valid, v_face_valid = [], [], []
    s_l_seq, s_r_seq, s_face_seq = [], [], []
    s_l_valid, s_r_valid, s_face_valid = [], [], []

    if rw_coverage < 0.80:
        print("  Falling back to MediaPipe video mode...")
        hands_v = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=cfg.min_detection_conf,
            min_tracking_confidence=cfg.min_tracking_conf,
            model_complexity=cfg.model_complexity
        )
        face_v = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=cfg.min_detection_conf,
            min_tracking_confidence=cfg.min_tracking_conf,
            refine_landmarks=False,
        )
        v_l_seq, v_r_seq, v_face_seq, v_l_valid, v_r_valid, v_face_valid = \
            _detect_pass(frames_rgb, frame_indices, hands_v, face_v, cfg)
        hands_v.close()
        face_v.close()

        total_coverage = max(len(rw_l_valid), len(rw_r_valid),
                            len(v_l_valid), len(v_r_valid)) / max(processed_count, 1)
        if total_coverage < 0.65:
            print("  Also running MediaPipe static mode...")
            hands_s = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=2,
                min_detection_confidence=cfg.min_detection_conf,
                min_tracking_confidence=cfg.min_tracking_conf,
                model_complexity=cfg.model_complexity
            )
            face_s = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                min_detection_confidence=cfg.min_detection_conf,
                min_tracking_confidence=cfg.min_tracking_conf,
                refine_landmarks=False,
            )
            s_l_seq, s_r_seq, s_face_seq, s_l_valid, s_r_valid, s_face_valid = \
                _detect_pass(frames_rgb, frame_indices, hands_s, face_s, cfg)
            hands_s.close()
            face_s.close()

    del frames_rgb  # Free memory

    # Pick best pass per hand (same logic as extract.py)
    def pick_best(candidates):
        non_empty = [(seq, valid) for seq, valid in candidates if valid]
        if not non_empty:
            return [], []
        return max(non_empty, key=lambda x: len(x[1]))

    l_seq, l_valid = pick_best([
        (rw_l_seq, rw_l_valid), (v_l_seq, v_l_valid), (s_l_seq, s_l_valid)])
    r_seq, r_valid = pick_best([
        (rw_r_seq, rw_r_valid), (v_r_seq, v_r_valid), (s_r_seq, s_r_valid)])
    face_seq, face_valid = pick_best([
        (rw_face_seq, rw_face_valid), (v_face_seq, v_face_valid), (s_face_seq, s_face_valid)])

    if not l_valid and not r_valid:
        print("  ERROR: No hands detected")
        return None

    # Temporal coherence rejection
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)

    if not l_valid and not r_valid:
        print("  ERROR: All detections rejected")
        return None

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)

    # Interpolation
    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
                              l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
                              r_valid, total_frames)
    face_full = interpolate_face(np.array(face_seq) if face_seq else np.zeros((0, 5, 3)),
                                 face_valid, total_frames)

    combined = np.concatenate([l_full, r_full, face_full], axis=1)

    # Temporal resample to 32 frames
    resampled = temporal_resample(combined, cfg.target_frames)

    # 1-Euro filter
    smoothed = one_euro_filter(resampled[:, :, :3])
    resampled[:, :, :3] = smoothed

    # Bone stabilization
    if l_ever:
        resampled = stabilize_bones(resampled, 0, 21)
    if r_ever:
        resampled = stabilize_bones(resampled, 21, 42)

    # Normalize
    normalized = normalize_sequence(resampled, l_ever, r_ever)

    # Per-frame mask
    T = cfg.target_frames
    per_frame_mask = np.zeros((1, T, NUM_NODES, 1), dtype=np.float32)
    if l_valid:
        l_cov = np.interp(np.linspace(0, total_frames - 1, T),
                          sorted(l_valid), np.ones(len(l_valid)))
        for t in range(T):
            per_frame_mask[0, t, 0:21, 0] = l_cov[t]
    if r_valid:
        r_cov = np.interp(np.linspace(0, total_frames - 1, T),
                          sorted(r_valid), np.ones(len(r_valid)))
        for t in range(T):
            per_frame_mask[0, t, 21:42, 0] = r_cov[t]
    if face_valid:
        per_frame_mask[0, :, 42:47, 0] = 1.0

    # Kinematics (Savitzky-Golay)
    final_data = compute_kinematics_batch(
        normalized[np.newaxis, ...], l_ever, r_ever, face_ever,
        per_frame_mask=per_frame_mask
    ).squeeze(0).astype(np.float16)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{label}_{basename}.npy")
    np.save(out_path, final_data)
    print(f"  Saved: {out_path} {final_data.shape} ({final_data.dtype})")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_video.py <video_or_folder> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

    if os.path.isdir(input_path):
        videos = sorted(glob.glob(os.path.join(input_path, '*.mp4')) +
                        glob.glob(os.path.join(input_path, '*.mov')))
        print(f"Found {len(videos)} videos in {input_path}")
        for v in videos:
            extract_single_video(v, output_dir)
    else:
        extract_single_video(input_path, output_dir)

    # List outputs
    npys = sorted(glob.glob(os.path.join(output_dir, '*.npy')))
    print(f"\n{'='*50}")
    print(f"Extracted {len(npys)} .npy files:")
    for f in npys:
        data = np.load(f)
        print(f"  {os.path.basename(f)}: {data.shape} {data.dtype}")


if __name__ == "__main__":
    main()
