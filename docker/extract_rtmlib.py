"""
Extract landmarks from video(s) using RTMW via rtmlib (ONNX).
No mmcv, mmpose, or mmdet required.
Produces .npy files identical to training data format: [32, 47, 10] float16.

Usage:
    python extract_rtmlib.py /app/input/video.mp4
    python extract_rtmlib.py /app/input/           # all .mp4/.mov files
"""
import sys, os, glob, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import cv2

# Import extract.py functions — but mediapipe import is at module level,
# so we patch it out before importing
import importlib
sys.path.insert(0, '/app/src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
# Create a fake mediapipe module so extract.py doesn't crash on import
import types
_fake_mp = types.ModuleType('mediapipe')
_fake_mp.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake_mp
sys.modules['mediapipe.solutions'] = _fake_mp.solutions

from extract import (
    PipelineConfig, NUM_NODES,
    interpolate_hand, interpolate_face, temporal_resample,
    normalize_sequence, compute_kinematics_batch,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)

# COCO-WholeBody 133 keypoint indices
_LHAND_START = 91
_RHAND_START = 112
# Face: 68-point face starts at index 23
_FACE_NOSE = 23 + 30
_FACE_CHIN = 23 + 8
_FACE_FOREHEAD = 23 + 27
_FACE_LEFT_EAR = 23 + 0
_FACE_RIGHT_EAR = 23 + 16
_FACE_INDICES = [_FACE_NOSE, _FACE_CHIN, _FACE_FOREHEAD, _FACE_LEFT_EAR, _FACE_RIGHT_EAR]

HAND_CONF_THRESHOLD = 0.25
FACE_CONF_THRESHOLD = 0.25


def extract_single_video(video_path, output_dir, wholebody, label=None):
    """Extract landmarks from a single video using RTMW via rtmlib."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    if label is None:
        label = basename.replace(' ', '_')

    print(f"\n[EXTRACT] {os.path.basename(video_path)}")

    cfg = PipelineConfig()

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

    # Run RTMW via rtmlib on each frame
    l_seq, r_seq, face_seq = [], [], []
    l_valid, r_valid, face_valid = [], [], []

    print("  Running RTMW-l (rtmlib/ONNX)...")
    for i, rgb in enumerate(frames_rgb):
        fidx = frame_indices[i]
        h, w = rgb.shape[:2]

        # rtmlib expects BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        keypoints, scores = wholebody(bgr)

        if keypoints is None or len(keypoints) == 0:
            continue

        # keypoints shape: [N_people, 133, 2], scores: [N_people, 133]
        # Pick person with highest mean score
        if keypoints.ndim == 3:
            person_idx = scores.mean(axis=1).argmax()
            kps = keypoints[person_idx]  # [133, 2]
            scs = scores[person_idx]     # [133]
        else:
            kps = keypoints  # [133, 2]
            scs = scores     # [133]

        if len(kps) < 133:
            continue

        # Extract left hand (91-111)
        l_hand_kps = kps[_LHAND_START:_LHAND_START + 21]
        l_hand_scores = scs[_LHAND_START:_LHAND_START + 21]
        l_mean_conf = l_hand_scores.mean()

        if l_mean_conf >= HAND_CONF_THRESHOLD:
            coords = [[kp[0] / w, kp[1] / h, 0.0] for kp in l_hand_kps]
            if not l_valid or l_valid[-1] != fidx:
                l_seq.append(coords)
                l_valid.append(fidx)

        # Extract right hand (112-132)
        r_hand_kps = kps[_RHAND_START:_RHAND_START + 21]
        r_hand_scores = scs[_RHAND_START:_RHAND_START + 21]
        r_mean_conf = r_hand_scores.mean()

        if r_mean_conf >= HAND_CONF_THRESHOLD:
            coords = [[kp[0] / w, kp[1] / h, 0.0] for kp in r_hand_kps]
            if not r_valid or r_valid[-1] != fidx:
                r_seq.append(coords)
                r_valid.append(fidx)

        # Extract face (5 landmarks from 68-point face)
        face_coords = []
        face_ok = True
        for face_idx in _FACE_INDICES:
            if face_idx < len(kps) and scs[face_idx] >= FACE_CONF_THRESHOLD:
                face_coords.append([kps[face_idx][0] / w, kps[face_idx][1] / h, 0.0])
            else:
                face_ok = False
                break

        if face_ok and len(face_coords) == 5:
            if not face_valid or face_valid[-1] != fidx:
                face_seq.append(face_coords)
                face_valid.append(fidx)

    rw_coverage = max(len(l_valid), len(r_valid)) / max(processed_count, 1)
    print(f"  RTMW: L={len(l_valid)} R={len(r_valid)} Face={len(face_valid)} ({rw_coverage:.0%})")

    del frames_rgb

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

    combined = np.concatenate([l_full, r_full, face_full], axis=1)  # [T_raw, 47, 3]

    def _process_segment(seg, l_ev, r_ev, f_ev):
        """Process a single segment: filter, stabilize, normalize, kinematics → [32, 47, 10].
        Each segment processed independently — matches how training data was created
        (each sign was extracted and processed as an individual file)."""
        seg32 = temporal_resample(seg, 32)
        smoothed = one_euro_filter(seg32[:, :, :3])
        seg32[:, :, :3] = smoothed
        if l_ev:
            seg32 = stabilize_bones(seg32, 0, 21)
        if r_ev:
            seg32 = stabilize_bones(seg32, 21, 42)
        normed = normalize_sequence(seg32, l_ev, r_ev)
        mask = np.zeros((1, 32, NUM_NODES, 1), dtype=np.float32)
        if l_ev: mask[0, :, 0:21, 0] = 1.0
        if r_ev: mask[0, :, 21:42, 0] = 1.0
        if f_ev: mask[0, :, 42:47, 0] = 1.0
        return compute_kinematics_batch(
            normed[np.newaxis, ...], l_ev, r_ev, f_ev, per_frame_mask=mask
        ).squeeze(0).astype(np.float16)

    os.makedirs(output_dir, exist_ok=True)

    # --- Output 1: Isolated sign [32, 47, 10] (whole video → 32 frames) ---
    isolated = _process_segment(combined, l_ever, r_ever, face_ever)
    iso_path = os.path.join(output_dir, f"{label}_{basename}.npy")
    np.save(iso_path, isolated)
    print(f"  Saved isolated: {iso_path} {isolated.shape}")

    # --- Output 1b: Raw XYZ for sliding window [T_raw, 47, 3] ---
    # Save raw interpolated XYZ so inference can slide Stage 1 across it,
    # processing each window independently (matching training data format).
    raw_frames = combined.shape[0]
    if raw_frames > 32:
        raw_path = os.path.join(output_dir, f"{label}_{basename}_raw.npy")
        raw_meta = {
            'l_ever': l_ever, 'r_ever': r_ever, 'face_ever': face_ever,
            'total_frames': raw_frames
        }
        np.save(raw_path, combined.astype(np.float16))
        # Save metadata alongside
        import json
        with open(raw_path.replace('.npy', '_meta.json'), 'w') as f:
            json.dump(raw_meta, f)
        print(f"  Saved raw: {raw_path} {combined.shape} (for sliding window)")

    # --- Output 2+: Multi-hypothesis [N*32, 47, 10] ---
    # Split raw XYZ into N segments at low-motion boundaries,
    # process each segment independently (matches training: each sign = separate file)
    raw_frames = combined.shape[0]
    max_signs = min(4, max(1, raw_frames // 12))

    if max_signs >= 2:
        xyz = combined[:, :42, :3]
        diff = xyz[1:] - xyz[:-1]
        energy = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=1)
        k = min(7, max(3, raw_frames // 6))
        smoothed_energy = np.convolve(energy, np.ones(k) / k, mode='same')

        for n_signs in range(2, max_signs + 1):
            min_seg = max(8, raw_frames // (n_signs + 2))
            margin = max(min_seg, int(raw_frames * 0.10))

            splits = []
            search_mask = np.zeros_like(smoothed_energy, dtype=bool)
            search_mask[:margin] = True
            search_mask[-margin:] = True

            for _ in range(n_signs - 1):
                masked = smoothed_energy.copy()
                masked[search_mask] = 999
                if masked.min() >= 999:
                    break
                best = np.argmin(masked)
                splits.append(best)
                lo = max(0, best - min_seg)
                hi = min(len(search_mask), best + min_seg)
                search_mask[lo:hi] = True

            if len(splits) < n_signs - 1:
                splits = [int(raw_frames * i / n_signs) for i in range(1, n_signs)]

            splits.sort()
            boundaries = [0] + splits + [raw_frames]

            segments = []
            for i in range(len(boundaries) - 1):
                seg = combined[boundaries[i]:boundaries[i+1]]
                if seg.shape[0] >= 4:
                    segments.append(_process_segment(seg, l_ever, r_ever, face_ever))

            if len(segments) >= 2:
                continuous = np.concatenate(segments, axis=0)
                cont_path = os.path.join(output_dir, f"{label}_{basename}_n{n_signs}.npy")
                np.save(cont_path, continuous)
                print(f"  Saved n={n_signs}: {cont_path} {continuous.shape} ({len(segments)} segments)")

    return iso_path


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_rtmlib.py <video_or_folder> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

    # Initialize rtmlib RTMW
    from rtmlib import Wholebody
    print("Initializing RTMW-l (rtmlib/ONNX, CPU)...")
    wholebody = Wholebody(
        to_openpose=False,
        mode='performance',  # uses RTMW-l 384x288
        backend='onnxruntime',
        device='cpu',
    )
    print("  Ready.")

    if os.path.isdir(input_path):
        videos = sorted(glob.glob(os.path.join(input_path, '*.mp4')) +
                        glob.glob(os.path.join(input_path, '*.mov')) +
                        glob.glob(os.path.join(input_path, '*.avi')))
        print(f"Found {len(videos)} videos in {input_path}")
        for v in videos:
            extract_single_video(v, output_dir, wholebody)
    else:
        extract_single_video(input_path, output_dir, wholebody)

    # List outputs
    npys = sorted(glob.glob(os.path.join(output_dir, '*.npy')))
    print(f"\n{'='*50}")
    print(f"Extracted {len(npys)} .npy files:")
    for f in npys:
        data = np.load(f)
        print(f"  {os.path.basename(f)}: {data.shape} {data.dtype}")


if __name__ == "__main__":
    main()
