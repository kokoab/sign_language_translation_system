"""
SLT 47-Point Extraction v7.0 (Aggressive Detection Edition)
42 Hand Landmarks + 5 Face Reference Points = 47 Total

v7.0 changes (from v6.x):
- Resolution bump: 384px -> 512px (more pixels for MediaPipe palm detection)
- CLAHE preprocessing: adaptive contrast enhancement before detection
- Adaptive frame skip: every frame for short videos (<80 frames), 2x oversampling for longer
- Two-pass detection: video mode + static mode, merge best per component (L hand, R hand, face)
- min_raw_frames=8, max_missing_ratio=0.40 (tightened back — v7.0 detection improvements make relaxation unnecessary)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import logging
import hashlib
import json
import concurrent.futures
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-Fast")

# ─────────────────────────────────────────────
# Label Aliases — merge synonym classes
# ─────────────────────────────────────────────
LABEL_ALIASES = {
    # Synonym collapse: map individual + composite folder names to one canonical label
    "DRIVE": "DRIVE_CAR",
    "CAR": "DRIVE_CAR",
    "HARD": "HARD_DIFFICULT",
    "DIFFICULT": "HARD_DIFFICULT",
    "MAKE": "MAKE_CREATE",
    "CREATE": "MAKE_CREATE",
    "EAT": "EAT_FOOD",
    "FOOD": "EAT_FOOD",
    # Composite folders → canonical single label (Issue #1 fix)
    "ALSO_SAME": "ALSO",
    "SAME": "ALSO",
    "MARKET_STORE": "STORE",
    "MARKET": "STORE",
    "US_WE": "WE",
    "US": "WE",
    "FEW_SEVERAL": "FEW",
    "SEVERAL": "FEW",
    "I_ME": "I",
    "ME": "I",
    "HE_SHE": "HE",
    "SHE": "HE",
}

# ─────────────────────────────────────────────
# Node Index Reference (47 Points Total)
# 0-20: Left Hand (21 points)
# 21-41: Right Hand (21 points)
# 42: Nose, 43: Chin, 44: Forehead, 45: Left Ear, 46: Right Ear
# ─────────────────────────────────────────────
NUM_NODES = 47
L_WRIST = 0; R_WRIST = 21
L_MIDDLE_MCP = 9; R_MIDDLE_MCP = 30

# Face landmark indices in MediaPipe FaceMesh
FACE_NOSE_IDX = 1
FACE_CHIN_IDX = 152
FACE_FOREHEAD_IDX = 10
FACE_LEFT_EAR_IDX = 234
FACE_RIGHT_EAR_IDX = 454
FACE_LANDMARK_INDICES = [FACE_NOSE_IDX, FACE_CHIN_IDX, FACE_FOREHEAD_IDX,
                          FACE_LEFT_EAR_IDX, FACE_RIGHT_EAR_IDX]

# Our node indices for face points
NOSE_NODE = 42
CHIN_NODE = 43
FOREHEAD_NODE = 44
L_EAR_NODE = 45
R_EAR_NODE = 46

@dataclass
class PipelineConfig:
    target_frames: int = 32
    min_raw_frames: int = 8
    max_missing_ratio: float = 0.40
    min_detection_conf: float = 0.80
    min_tracking_conf: float = 0.80
    model_complexity: int = 1
    max_num_hands: int = 2
    mirror_handedness: bool = False
    # Augmentation settings
    enable_augmentation: bool = False  # Disabled: Handled online in training scripts to save disk
    augment_probability: float = 0.5   # Probability of each augmentation type

    raw_video_dir: str = "data/raw_videos/ASL VIDEOS"
    output_dir: str = "ASL_landmarks_float16"
    seed: int = 42

CFG = PipelineConfig()

# ─────────────────────────────────────────────
# Core Math & Normalization Helpers (NumPy Native)
# ─────────────────────────────────────────────
def interpolate_hand(hand_seq: np.ndarray, valid_indices: list, total_frames: int) -> np.ndarray:
    if not valid_indices: return np.zeros((total_frames, 21, 3), dtype=np.float32)
    flat = hand_seq.reshape(len(hand_seq), -1)
    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, 21, 3).astype(np.float32)

    x_new = np.arange(total_frames, dtype=np.float64)
    xp = np.array(valid_indices, dtype=np.float64)
    result = np.column_stack([
        np.interp(x_new, xp, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(total_frames, 21, 3).astype(np.float32)


def interpolate_face(face_seq: np.ndarray, valid_indices: list, total_frames: int) -> np.ndarray:
    """Interpolate 5 face landmarks across frames."""
    if not valid_indices: return np.zeros((total_frames, 5, 3), dtype=np.float32)
    flat = face_seq.reshape(len(face_seq), -1)
    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, 5, 3).astype(np.float32)

    x_new = np.arange(total_frames, dtype=np.float64)
    xp = np.array(valid_indices, dtype=np.float64)
    result = np.column_stack([
        np.interp(x_new, xp, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(total_frames, 5, 3).astype(np.float32)


def temporal_resample(seq: np.ndarray, target_frames: int) -> np.ndarray:
    N, P, C = seq.shape
    if N == target_frames: return seq
    flat = seq.reshape(N, -1)
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, target_frames)
    result = np.column_stack([
        np.interp(x_new, x_old, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(target_frames, P, C).astype(np.float32)

def normalize_sequence(seq: np.ndarray, l_ever: bool, r_ever: bool) -> np.ndarray:
    """Normalize 47-point sequence. Face landmarks are normalized to the same center as hands."""
    norm_seq = seq.copy().astype(np.float64)
    valid_wrists = []
    if l_ever: valid_wrists.append(norm_seq[:, L_WRIST, :])
    if r_ever: valid_wrists.append(norm_seq[:, R_WRIST, :])

    if valid_wrists:
        all_wrists = np.concatenate(valid_wrists, axis=0)
        nonzero = all_wrists[np.linalg.norm(all_wrists, axis=-1) > 1e-6]
        center = np.median(nonzero, axis=0) if len(nonzero) > 0 else np.zeros(3)
    else: center = np.zeros(3)

    if l_ever: norm_seq[:, 0:21] -= center
    if r_ever: norm_seq[:, 21:42] -= center
    # Face landmarks normalized to same center for relative positioning
    norm_seq[:, 42:47] -= center

    bone_lengths = []
    if l_ever: bone_lengths.extend(np.linalg.norm(norm_seq[:, L_MIDDLE_MCP] - norm_seq[:, L_WRIST], axis=-1))
    if r_ever: bone_lengths.extend(np.linalg.norm(norm_seq[:, R_MIDDLE_MCP] - norm_seq[:, R_WRIST], axis=-1))

    if bone_lengths:
        filtered = [b for b in bone_lengths if b > 1e-6]
        if filtered: norm_seq /= (np.median(filtered) + 1e-8)

    return norm_seq.astype(np.float32)

def compute_kinematics_batch(seqs: np.ndarray, l_ever: bool, r_ever: bool, face_ever: bool = False) -> np.ndarray:
    """Compute kinematics for 47-point sequences."""
    B, F, P, _ = seqs.shape

    vel = np.zeros_like(seqs)
    vel[:, 1:-1] = (seqs[:, 2:] - seqs[:, :-2]) / 2.0
    vel[:, 0] = vel[:, 1]; vel[:, -1] = vel[:, -2]

    acc = np.zeros_like(seqs)
    acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / 2.0
    acc[:, 0] = acc[:, 1]; acc[:, -1] = acc[:, -2]

    mask = np.zeros((B, F, P, 1), dtype=np.float32)
    if l_ever: mask[:, :, 0:21, 0] = 1.0
    if r_ever: mask[:, :, 21:42, 0] = 1.0
    if face_ever: mask[:, :, 42:47, 0] = 1.0

    return np.concatenate([seqs, vel, acc, mask], axis=-1).astype(np.float32)

# ─────────────────────────────────────────────
# Augmentation Helpers for Multi-Pass Extraction
# ─────────────────────────────────────────────
import random

def temporal_speed_warp(xyz_seq: np.ndarray, speed_factor: float, target_frames: int = 32) -> np.ndarray:
    """Apply speed warp to XYZ sequence. speed_factor > 1 = faster, < 1 = slower."""
    T, P, C = xyz_seq.shape
    new_len = int(T / speed_factor)
    new_len = max(8, min(new_len, T * 2))

    flat = xyz_seq.reshape(T, -1)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_len)
    warped = np.column_stack([
        np.interp(x_new, x_old, flat[:, c]) for c in range(flat.shape[1])
    ]).reshape(new_len, P, C)

    return temporal_resample(warped, target_frames)


def mirror_hands_xyz(xyz_seq: np.ndarray) -> np.ndarray:
    """Flip left/right and mirror X coordinate for 47-point sequence."""
    mirrored = xyz_seq.copy()
    # Flip X coordinate (horizontal mirror)
    mirrored[:, :, 0] = -mirrored[:, :, 0]
    # Swap left (0:21) and right (21:42) hands
    left = mirrored[:, :21, :].copy()
    right = mirrored[:, 21:42, :].copy()
    mirrored[:, :21, :] = right
    mirrored[:, 21:42, :] = left
    # Face landmarks: X is already flipped, swap ear nodes
    l_ear = mirrored[:, L_EAR_NODE, :].copy()
    r_ear = mirrored[:, R_EAR_NODE, :].copy()
    mirrored[:, L_EAR_NODE, :] = r_ear
    mirrored[:, R_EAR_NODE, :] = l_ear
    return mirrored


def recompute_full_features(xyz_seq: np.ndarray, l_ever: bool, r_ever: bool, face_ever: bool = False) -> np.ndarray:
    """Recompute normalization and kinematics from augmented XYZ data."""
    normalized = normalize_sequence(xyz_seq, l_ever, r_ever)
    final_data = compute_kinematics_batch(normalized[np.newaxis, ...], l_ever, r_ever, face_ever)
    return final_data.squeeze(0).astype(np.float16)

# ─────────────────────────────────────────────
# Two-Pass Detection Helper
# ─────────────────────────────────────────────
def _detect_pass(frames_rgb, frame_indices, hands_detector, face_detector, cfg):
    """Run hand + face detection on preprocessed RGB frames.
    Returns (l_seq, r_seq, face_seq, l_valid, r_valid, face_valid).
    """
    l_seq, r_seq, l_valid, r_valid = [], [], [], []
    face_seq, face_valid = [], []

    for rgb, fidx in zip(frames_rgb, frame_indices):
        hand_res = hands_detector.process(rgb)
        if hand_res.multi_hand_landmarks:
            for hand_lm, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                h_label = handedness.classification[0].label
                if cfg.mirror_handedness:
                    h_label = "Right" if h_label == "Left" else "Left"
                if handedness.classification[0].score >= cfg.min_detection_conf:
                    coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                    if h_label == "Left" and (not l_valid or l_valid[-1] != fidx):
                        l_seq.append(coords); l_valid.append(fidx)
                    elif h_label == "Right" and (not r_valid or r_valid[-1] != fidx):
                        r_seq.append(coords); r_valid.append(fidx)

        face_res = face_detector.process(rgb)
        if face_res.multi_face_landmarks:
            face_lm = face_res.multi_face_landmarks[0]
            face_coords = []
            for idx in FACE_LANDMARK_INDICES:
                lm = face_lm.landmark[idx]
                face_coords.append([lm.x, lm.y, lm.z])
            face_seq.append(face_coords)
            face_valid.append(fidx)

    return l_seq, r_seq, face_seq, l_valid, r_valid, face_valid

# ─────────────────────────────────────────────
# Worker Process (v7.0: 512px + CLAHE + Adaptive Skip + Two-Pass)
# ─────────────────────────────────────────────
def process_single_video(task_info):
    try:
        root, video_name, label, cfg, _ = task_info
        out_path = Path(cfg.output_dir)
        video_path = os.path.join(root, video_name)

        cap = cv2.VideoCapture(video_path)
        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FAST-FAIL CHECK
        if 0 < total_est < cfg.min_raw_frames:
            cap.release()
            return 0, label, False, None

        # Adaptive skip: every frame for short videos, 2x oversampling for longer
        if total_est < 80:
            skip = 1
        else:
            skip = max(1, total_est // 64)

        # ── Phase 1: Read and preprocess all selected frames ──
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frames_rgb = []
        frame_indices = []
        frame_idx = 0

        while cap.isOpened():
            ret = cap.grab()
            if not ret: break

            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            ret, frame = cap.retrieve()
            if not ret: break

            h, w = frame.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # CLAHE contrast enhancement (L channel only — no color shift)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            frames_rgb.append(rgb)
            frame_indices.append(frame_idx)
            frame_idx += 1

        cap.release()
        total_frame_idx = frame_idx
        processed_count = len(frames_rgb)

        if processed_count < cfg.min_raw_frames:
            return 0, label, False, None

        # ── Phase 2: Two-pass detection ──
        # Pass 1 — Video mode (temporal tracking, good for smooth sequences)
        hands_v = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
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
        hands_v.close(); face_v.close()

        # Pass 2 — Static mode (independent per-frame, recovers when tracking fails)
        hands_s = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=cfg.max_num_hands,
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
        hands_s.close(); face_s.close()

        del frames_rgb  # Free frame buffer early to reduce memory pressure

        # ── Phase 3: Merge — pick pass with more detections per component ──
        if len(v_l_valid) >= len(s_l_valid):
            l_seq, l_valid = v_l_seq, v_l_valid
        else:
            l_seq, l_valid = s_l_seq, s_l_valid

        if len(v_r_valid) >= len(s_r_valid):
            r_seq, r_valid = v_r_seq, v_r_valid
        else:
            r_seq, r_valid = s_r_seq, s_r_valid

        if len(v_face_valid) >= len(s_face_valid):
            face_seq, face_valid = v_face_seq, v_face_valid
        else:
            face_seq, face_valid = s_face_seq, s_face_valid

        if not l_valid and not r_valid:
            return 0, label, False, None

        max_detected = max(len(l_valid), len(r_valid))
        if 1.0 - (max_detected / processed_count) > cfg.max_missing_ratio:
            return 0, label, False, None

        # ── Phase 4: Interpolation, normalization, kinematics (unchanged) ──
        l_full = interpolate_hand(np.array(l_seq), l_valid, total_frame_idx)
        r_full = interpolate_hand(np.array(r_seq), r_valid, total_frame_idx)
        face_full = interpolate_face(np.array(face_seq) if face_seq else np.zeros((0, 5, 3)),
                                      face_valid, total_frame_idx)

        combined = np.concatenate([l_full, r_full, face_full], axis=1)
        l_ever, r_ever = bool(l_valid), bool(r_valid)
        face_ever = bool(face_valid)

        resampled = temporal_resample(combined, cfg.target_frames)
        normalized = normalize_sequence(resampled, l_ever, r_ever)

        final_data = compute_kinematics_batch(normalized[np.newaxis, ...], l_ever, r_ever, face_ever)
        final_data = final_data.squeeze(0).astype(np.float16)

        file_hash = hashlib.md5(video_name.encode()).hexdigest()[:6]
        save_name = f"{label}_{Path(video_name).stem}_{file_hash}.npy"
        np.save(out_path / save_name, final_data)
        saved_count = 1

        # Multi-pass augmentation
        if cfg.enable_augmentation:
            xyz_base = resampled[:, :, :3]

            if random.random() < cfg.augment_probability:
                speed_factor = random.uniform(1.15, 1.25)
                aug_xyz = temporal_speed_warp(xyz_base, speed_factor, cfg.target_frames)
                aug_data = recompute_full_features(aug_xyz, l_ever, r_ever, face_ever)
                aug_name = f"{label}_{Path(video_name).stem}_fast_{file_hash}.npy"
                np.save(out_path / aug_name, aug_data)
                saved_count += 1

            if random.random() < cfg.augment_probability:
                speed_factor = random.uniform(0.75, 0.85)
                aug_xyz = temporal_speed_warp(xyz_base, speed_factor, cfg.target_frames)
                aug_data = recompute_full_features(aug_xyz, l_ever, r_ever, face_ever)
                aug_name = f"{label}_{Path(video_name).stem}_slow_{file_hash}.npy"
                np.save(out_path / aug_name, aug_data)
                saved_count += 1

            if random.random() < cfg.augment_probability:
                aug_xyz = mirror_hands_xyz(xyz_base)
                aug_data = recompute_full_features(aug_xyz, r_ever, l_ever, face_ever)
                aug_name = f"{label}_{Path(video_name).stem}_mirror_{file_hash}.npy"
                np.save(out_path / aug_name, aug_data)
                saved_count += 1

        # Diagnostics: which pass won, detection coverage
        diag = {
            'l_pass': 'video' if len(v_l_valid) >= len(s_l_valid) else 'static',
            'r_pass': 'video' if len(v_r_valid) >= len(s_r_valid) else 'static',
            'face_pass': 'video' if len(v_face_valid) >= len(s_face_valid) else 'static',
            'face_detected': face_ever,
            'detection_coverage': max(len(l_valid), len(r_valid)) / max(processed_count, 1),
        }

        return saved_count, label, True, diag

    except Exception as e:
        log.error(f"Failed processing {video_name}: {e}")
        return 0, task_info[2], False, None

def run_pipeline():
    out_path = Path(CFG.output_dir); out_path.mkdir(parents=True, exist_ok=True)

    # Track done files (base versions only)
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
        # Apply label aliases (merge synonym classes)
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
        # Still write manifest for existing files
        _write_manifest(out_path)
        return

    # Sort largest files first
    all_videos.sort(key=lambda x: x[4], reverse=True)

    safe_workers = max(1, min(multiprocessing.cpu_count() - 2, 9))
    chunk_size = 50

    log.info(f"Found {len(all_videos)} remaining videos. Processing with {safe_workers} workers (Chunksize: {chunk_size})...")

    # Per-class success/fail tracking + diagnostics
    class_success = Counter()
    class_fail = Counter()
    # Two-pass diagnostics: per-class pass winners
    class_l_video = Counter()   # times video mode won left hand
    class_l_static = Counter()  # times static mode won left hand
    class_r_video = Counter()
    class_r_static = Counter()
    class_face_video = Counter()
    class_face_static = Counter()
    class_face_detected = Counter()  # clips where face was detected
    class_coverage_sum = defaultdict(float)  # sum of detection coverage per class

    saved_count = 0
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        for result in executor.map(process_single_video, all_videos, chunksize=chunk_size):
            count, label, success, diag = result
            saved_count += count
            if success:
                class_success[label] += 1
                if diag:
                    if diag['l_pass'] == 'video': class_l_video[label] += 1
                    else: class_l_static[label] += 1
                    if diag['r_pass'] == 'video': class_r_video[label] += 1
                    else: class_r_static[label] += 1
                    if diag['face_pass'] == 'video': class_face_video[label] += 1
                    else: class_face_static[label] += 1
                    if diag['face_detected']: class_face_detected[label] += 1
                    class_coverage_sum[label] += diag['detection_coverage']
            else:
                class_fail[label] += 1
            if saved_count > 0 and saved_count % 100 == 0:
                print(f"  -> Saved {saved_count} sequences...", end="\r")

    elapsed = time.time() - t_start
    log.info(f"\nDATASET COMPLETE. Successfully processed {saved_count} new videos in {elapsed/60:.1f} min ({elapsed/max(saved_count,1):.2f} s/video).")

    # Log per-class extraction rates
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

    # Two-pass diagnostics summary
    total_success = sum(class_success.values())
    total_l_static = sum(class_l_static.values())
    total_r_static = sum(class_r_static.values())
    total_face_det = sum(class_face_detected.values())
    if total_success > 0:
        log.info(f"Two-pass stats ({total_success} clips):")
        log.info(f"  L-hand: static won {total_l_static}/{total_success} ({100*total_l_static/total_success:.0f}%)")
        log.info(f"  R-hand: static won {total_r_static}/{total_success} ({100*total_r_static/total_success:.0f}%)")
        log.info(f"  Face detected: {total_face_det}/{total_success} ({100*total_face_det/total_success:.0f}%)")

    # Save extraction_stats.json
    stats = {}
    for label in all_labels:
        s = class_success[label]
        f = class_fail[label]
        stats[label] = {
            'success': s, 'fail': f,
            'fail_rate': round(f / max(s + f, 1), 3),
            'l_video_wins': class_l_video[label], 'l_static_wins': class_l_static[label],
            'r_video_wins': class_r_video[label], 'r_static_wins': class_r_static[label],
            'face_video_wins': class_face_video[label], 'face_static_wins': class_face_static[label],
            'face_detected': class_face_detected[label],
            'avg_detection_coverage': round(class_coverage_sum[label] / max(s, 1), 3),
        }
    stats_path = out_path / 'extraction_stats.json'
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp, indent=2)
    log.info(f"Extraction stats saved: {stats_path}")

    # Write manifest
    _write_manifest(out_path)


def _write_manifest(out_path: Path):
    """Write manifest.json mapping filenames to labels by matching against raw video directory."""
    manifest = _build_manifest_from_raw(out_path)

    manifest_path = out_path / 'manifest.json'
    with open(manifest_path, 'w') as fp:
        json.dump(manifest, fp, indent=2)
    log.info(f"Manifest written: {len(manifest)} entries -> {manifest_path}")


def _build_manifest_from_raw(out_path: Path) -> dict:
    """Build manifest by matching output files to raw video folders."""
    manifest = {}
    existing_files = set(f for f in os.listdir(out_path) if f.endswith('.npy'))

    for root, _, files in os.walk(CFG.raw_video_dir):
        label = Path(root).name
        label = LABEL_ALIASES.get(label, label)
        for f in files:
            if not f.lower().endswith(('.mp4', '.mov')):
                continue
            file_hash = hashlib.md5(f.encode()).hexdigest()[:6]
            stem = Path(f).stem

            # Check all possible filenames (base + augmentations)
            base_name = f"{label}_{stem}_{file_hash}.npy"
            if base_name in existing_files:
                manifest[base_name] = label

            for aug_suf in ['fast', 'slow', 'mirror']:
                aug_name = f"{label}_{stem}_{aug_suf}_{file_hash}.npy"
                if aug_name in existing_files:
                    manifest[aug_name] = label

    return manifest


if __name__ == "__main__":
    run_pipeline()
