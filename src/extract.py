"""
SLT 47-Point Extraction v9.0 (Pure RTMW-l + Multi-GPU Edition)
42 Hand Landmarks + 5 Face Reference Points = 47 Total

v9.0 changes (from v8.x):
- Pure RTMW-l 384x288 as primary extractor (66.3% hand AP vs RTMPose-m's 47.5%)
  Eliminates silent bad detections where RTMPose returned wrong landmarks with high confidence
- Multi-GPU support: auto-detects all available GPUs, round-robin worker assignment
- GPU cascade simplified: RTMW-l → MediaPipe video → MediaPipe static
- CPU cascade unchanged: MediaPipe video → static → Tasks API
- Multi-person filtering: picks largest person by body bbox
- Full preprocessing pipeline: adaptive gamma → CLAHE → bilateral denoise → unsharp mask
- 1-Euro adaptive filter (preserves fast signing, removes jitter)
- Bone length stabilization (median normalization per sequence)
- Temporal coherence rejection (removes false detection jumps before interpolation)
- Confidence-weighted pass merge (spatial coherence tiebreak when counts are close)
- Relaxed thresholds: detection=0.70, tracking=0.65, max_missing=0.50
"""

import os
# ── Limit per-worker threading BEFORE importing any library ──
# Without this, each worker spawns 10-15 internal threads (OpenCV, MediaPipe, NumPy/MKL).
# 50 workers × 15 threads = 750 threads → OS thread exhaustion → crash.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
cv2.setNumThreads(0)  # Disable OpenCV's internal thread pool

import mediapipe as mp
import numpy as np
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
    max_missing_ratio: float = 0.50
    min_detection_conf: float = 0.70
    min_tracking_conf: float = 0.65
    model_complexity: int = 1
    max_num_hands: int = 2
    mirror_handedness: bool = False
    prefer_rtmw: bool = True  # RTMW-l first on GPU; auto-falls back to MediaPipe-first on CPU
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


def one_euro_filter(x, t_e=1.0/30, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
    """1-Euro adaptive low-pass filter. x shape: [T, N, C].
    Heavy smoothing for slow/still (removes jitter), light for fast motion (preserves signing).
    """
    def _alpha(t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    y = x.copy()
    dx = np.zeros_like(x)
    for i in range(1, len(x)):
        a_d = _alpha(t_e, d_cutoff)
        dx[i] = a_d * (x[i] - y[i-1]) / t_e + (1 - a_d) * dx[i-1]
        cutoff = min_cutoff + beta * np.abs(dx[i])
        a = _alpha(t_e, cutoff)
        y[i] = a * x[i] + (1 - a) * y[i-1]
    return y.astype(np.float32)


HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
]

def stabilize_bones(xyz, hand_start, hand_end):
    """Normalize bone lengths to median across sequence for one hand.
    xyz shape: [T, 47, 3]. Modifies in-place."""
    for parent, child in HAND_BONES:
        p, c = hand_start + parent, hand_start + child
        vecs = xyz[:, c] - xyz[:, p]
        lengths = np.linalg.norm(vecs, axis=-1, keepdims=True)
        valid = lengths.squeeze() > 1e-6
        if valid.sum() < 3:
            continue
        median_len = np.median(lengths[valid])
        unit = vecs / (lengths + 1e-8)
        xyz[:, c] = xyz[:, p] + unit * median_len
    return xyz


def reject_temporal_outliers(hand_seq, valid_indices, jump_threshold=0.15):
    """Remove frames where landmarks jump unrealistically between consecutive detections.
    jump_threshold: max allowed wrist displacement as fraction of [0,1] coord space.
    Returns cleaned (hand_seq, valid_indices).
    """
    if len(valid_indices) < 3:
        return hand_seq, valid_indices
    arr = np.array(hand_seq)  # [N, 21, 3]
    # Use wrist (index 0) as proxy for whole hand movement
    wrist = arr[:, 0, :2]  # XY only
    keep = [0]
    for i in range(1, len(wrist)):
        jump = np.linalg.norm(wrist[i] - wrist[keep[-1]])
        if jump < jump_threshold:
            keep.append(i)
    if len(keep) < 2:
        return hand_seq, valid_indices
    return [hand_seq[i] for i in keep], [valid_indices[i] for i in keep]


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

def compute_kinematics_batch(seqs: np.ndarray, l_ever: bool, r_ever: bool, face_ever: bool = False,
                              per_frame_mask: np.ndarray = None) -> np.ndarray:
    """Compute kinematics for 47-point sequences.
    Uses Savitzky-Golay filter for smoother velocity/acceleration when available,
    falls back to central difference otherwise.
    """
    B, F, P, _ = seqs.shape

    try:
        from scipy.signal import savgol_filter
        # Savitzky-Golay: smooth derivative, window=5 frames, poly=2
        # Much less noisy than central difference on jittery landmarks
        vel = np.zeros_like(seqs)
        acc = np.zeros_like(seqs)
        for b in range(B):
            for p in range(P):
                for c in range(3):
                    vel[b, :, p, c] = savgol_filter(seqs[b, :, p, c], window_length=min(7, F if F % 2 == 1 else F - 1), polyorder=2, deriv=1)
                    acc[b, :, p, c] = savgol_filter(seqs[b, :, p, c], window_length=min(7, F if F % 2 == 1 else F - 1), polyorder=2, deriv=2)
    except ImportError:
        # Fallback: central difference
        vel = np.zeros_like(seqs)
        vel[:, 1:-1] = (seqs[:, 2:] - seqs[:, :-2]) / 2.0
        vel[:, 0] = vel[:, 1]; vel[:, -1] = vel[:, -2]

        acc = np.zeros_like(seqs)
        acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / 2.0
        acc[:, 0] = acc[:, 1]; acc[:, -1] = acc[:, -2]

    # Per-frame mask if available, otherwise binary
    if per_frame_mask is not None:
        mask = per_frame_mask
    else:
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
# Tasks API Detection (better hand detection for fists, back-of-hand, occlusion)
# ─────────────────────────────────────────────
_TASKS_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'hand_landmarker.task')

def _detect_pass_tasks(frames_rgb, frame_indices, cfg):
    """Run hand detection using the MediaPipe Tasks API (HandLandmarker).
    Better at detecting fists, back-of-hand, and partially occluded hands.
    Returns (l_seq, r_seq, l_valid, r_valid) — no face detection (handled separately).
    """
    try:
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        from mediapipe.tasks.python import BaseOptions
        import mediapipe as _mp

        model_path = _TASKS_MODEL_PATH
        if not os.path.exists(model_path):
            return [], [], [], []

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=max(0.5, cfg.min_detection_conf - 0.15),  # more aggressive
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=cfg.min_tracking_conf,
        )
        landmarker = HandLandmarker.create_from_options(options)

        l_seq, r_seq, l_valid, r_valid = [], [], [], []

        for rgb, fidx in zip(frames_rgb, frame_indices):
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks and result.handedness:
                for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                    h_label = handedness[0].category_name
                    h_score = handedness[0].score
                    if cfg.mirror_handedness:
                        h_label = "Right" if h_label == "Left" else "Left"
                    if h_score >= max(0.5, cfg.min_detection_conf - 0.15):
                        coords = [[lm.x, lm.y, lm.z] for lm in hand_lm]
                        if h_label == "Left" and (not l_valid or l_valid[-1] != fidx):
                            l_seq.append(coords); l_valid.append(fidx)
                        elif h_label == "Right" and (not r_valid or r_valid[-1] != fidx):
                            r_seq.append(coords); r_valid.append(fidx)

        landmarker.close()
        return l_seq, r_seq, l_valid, r_valid

    except (ImportError, Exception) as e:
        log.warning(f"Tasks API detection failed: {e}")
        return [], [], [], []


# ─────────────────────────────────────────────
# RTMPose-WholeBody Detection (best for fists, body contact, occlusion)
# ─────────────────────────────────────────────
_RTMPOSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'rtmpose_wholebody.pth')

# COCO-WholeBody keypoint mapping to our 47-point system
# RTMPose-WholeBody outputs 133 keypoints:
#   Body: 0-16, Feet: 17-22, Face: 23-90 (68 pts), LHand: 91-111, RHand: 112-132
_RTMPOSE_LHAND_START = 91
_RTMPOSE_RHAND_START = 112
# Face keypoints in COCO-WholeBody 68-point face (offset by 23 in full 133)
_RTMPOSE_FACE_NOSE = 23 + 30       # nose tip
_RTMPOSE_FACE_CHIN = 23 + 8        # chin center (jawline point)
_RTMPOSE_FACE_FOREHEAD = 23 + 27   # between eyebrows (closest to forehead)
_RTMPOSE_FACE_LEFT_EAR = 23 + 0    # left face contour start
_RTMPOSE_FACE_RIGHT_EAR = 23 + 16  # right face contour end
_RTMPOSE_FACE_INDICES = [_RTMPOSE_FACE_NOSE, _RTMPOSE_FACE_CHIN, _RTMPOSE_FACE_FOREHEAD,
                          _RTMPOSE_FACE_LEFT_EAR, _RTMPOSE_FACE_RIGHT_EAR]

_rtmpose_verified = False
_rtmpose_inferencer = None

def _get_rtmpose_inferencer():
    """Get or create cached RTMPose inferencer (one per worker process)."""
    global _rtmpose_verified, _rtmpose_inferencer

    if _rtmpose_inferencer is not None:
        return _rtmpose_inferencer

    from mmpose.apis import MMPoseInferencer

    model_path = _RTMPOSE_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RTMPose checkpoint not found: {model_path}")

    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    inferencer = MMPoseInferencer(
        pose2d='wholebody',
        pose2d_weights=model_path,
        device=device,
    )

    if not _rtmpose_verified:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import numpy as _np
            _test_img = _np.zeros((256, 192, 3), dtype=_np.uint8)
            list(inferencer(_test_img, return_vis=False))
            size_warnings = [x for x in w if 'size mismatch' in str(x.message)]
            if size_warnings:
                raise RuntimeError(
                    f"RTMPose checkpoint/config mismatch! {len(size_warnings)} size mismatches detected. "
                    f"Checkpoint '{model_path}' does not match the 'wholebody' model config. "
                    f"Fix: run 'python models/download_rtmpose.py' to download the correct checkpoint."
                )
        _rtmpose_verified = True
        log.info("RTMPose verified — checkpoint matches config.")

    _rtmpose_inferencer = inferencer
    return inferencer

def _pick_primary_person(preds_list):
    """Pick the primary signer from multiple detected people.
    Strategy: largest bounding box area (most prominent person in frame).
    Returns the best prediction dict, or None.
    """
    if not preds_list:
        return None
    best_pred = None
    best_area = -1
    for pred in preds_list:
        keypoints = pred.get('keypoints', [])
        if len(keypoints) < 133:
            continue
        # Use body keypoints (0-16) to estimate bounding box
        body_kps = keypoints[:17]
        body_scores = pred.get('keypoint_scores', [])[:17]
        visible = [kp for kp, s in zip(body_kps, body_scores) if s > 0.3]
        if len(visible) < 3:
            continue
        xs = [kp[0] for kp in visible]
        ys = [kp[1] for kp in visible]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_pred = pred
    return best_pred


_RTMPOSE_BATCH_SIZE = 8

def _detect_pass_rtmpose(frames_rgb, frame_indices, cfg):
    """Run detection using RTMPose-WholeBody with batch inference.
    Processes frames in batches for GPU efficiency.
    Picks the largest person when multiple people are detected.
    Returns (l_seq, r_seq, face_seq, l_valid, r_valid, face_valid).
    """
    inferencer = _get_rtmpose_inferencer()

    l_seq, r_seq, face_seq = [], [], []
    l_valid, r_valid, face_valid = [], [], []

    hand_conf_threshold = 0.3

    # Process in batches for GPU throughput
    for batch_start in range(0, len(frames_rgb), _RTMPOSE_BATCH_SIZE):
        batch_frames = frames_rgb[batch_start:batch_start + _RTMPOSE_BATCH_SIZE]
        batch_indices = frame_indices[batch_start:batch_start + _RTMPOSE_BATCH_SIZE]

        # MMPose inferencer accepts a list of images for batch processing
        batch_results = list(inferencer(batch_frames, return_vis=False))

        for result, rgb, fidx in zip(batch_results, batch_frames, batch_indices):
            if not result:
                continue

            h, w = rgb.shape[:2]

            preds = result.get('predictions', [[]])
            if not preds or not preds[0]:
                continue

            # Pick primary signer (largest person) instead of blindly taking first
            pred = _pick_primary_person(preds[0])
            if pred is None:
                continue

            keypoints = pred.get('keypoints', [])
            scores = pred.get('keypoint_scores', [])

            # Extract left hand (21 keypoints, indices 91-111)
            l_hand_kps = keypoints[_RTMPOSE_LHAND_START:_RTMPOSE_LHAND_START + 21]
            l_hand_scores = scores[_RTMPOSE_LHAND_START:_RTMPOSE_LHAND_START + 21]
            l_mean_conf = sum(l_hand_scores) / 21

            if l_mean_conf >= hand_conf_threshold:
                coords = [[kp[0] / w, kp[1] / h, 0.0] for kp in l_hand_kps]
                if not l_valid or l_valid[-1] != fidx:
                    l_seq.append(coords)
                    l_valid.append(fidx)

            # Extract right hand (21 keypoints, indices 112-132)
            r_hand_kps = keypoints[_RTMPOSE_RHAND_START:_RTMPOSE_RHAND_START + 21]
            r_hand_scores = scores[_RTMPOSE_RHAND_START:_RTMPOSE_RHAND_START + 21]
            r_mean_conf = sum(r_hand_scores) / 21

            if r_mean_conf >= hand_conf_threshold:
                coords = [[kp[0] / w, kp[1] / h, 0.0] for kp in r_hand_kps]
                if not r_valid or r_valid[-1] != fidx:
                    r_seq.append(coords)
                    r_valid.append(fidx)

            # Extract face (5 keypoints from 68-point face)
            face_coords = []
            face_ok = True
            for face_idx in _RTMPOSE_FACE_INDICES:
                if face_idx < len(keypoints) and scores[face_idx] >= 0.3:
                    kp = keypoints[face_idx]
                    face_coords.append([kp[0] / w, kp[1] / h, 0.0])
                else:
                    face_ok = False
                    break

            if face_ok and len(face_coords) == 5:
                if not face_valid or face_valid[-1] != fidx:
                    face_seq.append(face_coords)
                    face_valid.append(fidx)

    return l_seq, r_seq, face_seq, l_valid, r_valid, face_valid


# ─────────────────────────────────────────────
# RTMW-l WholeBody Detection (Primary GPU extractor)
# Trained on 14 datasets (Cocktail14), distilled from RTMW-x teacher.
# 66.3% hand AP vs RTMPose-m's 47.5% — eliminates silent bad detections.
# ─────────────────────────────────────────────
_RTMW_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'rtmw_l_wholebody.pth')
_RTMW_CONFIG = 'rtmw-l_8xb320-270e_cocktail14-384x288'  # mmpose metafile Name (resolved via .mim/configs/)

# Per-device inferencer cache: {device_str: inferencer}
_rtmw_inferencers = {}
_rtmw_available = None  # None = not checked yet, True/False after first check

def _get_rtmw_inferencer(device=None):
    """Get or create cached RTMW-l inferencer for the given device.
    Supports multi-GPU: pass device='cuda:0', 'cuda:1', etc.
    Returns None if RTMW checkpoint is not available."""
    global _rtmw_inferencers, _rtmw_available

    if _rtmw_available is False:
        return None

    import torch
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device in _rtmw_inferencers:
        return _rtmw_inferencers[device]

    model_path = _RTMW_MODEL_PATH
    if not os.path.exists(model_path):
        log.info("RTMW-l checkpoint not found — skipping RTMW pass.")
        _rtmw_available = False
        return None

    try:
        from mmpose.apis import MMPoseInferencer

        inferencer = MMPoseInferencer(
            pose2d=_RTMW_CONFIG,
            pose2d_weights=model_path,
            device=device,
        )
        _rtmw_inferencers[device] = inferencer
        _rtmw_available = True
        log.info(f"RTMW-l loaded OK on {device}.")
        return inferencer
    except Exception as e:
        log.warning(f"RTMW-l init failed on {device}: {e}")
        _rtmw_available = False
        return None


_RTMW_BATCH_SIZE = 32  # 1 worker per GPU = full 24GB VRAM, large batches are fine

def _detect_pass_rtmw(frames_rgb, frame_indices, cfg, device=None):
    """Run detection using RTMW-l WholeBody with batch inference.
    Primary GPU extractor — 133 keypoints (COCO-WholeBody layout).
    Returns (l_seq, r_seq, face_seq, l_valid, r_valid, face_valid).
    """
    inferencer = _get_rtmw_inferencer(device=device)
    if inferencer is None:
        return [], [], [], [], [], []

    l_seq, r_seq, face_seq = [], [], []
    l_valid, r_valid, face_valid = [], [], []

    hand_conf_threshold = 0.25  # RTMW is high-quality; lower threshold is safe

    for batch_start in range(0, len(frames_rgb), _RTMW_BATCH_SIZE):
        batch_frames = frames_rgb[batch_start:batch_start + _RTMW_BATCH_SIZE]
        batch_indices = frame_indices[batch_start:batch_start + _RTMW_BATCH_SIZE]

        batch_results = list(inferencer(batch_frames, return_vis=False))

        for result, rgb, fidx in zip(batch_results, batch_frames, batch_indices):
            if not result:
                continue

            h, w = rgb.shape[:2]

            preds = result.get('predictions', [[]])
            if not preds or not preds[0]:
                continue

            pred = _pick_primary_person(preds[0])
            if pred is None:
                continue

            keypoints = pred.get('keypoints', [])
            scores = pred.get('keypoint_scores', [])

            if len(keypoints) < 133:
                continue

            # Extract left hand (91-111)
            l_hand_kps = keypoints[_RTMPOSE_LHAND_START:_RTMPOSE_LHAND_START + 21]
            l_hand_scores = scores[_RTMPOSE_LHAND_START:_RTMPOSE_LHAND_START + 21]
            l_mean_conf = sum(l_hand_scores) / 21

            if l_mean_conf >= hand_conf_threshold:
                coords = [[kp[0] / w, kp[1] / h, 0.0] for kp in l_hand_kps]
                if not l_valid or l_valid[-1] != fidx:
                    l_seq.append(coords)
                    l_valid.append(fidx)

            # Extract right hand (112-132)
            r_hand_kps = keypoints[_RTMPOSE_RHAND_START:_RTMPOSE_RHAND_START + 21]
            r_hand_scores = scores[_RTMPOSE_RHAND_START:_RTMPOSE_RHAND_START + 21]
            r_mean_conf = sum(r_hand_scores) / 21

            if r_mean_conf >= hand_conf_threshold:
                coords = [[kp[0] / w, kp[1] / h, 0.0] for kp in r_hand_kps]
                if not r_valid or r_valid[-1] != fidx:
                    r_seq.append(coords)
                    r_valid.append(fidx)

            # Extract face (5 landmarks from 68-point face)
            face_coords = []
            face_ok = True
            for face_idx in _RTMPOSE_FACE_INDICES:
                if face_idx < len(keypoints) and scores[face_idx] >= 0.25:
                    kp = keypoints[face_idx]
                    face_coords.append([kp[0] / w, kp[1] / h, 0.0])
                else:
                    face_ok = False
                    break

            if face_ok and len(face_coords) == 5:
                if not face_valid or face_valid[-1] != fidx:
                    face_seq.append(face_coords)
                    face_valid.append(fidx)

    return l_seq, r_seq, face_seq, l_valid, r_valid, face_valid


# ─────────────────────────────────────────────
# MediaPipe Detection Helper
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
# Frame Preprocessing Pipeline
# ─────────────────────────────────────────────
def _adaptive_gamma(frame):
    """Auto-correct brightness for over/underexposed frames.
    Computes mean luminance and adjusts gamma to target ~120/255."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = gray.mean()
    if mean_lum < 1:
        return frame
    target = 120.0
    gamma = np.log(target / 255.0) / np.log(mean_lum / 255.0 + 1e-8)
    gamma = np.clip(gamma, 0.4, 2.5)  # don't over-correct
    if 0.85 < gamma < 1.15:
        return frame  # close enough, skip
    lut = np.array([((i / 255.0) ** gamma) * 255
                     for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def _unsharp_mask(frame, sigma=1.0, strength=0.5):
    """Sharpen blurry frames to help landmark detection.
    Light sharpening — just enough to recover edges without amplifying noise."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def preprocess_frame(frame, clahe):
    """Lightweight preprocessing: conditional gamma + CLAHE only.
    RTMW-l is trained on 14 diverse datasets (Cocktail14) — robust to noise and blur.
    Skip gamma for well-lit videos, skip unsharp entirely."""
    # 1. Adaptive gamma only for extreme exposure (skip for normal lighting)
    mean_lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    if mean_lum < 60 or mean_lum > 200:
        frame = _adaptive_gamma(frame)

    # 2. CLAHE contrast enhancement (cheap and always helps)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


# ─────────────────────────────────────────────
# Worker Process (v9.0: RTMW-l primary + Multi-GPU + Quality Pipeline)
# ─────────────────────────────────────────────
def process_single_video(task_info):
    try:
        # task_info can be 5-tuple (local) or 6-tuple (extract_do with gpu_id)
        if len(task_info) == 6:
            root, video_name, label, cfg, _, gpu_id = task_info
        else:
            root, video_name, label, cfg, _ = task_info
            gpu_id = 0  # default to first GPU
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

            # Full preprocessing: gamma → CLAHE → bilateral denoise → sharpen
            frame = preprocess_frame(frame, clahe)

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

        # ── Phase 2: Cascaded detection ──
        # GPU:  RTMW-l (primary, 66.3% hand AP) → MediaPipe video → MediaPipe static
        # CPU:  MediaPipe video → static → Tasks API
        #
        # Initialize all pass results to empty (so diagnostics work even if passes are skipped)
        rw_l_seq, rw_r_seq, rw_face_seq, rw_l_valid, rw_r_valid, rw_face_valid = [], [], [], [], [], []
        v_l_seq, v_r_seq, v_face_seq, v_l_valid, v_r_valid, v_face_valid = [], [], [], [], [], []
        s_l_seq, s_r_seq, s_face_seq, s_l_valid, s_r_valid, s_face_valid = [], [], [], [], [], []
        t_l_seq, t_r_seq, t_l_valid, t_r_valid = [], [], [], []

        def _dominant_coverage(*hand_valids):
            """Coverage of the dominant hand (whichever detected more frames)."""
            if processed_count == 0:
                return 0.0
            best = max(len(v) for v in hand_valids)
            return best / processed_count

        # Auto-detect GPU availability
        import torch as _torch
        use_rtmw_first = cfg.prefer_rtmw and _torch.cuda.is_available()

        if use_rtmw_first:
            # ── GPU path: RTMW-l → MediaPipe video → MediaPipe static ──
            device = f'cuda:{gpu_id}' if _torch.cuda.is_available() else 'cpu'

            # Pass 1 — RTMW-l WholeBody (primary, high-accuracy)
            try:
                rw_l_seq, rw_r_seq, rw_face_seq, rw_l_valid, rw_r_valid, rw_face_valid = \
                    _detect_pass_rtmw(frames_rgb, frame_indices, cfg, device=device)
            except Exception as e:
                log.warning(f"RTMW-l failed on {device}, falling through to MediaPipe: {e}")

            # Cascade: if RTMW dominant hand >= 80%, done
            if _dominant_coverage(rw_l_valid, rw_r_valid) < 0.80:

                # Pass 2 — MediaPipe video mode (temporal tracking)
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

                # Cascade: if RTMW + video >= 65%, skip static
                if _dominant_coverage(rw_l_valid, rw_r_valid, v_l_valid, v_r_valid) < 0.65:

                    # Pass 3 — MediaPipe static mode
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

        else:
            # ── CPU path: MediaPipe first → Tasks API ──
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

            if _dominant_coverage(v_l_valid, v_r_valid, s_l_valid, s_r_valid) < 0.80:
                t_l_seq, t_r_seq, t_l_valid, t_r_valid = \
                    _detect_pass_tasks(frames_rgb, frame_indices, cfg)

        del frames_rgb  # Free frame buffer early to reduce memory pressure

        # ── Phase 3: Merge — confidence-weighted pass selection ──
        def _pick_best_pass(candidates):
            """Pick best pass: prefer highest count, but when counts are within 10%
            of each other, prefer the pass with tighter spatial coherence (lower wrist variance)."""
            non_empty = [(seq, valid) for seq, valid in candidates if valid]
            if not non_empty:
                return [], []
            best_count = max(len(v) for _, v in non_empty)
            # Candidates within 10% of best count
            close = [(seq, valid) for seq, valid in non_empty
                     if len(valid) >= best_count * 0.9]
            if len(close) == 1:
                return close[0]
            # Tiebreak: lowest wrist position variance (more stable detection)
            def _wrist_var(seq):
                arr = np.array(seq)
                if arr.ndim < 3 or len(arr) < 2:
                    return float('inf')
                wrist = arr[:, 0, :2]
                return np.mean(np.var(np.diff(wrist, axis=0), axis=0))
            return min(close, key=lambda x: _wrist_var(x[0]))

        all_l = [(rw_l_seq, rw_l_valid), (v_l_seq, v_l_valid), (s_l_seq, s_l_valid),
                 (t_l_seq, t_l_valid)]
        all_r = [(rw_r_seq, rw_r_valid), (v_r_seq, v_r_valid), (s_r_seq, s_r_valid),
                 (t_r_seq, t_r_valid)]

        l_seq, l_valid = _pick_best_pass(all_l)
        r_seq, r_valid = _pick_best_pass(all_r)

        all_face = [(rw_face_seq, rw_face_valid), (v_face_seq, v_face_valid),
                    (s_face_seq, s_face_valid)]
        face_seq, face_valid = max(all_face, key=lambda x: len(x[1]))

        if not l_valid and not r_valid:
            return 0, label, False, None

        # Temporal coherence: reject frames with unrealistic jumps (false detections)
        if l_valid:
            l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
        if r_valid:
            r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)

        if not l_valid and not r_valid:
            return 0, label, False, None

        max_detected = max(len(l_valid), len(r_valid))
        if 1.0 - (max_detected / processed_count) > cfg.max_missing_ratio:
            return 0, label, False, None

        # ── Phase 4: Interpolation, smoothing, normalization, kinematics ──
        l_full = interpolate_hand(np.array(l_seq), l_valid, total_frame_idx)
        r_full = interpolate_hand(np.array(r_seq), r_valid, total_frame_idx)
        face_full = interpolate_face(np.array(face_seq) if face_seq else np.zeros((0, 5, 3)),
                                      face_valid, total_frame_idx)

        combined = np.concatenate([l_full, r_full, face_full], axis=1)
        l_ever, r_ever = bool(l_valid), bool(r_valid)
        face_ever = bool(face_valid)

        resampled = temporal_resample(combined, cfg.target_frames)

        # 1-Euro adaptive smoothing on XYZ: heavy for still, light for fast signing
        smoothed_xyz = one_euro_filter(resampled[:, :, :3])
        resampled[:, :, :3] = smoothed_xyz

        # Bone length stabilization: normalize to median across sequence
        if l_ever:
            resampled = stabilize_bones(resampled, 0, 21)
        if r_ever:
            resampled = stabilize_bones(resampled, 21, 42)

        normalized = normalize_sequence(resampled, l_ever, r_ever)

        # Build per-frame detection mask (not binary per-hand)
        # Marks which frames had real detections vs interpolated
        T = cfg.target_frames
        per_frame_mask = np.zeros((1, T, NUM_NODES, 1), dtype=np.float32)
        if l_valid:
            l_coverage = np.interp(
                np.linspace(0, total_frame_idx - 1, T),
                sorted(l_valid),
                np.ones(len(l_valid))
            )
            # Frames near real detections get higher mask values
            for t in range(T):
                per_frame_mask[0, t, 0:21, 0] = l_coverage[t]
        if r_valid:
            r_coverage = np.interp(
                np.linspace(0, total_frame_idx - 1, T),
                sorted(r_valid),
                np.ones(len(r_valid))
            )
            for t in range(T):
                per_frame_mask[0, t, 21:42, 0] = r_coverage[t]
        if face_valid:
            per_frame_mask[0, :, 42:47, 0] = 1.0

        final_data = compute_kinematics_batch(normalized[np.newaxis, ...], l_ever, r_ever, face_ever,
                                               per_frame_mask=per_frame_mask)
        final_data = final_data.squeeze(0).astype(np.float16)

        file_hash = hashlib.md5(video_name.encode()).hexdigest()[:6]
        # Strip label prefix from stem to avoid doubled labels (e.g. CAN_CAN_10)
        stem = Path(video_name).stem
        if stem.startswith(label + "_"):
            stem = stem[len(label) + 1:]
        save_name = f"{label}_{stem}_{file_hash}.npy"
        np.save(out_path / save_name, final_data)
        saved_count = 1

        # Multi-pass augmentation
        if cfg.enable_augmentation:
            xyz_base = resampled[:, :, :3]

            if random.random() < cfg.augment_probability:
                speed_factor = random.uniform(1.15, 1.25)
                aug_xyz = temporal_speed_warp(xyz_base, speed_factor, cfg.target_frames)
                aug_data = recompute_full_features(aug_xyz, l_ever, r_ever, face_ever)
                aug_name = f"{label}_{stem}_fast_{file_hash}.npy"
                np.save(out_path / aug_name, aug_data)
                saved_count += 1

            if random.random() < cfg.augment_probability:
                speed_factor = random.uniform(0.75, 0.85)
                aug_xyz = temporal_speed_warp(xyz_base, speed_factor, cfg.target_frames)
                aug_data = recompute_full_features(aug_xyz, l_ever, r_ever, face_ever)
                aug_name = f"{label}_{stem}_slow_{file_hash}.npy"
                np.save(out_path / aug_name, aug_data)
                saved_count += 1

            if random.random() < cfg.augment_probability:
                aug_xyz = mirror_hands_xyz(xyz_base)
                aug_data = recompute_full_features(aug_xyz, r_ever, l_ever, face_ever)
                aug_name = f"{label}_{stem}_mirror_{file_hash}.npy"
                np.save(out_path / aug_name, aug_data)
                saved_count += 1

        # Diagnostics: which pass won, detection coverage
        def _best_pass(rw_count, v_count, s_count, t_count):
            counts = {'rtmw': rw_count, 'video': v_count, 'static': s_count, 'tasks': t_count}
            return max(counts, key=counts.get)

        diag = {
            'l_pass': _best_pass(len(rw_l_valid), len(v_l_valid), len(s_l_valid), len(t_l_valid)),
            'r_pass': _best_pass(len(rw_r_valid), len(v_r_valid), len(s_r_valid), len(t_r_valid)),
            'face_detected': face_ever,
            'detection_coverage': max(len(l_valid), len(r_valid)) / max(processed_count, 1),
            'rtmw_l_detections': len(rw_l_valid),
            'rtmw_r_detections': len(rw_r_valid),
            'video_l_detections': len(v_l_valid),
            'video_r_detections': len(v_r_valid),
            'static_l_detections': len(s_l_valid),
            'static_r_detections': len(s_r_valid),
            'tasks_l_detections': len(t_l_valid),
            'tasks_r_detections': len(t_r_valid),
        }

        return saved_count, label, True, diag

    except Exception as e:
        log.error(f"Failed processing {video_name}: {e}")
        return 0, task_info[2], False, None

# ─────────────────────────────────────────────
# Pipeline-mode functions (used by extract_do.py)
# Decomposes process_single_video into phases for GPU pipeline.
# ─────────────────────────────────────────────

def decode_video(task_info):
    """Phase 1: Read video and preprocess frames. Pure CPU, no GPU needed.
    Returns dict with decoded data, or None if video should be skipped.
    """
    if len(task_info) == 6:
        root, video_name, label, cfg, file_size, gpu_id = task_info
    else:
        root, video_name, label, cfg, file_size = task_info
        gpu_id = 0

    out_path = Path(cfg.output_dir)
    video_path = os.path.join(root, video_name)

    cap = cv2.VideoCapture(video_path)
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if 0 < total_est < cfg.min_raw_frames:
        cap.release()
        return None

    if total_est < 80:
        skip = 1
    else:
        skip = max(1, total_est // 64)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frames_rgb = []
    frame_indices = []
    frame_idx = 0

    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue
        ret, frame = cap.retrieve()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        frame = preprocess_frame(frame, clahe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        frames_rgb.append(rgb)
        frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    total_frame_idx = frame_idx
    processed_count = len(frames_rgb)

    if processed_count < cfg.min_raw_frames:
        return None

    # Build save name (same logic as process_single_video)
    file_hash = hashlib.md5(video_name.encode()).hexdigest()[:6]
    stem = Path(video_name).stem
    if stem.startswith(label + "_"):
        stem = stem[len(label) + 1:]
    save_name_base = f"{label}_{stem}_{file_hash}"

    return {
        'frames_rgb': frames_rgb,
        'frame_indices': frame_indices,
        'total_frame_idx': total_frame_idx,
        'processed_count': processed_count,
        'label': label,
        'cfg': cfg,
        'gpu_id': gpu_id,
        'out_path': str(out_path),
        'save_name_base': save_name_base,
        'video_name': video_name,
    }


def process_decoded_video(decoded, device=None):
    """Phase 2+3: RTMW inference + MediaPipe fallback + merge + normalize + save.

    Takes pre-decoded frames from decode_video() and runs the full detection cascade,
    postprocessing, and saving. Used by GPU pipeline workers.

    Args:
        decoded: dict from decode_video()
        device: CUDA device string (e.g. 'cuda:0'). None for CPU-only path.

    Returns: (count, label, success, diag)
    """
    try:
        frames_rgb = decoded['frames_rgb']
        frame_indices = decoded['frame_indices']
        total_frame_idx = decoded['total_frame_idx']
        processed_count = decoded['processed_count']
        label = decoded['label']
        cfg = decoded['cfg']
        out_path = Path(decoded['out_path'])
        save_name_base = decoded['save_name_base']
        video_name = decoded['video_name']

        # Initialize all pass results
        rw_l_seq, rw_r_seq, rw_face_seq = [], [], []
        rw_l_valid, rw_r_valid, rw_face_valid = [], [], []
        v_l_seq, v_r_seq, v_face_seq = [], [], []
        v_l_valid, v_r_valid, v_face_valid = [], [], []
        s_l_seq, s_r_seq, s_face_seq = [], [], []
        s_l_valid, s_r_valid, s_face_valid = [], [], []
        t_l_seq, t_r_seq, t_l_valid, t_r_valid = [], [], [], []

        def _dominant_coverage(*hand_valids):
            if processed_count == 0:
                return 0.0
            best = max(len(v) for v in hand_valids)
            return best / processed_count

        import torch as _torch
        use_rtmw = device is not None and _torch.cuda.is_available()

        if use_rtmw:
            # ── GPU path: RTMW-l → MediaPipe video → MediaPipe static ──
            try:
                rw_l_seq, rw_r_seq, rw_face_seq, rw_l_valid, rw_r_valid, rw_face_valid = \
                    _detect_pass_rtmw(frames_rgb, frame_indices, cfg, device=device)
            except Exception as e:
                log.warning(f"RTMW-l failed on {device}: {e}")

            if _dominant_coverage(rw_l_valid, rw_r_valid) < 0.80:
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

                if _dominant_coverage(rw_l_valid, rw_r_valid, v_l_valid, v_r_valid) < 0.65:
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
        else:
            # ── CPU path: MediaPipe → Tasks API ──
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

            if _dominant_coverage(v_l_valid, v_r_valid, s_l_valid, s_r_valid) < 0.80:
                t_l_seq, t_r_seq, t_l_valid, t_r_valid = \
                    _detect_pass_tasks(frames_rgb, frame_indices, cfg)

        del frames_rgb  # Free frame buffer early

        # ── Merge — confidence-weighted pass selection ──
        def _pick_best_pass(candidates):
            non_empty = [(seq, valid) for seq, valid in candidates if valid]
            if not non_empty:
                return [], []
            best_count = max(len(v) for _, v in non_empty)
            close = [(seq, valid) for seq, valid in non_empty
                     if len(valid) >= best_count * 0.9]
            if len(close) == 1:
                return close[0]
            def _wrist_var(seq):
                arr = np.array(seq)
                if arr.ndim < 3 or len(arr) < 2:
                    return float('inf')
                wrist = arr[:, 0, :2]
                return np.mean(np.var(np.diff(wrist, axis=0), axis=0))
            return min(close, key=lambda x: _wrist_var(x[0]))

        all_l = [(rw_l_seq, rw_l_valid), (v_l_seq, v_l_valid), (s_l_seq, s_l_valid),
                 (t_l_seq, t_l_valid)]
        all_r = [(rw_r_seq, rw_r_valid), (v_r_seq, v_r_valid), (s_r_seq, s_r_valid),
                 (t_r_seq, t_r_valid)]

        l_seq, l_valid = _pick_best_pass(all_l)
        r_seq, r_valid = _pick_best_pass(all_r)

        all_face = [(rw_face_seq, rw_face_valid), (v_face_seq, v_face_valid),
                    (s_face_seq, s_face_valid)]
        face_seq, face_valid = max(all_face, key=lambda x: len(x[1]))

        if not l_valid and not r_valid:
            return 0, label, False, None

        # Temporal coherence rejection
        if l_valid:
            l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
        if r_valid:
            r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)

        if not l_valid and not r_valid:
            return 0, label, False, None

        max_detected = max(len(l_valid), len(r_valid))
        if 1.0 - (max_detected / processed_count) > cfg.max_missing_ratio:
            return 0, label, False, None

        # ── Interpolation, smoothing, normalization, kinematics ──
        l_full = interpolate_hand(np.array(l_seq), l_valid, total_frame_idx)
        r_full = interpolate_hand(np.array(r_seq), r_valid, total_frame_idx)
        face_full = interpolate_face(np.array(face_seq) if face_seq else np.zeros((0, 5, 3)),
                                      face_valid, total_frame_idx)

        combined = np.concatenate([l_full, r_full, face_full], axis=1)
        l_ever, r_ever = bool(l_valid), bool(r_valid)
        face_ever = bool(face_valid)

        resampled = temporal_resample(combined, cfg.target_frames)
        smoothed_xyz = one_euro_filter(resampled[:, :, :3])
        resampled[:, :, :3] = smoothed_xyz

        if l_ever:
            resampled = stabilize_bones(resampled, 0, 21)
        if r_ever:
            resampled = stabilize_bones(resampled, 21, 42)

        normalized = normalize_sequence(resampled, l_ever, r_ever)

        T = cfg.target_frames
        per_frame_mask = np.zeros((1, T, NUM_NODES, 1), dtype=np.float32)
        if l_valid:
            l_coverage = np.interp(
                np.linspace(0, total_frame_idx - 1, T),
                sorted(l_valid),
                np.ones(len(l_valid))
            )
            for t in range(T):
                per_frame_mask[0, t, 0:21, 0] = l_coverage[t]
        if r_valid:
            r_coverage = np.interp(
                np.linspace(0, total_frame_idx - 1, T),
                sorted(r_valid),
                np.ones(len(r_valid))
            )
            for t in range(T):
                per_frame_mask[0, t, 21:42, 0] = r_coverage[t]
        if face_valid:
            per_frame_mask[0, :, 42:47, 0] = 1.0

        final_data = compute_kinematics_batch(normalized[np.newaxis, ...], l_ever, r_ever, face_ever,
                                               per_frame_mask=per_frame_mask)
        final_data = final_data.squeeze(0).astype(np.float16)

        save_name = f"{save_name_base}.npy"
        np.save(out_path / save_name, final_data)
        saved_count = 1

        # Augmentation
        if cfg.enable_augmentation:
            xyz_base = resampled[:, :, :3]
            if random.random() < cfg.augment_probability:
                aug_xyz = temporal_speed_warp(xyz_base, random.uniform(1.15, 1.25), cfg.target_frames)
                np.save(out_path / f"{save_name_base}_fast.npy",
                        recompute_full_features(aug_xyz, l_ever, r_ever, face_ever))
                saved_count += 1
            if random.random() < cfg.augment_probability:
                aug_xyz = temporal_speed_warp(xyz_base, random.uniform(0.75, 0.85), cfg.target_frames)
                np.save(out_path / f"{save_name_base}_slow.npy",
                        recompute_full_features(aug_xyz, l_ever, r_ever, face_ever))
                saved_count += 1
            if random.random() < cfg.augment_probability:
                aug_xyz = mirror_hands_xyz(xyz_base)
                np.save(out_path / f"{save_name_base}_mirror.npy",
                        recompute_full_features(aug_xyz, r_ever, l_ever, face_ever))
                saved_count += 1

        # Diagnostics
        def _best_pass(rw_count, v_count, s_count, t_count):
            counts = {'rtmw': rw_count, 'video': v_count, 'static': s_count, 'tasks': t_count}
            return max(counts, key=counts.get)

        diag = {
            'l_pass': _best_pass(len(rw_l_valid), len(v_l_valid), len(s_l_valid), len(t_l_valid)),
            'r_pass': _best_pass(len(rw_r_valid), len(v_r_valid), len(s_r_valid), len(t_r_valid)),
            'face_detected': face_ever,
            'detection_coverage': max(len(l_valid), len(r_valid)) / max(processed_count, 1),
            'rtmw_l_detections': len(rw_l_valid),
            'rtmw_r_detections': len(rw_r_valid),
            'video_l_detections': len(v_l_valid),
            'video_r_detections': len(v_r_valid),
            'static_l_detections': len(s_l_valid),
            'static_r_detections': len(s_r_valid),
            'tasks_l_detections': len(t_l_valid),
            'tasks_r_detections': len(t_r_valid),
        }

        return saved_count, label, True, diag

    except Exception as e:
        log.error(f"Failed processing {decoded.get('video_name', '?')}: {e}")
        return 0, decoded.get('label', '?'), False, None


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
                if stem.startswith(label + "_"):
                    stem = stem[len(label) + 1:]
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
    pass_names = ['rtmw', 'video', 'static', 'tasks']
    class_l_pass = {p: Counter() for p in pass_names}
    class_r_pass = {p: Counter() for p in pass_names}
    class_face_detected = Counter()
    class_coverage_sum = defaultdict(float)

    saved_count = 0
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        for result in executor.map(process_single_video, all_videos, chunksize=chunk_size):
            count, label, success, diag = result
            saved_count += count
            if success:
                class_success[label] += 1
                if diag:
                    l_winner = diag.get('l_pass', 'video')
                    r_winner = diag.get('r_pass', 'video')
                    if l_winner in class_l_pass:
                        class_l_pass[l_winner][label] += 1
                    if r_winner in class_r_pass:
                        class_r_pass[r_winner][label] += 1
                    if diag['face_detected']:
                        class_face_detected[label] += 1
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

    # Cascade diagnostics summary
    total_success = sum(class_success.values())
    total_face_det = sum(class_face_detected.values())
    if total_success > 0:
        log.info(f"Cascade stats ({total_success} clips):")
        for p in pass_names:
            l_wins = sum(class_l_pass[p].values())
            r_wins = sum(class_r_pass[p].values())
            if l_wins > 0 or r_wins > 0:
                log.info(f"  {p:>8s}: L-hand won {l_wins}/{total_success} ({100*l_wins/total_success:.0f}%), "
                         f"R-hand won {r_wins}/{total_success} ({100*r_wins/total_success:.0f}%)")
        log.info(f"  Face detected: {total_face_det}/{total_success} ({100*total_face_det/total_success:.0f}%)")

    # Save extraction_stats.json
    stats = {}
    for label in all_labels:
        s = class_success[label]
        f = class_fail[label]
        per_pass = {}
        for p in pass_names:
            per_pass[f'l_{p}_wins'] = class_l_pass[p][label]
            per_pass[f'r_{p}_wins'] = class_r_pass[p][label]
        stats[label] = {
            'success': s, 'fail': f,
            'fail_rate': round(f / max(s + f, 1), 3),
            **per_pass,
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
            if stem.startswith(label + "_"):
                stem = stem[len(label) + 1:]

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
