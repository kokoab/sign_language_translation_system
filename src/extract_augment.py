"""
SLT Extraction & Augmentation Pipeline v2.0
PhD-level redesign for DS-GCN + Seq2Seq Transformer compatibility.

Key improvements over v1:
  - Temporal warping augmentation (critical for 'Z', 'J', 'Hello')
  - Per-joint noise weighting (fingertips noisier than wrist/MCP)
  - Mirror (handedness) augmentation
  - Scale jitter augmentation
  - Roll wrap-around bug fixed → edge-pad temporal shift
  - Velocity & acceleration channels computed and saved
  - Cubic interpolation for smoother dynamic sign trajectories
  - Per-frame confidence tracking for better quality filtering
  - Modular config dataclass for M4 ↔ Kaggle consistency
  - Output shape: [32, 21, 9]  (xyz + vxvyvz + axayaz) or [32, 21, 3]
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from scipy.interpolate import interp1d, CubicSpline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-Extract")

# ─────────────────────────────────────────────
# MediaPipe Node Index Reference (for clarity)
# ─────────────────────────────────────────────
WRIST          = 0
THUMB_CMC      = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP      = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP     = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP       = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP      = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# Fingertip nodes — these get higher noise in augmentation
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
# High-velocity nodes for dynamic signs ('Z', 'J', 'Hello')
DYNAMIC_NODES = [INDEX_TIP, MIDDLE_TIP, THUMB_TIP, WRIST]

# ─────────────────────────────────────────────
# Per-joint noise weight vector [21]
# Wrist/MCPs are anatomically stable → low noise
# Tips move freely → higher noise during augmentation
# ─────────────────────────────────────────────
JOINT_NOISE_WEIGHTS = np.ones(21, dtype=np.float32)
JOINT_NOISE_WEIGHTS[FINGERTIPS] = 2.5    # fingertips get 2.5× noise
JOINT_NOISE_WEIGHTS[[THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]] = 1.8


# ─────────────────────────────────────────────
#  Configuration (single source of truth)
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    # ── Temporal ──────────────────────────────
    target_frames: int = 32
    min_raw_frames: int = 10          # discard shorter sequences
    max_missing_ratio: float = 0.30   # drop video if >30% frames undetected
    interpolation: str = "cubic"      # "linear" | "cubic" (cubic for dynamic signs)

    # ── MediaPipe ─────────────────────────────
    min_detection_conf: float = 0.70
    min_tracking_conf: float = 0.60
    model_complexity: int = 1         # 0=fast, 1=accurate (use 1 for training data)
    max_num_hands: int = 1

    # ── Augmentation ──────────────────────────
    variations_per_video: int = 12
    rotation_range_deg: float = 12.0   # ±12° per axis
    scale_jitter_range: tuple = (0.85, 1.15)
    temporal_shift_range: int = 4      # ±4 frames (edge-padded, no wrap)
    spatial_noise_std: float = 0.003   # base std before per-joint weighting
    time_warp_strength: float = 0.15   # how aggressively to warp the timeline
    mirror_prob: float = 0.5           # probability of applying mirror flip

    # ── Output ────────────────────────────────
    save_velocity: bool = True         # append velocity channels → [32, 21, 6]
    save_acceleration: bool = True     # append accel channels  → [32, 21, 9]
    output_dtype: str = "float32"

    # ── Paths ──────────────────────────────────
    raw_video_dir: str = "data/raw_videos"
    output_dir: str = "data/landmarks"
    meta_path: str = "data/dataset_meta.json"


CFG = PipelineConfig()


# ─────────────────────────────────────────────
#  Normalization
# ─────────────────────────────────────────────
def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Robust Global Scaling: Ensures the hand motion fits in a 1.0 unit sphere.
    """
    seq = sequence.copy().astype(np.float64)
    
    # 1. Wrist-centric (Standard)
    seq -= seq[:, WRIST:WRIST+1, :] 

    # 2. Sequence-wide Maximum Scaling
    # Instead of mean, we use max distance to prevent 'jitter' during movement
    dists = np.linalg.norm(seq, axis=-1)  # [F, 21]
    max_val = np.max(dists) + 1e-8
    seq /= max_val

    return seq.astype(np.float32)

# ─────────────────────────────────────────────
#  Kinematic Feature Computation
# ─────────────────────────────────────────────
def compute_kinematics(seq: np.ndarray) -> np.ndarray:
    """
    Appends velocity and acceleration channels.

    Input:  [F, 21, 3]
    Output: [F, 21, 3]  if save_velocity=False, save_acceleration=False
            [F, 21, 6]  if save_velocity=True,  save_acceleration=False
            [F, 21, 9]  if save_velocity=True,  save_acceleration=True

    Velocity  = central finite difference (first-order)
    Accel     = central finite difference of velocity (second-order)
    Edge frames use forward/backward differences.

    These channels are what lets the Transformer Encoder distinguish:
      • Static 'A' → near-zero velocity, near-zero acceleration
      • Dynamic 'Z' → high velocity at index tip, direction reversal in accel
    """
    if not CFG.save_velocity:
        return seq

    F = seq.shape[0]  # 32
    vel = np.zeros_like(seq)   # [F, 21, 3]

    # Central differences for interior frames
    vel[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    # Forward/backward at edges
    vel[0]    = seq[1]  - seq[0]
    vel[-1]   = seq[-1] - seq[-2]

    channels = [seq, vel]

    if CFG.save_acceleration:
        acc = np.zeros_like(seq)
        acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
        acc[0]    = vel[1]  - vel[0]
        acc[-1]   = vel[-1] - vel[-2]
        channels.append(acc)

    return np.concatenate(channels, axis=-1)  # [F, 21, 3|6|9]


# ─────────────────────────────────────────────
#  Temporal Interpolation
# ─────────────────────────────────────────────
def interpolate_to_target(raw: np.ndarray, target_frames: int,
                           method: str = "cubic") -> np.ndarray:
    """
    Resamples a variable-length sequence to exactly target_frames.

    Input:  [N, 21, 3]  (N = raw frame count)
    Output: [target_frames, 21, 3]

    Uses cubic spline for dynamic signs to preserve motion smoothness.
    Falls back to linear if N < 4 (cubic requires ≥4 points).
    """
    N = len(raw)
    x_src = np.linspace(0, 1, N)
    x_dst = np.linspace(0, 1, target_frames)

    if method == "cubic" and N >= 4:
        # CubicSpline expects [N, features] — reshape then restore
        flat = raw.reshape(N, -1)          # [N, 63]
        cs = CubicSpline(x_src, flat, bc_type='not-a-knot')
        out = cs(x_dst).reshape(target_frames, 21, 3)
    else:
        flat = raw.reshape(N, -1)
        f_interp = interp1d(x_src, flat, axis=0, kind='linear')
        out = f_interp(x_dst).reshape(target_frames, 21, 3)

    return out.astype(np.float32)


# ─────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────
def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Returns combined 3D rotation matrix [3, 3]."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx,  cx]])
    Ry = np.array([[cy, 0, sy], [0, 1,  0], [-sy, 0,  cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0,  0,   1]])
    return Rx @ Ry @ Rz


def _temporal_warp(seq: np.ndarray, strength: float) -> np.ndarray:
    """
    Non-linear temporal warping: randomly speeds up / slows down sections.

    This is the single most important augmentation for dynamic signs.
    'Z' and 'J' can be signed at different speeds; the model must be
    robust to this. Implemented as a monotonic cubic warp of the timeline.

    Input/Output: [F, 21, 3]
    """
    F = len(seq)
    # Create a warped time axis: starts at 0, ends at F-1, monotonically
    # increasing, but with random perturbations in the middle.
    n_knots = 5
    knot_x = np.linspace(0, 1, n_knots)
    knot_y = knot_x + np.random.uniform(-strength, strength, n_knots)
    # Clamp endpoints and enforce monotonicity
    knot_y[0] = 0.0
    knot_y[-1] = 1.0
    # Enforce monotonicity: sort interior knots
    knot_y[1:-1] = np.sort(knot_y[1:-1])
    knot_y = np.clip(knot_y, 0, 1)

    # Build warp function and sample at target frames
    warp_fn = CubicSpline(knot_x, knot_y)
    warped_t = np.clip(warp_fn(np.linspace(0, 1, F)), 0, 1) * (F - 1)

    # Re-interpolate sequence at warped time positions
    src_t = np.arange(F)
    flat = seq.reshape(F, -1)
    f_interp = interp1d(src_t, flat, axis=0, kind='linear',
                        bounds_error=False, fill_value="extrapolate")
    return f_interp(warped_t).reshape(F, 21, 3).astype(np.float32)


def _temporal_shift_edge_pad(seq: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts sequence along time axis and pads edges (no circular wrap).

    Circular wrap (np.roll) in v1 caused the last frame to appear at the
    start, which creates a false motion discontinuity — especially
    harmful for dynamic signs where the start/end pose matters.

    shift > 0 → sequence starts later  (pad start with first frame)
    shift < 0 → sequence starts earlier (pad end with last frame)
    """
    if shift == 0:
        return seq
    F = len(seq)
    result = np.empty_like(seq)
    if shift > 0:
        result[shift:] = seq[:F - shift]
        result[:shift]  = seq[0]          # replicate first frame
    else:
        s = abs(shift)
        result[:F - s] = seq[s:]
        result[F - s:] = seq[-1]          # replicate last frame
    return result


def _mirror_hand(seq: np.ndarray) -> np.ndarray:
    """
    Mirrors the hand by flipping the X axis.

    For a right-hand dominant dataset, this effectively synthesizes
    left-hand signing, doubling the effective dataset size and teaching
    the model handedness invariance where appropriate.
    """
    mirrored = seq.copy()
    mirrored[:, :, 0] *= -1.0   # flip X coordinate
    return mirrored


def generate_augmentations(base_data: np.ndarray,
                            num_variations: int,
                            cfg: PipelineConfig) -> list[np.ndarray]:
    """
    Generates num_variations augmented copies of base_data.

    Input:  base_data [F, 21, 3]  (already normalized)
    Output: list of [F, 21, 3] arrays

    Augmentation stack per variation (all operations preserve
    the [F, 21, 3] shape so normalization remains valid):
      1. 3D Rotation       — pose variation
      2. Scale jitter      — residual size variation
      3. Temporal warp     — speed variation (KEY for 'Z', 'J', 'Hello')
      4. Temporal shift    — temporal alignment variation
      5. Per-joint noise   — sensor noise simulation
      6. Mirror flip       — handedness variation (probabilistic)
    """
    F, P, C = base_data.shape
    aug_list = []

    for _ in range(num_variations):
        aug = base_data.copy()

        # 1. 3D Rotation
        angles = np.random.uniform(-cfg.rotation_range_deg,
                                    cfg.rotation_range_deg, 3)
        R = _rotation_matrix(*angles)
        aug = aug @ R.T   # [F, 21, 3] @ [3, 3] → broadcast correctly

        # 2. Scale jitter (uniform random scale around 1.0)
        scale = np.random.uniform(*cfg.scale_jitter_range)
        aug *= scale

        # 3. Temporal warp (most important for dynamic signs)
        aug = _temporal_warp(aug, cfg.time_warp_strength)

        # 4. Temporal shift with edge padding (no wrap-around)
        shift = np.random.randint(-cfg.temporal_shift_range,
                                   cfg.temporal_shift_range + 1)
        aug = _temporal_shift_edge_pad(aug, shift)

        # 5. Per-joint noise (fingertips noisier than proximal joints)
        #    noise_weights: [21] → broadcast to [F, 21, 3]
        noise = np.random.normal(0, cfg.spatial_noise_std,
                                 (F, P, C)).astype(np.float32)
        noise *= JOINT_NOISE_WEIGHTS[np.newaxis, :, np.newaxis]
        aug += noise

        # 6. Mirror augmentation (probabilistic)
        if np.random.random() < cfg.mirror_prob:
            aug = _mirror_hand(aug)

        aug_list.append(aug.astype(cfg.output_dtype))

    return aug_list


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────
def process_and_augment(cfg: PipelineConfig = CFG):
    """
    Full extraction + augmentation pipeline.

    For each video in cfg.raw_video_dir/<label>/:
      1. Extract MediaPipe landmarks frame-by-frame
      2. Quality filter (missing frame ratio)
      3. Cubic interpolate to TARGET_FRAMES
      4. Wrist-centric + palm-scale normalization
      5. Compute velocity/acceleration channels
      6. Save original + N augmented variants
      7. Write dataset metadata JSON

    Output .npy shape: [32, 21, 9]  (xyz + vel + accel)
                   or  [32, 21, 3]  (xyz only, if flags disabled)
    """
    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mp_hands_module = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_detector = mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=cfg.max_num_hands,
        min_detection_confidence=cfg.min_detection_conf,
        min_tracking_confidence=cfg.min_tracking_conf,
        model_complexity=cfg.model_complexity,
    )

    metadata = {
        "config": asdict(cfg),
        "samples": [],
        "label_counts": {},
    }

    paused = False
    total_saved = 0
    total_skipped = 0

    for root, _, files in os.walk(cfg.raw_video_dir):
        label = Path(root).name
        video_files = [f for f in files
                       if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        if not video_files:
            continue

        for video_name in video_files:
            video_path = os.path.join(root, video_name)
            cap = cv2.VideoCapture(video_path)
            total_video_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            raw_sequence = []        # list of [21, 3]
            frame_confidences = []   # per-frame detection confidence

            # ── Extraction loop ──────────────────────
            while cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = hands_detector.process(rgb)
                    rgb.flags.writeable = True

                    if result.multi_hand_landmarks and result.multi_handedness:
                        lm = result.multi_hand_landmarks[0]
                        conf = result.multi_handedness[0].classification[0].score
                        frame_confidences.append(conf)

                        # Only keep high-confidence frames
                        if conf >= cfg.min_detection_conf:
                            mp_draw.draw_landmarks(
                                frame, lm, mp_hands_module.HAND_CONNECTIONS)
                            raw_sequence.append(
                                [[l.x, l.y, l.z] for l in lm.landmark])

                    # ── HUD ─────────────────────────
                    status_color = (0, 200, 255) if not paused else (0, 80, 255)
                    cv2.rectangle(frame, (0, 0), (w, 85), (15, 15, 15), -1)
                    cv2.putText(frame,
                                f"LABEL: {label}  |  {'PAUSED' if paused else 'EXTRACTING...'}",
                                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 100), 2)
                    cv2.putText(frame,
                                f"FILE: {video_name}  |  "
                                f"Frames captured: {len(raw_sequence)}/{total_video_frames}",
                                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1)
                    cv2.putText(frame,
                                "SPACE: Pause/Play | Q: Quit | S: Skip video",
                                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                status_color, 1)
                    cv2.imshow("SLT Extraction Engine v2", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    paused = not paused
                elif key == ord('s'):    # skip this video
                    log.info(f"⏭  Skipped by user: {video_name}")
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    _write_metadata(metadata, cfg.meta_path)
                    log.info(f"\n🛑 Quit by user. Saved: {total_saved} | Skipped: {total_skipped}")
                    return

            cap.release()

            # ── Quality Filter ────────────────────────
            detected_frames = len(raw_sequence)
            missing_ratio = 1.0 - (detected_frames / total_video_frames)
            avg_conf = float(np.mean(frame_confidences)) if frame_confidences else 0.0

            if missing_ratio > cfg.max_missing_ratio:
                log.warning(
                    f"⚠️  Skipping {video_name}: "
                    f"Missing ratio {missing_ratio:.1%} > {cfg.max_missing_ratio:.0%}  "
                    f"(Detected {detected_frames}/{total_video_frames})")
                total_skipped += 1
                continue

            if detected_frames < cfg.min_raw_frames:
                log.warning(
                    f"⚠️  Skipping {video_name}: "
                    f"Too few frames ({detected_frames} < {cfg.min_raw_frames})")
                total_skipped += 1
                continue

            # ── Processing ───────────────────────────
            raw_np = np.array(raw_sequence, dtype=np.float32)  # [N, 21, 3]

            # Interpolate → normalize → kinematics
            interp_data = interpolate_to_target(
                raw_np, cfg.target_frames, method=cfg.interpolation)
            base_data = normalize_sequence(interp_data)         # [32, 21, 3]
            base_full = compute_kinematics(base_data)           # [32, 21, 3|6|9]

            # ── Save Original ────────────────────────
            stem = f"{label}_{Path(video_name).stem}"
            orig_path = out_path / f"{stem}_orig.npy"
            np.save(orig_path, base_full)
            _register_sample(metadata, label, str(orig_path),
                             base_full.shape, avg_conf, is_augmented=False)
            total_saved += 1

            # ── Save Augmentations ───────────────────
            # Augmentations operate on raw xyz [32, 21, 3] to avoid
            # corrupting velocity channels; we recompute kinematics after.
            aug_variants = generate_augmentations(base_data, cfg.variations_per_video, cfg)
            for v, aug_xyz in enumerate(aug_variants):
                aug_full = compute_kinematics(aug_xyz)   # recompute vel/accel
                aug_path = out_path / f"{stem}_aug{v:02d}.npy"
                np.save(aug_path, aug_full)
                _register_sample(metadata, label, str(aug_path),
                                 aug_full.shape, avg_conf, is_augmented=True)
                total_saved += 1

            metadata["label_counts"][label] = (
                metadata["label_counts"].get(label, 0) + cfg.variations_per_video + 1)

            log.info(
                f"✅  {label:>10} | {video_name:<30} → "
                f"{cfg.variations_per_video + 1} samples  "
                f"(conf={avg_conf:.2f}, missing={missing_ratio:.1%}, "
                f"shape={base_full.shape})")

    cv2.destroyAllWindows()
    hands_detector.close()
    _write_metadata(metadata, cfg.meta_path)

    log.info(f"\n{'─'*60}")
    log.info(f"✅  Done.  Saved: {total_saved} | Skipped: {total_skipped}")
    log.info(f"📁  Output: {cfg.output_dir}")
    log.info(f"📋  Metadata: {cfg.meta_path}")
    log.info(f"📐  Array shape per sample: [32, 21, "
             f"{3 + 3*cfg.save_velocity + 3*cfg.save_acceleration}]")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _register_sample(meta: dict, label: str, path: str,
                     shape: tuple, conf: float, is_augmented: bool):
    meta["samples"].append({
        "label": label,
        "path": path,
        "shape": list(shape),
        "avg_conf": round(conf, 4),
        "is_augmented": is_augmented,
    })


def _write_metadata(meta: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"📋  Metadata saved: {path}")


# ─────────────────────────────────────────────
#  Standalone verification utility
# ─────────────────────────────────────────────
def verify_sample(npy_path: str):
    """
    Quick sanity check for a saved .npy file.
    Verifies shape, value ranges, and wrist constraint.
    """
    data = np.load(npy_path)
    print(f"\n{'─'*50}")
    print(f"File   : {npy_path}")
    print(f"Shape  : {data.shape}")   # expect [32, 21, 9]
    xyz = data[:, :, :3]
    print(f"XYZ min/max : {xyz.min():.4f} / {xyz.max():.4f}")
    print(f"XYZ mean    : {xyz.mean():.4f}")
    wrist_deviation = np.abs(xyz[:, WRIST, :]).max()
    print(f"Wrist max deviation from (0,0,0): {wrist_deviation:.6f}  "
          f"{'✅' if wrist_deviation < 1e-5 else '⚠️ (normalization issue)'}")
    if data.shape[-1] >= 6:
        vel = data[:, :, 3:6]
        print(f"Velocity min/max : {vel.min():.4f} / {vel.max():.4f}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Customise config here before running ──
    CFG.raw_video_dir = "data/raw_videos"
    CFG.output_dir = "data/landmarks"
    CFG.variations_per_video = 12
    CFG.save_velocity = True
    CFG.save_acceleration = True
    CFG.interpolation = "cubic"   # use cubic for 'Z', 'J', 'Hello'

    process_and_augment(CFG)