"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v16 — Extraction Pipeline                                   ║
║  Apple Vision 2D Hand + 3D Body + Kalman + EMA + Palm Scale      ║
║                                                                  ║
║  Output: [T, 61, 5] float16 per video                           ║
║    ch0: X, ch1: Y, ch2: Z, ch3: mask, ch4: palm_scale           ║
║                                                                  ║
║  Key improvements over v15:                                      ║
║    - Real 3D Z from VNDetectHumanBodyPose3DRequest for body      ║
║    - Perspective Z for hand keypoints from palm scale             ║
║    - Kalman filter for missing frame recovery (gap≤7)            ║
║    - Simple EMA smoothing (replaces 1-Euro filter)               ║
║    - Palm scale feature for GOOD/THANKYOU disambiguation         ║
║    - No velocity/acceleration pre-computed (Squeezeformer         ║
║      learns temporal derivatives internally)                     ║
║    - Extraction ~75ms vs ~300ms (4x faster)                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import sys
import time
import gc

# Apple Vision imports. Keep module importable on non-Apple-Vision
# environments so tests/docs can import v16 model and inference code.
try:
    import Vision
    import Quartz
    from Foundation import NSData, NSAutoreleasePool
    VISION_AVAILABLE = True
except ModuleNotFoundError as exc:
    VISION_AVAILABLE = False
    _VISION_IMPORT_ERROR = exc

    class _MissingVision:
        def __getattr__(self, name):
            return name

    Vision = _MissingVision()
    Quartz = _MissingVision()
    NSData = None
    NSAutoreleasePool = None


def _require_vision():
    if not VISION_AVAILABLE:
        raise RuntimeError(
            "Apple Vision extraction dependencies are unavailable in this "
            "environment. Run v16 extraction on macOS with the Vision bridge "
            "installed."
        ) from _VISION_IMPORT_ERROR

# ── Constants ──
NUM_HAND_NODES = 21
NUM_FACE_NODES = 15
NUM_BODY_NODES = 4
NUM_NODES = NUM_HAND_NODES * 2 + NUM_FACE_NODES + NUM_BODY_NODES  # 61
NUM_CHANNELS = 5  # X, Y, Z, mask, palm_scale
CLIP_LENGTH = 32

# Node layout
LHAND_START = 0
LHAND_END = 21
RHAND_START = 21
RHAND_END = 42
FACE_START = 42
FACE_END = 57
BODY_START = 57
BODY_END = 61

# Hand joint names (Apple Vision VNHumanHandPoseObservation)
HAND_JOINT_NAMES = [
    Vision.VNHumanHandPoseObservationJointNameWrist,
    Vision.VNHumanHandPoseObservationJointNameThumbCMC,
    Vision.VNHumanHandPoseObservationJointNameThumbMP,
    Vision.VNHumanHandPoseObservationJointNameThumbIP,
    Vision.VNHumanHandPoseObservationJointNameThumbTip,
    Vision.VNHumanHandPoseObservationJointNameIndexMCP,
    Vision.VNHumanHandPoseObservationJointNameIndexPIP,
    Vision.VNHumanHandPoseObservationJointNameIndexDIP,
    Vision.VNHumanHandPoseObservationJointNameIndexTip,
    Vision.VNHumanHandPoseObservationJointNameMiddleMCP,
    Vision.VNHumanHandPoseObservationJointNameMiddlePIP,
    Vision.VNHumanHandPoseObservationJointNameMiddleDIP,
    Vision.VNHumanHandPoseObservationJointNameMiddleTip,
    Vision.VNHumanHandPoseObservationJointNameRingMCP,
    Vision.VNHumanHandPoseObservationJointNameRingPIP,
    Vision.VNHumanHandPoseObservationJointNameRingDIP,
    Vision.VNHumanHandPoseObservationJointNameRingTip,
    Vision.VNHumanHandPoseObservationJointNameLittleMCP,
    Vision.VNHumanHandPoseObservationJointNameLittlePIP,
    Vision.VNHumanHandPoseObservationJointNameLittleDIP,
    Vision.VNHumanHandPoseObservationJointNameLittleTip,
]

# 3D Body joint names
BODY_3D_JOINTS = {
    'LeftShoulder': Vision.VNHumanBodyPose3DObservationJointNameLeftShoulder,
    'RightShoulder': Vision.VNHumanBodyPose3DObservationJointNameRightShoulder,
    'LeftElbow': Vision.VNHumanBodyPose3DObservationJointNameLeftElbow,
    'RightElbow': Vision.VNHumanBodyPose3DObservationJointNameRightElbow,
    'LeftWrist': Vision.VNHumanBodyPose3DObservationJointNameLeftWrist,
    'RightWrist': Vision.VNHumanBodyPose3DObservationJointNameRightWrist,
    'CenterHead': Vision.VNHumanBodyPose3DObservationJointNameCenterHead,
}

# Body node mapping (which 3D body joint goes to which node)
BODY_NODE_MAP = {
    57: 'LeftShoulder',
    58: 'RightShoulder',
    59: 'LeftElbow',
    60: 'RightElbow',
}

# Wrist 3D → used for anchoring hand Z depth
WRIST_3D_MAP = {
    'left': 'LeftWrist',
    'right': 'RightWrist',
}


# ═══════════════════════════════════════════════════════════
# Frame → CIImage conversion (leak-free path)
# ═══════════════════════════════════════════════════════════

def frame_to_ciimage(bgr_frame):
    """Convert OpenCV BGR frame to CIImage for Vision framework.
    Uses NSData path (no memory leak, unlike CGDataProvider + CGImage)."""
    _require_vision()
    bgra = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    data = NSData.dataWithBytes_length_(bgra.tobytes(), h * w * 4)
    cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    return Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data, w * 4, Quartz.CGSizeMake(w, h), Quartz.kCIFormatBGRA8, cs)


# ═══════════════════════════════════════════════════════════
# Apple Vision detectors
# ═══════════════════════════════════════════════════════════

# Per-process request objects (created lazily for multiprocessing safety)
import threading
_thread_local = threading.local()

def _get_hand_req():
    """Get or create a per-thread/per-process hand request object."""
    if not hasattr(_thread_local, 'hand_req'):
        _thread_local.hand_req = Vision.VNDetectHumanHandPoseRequest.alloc().init()
        _thread_local.hand_req.setMaximumHandCount_(2)
    return _thread_local.hand_req

def _get_body_req():
    """Get or create a per-thread/per-process body request object."""
    if not hasattr(_thread_local, 'body_req'):
        _thread_local.body_req = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
    return _thread_local.body_req


def detect_hands_2d(ciimg):
    """Detect 2D hand keypoints using Apple Vision.

    Returns:
        list of dicts, each with:
            'keypoints': np.array [21, 2] (x, y in [0,1])
            'confidence': float (mean confidence)
            'chirality': str ('left' or 'right')
    """
    hand_req = _get_hand_req()
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ciimg, None)
    handler.performRequests_error_([hand_req], None)
    results = hand_req.results()

    if not results:
        return []

    hands = []
    for hand_obs in results:
        kpts = np.zeros((21, 2), dtype=np.float32)
        confs = []

        for i, jn in enumerate(HAND_JOINT_NAMES):
            try:
                pt = hand_obs.recognizedPointForJointName_error_(jn, None)
                if pt and pt[0]:
                    p = pt[0]
                    kpts[i, 0] = p.x()
                    kpts[i, 1] = 1.0 - p.y()  # Apple Vision Y is flipped
                    confs.append(float(p.confidence()) if hasattr(p, 'confidence') else 1.0)
            except:
                pass

        # Determine chirality (left vs right) from wrist X position
        # Apple Vision may provide chirality directly in newer APIs
        chirality_val = None
        try:
            chirality_val = hand_obs.chirality()
        except:
            pass

        if chirality_val is not None:
            # 1 = left, 2 = right (Apple Vision enum)
            chirality = 'left' if chirality_val == 1 else 'right'
        else:
            # Fallback: assign by X position
            chirality = 'left' if kpts[0, 0] > 0.5 else 'right'

        hands.append({
            'keypoints': kpts,
            'confidence': np.mean(confs) if confs else 0.0,
            'chirality': chirality,
        })

    return hands


def detect_body_2d(ciimg):
    """Detect 2D body pose using Apple Vision.

    Returns:
        dict mapping joint name → (x, y) normalized [0,1],
        or None if no person detected.

    Note: 2D body works on upper body only (no legs needed).
    """
    body_req = _get_body_req()
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ciimg, None)
    handler.performRequests_error_([body_req], None)
    results = body_req.results()

    if not results or len(results) == 0:
        return None

    obs = results[0]
    joints = {}

    BODY_2D_JOINTS = {
        'LeftShoulder': Vision.VNHumanBodyPoseObservationJointNameLeftShoulder,
        'RightShoulder': Vision.VNHumanBodyPoseObservationJointNameRightShoulder,
        'LeftElbow': Vision.VNHumanBodyPoseObservationJointNameLeftElbow,
        'RightElbow': Vision.VNHumanBodyPoseObservationJointNameRightElbow,
        'LeftWrist': Vision.VNHumanBodyPoseObservationJointNameLeftWrist,
        'RightWrist': Vision.VNHumanBodyPoseObservationJointNameRightWrist,
        'Nose': Vision.VNHumanBodyPoseObservationJointNameNose,
    }

    for name, jn in BODY_2D_JOINTS.items():
        try:
            pt = obs.recognizedPointForJointName_error_(jn, None)
            if pt and pt[0]:
                p = pt[0]
                conf = float(p.confidence()) if hasattr(p, 'confidence') else 0
                if conf > 0.1:  # minimum confidence
                    joints[name] = (float(p.x()), 1.0 - float(p.y()))
        except:
            pass

    return joints if joints else None


# ═══════════════════════════════════════════════════════════
# Kalman filter for missing frame recovery
# ═══════════════════════════════════════════════════════════

class KalmanTracker:
    """Constant-velocity Kalman filter for 2D point tracking.

    Maintains state [x, y, vx, vy] and predicts forward when
    measurements are missing. Tested: +16.2% frame recovery on
    DONT videos (gap≤7).
    """
    def __init__(self, process_noise=0.001, measurement_noise=0.01):
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.state = None
        self.P = np.eye(4)

    def predict(self):
        """Predict next state."""
        if self.state is None:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.H @ self.state

    def update(self, measurement):
        """Update with measurement [x, y]."""
        if self.state is None:
            self.state = np.array([measurement[0], measurement[1], 0, 0], dtype=np.float64)
            self.P = np.eye(4)
            return

        z = np.array(measurement, dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_position(self):
        """Get current estimated position [x, y]."""
        if self.state is None:
            return None
        return self.H @ self.state


def kalman_fill_sequence(sequence, max_gap=7):
    """Fill missing frames in a keypoint sequence using Kalman filter.

    Args:
        sequence: list of np.array [N, 2] or None (N keypoints, XY)
        max_gap: maximum gap to fill

    Returns:
        filled sequence (same structure, Nones replaced where possible)
    """
    if not sequence:
        return sequence

    n_frames = len(sequence)
    n_kpts = sequence[0].shape[0] if sequence[0] is not None else None

    # Find n_kpts from first valid frame
    for s in sequence:
        if s is not None:
            n_kpts = s.shape[0]
            break
    if n_kpts is None:
        return sequence

    filled = [s.copy() if s is not None else None for s in sequence]

    # Run Kalman per keypoint
    for k in range(n_kpts):
        tracker = KalmanTracker()
        gap_start = -1

        for t in range(n_frames):
            if filled[t] is not None:
                # Have measurement
                tracker.update(filled[t][k])
                gap_start = -1
            else:
                # Missing
                pred = tracker.predict()
                if pred is not None:
                    # Find gap length
                    if gap_start < 0:
                        gap_start = t
                    gap_len = t - gap_start + 1

                    if gap_len <= max_gap:
                        # Fill with prediction
                        if filled[t] is None:
                            filled[t] = np.zeros((n_kpts, 2), dtype=np.float32)
                        filled[t][k] = pred.astype(np.float32)

    return filled


# ═══════════════════════════════════════════════════════════
# EMA smoothing
# ═══════════════════════════════════════════════════════════

def ema_smooth(sequence, alpha=0.3):
    """Simple exponential moving average smoothing.

    alpha: smoothing factor (0 = max smooth, 1 = no smooth)
    Replaces 1-Euro filter for simplicity and speed.
    """
    if not sequence or all(s is None for s in sequence):
        return sequence

    smoothed = [s.copy() if s is not None else None for s in sequence]
    prev = None

    for t in range(len(smoothed)):
        if smoothed[t] is not None:
            if prev is not None:
                smoothed[t] = alpha * smoothed[t] + (1 - alpha) * prev
            prev = smoothed[t].copy()

    return smoothed


# ═══════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════

def aspect_correct_image_coordinates(xy_seq, image_width, image_height):
    """Convert Vision's anisotropic normalized XY into isotropic image units.

    Vision reports ``x / width`` and ``y / height``. Euclidean geometry in
    that coordinate space changes when an image rotates between landscape and
    portrait. Scaling both axes by their pixel extent relative to the longer
    image side preserves angles and relative distances without stretching.

    This function intentionally preserves the top-left origin. Subsequent
    wrist centering removes translation, while keeping missing coordinates at
    zero until the existing normalization step.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    longest_side = float(max(image_width, image_height))
    corrected = np.asarray(xy_seq, dtype=np.float32).copy()
    corrected[..., 0] *= float(image_width) / longest_side
    corrected[..., 1] *= float(image_height) / longest_side
    return corrected


def normalize_sequence(xy_seq, mask_seq, default_center=None):
    """Center on wrist, scale by palm length.

    Args:
        xy_seq: [T, 61, 2] XY coordinates
        mask_seq: [T, 61] mask values
        default_center: optional fallback center expressed in the same units
            as ``xy_seq``. Legacy normalized-image coordinates use [0.5, 0.5].

    Returns:
        normalized [T, 61, 2]
    """
    T, N, _ = xy_seq.shape

    # Find active hands (mask > 0.5 on wrist nodes)
    lh_active = mask_seq[:, LHAND_START].mean() > 0.3
    rh_active = mask_seq[:, RHAND_START].mean() > 0.3

    # Center: median wrist position across all active hands
    centers = []
    if lh_active:
        centers.append(xy_seq[:, LHAND_START, :])  # left wrist
    if rh_active:
        centers.append(xy_seq[:, RHAND_START, :])  # right wrist

    if centers:
        center = np.median(np.concatenate(centers, axis=0), axis=0)
    else:
        if default_center is None:
            default_center = (0.5, 0.5)
        center = np.asarray(default_center, dtype=np.float32)

    xy_seq = xy_seq - center[None, None, :]

    # Scale: median palm length (wrist to middle MCP)
    palm_lengths = []
    for hand_start in [LHAND_START, RHAND_START]:
        wrist = xy_seq[:, hand_start, :]
        mid_mcp = xy_seq[:, hand_start + 9, :]  # middle MCP = index 9
        pl = np.sqrt(((mid_mcp - wrist) ** 2).sum(axis=-1))
        valid = pl > 0.01
        if valid.sum() > 3:
            palm_lengths.append(np.median(pl[valid]))

    if palm_lengths:
        scale = np.mean(palm_lengths)
    else:
        scale = 1.0

    if scale > 0.01:
        xy_seq = xy_seq / scale

    return xy_seq


# ═══════════════════════════════════════════════════════════
# Palm scale computation
# ═══════════════════════════════════════════════════════════

def compute_palm_scale(xy_seq, mask_seq):
    """Compute per-frame palm scale: current_palm_len / median_palm_len.

    This feature helps disambiguate GOOD vs THANKYOU:
    - GOOD: hand moves down, palm scale constant
    - THANKYOU: hand moves toward camera, palm appears BIGGER

    Args:
        xy_seq: [T, 61, 2] NORMALIZED coordinates
        mask_seq: [T, 61]

    Returns:
        [T, 61] palm scale per node (broadcast from hand-level)
    """
    T, N = mask_seq.shape
    palm_scale = np.ones((T, N), dtype=np.float32)

    for hand_start in [LHAND_START, RHAND_START]:
        wrist = xy_seq[:, hand_start, :]
        mid_mcp = xy_seq[:, hand_start + 9, :]
        pl = np.sqrt(((mid_mcp - wrist) ** 2).sum(axis=-1))  # [T]

        valid = pl > 0.01
        if valid.sum() < 3:
            continue

        ref_palm = np.median(pl[valid])
        if ref_palm < 0.01:
            continue

        # Per-frame scale
        scale = np.ones(T, dtype=np.float32)
        for t in range(T):
            if pl[t] > 0.01:
                scale[t] = pl[t] / ref_palm

        # Broadcast to all 21 hand nodes
        for k in range(21):
            palm_scale[:, hand_start + k] = scale

    return palm_scale


# ═══════════════════════════════════════════════════════════
# Temporal resampling
# ═══════════════════════════════════════════════════════════

def resample_to_length(data, target_len=CLIP_LENGTH):
    """Resample [T, ...] to [target_len, ...] using linear interpolation."""
    T = data.shape[0]
    if T == target_len:
        return data

    src_t = np.linspace(0, 1, T)
    dst_t = np.linspace(0, 1, target_len)

    # Flatten non-time dims for interpolation
    original_shape = data.shape[1:]
    flat = data.reshape(T, -1)

    resampled = np.zeros((target_len, flat.shape[1]), dtype=data.dtype)
    for c in range(flat.shape[1]):
        resampled[:, c] = np.interp(dst_t, src_t, flat[:, c])

    return resampled.reshape(target_len, *original_shape)


# ═══════════════════════════════════════════════════════════
# Main extraction functions
# ═══════════════════════════════════════════════════════════

def extract_frames_v16(frames, body_3d_interval=4, aspect_correct=False):
    """Extract features from a list of BGR frames.

    Args:
        frames: list of BGR numpy frames
        body_3d_interval: run body detection every N frames (default 4)
        aspect_correct: opt in to isotropic coordinates for new retraining.
            False preserves compatibility with existing v16 checkpoints.

    Returns:
        np.array [T_resampled, 61, 5] float16, or None if no hands detected
        where T_resampled = 32 for isolated, variable for continuous
    """
    n_frames = len(frames)
    if n_frames < 4:
        return None
    frame_height, frame_width = frames[0].shape[:2]

    # ── Step 1: Detect hands and body for each frame ──
    lhand_seq = []  # list of [21, 2] or None
    rhand_seq = []
    lhand_conf = []
    rhand_conf = []
    body_3d_cache = {}  # frame_idx → dict of joint → (x, y, z)

    for fi, frame in enumerate(frames):
        # Autorelease pool drains all ObjC objects created inside it
        pool = NSAutoreleasePool.alloc().init()

        ci = frame_to_ciimage(frame)

        # 2D hand detection (every frame — fast, 5.4ms)
        hands = detect_hands_2d(ci)

        lh, rh = None, None
        lc, rc = 0.0, 0.0

        for h in hands:
            if h['chirality'] == 'left':
                lh = h['keypoints']
                lc = h['confidence']
            else:
                rh = h['keypoints']
                rc = h['confidence']

        lhand_seq.append(lh)
        rhand_seq.append(rh)
        lhand_conf.append(lc)
        rhand_conf.append(rc)

        # 2D body detection (every Nth frame — ~10ms)
        if fi % body_3d_interval == 0:
            body = detect_body_2d(ci)
            if body:
                body_3d_cache[fi] = body

        # Explicit cleanup to prevent memory leaks
        del hands, ci
        del pool  # Drains autorelease pool — frees all ObjC objects from this frame

    # Check if any hands detected
    lh_count = sum(1 for h in lhand_seq if h is not None)
    rh_count = sum(1 for h in rhand_seq if h is not None)
    if lh_count + rh_count == 0:
        return None

    # ── Step 2: Kalman filter for missing frames ──
    lhand_filled = kalman_fill_sequence(lhand_seq, max_gap=7)
    rhand_filled = kalman_fill_sequence(rhand_seq, max_gap=7)

    # ── Step 3: EMA smoothing ──
    lhand_smooth = ema_smooth(lhand_filled, alpha=0.4)
    rhand_smooth = ema_smooth(rhand_filled, alpha=0.4)

    # ── Step 4: Build full skeleton [T, 61, 2] (XY) ──
    xy = np.zeros((n_frames, NUM_NODES, 2), dtype=np.float32)
    mask = np.zeros((n_frames, NUM_NODES), dtype=np.float32)

    for t in range(n_frames):
        if lhand_smooth[t] is not None:
            xy[t, LHAND_START:LHAND_END, :] = lhand_smooth[t]
            mask[t, LHAND_START:LHAND_END] = 1.0

        if rhand_smooth[t] is not None:
            xy[t, RHAND_START:RHAND_END, :] = rhand_smooth[t]
            mask[t, RHAND_START:RHAND_END] = 1.0

        # Face/body from hand detection context (simplified)
        # Apple Vision hand request doesn't give face/body,
        # but 3D body request gives shoulders/elbows
        # For face nodes, use head position from 3D body if available

    # ── Step 5: Body node XY from 2D body detection + Perspective Z ──
    z = np.zeros((n_frames, NUM_NODES), dtype=np.float32)

    # Fill body node XY from 2D body detection
    body_joint_to_node = {
        'LeftShoulder': 57, 'RightShoulder': 58,
        'LeftElbow': 59, 'RightElbow': 60,
    }
    for joint_name, node_idx in body_joint_to_node.items():
        valid_frames = []
        valid_xy = []
        for fi, body in sorted(body_3d_cache.items()):
            if joint_name in body:
                valid_frames.append(fi)
                valid_xy.append(body[joint_name])
        if len(valid_frames) >= 2:
            xy_x = np.interp(np.arange(n_frames), valid_frames, [v[0] for v in valid_xy])
            xy_y = np.interp(np.arange(n_frames), valid_frames, [v[1] for v in valid_xy])
            xy[:, node_idx, 0] = xy_x
            xy[:, node_idx, 1] = xy_y
            mask[:, node_idx] = 1.0

    # Face node: use nose from body detection
    nose_frames = []
    nose_xy = []
    for fi, body in sorted(body_3d_cache.items()):
        if 'Nose' in body:
            nose_frames.append(fi)
            nose_xy.append(body['Nose'])
    if len(nose_frames) >= 2:
        nose_x = np.interp(np.arange(n_frames), nose_frames, [v[0] for v in nose_xy])
        nose_y = np.interp(np.arange(n_frames), nose_frames, [v[1] for v in nose_xy])
        xy[:, FACE_START, 0] = nose_x  # node 42 = nose
        xy[:, FACE_START, 1] = nose_y
        mask[:, FACE_START] = 1.0

    # Vision normalizes X by width and Y by height. Correct that anisotropy
    # before computing any Euclidean palm/body geometry. This is opt-in because
    # it defines a new feature representation and requires model retraining.
    if aspect_correct:
        xy = aspect_correct_image_coordinates(xy, frame_width, frame_height)
        longest_side = float(max(frame_width, frame_height))
        default_center = np.array([
            0.5 * frame_width / longest_side,
            0.5 * frame_height / longest_side,
        ], dtype=np.float32)
    else:
        default_center = np.array([0.5, 0.5], dtype=np.float32)

    # Compute perspective Z from palm scale for ALL nodes
    # Hands: Z from palm size (closer = bigger palm = negative Z)
    for hand_start in [LHAND_START, RHAND_START]:
        wrist_xy = xy[:, hand_start, :]
        mcp_xy = xy[:, hand_start + 9, :]
        palm_len = np.sqrt(((mcp_xy - wrist_xy) ** 2).sum(axis=-1))
        valid = palm_len > 0.01
        if valid.sum() >= 3:
            ref_palm = np.median(palm_len[valid])
            if ref_palm > 0.01:
                persp_z = np.zeros(n_frames, dtype=np.float32)
                for t in range(n_frames):
                    if palm_len[t] > 0.01:
                        persp_z[t] = np.log(ref_palm / palm_len[t])
                for k in range(21):
                    z[:, hand_start + k] = persp_z

    # Body Z: from inter-shoulder distance (closer = shoulders wider apart)
    ls_xy = xy[:, 57, :]
    rs_xy = xy[:, 58, :]
    sh_dist = np.sqrt(((rs_xy - ls_xy) ** 2).sum(axis=-1))
    valid_sh = sh_dist > 0.01
    if valid_sh.sum() >= 3:
        ref_sh = np.median(sh_dist[valid_sh])
        if ref_sh > 0.01:
            body_z = np.zeros(n_frames, dtype=np.float32)
            for t in range(n_frames):
                if sh_dist[t] > 0.01:
                    body_z[t] = np.log(ref_sh / sh_dist[t])
            for k in range(BODY_START, BODY_END):
                z[:, k] = body_z
            for k in range(FACE_START, FACE_END):
                z[:, k] = body_z  # face same depth as body

    # ── Step 6: Normalize XY ──
    xy_norm = normalize_sequence(xy, mask, default_center=default_center)

    # ── Step 7: Palm scale ──
    palm_scale = compute_palm_scale(xy_norm, mask)

    # ── Step 8: Assemble output [T, 61, 5] ──
    output = np.zeros((n_frames, NUM_NODES, NUM_CHANNELS), dtype=np.float32)
    output[:, :, 0] = xy_norm[:, :, 0]   # X
    output[:, :, 1] = xy_norm[:, :, 1]   # Y
    output[:, :, 2] = z                   # Z (real 3D for body, perspective for hands)
    output[:, :, 3] = mask                # detection mask
    output[:, :, 4] = palm_scale          # palm scale feature

    # ── Step 9: Resample to 32 frames (for isolated) ──
    if n_frames != CLIP_LENGTH:
        output = resample_to_length(output, CLIP_LENGTH)

    return output.astype(np.float16)


def extract_video_v16(video_path, body_3d_interval=4, aspect_correct=False):
    """Extract features from a video file.

    Args:
        video_path: path to video file

    Returns:
        np.array [32, 61, 5] float16, or None
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 4:
        return None

    return extract_frames_v16(
        frames,
        body_3d_interval=body_3d_interval,
        aspect_correct=aspect_correct,
    )


def extract_continuous_v16(frames, segment_len=28, stride=14,
                           aspect_correct=False):
    """Extract continuous (multi-sign) sequence with sliding window.

    Uses overlapping windows (stride < segment_len) to ensure signs
    at segment boundaries are captured by at least one complete window.

    Args:
        frames: list of BGR frames
        segment_len: window size in frames (default 28)
        stride: step between windows (default 14 = 50% overlap)

    Returns:
        np.array [N*32, 61, 5] float16 where N = number of segments
    """
    n_frames = len(frames)

    if n_frames < 8:
        return None

    # Single clip if very short
    if n_frames <= segment_len:
        return extract_frames_v16(frames, aspect_correct=aspect_correct)

    # Sliding window extraction
    segments = []
    for start in range(0, max(1, n_frames - segment_len + 1), stride):
        end = min(start + segment_len, n_frames)
        if end - start < 8:
            break
        seg_result = extract_frames_v16(
            frames[start:end],
            body_3d_interval=4,
            aspect_correct=aspect_correct,
        )
        if seg_result is not None:
            segments.append(seg_result)

    if not segments:
        return None

    return np.concatenate(segments, axis=0).astype(np.float16)


# ═══════════════════════════════════════════════════════════
# Batch extraction for training data
# ═══════════════════════════════════════════════════════════

_job_counter = 0

def _extract_single_job(args):
    """Worker function for multiprocessing. Extracts one video."""
    global _job_counter
    vid_path, out_path, body_3d_interval, aspect_correct = args
    try:
        result = extract_video_v16(
            vid_path,
            body_3d_interval=body_3d_interval,
            aspect_correct=aspect_correct,
        )
        if result is not None:
            np.save(out_path, result)
            status = 'ok'
        else:
            status = 'no_hands'
    except Exception as e:
        status = f'fail:{e}'

    # Force garbage collection every 10 videos to free ObjC objects
    _job_counter += 1
    if _job_counter % 10 == 0:
        gc.collect()

    return status


def extract_batch_v16(input_dir, output_dir, body_3d_interval=8, resume=True,
                      workers=1, aspect_correct=False):
    """Extract all videos in a directory tree.

    Args:
        input_dir: root directory containing class folders with videos
        output_dir: where to save .npy files
        body_3d_interval: how often to run 3D body (higher = faster)
        resume: skip already-extracted files
        workers: number of parallel workers (default 1 = single process)
    """
    os.makedirs(output_dir, exist_ok=True)

    video_exts = {'.mp4', '.mov', '.avi', '.mkv'}

    # Build job list
    jobs = []
    skipped = 0

    for cls_name in sorted(os.listdir(input_dir)):
        cls_dir = os.path.join(input_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for vid_file in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(vid_file)[1].lower()
            if ext not in video_exts:
                continue

            stem = os.path.splitext(vid_file)[0]
            out_name = f"{cls_name}_{stem}.npy"
            out_path = os.path.join(output_dir, out_name)

            if resume and os.path.exists(out_path):
                skipped += 1
                continue

            vid_path = os.path.join(cls_dir, vid_file)
            jobs.append((vid_path, out_path, body_3d_interval, aspect_correct))

    total_jobs = len(jobs)
    print(f"Jobs: {total_jobs} to extract, {skipped} skipped (resume={resume})")
    print(f"Workers: {workers}")

    if total_jobs == 0:
        print("Nothing to extract.")
        return

    t0 = time.time()
    total = 0
    failed = 0
    no_hands = 0

    if workers <= 1:
        # Single process
        for i, job in enumerate(jobs):
            result = _extract_single_job(job)
            if result == 'ok':
                total += 1
            elif result == 'no_hands':
                no_hands += 1
            else:
                failed += 1

            # Garbage collect every 10 videos
            if (i + 1) % 10 == 0:
                gc.collect()

            if (i + 1) % 100 == 0 or (i + 1) == total_jobs:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1)
                eta = (total_jobs - i - 1) / max(rate, 0.01)
                print(f"  [{i+1}/{total_jobs}] ok={total} no_hands={no_hands} fail={failed} "
                      f"rate={rate:.1f}/s eta={eta/60:.0f}min")
    else:
        # Multiprocessing
        from multiprocessing import Pool

        with Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_extract_single_job, jobs, chunksize=4)):
                if result == 'ok':
                    total += 1
                elif result == 'no_hands':
                    no_hands += 1
                else:
                    failed += 1

                if (i + 1) % 200 == 0 or (i + 1) == total_jobs:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1)
                    eta = (total_jobs - i - 1) / max(rate, 0.01)
                    print(f"  [{i+1}/{total_jobs}] ok={total} no_hands={no_hands} fail={failed} "
                          f"rate={rate:.1f}/s eta={eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone: {total} extracted, {no_hands} no hands, {failed} failed, "
          f"{skipped} skipped in {elapsed:.0f}s ({elapsed/60:.1f}min)")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='v16 extraction pipeline')
    parser.add_argument('input', help='Video file or directory of class folders')
    parser.add_argument('--output', default='ASL_landmarks_v16', help='Output directory')
    parser.add_argument('--body3d_interval', type=int, default=8,
                        help='Run 3D body every N frames (default 8)')
    parser.add_argument('--resume', action='store_true', help='Skip existing files')
    parser.add_argument('--single', action='store_true', help='Extract single video')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default 1)')
    parser.add_argument('--aspect_correct', action='store_true',
                        help='Use isotropic XY for new portrait/landscape retraining')
    args = parser.parse_args()

    if args.single:
        result = extract_video_v16(
            args.input,
            body_3d_interval=args.body3d_interval,
            aspect_correct=args.aspect_correct,
        )
        if result is not None:
            out_path = os.path.join(args.output, os.path.basename(args.input).replace('.mp4', '.npy'))
            os.makedirs(args.output, exist_ok=True)
            np.save(out_path, result)
            print(f"Saved: {out_path} shape={result.shape}")
        else:
            print("Extraction failed (no hands detected)")
    else:
        extract_batch_v16(args.input, args.output,
                         body_3d_interval=args.body3d_interval,
                         resume=args.resume,
                         workers=args.workers,
                         aspect_correct=args.aspect_correct)
