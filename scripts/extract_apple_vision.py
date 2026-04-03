"""
Apple Vision extraction for SLT — Mac M4 Neural Engine.
Uses native macOS Vision framework for hand + face detection.
Body (shoulders/elbows) estimated from face ear positions.
Parallelized via subprocess.Popen (multiprocessing.Pool hangs with Vision).

Output: ASL_landmarks_apple_vision/ with [32, 61, 10] .npy files + manifest.json
Also saves hand crops (128x128) for future hybrid CNN training.

Usage:
    python scripts/extract_apple_vision.py --workers 8
    python scripts/extract_apple_vision.py --workers 8 --resume
    python scripts/extract_apple_vision.py --mode phrases --workers 4
"""
import os, sys, glob, hashlib, warnings, time, argparse, json, subprocess
warnings.filterwarnings('ignore')
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_batch_fast_v2 import (
    interpolate_generic, normalize_sequence_v2, compute_kinematics_v2,
    NUM_NODES_V2, NUM_FACE_V2, NUM_BODY_V2,
)
from extract import (
    LABEL_ALIASES,
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)

import Vision
import Quartz
from Foundation import NSAutoreleasePool, NSData

# ============================================================
# Apple Vision detection
# ============================================================

HAND_JOINT_NAMES = [
    'wrist',
    'thumbCMC', 'thumbMP', 'thumbIP', 'thumbTip',
    'indexMCP', 'indexPIP', 'indexDIP', 'indexTip',
    'middleMCP', 'middlePIP', 'middleDIP', 'middleTip',
    'ringMCP', 'ringPIP', 'ringDIP', 'ringTip',
    'littleMCP', 'littlePIP', 'littleDIP', 'littleTip',
]

JOINT_KEYS = {
    'wrist': Vision.VNHumanHandPoseObservationJointNameWrist,
    'thumbCMC': Vision.VNHumanHandPoseObservationJointNameThumbCMC,
    'thumbMP': Vision.VNHumanHandPoseObservationJointNameThumbMP,
    'thumbIP': Vision.VNHumanHandPoseObservationJointNameThumbIP,
    'thumbTip': Vision.VNHumanHandPoseObservationJointNameThumbTip,
    'indexMCP': Vision.VNHumanHandPoseObservationJointNameIndexMCP,
    'indexPIP': Vision.VNHumanHandPoseObservationJointNameIndexPIP,
    'indexDIP': Vision.VNHumanHandPoseObservationJointNameIndexDIP,
    'indexTip': Vision.VNHumanHandPoseObservationJointNameIndexTip,
    'middleMCP': Vision.VNHumanHandPoseObservationJointNameMiddleMCP,
    'middlePIP': Vision.VNHumanHandPoseObservationJointNameMiddlePIP,
    'middleDIP': Vision.VNHumanHandPoseObservationJointNameMiddleDIP,
    'middleTip': Vision.VNHumanHandPoseObservationJointNameMiddleTip,
    'ringMCP': Vision.VNHumanHandPoseObservationJointNameRingMCP,
    'ringPIP': Vision.VNHumanHandPoseObservationJointNameRingPIP,
    'ringDIP': Vision.VNHumanHandPoseObservationJointNameRingDIP,
    'ringTip': Vision.VNHumanHandPoseObservationJointNameRingTip,
    'littleMCP': Vision.VNHumanHandPoseObservationJointNameLittleMCP,
    'littlePIP': Vision.VNHumanHandPoseObservationJointNameLittlePIP,
    'littleDIP': Vision.VNHumanHandPoseObservationJointNameLittleDIP,
    'littleTip': Vision.VNHumanHandPoseObservationJointNameLittleTip,
}

# Shoulder estimation constants (learned from RTMW training data)
_SHOULDER_L_OFFSET = np.array([0.885, 1.650, 0.0], dtype=np.float32)
_SHOULDER_R_OFFSET = np.array([-0.850, 1.685, 0.0], dtype=np.float32)
_ELBOW_L_OFFSET = np.array([1.52, 2.48, 0.0], dtype=np.float32)
_ELBOW_R_OFFSET = np.array([-0.97, 2.44, 0.0], dtype=np.float32)


def frame_to_ciimage(bgr_frame):
    """Convert OpenCV BGR frame to CIImage for Vision framework.
    Uses NSData path (no memory leak, unlike CGDataProvider)."""
    bgra = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    data = NSData.dataWithBytes_length_(bgra.tobytes(), h * w * 4)
    cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    return Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data, w * 4, Quartz.CGSizeMake(w, h), Quartz.kCIFormatBGRA8, cs)


def assign_hand_slots(hands_list, face_pts):
    """Assign all detected hands to left/right slots consistently.

    Rules:
    - 2 hands: use relative position (leftmost → left slot, rightmost → right slot)
    - 1 hand: use Apple Vision chirality as hint, but lock to one slot for the frame
    - Uses face center only when chirality is unknown

    Returns list of (slot, coords, conf) where slot is 'left' or 'right'.
    """
    valid = [(ch, coords, conf) for ch, coords, conf in hands_list if conf >= 0.25]
    if not valid:
        return []

    if len(valid) == 2:
        # Two hands: leftmost wrist (higher x in camera = signer's left) → left slot
        # Sort by wrist x descending (camera-right first = signer's left)
        sorted_hands = sorted(valid, key=lambda h: -h[1][0, 0])
        return [('left', sorted_hands[0][1], sorted_hands[0][2]),
                ('right', sorted_hands[1][1], sorted_hands[1][2])]

    # One hand: use chirality label (it's reliable when only 1 hand is detected)
    ch, coords, conf = valid[0]
    if ch == -1:
        return [('left', coords, conf)]    # signer's left
    elif ch == 1:
        return [('right', coords, conf)]   # signer's right
    else:
        # Unknown: use face position
        wrist_x = coords[0, 0]
        if face_pts is not None and len(face_pts) >= 5:
            face_cx = (face_pts[3, 0] + face_pts[4, 0]) / 2
        else:
            face_cx = 0.5
        slot = 'left' if wrist_x > face_cx else 'right'
        return [(slot, coords, conf)]


def estimate_body_from_face(face_pts):
    """Estimate shoulder/elbow positions from face ear landmarks.
    face_pts: [15, 2] array. Nodes 3=left_ear, 4=right_ear.
    Returns [4, 3] array or None."""
    if face_pts is None or len(face_pts) < 5:
        return None
    l_ear = face_pts[3]
    r_ear = face_pts[4]
    ear_mid = (l_ear + r_ear) / 2
    ear_dist = np.linalg.norm(r_ear - l_ear)
    if ear_dist < 1e-6:
        return None
    points = []
    for offset in [_SHOULDER_L_OFFSET, _SHOULDER_R_OFFSET, _ELBOW_L_OFFSET, _ELBOW_R_OFFSET]:
        pt = np.zeros(3, dtype=np.float32)
        pt[0] = ear_mid[0] + offset[0] * ear_dist
        pt[1] = ear_mid[1] + offset[1] * ear_dist
        points.append(pt)
    return np.array(points, dtype=np.float32)


def detect_all(ci_image, frame_w, frame_h):
    """Run hands + face in a single handler. Body estimated from face ears.
    Returns (hands_list, face_15x2, body_4x3)."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    hand_req = Vision.VNDetectHumanHandPoseRequest.alloc().init()
    hand_req.setMaximumHandCount_(2)
    face_req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    handler.performRequests_error_([hand_req, face_req], None)

    # Parse hands
    hands = []
    hand_results = hand_req.results()
    if hand_results:
        for obs in hand_results:
            chirality = obs.chirality()
            points = []
            confs = []
            for jname in HAND_JOINT_NAMES:
                key = JOINT_KEYS[jname]
                pt, err = obs.recognizedPointForJointName_error_(key, None)
                if pt and pt.confidence() > 0.01:
                    points.append([pt.location().x, 1.0 - pt.location().y])
                    confs.append(pt.confidence())
                else:
                    points.append([0.0, 0.0])
                    confs.append(0.0)
            hands.append((chirality, np.array(points, dtype=np.float32), np.mean(confs)))

    # Parse face
    face_pts = None
    face_results = face_req.results()
    if face_results:
        face = face_results[0]
        landmarks = face.landmarks()
        if landmarks:
            img_size = Quartz.CGSizeMake(frame_w, frame_h)

            def _pt(region, idx):
                if region is None or idx >= region.pointCount():
                    return None
                pts = region.pointsInImageOfSize_(img_size)
                p = pts[idx]
                return [p.x / frame_w, 1.0 - p.y / frame_h]

            nose = landmarks.nose()
            fc = landmarks.faceContour()
            ol = landmarks.outerLips()
            il = landmarks.innerLips()
            leb = landmarks.leftEyebrow()
            reb = landmarks.rightEyebrow()
            le = landmarks.leftEye()
            re = landmarks.rightEye()
            ml = landmarks.medianLine()

            fp = [
                _pt(nose, nose.pointCount()//2) if nose and nose.pointCount()>0 else None,
                _pt(fc, fc.pointCount()//2) if fc and fc.pointCount()>0 else None,
                _pt(ml, ml.pointCount()-1) if ml and ml.pointCount()>0 else None,
                _pt(fc, 0) if fc and fc.pointCount()>0 else None,
                _pt(fc, fc.pointCount()-1) if fc and fc.pointCount()>0 else None,
                _pt(ol, 0) if ol and ol.pointCount()>0 else None,
                _pt(ol, ol.pointCount()//2) if ol and ol.pointCount()>=6 else None,
                _pt(ol, ol.pointCount()*3//4) if ol and ol.pointCount()>=4 else None,
                _pt(il, il.pointCount()//2) if il and il.pointCount()>=4 else None,
                _pt(leb, leb.pointCount()-1) if leb and leb.pointCount()>0 else None,
                _pt(leb, 0) if leb and leb.pointCount()>0 else None,
                _pt(reb, 0) if reb and reb.pointCount()>0 else None,
                _pt(reb, reb.pointCount()-1) if reb and reb.pointCount()>0 else None,
                _pt(le, le.pointCount()//2) if le and le.pointCount()>0 else None,
                _pt(re, re.pointCount()//2) if re and re.pointCount()>0 else None,
            ]
            if all(p is not None for p in fp):
                face_pts = np.array(fp, dtype=np.float32)

    body_pts = estimate_body_from_face(face_pts) if face_pts is not None else None
    return hands, face_pts, body_pts


# ============================================================
# Extraction pipeline
# ============================================================

def _extract_from_frames(frames, is_phrase=False):
    """Core extraction from a list of BGR frames. No file I/O.
    If is_phrase: returns [N*32, 61, 10] (multi-clip).
    Else: returns [32, 61, 10] (single clip, resampled)."""
    total_frames = len(frames)
    if total_frames < 8:
        return None

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []

    for fi, bgr in enumerate(frames):
        pool = NSAutoreleasePool.alloc().init()
        h, w = bgr.shape[:2]
        ci = frame_to_ciimage(bgr)
        hands, face_pts, body_pts = detect_all(ci, w, h)
        del ci

        for slot, coords, conf in assign_hand_slots(hands, face_pts):
            coords_3d = np.column_stack([coords, np.zeros(21)])
            if slot == 'left':
                if not l_valid or l_valid[-1] != fi:
                    l_seq.append(coords_3d.tolist()); l_valid.append(fi)
            else:
                if not r_valid or r_valid[-1] != fi:
                    r_seq.append(coords_3d.tolist()); r_valid.append(fi)

        if face_pts is not None:
            face_3d = np.column_stack([face_pts, np.zeros(15)])
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_3d.tolist()); face_valid.append(fi)

        if body_pts is not None:
            if not body_valid or body_valid[-1] != fi:
                body_seq.append(body_pts.tolist()); body_valid.append(fi)

        del hands, face_pts, body_pts
        del pool
        fi += 1

    if not l_valid and not r_valid:
        return None
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)
    body_ever = bool(body_valid)

    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
    face_full = interpolate_generic(
        np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
        face_valid, total_frames, NUM_FACE_V2)
    body_full = interpolate_generic(
        np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
        body_valid, total_frames, NUM_BODY_V2)
    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

    if is_phrase:
        # Phrase mode is handled by extract_frames_continuous which calls
        # extract_frames_isolated per segment. This branch shouldn't be reached.
        return None
    else:
        # Single clip: resample to 32 frames
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
        return data


def extract_frames_isolated(frames):
    """Extract from a list of BGR frames → [32, 61, 10] float16."""
    return _extract_from_frames(frames, is_phrase=False)


def extract_frames_continuous(frames):
    """Extract from a list of BGR frames → [N*32, 61, 10] float16.
    Splits video into ~28-frame segments, extracts each independently
    as isolated (matching SyntheticCTCDataset training format)."""
    if len(frames) < 8:
        return None

    seg_size = 28
    clips = []
    for start in range(0, len(frames), seg_size):
        seg = frames[start:start + seg_size]
        if len(seg) < 8:
            continue
        clip = extract_frames_isolated(seg)
        if clip is not None:
            clips.append(clip)

    if not clips:
        return None
    return np.concatenate(clips, axis=0).astype(np.float16)


def extract_one_video(video_path):
    """Extract one video → [32, 61, 10] float16 or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []

    # Process one frame at a time
    fi = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # Autorelease pool drains all ObjC objects created inside it
        pool = NSAutoreleasePool.alloc().init()
        h, w = bgr.shape[:2]
        ci = frame_to_ciimage(bgr)
        del bgr
        hands, face_pts, body_pts = detect_all(ci, w, h)
        del ci

        for slot, coords, conf in assign_hand_slots(hands, face_pts):
            coords_3d = np.column_stack([coords, np.zeros(21)])
            if slot == 'left':
                if not l_valid or l_valid[-1] != fi:
                    l_seq.append(coords_3d.tolist()); l_valid.append(fi)
            else:
                if not r_valid or r_valid[-1] != fi:
                    r_seq.append(coords_3d.tolist()); r_valid.append(fi)

        if face_pts is not None:
            face_3d = np.column_stack([face_pts, np.zeros(15)])
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_3d.tolist()); face_valid.append(fi)

        if body_pts is not None:
            if not body_valid or body_valid[-1] != fi:
                body_seq.append(body_pts.tolist()); body_valid.append(fi)

        del hands, face_pts, body_pts
        del pool  # Drains autorelease pool — frees all ObjC objects from this frame
        fi += 1

    cap.release()
    total_frames = fi

    if total_frames < 8:
        return None
    if not l_valid and not r_valid:
        return None

    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    # No ghost hand duplication — let mask=0 signal "hand not present"
    # Model learns one-handed vs two-handed naturally
    face_ever = bool(face_valid)
    body_ever = bool(body_valid)

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
    return data


def extract_phrase_video(video_path):
    """Extract phrase video → [N*32, 61, 10] float16 (variable-length clips)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []

    fi = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        h, w = bgr.shape[:2]
        ci = frame_to_ciimage(bgr)
        del bgr
        hands, face_pts, body_pts = detect_all(ci, w, h)
        del ci

        for slot, coords, conf in assign_hand_slots(hands, face_pts):
            coords_3d = np.column_stack([coords, np.zeros(21)])
            if slot == 'left':
                if not l_valid or l_valid[-1] != fi:
                    l_seq.append(coords_3d.tolist()); l_valid.append(fi)
            else:
                if not r_valid or r_valid[-1] != fi:
                    r_seq.append(coords_3d.tolist()); r_valid.append(fi)

        if face_pts is not None:
            face_3d = np.column_stack([face_pts, np.zeros(15)])
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_3d.tolist()); face_valid.append(fi)
        if body_pts is not None:
            if not body_valid or body_valid[-1] != fi:
                body_seq.append(body_pts.tolist()); body_valid.append(fi)

        fi += 1

    cap.release()
    total_frames = fi

    if total_frames < 8:
        return None
    if not l_valid and not r_valid:
        return None
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)
    body_ever = bool(body_valid)

    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
    face_full = interpolate_generic(
        np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
        face_valid, total_frames, NUM_FACE_V2)
    body_full = interpolate_generic(
        np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
        body_valid, total_frames, NUM_BODY_V2)
    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

    # Split into 32-frame clips
    T = combined.shape[0]
    num_clips = T // 32
    remainder = T % 32
    if num_clips == 0:
        combined = np.concatenate([combined, np.zeros((32 - T, 61, 3), dtype=np.float32)])
        num_clips = 1
    elif remainder > 0:
        combined = np.concatenate([combined, np.zeros((32 - remainder, 61, 3), dtype=np.float32)])
        num_clips += 1

    clips = []
    for ci in range(num_clips):
        seg = combined[ci * 32:(ci + 1) * 32].copy()
        seg[:, :, :3] = one_euro_filter(seg[:, :, :3])
        if l_ever:
            seg = stabilize_bones(seg, 0, 21)
        if r_ever:
            seg = stabilize_bones(seg, 21, 42)
        normed = normalize_sequence_v2(seg, l_ever, r_ever)
        clips.append(compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever))

    return np.concatenate(clips, axis=0).astype(np.float16)


def save_hand_crop(video_path, crop_dir, bbox_center=None):
    """Save 128x128 hand crop from middle frame."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return
    # Seek to middle frame directly
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, mid = cap.read()
    cap.release()
    if not ret:
        return
    h, w = mid.shape[:2]
    # Use center of frame as person center
    cx, cy = w // 2, h // 2
    crop_size = min(w, h) // 2
    cy1, cy2 = max(0, cy - crop_size), min(h, cy + crop_size)
    cx1, cx2 = max(0, cx - crop_size), min(w, cx + crop_size)
    crop = mid[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return

    ch, cw = crop.shape[:2]
    max_side = max(ch, cw)
    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    y_off = (max_side - ch) // 2
    x_off = (max_side - cw) // 2
    padded[y_off:y_off+ch, x_off:x_off+cw] = crop
    crop = cv2.resize(padded, (128, 128))

    class_name = os.path.basename(os.path.dirname(video_path))
    class_name = LABEL_ALIASES.get(class_name, class_name)
    vid_hash = os.path.splitext(os.path.basename(video_path))[0]
    crop_path = os.path.join(crop_dir, f"{class_name}_{vid_hash}.jpg")
    cv2.imwrite(crop_path, crop)


# ============================================================
# Worker (called via subprocess.Popen)
# ============================================================

def worker_main(task_json_path, output_dir, crop_dir, mode):
    """Process a batch of videos. Called as a subprocess."""
    import gc

    with open(task_json_path) as f:
        tasks = json.load(f)

    results = {}
    for ti, task in enumerate(tasks):
        video_path = task['video_path']
        out_name = task['out_name']
        canonical = task['canonical']

        try:
            if mode == 'phrases':
                data = extract_phrase_video(video_path)
            else:
                data = extract_one_video(video_path)

            if data is not None:
                np.save(os.path.join(output_dir, out_name), data)
                results[out_name] = canonical
                if mode == 'isolated' and crop_dir:
                    save_hand_crop(video_path, crop_dir)
        except Exception:
            pass

        # Force garbage collection every 50 videos to free ObjC objects
        if (ti + 1) % 50 == 0:
            gc.collect()

    # Write results
    result_path = task_json_path.replace('.json', '_result.json')
    with open(result_path, 'w') as f:
        json.dump(results, f)


# ============================================================
# Phrase glosses (for Stage 2 training)
# ============================================================

PHRASE_GLOSSES = {
    "GOOD_MORNING": "GOOD MORNING",
    "HELLO_HOW_YOU": "HELLO HOW YOU",
    "PLEASE_HELP_I": "PLEASE HELP I",
    "PLEASE_HELP_ME": "PLEASE HELP I",
    "SORRY_I_LATE": "SORRY I LATE",
    "MY_NAME": "MY NAME",
    "YESTERDAY_TEACHER_MEET": "YESTERDAY TEACHER MEET",
    "THANKYOU_FRIEND": "THANKYOU FRIEND",
    "TOMORROW_SCHOOL_GO": "TOMORROW SCHOOL GO",
    "I_WANT_FOOD": "I WANT FOOD",
}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Handle --_worker flag (internal, called by subprocess)
    if '--_worker' in sys.argv:
        idx = sys.argv.index('--_worker')
        task_path = sys.argv[idx + 1]
        output_dir = sys.argv[idx + 2]
        crop_dir = sys.argv[idx + 3]
        mode = sys.argv[idx + 4]
        worker_main(task_path, output_dir, crop_dir, mode)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Apple Vision extraction for SLT")
    parser.add_argument("--input", default="data/raw_videos/ASL VIDEOS")
    parser.add_argument("--output", default="ASL_landmarks_apple_vision")
    parser.add_argument("--crops", default="ASL_hand_crops_av")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=["isolated", "phrases"], default="isolated")
    parser.add_argument("--phrase_input", default="data/raw_videos/phrases")
    parser.add_argument("--phrase_output", default="ASL_phrases_apple_vision")
    args = parser.parse_args()

    if args.mode == "phrases":
        # ---- Phrase extraction ----
        os.makedirs(args.phrase_output, exist_ok=True)
        print(f"=== PHRASE EXTRACTION (Apple Vision) ===")
        print(f"Input:  {args.phrase_input}")
        print(f"Output: {args.phrase_output}")

        count = 0
        manifest = {}
        for phrase_name, gloss_str in PHRASE_GLOSSES.items():
            pdir = os.path.join(args.phrase_input, phrase_name)
            if not os.path.isdir(pdir):
                print(f"  Skipping {phrase_name} (not found)")
                continue
            videos = sorted(glob.glob(os.path.join(pdir, "*.mp4")))
            print(f"  {phrase_name}: {len(videos)} videos")

            for vid in videos:
                data = extract_phrase_video(vid)
                if data is None:
                    continue
                vid_hash = os.path.splitext(os.path.basename(vid))[0]
                idx = len([f for f in os.listdir(args.phrase_output) if f.startswith(phrase_name)])
                out_name = f"{phrase_name}_{idx:04d}_{vid_hash[:8]}.npy"
                np.save(os.path.join(args.phrase_output, out_name), data)
                manifest[out_name] = gloss_str
                count += 1

        with open(os.path.join(args.phrase_output, 'manifest.json'), 'w') as fp:
            json.dump(manifest, fp, indent=2)
        print(f"\nDone: {count} phrase files, {len(set(manifest.values()))} phrases")

    else:
        # ---- Isolated sign extraction (parallelized) ----
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(args.crops, exist_ok=True)
        existing = set(os.listdir(args.output)) if args.resume else set()

        # Build task list
        tasks = []
        for class_dir in sorted(os.listdir(args.input)):
            class_path = os.path.join(args.input, class_dir)
            if not os.path.isdir(class_path):
                continue
            canonical = LABEL_ALIASES.get(class_dir, class_dir)
            for vid in sorted(os.listdir(class_path)):
                if not vid.endswith('.mp4'):
                    continue
                stem = os.path.splitext(vid)[0]
                hash_str = hashlib.md5(f"{canonical}_{stem}".encode()).hexdigest()[:6]
                out_name = f"{canonical}_{stem}_{hash_str}.npy"

                if args.resume and any(f.startswith(f"{canonical}_{stem}_") for f in existing):
                    continue
                tasks.append({
                    'video_path': os.path.join(class_path, vid),
                    'out_name': out_name,
                    'canonical': canonical,
                })

        total_tasks = len(tasks)
        print(f"=== ISOLATED SIGN EXTRACTION (Apple Vision) ===")
        print(f"Input:   {args.input}")
        print(f"Output:  {args.output} (keypoints) + {args.crops} (hand crops)")
        print(f"Videos:  {total_tasks} to extract")
        print(f"Workers: {args.workers}")
        if args.resume:
            print(f"Resumed: {len(existing)} existing files skipped")
        print()

        if total_tasks == 0:
            print("Nothing to extract.")
            sys.exit(0)

        # Split tasks across workers
        n_workers = min(args.workers, total_tasks)
        chunks = [[] for _ in range(n_workers)]
        for i, task in enumerate(tasks):
            chunks[i % n_workers].append(task)

        # Write task files and launch subprocesses
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix='slt_av_')
        procs = []
        task_files = []

        t0 = time.time()
        for wi, chunk in enumerate(chunks):
            if not chunk:
                continue
            task_path = os.path.join(tmpdir, f'tasks_{wi}.json')
            with open(task_path, 'w') as f:
                json.dump(chunk, f)
            task_files.append(task_path)

            env = os.environ.copy()
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            p = subprocess.Popen(
                [sys.executable, __file__, '--_worker', task_path, args.output, args.crops, 'isolated'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            procs.append((p, task_path, len(chunk)))
            print(f"  Worker {wi}: {len(chunk)} videos (PID {p.pid})")

        # Wait for all workers
        manifest = {}
        done = 0
        for p, task_path, n_tasks in procs:
            p.wait()
            result_path = task_path.replace('.json', '_result.json')
            if os.path.exists(result_path):
                with open(result_path) as f:
                    results = json.load(f)
                manifest.update(results)
            done += n_tasks
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total_tasks - done) / max(rate, 0.01)
            print(f"  Progress: {done}/{total_tasks} ({100*done/total_tasks:.1f}%) | "
                  f"{rate:.1f} vids/s | ETA: {eta/60:.0f}m | extracted: {len(manifest)}")

        # If resuming, merge with existing manifest
        manifest_path = os.path.join(args.output, 'manifest.json')
        if args.resume and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                old_manifest = json.load(f)
            old_manifest.update(manifest)
            manifest = old_manifest

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

        elapsed = time.time() - t0
        npy_count = len([f for f in os.listdir(args.output) if f.endswith('.npy')])
        crop_count = len([f for f in os.listdir(args.crops) if f.endswith('.jpg')])
        unique_classes = sorted(set(manifest.values()))
        print(f"\n=== DONE ===")
        print(f"Extracted: {npy_count} .npy files, {crop_count} hand crops")
        print(f"Manifest:  {len(manifest)} files, {len(unique_classes)} classes")
        print(f"Time:      {elapsed/60:.1f} minutes ({elapsed/max(total_tasks,1):.2f}s/video)")
        print(f"Output:    {args.output}/manifest.json")
