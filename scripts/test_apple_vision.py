"""
Test Apple Vision Framework extraction for SLT.
Verifies: hand/face/body detection, coordinate mapping, full pipeline output.

Usage:
    conda run -n sign_ai python scripts/test_apple_vision.py
"""
import os, sys, time, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Fake mediapipe
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_batch_fast_v2 import (
    normalize_sequence_v2, compute_kinematics_v2,
    interpolate_generic, NUM_NODES_V2, NUM_FACE_V2, NUM_BODY_V2,
)
from extract import (
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)

import Vision
import Quartz
from Foundation import NSData


# ============================================================
# Apple Vision landmark extraction
# ============================================================

# Hand joint names in Apple Vision order → our 21-joint order
# Apple Vision: wrist, thumbCMC, thumbMP, thumbIP, thumbTip,
#   indexMCP, indexPIP, indexDIP, indexTip, middleMCP, middlePIP, middleDIP, middleTip,
#   ringMCP, ringPIP, ringDIP, ringTip, littleMCP, littlePIP, littleDIP, littleTip
HAND_JOINT_NAMES = [
    'wrist',
    'thumbCMC', 'thumbMP', 'thumbIP', 'thumbTip',
    'indexMCP', 'indexPIP', 'indexDIP', 'indexTip',
    'middleMCP', 'middlePIP', 'middleDIP', 'middleTip',
    'ringMCP', 'ringPIP', 'ringDIP', 'ringTip',
    'littleMCP', 'littlePIP', 'littleDIP', 'littleTip',
]

# Map Apple Vision joint names to VNHumanHandPoseObservation keys
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

# Body joint keys for shoulders/elbows
BODY_JOINT_KEYS = {
    'left_shoulder': Vision.VNHumanBodyPoseObservationJointNameLeftShoulder,
    'right_shoulder': Vision.VNHumanBodyPoseObservationJointNameRightShoulder,
    'left_elbow': Vision.VNHumanBodyPoseObservationJointNameLeftElbow,
    'right_elbow': Vision.VNHumanBodyPoseObservationJointNameRightElbow,
}


def frame_to_ciimage(bgr_frame):
    """Convert OpenCV BGR frame to CIImage for Vision framework."""
    # Convert BGR to BGRA (Core Graphics expects 4 channels)
    bgra = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
    h, w, c = bgra.shape
    bytes_per_row = w * 4
    color_space = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    data_provider = Quartz.CGDataProviderCreateWithData(None, bgra.tobytes(), h * bytes_per_row, None)
    cg_image = Quartz.CGImageCreate(
        w, h, 8, 32, bytes_per_row, color_space,
        Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrder32Little,
        data_provider, None, False, Quartz.kCGRenderingIntentDefault,
    )
    ci = Quartz.CIImage.imageWithCGImage_(cg_image)
    return ci


def detect_hands(ci_image):
    """Detect hands in a CIImage, return list of (chirality, 21x2 coords, confidence)."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
    request.setMaximumHandCount_(2)
    handler.performRequests_error_([request], None)
    results = request.results()
    if not results:
        return []

    hands = []
    for obs in results:
        chirality = obs.chirality()  # 1=left, 2=right (from camera's perspective)
        points = []
        confs = []
        for jname in HAND_JOINT_NAMES:
            key = JOINT_KEYS[jname]
            pt, err = obs.recognizedPointForJointName_error_(key, None)
            if pt and pt.confidence() > 0.01:
                # Apple Vision: y=0 at bottom, we need y=0 at top
                points.append([pt.location().x, 1.0 - pt.location().y])
                confs.append(pt.confidence())
            else:
                points.append([0.0, 0.0])
                confs.append(0.0)
        hands.append((chirality, np.array(points, dtype=np.float32), np.mean(confs)))
    return hands


def detect_face(ci_image, frame_w, frame_h):
    """Detect face landmarks, return 15 face points as normalized [0,1] coords or None."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    landmarks_req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    handler.performRequests_error_([landmarks_req], None)
    face_results = landmarks_req.results()
    if not face_results:
        return None

    face = face_results[0]
    landmarks = face.landmarks()
    if not landmarks:
        return None

    img_size = Quartz.CGSizeMake(frame_w, frame_h)

    def get_region_point(region, idx):
        """Get a point from a face region using pointsInImageOfSize_, normalize to [0,1]."""
        if region is None or idx >= region.pointCount():
            return None
        pts = region.pointsInImageOfSize_(img_size)
        pt = pts[idx]
        # Vision y=0 at bottom, flip to y=0 at top, normalize to [0,1]
        return [pt.x / frame_w, 1.0 - pt.y / frame_h]

    nose = landmarks.nose()
    face_contour = landmarks.faceContour()
    outer_lips = landmarks.outerLips()
    inner_lips = landmarks.innerLips()
    left_eyebrow = landmarks.leftEyebrow()
    right_eyebrow = landmarks.rightEyebrow()
    left_eye = landmarks.leftEye()
    right_eye = landmarks.rightEye()
    median_line = landmarks.medianLine()

    # Map to our 15 face points:
    # 0: nose tip, 1: chin, 2: forehead, 3: left ear, 4: right ear
    # 5: left mouth corner, 6: right mouth corner, 7: upper lip, 8: lower lip
    # 9: left eyebrow inner, 10: left eyebrow outer, 11: right eyebrow inner, 12: right eyebrow outer
    # 13: left eye center, 14: right eye center
    face_points = []

    # Nose tip
    face_points.append(get_region_point(nose, nose.pointCount() // 2) if nose and nose.pointCount() > 0 else None)
    # Chin (middle of face contour)
    face_points.append(get_region_point(face_contour, face_contour.pointCount() // 2) if face_contour and face_contour.pointCount() > 0 else None)
    # Forehead (top of median line)
    face_points.append(get_region_point(median_line, median_line.pointCount() - 1) if median_line and median_line.pointCount() > 0 else None)
    # Left ear (start of face contour)
    face_points.append(get_region_point(face_contour, 0) if face_contour and face_contour.pointCount() > 0 else None)
    # Right ear (end of face contour)
    face_points.append(get_region_point(face_contour, face_contour.pointCount() - 1) if face_contour and face_contour.pointCount() > 0 else None)
    # Left mouth corner
    face_points.append(get_region_point(outer_lips, 0) if outer_lips and outer_lips.pointCount() > 0 else None)
    # Right mouth corner
    face_points.append(get_region_point(outer_lips, outer_lips.pointCount() // 2) if outer_lips and outer_lips.pointCount() >= 6 else None)
    # Upper lip center
    face_points.append(get_region_point(outer_lips, outer_lips.pointCount() * 3 // 4) if outer_lips and outer_lips.pointCount() >= 4 else None)
    # Lower lip center
    face_points.append(get_region_point(inner_lips, inner_lips.pointCount() // 2) if inner_lips and inner_lips.pointCount() >= 4 else None)
    # Left eyebrow inner
    face_points.append(get_region_point(left_eyebrow, left_eyebrow.pointCount() - 1) if left_eyebrow and left_eyebrow.pointCount() > 0 else None)
    # Left eyebrow outer
    face_points.append(get_region_point(left_eyebrow, 0) if left_eyebrow and left_eyebrow.pointCount() > 0 else None)
    # Right eyebrow inner
    face_points.append(get_region_point(right_eyebrow, 0) if right_eyebrow and right_eyebrow.pointCount() > 0 else None)
    # Right eyebrow outer
    face_points.append(get_region_point(right_eyebrow, right_eyebrow.pointCount() - 1) if right_eyebrow and right_eyebrow.pointCount() > 0 else None)
    # Left eye center
    face_points.append(get_region_point(left_eye, left_eye.pointCount() // 2) if left_eye and left_eye.pointCount() > 0 else None)
    # Right eye center
    face_points.append(get_region_point(right_eye, right_eye.pointCount() // 2) if right_eye and right_eye.pointCount() > 0 else None)

    if any(p is None for p in face_points):
        return None

    return np.array(face_points, dtype=np.float32)


def detect_body(ci_image):
    """Detect body pose, return 4 body points (shoulders + elbows) as normalized [0,1] or None.
    Uses low confidence threshold since ASL videos are often close-up upper body."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
    handler.performRequests_error_([request], None)
    results = request.results()
    if not results:
        return None

    obs = results[0]
    points = []
    min_conf = 0.05  # Very low — accept any reasonable detection
    for jname in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']:
        key = BODY_JOINT_KEYS[jname]
        pt, err = obs.recognizedPointForJointName_error_(key, None)
        if pt and pt.confidence() > min_conf:
            points.append([pt.location().x, 1.0 - pt.location().y, 0.0])
        else:
            return None  # Need all 4
    return np.array(points, dtype=np.float32)


# Shoulder estimation constants (learned from RTMW training data)
# Offsets from ear midpoint, in ear-distance units
_SHOULDER_L_OFFSET = np.array([0.885, 1.650, 0.0], dtype=np.float32)  # dx, dy, dz
_SHOULDER_R_OFFSET = np.array([-0.850, 1.685, 0.0], dtype=np.float32)
# Elbows have too much variance (1.0-1.6 MAE) — use shoulder offset * 1.6 as rough estimate
_ELBOW_L_OFFSET = np.array([1.52, 2.48, 0.0], dtype=np.float32)
_ELBOW_R_OFFSET = np.array([-0.97, 2.44, 0.0], dtype=np.float32)


def estimate_body_from_face(face_pts):
    """Estimate shoulder/elbow positions from face ear landmarks.

    face_pts: [15, 2] array with face landmarks (our 15-point layout).
    Nodes 3=left_ear, 4=right_ear in the face layout.
    Returns [4, 3] array: [L_shoulder, R_shoulder, L_elbow, R_elbow] or None.
    """
    if face_pts is None or len(face_pts) < 5:
        return None
    l_ear = face_pts[3]  # left ear
    r_ear = face_pts[4]  # right ear
    ear_mid = (l_ear + r_ear) / 2
    ear_dist = np.linalg.norm(r_ear - l_ear)
    if ear_dist < 1e-6:
        return None

    # Estimate body positions
    points = []
    for offset in [_SHOULDER_L_OFFSET, _SHOULDER_R_OFFSET, _ELBOW_L_OFFSET, _ELBOW_R_OFFSET]:
        pt = np.zeros(3, dtype=np.float32)
        pt[0] = ear_mid[0] + offset[0] * ear_dist
        pt[1] = ear_mid[1] + offset[1] * ear_dist
        pt[2] = 0.0
        points.append(pt)
    return np.array(points, dtype=np.float32)


def detect_all(ci_image, frame_w, frame_h):
    """Run hands + face in a single handler. Returns (hands_list, face_15x2, body_4x3).
    Body is estimated from face ears — no body pose request needed (saves ~7ms/frame)."""
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

    # Estimate body from face ears
    body_pts = estimate_body_from_face(face_pts) if face_pts is not None else None

    return hands, face_pts, body_pts


def extract_apple_vision(video_path):
    """Extract one video using Apple Vision → [32, 61, 10] float16 or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames < 8:
        return None

    # Subsample if too long
    max_process = 128
    indices = list(range(total_frames))
    if total_frames > max_process:
        step = total_frames / max_process
        indices = [int(i * step) for i in range(max_process)]
        frames = [frames[i] for i in indices]

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []

    for i, bgr in enumerate(frames):
        fi = indices[i] if total_frames > max_process else i
        h, w = bgr.shape[:2]
        ci = frame_to_ciimage(bgr)

        # Single batched detection: hands + face + estimated body
        hands, face_pts, body_pts = detect_all(ci, w, h)

        # Hands
        for chirality, coords, conf in hands:
            if conf < 0.25:
                continue
            coords_3d = np.column_stack([coords, np.zeros(21)])

            if chirality == -1:  # Signer's left hand → slots 0-20
                if not l_valid or l_valid[-1] != fi:
                    l_seq.append(coords_3d.tolist())
                    l_valid.append(fi)
            elif chirality == 1:  # Signer's right hand → slots 21-41
                if not r_valid or r_valid[-1] != fi:
                    r_seq.append(coords_3d.tolist())
                    r_valid.append(fi)
            else:  # Unknown chirality — assign by wrist x position
                wrist_x = coords[0, 0]
                if wrist_x > 0.5:
                    if not l_valid or l_valid[-1] != fi:
                        l_seq.append(coords_3d.tolist())
                        l_valid.append(fi)
                else:
                    if not r_valid or r_valid[-1] != fi:
                        r_seq.append(coords_3d.tolist())
                        r_valid.append(fi)

        # Face
        if face_pts is not None:
            face_3d = np.column_stack([face_pts, np.zeros(15)])
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_3d.tolist())
                face_valid.append(fi)

        # Body (estimated from face ears)
        if body_pts is not None:
            if not body_valid or body_valid[-1] != fi:
                body_seq.append(body_pts.tolist())
                body_valid.append(fi)

    if not l_valid and not r_valid:
        return None

    # Temporal outlier rejection
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None

    # Ghost hand: if only one hand ever seen, duplicate to the other
    l_ever, r_ever = bool(l_valid), bool(r_valid)
    if l_ever and not r_ever:
        r_seq, r_valid = list(l_seq), list(l_valid)
        r_ever = True
    elif r_ever and not l_ever:
        l_seq, l_valid = list(r_seq), list(r_valid)
        l_ever = True

    face_ever = bool(face_valid)
    body_ever = bool(body_valid)

    # Interpolation
    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
    face_full = interpolate_generic(
        np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
        face_valid, total_frames, NUM_FACE_V2)
    body_full = interpolate_generic(
        np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
        body_valid, total_frames, NUM_BODY_V2)

    # Combine [T, 61, 3]
    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

    # Pipeline: resample → filter → stabilize → normalize → kinematics
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


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    VIDEO_DIR = 'data/raw_videos/ASL VIDEOS'
    TEST_CLASSES = ['HELLO', 'GOOD', 'SORRY', 'THANKYOU', 'GO',
                    'MY', 'FRIEND', 'PLEASE', 'SCHOOL', 'WANT']

    print("=== APPLE VISION EXTRACTION TEST ===")
    print(f"Testing {len(TEST_CLASSES)} classes\n")

    results = {}
    for cls in TEST_CLASSES:
        cls_dir = os.path.join(VIDEO_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  {cls:10s}: SKIP (no directory)")
            continue
        vids = sorted(glob.glob(os.path.join(cls_dir, '*.mp4')))
        if not vids:
            print(f"  {cls:10s}: SKIP (no videos)")
            continue

        t0 = time.time()
        data = extract_apple_vision(vids[0])
        elapsed = time.time() - t0

        if data is not None:
            hand_range = [float(data[:, :42, 0].min()), float(data[:, :42, 0].max())]
            face_range = [float(data[:, 42:57, 0].min()), float(data[:, 42:57, 0].max())]
            body_range = [float(data[:, 57:61, 0].min()), float(data[:, 57:61, 0].max())]
            mask_vals = np.unique(data[:, :, 9])
            print(f"  {cls:10s}: shape={data.shape} dtype={data.dtype} | "
                  f"hand=[{hand_range[0]:.2f},{hand_range[1]:.2f}] "
                  f"face=[{face_range[0]:.2f},{face_range[1]:.2f}] "
                  f"body=[{body_range[0]:.2f},{body_range[1]:.2f}] "
                  f"mask={mask_vals} | {elapsed:.1f}s")
            results[cls] = data
        else:
            print(f"  {cls:10s}: FAILED | {elapsed:.1f}s")

    # Verification
    print(f"\n=== FORMAT VERIFICATION ===")
    print(f"Expected: [32, 61, 10] float16")
    all_ok = True
    for cls, data in results.items():
        if data.shape != (32, 61, 10):
            print(f"  FAIL: {cls} shape={data.shape}")
            all_ok = False
        if data.dtype != np.float16:
            print(f"  FAIL: {cls} dtype={data.dtype}")
            all_ok = False
        # Check mask channel makes sense
        mask = data[:, :, 9]
        if not (mask.min() >= 0 and mask.max() <= 1.0):
            print(f"  FAIL: {cls} mask out of range [{mask.min()}, {mask.max()}]")
            all_ok = False
        # Check no NaN/Inf
        if np.any(np.isnan(data.astype(np.float32))) or np.any(np.isinf(data.astype(np.float32))):
            print(f"  FAIL: {cls} has NaN/Inf")
            all_ok = False

    if all_ok:
        print(f"  ALL {len(results)}/{len(TEST_CLASSES)} PASSED ✓")
    else:
        print(f"  SOME FAILED")

    # Test bone features compatibility
    print(f"\n=== BONE FEATURES TEST ===")
    from src.train_stage_1 import compute_bone_features
    import torch
    for cls, data in list(results.items())[:3]:
        tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # [1, 32, 61, 10]
        try:
            with_bones = compute_bone_features(tensor)
            print(f"  {cls:10s}: {list(tensor.shape)} → {list(with_bones.shape)} ✓")
        except Exception as e:
            print(f"  {cls:10s}: BONE FEATURES FAILED: {e}")

    # Detection rate stats
    print(f"\n=== DETECTION STATS ===")
    print(f"  Extracted: {len(results)}/{len(TEST_CLASSES)} videos")
    if results:
        mask_stats = []
        for cls, data in results.items():
            lh = (data[:, 0, 9] > 0).sum()  # left hand mask
            rh = (data[:, 21, 9] > 0).sum()  # right hand mask
            fc = (data[:, 42, 9] > 0).sum()  # face mask
            bd = (data[:, 57, 9] > 0).sum()  # body mask
            mask_stats.append((cls, lh, rh, fc, bd))
        for cls, lh, rh, fc, bd in mask_stats:
            print(f"  {cls:10s}: LH={lh}/32 RH={rh}/32 Face={fc}/32 Body={bd}/32")
