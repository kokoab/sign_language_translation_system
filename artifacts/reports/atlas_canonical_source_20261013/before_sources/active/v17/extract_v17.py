#!/usr/bin/env python3
"""SLT v17 orientation-safe Apple Vision extractor.

Design rules:
- Decode or rotate pixels to upright orientation before Vision.
- Undo front-camera mirroring exactly once before handedness assignment.
- Convert normalized XY to isotropic image geometry before distances.
- Preserve per-joint confidence and keep missing spatial values at zero.
- Use bounded linear interpolation; never extrapolate hallucinated motion.
- Run hand/body/face requests together in one Vision handler per frame.
- Save a schema fingerprint with every sample.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Iterable
import math

import cv2
import numpy as np

try:
    import Quartz
    import Vision
    from Foundation import NSData, NSAutoreleasePool

    VISION_AVAILABLE = True
    _VISION_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    VISION_AVAILABLE = False
    _VISION_IMPORT_ERROR = exc
    Quartz = Vision = None
    NSData = NSAutoreleasePool = None

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.geometry_v17 import (
        body_relative_normalize,
        image_normalized_to_isotropic,
        interpolate_scalar_short_gaps,
        interpolate_short_gaps,
        resample_features,
    )
    from active.v17.schema_v17 import (
        BODY_END,
        BODY_START,
        FACE_END,
        FACE_START,
        FEATURE_CHANNELS,
        LHAND_END,
        LHAND_START,
        NUM_CHANNELS,
        NUM_NODES,
        RHAND_END,
        RHAND_START,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        V17Config,
        schema_fingerprint,
        schema_payload,
    )
else:
    from .geometry_v17 import (
        body_relative_normalize,
        image_normalized_to_isotropic,
        interpolate_scalar_short_gaps,
        interpolate_short_gaps,
        resample_features,
    )
    from .schema_v17 import (
        BODY_END,
        BODY_START,
        FACE_END,
        FACE_START,
        FEATURE_CHANNELS,
        LHAND_END,
        LHAND_START,
        NUM_CHANNELS,
        NUM_NODES,
        RHAND_END,
        RHAND_START,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        V17Config,
        schema_fingerprint,
        schema_payload,
    )


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def _require_vision() -> None:
    if not VISION_AVAILABLE:
        raise RuntimeError(
            "Apple Vision is unavailable. On this project's Python 3.9 setup, "
            "activate venv/ with PyObjC Vision and Quartz 11.1."
        ) from _VISION_IMPORT_ERROR


@dataclass
class HandDetection:
    xy: np.ndarray
    confidence: np.ndarray
    chirality: str
    score: float
    world_xyz: np.ndarray | None = None


@dataclass
class FrameDetection:
    hands: list[HandDetection]
    body_xy: np.ndarray
    body_confidence: np.ndarray
    face_xy: np.ndarray
    face_confidence: np.ndarray


@dataclass
class ExtractionResult:
    features: np.ndarray
    metadata: dict[str, object]
    diagnostics: dict[str, object]


def rotate_frame_clockwise(frame: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate clockwise without cropping or stretching the source pixels.

    Exact right-angle rotations use OpenCV's lossless transpose/flip operations.
    Other angles expand the canvas to contain all four transformed corners and use
    one affine resampling pass. This is suitable for explicit camera-roll correction
    and detector-space orientation augmentation.
    """
    value = float(degrees)
    if not math.isfinite(value):
        raise ValueError("rotation must be finite")
    normalized = value % 360.0
    if math.isclose(normalized, 0.0, abs_tol=1e-7) or math.isclose(
        normalized, 360.0, abs_tol=1e-7
    ):
        return frame
    if math.isclose(normalized, 90.0, abs_tol=1e-7):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if math.isclose(normalized, 180.0, abs_tol=1e-7):
        return cv2.rotate(frame, cv2.ROTATE_180)
    if math.isclose(normalized, 270.0, abs_tol=1e-7):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width = frame.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    # OpenCV's positive affine angle is counter-clockwise, hence the minus sign.
    matrix = cv2.getRotationMatrix2D(center, -normalized, 1.0)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    target_width = max(1, int(math.ceil(height * sine + width * cosine)))
    target_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += (target_width - width) / 2.0
    matrix[1, 2] += (target_height - height) / 2.0
    return cv2.warpAffine(
        frame,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def orient_frame(
    frame: np.ndarray, rotation_clockwise: float = 0, input_mirrored: bool = False
) -> np.ndarray:
    """Return upright, unmirrored pixels in the v17 canonical camera convention."""
    oriented = rotate_frame_clockwise(frame, rotation_clockwise)
    if input_mirrored:
        oriented = cv2.flip(oriented, 1)
    return oriented


def limit_image_side(frame: np.ndarray, maximum_side: int) -> np.ndarray:
    """Downscale without changing aspect ratio; never upscale."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= maximum_side:
        return frame
    scale = maximum_side / float(longest)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def frame_to_ciimage(frame_bgr: np.ndarray):
    _require_vision()
    bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    height, width = bgra.shape[:2]
    data = NSData.dataWithBytes_length_(bgra.tobytes(), height * width * 4)
    color_space = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    return Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data,
        width * 4,
        Quartz.CGSizeMake(width, height),
        Quartz.kCIFormatBGRA8,
        color_space,
    )


def _point_from_region(region, index: int):
    if region is None or region.pointCount() == 0:
        return None
    count = int(region.pointCount())
    resolved = index if index >= 0 else count + index
    if not 0 <= resolved < count:
        return None
    point = region.normalizedPoints()[resolved]
    return float(point.x), float(point.y)


def _face_point_to_image(point, bounding_box):
    if point is None:
        return None
    x = bounding_box.origin.x + point[0] * bounding_box.size.width
    y_bottom = bounding_box.origin.y + point[1] * bounding_box.size.height
    return float(x), float(1.0 - y_bottom)


class AppleVisionDetector:
    """Reusable, single-thread Apple Vision detector."""

    def __init__(self, minimum_confidence: float = 0.15):
        _require_vision()
        self.minimum_confidence = float(minimum_confidence)
        self.hand_request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
        self.hand_request.setMaximumHandCount_(2)
        self.body_request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        self.face_request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        self.hand_joint_names = (
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
        )
        self.body_joint_names = (
            Vision.VNHumanBodyPoseObservationJointNameLeftShoulder,
            Vision.VNHumanBodyPoseObservationJointNameRightShoulder,
            Vision.VNHumanBodyPoseObservationJointNameLeftElbow,
            Vision.VNHumanBodyPoseObservationJointNameRightElbow,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        include_body: bool,
        include_face: bool,
        include_hands: bool = True,
    ) -> FrameDetection:
        pool = NSAutoreleasePool.alloc().init()
        try:
            ciimage = frame_to_ciimage(frame_bgr)
            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_orientation_options_(
                ciimage, Quartz.kCGImagePropertyOrientationUp, None
            )
            requests = [self.hand_request] if include_hands else []
            if include_body:
                requests.append(self.body_request)
            if include_face:
                requests.append(self.face_request)
            success, error = handler.performRequests_error_(requests, None)
            if not success:
                raise RuntimeError(f"Apple Vision request failed: {error}")
            if not requests:
                raise ValueError("at least one Vision request must be enabled")
            hands = self._parse_hands() if include_hands else []
            body_xy, body_conf = self._parse_body() if include_body else (
                np.zeros((4, 2), np.float32), np.zeros(4, np.float32)
            )
            face_xy, face_conf = self._parse_face() if include_face else (
                np.zeros((15, 2), np.float32), np.zeros(15, np.float32)
            )
            return FrameDetection(hands, body_xy, body_conf, face_xy, face_conf)
        finally:
            del pool

    def _parse_hands(self) -> list[HandDetection]:
        detections: list[HandDetection] = []
        for observation in self.hand_request.results() or []:
            xy = np.zeros((21, 2), dtype=np.float32)
            confidence = np.zeros(21, dtype=np.float32)
            for index, joint_name in enumerate(self.hand_joint_names):
                point, _ = observation.recognizedPointForJointName_error_(
                    joint_name, None
                )
                if point is None:
                    continue
                score = float(point.confidence())
                if score < self.minimum_confidence:
                    continue
                xy[index] = (float(point.x()), 1.0 - float(point.y()))
                confidence[index] = score
            chirality_value = int(observation.chirality())
            if chirality_value == int(Vision.VNChiralityLeft):
                chirality = "left"
            elif chirality_value == int(Vision.VNChiralityRight):
                chirality = "right"
            else:
                chirality = "unknown"
            valid = confidence > 0
            score = float(confidence[valid].mean()) if valid.any() else 0.0
            if valid.sum() >= 5:
                detections.append(HandDetection(xy, confidence, chirality, score))
        return detections

    def _parse_body(self):
        xy = np.zeros((4, 2), dtype=np.float32)
        confidence = np.zeros(4, dtype=np.float32)
        results = self.body_request.results() or []
        if not results:
            return xy, confidence
        observation = max(results, key=lambda item: float(item.confidence()))
        for index, joint_name in enumerate(self.body_joint_names):
            point, _ = observation.recognizedPointForJointName_error_(
                joint_name, None
            )
            if point is None:
                continue
            score = float(point.confidence())
            if score < self.minimum_confidence:
                continue
            xy[index] = (float(point.x()), 1.0 - float(point.y()))
            confidence[index] = score
        return xy, confidence

    def _parse_face(self):
        xy = np.zeros((15, 2), dtype=np.float32)
        confidence = np.zeros(15, dtype=np.float32)
        results = self.face_request.results() or []
        if not results:
            return xy, confidence
        observation = max(
            results,
            key=lambda item: float(item.boundingBox().size.width * item.boundingBox().size.height),
        )
        landmarks = observation.landmarks()
        if landmarks is None:
            return xy, confidence
        bbox = observation.boundingBox()
        points = (
            _point_from_region(landmarks.leftPupil(), 0),
            _point_from_region(landmarks.rightPupil(), 0),
            _point_from_region(landmarks.leftEyebrow(), 0),
            _point_from_region(landmarks.leftEyebrow(), -1),
            _point_from_region(landmarks.rightEyebrow(), 0),
            _point_from_region(landmarks.rightEyebrow(), -1),
            _point_from_region(landmarks.noseCrest(), -1),
            _point_from_region(landmarks.outerLips(), 0),
            _point_from_region(landmarks.outerLips(), 7),
            _point_from_region(landmarks.outerLips(), 3),
            _point_from_region(landmarks.outerLips(), 10),
            _point_from_region(landmarks.faceContour(), 0),
            _point_from_region(landmarks.faceContour(), 8),
            _point_from_region(landmarks.faceContour(), -1),
            _point_from_region(landmarks.noseCrest(), 0),
        )
        score = max(float(observation.confidence()), self.minimum_confidence)
        for index, local_point in enumerate(points):
            image_point = _face_point_to_image(local_point, bbox)
            if image_point is not None:
                xy[index] = image_point
                confidence[index] = score
        return xy, confidence


def _assignment_cost(
    detection: HandDetection,
    slot: str,
    previous_wrist: np.ndarray | None,
) -> float:
    cost = -0.1 * detection.score
    if detection.chirality != "unknown":
        cost += 0.0 if detection.chirality == slot else 2.0
    elif detection.confidence[0] > 0:
        # In an upright, unmirrored camera image, the signer's anatomical left
        # is usually on the viewer's right. This is only a weak unknown-hand hint.
        expected_left = detection.xy[0, 0] >= 0.5
        if (slot == "left") != expected_left:
            cost += 0.2
    if previous_wrist is not None and detection.confidence[0] > 0:
        cost += float(np.linalg.norm(detection.xy[0] - previous_wrist))
    return cost


def assign_hands(
    detections: list[HandDetection],
    previous_wrists: dict[str, np.ndarray | None],
) -> dict[str, HandDetection | None]:
    """Assign up to two detections without relying on result order."""
    usable = sorted(detections, key=lambda item: item.score, reverse=True)[:2]
    result: dict[str, HandDetection | None] = {"left": None, "right": None}
    if not usable:
        return result
    if len(usable) == 1:
        detection = usable[0]
        slot = min(
            ("left", "right"),
            key=lambda name: _assignment_cost(
                detection, name, previous_wrists.get(name)
            ),
        )
        result[slot] = detection
        return result

    first, second = usable
    direct = _assignment_cost(first, "left", previous_wrists.get("left")) + _assignment_cost(
        second, "right", previous_wrists.get("right")
    )
    swapped = _assignment_cost(first, "right", previous_wrists.get("right")) + _assignment_cost(
        second, "left", previous_wrists.get("left")
    )
    if direct <= swapped:
        result["left"], result["right"] = first, second
    else:
        result["left"], result["right"] = second, first
    return result


def _upright_orientation_score(detection: FrameDetection) -> float:
    """Score whether auxiliary anatomy is upright in detector coordinates."""
    body_score, face_score = _orientation_score_components(detection)
    return face_score + body_score


def _orientation_score_components(detection: FrameDetection) -> tuple[float, float]:
    """Return body-detection strength and signed upright facial geometry."""
    body_score = float(detection.body_confidence.mean())
    eye_valid = detection.face_confidence[:2] > 0
    mouth_valid = detection.face_confidence[7:11] > 0
    face_score = 0.0
    if eye_valid.any() and mouth_valid.any():
        # Image Y grows downward. In an upright face, the mouth is below the eyes.
        eye_y = float(detection.face_xy[:2, 1][eye_valid].mean())
        mouth_y = float(detection.face_xy[7:11, 1][mouth_valid].mean())
        face_score = 10.0 * (mouth_y - eye_y)
    return body_score, face_score


def _upright_axis_score(detection: FrameDetection) -> float:
    """Return 1 for a horizontal shoulder/eye line and 0 for a vertical one."""
    for xy, confidence, first, second in (
        (detection.body_xy, detection.body_confidence, 0, 1),
        (detection.face_xy, detection.face_confidence, 0, 1),
    ):
        if confidence[first] > 0 and confidence[second] > 0:
            delta = xy[second] - xy[first]
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
            folded = abs((angle + 90.0) % 180.0 - 90.0)
            return 1.0 - folded / 90.0
    return 0.0


def _select_coarse_orientation_from_scores(
    scores: dict[str, dict[str, float]],
) -> float:
    """Resolve axis family first, then direction within the opposite pair."""
    families = ((0.0, 180.0), (90.0, 270.0))
    family = max(
        families,
        key=lambda angles: (
            max(scores[str(int(angle))]["axis"] for angle in angles),
            max(scores[str(int(angle))]["body"] for angle in angles),
        ),
    )
    return max(
        family,
        key=lambda angle: (
            scores[str(int(angle))]["face"],
            scores[str(int(angle))]["body"],
        ),
    )


def choose_coarse_orientation_v17(
    frames: list[np.ndarray],
    detector: AppleVisionDetector,
    *,
    probe_frames: int = 3,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Choose the lossless quadrant that makes anatomy most upright for Vision.

    Container orientation is applied before this function. This second-stage probe
    handles files whose pixels have arbitrary camera roll but whose metadata does not
    describe it. The classifier's continuous-roll augmentation handles the remaining
    residual angle, which is at most 45 degrees after the selected quadrant correction.
    """
    if not frames:
        raise ValueError("at least one frame is required for orientation selection")
    if probe_frames < 1:
        raise ValueError("probe_frames must be positive")
    count = min(probe_frames, len(frames))
    indices = np.rint(
        np.linspace(0.25 * (len(frames) - 1), 0.75 * (len(frames) - 1), count)
    ).astype(int)
    corrections = (0.0, 90.0, 180.0, 270.0)
    scores: dict[str, dict[str, float]] = {}
    reset_sequence = getattr(detector, "reset_sequence", None)
    for correction in corrections:
        body_values = []
        face_values = []
        axis_values = []
        for index in indices:
            detection = detector.detect(
                rotate_frame_clockwise(frames[int(index)], correction),
                include_body=True,
                include_face=True,
                include_hands=False,
            )
            body_score, face_score = _orientation_score_components(detection)
            body_values.append(body_score)
            face_values.append(face_score)
            axis_values.append(_upright_axis_score(detection))
        body_mean = float(np.mean(body_values))
        face_mean = float(np.mean(face_values))
        axis_mean = float(np.mean(axis_values))
        scores[str(int(correction))] = {
            "body": body_mean,
            "face": face_mean,
            "axis": axis_mean,
            "combined": body_mean + face_mean,
        }
    if reset_sequence is not None:
        reset_sequence()
    # Axis horizontalness chooses between the two opposite-quadrant families. Signed
    # facial anatomy then selects the direction within the family; body confidence is
    # only a fallback because it can disappear while the signer is actively moving.
    # Stable order deliberately resolves a fully uninformative probe to no correction.
    selected = _select_coarse_orientation_from_scores(scores)
    return selected, scores


def _sample_indices(frame_count: int, maximum_frames: int) -> set[int] | None:
    if frame_count <= 0 or frame_count <= maximum_frames:
        return None
    return set(
        np.rint(np.linspace(0, frame_count - 1, maximum_frames)).astype(int).tolist()
    )


def read_video_frames(
    video_path: str | Path,
    maximum_frames: int,
    maximum_image_side: int,
    rotation: str | float = "auto",
    input_mirrored: bool = False,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Decode an orientation-aware, memory-bounded frame sample."""
    path = str(video_path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    reported_rotation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META)))
    if rotation == "auto":
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        manual_rotation = 0
        orientation_mode = "opencv_metadata_auto"
    else:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        manual_rotation = float(rotation) % 360.0
        if not math.isfinite(manual_rotation):
            capture.release()
            raise ValueError("rotation must be finite")
        orientation_mode = "explicit_clockwise_rotation"

    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    selected_indices = _sample_indices(frame_count, maximum_frames)
    # If the container omits or under-reports its frame count, a deterministic
    # reservoir keeps memory bounded without truncating the sign to its first frames.
    reservoir_mode = selected_indices is None
    reservoir_rng = np.random.default_rng(17)
    frame_records: list[tuple[int, np.ndarray]] = []
    decoded = 0
    decoded_shape = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded_shape is None:
            decoded_shape = frame.shape[:2]
        if not reservoir_mode and decoded in selected_indices:
            upright = orient_frame(frame, manual_rotation, input_mirrored)
            frame_records.append(
                (decoded, limit_image_side(upright, maximum_image_side))
            )
        elif reservoir_mode:
            if len(frame_records) < maximum_frames:
                slot = len(frame_records)
            else:
                candidate = int(reservoir_rng.integers(0, decoded + 1))
                slot = candidate if candidate < maximum_frames else -1
            if slot >= 0:
                upright = orient_frame(frame, manual_rotation, input_mirrored)
                record = (decoded, limit_image_side(upright, maximum_image_side))
                if slot == len(frame_records):
                    frame_records.append(record)
                else:
                    frame_records[slot] = record
        decoded += 1
    capture.release()
    frame_records.sort(key=lambda item: item[0])
    frames = [frame for _, frame in frame_records]
    if not frames:
        raise RuntimeError(f"No decodable frames in {path}")
    shapes = {frame.shape[:2] for frame in frames}
    if len(shapes) != 1:
        raise RuntimeError(f"Frame dimensions changed within {path}: {sorted(shapes)}")
    height, width = frames[0].shape[:2]
    metadata = {
        "video_path": path,
        "reported_frame_count": frame_count,
        "decoded_frame_count": decoded,
        "sampled_frame_count": len(frames),
        "fps": fps,
        "reported_rotation_degrees": reported_rotation,
        "orientation_mode": orientation_mode,
        "explicit_rotation_clockwise": float(manual_rotation),
        "input_mirrored": bool(input_mirrored),
        "decoded_width": int(decoded_shape[1]) if decoded_shape else 0,
        "decoded_height": int(decoded_shape[0]) if decoded_shape else 0,
        "oriented_width": int(width),
        "oriented_height": int(height),
        "orientation": "square" if width == height else "portrait" if height > width else "landscape",
    }
    return frames, metadata


def extract_frames_v17(
    frames: list[np.ndarray],
    config: V17Config | None = None,
    *,
    rotation_clockwise: float = 0,
    input_mirrored: bool = False,
    detector: AppleVisionDetector | None = None,
    metadata: dict[str, object] | None = None,
) -> ExtractionResult | None:
    """Extract one isolated sign from already-decoded frames."""
    config = config or V17Config()
    config.validate()
    if len(frames) < 4:
        return None
    oriented = [
        limit_image_side(
            orient_frame(frame, rotation_clockwise, input_mirrored),
            config.maximum_image_side,
        )
        for frame in frames
    ]
    shapes = {frame.shape[:2] for frame in oriented}
    if len(shapes) != 1:
        raise ValueError(f"all frames must have the same dimensions, got {shapes}")
    height, width = oriented[0].shape[:2]
    detector = detector or AppleVisionDetector(config.minimum_point_confidence)
    reset_sequence = getattr(detector, "reset_sequence", None)
    if reset_sequence is not None:
        reset_sequence()
    count = len(oriented)
    xy = np.zeros((count, NUM_NODES, 2), dtype=np.float32)
    confidence = np.zeros((count, NUM_NODES), dtype=np.float32)
    detector_depth = np.zeros((count, NUM_NODES), dtype=np.float32)
    detector_depth_confidence = np.zeros((count, NUM_NODES), dtype=np.float32)
    previous_wrists: dict[str, np.ndarray | None] = {"left": None, "right": None}
    observed_hand_frames = 0
    chirality_counts = {"left": 0, "right": 0, "unknown": 0}
    started = time.perf_counter()

    for frame_index, frame in enumerate(oriented):
        detection = detector.detect(
            frame,
            include_body=frame_index % config.body_interval == 0,
            include_face=config.include_face and frame_index % config.face_interval == 0,
        )
        for hand in detection.hands:
            chirality_counts[hand.chirality] += 1
        assigned = assign_hands(detection.hands, previous_wrists)
        any_hand = False
        for slot, start, end in (
            ("left", LHAND_START, LHAND_END),
            ("right", RHAND_START, RHAND_END),
        ):
            hand = assigned[slot]
            if hand is None:
                continue
            xy[frame_index, start:end] = hand.xy
            confidence[frame_index, start:end] = hand.confidence
            if hand.world_xyz is not None:
                world_xyz = np.asarray(hand.world_xyz, dtype=np.float32)
                if world_xyz.shape != (21, 3):
                    raise ValueError(f"unexpected hand world shape {world_xyz.shape}")
                palm_scale = float(np.linalg.norm(world_xyz[9] - world_xyz[0]))
                if np.isfinite(palm_scale) and palm_scale > 1e-5:
                    valid_depth = hand.confidence > 0
                    detector_depth[frame_index, start:end] = (
                        world_xyz[:, 2] - world_xyz[0, 2]
                    ) / palm_scale
                    detector_depth_confidence[frame_index, start:end] = (
                        hand.confidence * valid_depth
                    )
            if hand.confidence[0] > 0:
                previous_wrists[slot] = hand.xy[0].copy()
            any_hand |= bool((hand.confidence > 0).any())
        observed_hand_frames += int(any_hand)
        xy[frame_index, BODY_START:BODY_END] = detection.body_xy
        confidence[frame_index, BODY_START:BODY_END] = detection.body_confidence
        if config.include_face:
            xy[frame_index, FACE_START:FACE_END] = detection.face_xy
            confidence[frame_index, FACE_START:FACE_END] = detection.face_confidence

    if observed_hand_frames < config.minimum_detected_hand_frames:
        return None

    untrimmed_count = count
    trim_start, trim_end = 0, count
    if config.trim_to_hand_activity:
        active = np.flatnonzero((confidence[:, :RHAND_END] > 0).sum(axis=1) >= 5)
        if active.size:
            trim_start = max(0, int(active[0]) - config.trim_context_frames)
            trim_end = min(count, int(active[-1]) + config.trim_context_frames + 1)
            if trim_end - trim_start >= 4:
                xy = xy[trim_start:trim_end]
                confidence = confidence[trim_start:trim_end]
                detector_depth = detector_depth[trim_start:trim_end]
                detector_depth_confidence = detector_depth_confidence[trim_start:trim_end]
                count = trim_end - trim_start

    valid = confidence > 0
    xy = image_normalized_to_isotropic(xy, width, height, valid)
    detector_depth[:, :RHAND_END], detector_depth_confidence[:, :RHAND_END] = (
        interpolate_scalar_short_gaps(
            detector_depth[:, :RHAND_END],
            detector_depth_confidence[:, :RHAND_END],
            config.hand_gap_frames,
        )
    )
    xy[:, :RHAND_END], confidence[:, :RHAND_END] = interpolate_short_gaps(
        xy[:, :RHAND_END], confidence[:, :RHAND_END], config.hand_gap_frames
    )
    xy[:, FACE_START:], confidence[:, FACE_START:] = interpolate_short_gaps(
        xy[:, FACE_START:], confidence[:, FACE_START:], config.auxiliary_gap_frames
    )
    normalized_xy, depth, normalization = body_relative_normalize(xy, confidence)
    direct_depth_valid = detector_depth_confidence > 0
    depth[direct_depth_valid] = detector_depth[direct_depth_valid]
    presence = (confidence > 0).astype(np.float32)
    features = np.zeros((count, NUM_NODES, NUM_CHANNELS), dtype=np.float32)
    features[..., 0:2] = normalized_xy
    features[..., 2] = depth
    features[..., 3] = presence
    features[..., 4] = np.clip(confidence, 0.0, 1.0)
    features = resample_features(features, config.target_frames).astype(np.float16)

    elapsed = time.perf_counter() - started
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "schema_fingerprint": schema_fingerprint(config),
            "feature_channels": list(FEATURE_CHANNELS),
            "source_frames_processed": count,
            "source_frames_before_hand_trim": untrimmed_count,
            "hand_trim_start_frame": trim_start,
            "hand_trim_end_frame_exclusive": trim_end,
            "oriented_width": int(width),
            "oriented_height": int(height),
            "orientation": "square" if width == height else "portrait" if height > width else "landscape",
            "frame_rotation_clockwise": float(rotation_clockwise) % 360.0,
            "frame_input_mirrored": bool(input_mirrored),
        }
    )
    diagnostics = {
        **normalization,
        "elapsed_seconds": elapsed,
        "observed_hand_frames": observed_hand_frames,
        "observed_hand_frame_fraction": observed_hand_frames / count,
        "observed_hand_frame_fraction_before_trim": observed_hand_frames / untrimmed_count,
        "hand_presence_fraction": float(features[:, :RHAND_END, 3].mean()),
        "face_presence_fraction": float(features[:, FACE_START:FACE_END, 3].mean()),
        "body_presence_fraction": float(features[:, BODY_START:BODY_END, 3].mean()),
        "chirality_observation_counts": chirality_counts,
        "finite": bool(np.isfinite(features).all()),
        "detector_world_depth_fraction": float(
            direct_depth_valid[:, :RHAND_END].mean()
        ),
    }
    if not diagnostics["finite"]:
        raise RuntimeError("non-finite feature value produced")
    return ExtractionResult(features, result_metadata, diagnostics)


def extract_video_v17(
    video_path: str | Path,
    config: V17Config | None = None,
    *,
    rotation: str | float = "auto",
    input_mirrored: bool = False,
    detector: AppleVisionDetector | None = None,
    vision_auto_orient: bool = True,
) -> ExtractionResult | None:
    config = config or V17Config()
    frames, video_metadata = read_video_frames(
        video_path,
        config.maximum_source_frames,
        config.maximum_image_side,
        rotation=rotation,
        input_mirrored=input_mirrored,
    )
    # read_video_frames applies container metadata first. Probe auxiliary anatomy only
    # in auto mode, because an explicit rotation is already a caller-declared fix.
    detector = detector or AppleVisionDetector(config.minimum_point_confidence)
    if rotation == "auto" and vision_auto_orient:
        correction, scores = choose_coarse_orientation_v17(frames, detector)
        if correction:
            frames = [rotate_frame_clockwise(frame, correction) for frame in frames]
        height, width = frames[0].shape[:2]
        video_metadata.update(
            {
                "vision_auto_orientation_enabled": True,
                "vision_coarse_rotation_clockwise": correction,
                "vision_orientation_scores": scores,
                "oriented_width": int(width),
                "oriented_height": int(height),
                "orientation": (
                    "square" if width == height else "portrait" if height > width else "landscape"
                ),
            }
        )
    else:
        video_metadata["vision_auto_orientation_enabled"] = False
        video_metadata["vision_coarse_rotation_clockwise"] = 0.0
    # Frames now use both container metadata and, when enabled, Vision's preferred
    # coarse quadrant. They are also already unmirrored.
    return extract_frames_v17(
        frames,
        config,
        detector=detector,
        metadata=video_metadata,
    )


def save_v17_result(path: str | Path, result: ExtractionResult, config: V17Config) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=result.features,
        metadata_json=np.array(json.dumps(result.metadata, sort_keys=True)),
        diagnostics_json=np.array(json.dumps(result.diagnostics, sort_keys=True)),
        schema_json=np.array(json.dumps(schema_payload(config), sort_keys=True)),
    )
    return destination


def load_v17_result(path: str | Path, config: V17Config | None = None) -> ExtractionResult:
    config = config or V17Config()
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"]
        metadata = json.loads(str(payload["metadata_json"]))
        diagnostics = json.loads(str(payload["diagnostics_json"]))
    expected = schema_fingerprint(config)
    if metadata.get("schema_fingerprint") != expected:
        raise ValueError(
            f"schema mismatch for {path}: {metadata.get('schema_fingerprint')} != {expected}"
        )
    expected_shape = (config.target_frames, NUM_NODES, NUM_CHANNELS)
    if features.shape != expected_shape:
        raise ValueError(f"unexpected feature shape {features.shape}; expected {expected_shape}")
    return ExtractionResult(features, metadata, diagnostics)


def iter_videos(input_dir: str | Path) -> Iterable[tuple[str, Path]]:
    root = Path(input_dir)
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for video in sorted(class_dir.iterdir()):
            if video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS:
                yield class_dir.name, video


def extract_batch_v17(
    input_dir: str | Path,
    output_dir: str | Path,
    config: V17Config | None = None,
    *,
    rotation: str | float = "auto",
    input_mirrored: bool = False,
    resume: bool = True,
    vision_auto_orient: bool = True,
) -> dict[str, int]:
    config = config or V17Config()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_schema_v17.json").write_text(
        json.dumps(schema_payload(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    detector = AppleVisionDetector(config.minimum_point_confidence)
    jobs = list(iter_videos(input_dir))
    counts = {"ok": 0, "no_hands": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()
    for index, (class_name, video_path) in enumerate(jobs, start=1):
        destination = output_root / class_name / f"{video_path.stem}.v17.npz"
        if resume and destination.exists():
            try:
                load_v17_result(destination, config)
                counts["skipped"] += 1
                continue
            except (ValueError, KeyError, OSError):
                pass
        try:
            result = extract_video_v17(
                video_path,
                config,
                rotation=rotation,
                input_mirrored=input_mirrored,
                detector=detector,
                vision_auto_orient=vision_auto_orient,
            )
            if result is None:
                counts["no_hands"] += 1
                print(f"NO_HANDS {video_path}", file=sys.stderr)
            else:
                save_v17_result(destination, result, config)
                counts["ok"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAILED {video_path}: {exc}", file=sys.stderr)
        if index == 1 or index % 25 == 0 or index == len(jobs):
            elapsed = time.perf_counter() - started
            print(
                f"[{index}/{len(jobs)}] ok={counts['ok']} no_hands={counts['no_hands']} "
                f"failed={counts['failed']} skipped={counts['skipped']} "
                f"rate={index / max(elapsed, 1e-6):.2f}/s"
            )
    return counts


def _parse_rotation(value: str) -> str | float:
    if value == "auto":
        return value
    try:
        rotation = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rotation must be auto or a finite number of degrees") from exc
    if not math.isfinite(rotation):
        raise argparse.ArgumentTypeError("rotation must be auto or a finite number of degrees")
    return rotation % 360.0


def main() -> None:
    parser = argparse.ArgumentParser(description="SLT v17 Apple Vision extractor")
    parser.add_argument("input", type=Path, help="Video file or root containing class directories")
    parser.add_argument("--output", type=Path, default=Path("data/local/ASL_landmarks_v17"))
    parser.add_argument("--rotation", type=_parse_rotation, default="auto")
    parser.add_argument(
        "--input-mirrored",
        action="store_true",
        help="Source pixels are mirrored; flip once before Vision",
    )
    parser.add_argument("--body-interval", type=int, default=8)
    parser.add_argument("--face-interval", type=int, default=8)
    parser.add_argument("--maximum-source-frames", type=int, default=96)
    parser.add_argument("--maximum-image-side", type=int, default=1280)
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument(
        "--no-vision-auto-orient",
        action="store_true",
        help="Trust container orientation only; skip the four-quadrant Vision probe",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = V17Config(
        maximum_source_frames=args.maximum_source_frames,
        maximum_image_side=args.maximum_image_side,
        body_interval=args.body_interval,
        face_interval=args.face_interval,
        include_face=not args.no_face,
    )
    config.validate()
    if args.input.is_file():
        result = extract_video_v17(
            args.input,
            config,
            rotation=args.rotation,
            input_mirrored=args.input_mirrored,
            vision_auto_orient=not args.no_vision_auto_orient,
        )
        if result is None:
            raise SystemExit("No usable hand detections")
        destination = args.output
        if destination.suffix.lower() != ".npz":
            destination = destination / f"{args.input.stem}.v17.npz"
        save_v17_result(destination, result, config)
        print(json.dumps({"output": str(destination), **result.diagnostics}, indent=2))
    else:
        counts = extract_batch_v17(
            args.input,
            args.output,
            config,
            rotation=args.rotation,
            input_mirrored=args.input_mirrored,
            resume=not args.no_resume,
            vision_auto_orient=not args.no_vision_auto_orient,
        )
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
