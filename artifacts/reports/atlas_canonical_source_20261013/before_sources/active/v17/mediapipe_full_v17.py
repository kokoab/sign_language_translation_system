"""MediaPipe-only v17 detector for the Android model family (hands, pose and face).

Produces the same ``FrameDetection`` contract as ``AppleVisionDetector`` so every v17 path
(isolated archives, hand crops, live/streaming observations) runs unchanged on it.
Archives are separately fingerprinted and must never be mixed with Apple archives.

Correspondence to Apple Vision was measured on Citizen train frames (2026-10-03):
- hands: MediaPipe handedness agrees with Apple chirality on unmirrored frames (272/277), no flip;
- body: pose 11/12/13/14 = Apple left/right shoulder, left/right elbow;
- face: ``FACE_MESH_INDICES`` are the nearest 478-mesh points to Apple's 15 samples
  (1-6 px median); the right brow uses the mirror partners of the left brow (293/276).
Hand tracking (VIDEO mode) is timestamp-independent; pose smoothing is not, so pose and face
run stateless (IMAGE mode) at the v17 auxiliary cadence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .extract_v17 import FrameDetection, HandDetection
from .schema_v17 import NODE_NAMES, NUM_CHANNELS, NUM_NODES, V17Config

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - environment-specific dependency
    mp = None
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None


SCHEMA_NAME = "slt_mediapipe_full_landmarks_v17"
SCHEMA_VERSION = 1
MEDIAPIPE_VERSION = "0.10.14"
DEFAULT_MODEL_DIR = Path("artifacts/model_assets/mediapipe")
MODEL_SHA256 = {
    "hand_landmarker": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "pose_landmarker_lite": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    "pose_landmarker_full": "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    "face_landmarker": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
POSE_INDICES = (11, 12, 13, 14)
FACE_MESH_INDICES = (468, 473, 63, 46, 293, 276, 344, 40, 291, 0, 17, 264, 152, 34, 168)
FEATURE_CHANNELS = (
    "x_body_relative",
    "y_body_relative",
    "relative_depth_log_scale",
    "presence",
    "confidence",
)


@dataclass(frozen=True)
class MediaPipeFullV17Config(V17Config):
    """v17 sampling/normalization plus the pinned MediaPipe runtime contract."""

    pose_model: str = "pose_landmarker_lite"
    backend: str = "gpu"
    hand_threshold: float = 0.5
    pose_visibility_threshold: float = 0.5
    mediapipe_version: str = MEDIAPIPE_VERSION

    def validate(self) -> None:
        super().validate()
        if self.pose_model not in ("pose_landmarker_lite", "pose_landmarker_full"):
            raise ValueError(f"unreviewed pose model {self.pose_model}")
        if self.backend not in ("gpu", "cpu"):
            raise ValueError("backend must be gpu or cpu")
        for name in ("hand_threshold", "pose_visibility_threshold"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def schema_payload(config: MediaPipeFullV17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "shape": [config.target_frames, NUM_NODES, NUM_CHANNELS],
        "dtype": "float16",
        "feature_channels": list(FEATURE_CHANNELS),
        "node_names": list(NODE_NAMES),
        "config": asdict(config),
        "extractor_contract": {
            "hands": "MediaPipe HandLandmarker VIDEO mode, 2 hands, handedness label as chirality",
            "hand_confidence": "handedness score broadcast to in-image joints; out-of-image joints missing",
            "body": "PoseLandmarker IMAGE mode, landmarks 11/12/13/14, visibility as confidence",
            "face": "FaceLandmarker IMAGE mode, 15 mesh points, confidence 1.0",
            "face_mesh_indices": list(FACE_MESH_INDICES),
            "model_sha256": {
                "hand_landmarker": MODEL_SHA256["hand_landmarker"],
                config.pose_model: MODEL_SHA256[config.pose_model],
                "face_landmarker": MODEL_SHA256["face_landmarker"],
            },
            "input": "upright, unmirrored frames; SRGBA on gpu, SRGB on cpu",
            "depth": "relative log-scale proxy only; no MediaPipe world depth",
            "missing_values": "all spatial/depth/confidence values are exactly zero",
        },
    }


def schema_fingerprint(config: MediaPipeFullV17Config) -> str:
    encoded = json.dumps(schema_payload(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


CROP_DETECTION = "landmark_pass_dense_tracking"


def derived_fingerprint(base_fingerprint: str, config: MediaPipeFullV17Config) -> str:
    """Fingerprint for a downstream archive (crops, embeddings) built from MediaPipe boxes.

    The base schema (crop size, embedding model) is unchanged, but the boxes come from
    MediaPipe, so these archives must never satisfy an Apple loader's expected fingerprint.
    Crop boxes reuse the dense landmark-pass hand detections of the same frames (Apple Vision is
    stateless, so its two passes agreed; sparse MediaPipe re-tracking would not).
    """
    payload = f"{base_fingerprint}+{SCHEMA_NAME}:{schema_fingerprint(config)}+{CROP_DETECTION}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(x: float, y: float) -> bool:
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


class MediaPipeFullDetector:
    """Reusable single-sequence MediaPipe detector with the AppleVisionDetector interface.

    Landmarkers are created once. ``reset_sequence`` drops hand tracking by feeding one blank
    frame, which gives output bit-identical to a freshly created landmarker.
    """

    def __init__(self, config: MediaPipeFullV17Config, model_dir: str | Path = DEFAULT_MODEL_DIR):
        if mp is None:
            raise RuntimeError("MediaPipe is unavailable") from _MEDIAPIPE_IMPORT_ERROR
        config.validate()
        if mp.__version__ != config.mediapipe_version:
            raise RuntimeError(f"MediaPipe {mp.__version__} != pinned {config.mediapipe_version}")
        self.config = config
        self._model_dir = Path(model_dir)
        self._create()

    def _create(self) -> None:
        config, model_dir = self.config, self._model_dir
        paths = {}
        for name in ("hand_landmarker", config.pose_model, "face_landmarker"):
            path = model_dir / f"{name}.task"
            actual = _sha256(path)
            if actual != MODEL_SHA256[name]:
                raise ValueError(f"{path} SHA-256 {actual} != pinned {MODEL_SHA256[name]}")
            paths[name] = str(path)
        vision = mp.tasks.vision
        delegate = (mp.tasks.BaseOptions.Delegate.GPU if config.backend == "gpu"
                    else mp.tasks.BaseOptions.Delegate.CPU)

        def base(name):
            return mp.tasks.BaseOptions(model_asset_path=paths[name], delegate=delegate)

        threshold = config.hand_threshold
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand_landmarker"), running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=threshold, min_hand_presence_confidence=threshold,
            min_tracking_confidence=threshold))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(config.pose_model), running_mode=vision.RunningMode.IMAGE, num_poses=1))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face_landmarker"), running_mode=vision.RunningMode.IMAGE, num_faces=1))
        self._clock_ms = 0
        self._tracking = False
        # GPU invocations so far. MediaPipe 0.10.14's macOS GPU path leaks one pixel buffer per call
        # and aborts after ~6,000 per landmarker instance; long runs call renew() between videos.
        self.calls = 0
        self._last_shape: tuple[int, int] | None = None

    def _image(self, frame_bgr: np.ndarray):
        if self.config.backend == "gpu":
            return mp.Image(image_format=mp.ImageFormat.SRGBA,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA))
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))

    def reset_sequence(self) -> None:
        if not self._tracking:
            return
        height, width = self._last_shape
        channels = 4 if self.config.backend == "gpu" else 3
        blank = mp.Image(image_format=mp.ImageFormat.SRGBA if channels == 4 else mp.ImageFormat.SRGB,
                         data=np.zeros((height, width, channels), np.uint8))
        self._clock_ms += 1000
        self.hand.detect_for_video(blank, self._clock_ms)
        self._clock_ms += 1000
        self._tracking = False

    def close(self) -> None:
        for landmarker in (self.hand, self.pose, self.face):
            landmarker.close()

    def renew(self) -> None:
        """Rebuild the landmarkers between sequences.

        The macOS GPU leak belongs to each landmarker instance (12,000 calls across four rebuilt
        instances ran clean), so long runs renew between videos. Tracking restarts either way.
        """
        self.close()
        self._create()

    def detect(self, frame_bgr: np.ndarray, include_body: bool, include_face: bool,
               include_hands: bool = True) -> FrameDetection:
        if not (include_hands or include_body or include_face):
            raise ValueError("at least one MediaPipe task must be enabled")
        image = self._image(frame_bgr)
        self.calls += int(include_hands) + int(include_body) + int(include_face)
        hands = self._hands(image) if include_hands else []
        body_xy, body_conf = self._body(image) if include_body else (
            np.zeros((4, 2), np.float32), np.zeros(4, np.float32))
        face_xy, face_conf = self._face(image) if include_face else (
            np.zeros((15, 2), np.float32), np.zeros(15, np.float32))
        return FrameDetection(hands, body_xy, body_conf, face_xy, face_conf)

    def _hands(self, image) -> list[HandDetection]:
        self._clock_ms += 33
        result = self.hand.detect_for_video(image, self._clock_ms)
        self._tracking = True
        self._last_shape = (image.height, image.width)
        detections = []
        for categories, landmarks in zip(result.handedness, result.hand_landmarks):
            if not categories or len(landmarks) != 21:
                continue
            category = categories[0]
            score = float(category.score)
            name = str(category.category_name).lower()
            chirality = name if name in ("left", "right") else "unknown"
            xy = np.zeros((21, 2), np.float32)
            confidence = np.zeros(21, np.float32)
            for index, point in enumerate(landmarks):
                if _inside(point.x, point.y):
                    xy[index] = (point.x, point.y)
                    confidence[index] = score
            if int((confidence > 0).sum()) >= 5:
                detections.append(HandDetection(xy, confidence, chirality, score))
        return detections

    def _body(self, image) -> tuple[np.ndarray, np.ndarray]:
        xy = np.zeros((4, 2), np.float32)
        confidence = np.zeros(4, np.float32)
        result = self.pose.detect(image)
        if not result.pose_landmarks:
            return xy, confidence
        landmarks = result.pose_landmarks[0]
        for slot, index in enumerate(POSE_INDICES):
            point = landmarks[index]
            visibility = float(point.visibility or 0.0)
            if visibility >= self.config.pose_visibility_threshold and _inside(point.x, point.y):
                xy[slot] = (point.x, point.y)
                confidence[slot] = visibility
        return xy, confidence

    def _face(self, image) -> tuple[np.ndarray, np.ndarray]:
        xy = np.zeros((15, 2), np.float32)
        confidence = np.zeros(15, np.float32)
        result = self.face.detect(image)
        if not result.face_landmarks:
            return xy, confidence
        mesh = result.face_landmarks[0]
        for slot, index in enumerate(FACE_MESH_INDICES):
            point = mesh[index]
            if _inside(point.x, point.y):
                xy[slot] = (point.x, point.y)
                confidence[slot] = 1.0
        return xy, confidence


class DetectionMemo:
    """Wrap a detector and replay its hand detections for pixel-identical frames.

    The landmark pass detects every sampled frame with dense tracking; the crop pass then asks
    again for 16 of those frames. Replaying keeps crop boxes identical to the landmark pass.
    Unseen frames (e.g. a downscaled pass) fall through to the real detector.
    """

    def __init__(self, detector):
        self.detector = detector
        self.memo: dict[bytes, FrameDetection] = {}
        self.hits = 0

    @staticmethod
    def _key(frame: np.ndarray) -> bytes:
        return hashlib.blake2b(frame.tobytes(), digest_size=16).digest() + str(frame.shape).encode()

    def clear(self) -> None:
        self.memo.clear()
        self.hits = 0

    def reset_sequence(self) -> None:
        self.detector.reset_sequence()

    def detect(self, frame_bgr, include_body, include_face, include_hands=True) -> FrameDetection:
        key = self._key(frame_bgr)
        if include_hands and not include_body and not include_face and key in self.memo:
            self.hits += 1
            stored = self.memo[key]
            return FrameDetection(stored.hands, np.zeros((4, 2), np.float32), np.zeros(4, np.float32),
                                  np.zeros((15, 2), np.float32), np.zeros(15, np.float32))
        detection = self.detector.detect(frame_bgr, include_body, include_face, include_hands)
        if include_hands:
            self.memo[key] = detection
        return detection


def stamp_result(result, config: MediaPipeFullV17Config, **extra):
    """Replace the Apple schema identity written by ``extract_frames_v17``."""
    result.metadata.update({
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": schema_fingerprint(config),
        "feature_channels": list(FEATURE_CHANNELS),
        "extractor": "mediapipe_full",
        "mediapipe_backend": config.backend,
        "pose_model": config.pose_model,
        **extra,
    })
    result.diagnostics["extractor_backend"] = f"mediapipe_full_{config.backend}"
    return result


def save_result(path: str | Path, result, config: MediaPipeFullV17Config) -> Path:
    """Write one archive; never overwrites an existing file."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial.npz")
    np.savez_compressed(
        temporary,
        features=result.features,
        metadata_json=np.array(json.dumps(result.metadata, sort_keys=True)),
        diagnostics_json=np.array(json.dumps(result.diagnostics, sort_keys=True)),
        schema_json=np.array(json.dumps(schema_payload(config), sort_keys=True)),
    )
    temporary.rename(destination)
    return destination


def load_result_features(path: str | Path, config: MediaPipeFullV17Config) -> np.ndarray:
    """Load features, rejecting any archive that is not this exact MediaPipe schema."""
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"]
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata.get("schema_fingerprint") != schema_fingerprint(config):
        raise ValueError(f"schema mismatch for {path}")
    if features.shape != (config.target_frames, NUM_NODES, NUM_CHANNELS):
        raise ValueError(f"unexpected feature shape {features.shape}")
    return features
