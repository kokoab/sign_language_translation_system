#!/usr/bin/env python3
"""MediaPipe-hand + optional Apple auxiliary extractor for the v17 bakeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_v17 import (
        AppleVisionDetector,
        ExtractionResult,
        FrameDetection,
        HandDetection,
        extract_frames_v17,
        iter_videos,
        read_video_frames,
    )
    from active.v17.schema_mediapipe_v17 import (
        FEATURE_CHANNELS,
        HAND_MODEL_SHA256,
        MediaPipeV17Config,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        schema_fingerprint,
        schema_payload,
    )
    from active.v17.schema_v17 import NUM_CHANNELS, NUM_NODES
else:
    from .extract_v17 import (
        AppleVisionDetector,
        ExtractionResult,
        FrameDetection,
        HandDetection,
        extract_frames_v17,
        iter_videos,
        read_video_frames,
    )
    from .schema_mediapipe_v17 import (
        FEATURE_CHANNELS,
        HAND_MODEL_SHA256,
        MediaPipeV17Config,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        schema_fingerprint,
        schema_payload,
    )
    from .schema_v17 import NUM_CHANNELS, NUM_NODES

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - environment-specific dependency
    mp = None
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None


DEFAULT_MODEL_PATH = Path("artifacts/model_assets/mediapipe/hand_landmarker.task")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MediaPipeHybridDetector:
    """MediaPipe video hand tracking with low-rate Apple body/face auxiliaries."""

    def __init__(self, model_path: str | Path, config: MediaPipeV17Config):
        if mp is None:
            raise RuntimeError("MediaPipe is unavailable") from _MEDIAPIPE_IMPORT_ERROR
        self.model_path = Path(model_path)
        self.config = config
        config.validate()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        actual_hash = sha256_file(self.model_path)
        if actual_hash != config.hand_model_sha256:
            raise ValueError(
                f"MediaPipe model SHA-256 mismatch: {actual_hash} != "
                f"{config.hand_model_sha256}"
            )
        self.auxiliary = (
            AppleVisionDetector(config.minimum_point_confidence)
            if config.include_apple_auxiliary
            else None
        )
        self.landmarker = None
        self.frame_index = 0

    def reset_sequence(self) -> None:
        self.close_landmarker()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.config.minimum_hand_detection_confidence,
            min_hand_presence_confidence=self.config.minimum_hand_presence_confidence,
            min_tracking_confidence=self.config.minimum_hand_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.frame_index = 0

    def close_landmarker(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def close(self) -> None:
        self.close_landmarker()

    def detect(
        self,
        frame_bgr: np.ndarray,
        include_body: bool,
        include_face: bool,
        include_hands: bool = True,
    ) -> FrameDetection:
        if not include_hands:
            raise ValueError("MediaPipe hybrid detector always supplies hands")
        if self.landmarker is None:
            self.reset_sequence()
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, self.frame_index * 33)
        self.frame_index += 1
        hands = []
        for categories, landmarks, world_landmarks in zip(
            result.handedness, result.hand_landmarks, result.hand_world_landmarks
        ):
            if not categories:
                continue
            category = categories[0]
            score = float(category.score)
            name = str(category.category_name).lower()
            chirality = name if name in ("left", "right") else "unknown"
            xy = np.asarray([(item.x, item.y) for item in landmarks], dtype=np.float32)
            world = np.asarray(
                [(item.x, item.y, item.z) for item in world_landmarks], dtype=np.float32
            )
            if xy.shape != (21, 2) or world.shape != (21, 3):
                continue
            confidence = np.full(21, score, dtype=np.float32)
            hands.append(HandDetection(xy, confidence, chirality, score, world))

        if self.auxiliary is not None and (include_body or include_face):
            auxiliary = self.auxiliary.detect(
                frame_bgr,
                include_body=include_body,
                include_face=include_face,
                include_hands=False,
            )
            body_xy, body_confidence = auxiliary.body_xy, auxiliary.body_confidence
            face_xy, face_confidence = auxiliary.face_xy, auxiliary.face_confidence
        else:
            body_xy = np.zeros((4, 2), dtype=np.float32)
            body_confidence = np.zeros(4, dtype=np.float32)
            face_xy = np.zeros((15, 2), dtype=np.float32)
            face_confidence = np.zeros(15, dtype=np.float32)
        return FrameDetection(
            hands, body_xy, body_confidence, face_xy, face_confidence
        )


def extract_frames_mediapipe_v17(
    frames: list[np.ndarray],
    config: MediaPipeV17Config,
    detector: MediaPipeHybridDetector,
    *,
    rotation_clockwise: int = 0,
    input_mirrored: bool = False,
    metadata: dict[str, object] | None = None,
) -> ExtractionResult | None:
    result = extract_frames_v17(
        frames,
        config,
        rotation_clockwise=rotation_clockwise,
        input_mirrored=input_mirrored,
        detector=detector,
        metadata=metadata,
    )
    if result is None:
        return None
    result.metadata.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "schema_fingerprint": schema_fingerprint(config),
            "feature_channels": list(FEATURE_CHANNELS),
            "hand_extractor": "mediapipe_hand_landmarker",
            "auxiliary_extractor": "apple_vision"
            if config.include_apple_auxiliary
            else "none",
            "hand_model_sha256": config.hand_model_sha256,
        }
    )
    result.diagnostics["extractor_backend"] = (
        "mediapipe_hand_apple_aux"
        if config.include_apple_auxiliary
        else "mediapipe_hand_only"
    )
    return result


def extract_video_mediapipe_v17(
    video_path: str | Path,
    config: MediaPipeV17Config,
    detector: MediaPipeHybridDetector,
    *,
    rotation: str | int = "auto",
    input_mirrored: bool = False,
) -> ExtractionResult | None:
    frames, metadata = read_video_frames(
        video_path,
        config.maximum_source_frames,
        config.maximum_image_side,
        rotation=rotation,
        input_mirrored=input_mirrored,
    )
    return extract_frames_mediapipe_v17(
        frames, config, detector, metadata=metadata
    )


def save_mediapipe_v17_result(
    path: str | Path, result: ExtractionResult, config: MediaPipeV17Config
) -> Path:
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


def load_mediapipe_v17_result(
    path: str | Path, config: MediaPipeV17Config
) -> ExtractionResult:
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


def extract_batch_mediapipe_v17(
    input_dir: str | Path,
    output_dir: str | Path,
    config: MediaPipeV17Config,
    detector: MediaPipeHybridDetector,
    *,
    rotation: str | int = "auto",
    input_mirrored: bool = False,
    resume: bool = True,
) -> dict[str, int]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_schema_v17.json").write_text(
        json.dumps(schema_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = {"extracted": 0, "skipped": 0, "no_hands": 0, "failed": 0}
    for index, (label, video) in enumerate(iter_videos(input_dir), start=1):
        destination = output_root / label / f"{video.stem}.v17.npz"
        if resume and destination.exists():
            try:
                load_mediapipe_v17_result(destination, config)
            except Exception:
                pass
            else:
                counts["skipped"] += 1
                continue
        try:
            result = extract_video_mediapipe_v17(
                video,
                config,
                detector,
                rotation=rotation,
                input_mirrored=input_mirrored,
            )
            if result is None:
                counts["no_hands"] += 1
                print(f"NO_HANDS {video}")
                continue
            save_mediapipe_v17_result(destination, result, config)
            counts["extracted"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAILED {video}: {exc}")
        if index % 25 == 0:
            print(json.dumps({"processed": index, **counts}, sort_keys=True))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rotation", choices=("auto", "0", "90", "180", "270"), default="auto")
    parser.add_argument("--input-mirrored", action="store_true")
    parser.add_argument(
        "--threshold", type=float, choices=(0.30, 0.50), default=0.50,
        help="Shared hand detection/presence/tracking threshold",
    )
    parser.add_argument("--no-apple-auxiliary", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = MediaPipeV17Config(
        minimum_hand_detection_confidence=args.threshold,
        minimum_hand_presence_confidence=args.threshold,
        minimum_hand_tracking_confidence=args.threshold,
        include_apple_auxiliary=not args.no_apple_auxiliary,
    )
    detector = MediaPipeHybridDetector(args.model, config)
    rotation: str | int = args.rotation if args.rotation == "auto" else int(args.rotation)
    try:
        if args.input.is_dir():
            counts = extract_batch_mediapipe_v17(
                args.input,
                args.output,
                config,
                detector,
                rotation=rotation,
                input_mirrored=args.input_mirrored,
                resume=not args.no_resume,
            )
            print(json.dumps(counts, indent=2, sort_keys=True))
        else:
            result = extract_video_mediapipe_v17(
                args.input,
                config,
                detector,
                rotation=rotation,
                input_mirrored=args.input_mirrored,
            )
            if result is None:
                raise SystemExit("no usable hand detections")
            destination = args.output
            if destination.is_dir() or not destination.suffix:
                destination = destination / f"{args.input.stem}.v17.npz"
            save_mediapipe_v17_result(destination, result, config)
            print(destination)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
