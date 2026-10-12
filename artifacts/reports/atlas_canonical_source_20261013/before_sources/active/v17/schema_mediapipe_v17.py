"""Separately fingerprinted MediaPipe-hand candidate contract for the v17 bakeoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .schema_v17 import (
    CLIP_LENGTH,
    NODE_NAMES,
    NUM_CHANNELS,
    NUM_NODES,
    V17Config,
)


SCHEMA_NAME = "slt_mediapipe_hand_apple_aux_v17"
SCHEMA_VERSION = 1
HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
FEATURE_CHANNELS = (
    "x_body_relative",
    "y_body_relative",
    "hand_world_z_over_palm_else_scale_depth",
    "presence",
    "detector_confidence",
)


@dataclass(frozen=True)
class MediaPipeV17Config(V17Config):
    hand_model_sha256: str = HAND_MODEL_SHA256
    minimum_hand_detection_confidence: float = 0.30
    minimum_hand_presence_confidence: float = 0.30
    minimum_hand_tracking_confidence: float = 0.30
    include_apple_auxiliary: bool = True

    def validate(self) -> None:
        super().validate()
        for name in (
            "minimum_hand_detection_confidence",
            "minimum_hand_presence_confidence",
            "minimum_hand_tracking_confidence",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.hand_model_sha256 != HAND_MODEL_SHA256:
            raise ValueError("unreviewed MediaPipe hand model fingerprint")


def schema_payload(config: MediaPipeV17Config) -> dict[str, object]:
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
            "hands": "MediaPipe Hand Landmarker float16 full task",
            "hand_model_sha256": config.hand_model_sha256,
            "auxiliary": "Apple Vision body and face at configured intervals"
            if config.include_apple_auxiliary
            else "none",
            "hand_depth": "world Z relative to wrist divided by world palm length",
            "missing_values": "all spatial/depth/confidence values are exactly zero",
            "gap_policy": "bounded linear interpolation only; no extrapolation",
        },
    }


def schema_fingerprint(config: MediaPipeV17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
