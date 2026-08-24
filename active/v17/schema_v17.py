"""Versioned feature contract for the v17 extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


SCHEMA_NAME = "slt_apple_vision_landmarks_v17"
SCHEMA_VERSION = 1
CLIP_LENGTH = 32

NUM_HAND_NODES = 21
NUM_FACE_NODES = 15
NUM_BODY_NODES = 4
NUM_NODES = NUM_HAND_NODES * 2 + NUM_FACE_NODES + NUM_BODY_NODES
NUM_CHANNELS = 5

LHAND_START, LHAND_END = 0, 21
RHAND_START, RHAND_END = 21, 42
FACE_START, FACE_END = 42, 57
MOUTH_START, MOUTH_END = 49, 53
BODY_START, BODY_END = 57, 61

FEATURE_CHANNELS = (
    "x_body_relative",
    "y_body_relative",
    "relative_depth_log_scale",
    "presence",
    "confidence",
)

HAND_NODE_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
)
FACE_NODE_NAMES = (
    "left_pupil", "right_pupil",
    "left_brow_start", "left_brow_end",
    "right_brow_start", "right_brow_end",
    "nose_tip",
    "mouth_left", "mouth_right", "upper_lip", "lower_lip",
    "jaw_left", "chin", "jaw_right", "nose_bridge",
)
BODY_NODE_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
)
NODE_NAMES = tuple(
    [f"left_hand_{name}" for name in HAND_NODE_NAMES]
    + [f"right_hand_{name}" for name in HAND_NODE_NAMES]
    + list(FACE_NODE_NAMES)
    + list(BODY_NODE_NAMES)
)


@dataclass(frozen=True)
class V17Config:
    """Extractor parameters that must match between training and inference."""

    target_frames: int = CLIP_LENGTH
    maximum_source_frames: int = 96
    maximum_image_side: int = 1280
    body_interval: int = 8
    face_interval: int = 8
    minimum_point_confidence: float = 0.15
    hand_gap_frames: int = 3
    auxiliary_gap_frames: int = 16
    trim_to_hand_activity: bool = True
    trim_context_frames: int = 2
    minimum_detected_hand_frames: int = 2
    include_face: bool = True

    def validate(self) -> None:
        if self.target_frames < 4:
            raise ValueError("target_frames must be at least 4")
        if self.maximum_source_frames < self.target_frames:
            raise ValueError("maximum_source_frames cannot be less than target_frames")
        if self.maximum_image_side < 320:
            raise ValueError("maximum_image_side must be at least 320 pixels")
        if self.body_interval < 1 or self.face_interval < 1:
            raise ValueError("body_interval and face_interval must be positive")
        if not 0.0 <= self.minimum_point_confidence <= 1.0:
            raise ValueError("minimum_point_confidence must be in [0, 1]")
        if self.hand_gap_frames < 0 or self.auxiliary_gap_frames < 0:
            raise ValueError("gap limits cannot be negative")
        if self.trim_context_frames < 0:
            raise ValueError("trim_context_frames cannot be negative")


def schema_payload(config: V17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "shape": [config.target_frames, NUM_NODES, NUM_CHANNELS],
        "dtype": "float16",
        "feature_channels": list(FEATURE_CHANNELS),
        "node_names": list(NODE_NAMES),
        "config": asdict(config),
        "coordinate_contract": {
            "input": "upright, unmirrored image coordinates",
            "image_geometry": "isotropic pixels divided by longest image side",
            "origin": "per-frame shoulder midpoint; wrist fallback",
            "scale": "sequence-median shoulder width; palm-length fallback",
            "missing_values": "all spatial/depth values are exactly zero",
        },
    }


def schema_fingerprint(config: V17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
