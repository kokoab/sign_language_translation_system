"""Feature contract for Apple-guided, real-pixel hand crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


SCHEMA_NAME = "slt_apple_guided_hand_rgb_v17"
SCHEMA_VERSION = 1
SEQUENCE_LENGTH = 16
CROP_SIZE = 256
VIEW_NAMES = ("left", "right", "union")


@dataclass(frozen=True)
class HandRGBV17Config:
    sequence_length: int = SEQUENCE_LENGTH
    crop_size: int = CROP_SIZE
    jpeg_quality: int = 92
    hand_box_scale: float = 1.70
    union_box_scale: float = 1.20
    minimum_box_long_side_fraction: float = 0.14
    maximum_reference_frames: int = 96
    minimum_joint_count: int = 5

    def validate(self) -> None:
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length must remain {SEQUENCE_LENGTH}")
        if self.crop_size != CROP_SIZE:
            raise ValueError(f"crop_size must remain {CROP_SIZE}")
        if not 80 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [80, 100]")
        if self.hand_box_scale < 1.0 or self.union_box_scale < 1.0:
            raise ValueError("box scales cannot shrink the detected hand")
        if not 0.05 <= self.minimum_box_long_side_fraction <= 0.5:
            raise ValueError("minimum box fraction is unreasonable")
        if self.minimum_joint_count < 3:
            raise ValueError("minimum_joint_count must be at least three")


def schema_payload(config: HandRGBV17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "views": list(VIEW_NAMES),
        "decoded_shape": [config.sequence_length, len(VIEW_NAMES), 3, config.crop_size, config.crop_size],
        "decoded_dtype": "uint8 RGB",
        "storage": "concatenated JPEG bytes plus explicit offsets",
        "config": asdict(config),
        "missing_contract": "invalid views have offset=-1, length=0, zero pixels, and valid=false",
        "pixel_contract": "upright, unmirrored, real source pixels; no synthesized image content",
        "temporal_contract": "same Apple-v17 hand-activity interval as landmark/RGB baselines",
    }


def schema_fingerprint(config: HandRGBV17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
