"""Contract for Apple-guided real-pixel face/mouth crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


SCHEMA_NAME = "slt_apple_guided_mouth_rgb_v17"
SCHEMA_VERSION = 1
SEQUENCE_LENGTH = 16
CROP_SIZE = 96


@dataclass(frozen=True)
class MouthRGBV17Config:
    sequence_length: int = SEQUENCE_LENGTH
    crop_size: int = CROP_SIZE
    jpeg_quality: int = 90
    minimum_face_confidence: float = 0.15
    mouth_width_scale: float = 2.50
    jaw_width_scale: float = 0.65
    minimum_crop_pixels: int = 32

    def validate(self) -> None:
        if self.sequence_length != SEQUENCE_LENGTH or self.crop_size != CROP_SIZE:
            raise ValueError("mouth RGB sequence length and crop size are schema-locked")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if not 0.0 <= self.minimum_face_confidence <= 1.0:
            raise ValueError("minimum_face_confidence must be in [0, 1]")
        if self.mouth_width_scale <= 0 or self.jaw_width_scale <= 0:
            raise ValueError("crop scales must be positive")
        if self.minimum_crop_pixels < 8:
            raise ValueError("minimum_crop_pixels must be at least 8")


def schema_payload(config: MouthRGBV17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "decoded_shape": [config.sequence_length, config.crop_size, config.crop_size, 3],
        "color_order": "BGR",
        "missing_crop": "valid=false and decoded pixels exactly zero",
        "config": asdict(config),
    }


def schema_fingerprint(config: MouthRGBV17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
