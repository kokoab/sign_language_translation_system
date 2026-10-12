"""Contract for full-utterance, face-aligned visual-speech crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


SCHEMA_NAME = "slt_visual_speech_rgb_v17"
SCHEMA_VERSION = 1
SEQUENCE_LENGTH = 32
CROP_SIZE = 112
VIEW_NAMES = ("mouth", "lower_face", "full_face")


@dataclass(frozen=True)
class VisualSpeechV17Config:
    sequence_length: int = SEQUENCE_LENGTH
    crop_size: int = CROP_SIZE
    jpeg_quality: int = 90
    maximum_reference_frames: int = 96
    minimum_face_confidence: float = 0.15
    mouth_width_scale: float = 2.60
    lower_face_eye_scale: float = 2.35
    full_face_scale: float = 1.30
    minimum_crop_pixels: int = 36
    motion_quantile: float = 0.60
    interval_context_fraction: float = 0.10
    minimum_interval_fraction: float = 0.45

    def validate(self) -> None:
        if self.sequence_length != SEQUENCE_LENGTH or self.crop_size != CROP_SIZE:
            raise ValueError("visual-speech sequence length and crop size are schema-locked")
        if not 80 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [80, 100]")
        if self.maximum_reference_frames < self.sequence_length:
            raise ValueError("reference frame count cannot be shorter than output")
        if not 0.0 <= self.minimum_face_confidence <= 1.0:
            raise ValueError("minimum_face_confidence must be in [0, 1]")
        if min(self.mouth_width_scale, self.lower_face_eye_scale, self.full_face_scale) <= 0:
            raise ValueError("crop scales must be positive")
        if self.minimum_crop_pixels < 16:
            raise ValueError("minimum_crop_pixels is too small")
        if not 0.0 < self.motion_quantile < 1.0:
            raise ValueError("motion_quantile must be inside (0, 1)")
        if not 0.0 <= self.interval_context_fraction <= 0.5:
            raise ValueError("interval context fraction is unreasonable")
        if not 0.25 <= self.minimum_interval_fraction <= 1.0:
            raise ValueError("minimum interval fraction is unreasonable")


def schema_payload(config: VisualSpeechV17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "views": list(VIEW_NAMES),
        "decoded_shape": [
            config.sequence_length,
            len(VIEW_NAMES),
            config.crop_size,
            config.crop_size,
            3,
        ],
        "decoded_dtype": "uint8 BGR",
        "storage": "concatenated JPEG bytes plus explicit per-frame/view offsets",
        "temporal_contract": (
            "full-video face reference pass; mouth-motion interval with context and "
            "minimum-duration/full-utterance fallback; never hand-activity trimmed"
        ),
        "alignment_contract": "per-frame eye-line rotation with reflected real-pixel border",
        "missing_contract": "invalid views have offset=-1, length=0, zero pixels, valid=false",
        "audio_accessed": False,
        "config": asdict(config),
    }


def schema_fingerprint(config: VisualSpeechV17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
