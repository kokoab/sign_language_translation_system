"""Feature contract for windowed multimodal v17 Stage-2 phrase archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .schema_hand_rgb_v17 import HandRGBV17Config, schema_fingerprint as hand_rgb_fingerprint
from .schema_v17 import V17Config, schema_fingerprint as landmark_fingerprint


@dataclass(frozen=True)
class Stage2FeatureV17Config:
    window_source_frames: int = 32
    maximum_source_frames: int = 256
    hand_frames_per_window: int = 16
    hand_views: int = 3
    minimum_tail_frames: int = 4

    def validate(self) -> None:
        if self.window_source_frames != 32 or self.hand_frames_per_window != 16:
            raise ValueError("Stage-1 temporal contracts require 32 landmark and 16 hand frames")
        if self.maximum_source_frames < self.window_source_frames:
            raise ValueError("maximum_source_frames is too small")
        if not 4 <= self.minimum_tail_frames <= self.window_source_frames:
            raise ValueError("invalid minimum tail length")


def landmark_config() -> V17Config:
    return V17Config(
        target_frames=32,
        maximum_source_frames=32,
        trim_to_hand_activity=False,
    )


def schema_payload(config: Stage2FeatureV17Config) -> dict[str, object]:
    config.validate()
    landmarks = landmark_config()
    hands = HandRGBV17Config()
    return {
        "schema_name": "slt_stage2_windowed_multimodal_v17",
        "schema_version": 1,
        "config": asdict(config),
        "landmark_window_shape": [32, 61, 5],
        "hand_window_shape": [16, 3],
        "landmark_schema_fingerprint": landmark_fingerprint(landmarks),
        "hand_rgb_schema_fingerprint": hand_rgb_fingerprint(hands),
        "temporal_contract": (
            "one orientation correction per source video; non-overlapping 32-source-frame "
            "windows; final tails shorter than four frames are dropped"
        ),
    }


def schema_fingerprint(config: Stage2FeatureV17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
