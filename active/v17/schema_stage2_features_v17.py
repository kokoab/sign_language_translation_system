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
    window_stride: int | None = None
    maximum_source_frames: int = 256
    hand_frames_per_window: int = 16
    hand_views: int = 3
    minimum_tail_frames: int = 4

    def validate(self) -> None:
        if not 8 <= self.window_source_frames <= 32 or self.hand_frames_per_window != 16:
            raise ValueError("Stage-1 online windows require 8-32 source frames and 16 hand frames")
        stride = self.window_source_frames if self.window_stride is None else self.window_stride
        if not 1 <= stride <= self.window_source_frames:
            raise ValueError("window_stride must be within the source window")
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
    config_payload = asdict(config)
    stride = config_payload.pop("window_stride") or config.window_source_frames
    # Preserve the locked default fingerprint; only non-default overlap changes it.
    if stride != config.window_source_frames:
        config_payload["window_stride"] = stride
    return {
        "schema_name": "slt_stage2_windowed_multimodal_v17",
        "schema_version": 1,
        "config": config_payload,
        "landmark_window_shape": [32, 61, 5],
        "hand_window_shape": [16, 3],
        "landmark_schema_fingerprint": landmark_fingerprint(landmarks),
        "hand_rgb_schema_fingerprint": hand_rgb_fingerprint(hands),
        "temporal_contract": (
            "one orientation correction per source video; non-overlapping 32-source-frame "
            "windows; final tails shorter than four frames are dropped"
            if stride == config.window_source_frames == 32 else
            f"one orientation correction per source video; non-overlapping "
            f"{config.window_source_frames}-source-frame windows resampled to the fixed "
            "32-frame Stage-1 input; final tails shorter than four frames are dropped"
            if stride == config.window_source_frames else
            f"one orientation correction per source video; overlapping "
            f"{config.window_source_frames}-source-frame windows at stride {stride}, resampled to the "
            "fixed 32-frame Stage-1 input"
        ),
    }


def schema_fingerprint(config: Stage2FeatureV17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
