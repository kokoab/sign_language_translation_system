"""Pre-pooled MobileCLIP2 spatial maps for hand-aware temporal fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .schema_hand_rgb_v17 import VIEW_NAMES, HandRGBV17Config, schema_fingerprint as crop_fingerprint
from .schema_mobileclip2_v17 import CHECKPOINT_SHA256, MODEL_NAME, PRETRAINED_TAG


SCHEMA_NAME = "slt_hand_spatial_mobileclip2_s0_v17"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HandSpatialMobileCLIP2V17Config:
    sequence_length: int = 16
    views: tuple[str, ...] = VIEW_NAMES
    channels: int = 512
    spatial_size: int = 8
    extraction_point: str = "fastvit_stage3_output_before_final_conv"

    def validate(self):
        if (self.sequence_length, self.channels, self.spatial_size) != (16, 512, 8):
            raise ValueError("MobileCLIP2 spatial contract changed")
        if tuple(self.views) != VIEW_NAMES:
            raise ValueError("view contract changed")


def schema_payload(config: HandSpatialMobileCLIP2V17Config):
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "shape": [config.sequence_length, len(config.views), config.channels, config.spatial_size, config.spatial_size],
        "dtype": "float16",
        "config": asdict(config),
        "crop_schema_fingerprint": crop_fingerprint(HandRGBV17Config()),
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "missing_contract": "invalid views have exact-zero maps and valid=false",
    }


def schema_fingerprint(config: HandSpatialMobileCLIP2V17Config):
    encoded = json.dumps(schema_payload(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
