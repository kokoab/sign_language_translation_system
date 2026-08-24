"""MobileCLIP2 embedding contract for windowed Stage-2 hand crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .schema_mobileclip2_v17 import CHECKPOINT_SHA256, EMBEDDING_DIM, MODEL_NAME, PRETRAINED_TAG
from .schema_stage2_features_v17 import Stage2FeatureV17Config, schema_fingerprint as crop_fingerprint


@dataclass(frozen=True)
class Stage2HandMobileCLIP2V17Config:
    sequence_length_per_window: int = 16
    views: int = 3
    embedding_dim: int = EMBEDDING_DIM
    normalized: bool = True

    def validate(self) -> None:
        if (self.sequence_length_per_window, self.views, self.embedding_dim) != (16, 3, 512):
            raise ValueError("Stage-2 MobileCLIP2 hand contract changed")


def schema_payload(config: Stage2HandMobileCLIP2V17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": "slt_stage2_hand_mobileclip2_s0_v17",
        "schema_version": 1,
        "config": asdict(config),
        "per_window_shape": [16, 3, 512],
        "dtype": "float16",
        "stage2_crop_schema_fingerprint": crop_fingerprint(Stage2FeatureV17Config()),
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "missing_contract": "invalid views have zero embeddings and valid=false",
    }


def schema_fingerprint(config: Stage2HandMobileCLIP2V17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
