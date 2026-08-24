"""Frozen MobileCLIP2 embeddings for high-resolution hand RGB views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .schema_hand_rgb_v17 import VIEW_NAMES, schema_fingerprint as crop_fingerprint, HandRGBV17Config
from .schema_mobileclip2_v17 import CHECKPOINT_SHA256, EMBEDDING_DIM, MODEL_NAME, PRETRAINED_TAG


SCHEMA_NAME = "slt_hand_mobileclip2_s0_v17"
SCHEMA_VERSION = 1
SEQUENCE_LENGTH = 16


@dataclass(frozen=True)
class HandMobileCLIP2V17Config:
    sequence_length: int = SEQUENCE_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    views: tuple[str, ...] = VIEW_NAMES
    normalize_embeddings: bool = True

    def validate(self) -> None:
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length must remain {SEQUENCE_LENGTH}")
        if self.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"embedding_dim must remain {EMBEDDING_DIM}")
        if tuple(self.views) != VIEW_NAMES:
            raise ValueError(f"views must remain {VIEW_NAMES}")


def schema_payload(config: HandMobileCLIP2V17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "shape": [config.sequence_length, len(config.views), config.embedding_dim],
        "dtype": "float16",
        "config": asdict(config),
        "crop_schema_fingerprint": crop_fingerprint(HandRGBV17Config()),
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "missing_contract": "invalid views have zero embeddings and valid=false",
    }


def schema_fingerprint(config: HandMobileCLIP2V17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
