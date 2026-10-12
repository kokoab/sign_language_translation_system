"""Frozen feature contract for the v17 MobileCLIP2-S0 RGB challenger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


SCHEMA_NAME = "slt_mobileclip2_s0_rgb_v17"
SCHEMA_VERSION = 1
MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED_TAG = "dfndr2b"
CHECKPOINT_SHA256 = (
    "ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95"
)
SEQUENCE_LENGTH = 16
EMBEDDING_DIM = 512
INPUT_SIZE = 256


@dataclass(frozen=True)
class MobileCLIP2V17Config:
    sequence_length: int = SEQUENCE_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    input_size: int = INPUT_SIZE
    maximum_reference_frames: int = 96
    square_policy: str = "aspect_preserving_zero_letterbox"
    temporal_policy: str = "apple_v17_hand_activity_then_uniform"
    normalize_embeddings: bool = True

    def validate(self) -> None:
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length must remain {SEQUENCE_LENGTH}")
        if self.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"embedding_dim must remain {EMBEDDING_DIM}")
        if self.input_size != INPUT_SIZE:
            raise ValueError(f"input_size must remain {INPUT_SIZE}")
        if self.maximum_reference_frames < self.sequence_length:
            raise ValueError("maximum_reference_frames is too small")


def schema_payload(config: MobileCLIP2V17Config) -> dict[str, object]:
    config.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "shape": [config.sequence_length, config.embedding_dim],
        "dtype": "float16",
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config": asdict(config),
        "input_contract": {
            "pixels": "upright, unmirrored RGB",
            "spatial": "preserve aspect ratio and zero-letterbox to 256x256",
            "temporal": "sample the frozen Apple-v17 hand-activity interval",
            "normalization": "OpenCLIP dfndr2b mean=(0,0,0), std=(1,1,1)",
        },
    }


def schema_fingerprint(config: MobileCLIP2V17Config) -> str:
    encoded = json.dumps(
        schema_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
