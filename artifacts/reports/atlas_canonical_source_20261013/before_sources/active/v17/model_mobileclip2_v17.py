"""Compact temporal classifier for frozen MobileCLIP2-S0 frame embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model_v17 import SqueezeformerBlockV17
from .schema_mobileclip2_v17 import EMBEDDING_DIM, SEQUENCE_LENGTH


@dataclass(frozen=True)
class MobileCLIP2Stage1Config:
    num_classes: int = 100
    input_dim: int = EMBEDDING_DIM
    sequence_length: int = SEQUENCE_LENGTH
    dim: int = 256
    depth: int = 3
    heads: int = 8
    conv_kernel: int = 7
    dropout: float = 0.12
    head_dropout: float = 0.25
    drop_path: float = 0.08

    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if self.input_dim != EMBEDDING_DIM:
            raise ValueError(f"input_dim must be {EMBEDDING_DIM}")
        if self.sequence_length != SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length must be {SEQUENCE_LENGTH}")
        if self.dim < 32 or self.depth < 1 or self.dim % self.heads:
            raise ValueError("invalid temporal model dimensions")
        if self.conv_kernel < 3 or self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd and at least three")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MobileCLIP2Stage1V17(nn.Module):
    """Squeezeformer temporal head; the 11.4M image tower remains frozen."""

    def __init__(self, config: MobileCLIP2Stage1Config | None = None):
        super().__init__()
        self.config = config or MobileCLIP2Stage1Config()
        self.config.validate()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.config.sequence_length, self.config.dim)
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        rates = torch.linspace(0.0, self.config.drop_path, self.config.depth).tolist()
        self.blocks = nn.ModuleList(
            SqueezeformerBlockV17(
                self.config.dim,
                self.config.heads,
                self.config.conv_kernel,
                self.config.dropout,
                rates[index],
            )
            for index in range(self.config.depth)
        )
        self.frame_attention = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim // 4),
            nn.GELU(),
            nn.Linear(self.config.dim // 4, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.dim, self.config.num_classes),
        )

    def forward_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.sequence_length,
            self.config.input_dim,
        )
        if embeddings.ndim != 3 or tuple(embeddings.shape[1:]) != expected:
            raise ValueError(f"expected [B, {expected[0]}, {expected[1]}], got {tuple(embeddings.shape)}")
        value = self.input_projection(embeddings) + self.position
        for block in self.blocks:
            value = block(value)
        weights = torch.softmax(self.frame_attention(value), dim=1)
        return (value * weights).sum(dim=1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(embeddings))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def make_checkpoint(
    model: MobileCLIP2Stage1V17,
    state_dict: dict[str, torch.Tensor],
    *,
    epoch: int,
    validation_metrics: dict[str, float],
    label_to_index: dict[str, int],
    manifest_sha256: str,
    schema_fingerprint: str,
) -> dict[str, object]:
    return {
        "format": "slt_stage1_mobileclip2_v17",
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": state_dict,
        "epoch": int(epoch),
        "validation_metrics": validation_metrics,
        "label_to_index": label_to_index,
        "manifest_sha256": manifest_sha256,
        "schema_fingerprint": schema_fingerprint,
        "test_evaluated": False,
    }
