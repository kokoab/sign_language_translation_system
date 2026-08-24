"""View-aware temporal classifier for high-resolution MobileCLIP2 hand crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model_v17 import SqueezeformerBlockV17
from .schema_hand_mobileclip2_v17 import SEQUENCE_LENGTH
from .schema_hand_rgb_v17 import VIEW_NAMES
from .schema_mobileclip2_v17 import EMBEDDING_DIM


@dataclass(frozen=True)
class HandMobileCLIP2Stage1Config:
    num_classes: int = 100
    input_dim: int = EMBEDDING_DIM
    sequence_length: int = SEQUENCE_LENGTH
    num_views: int = len(VIEW_NAMES)
    dim: int = 256
    depth: int = 3
    heads: int = 8
    conv_kernel: int = 7
    dropout: float = 0.12
    head_dropout: float = 0.25
    drop_path: float = 0.08

    def validate(self) -> None:
        if self.num_classes < 2 or self.input_dim != EMBEDDING_DIM:
            raise ValueError("invalid class or embedding dimensions")
        if self.sequence_length != SEQUENCE_LENGTH or self.num_views != len(VIEW_NAMES):
            raise ValueError("hand sequence/view contract changed")
        if self.dim < 32 or self.depth < 1 or self.dim % self.heads:
            raise ValueError("invalid temporal dimensions")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HandMobileCLIP2Stage1V17(nn.Module):
    def __init__(self, config: HandMobileCLIP2Stage1Config | None = None):
        super().__init__()
        self.config = config or HandMobileCLIP2Stage1Config()
        self.config.validate()
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.box_projection = nn.Sequential(
            nn.Linear(5, self.config.dim), nn.GELU(), nn.Linear(self.config.dim, self.config.dim)
        )
        self.view_embedding = nn.Parameter(torch.zeros(1, 1, self.config.num_views, self.config.dim))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        self.view_attention = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Linear(self.config.dim, self.config.dim // 4),
            nn.GELU(),
            nn.Linear(self.config.dim // 4, 1),
        )
        self.position = nn.Parameter(torch.zeros(1, self.config.sequence_length, self.config.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        rates = torch.linspace(0.0, self.config.drop_path, self.config.depth).tolist()
        self.blocks = nn.ModuleList(
            SqueezeformerBlockV17(
                self.config.dim, self.config.heads, self.config.conv_kernel,
                self.config.dropout, rates[index],
            )
            for index in range(self.config.depth)
        )
        self.frame_attention = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim // 4), nn.GELU(), nn.Linear(self.config.dim // 4, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.dim, self.config.num_classes),
        )

    def encode_frames(
        self, embeddings: torch.Tensor, valid: torch.Tensor, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_embeddings = (
            self.config.sequence_length, self.config.num_views, self.config.input_dim
        )
        if embeddings.ndim != 4 or tuple(embeddings.shape[1:]) != expected_embeddings:
            raise ValueError(f"unexpected embeddings {tuple(embeddings.shape)}")
        if tuple(valid.shape) != tuple(embeddings.shape[:3]):
            raise ValueError("valid mask shape mismatch")
        if tuple(boxes.shape) != tuple(embeddings.shape[:3]) + (4,):
            raise ValueError("box shape mismatch")
        valid_float = valid.to(embeddings.dtype)
        box_input = torch.cat((boxes, valid_float.unsqueeze(-1)), dim=-1)
        tokens = (
            self.embedding_projection(embeddings)
            + self.box_projection(box_input)
            + self.view_embedding
        ) * valid_float.unsqueeze(-1)
        scores = self.view_attention(tokens).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=2) * valid_float
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        frames = (tokens * weights.unsqueeze(-1)).sum(dim=2) + self.position
        frame_valid = valid.any(dim=2)
        frames = frames * frame_valid.unsqueeze(-1)
        for block in self.blocks:
            frames = block(frames) * frame_valid.unsqueeze(-1)
        return frames, frame_valid

    def forward_features(
        self, embeddings: torch.Tensor, valid: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        frames, frame_valid = self.encode_frames(embeddings, valid, boxes)
        frame_scores = self.frame_attention(frames).squeeze(-1).masked_fill(~frame_valid, -1e4)
        frame_weights = torch.softmax(frame_scores, dim=1) * frame_valid
        frame_weights = frame_weights / frame_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (frames * frame_weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, embeddings: torch.Tensor, valid: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(embeddings, valid, boxes))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def make_checkpoint(model, state_dict, **metadata):
    return {
        "format": "slt_stage1_hand_mobileclip2_v17",
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": state_dict,
        "test_evaluated": False,
        **metadata,
    }
