"""Signer-context transition-span prediction for v17 landmark trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .schema_v17 import NUM_CHANNELS, NUM_NODES


@dataclass(frozen=True)
class TransitionSpanV17Config:
    context_frames_per_side: int = 8
    nodes: int = NUM_NODES
    channels: int = NUM_CHANNELS
    minimum_span: int = 4
    maximum_span: int = 12
    dim: int = 128
    depth: int = 2
    heads: int = 4
    dropout: float = 0.10

    @property
    def context_frames(self) -> int:
        return self.context_frames_per_side * 2

    @property
    def frame_dim(self) -> int:
        return self.nodes * self.channels

    @property
    def classes(self) -> int:
        return self.maximum_span - self.minimum_span + 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TransitionSpanPredictorV17(nn.Module):
    """Infer transition duration from boundary motion and local signer rhythm."""

    def __init__(self, config: TransitionSpanV17Config | None = None):
        super().__init__()
        self.config = config or TransitionSpanV17Config()
        self.input_projection = nn.Linear(self.config.frame_dim, self.config.dim)
        self.position = nn.Parameter(
            torch.zeros(1, self.config.context_frames, self.config.dim)
        )
        self.side = nn.Parameter(torch.zeros(1, 2, self.config.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, self.config.depth)
        self.output_norm = nn.LayerNorm(self.config.dim)
        self.output = nn.Linear(self.config.dim, self.config.classes)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.side, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        expected = (
            len(context), self.config.context_frames,
            self.config.nodes, self.config.channels,
        )
        if tuple(context.shape) != expected:
            raise ValueError(f"unexpected transition context {tuple(context.shape)}")
        tokens = self.input_projection(context.flatten(2)) + self.position
        split = self.config.context_frames_per_side
        tokens = tokens + torch.cat((
            self.side[:, 0:1].expand(-1, split, -1),
            self.side[:, 1:2].expand(-1, split, -1),
        ), dim=1)
        tokens = self.sequence(tokens)
        return self.output(self.output_norm(tokens.mean(dim=1)))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def endpoint_only_context(context: torch.Tensor) -> torch.Tensor:
    """Remove local velocity/rhythm while preserving both boundary poses."""
    if context.ndim != 4 or context.shape[1] % 2:
        raise ValueError("context must have an even number of frames")
    split = context.shape[1] // 2
    left = context[:, split - 1:split]
    right = context[:, split:split + 1]
    return torch.cat((
        left.expand(-1, split, -1, -1),
        right.expand(-1, split, -1, -1),
    ), dim=1)


def kinematic_span(context: torch.Tensor, minimum: int = 4, maximum: int = 12) -> torch.Tensor:
    """Estimate elapsed frames from boundary distance and observed signer speed."""
    if context.ndim != 4 or context.shape[1] % 2:
        raise ValueError("context must have an even number of frames")
    split = context.shape[1] // 2
    left = context[:, :split]
    right = context[:, split:]
    left_boundary = left[:, -1]
    right_boundary = right[:, 0]
    boundary_present = (left_boundary[..., 3] > 0) & (right_boundary[..., 3] > 0)
    boundary_distance = torch.linalg.vector_norm(
        right_boundary[..., :3] - left_boundary[..., :3], dim=-1
    )
    boundary_distance = (
        (boundary_distance * boundary_present).sum(dim=1)
        / boundary_present.sum(dim=1).clamp_min(1)
    )

    speeds = []
    weights = []
    for side in (left, right):
        present = (side[:, 1:, :, 3] > 0) & (side[:, :-1, :, 3] > 0)
        velocity = torch.linalg.vector_norm(
            side[:, 1:, :, :3] - side[:, :-1, :, :3], dim=-1
        )
        speeds.append((velocity * present).sum(dim=(1, 2)))
        weights.append(present.sum(dim=(1, 2)))
    observed_speed = (
        torch.stack(speeds).sum(dim=0)
        / torch.stack(weights).sum(dim=0).clamp_min(1)
    )
    estimate = torch.round(boundary_distance / observed_speed.clamp_min(1e-4) - 1.0)
    return estimate.clamp(minimum, maximum).to(torch.long)
