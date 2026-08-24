"""Style-conditioned masked transition inpainting for genuine v17 trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .schema_v17 import NUM_CHANNELS, NUM_NODES


@dataclass(frozen=True)
class TransitionInpainterV17Config:
    frames: int = 32
    nodes: int = NUM_NODES
    channels: int = NUM_CHANNELS
    dim: int = 192
    depth: int = 4
    heads: int = 6
    dropout: float = 0.10

    @property
    def frame_dim(self) -> int:
        return self.nodes * self.channels

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def interpolate_masked_context(
    features: torch.Tensor, transition_mask: torch.Tensor
) -> torch.Tensor:
    """Fill each contiguous interior mask using its two genuine boundary frames."""
    if features.ndim != 4 or transition_mask.shape != features.shape[:2]:
        raise ValueError("features and transition mask have incompatible shapes")
    if transition_mask.dtype != torch.bool:
        raise ValueError("transition mask must be boolean")
    lengths = transition_mask.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("each row requires a non-empty masked interval")
    starts = transition_mask.to(torch.int64).argmax(dim=1)
    stops = starts + lengths
    if torch.any(starts == 0) or torch.any(stops >= features.shape[1]):
        raise ValueError("masked interval requires visible boundary frames")
    frame = torch.arange(features.shape[1], device=features.device)[None, :]
    expected_mask = (frame >= starts[:, None]) & (frame < stops[:, None])
    if not torch.equal(expected_mask, transition_mask):
        raise ValueError("transition mask must be one contiguous interval")
    row = torch.arange(len(features), device=features.device)
    left = features[row, starts - 1]
    right = features[row, stops]
    alpha = (
        (frame - starts[:, None] + 1).to(features.dtype)
        / (lengths[:, None] + 1).to(features.dtype)
    ).clamp(0.0, 1.0)
    interpolated = (
        left[:, None] * (1.0 - alpha[:, :, None, None])
        + right[:, None] * alpha[:, :, None, None]
    )
    return torch.where(transition_mask[:, :, None, None], interpolated, features)


class TransitionInpainterV17(nn.Module):
    """Reconstruct a missing contiguous motion interval from signer context.

    Style is inferred from visible frames of the same clip.  This makes the model
    capable of adapting to an unseen signer at inference without a learned signer ID.
    """

    def __init__(self, config: TransitionInpainterV17Config | None = None):
        super().__init__()
        self.config = config or TransitionInpainterV17Config()
        frame_dim = self.config.frame_dim
        self.input_projection = nn.Linear(frame_dim + 1, self.config.dim)
        self.style_projection = nn.Sequential(
            nn.Linear(frame_dim * 2, self.config.dim),
            nn.GELU(),
            nn.LayerNorm(self.config.dim),
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.config.frames, self.config.dim)
        )
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
        self.output_projection = nn.Linear(self.config.dim, frame_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        # Start at the strong interpolation baseline and learn only genuine-motion
        # residuals. This prevents a randomly initialized network from erasing the
        # two observed boundary conditions.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def visible_style(
        self, flattened: torch.Tensor, transition_mask: torch.Tensor
    ) -> torch.Tensor:
        visible = (~transition_mask).to(flattened.dtype).unsqueeze(-1)
        count = visible.sum(dim=1).clamp_min(1.0)
        mean = (flattened * visible).sum(dim=1) / count
        variance = (
            (flattened - mean.unsqueeze(1)).square() * visible
        ).sum(dim=1) / count
        return self.style_projection(torch.cat((mean, variance.sqrt()), dim=-1))

    def forward(
        self, features: torch.Tensor, transition_mask: torch.Tensor
    ) -> torch.Tensor:
        expected = (
            len(features), self.config.frames, self.config.nodes, self.config.channels
        )
        if tuple(features.shape) != expected:
            raise ValueError(f"unexpected transition tensor {tuple(features.shape)}")
        if tuple(transition_mask.shape) != expected[:2] or transition_mask.dtype != torch.bool:
            raise ValueError("transition mask must be boolean [batch, frames]")
        if not torch.all(transition_mask.any(dim=1)):
            raise ValueError("each row requires a non-empty masked interval")
        baseline = interpolate_masked_context(features, transition_mask)
        flattened = features.flatten(2)
        style = self.visible_style(flattened, transition_mask)
        corrupted = baseline.flatten(2)
        tokens = self.input_projection(torch.cat((
            corrupted, transition_mask.to(features.dtype).unsqueeze(-1)
        ), dim=-1))
        tokens = tokens + self.position + style.unsqueeze(1)
        tokens = self.sequence(tokens)
        residual = self.output_projection(self.output_norm(tokens)).reshape(expected)
        predicted = baseline + residual
        return torch.where(
            transition_mask[:, :, None, None], predicted, features
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
