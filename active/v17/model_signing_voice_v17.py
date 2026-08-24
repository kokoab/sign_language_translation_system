"""Content/style-disentangled isolated motion generator for a v17 signing voice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .schema_v17 import NUM_CHANNELS, NUM_NODES


@dataclass(frozen=True)
class SigningVoiceV17Config:
    classes: int = 100
    frames: int = 32
    nodes: int = NUM_NODES
    channels: int = NUM_CHANNELS
    dim: int = 128
    style_dim: int = 32
    encoder_depth: int = 2
    decoder_depth: int = 3
    heads: int = 4
    dropout: float = 0.10
    maximum_residual: float = 1.25

    @property
    def frame_dim(self) -> int:
        return self.nodes * self.channels

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SigningStyleEncoderV17(nn.Module):
    """Encode signer manner from a reference gloss, without receiving its label."""

    def __init__(self, config: SigningVoiceV17Config):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.frame_dim)
        self.input_projection = nn.Linear(config.frame_dim, config.dim)
        self.position = nn.Parameter(torch.zeros(1, config.frames, config.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, config.encoder_depth)
        self.output_norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim * 2, config.style_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        expected = (
            len(reference), self.config.frames, self.config.nodes, self.config.channels
        )
        if tuple(reference.shape) != expected:
            raise ValueError(f"unexpected signing-style reference {tuple(reference.shape)}")
        flattened = reference.flatten(2)
        tokens = self.input_projection(self.input_norm(flattened)) + self.position
        tokens = self.sequence(tokens)
        tokens = self.output_norm(tokens)
        pooled = torch.cat((tokens.mean(dim=1), tokens.std(dim=1)), dim=-1)
        return F.normalize(self.output(pooled), dim=-1)


class SigningVoiceGeneratorV17(nn.Module):
    """Generate one complete isolated gloss from content plus continuous style."""

    def __init__(self, config: SigningVoiceV17Config | None = None):
        super().__init__()
        self.config = config or SigningVoiceV17Config()
        self.style_encoder = SigningStyleEncoderV17(self.config)
        self.prototype_projection = nn.Linear(self.config.frame_dim, self.config.dim)
        self.content = nn.Embedding(self.config.classes, self.config.dim)
        self.style_projection = nn.Sequential(
            nn.Linear(self.config.style_dim, self.config.dim),
            nn.GELU(),
            nn.Linear(self.config.dim, self.config.dim),
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
        self.decoder = nn.TransformerEncoder(layer, self.config.decoder_depth)
        self.output_norm = nn.LayerNorm(self.config.dim)
        self.spatial_residual = nn.Linear(
            self.config.dim, self.config.nodes * 3
        )
        # A direct, zero-initialized low-parameter route preserves a signer-specific
        # motion basis that repeated Transformer layer normalization cannot wash out.
        self.style_spatial_residual = nn.Linear(
            self.config.style_dim,
            self.config.frames * self.config.nodes * 3,
            bias=False,
        )
        self.style_classifier: nn.Linear | None = None
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.spatial_residual.weight)
        nn.init.zeros_(self.spatial_residual.bias)
        nn.init.zeros_(self.style_spatial_residual.weight)

    def install_style_classifier(self, voices: int) -> None:
        self.style_classifier = nn.Linear(self.config.style_dim, voices)

    def encode_style(self, reference: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(reference)

    def generate_from_style(
        self,
        prototypes: torch.Tensor,
        targets: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            len(prototypes), self.config.frames, self.config.nodes, self.config.channels
        )
        if tuple(prototypes.shape) != expected:
            raise ValueError(f"unexpected content prototype {tuple(prototypes.shape)}")
        if targets.shape != (len(prototypes),) or style.shape != (
            len(prototypes), self.config.style_dim
        ):
            raise ValueError("content/style batch alignment changed")
        tokens = self.prototype_projection(prototypes.flatten(2))
        tokens = (
            tokens
            + self.content(targets)[:, None]
            + self.style_projection(style)[:, None]
            + self.position
        )
        tokens = self.decoder(tokens)
        decoded = self.spatial_residual(self.output_norm(tokens)).reshape(
            len(tokens), self.config.frames, self.config.nodes, 3
        )
        style_motion = self.style_spatial_residual(style).reshape(
            len(tokens), self.config.frames, self.config.nodes, 3
        )
        residual = self.config.maximum_residual * torch.tanh(decoded + style_motion)
        output = prototypes.clone()
        spatial = prototypes[..., :3] + residual
        present = prototypes[..., 3:4] > 0
        output[..., :3] = torch.where(present, spatial, torch.zeros_like(spatial))
        return output

    def forward(
        self,
        prototypes: torch.Tensor,
        targets: torch.Tensor,
        references: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        style = self.encode_style(references)
        return self.generate_from_style(prototypes, targets, style), style

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
