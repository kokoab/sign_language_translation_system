"""Small shared-frame visual-speech model for v17 mouth crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MouthRGBStage1Config:
    num_classes: int = 100
    dropout: float = 0.20

    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DepthwiseSeparable2D(nn.Module):
    def __init__(self, channels: int, output: int, expansion: int, spatial_stride: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(channels, expansion, 1, bias=False),
            nn.BatchNorm2d(expansion),
            nn.SiLU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expansion, expansion, 3, stride=spatial_stride,
                padding=1, groups=expansion, bias=False,
            ),
            nn.BatchNorm2d(expansion),
            nn.SiLU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expansion, output, 1, bias=False),
            nn.BatchNorm2d(output),
        )
        self.residual = channels == output and spatial_stride == 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.project(self.depthwise(self.expand(value)))
        return value + output if self.residual else output


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value).transpose(1, 2).contiguous()
        output = self.pointwise(F.silu(self.depthwise(normalized))).transpose(1, 2).contiguous()
        return value + output


class MouthRGBStage1(nn.Module):
    def __init__(self, config: MouthRGBStage1Config | None = None):
        super().__init__()
        self.config = config or MouthRGBStage1Config()
        self.config.validate()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
        )
        self.spatial = nn.Sequential(
            DepthwiseSeparable2D(24, 32, 48, 2),
            DepthwiseSeparable2D(32, 48, 72, 2),
            DepthwiseSeparable2D(48, 72, 108, 2),
            DepthwiseSeparable2D(72, 96, 144, 2),
        )
        self.temporal = nn.Sequential(
            TemporalBlock(96, 1, self.config.dropout),
            TemporalBlock(96, 2, self.config.dropout),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(96), nn.Dropout(self.config.dropout),
            nn.Linear(96, self.config.num_classes),
        )

    def forward(
        self, pixels: torch.Tensor, valid: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError("pixels must have shape [B,T,3,H,W]")
        if valid.shape != pixels.shape[:2]:
            raise ValueError("valid mask must have shape [B,T]")
        batch, frames = pixels.shape[:2]
        value = pixels.contiguous().view(batch * frames, *pixels.shape[2:])
        value = self.spatial(self.stem(value)).mean(dim=(-1, -2))
        value = value.reshape(batch, frames, -1)
        value = self.temporal(value)
        usable = valid.bool()
        has_valid = usable.any(dim=1, keepdim=True)
        usable = torch.where(has_valid, usable, torch.ones_like(usable))
        scores = self.attention(value).squeeze(-1)
        scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
        pooled = (value * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled)
        return (logits, pooled) if return_embeddings else logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
