"""ImageNet-pretrained MobileNetV3-Small temporal mouth classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .model_mouth_rgb_v17 import TemporalBlock


PRETRAINED_WEIGHT_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


@dataclass(frozen=True)
class MouthMobileNetV17Config:
    num_classes: int = 100
    temporal_dim: int = 128
    dropout: float = 0.20

    def validate(self) -> None:
        if self.num_classes < 2 or self.temporal_dim < 32:
            raise ValueError("invalid MobileNet mouth classifier dimensions")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MouthMobileNetV17(nn.Module):
    def __init__(self, config: MouthMobileNetV17Config | None = None):
        super().__init__()
        self.config = config or MouthMobileNetV17Config()
        self.config.validate()
        pretrained = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = pretrained.features
        self.projection = nn.Sequential(
            nn.Linear(576, self.config.temporal_dim),
            nn.LayerNorm(self.config.temporal_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.temporal = nn.Sequential(
            TemporalBlock(self.config.temporal_dim, 1, self.config.dropout),
            TemporalBlock(self.config.temporal_dim, 2, self.config.dropout),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(self.config.temporal_dim),
            nn.Linear(self.config.temporal_dim, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.temporal_dim), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.temporal_dim, self.config.num_classes),
        )
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, pixels: torch.Tensor, valid: torch.Tensor):
        if pixels.ndim != 5 or pixels.shape[2] != 3 or valid.shape != pixels.shape[:2]:
            raise ValueError("expected [B,T,3,H,W] pixels and [B,T] validity")
        batch, frames = pixels.shape[:2]
        value = pixels.contiguous().view(batch * frames, *pixels.shape[2:])
        value = ((value + 1.0) * 0.5 - self.image_mean) / self.image_std
        value = self.backbone(value).mean(dim=(-1, -2))
        value = self.projection(value).reshape(batch, frames, -1)
        value = self.temporal(value)
        usable = valid.bool()
        usable = torch.where(
            usable.any(dim=1, keepdim=True), usable, torch.ones_like(usable)
        )
        scores = self.attention(value).squeeze(-1)
        scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
        pooled = (value * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
