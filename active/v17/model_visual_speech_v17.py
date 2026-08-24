"""Auto-AVSR-pretrained visual frontend with an isolated-sign temporal head.

The frontend structure follows the Apache-2.0 Auto-AVSR implementation by
Pingchuan Ma et al.: https://github.com/mpc001/auto_avsr
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model_v17 import SqueezeformerBlockV17


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.SiLU(inplace=True)
        self.downsample = downsample

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.relu1(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu2(value + residual)


class _ResNet18Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(_BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        return self.avgpool(value).flatten(1)


class AutoAVSRVisualFrontend(nn.Module):
    """Exact key-compatible 3D stem plus per-frame ResNet-18 visual frontend."""

    def __init__(self):
        super().__init__()
        self.trunk = _ResNet18Trunk()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                padding=(2, 3, 3), bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
        )
        # Auto-AVSR uses MaxPool3d((1,3,3)); its temporal kernel is exactly one.
        # Per-frame MaxPool2d is mathematically identical and supported by Apple MPS.
        self.spatial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 5 or pixels.shape[2] != 1:
            raise ValueError("expected [B,T,1,H,W] grayscale visual-speech pixels")
        batch = pixels.shape[0]
        value = self.frontend3D(pixels.transpose(1, 2))
        frames = value.shape[2]
        value = value.transpose(1, 2).reshape(
            batch * frames, value.shape[1], value.shape[3], value.shape[4]
        )
        value = self.spatial_pool(value)
        return self.trunk(value).reshape(batch, frames, 512)


@dataclass(frozen=True)
class VisualSpeechTeacherV17Config:
    num_classes: int = 100
    dim: int = 256
    depth: int = 2
    heads: int = 8
    dropout: float = 0.15
    head_dropout: float = 0.25

    def validate(self) -> None:
        if self.num_classes < 2 or self.dim < 64 or self.depth < 1:
            raise ValueError("invalid visual-speech teacher dimensions")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VisualSpeechTeacherV17(nn.Module):
    def __init__(self, config: VisualSpeechTeacherV17Config | None = None):
        super().__init__()
        self.config = config or VisualSpeechTeacherV17Config()
        self.config.validate()
        self.frontend = AutoAVSRVisualFrontend()
        self.projection = nn.Sequential(
            nn.Linear(512, self.config.dim),
            nn.LayerNorm(self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.position = nn.Parameter(torch.zeros(1, 32, self.config.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.temporal = nn.ModuleList(
            SqueezeformerBlockV17(
                self.config.dim, self.config.heads, 7, self.config.dropout,
                0.04 * index / max(self.config.depth - 1, 1),
            )
            for index in range(self.config.depth)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim // 4),
            nn.GELU(),
            nn.Linear(self.config.dim // 4, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.dim, self.config.num_classes),
        )

    def forward(
        self, pixels: torch.Tensor, valid: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if valid.shape != pixels.shape[:2] or pixels.shape[1] > self.position.shape[1]:
            raise ValueError("visual-speech validity or sequence length mismatch")
        return self.forward_features(
            self.frontend(pixels), valid, return_embeddings=return_embeddings
        )

    def forward_features(
        self, features: torch.Tensor, valid: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 3 or features.shape[-1] != 512
            or valid.shape != features.shape[:2]
            or features.shape[1] > self.position.shape[1]
        ):
            raise ValueError("expected [B,T,512] frontend features and [B,T] validity")
        value = self.projection(features)
        value = value + self.position[:, : value.shape[1]]
        for block in self.temporal:
            value = block(value)
        usable = valid.bool()
        usable = torch.where(
            usable.any(dim=1, keepdim=True), usable, torch.ones_like(usable)
        )
        scores = self.attention(value).squeeze(-1)
        scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
        pooled = (value * F.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled)
        return (logits, pooled) if return_embeddings else logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class MultiViewVisualSpeechV17Config:
    num_classes: int = 100
    dim: int = 256
    view_dim: int = 128
    views: int = 2
    depth: int = 2
    heads: int = 8
    dropout: float = 0.15
    head_dropout: float = 0.25

    def validate(self) -> None:
        if (
            self.num_classes < 2 or self.dim < 64 or self.view_dim < 32
            or self.views != 2 or self.depth < 1
        ):
            raise ValueError("invalid multi-view visual-speech dimensions")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MultiViewVisualSpeechHeadV17(nn.Module):
    """Learned per-frame mouth/lower-face fusion over frozen frontend features."""

    def __init__(self, config: MultiViewVisualSpeechV17Config | None = None):
        super().__init__()
        self.config = config or MultiViewVisualSpeechV17Config()
        self.config.validate()
        self.view_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(512, self.config.view_dim),
                nn.LayerNorm(self.config.view_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            for _ in range(self.config.views)
        )
        fused_input = self.config.views * self.config.view_dim + self.config.views
        self.fusion = nn.Sequential(
            nn.Linear(fused_input, self.config.dim),
            nn.LayerNorm(self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.position = nn.Parameter(torch.zeros(1, 32, self.config.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.temporal = nn.ModuleList(
            SqueezeformerBlockV17(
                self.config.dim, self.config.heads, 7, self.config.dropout,
                0.04 * index / max(self.config.depth - 1, 1),
            )
            for index in range(self.config.depth)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim // 4),
            nn.GELU(),
            nn.Linear(self.config.dim // 4, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.dim, self.config.num_classes),
        )

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 4 or features.shape[-2:] != (self.config.views, 512)
            or valid.shape != features.shape[:3]
            or features.shape[1] > self.position.shape[1]
        ):
            raise ValueError(
                "expected [B,T,2,512] frontend features and [B,T,2] validity"
            )
        usable_views = valid.bool()
        projected = []
        for index, projection in enumerate(self.view_projections):
            value = projection(features[:, :, index])
            projected.append(value * usable_views[:, :, index, None])
        value = torch.cat(
            projected + [usable_views.to(features.dtype)], dim=-1
        )
        value = self.fusion(value) + self.position[:, : features.shape[1]]
        for block in self.temporal:
            value = block(value)
        usable_frames = usable_views.any(dim=-1)
        usable_frames = torch.where(
            usable_frames.any(dim=1, keepdim=True),
            usable_frames,
            torch.ones_like(usable_frames),
        )
        scores = self.attention(value).squeeze(-1)
        scores = scores.masked_fill(
            ~usable_frames, torch.finfo(scores.dtype).min
        )
        pooled = (
            value * F.softmax(scores, dim=1).unsqueeze(-1)
        ).sum(dim=1)
        logits = self.classifier(pooled)
        return (logits, pooled) if return_embeddings else logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_auto_avsr_frontend(
    frontend: AutoAVSRVisualFrontend, checkpoint_path: str
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("model_state_dict", checkpoint)
    candidates: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        normalized = str(key)
        for prefix in ("model.frontend.", "frontend."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        if normalized.startswith("trunk.") or normalized.startswith("frontend3D."):
            candidates[normalized] = value
    result = frontend.load_state_dict(candidates, strict=True)
    return {
        "loaded_keys": len(candidates),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
