"""Single Stage-1 classifier over v17 landmarks and hand-RGB embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1V17
from .model_v17 import SLTStage1V17


def per_sample_zscore(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=-1, keepdim=True)
    return centered / centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)


@dataclass(frozen=True)
class UnifiedMultimodalV17Config:
    num_classes: int = 100
    feature_dim: int = 256
    hidden_dim: int = 384
    landmark_weight: float = 0.75
    hand_weight: float = 0.25
    dropout: float = 0.15

    def validate(self) -> None:
        if self.num_classes < 2 or self.feature_dim < 32 or self.hidden_dim < 32:
            raise ValueError("invalid unified model dimensions")
        if self.landmark_weight < 0 or self.hand_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if abs(self.landmark_weight + self.hand_weight - 1.0) > 1e-8:
            raise ValueError("fusion weights must sum to one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class UnifiedFusionHeadV17(nn.Module):
    """Zero-residual head that starts as the frozen 75/25 score fusion."""

    def __init__(self, config: UnifiedMultimodalV17Config | None = None):
        super().__init__()
        self.config = config or UnifiedMultimodalV17Config()
        self.config.validate()
        dim = self.config.feature_dim
        classes = self.config.num_classes
        self.landmark_norm = nn.LayerNorm(dim)
        self.hand_norm = nn.LayerNorm(dim)
        self.residual = nn.Sequential(
            nn.Linear(dim * 2 + classes * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, classes),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        landmark_features: torch.Tensor,
        hand_features: torch.Tensor,
        landmark_logits: torch.Tensor,
        hand_logits: torch.Tensor,
        *,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        landmark_scores = per_sample_zscore(landmark_logits)
        hand_scores = per_sample_zscore(hand_logits)
        base = (
            self.config.landmark_weight * landmark_scores
            + self.config.hand_weight * hand_scores
        )
        landmark = self.landmark_norm(landmark_features)
        hand = self.hand_norm(hand_features)
        features = torch.cat((landmark, hand), dim=-1)
        residual = self.gate(features) * self.residual(
            torch.cat((features, landmark_scores, hand_scores), dim=-1)
        )
        output = base + residual
        return (output, residual) if return_residual else output


class UnifiedMultimodalStage1V17(nn.Module):
    """One classifier graph containing frozen landmark/hand temporal encoders."""

    def __init__(
        self,
        landmark_model: SLTStage1V17,
        hand_model: HandMobileCLIP2Stage1V17,
        fusion_head: UnifiedFusionHeadV17,
    ):
        super().__init__()
        self.landmark_model = landmark_model
        self.hand_model = hand_model
        self.fusion_head = fusion_head

    def forward(
        self,
        landmarks: torch.Tensor,
        hand_embeddings: torch.Tensor,
        hand_valid: torch.Tensor,
        hand_boxes: torch.Tensor,
    ) -> torch.Tensor:
        landmark_logits, landmark_features = self.landmark_model(
            landmarks, return_embeddings=True
        )
        hand_features = self.hand_model.forward_features(
            hand_embeddings, hand_valid, hand_boxes
        )
        hand_logits = self.hand_model.classifier(hand_features)
        return self.fusion_head(
            landmark_features, hand_features, landmark_logits, hand_logits
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
