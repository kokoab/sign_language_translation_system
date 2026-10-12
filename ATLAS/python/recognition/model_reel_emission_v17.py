"""A frozen 100-gloss Stage-1 model with a tiny learned temporal emission head."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config


@dataclass(frozen=True)
class ReelEmissionHeadConfig:
    stage1_dim: int = 256
    num_glosses: int = 100
    hidden_dim: int = 128
    dropout: float = 0.10

    @property
    def input_dim(self) -> int:
        return self.stage1_dim * 5 + self.num_glosses

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def pool_stage1_encoded(
    model: SLTStage1V17, encoded: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    if model.config.static_hand_token != "none":
        raise ValueError("reel emission wrapper currently requires static_hand_token=none")
    scores = model.frame_attention(encoded).squeeze(-1)
    has_active = active.any(dim=1, keepdim=True)
    usable = torch.where(has_active, active, torch.ones_like(active))
    scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=1)
    return (encoded * weights.unsqueeze(-1)).sum(dim=1)


def reel_temporal_summary(
    encoded: torch.Tensor, class_logits: torch.Tensor
) -> torch.Tensor:
    """Preserve coarse temporal order that ordinary Stage-1 pooling discards."""
    if encoded.ndim != 3 or encoded.shape[1] != 32:
        raise ValueError("expected a [B, 32, D] Stage-1 sequence")
    quarters = [encoded[:, start : start + 8].mean(dim=1) for start in range(0, 32, 8)]
    endpoint_change = encoded[:, 24:].mean(dim=1) - encoded[:, :8].mean(dim=1)
    normalized_logits = (
        class_logits - class_logits.mean(dim=1, keepdim=True)
    ) / class_logits.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.cat((*quarters, endpoint_change, normalized_logits), dim=1)


class ReelEmissionHeadV17(nn.Module):
    def __init__(self, config: ReelEmissionHeadConfig | None = None):
        super().__init__()
        self.config = config or ReelEmissionHeadConfig()
        self.network = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.network(summary)


class ReelEmissionStage1V17(nn.Module):
    """Return the untouched 100 gloss logits plus one ``NO_EMIT`` logit."""

    def __init__(self, base: SLTStage1V17, emission_head: ReelEmissionHeadV17):
        super().__init__()
        if base.config.num_classes != emission_head.config.num_glosses:
            raise ValueError("base and emission-head class counts disagree")
        if base.config.dim != emission_head.config.stage1_dim:
            raise ValueError("base and emission-head dimensions disagree")
        self.base = base
        self.emission_head = emission_head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, active = self.base.encode(features)
        pooled = pool_stage1_encoded(self.base, encoded, active)
        class_logits = self.base.classifier(pooled)
        summary = reel_temporal_summary(encoded, class_logits)
        no_emit = self.emission_head(summary)
        return torch.cat((class_logits, no_emit), dim=1)


def load_reel_emission_checkpoint(
    checkpoint: dict[str, object]
) -> ReelEmissionStage1V17:
    if checkpoint.get("format") != "slt_stage1_reel_emission_v17":
        raise ValueError("not a v17 Reel emission checkpoint")
    base = SLTStage1V17(Stage1V17Config(**checkpoint["base_model_config"]))
    base.load_state_dict(checkpoint["base_model_state_dict"], strict=True)
    head = ReelEmissionHeadV17(
        ReelEmissionHeadConfig(**checkpoint["emission_head_config"])
    )
    head.load_state_dict(checkpoint["emission_head_state_dict"], strict=True)
    return ReelEmissionStage1V17(base, head).eval()
