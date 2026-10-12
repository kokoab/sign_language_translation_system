"""Explicit Stage-1 landmark architectures beyond SLTStage1V17, for matched downstream chains.

The flat Transformer from the family benchmark (scripts/benchmark_stage1_families_v17.py)
returns logits only.  Fusion, phrase and interval stages need ``model(x,
return_embeddings=True) -> (logits, pooled)``.  This adapter builds the benchmark's own
modules (same parameter names, so a family state dict loads strictly), and splits the
benchmark forward at its attention-pooled 256-d vector, the input of its classifier MLP.
That is the analogue of SLTStage1V17's pooled embedding.  Logits must stay identical to
``TransformerBaseline.forward``; test/test_v17_family_transformer_adapter.py checks this.

Checkpoints carry ``model_config["architecture"]``.  ``Stage1V17Config`` rejects that key,
so loaders that still construct SLTStage1V17 directly fail instead of misreading the weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model_v17 import SLTStage1V17, Stage1V17Config

FLAT_TRANSFORMER = "flat_transformer_family_benchmark_v17"


@dataclass(frozen=True)
class FlatTransformerStage1Config:
    num_classes: int = 100
    architecture: str = FLAT_TRANSFORMER
    depth: int = 8
    local_conv: bool = False

    def validate(self) -> None:
        if self.architecture != FLAT_TRANSFORMER:
            raise ValueError(f"not a flat Transformer config: {self.architecture}")
        if self.num_classes < 2 or self.depth < 1:
            raise ValueError("invalid flat Transformer dimensions")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FlatTransformerStage1V17(nn.Module):
    """``TransformerBaseline`` with the Stage-1 ``return_embeddings`` interface."""

    def __init__(self, config: FlatTransformerStage1Config):
        super().__init__()
        config.validate()
        # Imported here: the benchmark imports train_stage_1_v17, which imports this module.
        from scripts.benchmark_stage1_families_v17 import TransformerBaseline

        reference = TransformerBaseline(
            config.num_classes, depth=config.depth, local_conv=config.local_conv
        )
        self.frame = reference.frame
        self.local_conv = reference.local_conv
        self.position = reference.position
        self.encoder = reference.encoder
        self.head = reference.head
        self.config = config

    def forward(
        self, features: torch.Tensor, return_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        value = self.encoder(self.local_conv(self.frame(features)) + self.position)
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        # AttentionClassifier.forward, split before its classifier MLP.
        safe = active.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        scores = self.head.attention(value).squeeze(-1).masked_fill(
            ~safe, torch.finfo(value.dtype).min
        )
        pooled = (value * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        pooled = pooled * (~empty).unsqueeze(-1).to(value.dtype)
        logits = self.head.classifier(pooled)
        return (logits, pooled) if return_embeddings else logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def stage1_architecture(model_config: dict[str, object]) -> str:
    return str(model_config.get("architecture", "slt_stage1_v17"))


def build_stage1_landmark_model(model_config: dict[str, object]) -> nn.Module:
    """SLTStage1V17 for every existing config; the flat Transformer only when named."""
    if stage1_architecture(model_config) == FLAT_TRANSFORMER:
        return FlatTransformerStage1V17(FlatTransformerStage1Config(**model_config))
    return SLTStage1V17(Stage1V17Config(**model_config))
