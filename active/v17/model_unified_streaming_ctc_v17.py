"""Causal CTC decoder over rich v17 Stage-1 window evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from active.v17.model_streaming_stage1_head_v17 import Stage1EvidenceBlock


@dataclass(frozen=True)
class UnifiedStreamingCTCConfig:
    stage1_dim: int = 256
    num_glosses: int = 100
    hidden_dim: int = 128
    blocks: int = 3
    kernel_size: int = 3
    dropout: float = 0.10

    @property
    def input_dim(self) -> int:
        return self.stage1_dim + self.num_glosses

    @property
    def other_index(self) -> int:
        return self.num_glosses + 1

    @property
    def receptive_field_steps(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(
            2**index for index in range(self.blocks)
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class UnifiedStreamingCTCHeadV17(nn.Module):
    """Emit blank, one of 100 locked glosses, or explicit OTHER signing."""

    def __init__(self, config: UnifiedStreamingCTCConfig | None = None):
        super().__init__()
        self.config = config or UnifiedStreamingCTCConfig()
        self.input_norm = nn.LayerNorm(self.config.input_dim)
        self.input_projection = nn.Linear(
            self.config.input_dim, self.config.hidden_dim
        )
        self.blocks = nn.ModuleList(
            Stage1EvidenceBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                2**index,
                self.config.dropout,
            )
            for index in range(self.config.blocks)
        )
        self.blank = nn.Linear(self.config.hidden_dim, 1)
        self.gloss_delta = nn.Linear(
            self.config.hidden_dim, self.config.num_glosses
        )
        self.other = nn.Linear(self.config.hidden_dim, 1)
        nn.init.zeros_(self.gloss_delta.weight)
        nn.init.zeros_(self.gloss_delta.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3 or evidence.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected [B,T,{self.config.input_dim}], got {tuple(evidence.shape)}"
            )
        stage1_logits = evidence[..., self.config.stage1_dim :]
        value = F.gelu(self.input_projection(self.input_norm(evidence)))
        for block in self.blocks:
            value = block(value)
        return torch.cat(
            (
                self.blank(value),
                stage1_logits + self.gloss_delta(value),
                self.other(value),
            ),
            dim=-1,
        )


def load_unified_streaming_head(
    checkpoint: dict[str, object], *, device: str | torch.device = "cpu"
) -> UnifiedStreamingCTCHeadV17:
    if checkpoint.get("format") != "slt_unified_streaming_ctc_v17":
        raise ValueError("not a unified v17 streaming CTC checkpoint")
    model = UnifiedStreamingCTCHeadV17(
        UnifiedStreamingCTCConfig(**checkpoint["head_config"])
    )
    model.load_state_dict(checkpoint["head_state_dict"], strict=True)
    return model.to(device).eval()
