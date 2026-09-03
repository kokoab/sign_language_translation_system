"""A tiny causal CTC head over the existing v17 Stage-1 evidence stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StreamingStage1HeadConfig:
    num_glosses: int = 100
    hidden_dim: int = 64
    blocks: int = 3
    kernel_size: int = 3
    dropout: float = 0.10

    @property
    def receptive_field_steps(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(2**i for i in range(self.blocks))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Stage1EvidenceBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.context = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            width, width, kernel_size, dilation=dilation, groups=width
        )
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        value = self.pointwise(F.gelu(value).transpose(1, 2)).transpose(1, 2)
        return self.dropout(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        padded = F.pad(value.transpose(1, 2), (self.context, 0)).transpose(1, 2)
        return self.norm(value + self.transform(padded))

    def step(
        self, value: torch.Tensor, history: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history is None:
            history = value.new_zeros(value.shape[0], self.context, value.shape[-1])
        joined = torch.cat((history, value), dim=1)
        return self.norm(value + self.transform(joined)), joined[:, -self.context :]


class StreamingStage1CTCHeadV17(nn.Module):
    """Uses Stage-1 logits as a strong residual and learns blank/timing causally."""

    def __init__(self, config: StreamingStage1HeadConfig | None = None):
        super().__init__()
        self.config = config or StreamingStage1HeadConfig()
        self.input_norm = nn.LayerNorm(self.config.num_glosses)
        self.input_projection = nn.Linear(self.config.num_glosses, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            Stage1EvidenceBlock(
                self.config.hidden_dim, self.config.kernel_size, 2**index,
                self.config.dropout,
            )
            for index in range(self.config.blocks)
        )
        self.blank = nn.Linear(self.config.hidden_dim, 1)
        self.gloss_delta = nn.Linear(self.config.hidden_dim, self.config.num_glosses)
        nn.init.zeros_(self.gloss_delta.weight)
        nn.init.zeros_(self.gloss_delta.bias)

    def encode(self, stage1_logits: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.input_projection(self.input_norm(stage1_logits)))
        for block in self.blocks:
            value = block(value)
        return value

    def classify(self, value: torch.Tensor, stage1_logits: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.blank(value), stage1_logits + self.gloss_delta(value)), dim=-1)

    def forward(self, stage1_logits: torch.Tensor) -> torch.Tensor:
        if stage1_logits.ndim != 3 or stage1_logits.shape[-1] != self.config.num_glosses:
            raise ValueError("expected Stage-1 logits shaped [B,T,100]")
        return self.classify(self.encode(stage1_logits), stage1_logits)

    def stream_step(
        self,
        stage1_logits: torch.Tensor,
        state: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if stage1_logits.ndim == 2:
            stage1_logits = stage1_logits.unsqueeze(1)
        if stage1_logits.shape[1:] != (1, self.config.num_glosses):
            raise ValueError("expected one Stage-1 evidence step shaped [B,100]")
        value = F.gelu(self.input_projection(self.input_norm(stage1_logits)))
        histories = state or tuple(None for _ in self.blocks)
        next_state = []
        for block, history in zip(self.blocks, histories):
            value, history = block.step(value, history)
            next_state.append(history)
        logits = self.classify(value[:, 0], stage1_logits[:, 0])
        return logits, tuple(next_state)


def load_streaming_stage1_head_checkpoint(
    checkpoint: dict[str, object], *, device: str | torch.device = "cpu"
) -> StreamingStage1CTCHeadV17:
    if checkpoint.get("format") != "slt_streaming_stage1_ctc_head_v17":
        raise ValueError("not a v17 streaming Stage-1 CTC head")
    model = StreamingStage1CTCHeadV17(
        StreamingStage1HeadConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
