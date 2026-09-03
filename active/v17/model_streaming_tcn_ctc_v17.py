"""Small causal landmark TCN for frame-synchronous v17 CTC recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


GROUPS = ((0, 21), (21, 42), (42, 57), (57, 61))


@dataclass(frozen=True)
class StreamingTCNConfig:
    num_glosses: int = 100
    input_channels: int = 5
    group_dim: int = 24
    hidden_dim: int = 96
    blocks: int = 4
    kernel_size: int = 3
    output_stride: int = 4
    dropout: float = 0.10
    use_face_body: bool = True

    @property
    def receptive_field_frames(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(2**i for i in range(self.blocks))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CausalDepthwiseBlock(nn.Module):
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
        return self.dropout(self.pointwise(F.gelu(value).transpose(1, 2)).transpose(1, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        padded = F.pad(value.transpose(1, 2), (self.context, 0)).transpose(1, 2)
        return self.norm(value + self.transform(padded))

    def step(
        self, value: torch.Tensor, history: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history is None:
            history = value.new_zeros(value.shape[0], self.context, value.shape[-1])
        joined = torch.cat((history, value), dim=1)
        output = self.norm(value + self.transform(joined))
        return output, joined[:, -self.context :]


class StreamingLandmarkTCNCTCV17(nn.Module):
    """Consumes ``[B,T,61,5]`` and emits blank-plus-100-gloss logits per frame."""

    def __init__(self, config: StreamingTCNConfig | None = None):
        super().__init__()
        self.config = config or StreamingTCNConfig()
        self.group_norms = nn.ModuleList(
            nn.LayerNorm((stop - start) * self.config.input_channels)
            for start, stop in GROUPS
        )
        self.group_projections = nn.ModuleList(
            nn.Linear((stop - start) * self.config.input_channels, self.config.group_dim)
            for start, stop in GROUPS
        )
        self.fusion = nn.Linear(self.config.group_dim * len(GROUPS), self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                2**index,
                self.config.dropout,
            )
            for index in range(self.config.blocks)
        )
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_glosses)

    def classify(self, value: torch.Tensor) -> torch.Tensor:
        glosses = self.classifier(value)
        return torch.cat((torch.zeros_like(glosses[..., :1]), glosses), dim=-1)

    def encode_groups(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-2:] != (61, 5):
            raise ValueError("expected v17 features shaped [B,T,61,5]")
        groups = []
        for index, ((start, stop), norm, projection) in enumerate(
            zip(GROUPS, self.group_norms, self.group_projections)
        ):
            value = features[:, :, start:stop].flatten(2)
            if not self.config.use_face_body and index >= 2:
                value = torch.zeros_like(value)
            groups.append(F.gelu(projection(norm(value))))
        return F.gelu(self.fusion(torch.cat(groups, dim=-1)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.encode_groups(features)
        for block in self.blocks:
            value = block(value)
        return self.classify(value)

    def stream_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Emit the newest frame exactly, retaining only each block's causal context."""
        if frame.ndim == 3:
            frame = frame.unsqueeze(1)
        if frame.shape[1:] != (1, 61, 5):
            raise ValueError("expected one v17 frame shaped [B,61,5]")
        histories = state or tuple(None for _ in self.blocks)
        if len(histories) != len(self.blocks):
            raise ValueError("streaming state has the wrong number of blocks")
        value = self.encode_groups(frame)
        next_state = []
        for block, history in zip(self.blocks, histories):
            value, history = block.step(value, history)
            next_state.append(history)
        return self.classify(value[:, 0]), tuple(next_state)


def load_streaming_tcn_checkpoint(
    checkpoint: dict[str, object], *, device: str | torch.device = "cpu"
) -> StreamingLandmarkTCNCTCV17:
    if checkpoint.get("format") != "slt_streaming_tcn_ctc_v17":
        raise ValueError("not a v17 streaming TCN checkpoint")
    model = StreamingLandmarkTCNCTCV17(
        StreamingTCNConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
