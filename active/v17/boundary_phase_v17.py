"""Small phase head and repeat-safe online decoder for v17 Stage 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from active.v17.model_reel_emission_v17 import pool_stage1_encoded, reel_temporal_summary
from active.v17.model_v17 import SLTStage1V17


KNOWN, UNKNOWN, TRANSITION = 0, 1, 2
PHASES = ("KNOWN", "UNKNOWN", "TRANSITION")
WINDOW_SECONDS = (0.27, 0.53)
STRIDE_SECONDS = 0.067
WINDOW_FRAMES = 32
CHECKPOINT_FORMAT = "slt_boundary_phase_v17"


class BoundaryPhaseHeadV17(nn.Module):
    def __init__(self, stage1_dim: int = 256, num_glosses: int = 100):
        super().__init__()
        self.stage1_dim = stage1_dim
        self.num_glosses = num_glosses
        self.network = nn.Sequential(
            nn.LayerNorm(stage1_dim * 5 + num_glosses),
            nn.Linear(stage1_dim * 5 + num_glosses, 128),
            nn.GELU(),
            nn.Linear(128, len(PHASES)),
        )

    def forward(self, encoded: torch.Tensor, gloss_logits: torch.Tensor) -> torch.Tensor:
        if gloss_logits.shape[-1] != self.num_glosses:
            raise ValueError("phase head requires exactly 100 gloss logits")
        return self.network(reel_temporal_summary(encoded, gloss_logits))


class BoundaryPhaseModelV17(nn.Module):
    def __init__(self, base: SLTStage1V17, phase_head: BoundaryPhaseHeadV17):
        super().__init__()
        if base.config.num_classes != phase_head.num_glosses:
            raise ValueError("base and phase-head gloss counts disagree")
        self.base = base
        self.phase_head = phase_head

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, active = self.base.encode(features)
        gloss = self.base.classifier(pool_stage1_encoded(self.base, encoded, active))
        return gloss, self.phase_head(encoded, gloss)


@dataclass
class BoundaryPhaseTranscript:
    agreements: int = 2

    def __post_init__(self) -> None:
        if self.agreements < 1:
            raise ValueError("agreements must be positive")
        self.words: list[str] = []
        self.provisional: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0
        self._last_emitted: str | None = None
        self._repeat_armed = False
        self._last_time = -np.inf

    def update(self, phase: int, label: str | None, end_seconds: float) -> list[str]:
        if phase not in (KNOWN, UNKNOWN, TRANSITION) or not np.isfinite(end_seconds):
            raise ValueError("valid phase and finite timestamp required")
        if end_seconds <= self._last_time:
            raise ValueError("timestamps must increase")
        self._last_time = float(end_seconds)
        if phase == TRANSITION:
            self._candidate = self.provisional = None
            self._candidate_count = 0
            self._repeat_armed = True
            return list(self.words)
        if phase == UNKNOWN:
            self._candidate = self.provisional = None
            self._candidate_count = 0
            return list(self.words)
        if not label:
            raise ValueError("KNOWN requires a locked-vocabulary label")
        self.provisional = label
        if label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = label, 1
        if self._candidate_count >= self.agreements and (
            label != self._last_emitted or self._repeat_armed
        ):
            self.words.append(label)
            self._last_emitted = label
            self._repeat_armed = False
        return list(self.words)

    def reset(self) -> None:
        self.__post_init__()
