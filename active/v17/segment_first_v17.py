"""Minimal segment-first boundary, rejection, and locked-gloss model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from active.v17.model_reel_emission_v17 import pool_stage1_encoded
from active.v17.model_v17 import SLTStage1V17


OUTSIDE, START, SIGNING, END = range(4)
IGNORE = -100
STATES = ("OUTSIDE", "START", "SIGNING", "END")
CHECKPOINT_FORMAT = "slt_segment_first_v17"


def frame_targets(
    sample_times: np.ndarray,
    accepted: list[tuple[float, float]],
    excluded: list[tuple[float, float]],
    negative_eligible: bool,
) -> np.ndarray:
    """Create exact edge targets while masking every questionable annotation."""
    times = np.asarray(sample_times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be strictly increasing")
    result = np.full(len(times), OUTSIDE if negative_eligible else IGNORE, np.int64)
    for start, end in accepted:
        if end <= start:
            raise ValueError("event intervals must be positive")
        inside = (times >= start) & (times <= end)
        result[inside] = SIGNING
        if times[0] <= start <= times[-1]:
            result[int(np.argmin(np.abs(times - start)))] = START
        if times[0] <= end <= times[-1]:
            result[int(np.argmin(np.abs(times - end)))] = END
    for start, end in excluded:
        result[(times >= start) & (times <= end)] = IGNORE
    return result


@dataclass(frozen=True)
class SegmentCandidate:
    start_seconds: float
    end_seconds: float
    score: float

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds or not np.isfinite(self.score):
            raise ValueError("candidate requires a positive interval and finite score")


def merge_candidates(
    candidates: list[SegmentCandidate], tolerance_seconds: float = 0.10
) -> list[SegmentCandidate]:
    """Merge repeated estimates of one boundary while retaining separate repeats."""
    if tolerance_seconds <= 0:
        raise ValueError("tolerance must be positive")
    merged: list[SegmentCandidate] = []
    for candidate in sorted(candidates, key=lambda value: value.end_seconds):
        if merged and candidate.end_seconds - merged[-1].end_seconds <= tolerance_seconds:
            if candidate.score > merged[-1].score:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return merged


class SegmentFirstV17(nn.Module):
    def __init__(self, base: SLTStage1V17):
        super().__init__()
        if base.config.num_classes != 100 or base.config.static_hand_token != "none":
            raise ValueError("segment-first requires the locked 100-gloss Stage-1 model")
        self.base = base
        self.boundary = nn.Linear(base.config.dim, len(STATES))
        self.known = nn.Sequential(nn.LayerNorm(base.config.dim), nn.Linear(base.config.dim, 2))

    def boundary_logits(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.base.encode(features)
        return self.boundary(encoded)

    def segment_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, active = self.base.encode(features)
        pooled = pool_stage1_encoded(self.base, encoded, active)
        return self.base.classifier(pooled), self.known(pooled)

