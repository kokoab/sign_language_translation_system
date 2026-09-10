"""Trainable OTHER adaptation over the accepted Stage-2 CTC selector."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .model_stage2_v17 import (
    Stage2GeneralCTCSelectorV17,
    Stage2OtherPreservingCTCV17,
    Stage2TemporalHeadV17,
    Stage2V17Config,
    _general_selector_from_checkpoint,
    make_stage2_checkpoint,
    preserve_known_ctc_logits,
)


class Stage2LiveAdaptedCTCV17(nn.Module):
    """Fine-tune accepted CTC evidence while retaining frozen OTHER features."""

    def __init__(self, preservation: Stage2OtherPreservingCTCV17):
        super().__init__()
        if not isinstance(preservation.accepted, Stage2GeneralCTCSelectorV17):
            raise ValueError("live adaptation requires the accepted general selector")
        if not hasattr(preservation.accepted, "primary") or not hasattr(preservation.accepted, "specialist"):
            raise ValueError("accepted selector must include context and specialist heads")
        if preservation.accepted.config.num_classes != 100 or preservation.evidence.config.num_classes != 101:
            raise ValueError("live adaptation requires 100 known classes and 101-class evidence")
        self.accepted = preservation.accepted
        self.evidence = preservation.evidence
        self.other_head = preservation.other_head
        self._preservation_margin = float(preservation.margin)
        self.other_shift = nn.Parameter(torch.tensor(-20.0))
        self.config = self.evidence.config
        self.accepted.requires_grad_(True)
        self.other_head.requires_grad_(True)
        self.evidence.requires_grad_(False)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.accepted.eval()
        self.evidence.eval()
        return self

    def other_log_odds(self, frozen_features, window_mask):
        with torch.no_grad():
            hidden, lengths = self.evidence.encode(frozen_features, window_mask)
            normalizer = self.evidence.ctc_head(hidden)[..., :101].logsumexp(
                -1, keepdim=True
            )
        return self.other_head(hidden) - normalizer + self.other_shift, lengths

    def forward(self, frozen_features, window_mask):
        known, lengths = self.accepted(frozen_features, window_mask)
        odds, evidence_lengths = self.other_log_odds(frozen_features, window_mask)
        if not torch.equal(lengths, evidence_lengths):
            raise ValueError("accepted/evidence temporal lengths differ")
        return preserve_known_ctc_logits(known, odds), lengths


def _accepted_checkpoint(accepted: Stage2GeneralCTCSelectorV17) -> dict[str, object]:
    adapter = accepted.primary
    target_class_indices = (
        adapter.target_projection.argmax(dim=1).detach().cpu().tolist()
    )
    return {
        "format": "slt_stage2_general_ctc_selector_v17",
        "model_config": accepted.config.to_dict(),
        "model_state_dict": accepted.state_dict(),
        "primary_context_adapter_config": {
            "feature_mode": adapter.feature_mode,
            "target_class_indices": [int(value) - 1 for value in target_class_indices],
            "weight": adapter.weight,
        },
        "selector_config": {
            "blend_weight": accepted.blend_weight,
            "blank_bias": accepted.blank_bias,
            "score_margin": accepted.score_margin,
            "minimum_tokens": accepted.minimum_tokens,
        },
    }


def make_stage2_live_adapted_checkpoint(
    model: Stage2LiveAdaptedCTCV17, *, input_contract: dict[str, object]
) -> dict[str, object]:
    """Build a self-contained artifact without an external preservation file."""
    if not isinstance(model, Stage2LiveAdaptedCTCV17):
        raise ValueError("expected a Stage2LiveAdaptedCTCV17 model")
    if not isinstance(input_contract, dict):
        raise ValueError("input_contract must be a dictionary")
    return {
        "format": "slt_stage2_live_adapted_ctc_v17",
        "input_contract": dict(input_contract),
        "preservation_checkpoint": {
            "format": "slt_stage2_other_preserving_ctc_v17",
            "margin": float(model._preservation_margin),
            "accepted_checkpoint": _accepted_checkpoint(model.accepted),
            "evidence_checkpoint": make_stage2_checkpoint(
                model.evidence, model.evidence.state_dict()
            ),
            "other_head_state_dict": model.other_head.state_dict(),
        },
        "model_state_dict": model.state_dict(),
    }


def _preservation_from_payload(payload: dict[str, object]) -> Stage2OtherPreservingCTCV17:
    if payload.get("format") != "slt_stage2_other_preserving_ctc_v17":
        raise ValueError("invalid embedded preservation checkpoint")
    accepted, _ = _general_selector_from_checkpoint(payload["accepted_checkpoint"])
    evidence_payload = payload["evidence_checkpoint"]
    evidence = Stage2TemporalHeadV17(
        Stage2V17Config(**evidence_payload["model_config"])
    )
    evidence.load_state_dict(evidence_payload["model_state_dict"], strict=True)
    other_head = nn.Linear(evidence.config.dim, 1)
    other_head.load_state_dict(payload["other_head_state_dict"], strict=True)
    return Stage2OtherPreservingCTCV17(
        accepted, evidence, other_head, float(payload["margin"])
    )


def load_stage2_live_adapted(
    path: str | Path,
) -> tuple[Stage2LiveAdaptedCTCV17, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_live_adapted_ctc_v17":
        raise ValueError("not a live-adapted v17 Stage-2 checkpoint")
    if not isinstance(payload.get("input_contract"), dict):
        raise ValueError("live-adapted checkpoint is missing input_contract")
    model = Stage2LiveAdaptedCTCV17(
        _preservation_from_payload(payload["preservation_checkpoint"])
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
