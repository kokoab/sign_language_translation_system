"""Frozen unified temporal encoder and compact CTC head for v17 Stage 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17
from .model_unified_multimodal_v17 import (
    UnifiedFusionHeadV17,
    UnifiedMultimodalV17Config,
)
from .model_v17 import SLTStage1V17, Stage1V17Config


FROZEN_TEMPORAL_FEATURE_DIM = 256 + 256 + 100


def load_frozen_unified_stage1(path: str | Path) -> tuple[
    SLTStage1V17, HandMobileCLIP2Stage1V17, UnifiedFusionHeadV17, dict[str, object]
]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_unified_multimodal_v17":
        raise ValueError(f"{path}: not the selected unified v17 checkpoint")
    landmark_config = dict(checkpoint["landmark_model_config"])
    if "phonology_head_sizes" in landmark_config:
        landmark_config["phonology_head_sizes"] = tuple(
            tuple(value) for value in landmark_config["phonology_head_sizes"]
        )
    landmark = SLTStage1V17(Stage1V17Config(**landmark_config))
    landmark.load_state_dict(checkpoint["landmark_model_state_dict"], strict=True)
    hand = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["hand_model_config"])
    )
    hand.load_state_dict(checkpoint["hand_model_state_dict"], strict=True)
    fusion = UnifiedFusionHeadV17(
        UnifiedMultimodalV17Config(**checkpoint["head_config"])
    )
    fusion.load_state_dict(checkpoint["head_state_dict"], strict=True)
    for model in (landmark, hand, fusion):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return landmark, hand, fusion, checkpoint


class FrozenUnifiedTemporalEncoderV17(nn.Module):
    """Apply the selected Stage-1 graph at frame resolution without pooling."""

    def __init__(
        self,
        landmark: SLTStage1V17,
        hand: HandMobileCLIP2Stage1V17,
        fusion: UnifiedFusionHeadV17,
    ):
        super().__init__()
        self.landmark = landmark
        self.hand = hand
        self.fusion = fusion
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(
        self,
        landmarks: torch.Tensor,
        hand_embeddings: torch.Tensor,
        hand_valid: torch.Tensor,
        hand_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if landmarks.ndim != 5 or tuple(landmarks.shape[2:]) != (32, 61, 5):
            raise ValueError(f"unexpected landmarks {tuple(landmarks.shape)}")
        batch, windows = landmarks.shape[:2]
        expected_hand = (batch, windows, 16, 3, 512)
        if tuple(hand_embeddings.shape) != expected_hand:
            raise ValueError(f"unexpected hand embeddings {tuple(hand_embeddings.shape)}")
        flat_landmarks = landmarks.reshape(batch * windows, 32, 61, 5)
        flat_hand = hand_embeddings.reshape(batch * windows, 16, 3, 512)
        flat_valid = hand_valid.reshape(batch * windows, 16, 3)
        flat_boxes = hand_boxes.reshape(batch * windows, 16, 3, 4)
        landmark_tokens, _ = self.landmark.encode(flat_landmarks)
        hand_tokens, hand_frame_valid = self.hand.encode_frames(
            flat_hand, flat_valid, flat_boxes
        )
        hand_tokens = F.interpolate(
            hand_tokens.transpose(1, 2), size=32, mode="linear", align_corners=False
        ).transpose(1, 2)
        hand_mask = F.interpolate(
            hand_frame_valid.to(hand_tokens.dtype).unsqueeze(1), size=32, mode="nearest"
        ).squeeze(1) > 0.5
        hand_tokens = hand_tokens * hand_mask.unsqueeze(-1)
        landmark_logits = self.landmark.classifier(landmark_tokens)
        hand_logits = self.hand.classifier(hand_tokens)
        fused_scores = self.fusion(
            landmark_tokens, hand_tokens, landmark_logits, hand_logits
        )
        features = torch.cat((landmark_tokens, hand_tokens, fused_scores), dim=-1)
        return features.reshape(batch, windows, 32, FROZEN_TEMPORAL_FEATURE_DIM)


@dataclass(frozen=True)
class Stage2V17Config:
    num_classes: int = 100
    blank_index: int = 0
    input_dim: int = FROZEN_TEMPORAL_FEATURE_DIM
    dim: int = 256
    tokens_per_window: int = 8
    max_windows: int = 8
    depth: int = 4
    heads: int = 8
    dropout: float = 0.15

    def validate(self) -> None:
        if self.num_classes != 100 or self.blank_index != 0:
            raise ValueError("the locked Stage-2 vocabulary is 100 signs plus blank=0")
        if self.input_dim != FROZEN_TEMPORAL_FEATURE_DIM:
            raise ValueError("frozen temporal feature dimension changed")
        if self.tokens_per_window != 8 or self.max_windows < 1:
            raise ValueError("invalid temporal compression contract")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Stage2TemporalHeadV17(nn.Module):
    """Trainable sequence model over cached frozen Stage-1 temporal evidence."""

    def __init__(self, config: Stage2V17Config | None = None):
        super().__init__()
        self.config = config or Stage2V17Config()
        self.config.validate()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.temporal_compression = nn.Sequential(
            nn.Conv1d(
                self.config.dim, self.config.dim, kernel_size=5, stride=4,
                padding=2, groups=self.config.dim,
            ),
            nn.GELU(),
            nn.Conv1d(self.config.dim, self.config.dim, kernel_size=1),
        )
        maximum_tokens = self.config.max_windows * self.config.tokens_per_window
        self.position = nn.Parameter(torch.zeros(1, maximum_tokens, self.config.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, num_layers=self.config.depth)
        self.output_norm = nn.LayerNorm(self.config.dim)
        self.ctc_head = nn.Linear(self.config.dim, self.config.num_classes + 1)

    def encode(
        self, frozen_features: torch.Tensor, window_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frozen_features.ndim != 4 or tuple(frozen_features.shape[2:]) != (
            32, self.config.input_dim
        ):
            raise ValueError(f"unexpected frozen features {tuple(frozen_features.shape)}")
        batch, windows = frozen_features.shape[:2]
        if windows > self.config.max_windows or tuple(window_mask.shape) != (batch, windows):
            raise ValueError("invalid window mask or too many windows")
        value = self.input_projection(frozen_features)
        value = value.reshape(batch * windows, 32, self.config.dim)
        value = self.temporal_compression(value.transpose(1, 2)).transpose(1, 2)
        if value.shape[1] != self.config.tokens_per_window:
            raise RuntimeError("temporal compression output changed")
        value = value.reshape(batch, windows * self.config.tokens_per_window, self.config.dim)
        token_mask = window_mask.unsqueeze(-1).expand(
            batch, windows, self.config.tokens_per_window
        ).reshape(batch, -1)
        value = (value + self.position[:, : value.shape[1]]) * token_mask.unsqueeze(-1)
        value = self.sequence(value, src_key_padding_mask=~token_mask)
        value = self.output_norm(value) * token_mask.unsqueeze(-1)
        lengths = window_mask.sum(dim=1).to(torch.long) * self.config.tokens_per_window
        return value, lengths

    def forward(
        self, frozen_features: torch.Tensor, window_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, lengths = self.encode(frozen_features, window_mask)
        return self.ctc_head(value), lengths

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class Stage2ContextAdapterV17(nn.Module):
    """Apply a compact train-only context adapter as a residual CTC prior.

    The adapter summarizes each 32-frame window independently, so it remains
    compatible with arbitrary-length continuous inputs.  Only explicitly
    selected locked-vocabulary classes are changed; blank and every other CTC
    logit remain byte-for-byte identical to the base head.
    """

    def __init__(
        self,
        base: Stage2TemporalHeadV17,
        *,
        feature_mode: str,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
        coefficients: torch.Tensor,
        intercept: torch.Tensor,
        class_indices: torch.Tensor,
        target_class_indices: tuple[int, ...],
        weight: float,
    ):
        super().__init__()
        if feature_mode not in {
            "mean", "mean_std", "mean_std_max", "mean_std_max_delta"
        }:
            raise ValueError(f"unknown context feature mode: {feature_mode}")
        multiplier = {
            "mean": 1,
            "mean_std": 2,
            "mean_std_max": 3,
            "mean_std_max_delta": 4,
        }[feature_mode]
        summary_dim = base.config.input_dim * multiplier
        scaler_mean = torch.as_tensor(scaler_mean, dtype=torch.float32)
        scaler_scale = torch.as_tensor(scaler_scale, dtype=torch.float32)
        coefficients = torch.as_tensor(coefficients, dtype=torch.float32)
        intercept = torch.as_tensor(intercept, dtype=torch.float32)
        class_indices = torch.as_tensor(class_indices, dtype=torch.long)
        if tuple(scaler_mean.shape) != (summary_dim,) or tuple(scaler_scale.shape) != (
            summary_dim,
        ):
            raise ValueError("context scaler dimension mismatch")
        if tuple(coefficients.shape) != (len(class_indices), summary_dim):
            raise ValueError("context coefficient dimension mismatch")
        if tuple(intercept.shape) != (len(class_indices),):
            raise ValueError("context intercept dimension mismatch")
        if not target_class_indices or not 0.0 < weight:
            raise ValueError("context targets and a positive residual weight are required")
        class_to_row = {int(value): row for row, value in enumerate(class_indices.tolist())}
        if any(value not in class_to_row for value in target_class_indices):
            raise ValueError("a target class is absent from the fitted adapter")
        if any(not 0 <= value < base.config.num_classes for value in target_class_indices):
            raise ValueError("target class is outside the locked vocabulary")

        self.base = base
        self.feature_mode = feature_mode
        self.weight = float(weight)
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("intercept", intercept)
        self.register_buffer("class_indices", class_indices)
        selected_rows = torch.as_tensor(
            [class_to_row[value] for value in target_class_indices], dtype=torch.long
        )
        self.register_buffer("selected_rows", selected_rows)
        projection = torch.zeros(len(target_class_indices), base.config.num_classes + 1)
        for row, class_index in enumerate(target_class_indices):
            projection[row, class_index + 1] = 1.0
        self.register_buffer("target_projection", projection)

    def summarize(self, frozen_features: torch.Tensor) -> torch.Tensor:
        parts = [frozen_features.mean(dim=2)]
        if self.feature_mode in {"mean_std", "mean_std_max", "mean_std_max_delta"}:
            parts.append(frozen_features.std(dim=2, unbiased=False))
        if self.feature_mode in {"mean_std_max", "mean_std_max_delta"}:
            parts.append(frozen_features.max(dim=2).values)
        if self.feature_mode == "mean_std_max_delta":
            edge = min(8, frozen_features.shape[2])
            parts.append(
                frozen_features[:, :, -edge:].mean(dim=2)
                - frozen_features[:, :, :edge].mean(dim=2)
            )
        return torch.cat(parts, dim=-1)

    def forward(
        self, frozen_features: torch.Tensor, window_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, lengths = self.base(frozen_features, window_mask)
        summary = self.summarize(frozen_features)
        standardized = (summary - self.scaler_mean) / self.scaler_scale.clamp_min(1e-6)
        scores = standardized @ self.coefficients.transpose(0, 1) + self.intercept
        scores = (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        selected = scores.index_select(-1, self.selected_rows)
        residual = selected @ self.target_projection
        residual = residual.repeat_interleave(self.base.config.tokens_per_window, dim=1)
        token_mask = window_mask.repeat_interleave(
            self.base.config.tokens_per_window, dim=1
        ).unsqueeze(-1)
        return logits + self.weight * residual * token_mask, lengths

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count


def load_stage2_context_adapted(
    path: str | Path,
) -> tuple[Stage2ContextAdapterV17, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage2_context_adapted_ctc_v17":
        raise ValueError(f"{path}: not a context-adapted v17 Stage-2 checkpoint")
    state = checkpoint["model_state_dict"]
    config = checkpoint["context_adapter_config"]
    base = Stage2TemporalHeadV17(Stage2V17Config(**checkpoint["model_config"]))
    model = Stage2ContextAdapterV17(
        base,
        feature_mode=str(config["feature_mode"]),
        scaler_mean=state["scaler_mean"],
        scaler_scale=state["scaler_scale"],
        coefficients=state["coefficients"],
        intercept=state["intercept"],
        class_indices=state["class_indices"],
        target_class_indices=tuple(int(value) for value in config["target_class_indices"]),
        weight=float(config["weight"]),
    )
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def ctc_exact_match_mask(
    logits: torch.Tensor, lengths: torch.Tensor, target_tokens: tuple[int, ...]
) -> torch.Tensor:
    """Return rows whose greedy CTC collapse is exactly ``target_tokens``.

    ``target_tokens`` uses CTC-space indices, including the +1 offset from the
    locked class indices. The implementation remains tensor-only so the gate can
    run on the same device as the temporal heads.
    """
    if logits.ndim != 3 or lengths.ndim != 1 or len(lengths) != len(logits):
        raise ValueError("invalid CTC gate tensors")
    if not target_tokens or any(token <= 0 for token in target_tokens):
        raise ValueError("CTC gate targets must be non-blank tokens")
    predictions = logits.argmax(dim=-1)
    positions = torch.arange(predictions.shape[1], device=predictions.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    previous = torch.cat((
        torch.full_like(predictions[:, :1], -1), predictions[:, :-1]
    ), dim=1)
    emitted = valid & predictions.ne(0) & predictions.ne(previous)
    order = emitted.cumsum(dim=1)
    matches = emitted.sum(dim=1).eq(len(target_tokens))
    for index, token in enumerate(target_tokens, start=1):
        selected = torch.where(
            emitted & order.eq(index), predictions, torch.zeros_like(predictions)
        ).sum(dim=1)
        matches = matches & selected.eq(token)
    return matches


def greedy_ctc_tokens(logits: torch.Tensor, length: int) -> tuple[int, ...]:
    """Collapse one CTC logit stream while retaining CTC-space token indices."""
    prediction = logits[:length].argmax(dim=-1).detach().cpu().tolist()
    output: list[int] = []
    previous = -1
    for value in prediction:
        token = int(value)
        if token != 0 and token != previous:
            output.append(token)
        previous = token
    return tuple(output)


def ctc_sequence_log_probability(
    logits: torch.Tensor, ctc_tokens: tuple[int, ...]
) -> torch.Tensor:
    """Exact log probability of one collapsed CTC sequence.

    This small forward dynamic program avoids a training-loss dependency and works
    on the same device as the logits.  Tokens are in CTC space, so zero is reserved
    for blank.
    """
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("expected a non-empty [time, classes] CTC stream")
    if any(token <= 0 or token >= logits.shape[-1] for token in ctc_tokens):
        raise ValueError("invalid non-blank CTC token")
    log_probabilities = logits.log_softmax(dim=-1)
    extended: list[int] = [0]
    for token in ctc_tokens:
        extended.extend((int(token), 0))
    states = torch.full(
        (len(extended),), -torch.inf,
        dtype=log_probabilities.dtype, device=log_probabilities.device,
    )
    states[0] = log_probabilities[0, 0]
    if ctc_tokens:
        states[1] = log_probabilities[0, extended[1]]
    for frame in range(1, len(log_probabilities)):
        following: list[torch.Tensor] = []
        for state, token in enumerate(extended):
            paths = [states[state]]
            if state >= 1:
                paths.append(states[state - 1])
            if state >= 2 and token != 0 and token != extended[state - 2]:
                paths.append(states[state - 2])
            following.append(
                torch.logsumexp(torch.stack(paths), dim=0)
                + log_probabilities[frame, token]
            )
        states = torch.stack(following)
    if not ctc_tokens:
        return states[0]
    return torch.logsumexp(states[-2:], dim=0)


class Stage2GeneralCTCSelectorV17(nn.Module):
    """Phrase-agnostic arbitration between the multi-voice and transition heads.

    The primary head remains in control by default.  A specialist may own a row
    only when its different greedy hypothesis has the same length (at least two
    signs) and has no lower full-path CTC probability under the specialist itself.
    No gloss identity, phrase allowlist, or signer identity is used.

    The data-dependent dynamic program is intended for the accuracy research model.
    Distill it before Core ML export rather than presenting this wrapper as the
    compact mobile graph.
    """

    def __init__(
        self,
        primary: Stage2ContextAdapterV17,
        specialist: Stage2TemporalHeadV17,
        *,
        blend_weight: float,
        blank_bias: float,
        score_margin: float = 0.0,
        minimum_tokens: int = 2,
    ):
        super().__init__()
        if primary.base.config.to_dict() != specialist.config.to_dict():
            raise ValueError("primary/specialist Stage-2 configurations differ")
        if not 0.0 < blend_weight < 0.5:
            raise ValueError("general selector blend weight must be in (0, 0.5)")
        if minimum_tokens < 2:
            raise ValueError("general selector requires at least two tokens")
        self.primary = primary
        self.specialist = specialist
        self.blend_weight = float(blend_weight)
        self.blank_bias = float(blank_bias)
        self.score_margin = float(score_margin)
        self.minimum_tokens = int(minimum_tokens)

    @property
    def config(self) -> Stage2V17Config:
        return self.primary.base.config

    def forward(
        self, frozen_features: torch.Tensor, window_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, lengths, _ = self.forward_with_selection(
            frozen_features, window_mask
        )
        return logits, lengths

    def forward_with_selection(
        self, frozen_features: torch.Tensor, window_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return inference logits plus the phrase-agnostic specialist row mask."""
        primary_logits, lengths = self.primary(frozen_features, window_mask)
        specialist_logits, specialist_lengths = self.specialist(
            frozen_features, window_mask
        )
        if not torch.equal(lengths, specialist_lengths):
            raise RuntimeError("primary/specialist CTC lengths differ")
        calibrated = (
            primary_logits * (1.0 - self.blend_weight)
            + specialist_logits * self.blend_weight
        )
        calibrated = calibrated.clone()
        calibrated[..., 0] += self.blank_bias
        selected = torch.zeros(len(calibrated), dtype=torch.bool, device=calibrated.device)
        for row, raw_length in enumerate(lengths.detach().cpu().tolist()):
            length = int(raw_length)
            base_tokens = greedy_ctc_tokens(calibrated[row], length)
            specialist_tokens = greedy_ctc_tokens(specialist_logits[row], length)
            if (
                base_tokens == specialist_tokens
                or len(base_tokens) < self.minimum_tokens
                or len(base_tokens) != len(specialist_tokens)
            ):
                continue
            base_score = ctc_sequence_log_probability(
                specialist_logits[row, :length], base_tokens
            )
            specialist_score = ctc_sequence_log_probability(
                specialist_logits[row, :length], specialist_tokens
            )
            selected[row] = bool(
                (specialist_score - base_score >= self.score_margin).detach().cpu()
            )
        return torch.where(
            selected[:, None, None], specialist_logits, calibrated
        ), lengths, selected

    @property
    def parameter_count(self) -> int:
        return self.primary.parameter_count + self.specialist.parameter_count


class Stage2DirectJoinSpecialistV17(nn.Module):
    """Blend a direct-isolated-join specialist with a stronger primary CTC model.

    The specialist receives a small global logit weight. When its greedy CTC
    sequence exactly matches a predeclared high-precision sequence, it owns that
    row. This retains complementary transition knowledge without allowing the
    weaker direct-join model to overwrite the primary model broadly.
    """

    def __init__(
        self,
        primary: Stage2ContextAdapterV17,
        specialist: Stage2TemporalHeadV17,
        *,
        blend_weight: float,
        gate_ctc_tokens: tuple[int, ...],
    ):
        super().__init__()
        if primary.base.config.to_dict() != specialist.config.to_dict():
            raise ValueError("primary/specialist Stage-2 configurations differ")
        if not 0.0 < blend_weight < 0.5:
            raise ValueError("direct-join blend weight must be in (0, 0.5)")
        if len(gate_ctc_tokens) < 2:
            raise ValueError("direct-join specialist gate must contain multiple signs")
        self.primary = primary
        self.specialist = specialist
        self.blend_weight = float(blend_weight)
        self.gate_ctc_tokens = tuple(int(token) for token in gate_ctc_tokens)

    @property
    def config(self) -> Stage2V17Config:
        return self.primary.base.config

    def forward(
        self, frozen_features: torch.Tensor, window_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary_logits, lengths = self.primary(frozen_features, window_mask)
        specialist_logits, specialist_lengths = self.specialist(frozen_features, window_mask)
        if not torch.equal(lengths, specialist_lengths):
            raise RuntimeError("primary/specialist CTC lengths differ")
        blended = (
            primary_logits * (1.0 - self.blend_weight)
            + specialist_logits * self.blend_weight
        )
        gate = ctc_exact_match_mask(
            specialist_logits, specialist_lengths, self.gate_ctc_tokens
        )
        logits = torch.where(gate[:, None, None], specialist_logits, blended)
        return logits, lengths

    @property
    def parameter_count(self) -> int:
        return self.primary.parameter_count + self.specialist.parameter_count


def load_stage2_direct_join_specialist(
    path: str | Path,
) -> tuple[Stage2DirectJoinSpecialistV17, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage2_direct_join_specialist_ctc_v17":
        raise ValueError(f"{path}: not a direct-join specialist v17 checkpoint")
    state = checkpoint["model_state_dict"]
    adapter = checkpoint["primary_context_adapter_config"]
    config = Stage2V17Config(**checkpoint["model_config"])
    primary_base = Stage2TemporalHeadV17(config)
    primary = Stage2ContextAdapterV17(
        primary_base,
        feature_mode=str(adapter["feature_mode"]),
        scaler_mean=state["primary.scaler_mean"],
        scaler_scale=state["primary.scaler_scale"],
        coefficients=state["primary.coefficients"],
        intercept=state["primary.intercept"],
        class_indices=state["primary.class_indices"],
        target_class_indices=tuple(int(value) for value in adapter["target_class_indices"]),
        weight=float(adapter["weight"]),
    )
    model = Stage2DirectJoinSpecialistV17(
        primary,
        Stage2TemporalHeadV17(config),
        blend_weight=float(checkpoint["specialist_config"]["blend_weight"]),
        gate_ctc_tokens=tuple(
            int(value) for value in checkpoint["specialist_config"]["gate_ctc_tokens"]
        ),
    )
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def load_stage2_general_ctc_selector(
    path: str | Path,
) -> tuple[Stage2GeneralCTCSelectorV17, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage2_general_ctc_selector_v17":
        raise ValueError(f"{path}: not a general CTC selector v17 checkpoint")
    state = checkpoint["model_state_dict"]
    adapter = checkpoint["primary_context_adapter_config"]
    config = Stage2V17Config(**checkpoint["model_config"])
    primary = Stage2ContextAdapterV17(
        Stage2TemporalHeadV17(config),
        feature_mode=str(adapter["feature_mode"]),
        scaler_mean=state["primary.scaler_mean"],
        scaler_scale=state["primary.scaler_scale"],
        coefficients=state["primary.coefficients"],
        intercept=state["primary.intercept"],
        class_indices=state["primary.class_indices"],
        target_class_indices=tuple(int(value) for value in adapter["target_class_indices"]),
        weight=float(adapter["weight"]),
    )
    selector = checkpoint["selector_config"]
    model = Stage2GeneralCTCSelectorV17(
        primary,
        Stage2TemporalHeadV17(config),
        blend_weight=float(selector["blend_weight"]),
        blank_bias=float(selector["blank_bias"]),
        score_margin=float(selector["score_margin"]),
        minimum_tokens=int(selector["minimum_tokens"]),
    )
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def load_stage2_model_v17(
    path: str | Path,
) -> tuple[nn.Module, dict[str, object]]:
    """Load any selected v17 Stage-2 inference artifact."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    artifact_format = checkpoint.get("format")
    if artifact_format == "slt_stage2_context_adapted_ctc_v17":
        return load_stage2_context_adapted(path)
    if artifact_format == "slt_stage2_direct_join_specialist_ctc_v17":
        return load_stage2_direct_join_specialist(path)
    if artifact_format == "slt_stage2_general_ctc_selector_v17":
        return load_stage2_general_ctc_selector(path)
    if artifact_format == "slt_stage2_ctc_v17":
        model = Stage2TemporalHeadV17(Stage2V17Config(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, checkpoint
    raise ValueError(f"{path}: unsupported v17 Stage-2 format {artifact_format!r}")


class Stage2DualHeadV17(nn.Module):
    """Locked deployment CTC plus a training-only expanded-gloss auxiliary head."""

    def __init__(self, config: Stage2V17Config, auxiliary_num_classes: int):
        super().__init__()
        if auxiliary_num_classes < config.num_classes:
            raise ValueError("auxiliary vocabulary cannot be smaller than the locked vocabulary")
        self.locked = Stage2TemporalHeadV17(config)
        self.auxiliary_num_classes = auxiliary_num_classes
        self.auxiliary_ctc_head = nn.Linear(config.dim, auxiliary_num_classes + 1)

    def forward_locked(self, frozen_features, window_mask):
        return self.locked(frozen_features, window_mask)

    def forward_auxiliary(self, frozen_features, window_mask):
        value, lengths = self.locked.encode(frozen_features, window_mask)
        return self.auxiliary_ctc_head(value), lengths

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def warm_start_dual_stage2(
    model: Stage2DualHeadV17, checkpoint: dict[str, object]
) -> None:
    """Load a selected 100-sign head, extending only its positional table."""
    if checkpoint.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("warm-start checkpoint is not v17 Stage 2")
    source = checkpoint["model_state_dict"]
    target = model.locked.state_dict()
    for name, value in source.items():
        if name == "position":
            if value.shape[0] != 1 or value.shape[2] != target[name].shape[2]:
                raise ValueError("warm-start positional dimension mismatch")
            if value.shape[1] > target[name].shape[1]:
                raise ValueError("warm-start positional table is longer than target")
            target[name][:, : value.shape[1]].copy_(value)
        else:
            if name not in target or target[name].shape != value.shape:
                raise ValueError(f"warm-start shape mismatch: {name}")
            target[name].copy_(value)
    model.locked.load_state_dict(target, strict=True)

    # Auxiliary indices 0--99 preserve the locked vocabulary after blank=0.
    with torch.no_grad():
        model.auxiliary_ctc_head.weight[0].copy_(model.locked.ctc_head.weight[0])
        model.auxiliary_ctc_head.bias[0].copy_(model.locked.ctc_head.bias[0])
        model.auxiliary_ctc_head.weight[1:101].copy_(model.locked.ctc_head.weight[1:101])
        model.auxiliary_ctc_head.bias[1:101].copy_(model.locked.ctc_head.bias[1:101])


class UnifiedMultimodalStage2V17(nn.Module):
    def __init__(self, encoder: FrozenUnifiedTemporalEncoderV17, head: Stage2TemporalHeadV17):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, landmarks, hand_embeddings, hand_valid, hand_boxes, window_mask):
        frozen = self.encoder(landmarks, hand_embeddings, hand_valid, hand_boxes)
        return self.head(frozen, window_mask)


def make_stage2_checkpoint(
    head: Stage2TemporalHeadV17,
    state_dict: dict[str, torch.Tensor],
    **metadata,
) -> dict[str, object]:
    return {
        "format": "slt_stage2_ctc_v17",
        "format_version": 1,
        "model_config": head.config.to_dict(),
        "model_state_dict": state_dict,
        "blank_index": 0,
        "test_evaluated": False,
        **metadata,
    }
