"""Compose a complete landmark utterance in a novel continuous signing style."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .geometry_v17 import resample_features
from .model_signing_voice_v17 import SigningVoiceGeneratorV17, SigningVoiceV17Config
from .model_transition_span_v17 import TransitionSpanPredictorV17, TransitionSpanV17Config
from .train_transition_diffusion_v17 import load_mean_model


@dataclass(frozen=True)
class NovelVoiceRecipe:
    name: str
    source_voice_indices: tuple[int, ...]
    weights: tuple[float, ...]


def load_signing_voice(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[SigningVoiceGeneratorV17, dict[str, object]]:
    row = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if row.get("format") != "slt_signing_voice_generator_v17":
        raise ValueError("unexpected signing-voice checkpoint")
    model = SigningVoiceGeneratorV17(SigningVoiceV17Config(**row["model_config"]))
    model.install_style_classifier(int(row["style_classifier_voices"]))
    missing, unexpected = model.load_state_dict(row["model_state_dict"], strict=False)
    allowed_missing = {"style_spatial_residual.weight"}
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"incompatible signing-voice state: missing={missing}, unexpected={unexpected}")
    model.eval().requires_grad_(False).to(device)
    return model, row


def load_transition_voice(
    mean_checkpoint: Path,
    timing_checkpoint: Path,
    device: torch.device | str = "cpu",
):
    device = torch.device(device)
    mean = load_mean_model(mean_checkpoint, device)
    row = torch.load(timing_checkpoint, map_location="cpu", weights_only=False)
    if row.get("format") != "slt_transition_span_predictor_v17":
        raise ValueError("unexpected transition timing checkpoint")
    timing = TransitionSpanPredictorV17(TransitionSpanV17Config(**row["model_config"]))
    timing.load_state_dict(row["model_state_dict"])
    timing.eval().requires_grad_(False).to(device)
    return mean, timing


def normalize_style_mix(
    centroids: torch.Tensor,
    source_voice_indices: Iterable[int],
    weights: Iterable[float],
) -> torch.Tensor:
    indices = tuple(int(value) for value in source_voice_indices)
    values = tuple(float(value) for value in weights)
    if len(indices) < 3 or len(indices) != len(values):
        raise ValueError("a novel voice requires at least three aligned source voices")
    if len(set(indices)) != len(indices) or min(values) <= 0 or max(values) > 0.60:
        raise ValueError("voice sources must be unique, positive, and non-dominant")
    if min(indices) < 0 or max(indices) >= len(centroids):
        raise ValueError("voice source index is outside the checkpoint")
    weight = torch.tensor(values, dtype=centroids.dtype, device=centroids.device)
    weight = weight / weight.sum()
    mixed = (centroids[list(indices)] * weight[:, None]).sum(dim=0)
    return F.normalize(mixed, dim=0)


def farthest_point_indices(centroids: torch.Tensor, count: int) -> list[int]:
    """Deterministically select diverse real-style anchors without labels."""
    if centroids.ndim != 2 or not 1 <= count <= len(centroids):
        raise ValueError("invalid centroid selection request")
    normalized = F.normalize(centroids.float(), dim=1)
    mean = F.normalize(normalized.mean(dim=0), dim=0)
    selected = [int(torch.argmin(normalized @ mean))]
    while len(selected) < count:
        similarity = normalized @ normalized[selected].T
        nearest = similarity.max(dim=1).values
        nearest[selected] = 2.0
        selected.append(int(torch.argmin(nearest)))
    return selected


def build_novel_voice_recipes(
    centroids: torch.Tensor,
    names: tuple[str, ...] = ("Aster", "Cobalt", "Juniper"),
    *,
    seed: int = 1701,
    candidates: int = 20_000,
) -> list[NovelVoiceRecipe]:
    if len(centroids) < 3 or candidates < len(names):
        raise ValueError("insufficient voice centroids or mixture candidates")
    normalized = F.normalize(centroids.float().cpu(), dim=1)
    rng = np.random.default_rng(seed)
    styles = []
    ingredients = []
    novelty = []
    for _ in range(candidates):
        indices = tuple(int(value) for value in rng.choice(len(normalized), 3, replace=False))
        weights = rng.dirichlet(np.full(3, 2.0))
        if weights.min() < 0.10 or weights.max() > 0.60:
            continue
        style = normalize_style_mix(normalized, indices, weights)
        styles.append(style)
        ingredients.append((indices, tuple(float(value) for value in weights)))
        novelty.append(float((normalized @ style).max()))
    if len(styles) < len(names):
        raise RuntimeError("novel voice search produced too few valid mixtures")
    style_matrix = torch.stack(styles)
    novelty_tensor = torch.tensor(novelty)
    selected = [int(torch.argmin(novelty_tensor))]
    while len(selected) < len(names):
        similarity = style_matrix @ style_matrix[selected].T
        # A candidate is only as distinct as its closest training or already-selected
        # voice. Minimax selection prevents several mixtures collapsing to one region.
        score = torch.maximum(similarity.max(dim=1).values, novelty_tensor)
        score[selected] = 2.0
        selected.append(int(torch.argmin(score)))
    return [
        NovelVoiceRecipe(name, ingredients[index][0], ingredients[index][1])
        for name, index in zip(names, selected)
    ]


def _take_end(features: np.ndarray, frames: int) -> np.ndarray:
    if len(features) >= frames:
        return features[-frames:].copy()
    return np.concatenate((
        np.repeat(features[:1], frames - len(features), axis=0), features
    ), axis=0)


def _take_start(features: np.ndarray, frames: int) -> np.ndarray:
    if len(features) >= frames:
        return features[:frames].copy()
    return np.concatenate((
        features, np.repeat(features[-1:], frames - len(features), axis=0)
    ), axis=0)


@torch.inference_mode()
def synthesize_boundary(
    left: np.ndarray,
    right: np.ndarray,
    mean_model,
    timing_model: TransitionSpanPredictorV17,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, int]:
    """Generate the previously unseen motion between two complete generated signs."""
    device = torch.device(device)
    context = np.concatenate((_take_end(left, 8), _take_start(right, 8)), axis=0)
    context_tensor = torch.from_numpy(context.astype(np.float32))[None].to(device)
    span = int(timing_model(context_tensor).argmax(dim=1).item())
    span += int(timing_model.config.minimum_span)
    start = (32 - span) // 2
    stop = start + span
    canvas = np.zeros((32, 61, 5), dtype=np.float32)
    canvas[:start] = _take_end(left, start)
    canvas[stop:] = _take_start(right, 32 - stop)
    mask = np.zeros((1, 32), dtype=np.bool_)
    mask[:, start:stop] = True
    features = torch.from_numpy(canvas)[None].to(device)
    generated = mean_model(features, torch.from_numpy(mask).to(device))[0]
    transition = generated[start:stop].cpu().numpy().astype(np.float32)
    transition[..., 3] = (transition[..., 3] >= 0.5).astype(np.float32)
    transition[..., 4] = np.clip(transition[..., 4], 0.0, 1.0)
    transition[..., :3] *= transition[..., 3:4]
    transition[..., 4] *= transition[..., 3]
    if not np.isfinite(transition).all():
        raise RuntimeError("transition generator emitted non-finite values")
    return transition, span


@torch.inference_mode()
def generate_isolated_signs(
    model: SigningVoiceGeneratorV17,
    checkpoint: dict[str, object],
    glosses: list[str],
    style: torch.Tensor,
    device: torch.device | str = "cpu",
) -> tuple[list[np.ndarray], list[int]]:
    device = torch.device(device)
    label_to_index = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    missing = [gloss for gloss in glosses if gloss not in label_to_index]
    if missing:
        raise ValueError(f"glosses are outside the 100-class vocabulary: {missing}")
    targets = torch.tensor([label_to_index[value] for value in glosses], device=device)
    prototypes = checkpoint["content_prototypes"][targets.cpu()].float().to(device)
    styles = style.to(device)[None].expand(len(targets), -1)
    generated = model.generate_from_style(prototypes, targets, styles)
    return [value.cpu().numpy().astype(np.float32) for value in generated], targets.cpu().tolist()


def voice_duration_ratio(checkpoint: dict[str, object], recipe: NovelVoiceRecipe) -> float:
    ratios = checkpoint["train_voice_duration_ratios"].float()
    weights = torch.tensor(recipe.weights, dtype=ratios.dtype)
    weights /= weights.sum()
    return float((ratios[list(recipe.source_voice_indices)] * weights).sum())


def compose_phrase(
    isolated_signs: list[np.ndarray],
    targets: list[int],
    duration_ratio: float,
    class_median_observed_frames: torch.Tensor,
    mean_model,
    timing_model: TransitionSpanPredictorV17,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    if not isolated_signs or len(isolated_signs) != len(targets):
        raise ValueError("isolated signs and targets must align")
    medians = class_median_observed_frames.cpu().numpy()
    signs = []
    for sign, target in zip(isolated_signs, targets):
        duration = int(np.clip(round(float(medians[target]) * duration_ratio), 8, 64))
        signs.append(resample_features(sign, duration))
    stream = signs[0]
    timeline: list[dict[str, int | str]] = [{
        "kind": "gloss", "target": targets[0], "start": 0, "stop": len(stream)
    }]
    for sign, target in zip(signs[1:], targets[1:]):
        transition, span = synthesize_boundary(
            stream, sign, mean_model, timing_model, device
        )
        boundary_start = len(stream)
        stream = np.concatenate((stream, transition, sign), axis=0)
        timeline.append({
            "kind": "transition", "start": boundary_start,
            "stop": boundary_start + span,
        })
        timeline.append({
            "kind": "gloss", "target": target,
            "start": boundary_start + span, "stop": len(stream),
        })
    return stream.astype(np.float32), timeline
