"""Compose a complete landmark utterance in a novel continuous signing style."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .geometry_v17 import resample_features
from .model_transition_inpainter_v17 import interpolate_masked_context
from .model_signing_voice_v17 import SigningVoiceGeneratorV17, SigningVoiceV17Config
from .model_transition_span_v17 import TransitionSpanPredictorV17, TransitionSpanV17Config
from .train_transition_diffusion_v17 import load_mean_model


HAND_TREE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


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


def trim_observed_span(
    features: np.ndarray, activity: np.ndarray | None = None
) -> np.ndarray:
    """Remove extractor padding so transition endpoints are observed motion."""
    observed = (
        (features[:, :42, 3] > 0).any(axis=1)
        if activity is None else np.asarray(activity, dtype=bool)
    )
    if observed.shape != (len(features),):
        raise ValueError("activity mask must align with the isolated sign")
    if not observed.any():
        observed = (features[..., 3] > 0).any(axis=1)
    indices = np.flatnonzero(observed)
    if not len(indices):
        raise ValueError("isolated sign has no observed frames")
    return features[indices[0]:indices[-1] + 1]


def trim_transition_span(
    features: np.ndarray, activity: np.ndarray | None = None
) -> np.ndarray:
    """Trim partial-hand edges only when an isolated clip will be joined."""
    features = trim_observed_span(features, activity)
    observed = np.ones(len(features), dtype=bool)
    complete_hands = np.stack([
        (features[:, start:start + 21, 3] > 0).all(axis=1)
        for start in (0, 21)
    ])
    participating = complete_hands.any(axis=1)
    if participating.any():
        supported = complete_hands[participating].all(axis=0)
        if not (observed & supported).any():
            supported = complete_hands[participating].any(axis=0)
        observed &= supported
    indices = np.flatnonzero(observed)
    return features[indices[0]:indices[-1] + 1]


def _complete_hand_frame(features: np.ndarray, start: int, reverse: bool) -> np.ndarray | None:
    order = range(len(features) - 1, -1, -1) if reverse else range(len(features))
    for frame in order:
        hand = features[frame, start:start + 21]
        if (hand[:, 3] > 0).all():
            return hand.copy()
    return None


def stabilize_transition_hands(
    transition: np.ndarray, left: np.ndarray, right: np.ndarray,
) -> np.ndarray:
    """Keep learned wrist motion while preventing transition hand-shape collapse."""
    output = np.asarray(transition, dtype=np.float32).copy()
    for start in (0, 21):
        first = _complete_hand_frame(left, start, True)
        last = _complete_hand_frame(right, start, False)
        if first is None and last is None:
            output[:, start:start + 21] = 0
            continue
        present_on_left = first is not None
        present_on_right = last is not None
        first = last.copy() if first is None else first
        last = first.copy() if last is None else last
        first_shape = first[:, :3] - first[:1, :3]
        last_shape = last[:, :3] - last[:1, :3]
        first_lengths = np.asarray([
            np.linalg.norm(first_shape[child] - first_shape[parent])
            for parent, child in HAND_TREE_EDGES
        ], dtype=np.float32)
        last_lengths = np.asarray([
            np.linalg.norm(last_shape[child] - last_shape[parent])
            for parent, child in HAND_TREE_EDGES
        ], dtype=np.float32)
        active = (output[:, start:start + 21, 3] > 0).any(axis=1)
        if present_on_left and present_on_right:
            active[:] = True
        for frame, alpha in enumerate(np.linspace(0.0, 1.0, len(output))):
            if not active[frame]:
                output[frame, start:start + 21] = 0
                continue
            blended = first_shape * (1.0 - alpha) + last_shape * alpha
            rebuilt = np.zeros((21, 3), dtype=np.float32)
            lengths = first_lengths * (1.0 - alpha) + last_lengths * alpha
            for edge, (parent, child) in enumerate(HAND_TREE_EDGES):
                direction = blended[child] - blended[parent]
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    direction = last_shape[child] - last_shape[parent]
                    norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    direction = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
                    norm = 1.0
                rebuilt[child] = rebuilt[parent] + direction / norm * lengths[edge]
            wrist = output[frame, start, :3]
            if output[frame, start, 3] <= 0:
                wrist = first[0, :3] * (1.0 - alpha) + last[0, :3] * alpha
            confidence = float(
                first[:, 4].mean() * (1.0 - alpha) + last[:, 4].mean() * alpha
            )
            hand = output[frame, start:start + 21]
            hand[:, :3] = wrist + rebuilt
            hand[:, 3] = 1.0
            hand[:, 4] = confidence
    return output


def _transition_motion_ratios(
    left: np.ndarray, transition: np.ndarray, right: np.ndarray,
) -> dict[str, float]:
    features = np.concatenate((left, transition, right), axis=0)
    generated_frames = np.zeros(len(features), dtype=bool)
    generated_frames[len(left):len(left) + len(transition)] = True
    ratios = {}
    for name, order in (("speed", 1), ("acceleration", 2), ("jerk", 3)):
        delta = np.linalg.norm(
            np.diff(features[:, :42, :3], n=order, axis=0), axis=-1
        )
        valid = np.ones(delta.shape, dtype=bool)
        present = features[:, :42, 3] > 0
        touches_generated = np.zeros(len(delta), dtype=bool)
        for offset in range(order + 1):
            valid &= present[offset:offset + len(delta)]
            touches_generated |= generated_frames[offset:offset + len(delta)]
        gloss = delta[valid & ~touches_generated[:, None]]
        generated = delta[valid & touches_generated[:, None]]
        reference = float(np.percentile(gloss, 95)) if len(gloss) else 0.0
        ratios[name] = (
            float(np.percentile(generated, 95) / reference)
            if len(generated) and reference > 0 else 0.0
        )
    return ratios


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

    def generate(candidate_span: int) -> np.ndarray:
        start = (32 - candidate_span) // 2
        stop = start + candidate_span
        canvas = np.zeros((32, 61, 5), dtype=np.float32)
        canvas[:start] = _take_end(left, start)
        canvas[stop:] = _take_start(right, 32 - stop)
        mask = np.zeros((1, 32), dtype=np.bool_)
        mask[:, start:stop] = True
        features = torch.from_numpy(canvas)[None].to(device)
        mask_tensor = torch.from_numpy(mask).to(device)
        generated = mean_model(features, mask_tensor)[0]
        # Keep the learned spatial residual, but anchor unreliable auxiliary
        # channels to genuine neighboring observations.
        envelope = interpolate_masked_context(features, mask_tensor)[0]
        value = generated[start:stop].cpu().numpy().astype(np.float32)
        envelope = envelope[start:stop].cpu().numpy().astype(np.float32)
        value[..., 3] = (envelope[..., 3] >= 0.5).astype(np.float32)
        value[..., 4] = np.clip(envelope[..., 4], 0.0, 1.0)
        value[..., :3] *= value[..., 3:4]
        value[..., 4] *= value[..., 3]
        return stabilize_transition_hands(value, left, right)

    transition = generate(span)
    ratios = _transition_motion_ratios(left, transition, right)
    if ratios["speed"] < 0.25:
        minimum = int(timing_model.config.minimum_span)
        for candidate_span in range(span - 1, minimum - 1, -1):
            candidate = generate(candidate_span)
            candidate_ratios = _transition_motion_ratios(left, candidate, right)
            if all(0.25 <= value <= 4.0 for value in candidate_ratios.values()):
                transition, span = candidate, candidate_span
                break
    if not np.isfinite(transition).all():
        raise RuntimeError("transition generator emitted non-finite values")
    return transition, span


def synthesize_join(
    left: np.ndarray,
    right: np.ndarray,
    mean_model,
    timing_model: TransitionSpanPredictorV17,
    device: torch.device | str = "cpu",
    *,
    maximum_entry_trim: int = 2,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate a join, skipping at most two noisy genuine entry frames if needed."""
    transition, span = synthesize_boundary(
        left, right, mean_model, timing_model, device
    )
    ratios = _transition_motion_ratios(left, transition, right)
    if all(0.25 <= value <= 4.0 for value in ratios.values()):
        return transition, right, span, 0

    participating = [
        bool((right[:, start:start + 21, 3] > 0).all(axis=1).any())
        for start in (0, 21)
    ]
    for offset in range(1, min(maximum_entry_trim, len(right) - 8) + 1):
        candidate_right = right[offset:]
        if any(
            required and not (candidate_right[0, start:start + 21, 3] > 0).all()
            for required, start in zip(participating, (0, 21))
        ):
            continue
        candidate, candidate_span = synthesize_boundary(
            left, candidate_right, mean_model, timing_model, device
        )
        candidate_ratios = _transition_motion_ratios(
            left, candidate, candidate_right
        )
        if all(0.25 <= value <= 4.0 for value in candidate_ratios.values()):
            return candidate, candidate_right, candidate_span, offset
    return transition, right, span, 0


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
    activity_masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    if not isolated_signs or len(isolated_signs) != len(targets):
        raise ValueError("isolated signs and targets must align")
    medians = class_median_observed_frames.cpu().numpy()
    if activity_masks is not None and len(activity_masks) != len(isolated_signs):
        raise ValueError("activity masks must align with isolated signs")
    signs = []
    for index, (sign, target) in enumerate(zip(isolated_signs, targets)):
        duration = int(np.clip(round(float(medians[target]) * duration_ratio), 8, 64))
        activity = None if activity_masks is None else activity_masks[index]
        signs.append(resample_features(trim_transition_span(sign, activity), duration))
    stream = signs[0]
    timeline: list[dict[str, int | str]] = [{
        "kind": "gloss", "target": targets[0], "start": 0, "stop": len(stream)
    }]
    for sign, target in zip(signs[1:], targets[1:]):
        transition, sign, span, entry_trim = synthesize_join(
            stream, sign, mean_model, timing_model, device
        )
        boundary_start = len(stream)
        stream = np.concatenate((stream, transition, sign), axis=0)
        timeline.append({
            "kind": "transition", "start": boundary_start,
            "right_entry_trim_frames": entry_trim,
            "stop": boundary_start + span,
        })
        timeline.append({
            "kind": "gloss", "target": target,
            "start": boundary_start + span, "stop": len(stream),
        })
    return stream.astype(np.float32), timeline
