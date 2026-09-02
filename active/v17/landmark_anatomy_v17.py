"""Animation-ready completion of sparse v17 detector landmarks."""

from __future__ import annotations

import numpy as np

from .schema_v17 import NUM_NODES


IMPUTED_CONFIDENCE = 0.05
HAND_SPECS = ((0, 21, 59), (21, 42, 60))


def _bridge_short_false_gaps(active: np.ndarray, maximum_gap: int) -> np.ndarray:
    output = np.asarray(active, dtype=bool).copy()
    known = np.flatnonzero(output)
    for left, right in zip(known[:-1], known[1:]):
        if right - left - 1 <= maximum_gap:
            output[left:right + 1] = True
    return output


def build_anatomy_template(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Learn canonical anatomy only from observed train-split landmarks."""
    values = np.asarray(landmarks, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (32, NUM_NODES, 5):
        raise ValueError("landmark pool must be [items,32,61,5]")
    absolute_xyz = np.zeros((NUM_NODES, 3), dtype=np.float32)
    for node in range(NUM_NODES):
        observed = values[:, :, node, 3] > 0
        if not observed.any():
            raise ValueError(f"train pool never observes node {node}")
        absolute_xyz[node] = np.median(values[:, :, node, :3][observed], axis=0)

    hand_shapes = []
    wrist_from_elbow = []
    for start, stop, elbow in HAND_SPECS:
        hand = values[:, :, start:stop]
        complete = (hand[..., 3] > 0).all(axis=2)
        if not complete.any():
            raise ValueError("train pool has no complete hand frame")
        relative = hand[complete, :, :3] - hand[complete, :1, :3]
        hand_shapes.append(np.median(relative, axis=0))
        paired = (hand[:, :, 0, 3] > 0) & (values[:, :, elbow, 3] > 0)
        if not paired.any():
            raise ValueError("train pool has no observed wrist/elbow pair")
        wrist_from_elbow.append(np.median(
            hand[:, :, 0, :3][paired] - values[:, :, elbow, :3][paired], axis=0
        ))
    return {
        "absolute_xyz": absolute_xyz,
        "hand_shapes": np.asarray(hand_shapes, dtype=np.float32),
        "wrist_from_elbow": np.asarray(wrist_from_elbow, dtype=np.float32),
    }


def anatomy_coverage(features: np.ndarray) -> float:
    """Score source detection coverage without requiring a second active hand."""
    value = np.asarray(features)
    if value.ndim != 3 or value.shape[1:] != (NUM_NODES, 5):
        raise ValueError("features must be [frames,61,5]")
    present = value[..., 3] > 0
    active = present[:, :42].any(axis=1)
    if not active.any():
        return 0.0
    left = present[active, :21].mean(axis=1)
    right = present[active, 21:42].mean(axis=1)
    face = present[active, 42:57].mean(axis=1)
    body = present[active, 57:61].mean(axis=1)
    return float(np.mean(
        0.45 * np.maximum(left, right)
        + 0.15 * np.minimum(left, right)
        + 0.20 * face
        + 0.20 * body
    ))


def complete_landmark_anatomy(
    features: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    imputed_confidence: float = IMPUTED_CONFIDENCE,
    hand_visibility_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a complete 61-node animation rig and the original observation mask.

    Real coordinates are never replaced. Intermittent gaps use the same node's
    observed trajectory. A hand never observed in the clip remains absent; this
    preserves whether the sign is genuinely one- or two-handed.
    """
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (NUM_NODES, 5) or not len(value):
        raise ValueError("features must be non-empty [frames,61,5]")
    if (
        not np.isfinite(value).all()
        or not 0.0 <= imputed_confidence <= 1.0
        or hand_visibility_gap < 0
    ):
        raise ValueError("invalid landmark values or imputed confidence")
    absolute = np.asarray(template["absolute_xyz"], dtype=np.float32)
    shapes = np.asarray(template["hand_shapes"], dtype=np.float32)
    wrists = np.asarray(template["wrist_from_elbow"], dtype=np.float32)
    if absolute.shape != (NUM_NODES, 3) or shapes.shape != (2, 21, 3) or wrists.shape != (2, 3):
        raise ValueError("invalid anatomy template")

    output = value.copy()
    observed = value[..., 3] > 0
    frames = np.arange(len(value))

    def fill_node(node: int, fallback_xyz: np.ndarray) -> None:
        known = observed[:, node]
        if known.any():
            for channel in range(3):
                output[:, node, channel] = np.interp(
                    frames, frames[known], value[known, node, channel]
                )
            output[:, node, 4] = np.interp(
                frames, frames[known], value[known, node, 4]
            )
        else:
            output[:, node, :3] = fallback_xyz
            output[:, node, 4] = imputed_confidence
        output[:, node, 3] = 1.0

    # Complete body and face first; hand rest poses depend on completed elbows.
    for node in range(42, NUM_NODES):
        fill_node(node, absolute[node])

    for side, (start, stop, elbow) in enumerate(HAND_SPECS):
        hand_observed = observed[:, start:stop]
        visibility = _bridge_short_false_gaps(
            hand_observed.any(axis=1), hand_visibility_gap
        )
        if not visibility.any():
            output[:, start:stop] = 0
            continue
        anchor = np.zeros((len(value), 3), dtype=np.float32)
        anchor_observed = np.zeros(len(value), dtype=bool)
        for frame in range(len(value)):
            nodes = np.flatnonzero(hand_observed[frame])
            if len(nodes):
                anchor[frame] = np.median(
                    value[frame, start + nodes, :3] - shapes[side, nodes], axis=0
                )
                anchor_observed[frame] = True
        if anchor_observed.any():
            for channel in range(3):
                anchor[:, channel] = np.interp(
                    frames, frames[anchor_observed], anchor[anchor_observed, channel]
                )
        for local_node in range(21):
            fill_node(start + local_node, anchor + shapes[side, local_node])
        output[~visibility, start:stop] = 0

    if not np.isfinite(output).all():
        raise RuntimeError("anatomy completion produced non-finite landmarks")
    return output, observed
