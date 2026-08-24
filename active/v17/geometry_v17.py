"""Pure NumPy geometry and temporal processing for v17."""

from __future__ import annotations

import numpy as np

from .schema_v17 import (
    BODY_START,
    FACE_END,
    FACE_START,
    LHAND_START,
    NUM_NODES,
    RHAND_START,
)


def image_normalized_to_isotropic(
    xy: np.ndarray,
    image_width: int,
    image_height: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Map Vision normalized XY to centered, isotropic image coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    values = np.asarray(xy, dtype=np.float32)
    if values.shape[-1] != 2:
        raise ValueError(f"expected XY in the final dimension, got {values.shape}")
    longest = float(max(image_width, image_height))
    result = values.copy()
    result[..., 0] = (result[..., 0] * image_width - image_width / 2.0) / longest
    result[..., 1] = (result[..., 1] * image_height - image_height / 2.0) / longest
    if valid is not None:
        result *= np.asarray(valid, dtype=np.float32)[..., None]
    return result


def interpolate_short_gaps(
    xy: np.ndarray,
    confidence: np.ndarray,
    max_gap: int,
    imputed_confidence_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill only bounded gaps; never extrapolate or invent a track."""
    values = np.asarray(xy, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32).copy()
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("xy must have shape [T, N, 2]")
    if conf.shape != values.shape[:2]:
        raise ValueError("confidence must have shape [T, N]")
    if max_gap <= 0:
        values[conf <= 0] = 0
        return values, conf

    frames, nodes = conf.shape
    for node in range(nodes):
        valid_indices = np.flatnonzero(conf[:, node] > 0)
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            gap = int(right - left - 1)
            if gap <= 0 or gap > max_gap:
                continue
            start = values[left, node]
            end = values[right, node]
            endpoint_conf = min(conf[left, node], conf[right, node])
            for offset in range(1, gap + 1):
                fraction = offset / float(gap + 1)
                values[left + offset, node] = start + fraction * (end - start)
                conf[left + offset, node] = endpoint_conf * imputed_confidence_scale
    values[conf <= 0] = 0
    return values, conf


def interpolate_scalar_short_gaps(
    values: np.ndarray,
    confidence: np.ndarray,
    max_gap: int,
    imputed_confidence_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded interpolation for one scalar per frame/node.

    This is used for genuine detector-provided depth. It follows the same no-
    extrapolation rule as XY interpolation and returns an independent validity map.
    """
    scalar = np.asarray(values, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32).copy()
    if scalar.ndim != 2 or conf.shape != scalar.shape:
        raise ValueError("values and confidence must both have shape [T, N]")
    if max_gap <= 0:
        scalar[conf <= 0] = 0
        return scalar, conf
    frames, nodes = conf.shape
    for node in range(nodes):
        valid_indices = np.flatnonzero(conf[:, node] > 0)
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            gap = int(right - left - 1)
            if gap <= 0 or gap > max_gap:
                continue
            endpoint_conf = min(conf[left, node], conf[right, node])
            for offset in range(1, gap + 1):
                fraction = offset / float(gap + 1)
                scalar[left + offset, node] = (
                    scalar[left, node]
                    + fraction * (scalar[right, node] - scalar[left, node])
                )
                conf[left + offset, node] = endpoint_conf * imputed_confidence_scale
    scalar[conf <= 0] = 0
    return scalar, conf


def _palm_lengths(xy: np.ndarray, confidence: np.ndarray, hand_start: int):
    wrist_valid = confidence[:, hand_start] > 0
    mcp_valid = confidence[:, hand_start + 9] > 0
    valid = wrist_valid & mcp_valid
    lengths = np.linalg.norm(
        xy[:, hand_start + 9] - xy[:, hand_start], axis=-1
    ).astype(np.float32)
    valid &= lengths > 1e-5
    return lengths, valid


def body_relative_normalize(
    xy: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Center on the torso and use one robust sequence scale.

    Missing nodes remain exactly zero. A time-varying torso center removes
    camera/body drift without removing hand movement relative to the signer.
    """
    values = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if values.shape != (values.shape[0], NUM_NODES, 2):
        raise ValueError(f"expected [T, {NUM_NODES}, 2], got {values.shape}")
    frames = values.shape[0]
    left_shoulder = BODY_START
    right_shoulder = BODY_START + 1
    shoulder_valid = (conf[:, left_shoulder] > 0) & (conf[:, right_shoulder] > 0)
    shoulder_width = np.linalg.norm(
        values[:, right_shoulder] - values[:, left_shoulder], axis=-1
    )
    shoulder_valid &= shoulder_width > 1e-5

    centers = np.zeros((frames, 2), dtype=np.float32)
    if shoulder_valid.any():
        centers[shoulder_valid] = (
            values[shoulder_valid, left_shoulder]
            + values[shoulder_valid, right_shoulder]
        ) / 2.0
        known = np.flatnonzero(shoulder_valid)
        timeline = np.arange(frames)
        for axis in range(2):
            centers[:, axis] = np.interp(
                timeline, known, centers[known, axis]
            ).astype(np.float32)
        scale = float(np.median(shoulder_width[shoulder_valid]))
        scale_source = "shoulder_width"
    else:
        wrist_points = []
        for hand_start in (LHAND_START, RHAND_START):
            valid = conf[:, hand_start] > 0
            if valid.any():
                wrist_points.append(values[valid, hand_start])
        fallback_center = (
            np.median(np.concatenate(wrist_points), axis=0)
            if wrist_points else np.zeros(2, dtype=np.float32)
        )
        centers[:] = fallback_center
        palm_values = []
        for hand_start in (LHAND_START, RHAND_START):
            lengths, valid = _palm_lengths(values, conf, hand_start)
            if valid.any():
                palm_values.extend(lengths[valid].tolist())
        scale = float(np.median(palm_values)) if palm_values else 1.0
        scale_source = "palm_length" if palm_values else "unit_fallback"

    if not np.isfinite(scale) or scale <= 1e-5:
        scale = 1.0
        scale_source = "unit_fallback"

    normalized = (values - centers[:, None, :]) / scale
    presence = conf > 0
    normalized *= presence[..., None]

    depth = np.zeros(conf.shape, dtype=np.float32)
    for hand_start in (LHAND_START, RHAND_START):
        lengths, valid = _palm_lengths(values, conf, hand_start)
        if not valid.any():
            continue
        reference = float(np.median(lengths[valid]))
        hand_depth = np.zeros(frames, dtype=np.float32)
        hand_depth[valid] = np.log(reference / lengths[valid])
        depth[:, hand_start:hand_start + 21] = hand_depth[:, None]

    if shoulder_valid.any():
        reference = float(np.median(shoulder_width[shoulder_valid]))
        torso_depth = np.zeros(frames, dtype=np.float32)
        torso_depth[shoulder_valid] = np.log(
            reference / shoulder_width[shoulder_valid]
        )
        depth[:, FACE_START:FACE_END] = torso_depth[:, None]
        depth[:, BODY_START:] = torso_depth[:, None]
    depth *= presence

    diagnostics = {
        "normalization_scale": scale,
        "normalization_scale_source": scale_source,
        "shoulder_coverage": float(shoulder_valid.mean()),
    }
    return normalized.astype(np.float32), depth, diagnostics


def resample_features(features: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample continuous channels linearly and presence with nearest-neighbor."""
    values = np.asarray(features, dtype=np.float32)
    frames = values.shape[0]
    if frames < 1:
        raise ValueError("cannot resample an empty sequence")
    if frames == target_frames:
        result = values.copy()
        result[..., 3] = (result[..., 3] >= 0.5).astype(np.float32)
        result[..., :3] *= result[..., 3:4]
        result[..., 4] *= result[..., 3]
        return result

    source_t = np.linspace(0.0, 1.0, frames)
    target_t = np.linspace(0.0, 1.0, target_frames)
    result = np.zeros((target_frames,) + values.shape[1:], dtype=np.float32)
    for node in range(values.shape[1]):
        for channel in (0, 1, 2, 4):
            result[:, node, channel] = np.interp(
                target_t, source_t, values[:, node, channel]
            )
    nearest = np.rint(target_t * (frames - 1)).astype(int)
    result[..., 3] = (values[nearest, :, 3] >= 0.5).astype(np.float32)
    result[..., :3] *= result[..., 3:4]
    result[..., 4] *= result[..., 3]
    return result
