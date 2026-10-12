"""Shared timestamp windows and revisable decoding for Stage-1 streaming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WINDOW_SECONDS = 0.53
STRIDE_SECONDS = 0.13
WINDOW_FRAMES = 32
NO_EMIT = "NO_EMIT"
CHECKPOINT_FORMAT = "slt_stage1_window_v17"


def _timestamps(value, expected: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (expected is not None and len(result) != expected):
        raise ValueError("timestamps must have shape [T]")
    if not np.isfinite(result).all() or len(result) < 2 or np.any(np.diff(result) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return result


def window_sample_times(
    timestamps, end_seconds: float, window_seconds: float = WINDOW_SECONDS,
    frames: int = WINDOW_FRAMES,
) -> np.ndarray:
    times = _timestamps(timestamps)
    if not np.isfinite(end_seconds) or window_seconds <= 0 or frames < 2:
        raise ValueError("window end and positive duration/frame count are required")
    start = max(float(times[0]), float(end_seconds) - window_seconds)
    stop = min(float(times[-1]), float(end_seconds))
    if stop <= start or np.count_nonzero((times >= start) & (times <= stop)) < 2:
        raise ValueError("window must contain at least two timestamped samples")
    return np.linspace(start, stop, frames, dtype=np.float64)


def select_time_window(
    features: np.ndarray, timestamps, end_seconds: float,
    window_seconds: float = WINDOW_SECONDS, frames: int = WINDOW_FRAMES,
) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1:] != (61, 5):
        raise ValueError("features must have shape [T, 61, 5]")
    times = _timestamps(timestamps, len(values))
    samples = window_sample_times(times, end_seconds, window_seconds, frames)
    result = np.empty((frames, 61, 5), dtype=np.float32)
    flat = values.reshape(len(values), -1)
    for column in range(flat.shape[1]):
        result.reshape(frames, -1)[:, column] = np.interp(
            samples, times, flat[:, column]
        )
    right = np.searchsorted(times, samples).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    nearest = np.where(samples - times[left] <= times[right] - samples, left, right)
    result[..., 3] = values[nearest, :, 3] >= .5
    result[..., :3] *= result[..., 3:4]
    result[..., 4] *= result[..., 3]
    return result


RAW_FORMAT = 'apple_vision_isotropic_xy_confidence_v1'


def raw_observation_features(observations):
    """Store Vision observations before any window-specific normalization."""
    from active.v17.geometry_v17 import image_normalized_to_isotropic
    raw = np.zeros((len(observations), 61, 5), np.float32)
    for index, item in enumerate(observations):
        for side, start in (('left', 0), ('right', 21)):
            hand = item.assigned[side]
            if hand is not None:
                if hand.world_xyz is not None:
                    raise ValueError('Stage1 window raw contract requires Apple Vision without world depth')
                raw[index, start:start+21, :2] = hand.xy
                raw[index, start:start+21, 4] = hand.confidence
        raw[index, 57:, :2] = item.detection.body_xy
        raw[index, 57:, 4] = item.detection.body_confidence
        if item.face_for_features:
            raw[index, 42:57, :2] = item.detection.face_xy
            raw[index, 42:57, 4] = item.detection.face_confidence
        height, width = item.frame.shape[:2]
        raw[index, :, 3] = raw[index, :, 4] > 0
        raw[index, :, :2] = image_normalized_to_isotropic(
            raw[index, :, :2], width, height, raw[index, :, 3] > 0,
        )
    return raw, np.asarray([o.seconds for o in observations], np.float64)


def normalize_time_window(raw, timestamps, end_seconds, window_seconds=WINDOW_SECONDS):
    """Normalize within the requested window, then resample its actual clock."""
    from active.v17.geometry_v17 import body_relative_normalize, interpolate_short_gaps
    values = np.asarray(raw, np.float32)
    times = _timestamps(timestamps, len(values))
    if values.shape != (len(times), 61, 5) or not np.isfinite(values).all():
        raise ValueError('invalid raw Vision tensor')
    keep = (times >= end_seconds - window_seconds - 1e-9) & (times <= end_seconds + 1e-9)
    values, times = values[keep].copy(), times[keep]
    _timestamps(times)
    if np.any(np.diff(times) > .26):
        raise ValueError('timestamp_gap')
    xy, confidence = values[..., :2].copy(), values[..., 4].copy()
    hand_frames = int((confidence[:, :42] > 0).any(axis=1).sum())
    xy[:, :42], confidence[:, :42] = interpolate_short_gaps(xy[:, :42], confidence[:, :42], 3)
    xy[:, 42:], confidence[:, 42:] = interpolate_short_gaps(xy[:, 42:], confidence[:, 42:], 16)
    normalized, depth, diagnostics = body_relative_normalize(xy, confidence)
    values[..., :2], values[..., 2] = normalized, depth
    values[..., 3], values[..., 4] = confidence > 0, confidence.clip(0, 1)
    features = select_time_window(values, times, end_seconds, window_seconds)
    diagnostics.update(source_frames=len(times), observed_hand_frames=hand_frames,
                       hand_presence_fraction=float(features[:, :42, 3].mean()))
    return features, diagnostics


def window_end_times(
    timestamps, window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS, *, include_final: bool = False,
) -> list[float]:
    times = _timestamps(timestamps)
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window and stride seconds must be positive")
    first = float(times[0]) + window_seconds
    count = max(0, int(np.floor((float(times[-1]) - first) / stride_seconds)) + 1)
    ends = [first + index * stride_seconds for index in range(count)]
    ends = [end for end in ends if np.count_nonzero(
        (times >= end - window_seconds) & (times <= end)
    ) >= 2]
    if include_final and (not ends or not np.isclose(ends[-1], times[-1])):
        try:
            window_sample_times(times, float(times[-1]), window_seconds)
        except ValueError:
            pass
        else:
            ends.append(float(times[-1]))
    return ends


@dataclass(frozen=True)
class WindowPrediction:
    label: str
    end_seconds: float


def _decode(predictions: list[WindowPrediction], agreements: int) -> list[str]:
    words: list[str] = []
    start = 0
    while start < len(predictions):
        end = start + 1
        while end < len(predictions) and predictions[end].label == predictions[start].label:
            end += 1
        label = predictions[start].label
        if label != NO_EMIT and end - start >= agreements:
            words.append(label)
        start = end
    return words


class Stage1WindowTranscript:
    def __init__(self, agreements: int = 2, revision_seconds: float = 2.0) -> None:
        if agreements < 1 or revision_seconds <= 0:
            raise ValueError("agreements and revision seconds must be positive")
        self.agreements = agreements
        self.revision_seconds = revision_seconds
        self.predictions: list[WindowPrediction] = []
        self.words: list[str] = []

    def update(self, label: str, end_seconds: float) -> list[str]:
        if not label or not np.isfinite(end_seconds):
            raise ValueError("prediction label and finite timestamp are required")
        if self.predictions and end_seconds <= self.predictions[-1].end_seconds:
            raise ValueError("prediction timestamps must be strictly increasing")
        self.predictions.append(WindowPrediction(label, float(end_seconds)))
        # ponytail: O(n) retained-run decode; cache the completed prefix if long-
        # utterance measurements show this competing with classifier latency.
        self.words = _decode(self.predictions, self.agreements)
        return list(self.words)

    def reset(self) -> None:
        self.predictions.clear()
        self.words.clear()

    def replace_recent(self, predictions: list[WindowPrediction]) -> list[str]:
        """Recompute a corrected recent tail without permitting old-history edits."""
        if not predictions:
            return list(self.words)
        times = [p.end_seconds for p in predictions]
        if any(not p.label or not np.isfinite(p.end_seconds) for p in predictions) or any(
            b <= a for a, b in zip(times, times[1:])
        ):
            raise ValueError('invalid replacement predictions')
        if self.predictions and times[0] < self.predictions[-1].end_seconds - self.revision_seconds:
            raise ValueError('replacement precedes the revision horizon')
        self.predictions = [p for p in self.predictions if p.end_seconds < times[0]] + list(predictions)
        self.words = _decode(self.predictions, self.agreements)
        return list(self.words)


def load_stage1_window_checkpoint(path, device="cpu"):
    """Load the one supported Stage-1 window checkpoint contract."""
    import torch

    from active.v17.model_reel_emission_v17 import (
        ReelEmissionHeadConfig,
        ReelEmissionHeadV17,
        ReelEmissionStage1V17,
    )
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    schedule = checkpoint.get("stage1_window", {})
    expected = {
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "training_window_seconds": [0.27, WINDOW_SECONDS, 1.07],
        "frames": WINDOW_FRAMES,
        "no_emit_index": 100,
    }
    if checkpoint.get("format") != CHECKPOINT_FORMAT or any(
        schedule.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("checkpoint does not match the v17 Stage-1 window contract")
    labels = checkpoint.get("label_to_index", {})
    if labels.get("__NO_EMIT__") != 100 or sorted(labels.values()) != list(range(101)):
        raise ValueError("checkpoint must contain 100 glosses plus __NO_EMIT__:100")
    base = SLTStage1V17(Stage1V17Config(**checkpoint["base_model_config"]))
    base.load_state_dict(checkpoint["base_model_state_dict"], strict=True)
    head = ReelEmissionHeadV17(
        ReelEmissionHeadConfig(**checkpoint["emission_head_config"])
    )
    head.load_state_dict(checkpoint["emission_head_state_dict"], strict=True)
    model = ReelEmissionStage1V17(base, head).to(device).eval()
    ordered = [key for key, _ in sorted(labels.items(), key=lambda row: row[1])]
    ordered[100] = NO_EMIT
    return model, ordered, checkpoint
