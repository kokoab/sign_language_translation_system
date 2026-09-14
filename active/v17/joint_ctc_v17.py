"""Research-only differentiable CTC over timestamp-owned Stage-1 frame tokens."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from active.v17.model_unified_streaming_ctc_v17 import (
    UnifiedStreamingCTCConfig, UnifiedStreamingCTCHeadV17,
)
from active.v17.stage1_window_v17 import normalize_time_window, window_sample_times


@dataclass
class Chunks:
    features: np.ndarray
    times: list[np.ndarray]
    keep: list[np.ndarray]
    ends: list[float]

    @property
    def length(self):
        return sum(int(k.sum()) for k in self.keep)


def chunks(raw, timestamps, seconds=.53):
    """Close bounded chunks; each output timestamp has exactly one owner.

    An endpoint shared by two input chunks is context only in the second chunk.
    A one-observation Finish tail borrows the previous observation as context.
    No token is available before its entire normalized chunk has been observed.
    """
    raw, times = np.asarray(raw), np.asarray(timestamps, dtype=np.float64)
    if (raw.shape != (len(times), 61, 5) or len(times) < 2 or seconds <= 0
            or not np.isfinite(raw).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid timestamped chunk input')
    if np.any(np.diff(times) > .26):
        raise ValueError('timestamp_gap')
    features, sample_times, keeps, ends = [], [], [], []
    previous = 0
    while previous < len(times)-1:
        end = int(np.searchsorted(times, times[previous]+seconds, side='right'))-1
        if end <= previous:
            raise ValueError('chunk_duration_cannot_hold_two_observations')
        start = previous
        local = times[start:end+1]
        duration = float(local[-1] - local[0])
        value, _ = normalize_time_window(raw[start:end+1], local, float(local[-1]), duration)
        sampled = window_sample_times(local, float(local[-1]), duration)
        keep = np.ones(32, bool) if not features else sampled > ends[-1] + 1e-9
        features.append(value); sample_times.append(sampled); keeps.append(keep)
        ends.append(float(local[-1])); previous = end
    return Chunks(np.stack(features), sample_times, keeps, ends)


def sequence_targets(row, labels):
    if row.get('all_signs_annotated') is not True:
        raise ValueError('incomplete annotations cannot supply full-sequence CTC')
    targets = []
    for event in row['intervals']:
        label = event['label']
        if label == '__OTHER__':
            target = 101
        elif label in labels:
            target = labels[label] + 1
        else:
            raise ValueError('unmapped complete-sequence label: ' + str(label))
        if 'ctc_index' in event and int(event['ctc_index']) != target:
            raise ValueError('annotation CTC index disagrees with frozen labels')
        targets.append(target)
    return tuple(targets)


def core_crop(raw, times, event):
    """Normalize observed positive-core poses; a single pose has no motion evidence."""
    start, end = float(event['start_seconds']), float(event['end_seconds'])
    keep = (times >= start) & (times <= end)
    count = int(keep.sum())
    if count == 0 or end <= start:
        raise ValueError('core_has_no_observation_or_invalid_duration')
    observed, clock = raw[keep], times[keep]
    if count == 1:
        # The 32-frame interface repeats this observed static pose, never a trajectory.
        observed = np.repeat(observed, 2, axis=0)
        clock = np.array([start, end])
    value, _ = normalize_time_window(observed, clock, end, end-start)
    return value, count


def ctc_loss(logits, targets, lengths):
    if logits.ndim != 3 or logits.shape[-1] != 102 or len(targets) != len(logits):
        raise ValueError('CTC requires [B,T,102] and one target per sequence')
    for target, length in zip(targets, lengths):
        required = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        if required > length or length > logits.shape[1] or length < 1:
            raise ValueError('infeasible CTC input/target length')
        if any(t < 1 or t > 101 for t in target):
            raise ValueError('blank or invalid target in CTC transcript')
    # MPS has no native CTC; this device transfer preserves encoder gradients.
    logp = logits.float().log_softmax(-1).transpose(0, 1).cpu()
    flat = torch.tensor([t for seq in targets for t in seq], dtype=torch.long)
    losses = F.ctc_loss(logp, flat, torch.tensor(lengths),
                        torch.tensor([len(t) for t in targets]),
                        blank=0, reduction='none', zero_infinity=False)
    losses = losses / torch.tensor([max(1, len(t)) for t in targets])
    if not torch.isfinite(losses).all():
        raise ValueError('nonfinite CTC loss')
    return losses.mean().to(logits.device)


class JointCTC(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = UnifiedStreamingCTCHeadV17(UnifiedStreamingCTCConfig(
            stage1_dim=base.config.dim))

    def evidence(self, features):
        encoded, _ = self.base.encode(features)
        return torch.cat((encoded, self.base.classifier(encoded)), dim=-1)

    def tokens(self, features):
        return self.head(self.evidence(features))

    def sequences(self, values):
        device = next(self.parameters()).device
        features = torch.from_numpy(np.concatenate([v.features for v in values])).to(device)
        encoded = torch.cat([self.evidence(x) for x in features.split(64)])
        sequences, cursor = [], 0
        for value in values:
            parts = []
            for keep in value.keep:
                parts.append(encoded[cursor, torch.as_tensor(keep, device=device)])
                cursor += 1
            sequences.append(torch.cat(parts))
        lengths = [len(v) for v in sequences]
        logits = self.head(nn.utils.rnn.pad_sequence(sequences, batch_first=True))
        return logits, lengths


def decode(path):
    """Collapse before removing OTHER, preserving repeated known signs around it."""
    output, previous = [], None
    for value in path:
        value = int(value)
        if value != previous and 1 <= value <= 100:
            output.append(value)
        previous = value
    return output
