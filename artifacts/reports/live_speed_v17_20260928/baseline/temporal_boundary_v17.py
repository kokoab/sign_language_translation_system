"""Native-clock ASL boundaries: causal geometry, bounded look-ahead, event decoding."""
from __future__ import annotations

from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'slt_temporal_boundary_v17'
FPS = 20
START, END = range(2)


def validate_clock(times):
    times = np.asarray(times, np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    return times


def sample_indices(times, fps=FPS):
    """Match Reel's source-clock cadence without interpolating future observations."""
    times = validate_clock(times)
    indices, deadline = [], float(times[0])
    for i, t in enumerate(times):
        if t + 1e-6 >= deadline:
            indices.append(i)
            deadline = max(deadline + 1 / fps, t)
    return np.asarray(indices, np.int64)


def boundary_features(raw, times, hand_geometry=True):
    times = validate_clock(times)
    raw = np.asarray(raw, np.float32)
    if raw.shape != (len(times), 61, 5) or not np.isfinite(raw).all():
        raise ValueError('expected finite raw Apple Vision [T,61,5]')
    present = (raw[..., 3] > .5) & (raw[..., 4] > 0)
    xy = raw[..., :2]
    normal = np.zeros_like(xy)
    center, scale, last_body = np.zeros(2), .3, -np.inf
    for i, t in enumerate(times):
        if present[i, 57:59].all():
            width = float(np.linalg.norm(xy[i, 58] - xy[i, 57]))
            if width > .02:
                center, scale, last_body = xy[i, 57:59].mean(0), width, t
        if t - last_body > .8:
            center, scale = np.zeros(2), .3
        normal[i] = np.clip((xy[i] - center) / scale, -4, 4) * present[i, :, None]
    velocity = np.zeros_like(normal)
    if len(times) > 1:
        valid = present[1:] & present[:-1] & (np.diff(times)[:, None] <= .26)
        velocity[1:] = np.clip(np.diff(normal, axis=0) / np.diff(times)[:, None, None], -10, 10) * valid[..., None] / 10
    values = [normal.reshape(len(times), -1), velocity.reshape(len(times), -1),
              present.astype(np.float32), raw[..., 4].clip(0, 1) * present]
    if hand_geometry:
        hands = []
        for start in (0, 21):
            palm = np.linalg.norm(xy[:, start + 9] - xy[:, start], axis=-1)
            valid = present[:, start:start + 21] & present[:, start, None] & present[:, start + 9, None] & (palm[:, None] > .005)
            local = (xy[:, start:start + 21] - xy[:, start, None]) / np.maximum(palm[:, None, None], .005)
            hands.append(np.clip(local, -4, 4) * valid[..., None])
        values.extend(h.reshape(len(times), -1) for h in hands)
    return np.concatenate(values, axis=-1).astype(np.float32)


def boundary_targets(times, accepted, excluded, complete=False):
    """Independent edge targets permit START and END at the same frame; -1 is unknown."""
    times = validate_clock(times)
    y = np.full((len(times), 2), 0 if complete else -1, np.float32)
    for start, end in accepted:
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('invalid sign interval')
        inside = (times >= start - 1e-8) & (times <= end + 1e-8)
        y[inside[:, None] & (y < 0)] = 0
        for edge, channel in ((start, START), (end, END)):
            if times[0] - .025 <= edge <= times[-1] + .025:
                # A 100ms band tolerates annotation/sampling uncertainty, without
                # turning an unverified surrounding region into negative supervision.
                near = np.abs(times - edge) <= .050001
                y[near, channel] = 1
    for start, end in excluded:
        y[(times >= start - .05) & (times <= end + .05)] = -1
    return y


class TemporalBoundary(nn.Module):
    def __init__(self, input_dim=450, hidden=64, lookahead=4):
        super().__init__()
        if input_dim <= 0 or hidden <= 0 or not 0 <= lookahead <= 10:
            raise ValueError('invalid model dimensions/lookahead')
        self.config = dict(input_dim=input_dim, hidden=hidden, lookahead=lookahead)
        self.lookahead = lookahead
        self.project = nn.Conv1d(input_dim, hidden, 1)
        self.layers = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, dilation=d) for d in (1, 2, 4, 8))
        self.output = nn.Conv1d(hidden, 2, 1)
        self.left_context = 30

    def forward(self, features):
        x = F.gelu(self.project(features.transpose(1, 2)))
        for layer in self.layers:
            x = x + F.gelu(layer(F.pad(x, (2 * layer.dilation[0], 0))))
        return self.output(x).transpose(1, 2)[:, self.lookahead:]


def masked_boundary_loss(logits, targets, positive_weight):
    if logits.shape != targets.shape:
        raise ValueError('boundary logits/targets must align')
    valid = targets >= 0
    if not valid.any():
        raise ValueError('batch has no supervised frames')
    loss = F.binary_cross_entropy_with_logits(logits, targets.clamp_min(0),
                                             pos_weight=positive_weight, reduction='none')
    return loss[valid].mean()


class BoundaryDecoder:
    """One interval per edge pair; lexical identity never suppresses a real repeat."""
    def __init__(self, threshold=.5, minimum=.15, maximum=4.):
        if not 0 < threshold < 1 or not 0 < minimum < maximum:
            raise ValueError('invalid decoder settings')
        self.threshold, self.minimum, self.maximum = threshold, minimum, maximum
        self.reset()

    def reset(self):
        self.start = None
        self.last_time = -np.inf
        self.previous = np.zeros(2, bool)

    def _close(self, time):
        event = []
        if self.start is not None and self.minimum - 1e-8 <= time - self.start <= self.maximum:
            event = [dict(start_seconds=self.start, end_seconds=float(time))]
        self.start = None
        return event

    def update(self, time, probabilities):
        p = np.asarray(probabilities, np.float64)
        if p.shape != (2,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isfinite(time) or time <= self.last_time:
            raise ValueError('finite probabilities and increasing timestamps required')
        if time - self.last_time > .26:
            self.reset()
        high = p >= self.threshold
        rising = high & ~self.previous
        events = []
        if self.start is not None:
            if rising[END]:
                events.extend(self._close(time))
            elif rising[START] and time - self.start >= self.minimum:
                self.start = None  # Replace an unmatched start; it is not an end.
            elif time - self.start > self.maximum:
                self._close(time)  # Expiry discards; it never fabricates a completed sign.
        if rising[START] and self.start is None:
            self.start = float(time)
        self.previous, self.last_time = high, float(time)
        return events

    def finish(self):
        self.start = None
        return []


class BoundaryStream:
    """Same features/network as training, bounded storage and no motion activation gate."""
    def __init__(self, model, hand_geometry=True, threshold=.5):
        self.model = model.eval()
        self.hand_geometry = hand_geometry
        self.decoder = BoundaryDecoder(threshold)
        self.reset()

    def reset(self):
        self.raw, self.raw_times = deque(), deque()
        self.features = deque(maxlen=self.model.left_context + 1)
        self.times = deque(maxlen=self.model.lookahead + 1)
        self.decoder.reset()
        self.last_time = -np.inf

    @torch.inference_mode()
    def update(self, raw_frame, seconds):
        if not np.isfinite(seconds) or seconds <= self.last_time:
            raise ValueError('stream timestamps must increase')
        if seconds - self.last_time > .26:
            self.reset()
        self.last_time = float(seconds)
        self.raw.append(np.asarray(raw_frame, np.float32))
        self.raw_times.append(float(seconds))
        while len(self.raw_times) > 2 and self.raw_times[1] < seconds - 1.2:
            self.raw.popleft(); self.raw_times.popleft()
        feature = boundary_features(np.asarray(self.raw), self.raw_times, self.hand_geometry)[-1]
        self.features.append(feature)
        self.times.append(float(seconds))
        if len(self.features) <= self.model.lookahead:
            return None
        device = next(self.model.parameters()).device
        x = torch.as_tensor(np.asarray(self.features)[None], device=device)
        p = self.model(x)[0, -1].sigmoid().cpu().numpy()
        source_time = self.times[0]
        events = self.decoder.update(source_time, p)
        return dict(seconds=source_time, available_seconds=float(seconds),
                    probabilities=p.tolist(), events=events)
