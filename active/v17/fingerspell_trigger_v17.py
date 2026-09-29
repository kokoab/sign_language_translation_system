"""FINGERSPELL-sign trigger: the ASL sign FINGERSPELL (open 5, fingers wiggling, sideways slide) as a
spell-mode switch. Window features over raw Apple Vision landmarks [T,61,5] (the live raw contract,
20 Hz) and a boosted-tree classifier exported to JSON (artifacts/models/fingerspell_trigger_v17).

Trained and evaluated in artifacts/reports/fingerspell_detector_v17_20260930. It detects the deliberate
(citation) form, ~1 s; the quick in-sentence FINGERSPELL is not detected.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

WINDOW, STEP = 20, 2
TIPS, PIPS = [8, 12, 16, 20], [6, 10, 14, 18]
N_HAND = 18
MODEL = Path(__file__).resolve().parents[2] / 'artifacts/models/fingerspell_trigger_v17/model.json'


def hand_features(raw, s):
    """N_HAND features of hand block s (0 or 21) over one window of raw [W,61,5] (isotropic xy)."""
    h = raw[:, s:s + 21]
    present = (h[:, :, 4] > 0).sum(1) >= 15
    f = np.zeros(N_HAND, np.float32)
    f[0] = present.mean()
    if present.sum() < 4:
        return f
    xy = h[present, :, :2]
    rel = xy - xy[:, :1]
    palm = np.linalg.norm(rel[:, 9], axis=1)
    palm_m = float(np.median(palm))
    good = palm > .6 * palm_m                                # drop collapsed / mis-tracked frames
    if good.sum() < 4 or palm_m < 1e-3:
        return f
    xy, rel, palm = xy[good], rel[good], palm[good]
    reln = rel / palm[:, None, None]
    tip_d = np.linalg.norm(reln[:, TIPS], axis=2); pip_d = np.linalg.norm(reln[:, PIPS], axis=2)
    ext = tip_d / np.maximum(pip_d, 1e-3)
    f[1] = ext.mean(); f[2] = ext.min(1).mean()                                     # finger extension
    v = np.diff(tip_d, axis=0)                                                      # per-finger length velocity
    f[3] = np.abs(v).mean()                                                         # wiggle amount
    f[4] = (np.diff(np.sign(v), axis=0) != 0).mean() if len(v) > 2 else 0          # wiggle reversals
    f[5] = v.std(1).mean()                                                          # asynchronous fingers
    wrist = xy[:, 0] / palm_m
    d = wrist[-1] - wrist[0]
    f[6], f[7] = d
    path = np.linalg.norm(np.diff(wrist, axis=0), axis=1).sum()
    f[8] = path; f[9] = np.linalg.norm(d) / max(path, 1e-3)                        # straightness
    f[10] = np.mean([np.linalg.norm(reln[:, a] - reln[:, b], axis=1).mean() for a, b in zip(TIPS, TIPS[1:])])
    direction = reln[:, 12] - reln[:, 9]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    f[11], f[12] = direction[:, 0].mean(), direction[:, 1].mean()
    f[13] = np.linalg.norm(reln[:, 4] - reln[:, 5], axis=1).mean()                  # thumb out
    body = raw[:, 57:59]
    ok = (body[:, :, 4] > 0).all(1)
    if ok.any():
        width = np.linalg.norm(body[ok, 1, :2] - body[ok, 0, :2], axis=1).mean()
        centre = body[ok, :, :2].mean(1).mean(0)
        if width > 1e-3:
            f[14] = (centre[1] - xy[:, 0, 1].mean()) / width
            f[15] = np.abs(xy[:, 0, 0].mean() - centre[0]) / width
            f[16] = palm_m / width
            f[17] = 1
    return np.clip(np.nan_to_num(f), -20, 20)


def window_row(window):
    """One [N_HAND + 4] row: the more present/moving hand, then a summary of the other hand."""
    a, b = hand_features(window, 0), hand_features(window, 21)
    if (b[0], b[8]) > (a[0], a[8]):
        a, b = b, a
    return np.concatenate([a, b[[0, 1, 3, 8]]])


def window_features(raw):
    """[n_windows, N_HAND + 4] for windows starting at 0, STEP, 2*STEP, ..."""
    raw = np.asarray(raw, np.float32)
    if len(raw) < WINDOW:
        raw = np.concatenate([raw, np.zeros((WINDOW - len(raw),) + raw.shape[1:], np.float32)])
    return np.stack([window_row(raw[s:s + WINDOW]) for s in range(0, len(raw) - WINDOW + 1, STEP)])


def export(model, threshold, path=MODEL, **meta):
    """sklearn HistGradientBoostingClassifier (binary, numeric features) -> portable JSON trees."""
    trees = []
    for (predictor,) in model._predictors:
        n = predictor.nodes
        trees.append(dict(feature=n['feature_idx'].tolist(), threshold=n['num_threshold'].tolist(),
                          left=n['left'].tolist(), right=n['right'].tolist(),
                          leaf=n['is_leaf'].astype(int).tolist(), value=n['value'].tolist()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(format='hgb_binary_v1', window=WINDOW, step=STEP, smooth=3,
                                    features=N_HAND + 4, baseline=float(np.ravel(model._baseline_prediction)[0]),
                                    threshold=float(threshold), trees=trees, **meta)))


class TriggerModel:
    def __init__(self, path=MODEL):
        d = json.loads(Path(path).read_text())
        self.baseline, self.threshold, self.trees = d['baseline'], d['threshold'], d['trees']
        self.smooth = d['smooth']

    def probability(self, row):
        raw = self.baseline
        for t in self.trees:
            i = 0
            while not t['leaf'][i]:
                i = t['left'][i] if row[t['feature'][i]] <= t['threshold'][i] else t['right'][i]
            raw += t['value'][i]
        return 1 / (1 + math.exp(-raw))


class FingerspellTrigger:
    """Streaming detector: push one raw frame at a time; returns (start, end) seconds when it fires.

    Score = mean of the last 3 window probabilities (window 20 frames, every 2 frames). After a firing
    the score must fall below threshold and 1.5 s must pass before it can fire again, so one sign
    never switches spelling on and straight back off.
    """
    refractory = 1.5

    def __init__(self, model: TriggerModel | None = None):
        self.model = model or TriggerModel()
        self.reset()

    def reset(self):
        self.frames, self.times, self.scores = [], [], []
        self.count, self.last_fire, self.armed, self.above_since = 0, -math.inf, True, None
        self.score = 0.0

    def push(self, raw_frame, seconds):
        self.frames.append(np.asarray(raw_frame, np.float32)); self.times.append(float(seconds))
        self.frames, self.times = self.frames[-WINDOW:], self.times[-WINDOW:]
        self.count += 1
        if self.count < WINDOW or (self.count - WINDOW) % STEP:
            return None
        self.scores = (self.scores + [self.model.probability(window_row(np.stack(self.frames)))])[-self.model.smooth:]
        self.score = sum(self.scores) / len(self.scores)
        above = self.score > self.model.threshold
        if not above:
            self.above_since = None
            if seconds - self.last_fire >= self.refractory:
                self.armed = True
            return None
        if self.above_since is None:
            self.above_since = self.times[0]
        if self.armed and seconds - self.last_fire >= self.refractory:
            self.armed, self.last_fire = False, seconds
            return (self.above_since, seconds)
        return None
