"""Fist letters (A E M N S T) differ by thumb placement, which the letter head reads poorly
(validation 77%). A hand-geometry classifier on the span landmarks separates them far better (93%).

FistGeometry.refine(letter_logits, landmarks): when the head's top letter is a fist letter, the
fist-group probability is re-shared as head^w x geometry (renormalised to the group's original mass);
every other class keeps its probability. w = 0.25 (a confident head cannot outvote clear geometry).
Validation (other signers): 1101 -> 1153 of 1271 letters, fist 230 -> 282 of 299
(w 1: 1144, .5: 1148, 0: 1142); no letter outside the group changed (reports/fist_letters_v17_20260929).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'artifacts/models/fist_geometry_v17/model.json'
FIST = list('AEMNST')
LETTERS = [chr(65 + i) for i in range(26)]


def features(landmarks):
    """Span landmarks [32,61,5] -> 54 values, or None: dominant hand, frames 8..23, origin wrist,
    y along wrist -> middle MCP, unit palm, mirrored so the index MCP is on +x; 19 joints (x, y)
    without the constant middle MCP,
    plus thumb-tip distances to the 16 finger joints; median over frames."""
    lm = np.asarray(landmarks, np.float32)
    s = 0 if lm[:, :21, 3].sum() >= lm[:, 21:42, 3].sum() else 21
    rows = []
    for f in lm[8:24]:
        h = f[s:s + 21]
        if (h[:, 3] > .5).sum() < 18:
            continue
        xy = h[:, :2] - h[0, :2]
        up = xy[9]
        palm = float(np.linalg.norm(up))
        if palm < 1e-4:
            continue
        ey = up / palm
        ex = np.array([ey[1], -ey[0]], np.float32)
        c = np.stack([xy @ ex, xy @ ey], 1) / palm
        if c[5, 0] < c[17, 0]:
            c[:, 0] *= -1
        dist = np.linalg.norm(c[5:] - c[4], axis=1)
        # The middle MCP (joint 9) is (0, 1) by construction: constant, so it is left out (with a
        # near-zero standardiser scale it would amplify rounding noise into large logit swings).
        joints = np.concatenate([c[1:9], c[10:]])
        rows.append(np.concatenate([joints.reshape(-1), dist]))
    return np.median(np.stack(rows), 0) if len(rows) >= 3 else None


def load_split(split, root=ROOT / 'data/local/fingerspelling_letters_v17'):
    corrections = {}
    path = root / 'label_corrections.json'
    if path.exists():
        corrections = {k: v['now'] for k, v in json.loads(path.read_text())['corrections'].items()}
    X, y = [], []
    for letter in FIST:
        for p in sorted(glob.glob(str(root / 'spans' / split / letter / '*.npz'))):
            key = str(Path(p).relative_to(root / 'spans'))
            if corrections.get(key, letter) != letter:
                continue
            d = np.load(p)
            if 'landmarks' not in d:
                continue
            f = features(d['landmarks'].astype(np.float32))
            if f is not None:
                X.append(f); y.append(letter)
    return np.array(X), np.array(y)


class FistGeometry:
    def __init__(self, path=MODEL):
        m = json.loads(Path(path).read_text())
        self.classes = m['classes']
        self.mean, self.scale = np.array(m['mean']), np.array(m['scale'])
        self.coef, self.intercept = np.array(m['coef']), np.array(m['intercept'])
        self.index = [LETTERS.index(c) for c in self.classes]

    def probabilities(self, f):
        z = self.coef @ ((f - self.mean) / self.scale) + self.intercept
        z = np.exp(z - z.max())
        return z / z.sum()

    def refine(self, letter_logits, landmarks, head_weight=0.25):
        """letter_logits: 27 (A..Z, NONE). Returns log-probabilities with the fist group re-shared."""
        l = np.asarray(letter_logits, np.float64)
        p = np.exp(l - l.max()); p /= p.sum()
        if int(p.argmax()) not in self.index:
            return letter_logits
        f = features(landmarks)
        if f is None:
            return letter_logits
        mass = p[self.index].sum()
        new = p[self.index] ** head_weight * self.probabilities(f)
        if new.sum() <= 0:
            return letter_logits
        p[self.index] = new / new.sum() * mass
        return np.log(np.clip(p, 1e-12, 1)).astype(np.float32)
