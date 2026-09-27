"""Apple Vision boundary student: DGS-quality sign BIO from the live Vision features.

The pinned DGS pose segmenter needs MediaPipe Holistic (~5-7x slower than Apple Vision per
frame here), so it is used only as an offline teacher. This student reads the same
`boundary_features` the live Reel already computes, over a bounded window with an explicit
lookahead, and predicts per-frame O/B/I at the read position.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from active.v17.temporal_boundary_v17 import boundary_features

FORMAT = 'slt_av_boundary_student_v17'
FPS = 20
FRAMES = 64
CLASSES = ('O', 'B', 'I')


class AVBoundary(nn.Module):
    def __init__(self, input_dim=450, hidden=256, layers=4, heads=8, lookahead=6, dropout=.1):
        super().__init__()
        if not 0 <= lookahead < FRAMES // 2:
            raise ValueError('invalid lookahead')
        self.config = dict(input_dim=input_dim, hidden=hidden, layers=layers, heads=heads,
                           lookahead=lookahead, dropout=dropout)
        self.lookahead = lookahead
        self.read = FRAMES - 1 - lookahead
        self.inp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.pos = nn.Parameter(torch.zeros(FRAMES, hidden))
        nn.init.normal_(self.pos, std=.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=dropout, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, len(CLASSES))

    def forward(self, x, valid):
        """x [B,FRAMES,D], valid [B,FRAMES] bool -> logits [B,3] at the read position."""
        h = self.inp(x) + self.pos
        h = self.encoder(h, src_key_padding_mask=~valid)
        return self.head(self.norm(h[:, self.read]))


def windows(features, lookahead):
    """All read-position windows for a whole clip: x [n,FRAMES,D], valid [n,FRAMES]."""
    n, d = features.shape
    read = FRAMES - 1 - lookahead
    idx = np.arange(n)[:, None] + np.arange(FRAMES)[None] - read
    valid = (idx >= 0) & (idx < n)
    x = features[np.clip(idx, 0, n - 1)] * valid[..., None]
    return x.astype(np.float32), valid


def clip_bio(model, raw, times, device='cpu', batch=256):
    """[n,4] log-probs in the DGS order (UNK,O,B,I) for offline decoding; UNK gets ~0 mass."""
    features = boundary_features(raw, times, True)
    x, valid = windows(features, model.lookahead)
    out = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            logits = model(torch.from_numpy(x[i:i + batch]).to(device), torch.from_numpy(valid[i:i + batch]).to(device))
            out.append(torch.log_softmax(logits.float(), -1).cpu().numpy())
    obi = np.concatenate(out)
    return np.concatenate([np.full((len(obi), 1), np.log(1e-6)), obi], axis=1).astype(np.float32)


class AVBoundaryStream:
    """Live per-frame BIO with the trained lookahead; bounded history, no MediaPipe."""

    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        self.reset()

    def reset(self):
        self.raw, self.times = deque(), deque()
        self.features = deque(maxlen=FRAMES)
        self.last = -np.inf

    @torch.inference_mode()
    def update(self, raw_frame, seconds):
        """Returns (frame_seconds, log-probs[4]) for the frame `lookahead` steps back, or None."""
        if seconds - self.last > .26:
            self.reset()
        self.last = float(seconds)
        self.raw.append(np.asarray(raw_frame, np.float32))
        self.times.append(float(seconds))
        while len(self.times) > 2 and self.times[1] < seconds - 1.2:
            self.raw.popleft(); self.times.popleft()
        self.features.append(boundary_features(np.asarray(self.raw), self.times, True)[-1])
        if len(self.features) <= self.model.lookahead:
            return None
        f = np.asarray(self.features)
        pad = FRAMES - len(f)
        x = np.concatenate([np.zeros((pad, f.shape[1]), np.float32), f]) if pad > 0 else f
        valid = np.arange(FRAMES) >= max(pad, 0)
        logits = self.model(torch.from_numpy(x[None]).to(self.device), torch.from_numpy(valid[None]).to(self.device))
        obi = torch.log_softmax(logits[0].float(), -1).cpu().numpy()
        return self.times[-1 - self.model.lookahead] if len(self.times) > self.model.lookahead else None, \
            np.concatenate([[np.log(1e-6)], obi])

    @torch.inference_mode()
    def flush(self):
        """Estimates for the last `lookahead` frames with the unseen future masked (as in training
        windows at a clip end). Returns a list of log-prob rows in frame order."""
        if not self.features:
            return []
        f = np.asarray(self.features)
        n, look = len(f), self.model.lookahead
        read = FRAMES - 1 - look
        out = []
        for k in range(look, 0, -1):
            target = n - k                    # index in f of the pending frame
            if target < 0:
                continue
            idx = np.arange(FRAMES) + target - read
            valid = (idx >= 0) & (idx < n)
            x = f[np.clip(idx, 0, n - 1)] * valid[:, None]
            logits = self.model(torch.from_numpy(x[None].astype(np.float32)).to(self.device),
                                torch.from_numpy(valid[None]).to(self.device))
            obi = torch.log_softmax(logits[0].float(), -1).cpu().numpy()
            out.append(np.concatenate([[np.log(1e-6)], obi]))
        return out

