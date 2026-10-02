"""Continuous fingerspelling reader: per-frame CTC over letters on the live 20 Hz Apple Vision inputs.

Inputs per frame are what the phone already computes: raw Apple Vision landmarks [61, 5] and the frozen
hand-crop embeddings [3, 512] (left, right, both-hands union) with their valid flags. Landmarks are made
camera-invariant: each hand's joints relative to its wrist in palm lengths, wrist position relative to
the shoulder midpoint in shoulder widths (per-clip medians), hand size in shoulder widths, and frame
deltas. The model is a stack of symmetric depthwise-convolution blocks, so its look-ahead is fixed
(layers * (kernel // 2) frames) and it can run streaming with that latency.

Classes: 0 blank, 1-26 A-Z, 27 '#' (any signed non-letter character: digit or punctuation). Spaces are not
targets: FSboard signers mostly do not mark them.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
from torch import nn

FORMAT = 'letter_ctc_v17_a'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
OTHER = '#'
CLASSES = ['<blank>'] + list(LETTERS) + [OTHER]
FPS = 20.
HANDS = (0, 21)
L_SHOULDER, R_SHOULDER = 57, 58
BASE_DIM = 2 * (40 + 2 + 1 + 1)                    # per hand: rel joints, wrist pos, size, present
LM_DIM = BASE_DIM + 2 * (40 + 2)                    # + deltas of rel joints and wrist pos
LM_DIM_LAG2 = LM_DIM + 2 * (40 + 2)                 # + lag-2 deltas (Sohn: lag 1 and 2 motion)


def targets(phrase):
    """Phrase -> class ids: letters, '#' for other signed characters, spaces dropped."""
    out = []
    for ch in str(phrase).upper():
        if ch in LETTERS:
            out.append(1 + LETTERS.index(ch))
        elif not ch.isspace():
            out.append(len(CLASSES) - 1)
    return out


def ctc_min_frames(ids):
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def body_reference(raw):
    """Per-clip shoulder midpoint and width (medians); fallbacks keep scale sane without a body."""
    sh = raw[:, [L_SHOULDER, R_SHOULDER]]
    ok = (sh[:, :, 4] > 0).all(1)
    if ok.sum() >= 1:
        centre = np.median(sh[ok, :, :2].mean(1), 0)
        width = float(np.median(np.linalg.norm(sh[ok, 0, :2] - sh[ok, 1, :2], axis=1)))
        if width > 1e-3:
            return centre, width
    face = raw[:, 42:57]
    fp = face[:, :, 4] > 0
    hands = [raw[:, s:s + 21] for s in HANDS]
    palms = [np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=1)[(h[:, :, 4] > 0).sum(1) >= 15] for h in hands]
    palm = float(np.median(np.concatenate(palms))) if sum(len(p) for p in palms) else .1
    palm = max(palm, .02)                                   # collapsed hands must not zero the scale
    if fp.any():
        return face[fp][:, :2].mean(0) + np.array([0., 2.5 * palm]), 3.5 * palm
    return np.zeros(2), 3.5 * palm


def base_features(raw):
    """[T, BASE_DIM] float32 plus [T, 2] hand presence, from raw [T, 61, 5]."""
    raw = np.asarray(raw, np.float32)
    centre, width = body_reference(raw)
    T = len(raw)
    feats, present = [], []
    for s in HANDS:
        h = raw[:, s:s + 21]
        p = (h[:, :, 4] > 0).sum(1) >= 15
        xy = h[:, :, :2]
        palm = np.linalg.norm(xy[:, 9] - xy[:, 0], axis=1)
        med = float(np.median(palm[p])) if p.any() else 1.
        palm = np.where(p & (palm > .5 * med), palm, med)
        rel = (xy[:, 1:] - xy[:, :1]) / np.maximum(palm, 1e-4)[:, None, None]
        pos = (xy[:, 0] - centre) / width
        size = (palm / width)[:, None]
        f = np.concatenate([rel.reshape(T, 40), pos, size, p[:, None].astype(np.float32)], 1)
        f[~p] = 0
        feats.append(f); present.append(p)
    out = np.nan_to_num(np.concatenate(feats, 1), nan=0., posinf=0., neginf=0.)
    return np.clip(out, -8, 8).astype(np.float32), np.stack(present, 1)


def add_deltas(base, present, lag2=False):
    """Append frame deltas (lag 1, optionally lag 2) of each hand's relative joints and wrist position
    (zero across absences)."""
    parts = []
    for lag in ((1, 2) if lag2 else (1,)):
        for k in range(2):
            o = k * 44
            x = base[:, o:o + 42]
            d = np.zeros_like(x)
            d[lag:] = x[lag:] - x[:-lag]
            both = np.zeros(len(x), bool)
            both[lag:] = present[lag:, k] & present[:-lag, k]
            d[~both] = 0
            parts.append(d)
    return np.concatenate([base] + parts, 1).astype(np.float32)


def model_features(config, base, present):
    return add_deltas(base, present, lag2=config.get('lag2', False))


def mirror(base, present):
    """Horizontal mirror of base features: x of relative joints and wrist position negated, hands swapped."""
    b = base.copy()
    for k in range(2):
        o = k * 44
        b[:, o:o + 40:2] *= -1
        b[:, o + 40] *= -1
    return np.concatenate([b[:, 44:88], b[:, 0:44]], 1), present[:, ::-1].copy()


class Block(nn.Module):
    def __init__(self, d, kernel, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.pw = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.norm1(x).masked_fill(~mask[..., None], 0)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.pw(nn.functional.gelu(y)))
        return x + self.drop(self.ffn(self.norm2(x)))


class LetterCTC(nn.Module):
    def __init__(self, d=192, layers=6, kernel=7, dropout=.15, use_embeddings=True):
        super().__init__()
        self.config = dict(d=d, layers=layers, kernel=kernel, dropout=dropout, use_embeddings=use_embeddings)
        self.lm = nn.Sequential(nn.LayerNorm(LM_DIM), nn.Linear(LM_DIM, d))
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.emb = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d // 2))
            self.fuse = nn.Linear(d + 3 * (d // 2), d)
        self.blocks = nn.ModuleList(Block(d, kernel, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, len(CLASSES))

    @property
    def lookahead_frames(self):
        return self.config['layers'] * (self.config['kernel'] // 2)

    def forward(self, lm, emb, valid, mask):
        """lm [B,T,LM_DIM], emb [B,T,3,512], valid [B,T,3] bool, mask [B,T] bool -> log-probs [B,T,C]."""
        x = self.lm(lm)
        if self.use_embeddings:
            e = self.emb(emb) * valid[..., None]
            x = self.fuse(torch.cat([x, e.flatten(2)], -1))
        for b in self.blocks:
            x = b(x, mask)
        return self.out(self.norm(x)).log_softmax(-1)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class CausalConvBlock(nn.Module):
    """Sohn-style Conv1DBlock: expand -> causal depthwise conv (large kernel) -> project, drop path."""

    def __init__(self, d, kernel, expand, dropout, drop_path):
        super().__init__()
        self.kernel = kernel
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, d * expand)
        self.dw = nn.Conv1d(d * expand, d * expand, kernel, groups=d * expand)
        self.norm2 = nn.LayerNorm(d * expand)
        self.out = nn.Linear(d * expand, d)
        self.drop = nn.Dropout(dropout)
        self.dp = DropPath(drop_path)

    def forward(self, x, mask, attn_mask=None):
        y = nn.functional.silu(self.inp(self.norm(x))).masked_fill(~mask[..., None], 0)
        y = nn.functional.pad(y.transpose(1, 2), (self.kernel - 1, 0))
        y = self.dw(y).transpose(1, 2)
        y = self.out(nn.functional.silu(self.norm2(y)))
        return (x + self.dp(self.drop(y))).masked_fill(~mask[..., None], 0)


class LocalAttnBlock(nn.Module):
    """Transformer block whose attention sees `back` past and `ahead` future frames (streamable)."""

    def __init__(self, d, heads, expand, dropout, drop_path):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * expand), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d * expand, d))
        self.dp = DropPath(drop_path)

    def forward(self, x, mask, attn_mask=None):
        h = self.norm1(x)
        y = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.dp(y)
        x = x + self.dp(self.ffn(self.norm2(x)))
        return x.masked_fill(~mask[..., None], 0)


class ConvFormerCTC(nn.Module):
    """Hoyeol Sohn's 1D-CNN + Transformer encoder (Kaggle ISLR 1st / fingerspelling 2nd), made streamable:
    causal convolutions and windowed attention. Look-ahead = attention layers * ahead frames."""

    def __init__(self, d=192, groups=3, convs=3, kernel=17, expand=2, heads=4, back=64, ahead=6,
                 dropout=.1, drop_path=.2, head_dropout=.3, lag2=True):
        super().__init__()
        self.config = dict(arch='convformer', d=d, groups=groups, convs=convs, kernel=kernel, expand=expand,
                           heads=heads, back=back, ahead=ahead, dropout=dropout, drop_path=drop_path,
                           head_dropout=head_dropout, lag2=lag2)
        dim = LM_DIM_LAG2 if lag2 else LM_DIM
        self.stem = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
        blocks = []
        for _ in range(groups):
            blocks += [CausalConvBlock(d, kernel, expand, dropout, drop_path) for _ in range(convs)]
            blocks.append(LocalAttnBlock(d, heads, expand, dropout, drop_path))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)
        self.head_drop = nn.Dropout(head_dropout)
        self.out = nn.Linear(d, len(CLASSES))

    @property
    def lookahead_frames(self):
        return self.config['groups'] * self.config['ahead']

    def forward(self, lm, emb, valid, mask):
        B, T = mask.shape
        i = torch.arange(T, device=lm.device)
        rel = i[None, :] - i[:, None]                                       # key - query
        allowed = (rel >= -self.config['back']) & (rel <= self.config['ahead'])
        allowed = allowed[None] & mask[:, None, :]
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=lm.device)[None]   # no empty rows
        attn_mask = (~allowed).repeat_interleave(self.config['heads'], 0)
        x = self.stem(lm).masked_fill(~mask[..., None], 0)
        for b in self.blocks:
            x = b(x, mask, attn_mask)
        return self.out(self.head_drop(self.norm(x))).log_softmax(-1)


def greedy(log_probs):
    """[T, C] -> (string, [(class, first_frame, last_frame)]) with CTC collapse; '#' kept."""
    ids = log_probs.argmax(-1)
    out, prev = [], 0
    for t, c in enumerate(ids.tolist()):
        if c != prev and c != 0:
            out.append([c, t, t])
        elif c == prev and c != 0:
            out[-1][2] = t
        prev = c
    return ''.join(CLASSES[c] for c, _, _ in out), out


def letters_only(s):
    return re.sub(r'[^A-Z]', '', s.upper())


def cer_counts(hyp, ref):
    """Edit distance and reference length."""
    d = np.arange(len(ref) + 1)
    for i, h in enumerate(hyp, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (h != r))
    return int(d[-1]), len(ref)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT:
        raise ValueError('not a letter_ctc_v17 checkpoint')
    config = dict(payload['config'])
    model = ConvFormerCTC(**{k: v for k, v in config.items() if k != 'arch'}) if config.get('arch') == 'convformer' \
        else LetterCTC(**config)
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), payload
