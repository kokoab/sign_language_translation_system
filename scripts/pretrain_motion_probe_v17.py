"""Does YouTube-ASL landmark motion carry learnable temporal structure?

Gate experiment for encoder pretraining. Trains a STANDALONE temporal encoder with a
masked-frame reconstruction objective and compares it against two trivial baselines.
Touches no existing weights: the frozen landmark/hand encoders, Stage 1/2 heads, Reel and
every CoreML export are untouched. Nothing is promoted.

Pass condition: masked reconstruction must beat BOTH trivial baselines by a clear margin.
  copy_previous  - repeat the last observed frame across the mask
  linear_interp  - linearly interpolate across the mask from its edges
If the model only matches these, the corpus offers nothing beyond smoothness and encoder
pretraining on it is not justified.

Held out BY SOURCE VIDEO so no clip leaks between train and validation.
"""
from __future__ import annotations
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/reports/motion_pretrain_probe_v17_20260922'
PATTERN = 'data/local/youtube_asl_transition_landmarks_v17/**/*.npz'
POINTS, CHANNELS, FRAMES = 61, 2, 32
HAND_POINTS = 42          # 21 left + 21 right; the channels that matter for signing
MASK_SPAN = int(__import__('os').environ.get('MASK_SPAN', 6))


def load_windows(limit_files=None, seed=17621):
    """Returns (windows, presence, group) with group = source video id."""
    files = sorted(glob.glob(str(ROOT / PATTERN), recursive=True))
    if limit_files:
        files = files[:limit_files]
    xs, ms, gs = [], [], []
    for path in files:
        try:
            z = np.load(path, allow_pickle=True)
            a = np.asarray(z['landmarks'], np.float32)
            valid = np.asarray(z['window_valid']).astype(bool)
            meta = json.loads(str(z['metadata_json'].item()))
        except Exception:
            continue
        if a.ndim != 4 or a.shape[1:] != (FRAMES, POINTS, 5):
            continue
        group = str(meta.get('source_item_id') or Path(path).stem)
        for window, ok in zip(a, valid):
            if not ok:
                continue
            xy = window[:, :HAND_POINTS, :CHANNELS]
            presence = window[:, :HAND_POINTS, 3] > 0
            if presence.mean() < 0.5 or not np.isfinite(xy).all():
                continue
            xs.append(xy)
            ms.append(presence)
            gs.append(group)
    if not xs:
        raise SystemExit('no usable windows found')
    x = np.stack(xs)
    m = np.stack(ms)
    groups = np.array(gs)
    # Per-window centring and scaling keeps the objective about MOTION, not position.
    flat = x.reshape(len(x), -1, CHANNELS)
    centre = flat.mean(axis=1, keepdims=True)
    x = x - centre[:, None]
    scale = np.sqrt((x ** 2).mean(axis=(1, 2, 3), keepdims=True)) + 1e-6
    x = x / scale
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    holdout = set(unique[:max(1, len(unique) // 5)].tolist())
    is_val = np.array([g in holdout for g in groups])
    return x, m, groups, is_val


class TemporalEncoder(nn.Module):
    """Small causal-agnostic transformer over frames; standalone, not the v17 encoder."""

    def __init__(self, width=192, depth=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(HAND_POINTS * CHANNELS, width)
        self.pos = nn.Parameter(torch.zeros(1, FRAMES, width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(width, HAND_POINTS * CHANNELS)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, masked):
        b = x.shape[0]
        h = self.inp(x.reshape(b, FRAMES, -1))
        h = torch.where(masked[..., None], self.mask_token.expand(b, FRAMES, -1), h)
        h = self.body(h + self.pos)
        return self.out(h).reshape(b, FRAMES, HAND_POINTS, CHANNELS)


def make_masks(n, rng):
    masked = np.zeros((n, FRAMES), bool)
    starts = rng.integers(2, FRAMES - MASK_SPAN - 2, size=n)
    for i, s in enumerate(starts):
        masked[i, s:s + MASK_SPAN] = True
    return masked


def baselines(x, masked):
    """copy_previous and linear_interp reconstructions over the masked span."""
    copy = x.copy()
    interp = x.copy()
    for i in range(len(x)):
        idx = np.flatnonzero(masked[i])
        lo, hi = idx[0] - 1, idx[-1] + 1
        copy[i, idx] = x[i, lo]
        w = (np.arange(len(idx)) + 1) / (len(idx) + 1)
        interp[i, idx] = ((1 - w)[:, None, None] * x[i, lo][None]
                          + w[:, None, None] * x[i, hi][None])
    return copy, interp


def masked_mse(pred, truth, masked, presence):
    sel = masked[..., None] & presence
    if not sel.any():
        return float('nan')
    d = ((pred - truth) ** 2).sum(-1)
    return float(d[sel].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--limit-files', type=int, default=None)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)

    x, presence, groups, is_val = load_windows(args.limit_files)
    rng = np.random.default_rng(17621)
    torch.manual_seed(17621)
    tr, va = ~is_val, is_val
    print('windows %d (train %d / val %d) from %d source videos; val videos %d'
          % (len(x), tr.sum(), va.sum(), len(np.unique(groups)), len(np.unique(groups[va]))), flush=True)

    val_masked = make_masks(int(va.sum()), rng)
    vx, vp = x[va], presence[va]
    copy, interp = baselines(vx, val_masked)
    base = dict(copy_previous=masked_mse(copy, vx, val_masked, vp),
                linear_interp=masked_mse(interp, vx, val_masked, vp))
    print('baselines: copy_previous %.5f | linear_interp %.5f' % (base['copy_previous'], base['linear_interp']), flush=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = TemporalEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    xt = torch.from_numpy(x[tr]).to(device)
    pt = torch.from_numpy(presence[tr]).to(device)
    vxt = torch.from_numpy(vx).to(device)
    vmt = torch.from_numpy(val_masked).to(device)
    vpt = torch.from_numpy(vp).to(device)

    history, best = [], float('inf')
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(xt))
        total = count = 0.
        for s in range(0, len(order), args.batch):
            idx = torch.from_numpy(order[s:s + args.batch]).to(device)
            xb, pb = xt[idx], pt[idx]
            mb = torch.from_numpy(make_masks(len(idx), rng)).to(device)
            pred = model(torch.where(mb[..., None, None], torch.zeros_like(xb), xb), mb)
            sel = mb[..., None] & pb
            loss = (((pred - xb) ** 2).sum(-1)[sel]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        model.eval()
        with torch.inference_mode():
            pred = model(torch.where(vmt[..., None, None], torch.zeros_like(vxt), vxt), vmt)
            sel = vmt[..., None] & vpt
            val = float((((pred - vxt) ** 2).sum(-1)[sel]).mean())
        history.append(dict(epoch=epoch, train=total / count, validation=val))
        best = min(best, val)
        if epoch % 10 == 0 or epoch == 1:
            print('epoch %3d  train %.5f  val %.5f  (best %.5f)  %.0fs'
                  % (epoch, total / count, val, best, time.perf_counter() - started), flush=True)

    verdict = ('PASS: beats both trivial baselines' if best < 0.8 * min(base.values())
               else 'FAIL: no clear gain over trivial reconstruction')
    print('\nbest val masked MSE %.5f vs copy_previous %.5f, linear_interp %.5f'
          % (best, base['copy_previous'], base['linear_interp']))
    print('improvement over best baseline: %.1f%%' % (100 * (1 - best / min(base.values()))))
    print(verdict)

    (REPORT / 'probe.json').write_text(json.dumps(dict(
        scope='standalone masked-frame reconstruction on YouTube-ASL transition landmarks; '
              'gate experiment for encoder pretraining. No existing weights touched, nothing promoted.',
        data=dict(windows=int(len(x)), train=int(tr.sum()), validation=int(va.sum()),
                  source_videos=int(len(np.unique(groups))), held_out_by='source video id'),
        mask=dict(span_frames=MASK_SPAN, frames=FRAMES, points=HAND_POINTS),
        baselines=base, best_validation=best, verdict=verdict, history=history), indent=1) + '\n')
    print('wrote', REPORT / 'probe.json')


if __name__ == '__main__':
    main()
