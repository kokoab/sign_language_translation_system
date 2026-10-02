"""Train the continuous fingerspelling reader (active/v17/letter_ctc_v17.py) on FSboard.

Data: data/local/fsboard_v17 batch1 train (117 signers) + pilot (train split); selection by greedy
letter CER on batch1 validation (15 other signers). The FSboard test split and the ASLLRP final-test
utterances are never read here. Clips whose annotated span is shorter than CTC needs are dropped.

Generalization measures: signer-disjoint validation, speed augmentation (FSboard spells 2-4 letters/s,
natural in-sentence spelling ~9), rotation/scale/position jitter (camera framing), landmark noise,
frame repeats (uneven phone frame rate), hand-embedding dropout (appearance: skin, light, background).
"""
from __future__ import annotations

import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.8')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.6')

import argparse
import glob
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.letter_ctc_v17 import (CLASSES, FORMAT, ConvFormerCTC, LetterCTC, add_deltas, base_features, cer_counts,
                                       ctc_min_frames, greedy, letters_only, mirror, targets)

FS = ROOT / 'data/local/fsboard_v17'
FSW = ROOT / 'data/local/chicago_fswild/features'


def load(files):
    items = []
    for f in files:
        z = np.load(f)
        base, pres = base_features(z['raw'])
        ids = targets(z['phrase'])
        t = z['times']; a, b = z['annotation']
        inside = np.where((t >= a) & (t <= b))[0]
        if not ids or len(inside) < ctc_min_frames(ids):
            continue
        items.append(dict(file=f, base=base, pres=pres, emb=z['hand_embeddings'], valid=z['hand_valid'],
                          ids=ids, phrase=str(z['phrase']), signer=str(z['signer']),
                          first=int(inside[0]), last=int(inside[-1])))
    return items


def resample(x, src, linear):
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, len(x) - 1); w = (src - lo)[:, None]
    if not linear:
        return x[np.rint(src).astype(int)]
    return (x[lo] * (1 - w) + x[hi] * w).astype(x.dtype)


def augment(it, rng, a):
    base, pres, emb, valid = it['base'].copy(), it['pres'], it['emb'], it['valid'].copy()
    first, last = it['first'], it['last']
    s = rng.integers(0, min(first, 8) + 1); e = len(base) - rng.integers(0, min(len(base) - 1 - last, 8) + 1)
    base, pres, emb, valid = base[s:e], pres[s:e], emb[s:e], valid[s:e]
    T = len(base)
    if rng.random() < a.hflip:                                    # Sohn: mirror (left-handed signers)
        base, pres = mirror(base, pres)
        emb, valid = emb[:, [1, 0, 2]], valid[:, [1, 0, 2]]
    # Camera framing: rotate/scale/shear hand shapes, shift/scale positions.
    ang = math.radians(rng.uniform(-a.rotate, a.rotate)); sc = rng.uniform(.9, 1.1, 2)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) * sc
    if a.shear:
        R = R @ np.array([[1, rng.uniform(-a.shear, a.shear)], [rng.uniform(-a.shear, a.shear), 1]])
    for k in range(2):
        o = k * 44
        rel = base[:, o:o + 40].reshape(T, 20, 2) @ R.T
        rel += rng.normal(0, a.noise, rel.shape)
        if rng.random() < a.finger_drop:                         # Kaggle 1st: finger dropout
            f = rng.integers(0, 5); rel[:, f * 4:(f + 1) * 4] = 0
        base[:, o:o + 40] = rel.reshape(T, 40)
        base[:, o + 40:o + 42] = base[:, o + 40:o + 42] * rng.uniform(.85, 1.15) + rng.uniform(-.2, .2, 2)
        base[:, o + 42] *= rng.uniform(.85, 1.15)
        base[~pres[:, k], o:o + 44] = 0
    # Signing speed.
    if rng.random() < a.speed_p:
        f = math.exp(rng.uniform(math.log(a.speed_min), math.log(a.speed_max)))
        n = max(ctc_min_frames(it['ids']) + 2, int(round(T / f)))
        if n < T or f < 1:
            src = np.linspace(0, T - 1, n)
            base = resample(base, src, True)
            pnear = resample(pres, src, False)
            base[~pnear[:, 0], 0:44] = 0; base[~pnear[:, 1], 44:88] = 0
            pres, emb, valid = pnear, resample(emb, src, False), resample(valid, src, False)
            T = n
    # Uneven frame rate: repeat the previous frame.
    if rng.random() < .5:
        rep = np.where(rng.random(T) < .05)[0]
        rep = rep[rep > 0]
        idx = np.arange(T); idx[rep] = rep - 1
        base, pres, emb, valid = base[idx], pres[idx], emb[idx], valid[idx]
    # Sohn: random temporal masking (short spans).
    if rng.random() < a.time_mask:
        for _ in range(rng.integers(1, 3)):
            w = int(rng.integers(1, 4)); st = int(rng.integers(0, max(1, T - w)))
            base[st:st + w] = 0
    # Appearance: drop embeddings.
    valid = valid & (rng.random(valid.shape) >= a.emb_drop)
    if rng.random() < .1:
        valid[:] = False
    return add_deltas(base, pres, lag2=a.lag2), emb, valid


def plain(it, lag2=False):
    return add_deltas(it['base'], it['pres'], lag2=lag2), it['emb'], it['valid']


def collate(rows, device, bucket=32):
    T = max(len(r[0]) for r in rows); B = len(rows)
    T = -(-T // bucket) * bucket            # few distinct lengths: MPS caches one attention graph per shape
    lm = np.zeros((B, T, rows[0][0].shape[1]), np.float32)
    emb = np.zeros((B, T, 3, 512), np.float16); valid = np.zeros((B, T, 3), bool); mask = np.zeros((B, T), bool)
    for i, (l, e, v) in enumerate(rows):
        n = len(l); lm[i, :n] = l; emb[i, :n] = e; valid[i, :n] = v; mask[i, :n] = True
    t = lambda x, dt=None: torch.from_numpy(x).to(device, dt)
    return t(lm), t(emb, torch.float32), t(valid), t(mask)


@torch.no_grad()
def evaluate(model, items, device, batch=48):
    model.eval()
    edits = total = 0
    order = sorted(range(len(items)), key=lambda i: len(items[i]['base']))
    for k in range(0, len(order), batch):
        chunk = [items[i] for i in order[k:k + batch]]
        lm, emb, valid, mask = collate([plain(it, model.config.get('lag2', False)) for it in chunk], device)
        lp = model(lm, emb, valid, mask).float().cpu()
        for i, it in enumerate(chunk):
            hyp, _ = greedy(lp[i, :len(it['base'])])
            e, n = cer_counts(letters_only(hyp), letters_only(it['phrase']))
            edits += e; total += n
    model.train()
    return edits / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fraction', type=float, default=1.)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--kernel', type=int, default=7)
    ap.add_argument('--dropout', type=float, default=.15)
    ap.add_argument('--no-embeddings', action='store_true')
    ap.add_argument('--speed-p', type=float, default=.8)
    ap.add_argument('--speed-min', type=float, default=.8)
    ap.add_argument('--speed-max', type=float, default=2.6)
    ap.add_argument('--rotate', type=float, default=12.)
    ap.add_argument('--noise', type=float, default=.03)
    ap.add_argument('--emb-drop', type=float, default=.15)
    ap.add_argument('--arch', choices=('conv', 'convformer'), default='conv')
    ap.add_argument('--hflip', type=float, default=0.)
    ap.add_argument('--shear', type=float, default=0.)
    ap.add_argument('--finger-drop', type=float, default=0.)
    ap.add_argument('--time-mask', type=float, default=0.)
    ap.add_argument('--drop-path', type=float, default=.2)
    ap.add_argument('--fswild', action='store_true', help='add ChicagoFSWild train; select on its dev signers')
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_files = sorted(glob.glob(str(FS / 'batch1/features/train/*.npz'))) + sorted(glob.glob(str(FS / 'pilot/cmp_ffmpeg/*.npz')))
    train = load(train_files)
    val = load(sorted(glob.glob(str(FS / 'batch1/features/validation/*.npz'))))
    dev = []
    if a.fswild:
        wild = load(sorted(glob.glob(str(FSW / 'train/*.npz'))))
        dev = load(sorted(glob.glob(str(FSW / 'dev/*.npz'))))
        print(f'ChicagoFSWild train {len(wild)} / dev {len(dev)} clips', flush=True)
    if a.fraction < 1:                                       # keep every signer, fewer clips each
        by = {}
        for it in train:
            by.setdefault(it['signer'], []).append(it)
        sub = []
        for s in sorted(by):
            g = sorted(by[s], key=lambda it: it['file']); random.Random(a.seed).shuffle(g)
            sub += g[:max(1, math.ceil(a.fraction * len(g)))]
        train = sub
    if a.fswild:
        train = train + wild
    assert not {it['signer'] for it in train} & {it['signer'] for it in dev}, 'signer overlap (FSWild dev)'
    assert not {it['signer'] for it in train} & {it['signer'] for it in val}, 'signer overlap'
    print(f'train {len(train)} clips / {len({it["signer"] for it in train})} signers, val {len(val)} / '
          f'{len({it["signer"] for it in val})}; loaded in {time.time() - t0:.0f}s', flush=True)
    if a.arch == 'convformer':
        model = ConvFormerCTC(d=a.d, dropout=a.dropout, drop_path=a.drop_path).to(a.device)
    else:
        model = LetterCTC(a.d, a.layers, a.kernel, a.dropout, not a.no_embeddings).to(a.device)
    a.lag2 = model.config.get('lag2', False)
    print('params', sum(p.numel() for p in model.parameters()), 'lookahead frames', model.lookahead_frames, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.05)
    steps = a.epochs * math.ceil(len(train) / a.batch); warm = 3 * math.ceil(len(train) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * .5 * (1 + math.cos(math.pi * min(1, s / steps))))
    best, history = 9., []
    for epoch in range(a.epochs):
        order = sorted(range(len(train)), key=lambda i: len(train[i]['base']) + rng.integers(0, 40))
        batches = [order[k:k + a.batch] for k in range(0, len(order), a.batch)]
        rng.shuffle(batches)
        losses = []
        for bi in batches:
            rows = [augment(train[i], rng, a) for i in bi]
            lm, emb, valid, mask = collate(rows, a.device)
            lp = model(lm, emb, valid, mask).float().cpu()                 # CTC is unsupported on MPS
            tg = [torch.tensor(train[i]['ids']) for i in bi]
            loss = F.ctc_loss(lp.transpose(0, 1), torch.cat(tg), mask.sum(1).cpu(), torch.tensor([len(x) for x in tg]),
                              blank=0, zero_infinity=True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step()
            losses.append(float(loss.detach()))
            if a.device == 'mps' and len(losses) % 100 == 0:
                torch.mps.empty_cache()
        row = dict(epoch=epoch, loss=float(np.mean(losses)), minutes=(time.time() - t0) / 60)
        if a.device == 'mps':
            row['mps_gb'] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        if epoch % 2 == 1 or epoch == a.epochs - 1:
            row['val_cer'] = evaluate(model, val, a.device)
            if dev:
                row['fswild_dev_cer'] = evaluate(model, dev, a.device)
            score = row['fswild_dev_cer'] if dev else row['val_cer']
            if score < best:
                best = score
                torch.save(dict(format=FORMAT, config=model.config, state_dict=model.state_dict(), epoch=epoch,
                                val_cer=row['val_cer'], fswild_dev_cer=row.get('fswild_dev_cer'), args=vars(a) | dict(out=str(a.out)), train_clips=len(train),
                                train_signers=sorted({it['signer'] for it in train})), a.out / 'best.pt')
        history.append(row)
        print(json.dumps(row), flush=True)
    (a.out / 'history.json').write_text(json.dumps(dict(best_val_cer=best, history=history), indent=1))
    print('best val CER', round(best, 4), flush=True)


if __name__ == '__main__':
    main()
