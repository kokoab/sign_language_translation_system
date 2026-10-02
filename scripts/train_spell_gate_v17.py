"""Train the spelling-vs-signing gate (active/v17/spell_gate_v17.py).

Spelling (1): FSboard train clips inside their annotated span; ChicagoFSWild train sequences.
Not spelling (0): ASL Citizen train-signer clips (isolated signs with rest); FSboard clip margins
(hand raising/lowering) outside the span by >= 0.4 s; O5S5 sign glosses of the five train signers.
O5S5 'FS' glosses are spelling; unannotated O5S5 frames are ignored (transcript completeness unproven).
Training sequences concatenate 1-4 random clips so the gate sees spelling<->signing transitions; O5S5
contributes 8 s windows with real transitions.

Validation (selection, fixed before results): mean of FSWild dev spelling-frame recall, Citizen val-signer
frame rejection rate, and O5S5 LG balanced accuracy, all at probability 0.5. ASLLRP, FSWild test and the
official Citizen/FSboard test splits are never read here.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.letter_ctc_v17 import add_deltas, base_features
from active.v17.spell_gate_v17 import FORMAT, SpellGate

O5 = ROOT / 'data/local/open_asl_alternatives_20260913/o5s5'
O5_EVAL = {'LG'}


def feats(raw):
    base, pres = base_features(raw)
    return base, pres


def clip_items(files, label_fn):
    out = []
    for f in files:
        z = np.load(f)
        base, pres = feats(z['raw'])
        y = label_fn(z)
        if len(base) >= 4:
            out.append(dict(base=base, pres=pres, y=y.astype(np.float32), file=f))
    return out


def fsboard_labels(z):
    t = z['times']; a, b = z['annotation']
    y = np.full(len(t), -1.)
    y[(t >= a + .1) & (t <= b - .1)] = 1
    y[(t < a - .4) | (t > b + .4)] = 0
    return y


def all_label(v):
    return lambda z: np.full(len(z['times']), float(v))


def o5_labels(code, times):
    eaf = next(f for f in glob.glob(str(O5 / 'annotations/*.eaf')) if re.search(rf'_N_.*', f) and code_of(f) == code)
    root = ET.parse(eaf).getroot()
    slots = {t.get('TIME_SLOT_ID'): int(t.get('TIME_VALUE') or 0) / 1000 for t in root.iter('TIME_SLOT')}
    y = np.full(len(times), -1.)
    spans = []
    for tier in root.iter('TIER'):
        if 'Hand' not in tier.get('TIER_ID', ''):
            continue
        for a in tier.iter('ALIGNABLE_ANNOTATION'):
            v = (a.findtext('ANNOTATION_VALUE') or '').strip()
            s, e = slots[a.get('TIME_SLOT_REF1')], slots[a.get('TIME_SLOT_REF2')]
            spans.append((s, e, v == 'FS'))
    for s, e, fs in sorted(spans, key=lambda x: x[2]):           # FS written last: wins on overlap
        y[(times >= s) & (times <= e)] = 1. if fs else 0.
    return y


def code_of(path):
    m = re.search(r'O5S5_(\d{3})', Path(path).name)
    return {'002': 'LG', '007': 'JAH', '025': 'CK', '026': 'DR', '027': 'LR', '029': 'RD'}[m.group(1)]


def o5_items(eval_only):
    out = []
    for f in sorted(glob.glob(str(O5 / 'live_features/*.npz'))):
        code = Path(f).stem.split('_')[-1]
        if (code in O5_EVAL) != eval_only:
            continue
        z = np.load(f)
        base, pres = feats(z['raw'])
        out.append(dict(base=base, pres=pres, y=o5_labels(code, z['times']).astype(np.float32), file=f, code=code))
    return out


def augment(base, pres, rng, speed=True):
    base = base.copy(); T = len(base)
    ang = math.radians(rng.uniform(-12, 12)); sc = rng.uniform(.9, 1.1, 2)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) * sc
    for k in range(2):
        o = k * 44
        rel = base[:, o:o + 40].reshape(T, 20, 2) @ R.T + rng.normal(0, .03, (T, 20, 2))
        base[:, o:o + 40] = rel.reshape(T, 40)
        base[:, o + 40:o + 42] = base[:, o + 40:o + 42] * rng.uniform(.85, 1.15) + rng.uniform(-.2, .2, 2)
        base[~pres[:, k], o:o + 44] = 0
    return base


def resample(base, pres, y, f):
    T = len(base); n = max(4, int(round(T / f)))
    src = np.linspace(0, T - 1, n); lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1); w = (src - lo)[:, None]
    near = np.rint(src).astype(int)
    b = base[lo] * (1 - w) + base[hi] * w
    p = pres[near]; b[~p[:, 0], :44] = 0; b[~p[:, 1], 44:88] = 0
    return b.astype(np.float32), p, y[near]


def composite(pools, rng, max_len=400):
    k = rng.integers(1, 5)
    parts = []
    for _ in range(k):
        pool = pools[rng.integers(len(pools))]
        it = pool[rng.integers(len(pool))]
        base, pres, y = it['base'], it['pres'], it['y']
        if rng.random() < .6:
            base, pres, y = resample(base, pres, y, math.exp(rng.uniform(math.log(.8), math.log(2.))))
        parts.append((augment(base, pres, rng), pres, y))
    base = np.concatenate([p[0] for p in parts]); pres = np.concatenate([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
    return add_deltas(base[:max_len], pres[:max_len]), y[:max_len]


def window(items, rng, n=160):
    it = items[rng.integers(len(items))]
    T = len(it['base'])
    fs = np.where(it['y'] == 1)[0]
    if len(fs) and rng.random() < .5:                            # oversample real spelling in context
        s = int(np.clip(fs[rng.integers(len(fs))] - rng.integers(0, n), 0, max(0, T - n)))
    else:
        s = int(rng.integers(0, max(1, T - n)))
    base, pres, y = it['base'][s:s + n], it['pres'][s:s + n], it['y'][s:s + n]
    return add_deltas(augment(base, pres, rng), pres), y


def batch(rows, device):
    T = max(len(r[0]) for r in rows)
    lm = np.zeros((len(rows), T, rows[0][0].shape[1]), np.float32); y = np.full((len(rows), T), -1., np.float32)
    mask = np.zeros((len(rows), T), bool)
    for i, (l, yy) in enumerate(rows):
        lm[i, :len(l)] = l; y[i, :len(l)] = yy; mask[i, :len(l)] = True
    return torch.from_numpy(lm).to(device), torch.from_numpy(y).to(device), torch.from_numpy(mask).to(device)


@torch.no_grad()
def probs(model, items, device):
    model.eval(); out = []
    for it in items:
        lm = torch.from_numpy(add_deltas(it['base'], it['pres']))[None].to(device)
        out.append(torch.sigmoid(model(lm, torch.ones(lm.shape[:2], dtype=torch.bool, device=device)))[0].cpu().numpy())
    model.train()
    return out


def validate(model, val, device):
    r = {}
    p = np.concatenate(probs(model, val['fswild_dev'], device)); r['fswild_dev_recall'] = float((p >= .5).mean())
    p = np.concatenate(probs(model, val['citizen_val'], device)); r['citizen_val_reject'] = float((p < .5).mean())
    ps = probs(model, val['o5s5_lg'], device)
    y = np.concatenate([it['y'] for it in val['o5s5_lg']]); p = np.concatenate(ps)
    tpr = float((p[y == 1] >= .5).mean()) if (y == 1).any() else 0.; tnr = float((p[y == 0] < .5).mean())
    r['lg_fs_recall'], r['lg_sign_reject'] = tpr, tnr
    r['score'] = (r['fswild_dev_recall'] + r['citizen_val_reject'] + (tpr + tnr) / 2) / 3
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fsb = clip_items(sorted(glob.glob(str(ROOT / 'data/local/fsboard_v17/batch1/features/train/*.npz'))), fsboard_labels)
    wild = clip_items(sorted(glob.glob(str(ROOT / 'data/local/chicago_fswild/features/train/*.npz'))), all_label(1))
    cit = clip_items(sorted(glob.glob(str(ROOT / 'data/local/citizen100_v17/live_features/train/*.npz'))), all_label(0))
    o5 = o5_items(eval_only=False)
    val = dict(fswild_dev=clip_items(sorted(glob.glob(str(ROOT / 'data/local/chicago_fswild/features/dev/*.npz'))), all_label(1)),
               citizen_val=clip_items(sorted(glob.glob(str(ROOT / 'data/local/citizen100_v17/live_features/val/*.npz'))), all_label(0)),
               o5s5_lg=o5_items(eval_only=True))
    lab = lambda items: (sum(int((it['y'] == 1).sum()) for it in items), sum(int((it['y'] == 0).sum()) for it in items))
    print('train frames (spell, not):', 'fsboard', lab(fsb), 'fswild', lab(wild), 'citizen', lab(cit), 'o5s5', lab(o5),
          '| val clips', {k: len(v) for k, v in val.items()}, f'| loaded {time.time() - t0:.0f}s', flush=True)
    model = SpellGate().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * .5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    pools = [fsb, wild, cit, cit]                                  # signing sampled as often as spelling
    best, history, losses = -1., [], []
    for step in range(a.steps):
        rows = [composite(pools, rng) for _ in range(a.batch - 8)] + [window(o5, rng) for _ in range(8)]
        lm, y, mask = batch(rows, a.device)
        logits = model(lm, mask)
        keep = mask & (y >= 0)
        loss = F.binary_cross_entropy_with_logits(logits[keep], y[keep])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            r = validate(model, val, a.device) | dict(step=step + 1, loss=float(np.mean(losses[-250:])), minutes=(time.time() - t0) / 60)
            history.append(r)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            if r['score'] > best:
                best = r['score']
                torch.save(dict(format=FORMAT, config=model.config, state_dict=model.state_dict(), step=step + 1, val=r,
                                args=vars(a) | dict(out=str(a.out))), a.out / 'best.pt')
    (a.out / 'history.json').write_text(json.dumps(dict(best_score=best, history=history), indent=1))
    print('best score', round(best, 4), flush=True)


if __name__ == '__main__':
    main()
