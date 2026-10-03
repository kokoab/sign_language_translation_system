"""Distil the frozen DGS pose segmenter (MediaPipe, 500 ms lookahead) into an Apple Vision student.

Teacher targets are DGS sign-BIO distributions at lookahead 10 frames, read from the cached
all-window outputs in artifacts/cache/segmental_decoder_v17/dgs. Student inputs are the live
`boundary_features` from Apple Vision raw observations cached alongside span scoring. No human
labels are needed, so any video with both caches can be used. Test, tuning and ASLLRP
validation videos are always excluded. Selection: held-out-video KL to the teacher.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.av_boundary_v17 import AVBoundary, FORMAT, FRAMES, windows
from active.v17.temporal_boundary_v17 import boundary_features
from scripts import segmental_lab_v17 as lab


def teacher_obi(key, teacher_lookahead=10):
    path = lab.CACHE / 'dgs' / (key + '.npz')
    if not path.exists():
        return None
    sign = np.load(path)['sign'].astype(np.float32)
    p = np.exp(lab.bio_at_lookahead(sign, teacher_lookahead))
    obi = np.stack([p[:, 1] + p[:, 0], p[:, 2], p[:, 3]], 1)
    return obi / obi.sum(1, keepdims=True)


# Android family: --av-raw-dir points student inputs at MediaPipe captures of the same videos.
AV_RAW_DIR = None


def student_inputs(key):
    apple = lab.CACHE / 'av_raw' / (key + '.npz')
    if not apple.exists() or apple.stat().st_size == 0:
        return None  # identical video set to the Apple student
    path = apple if AV_RAW_DIR is None else AV_RAW_DIR / (key + '.npz')
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = np.load(path)
    raw, times = data['raw'].astype(np.float32), data['times'].astype(np.float64)
    if len(times) < 4:
        return None
    return boundary_features(raw, times, True), times


def aligned(key, teacher_lookahead=10):
    t = teacher_obi(key, teacher_lookahead)
    s = student_inputs(key)
    if t is None or s is None:
        return None
    feats, times = s
    grid = np.clip(np.round(times * lab.FPS).astype(int), 0, len(t) - 1)
    return feats, t[grid]


def letter_clips(role, cap, seed):
    """Held static letters: O before the hand appears, B at its first frame, I while it is up."""
    import glob
    import random
    paths = sorted(glob.glob(str(ROOT / 'data/local/fingerspelling_letters_v17/spans' / role / '*' / '*.npz')))
    random.Random(seed).shuffle(paths)
    out = []
    for path in paths:
        d = np.load(path)
        if 'raw' not in d:
            continue
        raw, times = d['raw'].astype(np.float32), d['times'].astype(np.float64)
        if len(times) < 4:
            continue
        present = (raw[:, :42, 4] > 0).any(1)
        for i in range(1, len(present) - 1):          # bridge single-frame detector dropouts
            if not present[i] and present[i - 1] and present[i + 1]:
                present[i] = True
        if present.sum() < 3:
            continue
        y = np.zeros((len(times), 3), np.float32)
        y[~present, 0] = 1
        y[present, 2] = 1
        first = int(np.flatnonzero(present)[0])
        y[first] = [0, 1, 0]
        out.append((boundary_features(raw, times, True), y))
        if cap and len(out) >= cap:
            break
    return out


def collect(excluded):
    keys = []
    for path in sorted((lab.CACHE / 'av_raw').glob('*.npz')):
        k = path.stem
        if k in excluded:
            continue
        if (lab.CACHE / 'dgs' / (k + '.npz')).exists():
            keys.append(k)
    return keys


def batches(clips, lookahead, batch, rng):
    """Random read-position windows across all clips."""
    index = [(c, t) for c, (f, _) in enumerate(clips) for t in range(len(f))]
    rng.shuffle(index)
    read = FRAMES - 1 - lookahead
    for i in range(0, len(index), batch):
        chunk = index[i:i + batch]
        xs, vs, ys = [], [], []
        for c, t in chunk:
            f, y = clips[c]
            n = len(f)
            idx = np.arange(FRAMES) + t - read
            valid = (idx >= 0) & (idx < n)
            xs.append(f[np.clip(idx, 0, n - 1)] * valid[:, None])
            vs.append(valid)
            ys.append(y[t])
        yield (torch.from_numpy(np.stack(xs).astype(np.float32)), torch.from_numpy(np.stack(vs)),
               torch.from_numpy(np.stack(ys).astype(np.float32)))


@torch.inference_mode()
def evaluate(model, clips, device):
    model.eval()
    kl = agree = n = 0.
    for f, y in clips:
        x, v = windows(f, model.lookahead)
        logits = model(torch.from_numpy(x).to(device), torch.from_numpy(v).to(device)).float().cpu()
        logp = torch.log_softmax(logits, -1)
        t = torch.from_numpy(y.astype(np.float32))
        kl += float((t * (torch.log(t.clamp_min(1e-8)) - logp)).sum())
        agree += float((logp.argmax(1) == t.argmax(1)).sum())
        n += len(y)
    return dict(kl=kl / n, agreement=agree / n, frames=int(n))


@torch.inference_mode()
def letter_activity(model, clips, device):
    """Share of held-out letter clips with >= 3 frames labelled as signing (B or I)."""
    model.eval()
    ok = 0
    for f, _ in clips:
        x, v = windows(f, model.lookahead)
        arg = model(torch.from_numpy(x).to(device), torch.from_numpy(v).to(device)).argmax(-1).cpu().numpy()
        ok += int((arg >= 1).sum() >= 3)
    return ok / max(len(clips), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--lookahead', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=27930)
    ap.add_argument('--val-fraction', type=float, default=.08)
    ap.add_argument('--teacher-lookahead', type=int, default=10)
    ap.add_argument('--dropout', type=float, default=.1)
    ap.add_argument('--letter-clips', type=int, default=0, help='add up to N train-session letter clips')
    ap.add_argument('--init', type=Path, default=None, help='start from an existing student')
    ap.add_argument('--av-raw-dir', type=Path, default=None, help='MediaPipe continuous/av_raw (Android family)')
    ap.add_argument('--keys-file', type=Path, default=None, help='restrict to an exact video-key list (JSON)')
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    global AV_RAW_DIR
    AV_RAW_DIR = args.av_raw_dir
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    excluded = {lab.key(r['source_item_id']) for name in ('test', 'tune', 'asllrp_val') for r in lab.rows_for(name)}
    keys = collect(excluded)
    if args.keys_file is not None:
        allowed = set(json.loads(args.keys_file.read_text()))
        keys = [k for k in keys if k in allowed]
    rng = random.Random(args.seed)
    import hashlib
    order = sorted(keys, key=lambda k: hashlib.sha256(f'{args.seed}:{k}'.encode()).hexdigest())
    n_val = max(8, int(len(order) * args.val_fraction))
    val_keys, train_keys = order[:n_val], order[n_val:]
    train = [c for k in train_keys if (c := aligned(k, args.teacher_lookahead)) is not None]
    val = [c for k in val_keys if (c := aligned(k, args.teacher_lookahead)) is not None]
    letters_val = []
    if args.letter_clips:
        train += letter_clips('train', args.letter_clips, args.seed)
        letters_val = letter_clips('validation', 400, args.seed)
    print(json.dumps(dict(train_clips=len(train), val_clips=len(val), train_frames=sum(len(f) for f, _ in train),
                          letter_val_clips=len(letters_val),
                          excluded=len(excluded))), flush=True)

    model = AVBoundary(hidden=args.hidden, layers=args.layers, lookahead=args.lookahead, dropout=args.dropout).to(device)
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=False)['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * ((sum(len(f) for f, _ in train) + args.batch - 1) // args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=.1)
    history, best = [], None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = seen = 0.
        for x, v, y in batches(train, args.lookahead, args.batch, rng):
            x, v, y = x.to(device), v.to(device), y.to(device)
            logp = torch.log_softmax(model(x, v), -1)
            loss = -(y * logp).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()
            total += float(loss.detach()) * len(y); seen += len(y)
        row = dict(epoch=epoch, loss=total / seen, val=evaluate(model, val, device),
                   letters_val=letter_activity(model, letters_val, device) if letters_val else None,
                   minutes=(time.perf_counter() - started) / 60)
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row['val']['kl'] < best['val']['kl']:
            best = dict(row, state=copy.deepcopy(model.state_dict()))
    args.output_dir.mkdir(parents=True)
    state = best.pop('state')
    torch.save(dict(format=FORMAT, config=model.config, state_dict=state, epoch=best['epoch'],
                    teacher='pose_boundary_dgs_2026 sign BIO at lookahead %d (MediaPipe)' % args.teacher_lookahead,
                    train_keys=train_keys, val_keys=val_keys, excluded_sets=['test', 'tune', 'asllrp_val'],
                    selected=best), args.output_dir / 'model.pth')
    (args.output_dir / 'history.json').write_text(json.dumps(dict(args=vars(args) | dict(output_dir=str(args.output_dir)),
                                                                  history=history, selected=best), indent=1, default=str) + '\n')
    print(json.dumps(dict(selected=best['epoch'], val=best['val'])), flush=True)


def load(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT:
        raise ValueError('not an AV boundary student')
    model = AVBoundary(**payload['config'])
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval()


if __name__ == '__main__':
    main()
