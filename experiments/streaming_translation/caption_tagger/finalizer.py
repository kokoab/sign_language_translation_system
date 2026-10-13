"""Learned clause finalizer: at most one sign move + full-context rendering.

When a clause closes, the live caption for that clause may be revised once:
 1. Mover (bidirectional): picks which sign, if any, moves and where it is
    inserted (WH-fronting, negation placement, time placement).
 2. Labeler (bidirectional tagger): renders the reordered clause with the same
    closed insertion/form/punctuation inventory as the live tagger.
Signs can only move, never disappear or be invented. Trained on corpus sentences
whose English order differs from sign order by at most one moved sign (97% of
the train split); targets derive from the corpus English alone.
"""
import bisect
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align import corpus_rows, row_sentences, labels, DROP  # noqa: E402
from forms import FORMS, tokenize  # noqa: E402
from tagger import Tagger, batchify, legal_mask, bucket, PAD, MAXLEN  # noqa: E402

IGNORE = -100


def english_positions(g, toks):
    used, pos = set(), []
    for x in g:
        best = None
        for f in FORMS[x]:
            L = len(f)
            for j in range(len(toks) - L + 1):
                if tuple(toks[j:j + L]) == f and j not in used:
                    if best is None or j < best:
                        best = j
                    break
        pos.append(best)
        if best is not None:
            used.add(best)
    return pos


def single_move(pos):
    """(src, dst) such that moving gloss src before index dst sorts pos; (None, None)
    if already sorted; False if more than one move is needed. dst indexes the
    original sequence (len = end)."""
    idx = [i for i, p in enumerate(pos) if p is not None]
    vals = [pos[i] for i in idx]
    if all(a < b for a, b in zip(vals, vals[1:])):
        return None, None
    for s in idx:
        rest = [i for i in idx if i != s]
        rv = [pos[i] for i in rest]
        if all(a < b for a, b in zip(rv, rv[1:])):
            # destination: before the first remaining gloss with a later English position
            later = [i for i in rest if pos[i] > pos[s]]
            return s, (later[0] if later else len(pos))
    return False


def apply_move(seq, src, dst):
    if src is None:
        return list(seq)
    item = seq[src]
    out = [x for i, x in enumerate(seq) if i != src]
    d = dst if dst <= src else dst - 1
    out.insert(d, item)
    return out


def build():
    D = json.loads((HERE / 'data.json').read_text())
    pid = {v: i for i, v in enumerate(D['pre_vocab'])}
    qid = {v: i for i, v in enumerate(D['post_vocab'])}
    fid = {v: i for i, v in enumerate(D['form_vocab'])}
    gid = {g: i + 1 for i, g in enumerate(D['glosses'])}
    out = {'train': [], 'validation': []}
    stats = {'monotone': 0, 'one_move': 0, 'excluded': 0}
    for r in corpus_rows():
        k = 0
        for g, s in row_sentences(r):
            conf = [bucket(c) for c in r['confidences'][k:k + len(g)]]; k += len(g)
            toks = tokenize(s)
            mv = single_move(english_positions(g, toks))
            if mv is False:
                stats['excluded'] += r['split'] == 'train'; continue
            src, dst = mv
            stats['monotone' if src is None else 'one_move'] += r['split'] == 'train'
            g2 = apply_move(g, src, dst); c2 = apply_move(conf, src, dst)
            lab, m, _ = labels(g2, toks)
            enc = {'x': [gid[x] for x in g], 'conf': conf,
                   'src': len(g) if src is None else src, 'dst': len(g) if dst is None else dst,
                   'x2': [gid[x] for x in g2], 'conf2': c2,
                   'pre': [pid.get(l['pre'], IGNORE) for l in lab],
                   'form': [fid[l['form']] for l in lab],
                   'post': [qid.get(l['post'], IGNORE) for l in lab]}
            out[r['split']].append(enc)
    (HERE / 'finalizer_data.json').write_text(json.dumps(out))
    print(stats, {k: len(v) for k, v in out.items()})


class Mover(nn.Module):
    """Bidirectional encoder; src head over positions (+ 'none' at index n),
    dst head over insertion slots 0..n."""

    def __init__(self, d=160, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Embedding(102, d, padding_idx=PAD)
        self.conf = nn.Embedding(4, d, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.src = nn.Linear(d, 1)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)

    def forward(self, x, conf):
        """x has an EOS token at index n which doubles as 'none'/'end'."""
        T = x.shape[1]
        h = self.tok(x) + self.conf(conf) + self.pos(torch.arange(T, device=x.device))[None]
        h = self.norm(self.enc(h, src_key_padding_mask=x == PAD))
        pad = x == PAD
        src = self.src(h).squeeze(-1).masked_fill(pad, -1e4)
        b = torch.arange(x.shape[0], device=x.device)
        return h, src, pad

    def dst_logits(self, h, pad, src_idx):
        b = torch.arange(h.shape[0], device=h.device)
        q = self.q(h[b, src_idx])  # [B,d]
        return torch.einsum('bd,btd->bt', q, self.k(h)).masked_fill(pad, -1e4)


def mover_batch(rows, device):
    T = max(len(r['x']) for r in rows) + 1
    x = torch.zeros(len(rows), T, dtype=torch.long); c = torch.zeros_like(x)
    for b, r in enumerate(rows):
        n = len(r['x'])
        x[b, :n] = torch.tensor(r['x']); x[b, n] = 101
        c[b, :n] = torch.tensor(r['conf']); c[b, n] = 1
    src = torch.tensor([r['src'] for r in rows]); dst = torch.tensor([r['dst'] for r in rows])
    return x.to(device), c.to(device), src.to(device), dst.to(device)


def train_mover(epochs, device, log):
    D = json.loads((HERE / 'finalizer_data.json').read_text())
    torch.manual_seed(0); random.seed(0)
    m = Mover().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    tr, va = D['train'], D['validation']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * math.ceil(len(tr) / 256),
                                                pct_start=0.1)
    t0, best, hist = time.time(), -1, []
    for ep in range(epochs):
        m.train(); random.shuffle(tr)
        for s in range(0, len(tr), 256):
            x, c, src, dst = mover_batch(tr[s:s + 256], device)
            h, sl, pad = m(x, c)
            dl = m.dst_logits(h, pad, src)  # teacher-forced source
            moved = src != (x != PAD).sum(1) - 1
            loss = F.cross_entropy(sl, src) + (F.cross_entropy(dl[moved], dst[moved]) if moved.any() else 0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = eval_mover(m, va, device)
        hist.append({'epoch': ep + 1, **acc, 'elapsed_s': time.time() - t0}); print('mover', hist[-1], flush=True)
        if acc['exact'] > best:
            best = acc['exact']; torch.save(m.state_dict(), HERE / 'mover.pt')
    log['mover'] = {'history': hist, 'best_validation_exact': best, 'seconds': time.time() - t0,
                    'params': sum(p.numel() for p in m.parameters())}


@torch.no_grad()
def eval_mover(m, rows, device):
    m.eval(); ok = n = moved_ok = moved_n = 0
    for s in range(0, len(rows), 512):
        x, c, src, dst = mover_batch(rows[s:s + 512], device)
        h, sl, pad = m(x, c)
        ps = sl.argmax(-1)
        pd = m.dst_logits(h, pad, ps).argmax(-1)
        none = ps == (x != PAD).sum(1) - 1
        e = (ps == src) & (none | (pd == dst))
        ok += int(e.sum()); n += len(src)
        gm = src != (x != PAD).sum(1) - 1
        moved_ok += int((e & gm).sum()); moved_n += int(gm.sum())
    return {'exact': round(ok / n, 4), 'moved_exact': round(moved_ok / max(1, moved_n), 4)}


def train_labeler(epochs, device, log):
    """Bidirectional tagger on reordered clauses (reuses tagger.Tagger with k='full')."""
    import tagger as T
    D = json.loads((HERE / 'data.json').read_text())
    F_ = json.loads((HERE / 'finalizer_data.json').read_text())
    rows = {s: [{'x': r['x2'], 'conf': r['conf2'], 'pre': r['pre'], 'form': r['form'], 'post': r['post']}
                for r in F_[s]] for s in F_}
    torch.manual_seed(0); random.seed(0)
    model = Tagger(len(D['pre_vocab']), len(D['form_vocab']), len(D['post_vocab']), 'full').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    tr, va = rows['train'], rows['validation']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * math.ceil(len(tr) / 256),
                                                pct_start=0.1)
    lm = legal_mask(D, device)
    t0, best, hist = time.time(), -1, []
    for ep in range(epochs):
        model.train(); random.shuffle(tr)
        for s in range(0, len(tr), 256):
            x, c, L, y = batchify(tr[s:s + 256], device)
            pre, form, post = model(x, c, L)
            form = form + lm[x]
            loss = sum(F.cross_entropy(o.flatten(0, 1), y[f].flatten(), ignore_index=-100)
                       for o, f in ((pre, 'pre'), (form, 'form'), (post, 'post')))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = T.evaluate(model, va, lm, device)
        hist.append({'epoch': ep + 1, **acc, 'elapsed_s': time.time() - t0}); print('labeler', hist[-1], flush=True)
        if acc['joint'] > best:
            best = acc['joint']; torch.save(model.state_dict(), HERE / 'labeler.pt')
    log['labeler'] = {'history': hist, 'best_validation_joint': best, 'seconds': time.time() - t0,
                      'params': sum(p.numel() for p in model.parameters())}


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        lp = HERE / 'training_log.json'
        log = json.loads(lp.read_text())
        {'mover': train_mover, 'labeler': train_labeler}[sys.argv[1]](int(sys.argv[2]), device, log)
        lp.write_text(json.dumps(log, indent=1))
