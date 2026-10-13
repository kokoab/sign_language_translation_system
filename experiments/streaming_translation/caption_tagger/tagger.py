"""Lookahead-k caption tagger: model, training and decoding.

Causal transformer over glosses (+ confidence bucket) with an <eos> token appended
at utterance end. The labels for gloss i are read from hidden states h_i and
h_min(i+k, eos), so a gloss's rendering depends only on inputs up to i+k: exactly
what is available when the (i+k)-th sign has been recognized. k='full' is a
bidirectional upper bound that sees the whole utterance.
"""
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
from forms import detokenize  # noqa: E402

PAD, EOS = 0, 101
MAXLEN = 48


def bucket(score):
    return 1 if score >= .6 else 2 if score >= .4 else 3


class Tagger(nn.Module):
    def __init__(self, n_pre, n_form, n_post, k, d=192, layers=3, heads=4):
        super().__init__()
        self.k = k
        self.tok = nn.Embedding(102, d, padding_idx=PAD)
        self.conf = nn.Embedding(4, d, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN + 1, d)
        self.off = nn.Embedding(8, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.mix = nn.Sequential(nn.Linear(3 * d, d), nn.GELU())
        self.pre, self.form, self.post = nn.Linear(d, n_pre), nn.Linear(d, n_form), nn.Linear(d, n_post)

    def forward(self, x, conf, lengths):
        """x: [B,T] glosses with EOS at index lengths. Returns logits for positions < lengths."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.conf(conf) + self.pos(pos)[None]
        pad = x == PAD
        mask = None if self.k == 'full' else torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = self.norm(self.enc(h, mask=mask, src_key_padding_mask=pad))
        if self.k == 'full':
            j = pos[None].expand(B, T)
        else:
            j = torch.minimum(pos[None] + self.k, lengths[:, None])  # eos sits at index `lengths`
        hj = torch.gather(h, 1, j[..., None].expand(B, T, h.shape[-1]))
        off = (j - pos[None]).clamp(0, 7)
        z = self.mix(torch.cat([h, hj, self.off(off)], -1))
        return self.pre(z), self.form(z), self.post(z)


def batchify(rows, device):
    T = max(len(r['x']) for r in rows) + 1
    x = torch.zeros(len(rows), T, dtype=torch.long)
    c = torch.zeros_like(x)
    ys = {f: torch.full((len(rows), T), -100, dtype=torch.long) for f in ('pre', 'form', 'post')}
    lengths = torch.tensor([len(r['x']) for r in rows])
    for b, r in enumerate(rows):
        n = len(r['x'])
        x[b, :n] = torch.tensor(r['x']); x[b, n] = EOS
        c[b, :n] = torch.tensor(r['conf']); c[b, n] = 1
        for f in ys:
            ys[f][b, :n] = torch.tensor(r[f])
    return x.to(device), c.to(device), lengths.to(device), {f: v.to(device) for f, v in ys.items()}


def legal_mask(D, device):
    m = torch.full((102, len(D['form_vocab'])), -1e4)
    m[PAD] = 0; m[EOS] = 0  # never labelled; keeps ignored positions finite
    for i, g in enumerate(D['glosses']):
        m[i + 1, D['legal_forms'][g]] = 0
    return m.to(device)


def train(k, epochs, device, log, data='data.json', suffix=''):
    D = json.loads((HERE / data).read_text())
    torch.manual_seed(0); random.seed(0)
    model = Tagger(len(D['pre_vocab']), len(D['form_vocab']), len(D['post_vocab']), k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    tr, va = D['train'], D['validation']
    steps = epochs * math.ceil(len(tr) / 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    lm = legal_mask(D, device)
    best, t0, history = -1, time.time(), []
    tag = f'k{k}{suffix}'
    for ep in range(epochs):
        model.train(); random.shuffle(tr)
        for s in range(0, len(tr), 128):
            x, c, L, y = batchify(tr[s:s + 128], device)
            pre, form, post = model(x, c, L)
            form = form + lm[x]
            loss = sum(F.cross_entropy(o.flatten(0, 1), y[f].flatten(), ignore_index=-100)
                       for o, f in ((pre, 'pre'), (form, 'form'), (post, 'post')))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = evaluate(model, va, lm, device)
        history.append({'epoch': ep + 1, 'loss': float(loss.detach()), **acc, 'elapsed_s': time.time() - t0})
        print(tag, history[-1], flush=True)
        if acc['joint'] > best:
            best = acc['joint']
            torch.save(model.state_dict(), HERE / f'tagger_{tag}.pt')
    log[tag] = {'history': history, 'best_validation_joint': best, 'seconds': time.time() - t0,
                'params': sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def evaluate(model, rows, lm, device):
    model.eval()
    hits = {f: 0 for f in ('pre', 'form', 'post', 'joint')}; n = 0
    for s in range(0, len(rows), 256):
        x, c, L, y = batchify(rows[s:s + 256], device)
        pre, form, post = model(x, c, L)
        p = {'pre': pre.argmax(-1), 'form': (form + lm[x]).argmax(-1), 'post': post.argmax(-1)}
        valid = (y['pre'] != -100) & (y['form'] != -100) & (y['post'] != -100)
        ok = torch.ones_like(valid)
        for f in ('pre', 'form', 'post'):
            e = (p[f] == y[f]) & valid
            hits[f] += int(e.sum()); ok &= e
        hits['joint'] += int(ok.sum()); n += int(valid.sum())
    return {f: round(v / n, 4) for f, v in hits.items()}


class Captioner:
    """Loads a trained tagger and renders gloss prefixes to caption text."""

    def __init__(self, k, device='cpu'):
        self.D = json.loads((HERE / 'data.json').read_text())
        D = self.D
        self.model = Tagger(len(D['pre_vocab']), len(D['form_vocab']), len(D['post_vocab']), k)
        self.model.load_state_dict(torch.load(HERE / f'tagger_k{k}.pt', map_location=device))
        self.model.eval(); self.k = k
        self.gid = {g: i + 1 for i, g in enumerate(D['glosses'])}
        self.lm = legal_mask(D, device)
        # Joint (insertion, form) decoding over pairs seen >= 5 times in the train split.
        pairs = torch.zeros(len(D['pre_vocab']), len(D['form_vocab']))
        for r in D['train']:
            for p, f in zip(r['pre'], r['form']):
                if p != -100 and f != -100:
                    pairs[p, f] += 1
        self.pair_mask = torch.where(pairs >= 5, 0.0, -1e4)

    @torch.no_grad()
    def tags(self, glosses, scores, ended):
        """Per-gloss (pre, form, post) strings given the prefix; ended appends <eos>.

        Without <eos>, position min(i+k, n-1) is the latest available state; the
        caller only uses gloss i once its lookahead is available (or at end).
        """
        x = [self.gid[g] for g in glosses]
        conf = [bucket(s) for s in scores]
        n = len(x)
        xt = torch.tensor([x + ([EOS] if ended else [])])
        ct = torch.tensor([conf + ([1] if ended else [])])
        L = torch.tensor([n if ended else n - 1])  # clamp index for missing lookahead
        pre, form, post = self.model(xt, ct, L)
        lp_pre = pre.log_softmax(-1)[0]
        lp_form = (form + self.lm[xt]).log_softmax(-1)[0]
        out = []
        for i in range(n):
            joint = lp_pre[i][:, None] + lp_form[i][None, :] + self.pair_mask
            p, f = divmod(int(joint.argmax()), joint.shape[1])
            out.append((self.D['pre_vocab'][p], self.D['form_vocab'][f],
                        self.D['post_vocab'][post[0, i].argmax()]))
        return out


def render(tags):
    toks = []
    for pre, form, post in tags:
        toks += pre.split()
        if form != '<drop>':
            toks += form.split()
        toks += post.split()
    return detokenize(toks)


def prepare_conf():
    """Attach confidence buckets to data.json rows from the corpus (same order)."""
    D = json.loads((HERE / 'data.json').read_text())
    from align import corpus_rows
    split_rows = {'train': [], 'validation': []}
    for r in corpus_rows():
        split_rows[r['split']].append(r)
    for split in ('train', 'validation'):
        for enc, r in zip(D[split], split_rows[split]):
            enc['conf'] = [bucket(s) for s in r['confidences']]
    (HERE / 'data.json').write_text(json.dumps(D))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare_conf()
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        log_path = HERE / 'training_log.json'
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        k = sys.argv[1] if sys.argv[1] == 'full' else int(sys.argv[1])
        if len(sys.argv) > 3:  # e.g. `tagger.py 1 3 b` -> data_b.json, tagger_k1b.pt
            train(k, int(sys.argv[2]), device, log, f'data_{sys.argv[3]}.json', sys.argv[3])
        else:
            train(k, int(sys.argv[2]), device, log)
        log_path.write_text(json.dumps(log, indent=1))
