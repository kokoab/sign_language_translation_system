"""Part A2: how much right context does each edit decision need?

Count-based backoff predictor P(tag | 2 left glosses, gloss, k right glosses),
fitted on train, scored on validation labelled positions. Not the proposed model;
a transparent estimate of decidability at lookahead k.
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'data.json').read_text())
FIELDS = ('pre', 'form', 'post')


def contexts(x, i, left, k):
    """Backoff chain from most to least specific context keys."""
    right = tuple(x[i + 1:i + 1 + k]) + (('<eos>',) if i + 1 + k > len(x) else ())
    keys = []
    for L in range(left, -1, -1):
        keys.append((L, k, tuple(x[max(0, i - L):i]), x[i], right))
    # then shrink right context too
    for kk in range(k - 1, -1, -1):
        r = tuple(x[i + 1:i + 1 + kk]) + (('<eos>',) if i + 1 + kk > len(x) else ())
        keys.append((0, kk, (), x[i], r))
    return keys


def fit(rows, field, left, k):
    table = collections.defaultdict(collections.Counter)
    for r in rows:
        for i, y in enumerate(r[field]):
            if y == -100:
                continue
            for key in contexts(r['x'], i, left, k):
                table[key][y] += 1
    return {key: c.most_common(1)[0][0] for key, c in table.items()}


def score(rows, model, field, left, k):
    ok = n = 0
    for r in rows:
        for i, y in enumerate(r[field]):
            if y == -100:
                continue
            for key in contexts(r['x'], i, left, k):
                if key in model:
                    ok += model[key] == y; break
            n += 1
    return ok / n


def joint(rows):
    for r in rows:
        r['joint'] = [-100 if -100 in t else t[0] * 100000 + t[1] * 1000 + t[2]
                      for t in zip(r['pre'], r['form'], r['post'])]


def main():
    joint(D['train']); joint(D['validation'])
    out = {}
    for k in (0, 1, 2, 3):
        out[k] = {}
        for field in FIELDS + ('joint',):
            m = fit(D['train'], field, 2, k)
            out[k][field] = round(score(D['validation'], m, field, 2, k), 4)
        print(k, out[k], flush=True)
    (HERE / 'lookahead_counts.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
