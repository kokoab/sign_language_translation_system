"""Model-independent audit of the letter labels (train + validation).

Hand-shape descriptor per clip: the dominant hand's 21 joints relative to the wrist, scaled by
the palm (wrist -> middle MCP), median over the middle half of the clip. Orientation is kept
(it separates G/Q, H/U, K/P); left/right hands are treated as mirror images (distance = min over
an x-flip). Every clip's 10 nearest neighbours are compared with its label; a clip is flagged
when >= 7 of 10 neighbours share one *other* label. Label corrections already made are applied.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.train_letter_head_v17 import corrected_labels
LET = ROOT / 'data/local/fingerspelling_letters_v17/spans'
corr = corrected_labels()


def descriptor(raw):
    middle = raw[len(raw) // 4:max(len(raw) // 4 + 1, 3 * len(raw) // 4)]
    rows = []
    for row in middle:
        s = 0 if row[:21, 4].sum() >= row[21:42, 4].sum() else 21
        h = row[s:s + 21]
        if (h[:, 4] > 0).sum() < 18:
            continue
        xy = h[:, :2] - h[0, :2]
        palm = np.linalg.norm(xy[9])
        if palm < 1e-3:
            continue
        rows.append((xy[1:] / palm).reshape(-1))
    return np.median(np.stack(rows), 0) if len(rows) >= 3 else None


items = []
for split in ('train', 'validation'):
    for p in sorted(LET.glob(f'{split}/*/*.npz')):
        with np.load(p) as d:
            if 'raw' not in d:
                continue
            desc = descriptor(d['raw'].astype(np.float32))
            meta = json.loads(str(d['meta']))
        if desc is None:
            continue
        key = str(p.relative_to(LET))
        items.append(dict(key=key, split=split, label=corr.get(key, p.parent.name), name=meta['name'],
                          session=meta.get('session'), desc=desc))
X = np.stack([i['desc'] for i in items]).astype(np.float32)
flip = X.reshape(len(X), 20, 2) * np.array([-1, 1], np.float32)
flip = flip.reshape(len(X), -1)


def sqdist(A, B):
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T


D = np.minimum(sqdist(X, X), sqdist(X, flip))
np.fill_diagonal(D, np.inf)
labels = np.array([i['label'] for i in items])
nn = np.argsort(D, 1)[:, :10]
flags = []
for k, row in enumerate(nn):
    votes = Counter(labels[row])
    top, count = votes.most_common(1)[0]
    if top != labels[k] and count >= 7:
        flags.append(dict(key=items[k]['key'], split=items[k]['split'], label=labels[k], neighbours=top,
                          votes=count, own_votes=int(votes[labels[k]]), name=items[k]['name'], session=items[k]['session']))
pairs = Counter((f['label'], f['neighbours']) for f in flags)
per_letter = Counter(i['label'] for i in items)
summary = dict(clips=len(items), flagged=len(flags),
               pairs=[dict(label=a, looks_like=b, clips=c, of=per_letter[a]) for (a, b), c in pairs.most_common()])
(Path(__file__).parent / 'audit.json').write_text(json.dumps(dict(summary=summary, flags=flags), indent=1))
print(json.dumps(summary['clips']), 'clips;', len(flags), 'flagged')
for p in summary['pairs']:
    print(f"  labelled {p['label']} but looks like {p['looks_like']}: {p['clips']} (of {p['of']})")
