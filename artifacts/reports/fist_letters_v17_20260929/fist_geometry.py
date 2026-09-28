"""Can hand geometry separate the fist letters (A E M N S T)? Train on the letter train split,
test on the validation split (other signers) and on the user's harvested E spans.

Features per clip from the recognizer's span landmarks [32,61,5] (body-relative, resampled):
dominant hand, middle frames, canonicalised: origin wrist, y along wrist -> middle MCP, unit palm,
mirrored so the index MCP is on +x. 20 joints (40 values) + thumb-tip distances to every finger joint.
"""
import json, sys, glob
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LET = ROOT / 'data/local/fingerspelling_letters_v17/spans'
USER = ROOT / 'data/local/fingerspelling_letters_v17/user_sessions'
FIST = list('AEMNST')


def features(lm):
    lm = np.asarray(lm, np.float32)
    s = 0 if lm[:, :21, 3].sum() >= lm[:, 21:42, 3].sum() else 21
    rows = []
    for f in lm[8:24]:
        h = f[s:s + 21]
        if (h[:, 3] > .5).sum() < 18:
            continue
        xy = h[:, :2] - h[0, :2]
        up = xy[9]; palm = np.linalg.norm(up)
        if palm < 1e-4:
            continue
        ey = up / palm; ex = np.array([ey[1], -ey[0]])
        c = np.stack([xy @ ex, xy @ ey], 1) / palm
        if c[5, 0] < c[17, 0]:
            c[:, 0] *= -1
        thumb = c[4]
        dist = np.linalg.norm(c[5:] - thumb, axis=1)
        rows.append(np.concatenate([c[1:].reshape(-1), dist]))
    return np.median(np.stack(rows), 0) if len(rows) >= 3 else None


def load(split):
    X, y, names = [], [], []
    corr = {}
    cf = LET.parent / 'label_corrections.json'
    if cf.exists():
        corr = {k: v['now'] for k, v in json.loads(cf.read_text())['corrections'].items()}
    for L in FIST:
        for p in sorted(glob.glob(str(LET / split / L / '*.npz'))):
            key = str(Path(p).relative_to(LET))
            if corr.get(key, L) != L:
                continue
            d = np.load(p)
            if 'landmarks' not in d:
                continue
            f = features(d['landmarks'].astype(np.float32))
            if f is not None:
                X.append(f); y.append(L); names.append(key)
    return np.array(X), np.array(y), names


if __name__ == '__main__':
    Xtr, ytr, _ = load('train'); Xva, yva, nva = load('validation')
    print('train', len(ytr), 'validation', len(yva))
    models = {'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0)),
              'boosted': HistGradientBoostingClassifier(max_iter=300, learning_rate=.05)}
    out = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        acc = {L: f"{int(((pred == L) & (yva == L)).sum())}/{int((yva == L).sum())}" for L in FIST}
        es = (yva == 'E') | (yva == 'S')
        print(name, 'validation fist acc', round(float((pred == yva).mean()), 3), acc,
              'E/S pair acc', round(float((pred[es] == yva[es]).mean()), 3))
        out[name] = dict(acc=float((pred == yva).mean()), per_letter=acc)
    # the letter head on the same validation clips (from the earlier full check, head a, before geometry)
    rows = json.load(open(ROOT / 'artifacts/reports/live_correctness_v17_20260929/full_letter_check.json'))['rows']
    fist = [r for r in rows if r['expected'] in FIST]
    head = {L: f"{sum(r['before_correct'] for r in fist if r['expected'] == L)}/{sum(r['expected'] == L for r in fist)}" for L in FIST}
    print('letter head (current) validation fist acc', round(sum(r['before_correct'] for r in fist) / len(fist), 3), head)
    # the user's harvested E spans
    m = models['boosted']
    for L in 'E':
        ups = sorted(glob.glob(str(USER / '*' / L / '*.npz')))
        fx = [(json.loads(str(np.load(p)['meta'])), features(np.load(p)['landmarks'].astype(np.float32))) for p in ups]
        fx = [(meta, f) for meta, f in fx if f is not None]
        pred = m.predict(np.stack([f for _, f in fx])) if fx else []
        print('user E spans', len(fx), 'geometry says', dict(zip(*np.unique(pred, return_counts=True))) if len(fx) else {},
              'model said', dict(zip(*np.unique([meta['predicted'] for meta, _ in fx], return_counts=True))))
        for (meta, _), p_ in zip(fx, pred):
            if meta['predicted'] != 'E':
                print('   model', meta['predicted'], '-> geometry', p_, meta['session'], meta['run'])
    json.dump(out, open(Path(__file__).parent / 'fist_geometry.json', 'w'), indent=1)
