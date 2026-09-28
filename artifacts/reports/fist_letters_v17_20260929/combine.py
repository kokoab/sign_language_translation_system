"""Letter head + geometry: which combination, which letter group. Train split -> validation split.
Head probabilities come from the current head (letter_head_v17_a) on the same validation clips."""
import json, sys, glob
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from fist_geometry import LET
from active.v17.fist_geometry_v17 import features
from active.v17.segmental_runtime_v17 import build_runtime, log_softmax
torch.set_num_threads(6)
rec = build_runtime(backend='torch', device='cpu').recognizer
corr = {k: v['now'] for k, v in json.loads((LET.parent / 'label_corrections.json').read_text())['corrections'].items()}
ALPHA = [chr(65 + i) for i in range(26)]


def load(split, letters):
    X, y, inp = [], [], []
    for L in letters:
        for p in sorted(glob.glob(str(LET / split / L / '*.npz'))):
            if corr.get(str(Path(p).relative_to(LET)), L) != L: continue
            d = np.load(p)
            if 'landmarks' not in d: continue
            f = features(d['landmarks'].astype(np.float32))
            if f is None: continue
            X.append(f); y.append(L)
            inp.append(tuple(d[k].astype(np.float32) if k != 'hand_valid' else d[k] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')))
    return np.array(X), np.array(y), inp


results = {}
for group_name, group in (('fist', list('AEMNST')), ('fist+OC', list('AEMNSTOC'))):
    Xtr, ytr, _ = load('train', group)
    # validation: every letter (to see what the rule does when the true letter is outside the group)
    Xva, yva, inp = load('validation', ALPHA)
    geo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(Xtr, ytr)
    logits = np.concatenate([rec.logits(inp[i:i + 32])[:, 100:] for i in range(0, len(inp), 32)])
    head = np.exp(np.stack([log_softmax(l) for l in logits]))[:, :26]
    gidx = [ALPHA.index(L) for L in group]
    pg = np.zeros((len(yva), 26)); pg[:, [ALPHA.index(c) for c in geo.classes_]] = geo.predict_proba(Xva)
    variants = {}
    for mode in ('head', 'replace', 'product'):
        pred = []
        for h, g in zip(head, pg):
            top = int(h.argmax())
            if mode == 'head' or top not in gidx:
                pred.append(ALPHA[top]); continue
            mass = h[gidx].sum()
            if mode == 'replace':
                new = g[gidx] * mass
            else:
                new = h[gidx] * g[gidx]; new = new / max(new.sum(), 1e-12) * mass
            q = h.copy(); q[gidx] = new
            pred.append(ALPHA[int(q.argmax())])
        pred = np.array(pred)
        ing = np.isin(yva, group)
        variants[mode] = dict(all_letters=f'{int((pred == yva).sum())}/{len(yva)}',
                              group=f'{int((pred[ing] == yva[ing]).sum())}/{int(ing.sum())}',
                              outside_group_changed_to_wrong=int(((pred != yva) & ~ing & (head.argmax(1) == np.array([ALPHA.index(v) for v in yva]))).sum()),
                              per_letter={L: f'{int(((pred == L) & (yva == L)).sum())}/{int((yva == L).sum())}' for L in group})
        print(group_name, mode, variants[mode]['all_letters'], 'group', variants[mode]['group'],
              'broke outside', variants[mode]['outside_group_changed_to_wrong'], variants[mode]['per_letter'], flush=True)
    results[group_name] = variants
json.dump(results, open(Path(__file__).parent / 'combine.json', 'w'), indent=1)
