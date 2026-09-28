"""head^w x geometry within the fist group (w=1 product, w=0 replace); validation split."""
import json, sys, glob
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.fist_geometry_v17 import FistGeometry, features
from active.v17.segmental_runtime_v17 import build_runtime, log_softmax
torch.set_num_threads(6)
rec = build_runtime(backend='torch', device='cpu').recognizer
LET = ROOT / 'data/local/fingerspelling_letters_v17/spans'
ALPHA = [chr(65 + i) for i in range(26)]
fg = FistGeometry()
y, inp, feats = [], [], []
for p in sorted(LET.glob('validation/*/*.npz')):
    d = np.load(p)
    if 'landmarks' not in d: continue
    y.append(p.parent.name)
    inp.append(tuple(d[k].astype(np.float32) if k != 'hand_valid' else d[k] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')))
    feats.append(features(d['landmarks'].astype(np.float32)))
logits = np.concatenate([rec.logits(inp[i:i + 32])[:, 100:] for i in range(0, len(inp), 32)])
y = np.array(y); gidx = fg.index
for w in (1.0, .75, .5, .25, 0.0):
    pred = []
    for l, f in zip(logits, feats):
        p = np.exp(log_softmax(l))
        top = int(p.argmax())
        if top in gidx and f is not None:
            mass = p[gidx].sum()
            new = p[gidx] ** w * fg.probabilities(f)
            p = p.copy(); p[gidx] = new / new.sum() * mass
        pred.append(ALPHA[int(p[:26].argmax())])
    pred = np.array(pred); ing = np.isin(y, [ALPHA[i] for i in gidx])
    print(f'w={w}: all {int((pred == y).sum())}/{len(y)}  fist {int((pred[ing] == y[ing]).sum())}/{int(ing.sum())}',
          {L: f'{int(((pred == L) & (y == L)).sum())}/{int((y == L).sum())}' for L in 'AEMNST'}, flush=True)
