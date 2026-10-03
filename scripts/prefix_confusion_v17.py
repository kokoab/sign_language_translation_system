"""Which signs' beginnings look like other signs? (drives the early-display guard)

Runs a span recognizer on the first fraction of every isolated validation clip and counts how often
the prefix of class h is predicted as class g. Glosses g that frequently stand in for other signs'
beginnings are 'early-unsafe': the streaming decoder must not show them before their segment closes.
Validation data only; no test access.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.geometry_v17 import resample_features
from active.v17.export_unified_multimodal_coreml_v17 import load_model

CACHE = ROOT / 'artifacts/generated/unfrozen_phrase_adapt_v17'


def prefix(values, fraction):
    lm, he, hv, hb = values
    k = max(4, int(round(32 * fraction)))
    lm = resample_features(lm[:k], 32).astype(np.float32)
    h = max(2, int(round(16 * fraction)))
    idx = np.rint(np.linspace(0, h - 1, 16)).astype(int)
    return lm, he[idx], hv[idx], hb[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='artifacts/models/span_recognizer_v17_local_a/best_model.pth')
    ap.add_argument('--fractions', default='0.35,0.5,0.65')
    ap.add_argument('--threshold', type=float, default=.2)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--raw-cache', type=Path, default=CACHE, help='isolated raw cache (MediaPipe family: its own)')
    a = ap.parse_args()
    if torch.load(a.checkpoint, map_location='cpu', weights_only=False).get('format') == 'slt_stage1_v17':
        from scripts.train_span_recognizer_v17 import load_landmark_only  # Android landmark-only recognizer
        model, ck = load_landmark_only(Path(a.checkpoint))
    else:
        model, ck = load_model(Path(a.checkpoint))
    model.eval()
    labels = [l for l, _ in sorted(ck['label_to_index'].items(), key=lambda r: r[1])]
    fractions = [float(f) for f in a.fractions.split(',')]
    hits = defaultdict(Counter)     # true h -> Counter(pred g) over prefixes
    totals = Counter()
    full_ok = Counter()
    for source in ('citizen', 'semlex', 'local'):
        with np.load(a.raw_cache / f'{source}_val.npz', allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
        n = len(data['targets'])
        for i in range(0, n, 128):
            sl = slice(i, i + 128)
            targets = data['targets'][sl]
            batch = [data['landmarks'][sl].astype(np.float32), data['hand_embeddings'][sl].astype(np.float32),
                     data['hand_valid'][sl].astype(bool), data['hand_boxes'][sl].astype(np.float32)]
            with torch.inference_mode():
                full = model(*(torch.from_numpy(x) for x in batch)).argmax(1).numpy()
            for j, t in enumerate(targets):
                full_ok[int(t)] += int(full[j] == t)
            for f in fractions:
                views = [prefix(tuple(x[j] for x in batch), f) for j in range(len(targets))]
                stacked = [np.stack([v[q] for v in views]) for q in range(4)]
                with torch.inference_mode():
                    pred = model(torch.from_numpy(stacked[0]), torch.from_numpy(stacked[1]),
                                 torch.from_numpy(stacked[2]), torch.from_numpy(stacked[3])).argmax(1).numpy()
                for j, t in enumerate(targets):
                    hits[int(t)][int(pred[j])] += 1
                    totals[int(t)] += 1
    pairs = []
    for h, counter in hits.items():
        for g, c in counter.items():
            if g != h and totals[h] >= 6 and c / totals[h] >= a.threshold:
                pairs.append(dict(prefix_of=labels[h], looks_like=labels[g], rate=c / totals[h], n=totals[h]))
    pairs.sort(key=lambda p: -p['rate'])
    unsafe = sorted({p['looks_like'] for p in pairs})
    result = dict(checkpoint=a.checkpoint, fractions=fractions, threshold=a.threshold, pairs=pairs,
                  early_unsafe=unsafe, scope='isolated validation (Citizen, SemLex, local); no test data')
    a.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(early_unsafe=unsafe, pairs=[(p['prefix_of'], p['looks_like'], round(p['rate'], 2), p['n']) for p in pairs]), indent=0))


if __name__ == '__main__':
    main()
